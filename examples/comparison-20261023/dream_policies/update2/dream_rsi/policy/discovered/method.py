"""Prefix-only trajectory portfolios with a fixed beta schedule.

Retains successful anchors through repair episodes, but uses recent successful
transitions and successful stagnation rather than an indefinitely positive old
trend. Deterministic queues compete on anchor, gain, regressions, repair history,
semantic novelty, waiting and explored depth. Plausible reserves fill otherwise
unused workers; at most one actual repair receives spare normal-refinement
capacity. Closure is recomputed from current episodes, never from validity counts.
All policy thresholds live in _schedule. Default beta remains 0.6 because the
supplied sweep lacks paired live-cycle evidence supporting a default change.
Its one-worker runs cannot establish a batching defect. Grid planning uses only
completed context history, or explicitly returns the capped fallback when
attributed evidence is insufficient. No execution or performance claim is made.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping

from dream_rsi.policy.api import (
    GridPlan, GridPlanningContext, LLMDesignedMethod, SimResult,
    _budget_done, _record_curve, finalize_result,
)


def _get(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, Mapping) else getattr(obj, key, default)


def _number(value):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _success(obs):
    return bool(_get(obs, 'evaluated', False)) and _get(obs, 'error') is None and _get(obs, 'fail_class') == 'ok'


def _kind(obs):
    if _success(obs):
        return 'success'
    category = str(_get(obs, 'fail_class', '')).lower()
    text = category + ' ' + str(_get(obs, 'error', '') or '').lower()
    repair = (
        'mismatch', 'correctness', 'incorrect', 'wrong output', 'assertion',
        'shared memory', 'shared_memory', 'resource', 'out of memory',
        'out_of_memory', 'register limit', 'too many registers', 'mask',
        'layout', 'shape', 'dimension', 'stride', 'broadcast', 'indexerror',
        'nameerror', 'undefined', 'unbound', 'syntax', 'typeerror', 'variable',
        'illegal memory', 'invalid argument', 'launch', 'code generation',
    )
    hard = (
        'modulenotfound', 'no module named', 'missing dependency',
        'dependency unavailable', 'driver unavailable', 'no cuda device',
        'cuda driver version', 'unsupported architecture', 'permission denied',
        'compiler not found', 'toolchain unavailable', 'environment unavailable',
    )
    if any(word in text for word in repair):
        return 'repair'
    if category == 'compile_env' or any(word in text for word in hard):
        return 'hard'
    if category in {'compile_other', 'compile', 'runtime', 'timeout', 'validation'}:
        return 'repair'
    # Neither valid=False nor n_valid=0 establishes impossibility.
    return 'weak'


def _tags(meta):
    value = _get(meta, 'tags', ())
    if isinstance(value, Mapping):
        return tuple(sorted(str(k) + '=' + str(v) for k, v in value.items()))
    if isinstance(value, str):
        return (value,)
    return tuple(sorted(str(v) for v in (value or ())))


class OptimalPolicy(LLMDesignedMethod):
    NAME = 'OptimalPolicy'

    def __init__(self, config=None):
        super().__init__(config)
        beta = float(self.config.get('beta', 0.6))
        self.beta = min(1.0, max(0.0, beta)) if math.isfinite(beta) else 0.6
        self.sch = self._schedule(self.beta)

    def _schedule(self, beta):
        return {
            'width': 0.25 + 0.75 * beta,
            'yield_width': 0.20 * beta,
            'novel_width': 0.15 * beta,
            'min_success': 2 + int(3 * beta),
            'patience': 2 + int(4 * beta),
            'leader_patience': 3 + int(7 * beta),
            'recent': 2 + int(2 * beta),
            'hard_patience': 2 + int(2 * beta),
            'repair_patience': 2 + int(4 * beta),
            'weak_patience': 2 + int(4 * beta),
            'gap_close': 0.25 + 0.65 * beta,
            'active': 0.30 - 0.60 * beta,
            'reserve': 0.05 - 0.55 * beta,
            'wait': 2 + int(5 * beta),
            'age': 0.10 + 0.08 * beta,
            'root': 0.55 + 0.60 * beta,
            'novelty': 0.35,
            'anchor': 0.70,
            'gap': 0.55 - 0.20 * beta,
            'trend': 0.40,
            'gain': 0.40,
            'regression': 0.25,
            'stale': 0.14 - 0.08 * beta,
            'depth': 0.15 + 0.10 * beta,
            'under': 0.65 + 0.30 * beta,
            'normal': 0.45,
            'recovery': 0.40 + 0.35 * beta,
            'repair_history': 0.35,
            'failure': 0.28 - 0.14 * beta,
            'overdue': 0.55 - 0.25 * beta,
            'hard_cost': 0.30,
            'clip': 2.0,
            'history': 3,
            'history_min': 2,
            'grid_step': 0.10 + 0.10 * beta,
            'majority': 0.60,
            'late': 0.50,
            'distinct': 0.50,
            'hard_fraction': 0.50,
        }

    def _trajectory(self, rows, prefix, question):
        rows = sorted(rows, key=lambda p: (question.meta(p[0]).attempt, question.meta(p[0]).seq, p[0]))
        values, kinds, transitions = [], [], []
        anchor = None
        stale = 0
        episodes = recovered = 0
        repairing = False
        latest_parent_gain = 0.0
        for cell, obs in rows:
            kind = _kind(obs)
            kinds.append(kind)
            latest_parent_gain = 0.0
            if kind == 'repair' and not repairing:
                episodes += 1
                repairing = True
            if kind != 'success':
                continue
            if repairing:
                recovered += 1
                repairing = False
            score = _number(_get(obs, 'score'))
            if score is None:
                continue
            if values:
                transitions.append(score - values[-1])
            values.append(score)
            if anchor is None or score > anchor:
                anchor, stale = score, 0
            else:
                stale += 1
            parent = prefix.get(_get(question.meta(cell), 'parent_id'))
            if parent is not None and _success(parent):
                parent_score = _number(_get(parent, 'score'))
                gain = _number(_get(obs, 'delta_vs_parent'))
                if gain is None and parent_score is not None:
                    gain = score - parent_score
                latest_parent_gain = gain if gain is not None else 0.0
            elif len(values) > 1:
                # Recovery gain compares successful anchors, not failed scores.
                latest_parent_gain = transitions[-1]
        episode = []
        for kind in reversed(kinds):
            if kind == 'success':
                break
            episode.append(kind)
        recent = transitions[-self.sch['recent']:]
        return {
            'anchor': anchor, 'successes': len(values), 'stale': stale,
            'trend': sum(recent) / len(recent) if recent else 0.0,
            'gain': latest_parent_gain,
            'regression': sum(g < 0 for g in transitions) / max(1, len(transitions)),
            'episode': episode, 'failures': len(episode), 'last': kinds[-1],
            'repair_rate': recovered / episodes if episodes else 0.0,
            'depth': question.meta(rows[-1][0]).attempt,
            'count': len(rows),
            # The API exposes no enumerable future depth. A legal frontier
            # proves one remaining step; no additional depth is fabricated.
            'remaining_lower_bound': 1,
        }

    def solve(self, question, budget=None):
        question.reset()
        result = SimResult()
        s = self.sch
        waiting = {}
        while not _budget_done(question, budget):
            # Copy: observed() may return a live dictionary mutated by probing.
            prefix = dict(question.observed())
            legal = list(dict.fromkeys(question.legal_actions()))
            capacity = max(1, int(question.max_parallelism or 1))
            if budget is not None:
                capacity = min(capacity, max(0, int(budget) - len(prefix)))
            if not legal or capacity <= 0:
                break
            roots = set(question.legal_roots()).intersection(legal)
            grouped = defaultdict(list)
            for cell, obs in prefix.items():
                grouped[question.meta(cell).branch].append((cell, obs))
            trajectories = {b: self._trajectory(rows, prefix, question) for b, rows in grouped.items()}
            baseline = float(question.baseline_score)
            scores = [baseline]
            scores.extend(v for obs in prefix.values() if _success(obs)
                          for v in [_number(_get(obs, 'score'))] if v is not None)
            best = max(scores)
            scale = max(max(scores) - min(scores), max(math.ulp(v) for v in scores))

            def signal(value):
                return max(-s['clip'], min(s['clip'], value / scale))

            counts = Counter()
            for rows in grouped.values():
                first = min(rows, key=lambda p: question.meta(p[0]).attempt)[0]
                counts.update(_tags(question.meta(first)))

            def novelty(tags, counter):
                return sum(1.0 / (1 + counter[t]) for t in tags) / len(tags) if tags else 0.0

            total_width = len(grouped) + len(roots)
            productive = sum(t['anchor'] is not None and t['anchor'] > baseline for t in trajectories.values())
            root_yield = productive / max(1, len(grouped))
            unseen = max((novelty(_tags(question.meta(c)), counts) for c in roots), default=0.0)
            fraction = min(1.0, s['width'] + s['yield_width'] * root_yield + s['novel_width'] * unseen)
            width_limit = min(total_width, max(capacity, math.ceil(total_width * fraction)))
            root_slots = max(0, width_limit - len(grouped))
            candidates = []
            closed = set()
            for cell in legal:
                meta = question.meta(cell)
                tags = _tags(meta)
                age = min(waiting.get(cell, 0), s['wait'])
                bonus = s['age'] * age
                if cell in roots:
                    # Roots outside the evidence-adjusted width envelope remain
                    # reserved and are reconsidered whenever the prefix changes.
                    score = s['root'] + s['novelty'] * novelty(tags, counts) + bonus
                    candidates.append(dict(cell=cell, branch=meta.branch, root=True,
                        role='explore', score=score, eligible=root_slots > 0,
                        active=root_slots > 0, tags=tags, seq=meta.seq))
                    continue
                t = trajectories.get(meta.branch)
                if t is None:
                    continue
                anchor = t['anchor'] if t['anchor'] is not None else baseline
                gap = max(0.0, (best - anchor) / scale)
                under = t['successes'] < s['min_success']
                hard_closed = (t['last'] == 'hard' and len(t['episode']) >= s['hard_patience']
                               and all(k == 'hard' for k in t['episode']))
                stale_closed = (t['last'] == 'success' and not under
                    and t['stale'] >= s['patience'] and t['trend'] <= 0 and t['gain'] <= 0
                    and (gap > s['gap_close'] or t['stale'] >= s['leader_patience']))
                if hard_closed or stale_closed:
                    closed.add(meta.branch)
                    continue
                score = (s['normal'] + s['anchor'] * signal(anchor - baseline)
                    - s['gap'] * min(gap, s['clip']) + s['trend'] * signal(t['trend'])
                    + s['gain'] * signal(t['gain']) - s['regression'] * t['regression']
                    - s['stale'] * t['stale']
                    + s['depth'] * t['remaining_lower_bound'] / (1 + t['depth']) + bonus)
                if t['last'] == 'repair':
                    role = 'recover'
                    score += s['recovery'] + s['repair_history'] * t['repair_rate']
                    score -= s['failure'] * max(0, t['failures'] - 1)
                    score -= s['overdue'] * max(0, t['failures'] - s['repair_patience'])
                    guaranteed = t['failures'] <= s['repair_patience']
                elif t['last'] in {'weak', 'hard'}:
                    role = 'explore'
                    score += s['under'] - s['failure'] * t['failures']
                    score -= s['hard_cost'] if t['last'] == 'hard' else 0.0
                    score -= s['overdue'] * max(0, t['failures'] - s['weak_patience'])
                    guaranteed = t['failures'] < s['weak_patience']
                else:
                    role = 'explore' if under else 'normal'
                    score += s['under'] if under else 0.0
                    guaranteed = under or t['gain'] > 0
                eligible = guaranteed or score >= s['reserve']
                active = guaranteed or score >= s['active'] or (eligible and age >= s['wait'])
                candidates.append(dict(cell=cell, branch=meta.branch, root=False,
                    role=role, score=score, eligible=eligible, active=active,
                    tags=tags, seq=meta.seq))

            # Every nonclosed action has an active/reserved decision. Plausible
            # reserves participate in packing, not only after the active set dies.
            pool = [c for c in candidates if c['eligible']]
            batch, branches = [], set()
            batch_tags = counts.copy()
            recovery_used = False

            def feasible(c):
                return (c['branch'] not in branches and c['cell'] not in batch
                    and (not c['root'] or root_slots > 0)
                    and (c['role'] != 'recover' or not recovery_used))

            def rank(c):
                diversity = s['novelty'] * novelty(c['tags'], batch_tags) if c['root'] else 0.0
                # seq/cell only break otherwise indistinguishable structural ties.
                return (not c['active'], -(c['score'] + diversity), c['tags'], c['seq'], c['cell'])

            def choose(role=None):
                available = [c for c in pool if feasible(c) and (role is None or c['role'] == role)]
                return min(available, key=rank) if available else None

            def add(c):
                nonlocal root_slots, recovery_used
                if c is None or len(batch) >= capacity or not feasible(c):
                    return False
                batch.append(c['cell'])
                branches.add(c['branch'])
                if c['root']:
                    root_slots -= 1
                    batch_tags.update(c['tags'])
                if c['role'] == 'recover':
                    recovery_used = True
                return True

            if capacity == 1:
                # Recovery competes with exploration when there is no eligible
                # normal refinement to protect, including on one-worker traces.
                normal = choose('normal')
                add(normal if normal is not None else choose())
            else:
                add(choose('normal'))
                add(choose('explore'))
                # Recovery cannot displace eligible normal refinements. Protect
                # their remaining capacity before assigning the recovery seat.
                normal_branches = {c['branch'] for c in pool if c['role'] == 'normal' and feasible(c)}
                if len(batch) + len(normal_branches) < capacity:
                    add(choose('recover'))
                while len(batch) < capacity:
                    ordinary = [c for c in pool if c['role'] != 'recover' and feasible(c)]
                    if not ordinary:
                        break
                    add(min(ordinary, key=rank))
                if len(batch) < capacity:
                    add(choose('recover'))
            # Multiple actual repairs cannot share a batch by specification.
            # Otherwise all independently eligible capacity has been considered.
            if not batch:
                break
            selected = set(batch)
            waiting = {c['cell']: waiting.get(c['cell'], 0) + 1
                       for c in candidates if c['cell'] not in selected}
            before = len(prefix)
            question.probe_batch(batch, on_reveal=lambda _: _record_curve(result, question))
            if len(question.observed()) <= before:
                break
        return finalize_result(question, result)

    def plan_grid(self, context: GridPlanningContext) -> GridPlan:
        s = self.sch
        width_cap = int(context.hard_max_branch_count)
        depth_cap = int(context.hard_max_refine_count)
        if context.trace_branch_count is not None:
            width_cap = min(width_cap, int(context.trace_branch_count))
        if context.trace_refine_count is not None:
            depth_cap = min(depth_cap, int(context.trace_refine_count))
        width = max(1, min(width_cap, int(context.fallback_branch_count)))
        depth = max(0, min(depth_cap, int(context.fallback_refine_count)))
        reason = 'Insufficient attributed historical evidence; explicitly retain the context fallback within hard caps and replay support.'
        votes = []
        late_classes = []
        for manifest in list(context.history)[-s['history']:]:
            if not isinstance(manifest, Mapping):
                votes.append('uncertain')
                continue
            rows = manifest.get('observations')
            if isinstance(rows, Mapping):
                rows = list(rows.values())
            if not isinstance(rows, (list, tuple)):
                votes.append('uncertain')
                continue
            groups = defaultdict(list)
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                branch, attempt = row.get('branch'), row.get('attempt')
                if isinstance(branch, int) and isinstance(attempt, int) and attempt >= 0:
                    groups[branch].append(row)
            baseline = _number(manifest.get('baseline_score'))
            adequate = early = late = hard = plateau = 0
            distinct, late_labels = set(), set()
            for records in groups.values():
                records.sort(key=lambda r: r['attempt'])
                episode = []
                for row in reversed(records):
                    if _success(row):
                        break
                    episode.append(_kind(row))
                if len(episode) >= s['hard_patience'] and all(k == 'hard' for k in episode):
                    hard += 1
                valid = [r for r in records if _success(r) and _number(r.get('score')) is not None]
                if len(valid) < s['min_success']:
                    continue
                adequate += 1
                peak_index = max(range(len(valid)), key=lambda i: float(valid[i]['score']))
                peak = valid[peak_index]
                improved = float(peak['score']) > float(valid[0]['score'])
                label = records[0].get('direction_class')
                stale = len(valid) - 1 - peak_index
                if stale >= s['patience']:
                    plateau += 1
                if improved and peak['attempt'] > s['late'] * records[-1]['attempt']:
                    late += 1
                    if label is not None:
                        late_labels.add(str(label))
                if (baseline is not None and valid[0]['attempt'] == 0
                    and float(valid[0]['score']) > baseline and not improved
                    and stale >= s['patience']):
                    early += 1
                    if label is not None:
                        distinct.add(str(label))
            if groups and hard / len(groups) >= s['hard_fraction']:
                votes.append('contract')
            elif adequate and late / adequate >= s['majority'] and late_labels:
                votes.append('depth')
                late_classes.append(late_labels)
            elif (adequate and early / adequate >= s['majority']
                  and len(distinct) >= max(s['history_min'], math.ceil(adequate * s['distinct']))):
                votes.append('width')
            elif adequate == len(groups) and adequate and plateau == adequate and manifest.get('uncovered_direction_classes'):
                votes.append('width')
            else:
                votes.append('uncertain')
        if len(votes) >= s['history_min'] and len(set(votes)) == 1:
            step_w = max(1, math.ceil(width * s['grid_step']))
            step_d = max(1, math.ceil(max(1, depth) * s['grid_step']))
            if votes[0] == 'width':
                width += step_w
                reason = 'Completed trajectories consistently favor distinct early gains with stalled refinement, or sufficiently explored plateaus with uncovered classes; increase width and hold depth.'
            elif votes[0] == 'depth' and late_classes and set.intersection(*late_classes):
                depth += step_d
                reason = 'Repeated completed trajectories attribute late gains to overlapping direction classes; hold width and increase refinement depth.'
            elif votes[0] == 'contract':
                width -= step_w
                depth -= step_d
                reason = 'Repeated completed trajectories show predominantly unrecovered environmental failures; conservatively reduce width and depth.'
        width = max(1, min(width_cap, width))
        depth = max(0, min(depth_cap, depth))
        return GridPlan(branch_count=width, refine_count=depth, reason=reason)
