"""Prefix-only trajectory portfolios with a fixed, scheduled beta.

Ranks historical successful anchors, gains, regressions, repair episodes, semantic
novelty and explored depth. Independent normal/exploration probes fill workers;
at most one actual recovery uses capacity not needed by successful refinements.
Repairable failures retain anchors and compete again after every prefix change.
Closure requires repeated current-episode environmental failures or sufficient
successful stagnation evidence. Reserved actions are reconsidered before stopping.
Default beta is 0.6: supplied history has no paired live trend and informative
beta sweep supporting a different default. No evaluation is claimed here.
Grid planning uses completed historical trajectories when supplied, otherwise an
explicit context-derived bootstrap. Unknown remaining depth is never invented;
a legal frontier proves only that at least one further probe is available.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping

from dream_rsi.policy.api import (
    GridPlan, GridPlanningContext, LLMDesignedMethod, SimResult,
    _budget_done, _record_curve, finalize_result,
)


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _get(value, key, default=None):
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def _successful(obs):
    # Validity counts do not override the documented evaluation semantics.
    return bool(_get(obs, 'evaluated', False)) and _get(obs, 'error') is None and _get(obs, 'fail_class') == 'ok'


def _failure(obs):
    if _successful(obs):
        return 'success'
    category = str(_get(obs, 'fail_class', '')).lower()
    message = str(_get(obs, 'error', '') or '').lower()
    text = category + ' ' + message
    repair = (
        'mismatch', 'correctness', 'incorrect', 'wrong output', 'assertion',
        'shared memory', 'shared_memory', 'resource', 'out of memory',
        'out_of_memory', 'too many registers', 'register limit', 'mask',
        'layout', 'shape', 'dimension', 'stride', 'broadcast', 'indexerror',
        'nameerror', 'undefined', 'unbound', 'syntax', 'typeerror', 'variable',
        'illegal memory', 'invalid argument', 'launch', 'code generation',
    )
    environmental = (
        'modulenotfound', 'no module named', 'missing dependency',
        'dependency unavailable', 'driver unavailable', 'no cuda device',
        'cuda driver version', 'unsupported architecture', 'permission denied',
        'compiler not found', 'toolchain unavailable', 'environment unavailable',
    )
    if any(token in text for token in repair):
        return 'repair'
    if category == 'compile_env' or any(token in text for token in environmental):
        return 'hard'
    if category in {'compile_other', 'runtime', 'compile', 'timeout', 'correctness', 'validation'}:
        return 'repair'
    # Zero valid cases alone establish neither repairability nor impossibility.
    return 'weak'


def _tags(meta):
    tags = getattr(meta, 'tags', ())
    if isinstance(tags, Mapping):
        return tuple(sorted(str(k) + '=' + str(v) for k, v in tags.items()))
    if isinstance(tags, str):
        return (tags,)
    try:
        return tuple(sorted(str(t) for t in tags))
    except TypeError:
        return ()


class OptimalPolicy(LLMDesignedMethod):
    NAME = 'OptimalPolicy'

    def __init__(self, config=None):
        super().__init__(config)
        beta = float(self.config.get('beta', 0.6))
        self.beta = min(1.0, max(0.0, beta)) if math.isfinite(beta) else 0.6
        self.sch = self._schedule(self.beta)

    def _schedule(self, beta):
        return {
            'width_fraction': 0.25 + 0.75 * beta,
            'width_yield': 0.25 * beta,
            'min_successes': 2 + int(3 * beta),
            'patience': 2 + int(5 * beta),
            'leader_patience': 3 + int(7 * beta),
            'hard_patience': 2 + int(2 * beta),
            'repair_patience': 2 + int(5 * beta),
            'weak_patience': 2 + int(4 * beta),
            'gap_close': 0.25 + 0.65 * beta,
            'active_floor': 0.30 - 0.70 * beta,
            'reserve_floor': 0.05 - 0.60 * beta,
            'wait_limit': 2 + int(5 * beta),
            'age_weight': 0.10 + 0.08 * beta,
            'root_base': 0.55 + 0.65 * beta,
            'novelty_weight': 0.30,
            'anchor_weight': 0.65,
            'gap_weight': 0.55 - 0.20 * beta,
            'trend_weight': 0.40,
            'gain_weight': 0.35,
            'regression_weight': 0.25,
            'stale_weight': 0.12 - 0.06 * beta,
            'depth_weight': 0.15 + 0.10 * beta,
            'under_bonus': 0.75 + 0.25 * beta,
            'normal_base': 0.45,
            'recovery_base': 0.45 + 0.30 * beta,
            'repair_history_weight': 0.30,
            'failure_weight': 0.27 - 0.13 * beta,
            'overdue_failure_weight': 0.55 - 0.25 * beta,
            'hard_weight': 0.30,
            'weak_weight': 0.20,
            'signal_clip': 2.0,
            'history_cycles': 3,
            'history_min': 2,
            'grid_step': 0.10 + 0.10 * beta,
            'grid_majority': 0.60,
            'late_fraction': 0.50,
            'distinct_fraction': 0.50,
            'grid_hard_fraction': 0.50,
        }

    def _trajectory(self, items, prefix, question):
        items = sorted(items, key=lambda pair: (question.meta(pair[0]).attempt, question.meta(pair[0]).seq, pair[0]))
        successes, gains, kinds = [], [], []
        anchor = None
        anchor_index = -1
        repair_runs = repairs = recovered = 0
        in_repair = False
        for index, (cell, obs) in enumerate(items):
            kind = _failure(obs)
            kinds.append(kind)
            if kind == 'repair':
                repairs += 1
                if not in_repair:
                    repair_runs += 1
                in_repair = True
            if kind == 'success':
                if in_repair:
                    recovered += 1
                in_repair = False
                score = _number(_get(obs, 'score'))
                if score is not None:
                    successes.append((index, score))
                    if anchor is None or score > anchor:
                        anchor, anchor_index = score, index
                    parent = prefix.get(getattr(question.meta(cell), 'parent_id', None))
                    if parent is not None and _successful(parent):
                        parent_score = _number(_get(parent, 'score'))
                        gain = _number(_get(obs, 'delta_vs_parent'))
                        if gain is None and parent_score is not None:
                            gain = score - parent_score
                        if gain is not None:
                            gains.append(gain)
        episode = []
        for kind in reversed(kinds):
            if kind == 'success':
                break
            episode.append(kind)
        differences = [b[1] - a[1] for a, b in zip(successes, successes[1:])]
        return {
            'anchor': anchor, 'successes': len(successes),
            'trend': sum(differences) / len(differences) if differences else 0.0,
            'gain': gains[-1] if gains else 0.0,
            'regressions': sum(d < 0 for d in differences),
            'transitions': len(differences), 'episode': episode,
            'last': kinds[-1], 'failures': len(episode),
            'stale': len(items) - 1 - anchor_index,
            'depth': question.meta(items[-1][0]).attempt,
            'count': len(items), 'repairs': repairs,
            'repair_rate': recovered / repair_runs if repair_runs else 0.0,
            # Only a legal frontier, not inaccessible future metadata, proves this.
            'remaining_lower_bound': 1,
        }

    def solve(self, question, budget=None):
        question.reset()
        result = SimResult()
        schedule = self.sch
        waiting = {}
        while not _budget_done(question, budget):
            prefix = question.observed()
            legal = list(dict.fromkeys(question.legal_actions()))
            if not legal:
                break
            workers = max(1, int(question.max_parallelism or 1))
            capacity = workers
            if budget is not None:
                capacity = min(capacity, max(0, int(budget) - len(prefix)))
            if capacity == 0:
                break
            roots = set(question.legal_roots()).intersection(legal)
            grouped = defaultdict(list)
            for cell, obs in prefix.items():
                grouped[question.meta(cell).branch].append((cell, obs))
            trajectories = {b: self._trajectory(rows, prefix, question) for b, rows in grouped.items()}
            baseline = float(question.baseline_score)
            scores = [baseline]
            scores.extend(v for obs in prefix.values() if _successful(obs) for v in [_number(_get(obs, 'score'))] if v is not None)
            best = max(scores)
            scale = max(max(scores) - min(scores), max(math.ulp(v) for v in scores))
            tag_counts = Counter()
            for rows in grouped.values():
                root = min(rows, key=lambda pair: question.meta(pair[0]).attempt)[0]
                tag_counts.update(_tags(question.meta(root)))
            total_width = len(grouped) + len(roots)
            improving = sum(t['anchor'] is not None and t['anchor'] > baseline for t in trajectories.values())
            root_yield = improving / len(grouped) if grouped else 0.0
            width_fraction = min(1.0, schedule['width_fraction'] + schedule['width_yield'] * root_yield)
            width_limit = min(total_width, max(workers, math.ceil(total_width * width_fraction)))
            closed, candidates = set(), []

            def bounded(value):
                return max(-schedule['signal_clip'], min(schedule['signal_clip'], value / scale))

            for cell in legal:
                meta = question.meta(cell)
                age = min(waiting.get(cell, 0), schedule['wait_limit'])
                age_bonus = schedule['age_weight'] * age
                tags = _tags(meta)
                novelty = sum(1.0 / (1 + tag_counts[t]) for t in tags) / len(tags) if tags else 0.0
                if cell in roots:
                    score = schedule['root_base'] + schedule['novelty_weight'] * novelty + age_bonus
                    status = 'active' if len(grouped) < width_limit else 'reserve'
                    candidates.append({'cell': cell, 'branch': meta.branch, 'role': 'explore', 'score': score, 'status': status, 'root': True, 'tie': (tags, str(meta.seq), cell)})
                    continue
                trajectory = trajectories.get(meta.branch)
                if trajectory is None:
                    continue
                t = trajectory
                anchor = t['anchor'] if t['anchor'] is not None else baseline
                gap = max(0.0, (best - anchor) / scale)
                under = t['successes'] < schedule['min_successes']
                # Closure is rebuilt each round: a later successful evaluation
                # cancels every closure based on an earlier failure episode.
                hard_closed = (t['last'] == 'hard' and len(t['episode']) >= schedule['hard_patience'] and all(k == 'hard' for k in t['episode']))
                stale_closed = (t['last'] == 'success' and not under and t['stale'] >= schedule['patience'] and t['trend'] <= 0 and t['gain'] <= 0 and (gap > schedule['gap_close'] or t['stale'] >= schedule['leader_patience']))
                if hard_closed or stale_closed:
                    closed.add(meta.branch)
                    continue
                score = (
                    schedule['normal_base']
                    + schedule['anchor_weight'] * bounded(anchor - baseline)
                    - schedule['gap_weight'] * min(gap, schedule['signal_clip'])
                    + schedule['trend_weight'] * bounded(t['trend'])
                    + schedule['gain_weight'] * bounded(t['gain'])
                    - schedule['regression_weight'] * t['regressions'] / max(1, t['transitions'])
                    - schedule['stale_weight'] * t['stale']
                    + schedule['depth_weight'] * t['remaining_lower_bound'] / (1 + t['depth'])
                    + age_bonus
                )
                if t['last'] == 'repair':
                    role = 'recover'
                    score += schedule['recovery_base'] + schedule['repair_history_weight'] * t['repair_rate']
                    score -= schedule['failure_weight'] * max(0, t['failures'] - 1)
                    score -= schedule['overdue_failure_weight'] * max(0, t['failures'] - schedule['repair_patience'])
                    # First repair always gets a fair eligibility decision,
                    # independent of validity counts or a damaged latest score.
                    guaranteed = t['failures'] <= schedule['repair_patience']
                elif t['last'] in {'weak', 'hard'}:
                    role = 'explore'
                    score += schedule['under_bonus']
                    score -= schedule['weak_weight'] * t['failures']
                    if t['last'] == 'hard':
                        score -= schedule['hard_weight']
                    score -= schedule['overdue_failure_weight'] * max(0, t['failures'] - schedule['weak_patience'])
                    guaranteed = t['failures'] < schedule['weak_patience']
                else:
                    role = 'explore' if under else 'normal'
                    score += schedule['under_bonus'] if under else 0.0
                    guaranteed = under or t['gain'] > 0 or t['trend'] > 0
                status = 'active' if guaranteed or score >= schedule['active_floor'] else 'reserve'
                # A finite waiting allowance promotes plausible reserves;
                # repeated failure penalties can still justify stopping.
                if status == 'reserve' and score >= schedule['reserve_floor'] and age >= schedule['wait_limit']:
                    status = 'active'
                candidates.append({'cell': cell, 'branch': meta.branch, 'role': role, 'score': score, 'status': status, 'root': False, 'tie': (tags, str(meta.seq), cell)})

            def ranking(candidate):
                return (-candidate['score'], candidate['tie'])

            active = sorted((c for c in candidates if c['status'] == 'active'), key=ranking)
            # Portfolio-level stop: plausible reserves can run immediately if
            # there is nothing else to wait for. Width reserves remain justified
            # by the scheduled width allowance and observed root yield.
            if not active:
                active = sorted((c for c in candidates if not c['root'] and c['score'] >= schedule['reserve_floor']), key=ranking)
            if not active:
                break
            normals = [c for c in active if c['role'] == 'normal']
            exploration = [c for c in active if c['role'] == 'explore']
            recovery = [c for c in active if c['role'] == 'recover']
            batch, branches = [], set()
            root_slots = max(0, width_limit - len(grouped))

            def add(candidate):
                nonlocal root_slots
                if len(batch) >= capacity or candidate['branch'] in branches:
                    return False
                if candidate['root'] and root_slots <= 0:
                    return False
                batch.append(candidate['cell'])
                branches.add(candidate['branch'])
                if candidate['root']:
                    root_slots -= 1
                return True

            # Protect a strong normal refinement; an exploration seat is useful
            # when parallel capacity exists. A one-worker run uses ranked normal
            # and exploration competition rather than a fictitious parallel batch.
            if capacity == 1:
                ordinary = sorted(normals + exploration, key=ranking)
                if ordinary:
                    add(ordinary[0])
                elif recovery:
                    add(recovery[0])
            else:
                if normals:
                    add(normals[0])
                if exploration:
                    add(exploration[0])
                # Recovery cannot evict an eligible normal successful refinement.
                remaining_normals = sum(c['branch'] not in branches for c in normals)
                if recovery and len(batch) + remaining_normals < capacity:
                    add(recovery[0])
                for candidate in sorted(normals + exploration, key=ranking):
                    add(candidate)
                if recovery and not any(c in batch for c in [r['cell'] for r in recovery]):
                    add(recovery[0])
            if not batch:
                break
            selected = set(batch)
            waiting = {c['cell']: (0 if c['cell'] in selected else waiting.get(c['cell'], 0) + 1) for c in candidates}
            question.probe_batch(batch, on_reveal=lambda _: _record_curve(result, question))
            # Guard a non-progressing implementation without assuming a budget.
            if len(question.observed()) <= len(prefix):
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
        reason = 'Insufficient attributed historical width/depth evidence; retain the explicit fallback grid within hard caps and replay support.'
        votes = []
        # The manifest schema is not specified to include trajectories. Never
        # infer early/late gains from final score alone. Optional observation rows
        # must carry their own branch, attempt and success evidence.
        for manifest in list(context.history)[-s['history_cycles']:]:
            if not isinstance(manifest, Mapping):
                continue
            rows = manifest.get('observations')
            if isinstance(rows, Mapping):
                rows = list(rows.values())
            if not isinstance(rows, (list, tuple)):
                continue
            groups = defaultdict(list)
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                branch, attempt = row.get('branch'), row.get('attempt')
                if isinstance(branch, int) and isinstance(attempt, int) and attempt >= 0:
                    groups[branch].append(row)
            if not groups:
                continue
            prior_base = _number(manifest.get('baseline_score'))
            early, late, hard, plateau, distinct = 0, 0, 0, 0, set()
            adequate = 0
            for records in groups.values():
                records.sort(key=lambda row: row['attempt'])
                valid = [r for r in records if _successful(r) and _number(r.get('score')) is not None]
                terminal = []
                for row in reversed(records):
                    if _successful(row):
                        break
                    terminal.append(_failure(row))
                if len(terminal) >= s['hard_patience'] and all(k == 'hard' for k in terminal):
                    hard += 1
                if len(valid) < s['min_successes']:
                    continue
                adequate += 1
                best_row = max(valid, key=lambda row: float(row['score']))
                last_depth = records[-1]['attempt']
                improves_root = float(best_row['score']) > float(valid[0]['score'])
                if improves_root and best_row['attempt'] > s['late_fraction'] * last_depth:
                    late += 1
                if prior_base is not None and valid[0]['attempt'] == 0 and float(valid[0]['score']) > prior_base and not improves_root:
                    early += 1
                    label = valid[0].get('direction_class')
                    if label is not None:
                        distinct.add(str(label))
                if len(valid) - 1 - valid.index(best_row) >= s['patience']:
                    plateau += 1
            if hard / len(groups) >= s['grid_hard_fraction']:
                votes.append('contract')
            elif adequate and late / adequate >= s['grid_majority']:
                votes.append('depth')
            elif adequate and early / adequate >= s['grid_majority'] and len(distinct) >= max(2, math.ceil(adequate * s['distinct_fraction'])):
                votes.append('width')
            elif adequate and plateau == adequate and manifest.get('uncovered_direction_classes'):
                votes.append('width')
            else:
                votes.append('uncertain')
        if len(votes) >= s['history_min'] and len(set(votes)) == 1:
            step_w = max(1, math.ceil(width * s['grid_step']))
            step_d = max(1, math.ceil(max(1, depth) * s['grid_step']))
            if votes[0] == 'width':
                width += step_w
                reason = 'Completed attributed trajectories repeatedly favor distinct early directions or report sufficiently explored plateaus with uncovered classes; increase width and hold depth.'
            elif votes[0] == 'depth':
                depth += step_d
                reason = 'Completed attributed trajectories repeatedly place improvement late in explored branches; hold width and increase depth.'
            elif votes[0] == 'contract':
                width -= step_w
                depth -= step_d
                reason = 'Completed trajectories repeatedly show unrecovered environmental failure episodes; conservatively reduce width and depth.'
        width = max(1, min(width_cap, width))
        depth = max(0, min(depth_cap, depth))
        return GridPlan(branch_count=width, refine_count=depth, reason=reason)
