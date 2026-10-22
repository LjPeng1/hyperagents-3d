from agent.base_agent import AgentSystem
from agent.llm_withtools import chat_with_agent
from portable_context import get_context
import json
import math
import re


def metric(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return float(value) if math.isfinite(value) and 0 <= value <= 1 else None
    except (OverflowError, ValueError):
        return None


def clean(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, list):
        return [clean(item) for item in value]
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items() if isinstance(key, str)}
    return None


def group_key(dev):
    if not isinstance(dev, dict) or dev.get('split') != 'dev':
        return None
    version, digest = dev.get('benchmark_version'), dev.get('scene_sha256')
    backend = dev.get('evaluation_backend')
    if not isinstance(version, str) or not version or not isinstance(digest, str) or not digest:
        return None
    if backend is not None and not isinstance(backend, str):
        return None
    return version, digest, 'dev', backend


def ceiling(dev):
    if any(metric(dev.get(key)) != 1.0 for key in ('combined_score', 'valid_scene_rate', 'strict_success_rate')):
        return False
    components = dev.get('components')
    if not isinstance(components, dict) or any(metric(components.get(key)) != 1.0 for key in ('support', 'collision_free', 'relations')):
        return False
    scenes = dev.get('scenes')
    if not isinstance(scenes, list) or not scenes:
        return False
    seen = set()
    for scene in scenes:
        if not isinstance(scene, dict):
            return False
        sid = scene.get('id')
        if not isinstance(sid, str) or sid in seen:
            return False
        seen.add(sid)
        if scene.get('valid') is not True or scene.get('strict_success') is not True or metric(scene.get('combined_score')) != 1.0:
            return False
        if 'collisions' in scene and scene['collisions'] != []:
            return False
        relations = scene.get('relation_results')
        if relations is not None and (not isinstance(relations, list) or any(not isinstance(r, dict) or r.get('satisfied') is not True for r in relations)):
            return False
    count = dev.get('scene_count')
    return count is None or (isinstance(count, int) and not isinstance(count, bool) and count == len(scenes))


def summarize(observed):
    candidates = [observed.get('best'), observed.get('initial')]
    history = observed.get('history')
    if isinstance(history, list):
        candidates.extend(history)
    groups, seen = {}, set()
    for record in candidates:
        if not isinstance(record, dict):
            continue
        source, dev = record.get('program'), record.get('dev')
        key = group_key(dev)
        if key is None or not isinstance(source, str) or not source.strip():
            continue
        signature = (source, json.dumps(clean(dev), sort_keys=True, ensure_ascii=True))
        if signature in seen:
            continue
        seen.add(signature)
        programs = groups.setdefault(key, {})
        observations = programs.setdefault(source, [])
        scenes = dev.get('scenes')
        failures = []
        unknown = 0
        if isinstance(scenes, list):
            for scene in scenes:
                if not isinstance(scene, dict):
                    unknown += 1
                    continue
                score = metric(scene.get('combined_score'))
                relations = scene.get('relation_results')
                unsatisfied = [r for r in relations if isinstance(r, dict) and r.get('satisfied') is False] if isinstance(relations, list) else []
                collisions = scene.get('collisions')
                explicit_failure = (scene.get('valid') is False or scene.get('strict_success') is False
                                    or (score is not None and score < 1) or bool(unsatisfied)
                                    or (isinstance(collisions, list) and bool(collisions)))
                if explicit_failure:
                    failures.append({'id': scene.get('id'), 'valid': scene.get('valid'),
                                     'strict_success': scene.get('strict_success'), 'combined_score': score,
                                     'components': scene.get('components'), 'collisions': collisions,
                                     'unsatisfied_relations': unsatisfied})
                elif scene.get('valid') is not True or scene.get('strict_success') is not True or score is None:
                    unknown += 1
        else:
            unknown = None
        observations.append({'id': record.get('id'), 'metrics': {name: metric(dev.get(name)) for name in ('combined_score', 'valid_scene_rate', 'strict_success_rate')},
                             'components': dev.get('components'), 'failures': failures,
                             'unknown_scene_results': unknown,
                             'scene_count': len(scenes) if isinstance(scenes, list) else None,
                             'verified_development_ceiling': ceiling(dev),
                             'evaluation_seconds': dev.get('evaluation_seconds'),
                             'blender_verified': dev.get('blender_verified')})
    result = []
    for key, programs in groups.items():
        entries = []
        for source, observations in programs.items():
            entries.append({'program': source, 'observations': observations,
                            'all_observations_at_ceiling': all(run['verified_development_ceiling'] for run in observations)})
        result.append({'benchmark_version': key[0], 'scene_sha256': key[1], 'split': key[2],
                       'evaluation_backend': key[3], 'programs': entries,
                       'policy': 'preserve measured solver; improve agent reliability' if any(entry['all_observations_at_ceiling'] for entry in entries) else 'use explicit incumbent failures only'})
    return clean({'domain': observed.get('domain'), 'problem': observed.get('problem'),
                  'groups': result, 'context_iterations_left': observed.get('iterations_left')})


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError('Nonfinite JSON constant')


def valid_response(history):
    if not isinstance(history, list) or not history or not isinstance(history[-1], dict):
        return False
    text = history[-1].get('text')
    if not isinstance(text, str):
        return False
    text = text.strip()
    if not text.startswith('<json>') or not text.endswith('</json>'):
        return False
    try:
        payload = json.loads(text[6:-7], object_pairs_hook=unique_object, parse_constant=reject_constant)
    except (ValueError, TypeError, RecursionError):
        return False
    if not isinstance(payload, dict) or set(payload) != {'task_agent', 'meta_agent'}:
        return False
    for key, name in (('task_agent', 'TaskAgent'), ('meta_agent', 'MetaAgent')):
        source = payload[key]
        if not isinstance(source, str) or not source.strip() or '\x00' in source:
            return False
        if not re.search(r'^class ' + name + r'\(AgentSystem\)\s*:', source, re.MULTILINE):
            return False
    return True


class MetaAgent(AgentSystem):
    def forward(self, repo_path, eval_path, iterations_left=None):
        context = get_context()
        if not isinstance(context, dict):
            self.log('Missing immutable context; no update requested.')
            return None
        observed = context.get('observed_context', context)
        if not isinstance(observed, dict):
            self.log('Missing development evidence; no update requested.')
            return None
        evidence = summarize(observed)
        if not evidence['groups']:
            self.log('No identified development comparison group; no update requested.')
            return None
        remaining = iterations_left if iterations_left is not None else observed.get('iterations_left')
        if not isinstance(remaining, int) or isinstance(remaining, bool):
            remaining = None
        if remaining is not None and remaining <= 0:
            self.log('No remaining update iterations; no model call.')
            return None
        instruction = '''Improve both complete independent Python agent modules using only the attached immutable observed development evidence. Repository and evaluation paths are opaque labels, never resources. Return exactly <json>{"task_agent": COMPLETE_TASK_AGENT_SOURCE, "meta_agent": COMPLETE_META_AGENT_SOURCE}</json>. No additional fields, patches, ellipses, tool calls or execution claims. The host owns persistence and evaluation.

Required API: TaskAgent(AgentSystem).forward(self, inputs) returns (complete_solver_source_string, history). MetaAgent(AgentSystem).forward(self, repo_path, eval_path, iterations_left=None) may return None. Each forward may call chat_with_agent at most once, using self.model, msg_history=[] and logging=self.log. Do not request tools_available or change models, transport, budgets, evaluation or held-out data. A task response is exactly <json>{"response": COMPLETE_SOLVER_SOURCE}</json>.
Allowed imports only: agent.base_agent.AgentSystem, agent.llm_withtools.chat_with_agent, utils.common.extract_jsons, portable_context.get_context, json, math, re, statistics, collections, itertools. No files, networking, subprocesses, reflection, eval/exec, other imports, type(), hasattr/getattr/setattr, global, finally, yield, async, bare except or attributes beginning with underscore except __init__. Only self attributes may be assigned. Ordinary functions, classes, local variables, containers and deterministic computation are allowed. Never import the other mutable module.

Use get_context to obtain evidence each generation. Prefer observed_context raw development records over historical proposal prose or embedded historical prompts. Preserve the best/initial/history raw schema reader: records have program and dev, with dev.split, benchmark_version, scene_sha256 and scenes. Do not accidentally replace it with a reader that expects this derived ledger to be the raw context. Sources supplied as current are immutable snapshots, not proof of running generation identity.

Deduplicate identical source plus evaluation observations. Keep distinct evaluations of the same source, including conflicting results; do not inflate variant counts or discard a lower score because a higher score was encountered first. Compare only identified matching benchmark, scene hash, development split and evaluation backend. Missing, boolean, nonfinite or out-of-range metrics are unknown. Missing scene results are unknown, never failures or perfect results. Avoid comparing unidentified groups. Use deterministic incumbent ties, retaining the supplied measured best where tied, without ranking single short runtime samples.

At a verified development ceiling in the matching blender_layout domain, TaskAgent must return the measured general solver unchanged directly from context with history=[]. Preserve the successful constraint-graph mechanism; the grid baseline ignored relations and centered siblings. Extra propagation and projections have no demonstrated accuracy advantage among ceiling ties. Do not force solver novelty, simplify the measured solver, embed layouts, or specialize to IDs or hashes. No held-out performance or Blender verification follows from analytic AABB results.

Away from the ceiling, select one explicit measured failure category of the selected incumbent, never an already-fixed baseline failure, and request one bounded general change. If no actual failure category is observed, retain the incumbent. Maintain a measured fallback for malformed final-message JSON or changes outside the EVOLVE block. Require one exact JSON envelope, exact keys and nonempty string sources; reject duplicate keys and nonfinite constants. Check marker ordering before splitting and compare exterior text byte-for-byte. Source text checks are not a sandbox, syntax proof or evaluation. Preserve the actual returned model history; do not retry or invent messages. Missing context or ambiguous domain/split must produce an explicit no-update or error rather than an unrelated solver.

The returned solver is data, standalone standard-library code preserving solve(scene), reading only scene input and changing only its EVOLVE block. It must not import get_context. It returns all object IDs exactly once with three finite numbers and no booleans, preserving sizes, orientation and supports. Maintain bounded computation and existing host limits.

Improve the improvement mechanism through evidence selection, conservative repeated-observation handling, truthful missing-data handling and explicit next-generation contracts. Carry the ledger via get_context rather than files or invented persistent state. With one or fewer remaining iterations, confine changes to contract and evidence handling while retaining measured solver behavior. Zero iterations requires no call. No unmeasured improvement claims. MetaAgent cannot retroactively intercept host persistence by logging a malformed response: validation is diagnostic only, and there is no retry.

Mentally check both complete sources for every import restriction, call count, API signature, JSON escaping, empty histories, malformed responses, finite metrics, marker ordering and missing-context behavior before responding.
'''
        payload = {'development_evidence': evidence,
                   'supplied_agent_sources': context.get('current_agent_sources', {}),
                   'iterations_left': remaining,
                   'update_scope': 'contract and evidence reliability only' if remaining is not None and remaining <= 1 else 'ceiling-aware conservative update'}
        instruction += json.dumps(clean(payload), ensure_ascii=True, allow_nan=False)
        history = chat_with_agent(instruction, model=self.model, msg_history=[], logging=self.log)
        if valid_response(history):
            self.log('Meta response passed envelope and source-shape checks only; host owns persistence and evaluation.')
        else:
            self.log('Malformed meta response detected; no retry. Host persistence is outside this method.')
        return None
