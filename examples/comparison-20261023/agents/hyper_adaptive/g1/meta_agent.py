from agent.base_agent import AgentSystem
from agent.llm_withtools import chat_with_agent
from portable_context import get_context
import json
import math


def metric(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def summarize(context):
    observed = context.get('observed_context', context)
    if not isinstance(observed, dict):
        observed = {}
    candidates = [observed.get('initial'), observed.get('best')]
    history = observed.get('history', [])
    if isinstance(history, list):
        candidates.extend(history)
    unique = []
    seen = set()
    for record in candidates:
        if not isinstance(record, dict):
            continue
        dev = record.get('dev')
        source = record.get('program')
        if not isinstance(dev, dict) or dev.get('split') != 'dev' or not isinstance(source, str):
            continue
        key = (record.get('id'), source, dev.get('benchmark_version'), dev.get('scene_sha256'))
        if key in seen:
            continue
        seen.add(key)
        failures = []
        scenes = dev.get('scenes', [])
        if isinstance(scenes, list):
            for scene in scenes:
                if not isinstance(scene, dict):
                    continue
                if scene.get('strict_success') is not True:
                    failures.append({'id': scene.get('id'), 'valid': scene.get('valid'),
                                     'combined_score': scene.get('combined_score'),
                                     'components': scene.get('components'),
                                     'collisions': scene.get('collisions'),
                                     'unsatisfied_relations': [r for r in scene.get('relation_results', [])
                                                               if isinstance(r, dict) and r.get('satisfied') is not True]})
        unique.append({'id': record.get('id'), 'program': source,
                       'combined_score': metric(dev.get('combined_score')),
                       'valid_scene_rate': metric(dev.get('valid_scene_rate')),
                       'strict_success_rate': metric(dev.get('strict_success_rate')),
                       'components': dev.get('components'), 'failures': failures,
                       'scene_count': len(scenes) if isinstance(scenes, list) else None,
                       'benchmark_version': dev.get('benchmark_version'),
                       'scene_sha256': dev.get('scene_sha256'),
                       'evaluation_seconds': metric(dev.get('evaluation_seconds')),
                       'evaluation_backend': dev.get('evaluation_backend'),
                       'blender_verified': dev.get('blender_verified')})
    groups = {}
    for record in unique:
        key = (record['benchmark_version'], record['scene_sha256'])
        groups.setdefault(key, []).append(record)
    comparisons = []
    for key, group in groups.items():
        scored = [record for record in group if record['combined_score'] is not None]
        if not scored:
            continue
        ceiling = [record['id'] for record in scored
                   if record['combined_score'] == 1.0 and record['valid_scene_rate'] == 1.0
                   and record['strict_success_rate'] == 1.0]
        comparisons.append({'benchmark_version': key[0], 'scene_sha256': key[1],
                            'best_observed_score': max(record['combined_score'] for record in scored),
                            'perfect_record_ids': ceiling,
                            'policy': 'preserve measured solver; improve agent reliability' if ceiling
                                      else 'target observed failures with one bounded mechanism change'})
    return {'problem': observed.get('problem'), 'domain': observed.get('domain'),
            'records': unique, 'comparisons': comparisons,
            'context_iterations_left': observed.get('iterations_left')}


class MetaAgent(AgentSystem):
    def forward(self, repo_path, eval_path, iterations_left=None):
        context = get_context()
        if not isinstance(context, dict):
            raise ValueError('Expected immutable context dictionary')
        evidence = summarize(context)
        remaining = iterations_left if iterations_left is not None else evidence['context_iterations_left']
        instruction = '''Improve both complete Python agent modules using only the supplied immutable development evidence.
This is pure source generation. Repository and evaluation paths are opaque labels, not accessible resources.
Return exactly <json>{"task_agent": COMPLETE_TASK_AGENT_SOURCE, "meta_agent": COMPLETE_META_AGENT_SOURCE}</json>.
The host persists this model output directly. No patches, ellipses, extra files, execution claims or tool calls.

Mandatory API and execution contract:
TaskAgent(AgentSystem).forward(self, inputs) returns (complete_solver_source_string, history).
MetaAgent(AgentSystem).forward(self, repo_path, eval_path, iterations_left=None) may return None.
Each forward may call chat_with_agent at most once, using self.model, msg_history=[] and logging=self.log.
Never request tools_available or alter transport, model, evaluation, host budgets or held-out data.
A task model response is <json>{"response": COMPLETE_SOLVER_SOURCE}</json>.
A meta model response contains both complete agent sources as above.
Allowed imports only: agent.base_agent.AgentSystem, agent.llm_withtools.chat_with_agent,
utils.common.extract_jsons, portable_context.get_context, json, math, re, statistics, collections, itertools.
No files, network, subprocesses, reflection, eval/exec, other imports, type(), hasattr/getattr/setattr,
global, finally, yield, async, bare except or private attributes except __init__. Only self attributes
may be assigned. Ordinary local variables, functions, containers and deterministic computation are allowed.
The solver source is data returned by TaskAgent: it must be standalone, use only standard-library computation,
read only scene input, preserve solve(scene), and change only its EVOLVE block. It must not import get_context.

Evidence-driven update policy:
1. Prefer measured development records to statements inside historical proposal prose. Deduplicate records;
compare only the same benchmark and scene split. Missing results are unknown, never success or failure.
2. The observed grid baseline ignores relations and centers siblings. Constraint-graph solvers eliminate these
measured failures. Preserve that successful general mechanism instead of rediscovering the grid.
3. Multiple variants already reach the development ceiling. These ties establish no accuracy advantage for
extra propagation or projections. Single short runtime measurements do not establish a reliable speed ranking.
There is no held-out feedback and no Blender verification. Never describe a new variant as measured better.
4. At a verified development ceiling, retain the evaluated general solver deterministically for the matching
layout domain. Returning source from get_context is permitted; embedding development layouts or ID-specific
answers is not. An empty history is truthful when no model call is made. Improve the agents' evidence selection,
contract handling and fallback behavior instead of forcing unsupported solver novelty.
5. Away from the ceiling, select one actual measured failure category and request one bounded general change.
Keep a measured incumbent as fallback for malformed response data or changes outside the EVOLVE block.
Validate final-message JSON structure and string values without executing source. Text checks are not a sandbox,
a syntax proof, or a correctness test. Do not retry the model, fabricate history, or silently report validation
as evaluation. Preserve host-owned evaluation and persistence.
6. Improve MetaAgent itself by retaining this deduplicated evidence ledger, ceiling-aware policy and explicit
next-generation contracts. Carry evidence through get_context, not files or invented persistent state.
Keep future updates conservative when remaining iterations are few, without changing any host budget.
7. Preserve module independence: do not import the other mutable module. Sources supplied as current may be an
immutable earlier snapshot; do not infer that they prove which generation is currently running.

Before returning, mentally check both source files against every restriction, call count, API signature,
JSON escaping, empty history, malformed response, finite numeric metrics and missing-context behavior.
'''
        instruction += json.dumps({'evidence': evidence,
                                   'supplied_agent_sources': context.get('current_agent_sources', {}),
                                   'iterations_left': remaining}, ensure_ascii=True)
        chat_with_agent(instruction, model=self.model, msg_history=[], logging=self.log)
        return None
