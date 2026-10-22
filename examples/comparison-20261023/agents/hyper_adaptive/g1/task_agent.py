from agent.base_agent import AgentSystem
from agent.llm_withtools import chat_with_agent
from utils.common import extract_jsons
from portable_context import get_context
import json
import math
import re


def number(value, default=-1.0):
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return default


def block_parts(source):
    if not isinstance(source, str):
        return None
    start = '# EVOLVE-BLOCK-START'
    end = '# EVOLVE-BLOCK-END'
    if source.count(start) != 1 or source.count(end) != 1:
        return None
    before, tail = source.split(start, 1)
    body, after = tail.split(end, 1)
    if not re.search(r'^def solve\(scene\)\s*:', body, re.MULTILINE):
        return None
    return before + start, body, end + after


def observed_data(context):
    if not isinstance(context, dict):
        return {}
    observed = context.get('observed_context', context)
    return observed if isinstance(observed, dict) else {}


def records(observed):
    result = []
    seen = set()
    candidates = [observed.get('best'), observed.get('initial')]
    history = observed.get('history', [])
    if isinstance(history, list):
        candidates.extend(history)
    for record in candidates:
        if not isinstance(record, dict):
            continue
        source = record.get('program')
        dev = record.get('dev')
        if block_parts(source) is None or not isinstance(dev, dict):
            continue
        if dev.get('split') != 'dev':
            continue
        signature = (source, dev.get('benchmark_version'), dev.get('scene_sha256'))
        if signature in seen:
            continue
        seen.add(signature)
        result.append(record)
    return result


def quality(record):
    dev = record['dev']
    return (number(dev.get('combined_score')), number(dev.get('valid_scene_rate')),
            number(dev.get('strict_success_rate')))


def select_incumbent(observed):
    candidates = records(observed)
    if not candidates:
        return None
    reference = candidates[0]['dev']
    comparable = [record for record in candidates
                  if record['dev'].get('benchmark_version') == reference.get('benchmark_version')
                  and record['dev'].get('scene_sha256') == reference.get('scene_sha256')]
    return max(comparable, key=quality)


def saturated(record):
    if quality(record) != (1.0, 1.0, 1.0):
        return False
    scenes = record['dev'].get('scenes')
    if not isinstance(scenes, list) or not scenes:
        return False
    return all(isinstance(scene, dict) and scene.get('valid') is True
               and scene.get('strict_success') is True
               and number(scene.get('combined_score')) == 1.0 for scene in scenes)


def response_source(history):
    if not isinstance(history, list) or not history:
        return None
    message = history[-1]
    if not isinstance(message, dict) or not isinstance(message.get('text'), str):
        return None
    text = message['text']
    envelopes = re.findall(r'<json>\s*(.*?)\s*</json>', text, re.DOTALL)
    if envelopes:
        try:
            payload = json.loads(envelopes[-1])
        except (ValueError, TypeError):
            payload = None
        if isinstance(payload, dict) and isinstance(payload.get('response'), str):
            return payload['response']
    try:
        payloads = extract_jsons(text)
    except Exception:
        return None
    if isinstance(payloads, list) and payloads:
        payload = payloads[-1]
        if isinstance(payload, dict) and isinstance(payload.get('response'), str):
            return payload['response']
    return None


class TaskAgent(AgentSystem):
    def forward(self, inputs):
        observed = observed_data(get_context())
        incumbent = select_incumbent(observed)
        if incumbent is None:
            raise ValueError('No development-evaluated solver is available in the supplied context')
        fallback = incumbent['program']
        domain = inputs.get('domain', observed.get('domain')) if isinstance(inputs, dict) else observed.get('domain')
        # Preserve a measured ceiling instead of manufacturing an untested gain.
        # This reuses a general solve(scene) implementation, never saved coordinates.
        if domain == observed.get('domain') == 'blender_layout' and saturated(incumbent):
            self.log('Reusing the development-perfect general solver; no new evaluation or held-out claim.')
            return fallback, []
        evidence = []
        for record in records(observed):
            dev = record['dev']
            evidence.append({'id': record.get('id'), 'combined_score': dev.get('combined_score'),
                             'valid_scene_rate': dev.get('valid_scene_rate'),
                             'strict_success_rate': dev.get('strict_success_rate'),
                             'components': dev.get('components'), 'scenes': dev.get('scenes'),
                             'benchmark_version': dev.get('benchmark_version'),
                             'scene_sha256': dev.get('scene_sha256')})
        instruction = '''Generate a complete reusable Python solver source, not a layout and not an agent module.
The scene-output instruction in the problem applies to solve(scene), while your response contains source code.
Use only the supplied task and observed development evidence. Treat historical proposals as hypotheses;
measured dev records take precedence. Do not access files, tools, evaluation state, network or held-out data.
Keep all text outside the EVOLVE block byte-for-byte unchanged. Preserve solve(scene) and standard-library-only
execution. Return every object exactly once with three finite numerical coordinates, immutable sizes and supports.
Retain the incumbent's support-aware difference constraints, collision separation and proximity handling.
If measured deficiencies exist, make one targeted general change addressing the largest actual deficit.
Do not specialize to scene names, IDs, recorded coordinates, hashes or relation patterns. Preserve bounded work
under the supplied total timeout. Check clamping against support and collision scoring; do not confuse a failed
projection with infeasibility. Do not increase search limits merely to appear different. If there is no supported
reason to change the incumbent, return it unchanged. Mentally audit empty input, exact touching, support chains,
conflicting relations and numerical boundaries. Do not claim execution or measured improvement.
Return exactly <json>{"response": COMPLETE_SOLVER_SOURCE}</json> with no additional fields.
'''
        instruction += json.dumps({'task_input': inputs, 'problem': observed.get('problem'),
                                   'development_evidence': evidence, 'incumbent_source': fallback}, ensure_ascii=True)
        history = chat_with_agent(instruction, model=self.model, msg_history=[], logging=self.log)
        candidate = response_source(history)
        candidate_parts = block_parts(candidate)
        original_parts = block_parts(fallback)
        if candidate_parts is None:
            self.log('Missing or malformed source envelope; retaining the evaluated incumbent.')
            return fallback, history
        if candidate_parts[0] != original_parts[0] or candidate_parts[2] != original_parts[2]:
            self.log('Candidate changed text outside the EVOLVE block; retaining the evaluated incumbent.')
            return fallback, history
        # Structural checks are not execution or proof of geometric correctness.
        return candidate, history
