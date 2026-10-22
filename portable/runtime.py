"""Execute versioned task/meta Python and use the shared DREAM Codex transport.

This is a restricted portable variant, not the native Docker/tool environment.
One task invocation permits zero or one model episode; meta updates use one.
An evaluated solver may be returned directly without calling a model. The changed MetaAgent
is executed when producing its descendants. Frozen host code supplies observed
context, validates outputs and evaluates geometry outside the agent process.
The AST checks and subprocess boundary are not a hostile-code security sandbox.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from dream_rsi.agents.codex import from_env as codex_from_env
from dream_rsi.agents.codex_json import _reject_tool_use, _strict_json

ROOT = Path(__file__).resolve().parents[1]
MAX_CONTEXT_BYTES = 750_000
MAX_SOURCE_BYTES = 100_000
API_DESCRIPTION = """
The two mutable files are task_agent.py and meta_agent.py, each a complete Python
module. Preserve TaskAgent(AgentSystem).forward(self, inputs) returning
(program_source_string, history), and MetaAgent(AgentSystem).forward(self,
repo_path, eval_path, iterations_left=None). Both may call chat_with_agent at
most ONCE. That function returns a list of messages with 'text' fields.
For TaskAgent the final text is <json>{"response": COMPLETE_SOLVER_SOURCE}</json>.
For MetaAgent it is <json>{"task_agent": SOURCE,"meta_agent": SOURCE}</json>.
The host persists MetaAgent's model output; MetaAgent may return None.
Available imports: agent.base_agent.AgentSystem, agent.llm_withtools.chat_with_agent,
utils.common.extract_jsons, portable_context.get_context, json, math, re,
statistics, collections, itertools. get_context() returns the full immutable
observed context below. AgentSystem provides self.model and self.log(message).
No files, tools, networking, subprocesses, reflection, eval/exec or other imports.
Avoid type(), hasattr/getattr/setattr, global, finally, yield, async, bare
except, and attributes beginning with underscore (except __init__). Use isinstance
and public helper methods. Only self attributes may be assigned, never module
attributes. Generator expressions are allowed. Restrictions are checked before execution.
Use ordinary class definitions, functions, data and deterministic computation.
Do not change the model, budget, transport, evaluation, or held-out data.
The next generation really executes your changed TaskAgent and MetaAgent code.
"""


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def initial_bundle(path=None):
    directory = Path(path) if path else ROOT
    return {role: (directory / f'{role}_agent.py').read_text(encoding='utf-8') for role in ('task', 'meta')}


def _worker(role, source, inputs, responses):
    payload = {'role': role, 'source': source, 'inputs': inputs,
               'model': 'fixed-host-model', 'responses': responses}
    cp = subprocess.run([sys.executable, '-I', str(ROOT / 'portable' / 'worker.py')],
                        input=json.dumps(payload, ensure_ascii=False, allow_nan=False),
                        capture_output=True, text=True, encoding='utf-8', errors='strict',
                        timeout=15, cwd=ROOT)
    if cp.returncode:
        raise ValueError(f'agent worker failed ({cp.returncode}): {cp.stderr[-2000:]}')
    if len(cp.stdout.encode('utf-8')) > 2_000_000:
        raise ValueError('worker output exceeds 2 MB')
    result = _strict_json(cp.stdout)
    if result.get('status') not in ('request', 'completed'):
        raise ValueError(f'agent execution error: {result}')
    return result


def _structured_call(role, request, bundle, context, call_dir):
    """One existing Codex CLI episode; no shell/tools authorized in its prompt."""
    log_dir = call_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=False)
    fields = ('proposal', 'program') if role == 'task' else ('task_agent', 'meta_agent')
    schema = {'type': 'object', 'properties': {f: {'type': 'string'} for f in fields},
              'required': list(fields), 'additionalProperties': False}
    schema_path = (log_dir / 'response_schema.json').resolve()
    write_json(schema_path, schema)
    instruction = (
        'This is a self-contained pure code-generation request. Return exactly one JSON '
        'object matching the output schema with complete source strings. Do not use ANY '
        'tools, browse, execute code, read/write files or spawn agents. All allowed context '
        'is attached below. File names and read/edit requests refer to the attached data '
        'and returned source strings, never filesystem access. Do not claim execution. '
        'The host enforces fixed output destinations and counts this single model episode.\n'
    )
    if role == 'task':
        instruction += ('TASK: Improve a reusable solve(scene) Python program. Return proposal '
                        '(brief explanation) and program (complete solver source, not a placement '
                        'or TaskAgent class). Interpret the TaskAgent response schema through '
                        'this fixed transport: your program will be delivered as its response.\n')
    else:
        instruction += ('META: Improve the task-solving agent AND its improvement mechanism '
                        'using only observed development outcomes. Return both complete files '
                        'task_agent and meta_agent. They must implement the API below.\n')
    body = {'executed_agent_request': request, 'current_agent_sources': bundle,
            'observed_context': context}
    prompt = instruction + API_DESCRIPTION + '\nCONTEXT_JSON\n' + json.dumps(body, ensure_ascii=False)
    raw = prompt.encode('utf-8')
    write_json(log_dir / 'context_manifest.json', {
        'role': role, 'prompt_bytes': len(raw), 'prompt_sha256': digest(prompt),
        'context_limit_bytes': MAX_CONTEXT_BYTES, 'truncated': False,
        'task_sha256': digest(bundle['task']), 'meta_sha256': digest(bundle['meta']),
        'heldout_feedback_supplied': False,
    })
    if len(raw) > MAX_CONTEXT_BYTES:
        raise ValueError('Full context exceeds limit; refusing silent truncation')
    codex = copy.copy(codex_from_env())
    if any(arg.split('=', 1)[0] in ('--output-schema', '--output-last-message', '-o') for arg in codex.extra_args):
        raise ValueError('Portable transport owns schema and output path')
    codex.extra_args = [*codex.extra_args, '--output-schema', str(schema_path)]
    codex.plain_exec = False
    result = codex.run(prompt=prompt, workspace=call_dir, node_dir=call_dir,
                       log_dir=log_dir, guide={}, run_cwd=call_dir, stage=role)
    if result.status != 'ok':
        raise RuntimeError(f'Codex episode failed: {result.message}')
    _reject_tool_use(log_dir)
    data = _strict_json((log_dir / 'codex_last_message.txt').read_text(encoding='utf-8'))
    if not isinstance(data, dict) or set(data) != set(fields):
        raise ValueError(f'Expected exactly {fields}')
    if any(not isinstance(v, str) or not v.strip() for v in data.values()):
        raise ValueError('Nonempty strings required')
    for key in ('program',) if role == 'task' else fields:
        if len(data[key].encode('utf-8')) > MAX_SOURCE_BYTES:
            raise ValueError('Source exceeds 100 KB')
        compile(ast.parse(data[key]), key + '.py', 'exec')
    return data


def _invoke(role, bundle, context, call_dir):
    from portable.worker import validate_source
    for name in ('task', 'meta'):
        validate_source(bundle[name], name)
    call_dir = Path(call_dir).resolve()
    if (call_dir / 'started.json').exists():
        raise RuntimeError(f'Refusing to repeat a possibly paid invocation: {call_dir}')
    call_dir.mkdir(parents=True, exist_ok=True)
    write_json(call_dir / 'started.json', {'role': role, 'task_sha256': digest(bundle['task']),
                                         'meta_sha256': digest(bundle['meta'])})
    write_json(call_dir / 'input_bundle.json', bundle)
    context = copy.deepcopy(context)
    context['domain'] = 'blender_layout'
    first = _worker(role, bundle[role], context, [])
    if role == 'task' and first['status'] == 'completed' and first.get('request_hashes') == []:
        result = first.get('result')
        if not isinstance(result, (list, tuple)) or len(result) != 2 or not isinstance(result[0], str) or result[1] != []:
            raise ValueError('Zero-call TaskAgent must return (program_source, empty history)')
        if len(result[0].encode('utf-8')) > MAX_SOURCE_BYTES:
            raise ValueError('Source exceeds 100 KB')
        compile(ast.parse(result[0]), 'program.py', 'exec')
        write_json(call_dir / 'completed.json', {'role': role, 'request_hashes': [],
                    'executed_source_sha256': digest(bundle[role]), 'model_episodes': 0})
        return {'program': result[0], 'proposal': 'TaskAgent returned solver source directly without a model call.',
                'model_episodes': 0}
    if first['status'] != 'request' or len(first.get('request_hashes', [])) != 1:
        raise ValueError('One model request is required per invocation')
    if first['request'].get('model') != 'fixed-host-model':
        raise ValueError('Agent cannot choose a different model')
    write_json(call_dir / 'executed_request.json', first)
    data = _structured_call(role, first['request'], bundle, context, call_dir)
    bridged = {'response': data['program']} if role == 'task' else data
    history = [{'role': 'assistant', 'text': '<json>' + json.dumps(bridged, ensure_ascii=False) + '</json>'}]
    completed = _worker(role, bundle[role], context, [history])
    if completed['status'] != 'completed' or completed.get('request_hashes') != first['request_hashes']:
        raise ValueError('Execution replay diverged or attempted an additional model call')
    if role == 'task':
        result = completed.get('result')
        if not isinstance(result, (list, tuple)) or len(result) != 2 or not isinstance(result[0], str):
            raise ValueError('TaskAgent must return (program_source, history)')
        # Honor the actual executed task agent's postprocessing, not just raw LLM output.
        data['program'] = result[0]
        compile(ast.parse(data['program']), 'program.py', 'exec')
    else:
        for name in ('task', 'meta'):
            validate_source(data[name + '_agent'], name)
    write_json(call_dir / 'completed.json', {'role': role, 'request_hashes': first['request_hashes'],
                                           'executed_source_sha256': digest(bundle[role]), 'model_episodes': 1})
    data['model_episodes'] = 1
    return data


def generate(bundle, context, call_dir):
    data = _invoke('task', bundle, context, call_dir)
    result = {**data, 'task_sha256': digest(bundle['task']), 'meta_sha256': digest(bundle['meta'])}
    write_json(Path(call_dir) / 'result.json', result)
    return result


def improve(bundle, context, call_dir, freeze_meta=False):
    data = _invoke('meta', bundle, context, call_dir)
    child = {'task': data['task_agent'], 'meta': bundle['meta'] if freeze_meta else data['meta_agent']}
    result = {'bundle': child, 'executed_meta_sha256': digest(bundle['meta']),
              'proposed_meta_sha256': digest(data['meta_agent']),
              'task_sha256': digest(child['task']), 'meta_sha256': digest(child['meta']),
              'freeze_meta': freeze_meta}
    write_json(Path(call_dir) / 'result.json', result)
    return result
