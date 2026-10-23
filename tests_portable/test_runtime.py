"""Verify actual inherited-code execution without any model/network calls."""
from pathlib import Path
import pytest
from portable import runtime

SOLVER = 'def solve(scene):\n    return {}\n'


def test_task_bridge_honors_executed_postprocessing(tmp_path, monkeypatch):
    bundle = runtime.initial_bundle()
    bundle['task'] = bundle['task'].replace('return prediction, new_msg_history',
                                           'return prediction + "\\n# postprocessed", new_msg_history')
    monkeypatch.setattr(runtime, '_structured_call', lambda *a: {'proposal': 'p', 'program': SOLVER})
    result = runtime.generate(bundle, {'problem': 'test'}, tmp_path / 'call')
    assert result['program'].endswith('# postprocessed')
    assert result['task_sha256'] == runtime.digest(bundle['task'])
    with pytest.raises(RuntimeError, match='repeat'):
        runtime.generate(bundle, {}, tmp_path / 'call')


def test_changed_meta_is_executed_next_generation(tmp_path, monkeypatch):
    original = runtime.initial_bundle()
    changed = original['meta'].replace('Modify any part', 'SECOND GENERATION: Modify any part')
    seen = []
    def fake(role, request, bundle, context, call_dir):
        seen.append(request['instruction'])
        return {'task_agent': bundle['task'], 'meta_agent': changed}
    monkeypatch.setattr(runtime, '_structured_call', fake)
    first = runtime.improve(original, {}, tmp_path / 'm0')
    second = runtime.improve(first['bundle'], {}, tmp_path / 'm1')
    assert 'SECOND GENERATION' not in seen[0]
    assert 'SECOND GENERATION' in seen[1]
    assert second['executed_meta_sha256'] == first['meta_sha256']
    assert second['executed_meta_sha256'] != first['executed_meta_sha256']


def test_fixed_meta_discards_proposal(tmp_path, monkeypatch):
    bundle = runtime.initial_bundle()
    changed = bundle['meta'].replace('Modify any part', 'Changed: Modify any part')
    monkeypatch.setattr(runtime, '_structured_call', lambda *a: {'task_agent': bundle['task'], 'meta_agent': changed})
    result = runtime.improve(bundle, {}, tmp_path, freeze_meta=True)
    assert result['bundle']['meta'] == bundle['meta']
    assert result['proposed_meta_sha256'] != result['meta_sha256']


def test_task_files_identical_to_shared_dream_snapshot():
    reference = runtime.ROOT.parent / 'dream-rsi-3d' / 'tasks' / 'blender_layout'
    if not reference.exists():
        pytest.skip('Sibling DREAM checkout unavailable; manifest checked by runner')
    for name in ('problem.txt', 'init_program.py', 'scenes.py', 'worker.py', 'evaluator.py', 'export.py'):
        assert (runtime.ROOT / 'portable' / 'task' / name).read_bytes() == (reference / name).read_bytes()


def test_zero_call_task_is_evaluated_without_fabricated_model_usage(tmp_path, monkeypatch):
    bundle = runtime.initial_bundle()
    bundle['task'] = '''from agent.base_agent import AgentSystem
class TaskAgent(AgentSystem):
    def forward(self, inputs):
        return inputs['program'], []
'''
    def unexpected(*args):
        pytest.fail('A completed zero-call task must not request a model episode')
    monkeypatch.setattr(runtime, '_structured_call', unexpected)
    result = runtime.generate(bundle, {'program': SOLVER}, tmp_path / 'zero')
    assert result['program'] == SOLVER
    assert result['model_episodes'] == 0
    assert not (tmp_path / 'zero' / 'logs').exists()


@pytest.mark.parametrize('context,expected', [({}, 'blender_layout'), ({'domain': 'mesh_edit'}, 'mesh_edit')])
def test_runtime_respects_task_domain_without_changing_input(tmp_path, monkeypatch, context, expected):
    bundle = runtime.initial_bundle()
    seen = []
    def fake(role, request, bundle, observed, call_dir):
        seen.append(observed['domain'])
        assert expected in request['instruction']
        return {'proposal': 'mesh or layout program', 'program': SOLVER}
    monkeypatch.setattr(runtime, '_structured_call', fake)
    before = context.copy()
    runtime.generate(bundle, context, tmp_path / 'call')
    assert seen == [expected]
    assert context == before
