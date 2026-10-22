"""Offline tests of role execution, replay, and conservative validation."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "portable" / "worker.py"
spec = importlib.util.spec_from_file_location("portable_worker_test", WORKER)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def run_worker(source, role="task", responses=None, inputs=None):
    payload = {"source": source, "role": role, "model": "offline-model",
               "inputs": inputs if inputs is not None else {"domain": "blender_layout", "scene": {"objects": []}},
               "responses": responses or []}
    result = subprocess.run([sys.executable, "-I", str(WORKER)], input=json.dumps(payload),
                            text=True, encoding="utf-8", capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_original_task_agent_parses_response_and_replay_hash_is_stable():
    source = (ROOT / "task_agent.py").read_text(encoding="utf-8")
    request = run_worker(source)
    assert request["status"] == "request"
    assert request["request"]["model"] == "offline-model"
    history = [{"role": "assistant", "text": '<json>{"response":{"cube":[1,2,3]}}</json>'}]
    completed = run_worker(source, responses=[history])
    assert completed["status"] == "completed"
    assert completed["result"] == [{"cube": [1, 2, 3]}, history]
    assert completed["request_hashes"] == request["request_hashes"]
    canonical = json.dumps(request["request"], sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    assert request["request_hashes"] == [hashlib.sha256(canonical.encode()).hexdigest()]


def test_original_meta_requests_and_finishes_without_native_tools():
    source = (ROOT / "meta_agent.py").read_text(encoding="utf-8")
    request = run_worker(source, "meta", inputs={"iterations_left": 2})
    assert request["status"] == "request"
    assert request["request"]["instruction"] == "Modify any part of the codebase at `version`."
    complete = run_worker(source, "meta", responses=[[{"role": "assistant", "text": "done"}]])
    assert complete["status"] == "completed" and complete["result"] is None


def test_changed_meta_code_changes_actual_prompt_and_uses_context():
    source = '''from agent.base_agent import AgentSystem
from agent.llm_withtools import chat_with_agent
from portable_context import get_context
class MetaAgent(AgentSystem):
    def forward(self, repo_path, eval_path, iterations_left=None):
        context = get_context()
        context['feedback'] = 'local mutation'
        return chat_with_agent('Improve support: ' + get_context()['feedback'], model=self.model)
'''
    result = run_worker(source, "meta", inputs={"feedback": "two collisions"})
    assert result["status"] == "request"
    assert result["request"]["instruction"] == "Improve support: two collisions"


def test_extra_call_is_rejected_even_with_exception_handler():
    source = '''from agent.base_agent import AgentSystem
from agent.llm_withtools import chat_with_agent
class TaskAgent(AgentSystem):
    def forward(self, inputs):
        first = chat_with_agent('one', model=self.model)
        try:
            return chat_with_agent('two', model=self.model)
        except Exception:
            return first
'''
    result = run_worker(source, responses=[[{"role": "assistant", "text": "first"}]])
    assert result["status"] == "error"
    assert "budget exceeded" in result["error"]
    assert len(result["request_hashes"]) == 1


@pytest.mark.parametrize("operation", [
    "import os", "from pathlib import Path", "import socket", "import subprocess",
    "open('secret.txt')", "eval('2+2')", "getattr(self, 'model')",
    "self.__class__", "import agent.base_agent as __hidden", "from math import *",
    "import json; json.dumps = str",
])
def test_io_reflection_and_unsafe_imports_rejected(operation):
    source = "from agent.base_agent import AgentSystem\nclass TaskAgent(AgentSystem):\n    def forward(self, inputs):\n        " + operation + "\n        return None\n"
    with pytest.raises(ValueError):
        worker.validate_source(source, "task")
    result = run_worker(source)
    assert result["status"] == "error" and result["request_hashes"] == []


def test_init_and_standard_library_helpers_work_without_original_dependencies():
    source = '''from agent.base_agent import AgentSystem
from collections import Counter
import math
class TaskAgent(AgentSystem):
    def __init__(self, model):
        super().__init__(model=model)
    def forward(self, inputs):
        print('candidate diagnostics')
        return dict(Counter([math.floor(1.2), 1])), []
'''
    result = run_worker(source)
    assert result["status"] == "completed"
    assert result["result"] == [{"1": 2}, []]


def test_source_limit_and_control_exception_interception_rejected():
    base = "from agent.base_agent import AgentSystem\nclass TaskAgent(AgentSystem):\n    def forward(self, inputs):\n        return None\n"
    with pytest.raises(ValueError, match="100 KB"):
        worker.validate_source(base + "#" * 100_000, "task")
    for fragment in ("try:\n            pass\n        except:\n            pass", "try:\n            pass\n        finally:\n            return None"):
        with pytest.raises(ValueError, match="replay boundary"):
            worker.validate_source(base.replace("return None", fragment), "task")


def test_nonfinite_result_is_an_error():
    source = "from agent.base_agent import AgentSystem\nclass TaskAgent(AgentSystem):\n    def forward(self, inputs):\n        return float('nan')\n"
    assert run_worker(source)["status"] == "error"
