"""Replayable execution of a versioned TaskAgent or MetaAgent.

This is a conservative API/AST boundary, NOT a security sandbox. The host must
still enforce a subprocess timeout and an output-size limit. No native tools,
provider APIs, or upstream heavyweight dependencies are imported here.
"""
from __future__ import annotations

import ast
import builtins
import contextlib
import copy
import hashlib
import importlib
import json
import re
import sys
import types

MAX_SOURCE_BYTES = 100_000
MAX_MODEL_CALLS = 1
STDLIB_MODULES = frozenset({"json", "math", "re", "statistics", "collections", "itertools"})
SHIM_EXPORTS = {
    "agent.base_agent": frozenset({"AgentSystem"}),
    "agent.llm_withtools": frozenset({"chat_with_agent"}),
    "utils.common": frozenset({"extract_jsons"}),
    "portable_context": frozenset({"get_context"}),
}
FORBIDDEN_NAMES = frozenset({
    "open", "input", "exec", "eval", "compile", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "hasattr", "dir", "help", "breakpoint",
    "type", "__import__", "__builtins__", "BaseException", "SystemExit",
    "KeyboardInterrupt", "GeneratorExit", "exit", "quit", "memoryview",
})


def validate_source(source, role):
    """Return the parsed AST or raise ValueError before any candidate executes."""
    if role not in ("task", "meta"):
        raise ValueError("role must be task or meta")
    if not isinstance(source, str):
        raise ValueError("source must be text")
    if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise ValueError("source exceeds 100 KB")
    try:
        tree = ast.parse(source, filename=f"<{role}_agent>")
    except (SyntaxError, RecursionError, ValueError) as exc:
        raise ValueError(f"invalid Python source: {exc}") from exc
    expected = "TaskAgent" if role == "task" else "MetaAgent"
    definitions = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == expected]
    if len(definitions) != 1:
        raise ValueError(f"source must define exactly one top-level {expected}")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in STDLIB_MODULES and alias.name not in SHIM_EXPORTS:
                    raise ValueError(f"import is not allowed: {alias.name}")
                if alias.asname and alias.asname.startswith("__"):
                    raise ValueError("dunder import alias is not allowed")
        elif isinstance(node, ast.ImportFrom):
            if node.level or node.module not in STDLIB_MODULES | SHIM_EXPORTS.keys():
                raise ValueError(f"import is not allowed: {node.module}")
            for alias in node.names:
                if alias.name == "*" or alias.name.startswith("_"):
                    raise ValueError("wildcard/private imports are not allowed")
                if node.module in SHIM_EXPORTS and alias.name not in SHIM_EXPORTS[node.module]:
                    raise ValueError(f"unsupported shim export: {node.module}.{alias.name}")
                if alias.asname and alias.asname.startswith("__"):
                    raise ValueError("dunder import alias is not allowed")
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES or node.id.startswith("__"):
                raise ValueError(f"name is not allowed: {node.id}")
        elif isinstance(node, ast.Attribute):
            if isinstance(node.ctx, (ast.Store, ast.Del)) and not (isinstance(node.value, ast.Name) and node.value.id == "self"):
                raise ValueError("only instance attributes may be changed; module mutation is not allowed")
            if node.attr.startswith("_") and node.attr != "__init__":
                raise ValueError(f"private/reflection attribute is not allowed: {node.attr}")
            if node.attr in FORBIDDEN_NAMES:
                raise ValueError(f"attribute is not allowed: {node.attr}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if isinstance(node, ast.AsyncFunctionDef):
                raise ValueError("async execution is not supported")
            if node.name.startswith("__") and node.name != "__init__":
                raise ValueError(f"dunder definition is not allowed: {node.name}")
        elif isinstance(node, ast.arg):
            if node.arg.startswith("__") or node.arg in FORBIDDEN_NAMES:
                raise ValueError(f"argument name is not allowed: {node.arg}")
        elif isinstance(node, ast.ExceptHandler) and node.type is None:
            raise ValueError("bare except could intercept the replay boundary")
        elif isinstance(node, (ast.Try, getattr(ast, "TryStar", ast.Try))) and node.finalbody:
            raise ValueError("finally blocks could intercept the replay boundary")
        elif isinstance(node, (ast.Global, ast.AsyncFunctionDef, ast.Await, ast.Yield, ast.YieldFrom)):
            raise ValueError(f"unsupported execution construct: {type(node).__name__}")
    return tree


class PendingCall(BaseException):
    def __init__(self, request):
        self.request = request


class WorkerFault(BaseException):
    pass


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _json_copy(value):
    return json.loads(_canonical(value))


def _extract_jsons(response):
    """Match the upstream extractor's tagged JSON and fenced JSON behavior."""
    extracted = []
    for pattern in (r"<json>(.*?)</json>", r"```json(.*?)```"):
        for match in re.findall(pattern, response, re.DOTALL):
            try:
                extracted.append(json.loads(match.strip()))
            except json.JSONDecodeError:
                continue
    return extracted or None


class _QuietOutput:
    """Discard candidate diagnostics without corrupting protocol stdout."""
    def __init__(self):
        self.count = 0

    def write(self, value):
        self.count += len(value)
        if self.count > 65_536:
            raise WorkerFault("candidate diagnostic output exceeds 64 KB")
        return len(value)

    def flush(self):
        pass


def execute(payload):
    """Execute one stateless replay; this function never contacts a model."""
    request_hashes = []
    try:
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")
        role = payload.get("role")
        tree = validate_source(payload.get("source"), role)
        inputs = payload.get("inputs", {})
        model = payload.get("model", "")
        responses = payload.get("responses", [])
        if not isinstance(inputs, dict) or not isinstance(model, str):
            raise ValueError("inputs must be an object and model must be text")
        if not isinstance(responses, list) or len(responses) > MAX_MODEL_CALLS:
            raise ValueError("responses must contain at most one history array")
        if any(not isinstance(history, list) for history in responses):
            raise ValueError("every replay response must be a history array")
        inputs = _json_copy(inputs)
        responses = _json_copy(responses)

        def get_context():
            return copy.deepcopy(inputs)

        def log(*args, **kwargs):
            pass

        class AgentSystem:
            def __init__(self, model=model, chat_history_file=None):
                self.model = model
                self.log = log

        def chat_with_agent(instruction, model=model, msg_history=None, logging=None,
                            tools_available=None, **kwargs):
            if len(request_hashes) >= MAX_MODEL_CALLS:
                raise WorkerFault("model call budget exceeded: at most one call is allowed")
            if not isinstance(instruction, str) or not isinstance(model, str):
                raise WorkerFault("instruction and model must be text")
            history = [] if msg_history is None else msg_history
            if not isinstance(history, list):
                raise WorkerFault("msg_history must be an array")
            request = _json_copy({"instruction": instruction, "model": model, "msg_history": history})
            request_hashes.append(hashlib.sha256(_canonical(request).encode("utf-8")).hexdigest())
            index = len(request_hashes)-1
            if index >= len(responses):
                raise PendingCall(request)
            # tools_available is intentionally not executed: the frozen host
            # receives a model request and owns all model interaction.
            return copy.deepcopy(responses[index])

        modules = {}
        for name, exports in {
            "agent.base_agent": {"AgentSystem": AgentSystem},
            "agent.llm_withtools": {"chat_with_agent": chat_with_agent},
            "utils.common": {"extract_jsons": _extract_jsons},
            "portable_context": {"get_context": get_context},
        }.items():
            module = types.ModuleType(name)
            module.__dict__.update(exports)
            modules[name] = module
        for root in ("agent", "utils"):
            modules[root] = types.ModuleType(root)
        modules["agent"].base_agent = modules["agent.base_agent"]
        modules["agent"].llm_withtools = modules["agent.llm_withtools"]
        modules["utils"].common = modules["utils.common"]

        # Avoid exposing the worker's own module object to candidate mutation.
        # This copy is another integrity guard, not a claim of secure isolation.
        standard_modules = {}
        for name in STDLIB_MODULES:
            module = types.ModuleType(name)
            module.__dict__.update(importlib.import_module(name).__dict__)
            standard_modules[name] = module

        def limited_import(name, globals=None, locals=None, fromlist=(), level=0):
            if level:
                raise ImportError("relative imports are not supported")
            if name in STDLIB_MODULES:
                return standard_modules[name]
            if name in SHIM_EXPORTS:
                return modules[name] if fromlist else modules[name.split(".")[0]]
            raise ImportError(f"import is not allowed: {name}")

        allowed = (
            "abs all any bool bytes chr dict divmod enumerate filter float format frozenset "
            "int isinstance iter len list map max min next object ord pow print range repr "
            "reversed round set slice sorted str sum super tuple zip staticmethod classmethod "
            "property Exception ValueError TypeError RuntimeError KeyError IndexError "
            "ZeroDivisionError OverflowError ArithmeticError AssertionError "
            "NotImplementedError StopIteration"
        ).split()
        safe_builtins = {name: getattr(builtins, name) for name in allowed}
        safe_builtins["__build_class__"] = builtins.__build_class__
        safe_builtins["__import__"] = limited_import
        namespace = {"__builtins__": safe_builtins, "__name__": "portable_candidate"}
        quiet = _QuietOutput()
        with contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
            exec(compile(tree, f"<{role}_agent>", "exec"), namespace)
            expected = "TaskAgent" if role == "task" else "MetaAgent"
            agent_type = namespace[expected]
            if not isinstance(agent_type, type) or not issubclass(agent_type, AgentSystem):
                raise ValueError(f"{expected} must inherit AgentSystem")
            agent = agent_type(model=model)
            if role == "task":
                result = agent.forward(copy.deepcopy(inputs))
            else:
                result = agent.forward("version", "observed_feedback", iterations_left=inputs.get("iterations_left"))
        return {"status": "completed", "result": _json_copy(result), "request_hashes": request_hashes}
    except PendingCall as pending:
        return {"status": "request", "request": pending.request, "request_hashes": request_hashes}
    except (Exception, WorkerFault) as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"[:4096], "request_hashes": request_hashes}


def main():
    try:
        payload = json.load(sys.stdin)
        output = execute(payload)
    except Exception as exc:
        output = {"status": "error", "error": f"{type(exc).__name__}: {exc}"[:4096], "request_hashes": []}
    print(json.dumps(output, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
