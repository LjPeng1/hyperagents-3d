"""Public-input-only surface-repair worker; no clean reference is loaded.

This AST/import/audit boundary reduces accidental file/network/process access.
It is NOT an OS security sandbox for hostile Python or native library code.
"""
from __future__ import annotations

import ast
import contextlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import scipy
import scipy.linalg
import scipy.ndimage
import scipy.optimize
import scipy.sparse
import scipy.sparse.linalg
import scipy.spatial
import scipy.spatial.distance

OUTPUT_LIMIT_BYTES = 32 * 1024 * 1024
ALLOWED_IMPORTS = {"numpy", "scipy", "math", "collections", "itertools", "functools", "heapq", "statistics"}
FORBIDDEN_NAMES = {"open", "input", "eval", "exec", "compile", "globals", "locals", "vars", "dir",
                   "getattr", "setattr", "delattr", "hasattr", "type", "__import__", "__builtins__",
                   "breakpoint", "help", "exit", "quit"}
FORBIDDEN_ATTRIBUTES = {"ctypes", "ctypeslib", "load_library", "system", "popen", "spawn", "fork"}


def validate_source(source):
    if len(source.encode("utf-8")) > 100_000:
        raise ValueError("candidate source exceeds 100 KB")
    tree = ast.parse(source, filename="<mesh_repair_candidate>")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            if getattr(node, "level", 0) or any(name.split(".")[0] not in ALLOWED_IMPORTS for name in names):
                raise ValueError("candidate imports are restricted to numeric and standard-library helpers")
            if any(alias.name.startswith("_") or alias.name == "*" or (alias.asname or "").startswith("__") for alias in node.names):
                raise ValueError("private/wildcard imports are not allowed")
        elif isinstance(node, ast.Name) and (node.id in FORBIDDEN_NAMES or node.id.startswith("__")):
            raise ValueError(f"reflection or execution name is not allowed: {node.id}")
        elif isinstance(node, ast.Attribute) and (node.attr.startswith("_") or node.attr in FORBIDDEN_ATTRIBUTES):
            raise ValueError(f"private/reflection/native attribute is not allowed: {node.attr}")
    return tree


def _audit_hook(event, args):
    if event == "open":
        path, mode, flags = args
        if isinstance(path, int):
            raise PermissionError("candidate file-descriptor access is not allowed")
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        try:
            resolved = Path(path).resolve()
        except (TypeError, ValueError, OSError):
            raise PermissionError("candidate file access is not allowed")
        if writing or not any(resolved.is_relative_to(root) for root in LIBRARY_ROOTS):
            raise PermissionError("only read-only numeric/standard-library imports are allowed")
    elif event.startswith(("socket.", "subprocess.", "ctypes.", "winreg.")) or event in {
        "os.system", "os.fork", "os.forkpty", "os.posix_spawn", "os.spawn", "os.startfile",
        "os.remove", "os.rename", "os.rmdir", "os.mkdir", "os.link", "os.symlink",
        "os.chmod", "os.chown", "os.chdir", "os.listdir", "os.scandir", "os.truncate",
    }:
        raise PermissionError(f"candidate operation is not allowed: {event}")


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"unsupported result value: {type(value).__name__}")


def main():
    public = json.load(sys.stdin)
    if not isinstance(public, dict) or set(public) - {"id", "task", "vertices", "faces", "editable", "noise_hint"}:
        raise ValueError("worker accepts only the documented public case")
    source = Path(sys.argv[1]).read_text(encoding="utf-8")
    tree = validate_source(source)
    code = compile(tree, "<mesh_repair_candidate>", "exec")
    global LIBRARY_ROOTS
    LIBRARY_ROOTS = tuple({Path(np.__file__).resolve().parent, Path(scipy.__file__).resolve().parent,
                          Path(os.__file__).resolve().parent})
    # Preloading covers common lazy paths. The audit hook still permits library
    # source/dynamic-module reads needed by numeric imports, never benchmark data.
    sys.addaudithook(_audit_hook)
    namespace = {"__name__": "mesh_repair_candidate"}
    with contextlib.redirect_stdout(sys.stderr):
        exec(code, namespace)
        result = namespace["solve"](public)
    output = json.dumps(result, default=_json_default, allow_nan=False)
    if len(output.encode("utf-8")) > OUTPUT_LIMIT_BYTES:
        raise ValueError("candidate output exceeds 32 MB")
    sys.stdout.write(output)


if __name__ == "__main__":
    main()
