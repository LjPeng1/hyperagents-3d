"""Host-only reference scoring for fixed-topology surface repair.

Scenes use coordinates normalized by the source asset's maximum bounding-box
side. The worker receives only ``case['public']``. Clean vertices never enter
the candidate process. The process/AST/audit restrictions are not an OS sandbox.
Normal reversal against the clean reference is meaningful for this repair task,
not for arbitrary deformation with legitimate large rotations. No full
self-intersection or manifold-validity claim is made.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np

TASK_DIR = Path(__file__).resolve().parent
OUTPUT_LIMIT_BYTES = 32 * 1024 * 1024
STDERR_LIMIT_BYTES = 1024 * 1024
THRESHOLDS = {
    "protected_max_displacement": 1e-8,
    "collapse_absolute_area": 1e-12,
    "collapse_reference_area_fraction": 1e-4,
    "new_collapse_fraction_zero": 0.01,
    "new_reversed_fraction_zero": 0.05,
    "restoration_full_improvement": 0.75,
    "detail_full_normal_error_degrees": 8.0,
    "strict_max_rmse_ratio": 0.35,
    "strict_max_normal_error_degrees": 12.0,
}
COMPONENTS = ("restoration", "detail", "protection_gate", "geometry_gate")


def _load_scenes():
    # Never import the unrelated layout task's globally cached ``scenes``.
    path = TASK_DIR / "scenes.py"
    name = "_mesh_edit_scenes_" + hashlib.sha256(str(path).encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _stop_process_tree(proc):
    if proc.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            pass
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if proc.poll() is None:
        proc.kill()
    proc.wait(timeout=10)


def run_candidate(program_path, public, timeout=None):
    """Run one public case in a fresh isolated-mode interpreter, with quotas."""
    timeout = float(timeout if timeout is not None else os.environ.get("EVALUATOR_TIMEOUT_SECONDS", "120"))
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive and finite")
    root = Path(os.environ.get("MESH_EDIT_TMPDIR", str(TASK_DIR / ".runtime"))).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mesh-repair-", dir=root) as temporary:
        directory = Path(temporary)
        candidate = directory / "candidate.py"
        candidate.write_bytes(Path(program_path).resolve().read_bytes())
        payload = directory / "public.json"
        payload.write_text(json.dumps(public, allow_nan=False), encoding="utf-8")
        stdout_path, stderr_path = directory / "stdout.json", directory / "stderr.txt"
        kwargs = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
        env = dict(os.environ)
        env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
        started = time.monotonic()
        with payload.open("rb") as stdin, stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            proc = subprocess.Popen([sys.executable, "-I", str(TASK_DIR / "worker.py"), str(candidate)],
                                    stdin=stdin, stdout=stdout, stderr=stderr, cwd=directory, env=env, **kwargs)
            try:
                while proc.poll() is None:
                    if time.monotonic() - started > timeout:
                        raise TimeoutError(f"candidate exceeded {timeout:g} seconds")
                    if stdout_path.stat().st_size > OUTPUT_LIMIT_BYTES:
                        raise ValueError("candidate output exceeds 32 MB limit")
                    if stderr_path.stat().st_size > STDERR_LIMIT_BYTES:
                        raise ValueError("candidate diagnostics exceed 1 MB limit")
                    time.sleep(0.025)
            finally:
                if proc.poll() is None:
                    _stop_process_tree(proc)
        if stdout_path.stat().st_size > OUTPUT_LIMIT_BYTES:
            raise ValueError("candidate output exceeds 32 MB limit")
        if stderr_path.stat().st_size > STDERR_LIMIT_BYTES:
            raise ValueError("candidate diagnostics exceed 1 MB limit")
        if proc.returncode:
            error = stderr_path.read_text(encoding="utf-8", errors="replace")[-2000:]
            raise ValueError(f"candidate exited {proc.returncode}: {error}")
        return json.loads(stdout_path.read_text(encoding="utf-8"), object_pairs_hook=_unique_pairs,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"invalid constant {value}")))


def _vertices(value, count, label):
    # Check scalar types before NumPy conversion, which otherwise hides bools.
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != count:
        raise ValueError(f"{label}: expected {count} vertices")
    for row in value:
        if not isinstance(row, (list, tuple, np.ndarray)) or len(row) != 3:
            raise ValueError(f"{label}: every vertex must have three coordinates")
        if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, float, np.integer, np.floating)) for x in row):
            raise ValueError(f"{label}: coordinates must be real numbers, not booleans")
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (count, 3) or not np.isfinite(result).all():
        raise ValueError(f"{label}: coordinates must be finite N by 3 values")
    return result


def _face_geometry(vertices, faces):
    points = vertices[faces]
    cross = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    norms = np.linalg.norm(cross, axis=1)
    normals = np.divide(cross, norms[:, None], out=np.zeros_like(cross), where=norms[:, None] > 0)
    return norms * 0.5, normals


def _failure(case, exc, stage="output_validation"):
    return {"id": case["public"].get("id"), "valid": False, "strict_success": False,
            "combined_score": 0.0, "components": {key: 0.0 for key in COMPONENTS},
            "failure_stage": stage, "error": f"{type(exc).__name__}: {exc}"[:2000]}


@np.errstate(over="raise", invalid="raise", divide="raise")
def score_case(case, output):
    """Score one output; all clean-reference calculations remain in the host."""
    try:
        public = case["public"]
        count = len(public["vertices"])
        noisy = _vertices(public["vertices"], count, "input")
        clean = _vertices(case["reference"], count, "reference")
        if not isinstance(output, dict) or set(output) != {"vertices"}:
            raise ValueError("output must contain exactly the vertices key; faces and vertex order are immutable")
        edited = _vertices(output["vertices"], count, "output")
        raw_mask = public["editable"]
        if len(raw_mask) != count or any(not isinstance(x, (bool, np.bool_)) for x in raw_mask):
            raise ValueError("editable must be a boolean mask of length N")
        mask = np.asarray(raw_mask, dtype=bool)
        faces = np.asarray(public["faces"])
        if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in "iu" or not len(faces):
            raise ValueError("input faces must be a nonempty integer M by 3 array")
        if np.any(faces < 0) or np.any(faces >= count) or not mask.any():
            raise ValueError("invalid face indices or empty editable region")
        ref_area, ref_normal = _face_geometry(clean, faces)
        old_area, old_normal = _face_geometry(noisy, faces)
        area, normal = _face_geometry(edited, faces)
        affected = mask[faces].any(axis=1) & (ref_area > THRESHOLDS["collapse_absolute_area"])
        if not affected.any():
            raise ValueError("editable region has no nondegenerate reference faces")
        weights = np.zeros(count)
        for column in range(3):
            np.add.at(weights, faces[:, column], ref_area / 3.0)
        weights = weights[mask]
        if weights.sum() <= 0:
            raise ValueError("editable region has no reference area")
        def rmse(vertices):
            squared = np.sum((vertices[mask] - clean[mask]) ** 2, axis=1)
            return float(np.sqrt(np.dot(weights, squared) / weights.sum()))
        original_rmse, edited_rmse = rmse(noisy), rmse(edited)
        if original_rmse <= 1e-15:
            raise ValueError("repair case has no measurable input corruption")
        ratio = edited_rmse / original_rmse
        restoration = float(np.clip((1.0-ratio) / THRESHOLDS["restoration_full_improvement"], 0, 1))
        protected = float(np.max(np.linalg.norm(edited[~mask]-noisy[~mask], axis=1))) if (~mask).any() else 0.0
        protection_gate = float(protected <= THRESHOLDS["protected_max_displacement"])
        cutoff = np.maximum(THRESHOLDS["collapse_absolute_area"], ref_area * THRESHOLDS["collapse_reference_area_fraction"])
        collapsed, old_collapsed = area <= cutoff, old_area <= cutoff
        dots = np.clip(np.einsum("ij,ij->i", normal, ref_normal), -1, 1)
        angles = np.degrees(np.arccos(dots))
        angles[collapsed] = 180.0
        normal_error = float(np.average(angles[affected], weights=ref_area[affected]))
        detail = min(1.0, THRESHOLDS["detail_full_normal_error_degrees"] / max(normal_error, 1e-12))
        reversed_faces = (dots < 0) & ~collapsed
        old_reversed = (np.einsum("ij,ij->i", old_normal, ref_normal) < 0) & ~old_collapsed
        new_collapsed = collapsed & ~old_collapsed & affected
        new_reversed = reversed_faces & ~old_reversed & affected
        n = int(affected.sum())
        collapse_quality = max(0.0, 1.0-float(new_collapsed.sum())/n/THRESHOLDS["new_collapse_fraction_zero"])
        reversed_quality = max(0.0, 1.0-float(new_reversed.sum())/n/THRESHOLDS["new_reversed_fraction_zero"])
        geometry_gate = collapse_quality * reversed_quality
        combined = restoration ** 0.7 * detail ** 0.3 * protection_gate * geometry_gate
        strict = bool(ratio <= THRESHOLDS["strict_max_rmse_ratio"] and
                      normal_error <= THRESHOLDS["strict_max_normal_error_degrees"] and
                      protection_gate and not new_collapsed.any() and not new_reversed.any())
        counts = {"affected_faces": n,
                  "input_collapsed": int((old_collapsed & affected).sum()),
                  "output_collapsed": int((collapsed & affected).sum()),
                  "new_collapsed": int(new_collapsed.sum()),
                  "input_reversed": int((old_reversed & affected).sum()),
                  "output_reversed": int((reversed_faces & affected).sum()),
                  "new_reversed": int(new_reversed.sum())}
        metadata = case.get("metadata", {})
        return {"id": public.get("id"), "asset_id": metadata.get("asset_id"), "family": metadata.get("family"),
                "valid": True, "strict_success": strict, "combined_score": float(combined),
                "components": {"restoration": restoration, "detail": detail,
                               "protection_gate": protection_gate, "geometry_gate": geometry_gate},
                "input_rmse": original_rmse, "edited_rmse": edited_rmse, "rmse_ratio": float(ratio),
                "normal_error_degrees": normal_error, "protected_max_displacement": protected,
                "geometry": counts, "self_intersection_checked": False}
    except (ValueError, TypeError, KeyError, IndexError, OverflowError, FloatingPointError) as exc:
        return _failure(case, exc)


def evaluate_split(program_path, split="dev"):
    started = time.monotonic()
    budget = float(os.environ.get("EVALUATOR_TIMEOUT_SECONDS", "120"))
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError("split timeout must be positive and finite")
    deadline = started + budget
    scenes = _load_scenes()
    cases = scenes.get_cases(split)
    if not cases:
        raise ValueError("benchmark split cannot be empty")
    fingerprint = hashlib.sha256(json.dumps(cases, sort_keys=True, allow_nan=False).encode()).hexdigest()
    # Fail host setup explicitly rather than disguising an invalid benchmark as
    # a candidate failure. Identity must be a valid output, although its score is 0.
    for case in cases:
        check = score_case(case, {"vertices": case["public"]["vertices"]})
        if not check["valid"]:
            raise ValueError(f"invalid benchmark case {case['public'].get('id')}: {check['error']}")
    results = []
    execution_seconds = 0.0
    solver_processes = 0
    for case in cases:
        case_started = time.monotonic()
        remaining = deadline - case_started
        execution_started = None
        try:
            if remaining <= 0:
                raise TimeoutError("split execution budget exhausted; no worker started")
            execution_started = time.monotonic()
            solver_processes += 1
            output = run_candidate(program_path, case["public"], timeout=remaining)
        except (TimeoutError, ValueError, UnicodeError, subprocess.SubprocessError) as exc:
            result = _failure(case, exc, stage="candidate_execution")
        else:
            result = score_case(case, output)
        finally:
            if execution_started is not None:
                execution_seconds += time.monotonic() - execution_started
        result["case_elapsed"] = time.monotonic() - case_started
        results.append(result)
    families = {}
    for case, result in zip(cases, results):
        family = case.get("metadata", {}).get("family", "unknown")
        families.setdefault(family, []).append(result)
    execution_errors = sum(r.get("failure_stage") == "candidate_execution" for r in results)
    validation_errors = sum(r.get("failure_stage") == "output_validation" for r in results)
    return {"combined_score": float(np.mean([r["combined_score"] for r in results])),
            "strict_success_rate": float(np.mean([r["strict_success"] for r in results])),
            "valid_scene_rate": float(np.mean([r["valid"] for r in results])),
            "components": {key: float(np.mean([r["components"][key] for r in results])) for key in COMPONENTS},
            "by_family": {family: {"count": len(rows), "combined_score": float(np.mean([r["combined_score"] for r in rows])),
                                   "strict_success_rate": float(np.mean([r["strict_success"] for r in rows]))}
                          for family, rows in families.items()},
            "scenes": results, "split": split, "benchmark_version": scenes.VERSION,
            "scene_sha256": fingerprint, "evaluation_seconds": time.monotonic()-started,
            "solver_processes": solver_processes, "execution_seconds": execution_seconds,
            "execution_error_count": execution_errors, "validation_error_count": validation_errors,
            "error": f"{execution_errors} candidate execution failure(s), {validation_errors} invalid output(s)"
                     if execution_errors or validation_errors else None,
            "split_timeout_seconds": budget,
            "evaluation_backend": "reference_mesh_repair", "blender_verified": False,
            "self_intersection_checked": False, "thresholds": dict(THRESHOLDS),
            "coordinate_convention": "asset maximum bounding-box side equals one; no evaluator rescaling"}


def evaluate(program_path):
    return evaluate_split(program_path, "dev")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", required=True)
    parser.add_argument("--split", default="dev", choices=("dev", "calibration", "heldout"))
    args = parser.parse_args()
    print(json.dumps(evaluate_split(args.program, args.split), indent=2, allow_nan=False))
