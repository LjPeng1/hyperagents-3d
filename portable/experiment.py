"""Bounded fresh-trace comparison; run with the sibling DREAM Python environment.

python -m portable.experiment --phase all --output workspace/comparison --timeout 300
An interrupted or failed paid step is deliberately not resumable. Completed
steps may be reused; ambiguous steps require investigation and a new experiment.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.util
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys

from portable import runtime

ROOT = Path(__file__).resolve().parents[1]
DREAM_ROOT = ROOT.parent / "dream-rsi-3d"
TASK_NAME = "blender_layout"
TASK = ROOT / "portable" / "task"
TASK_MANIFEST = ROOT / "portable" / "task_manifest.json"
TASK_FILES = tuple(json.loads(TASK_MANIFEST.read_text(encoding="utf-8")))
CALIBRATION_THRESHOLDS = {"saturation_score": 0.95, "saturation_strict": 0.8, "floor_score": 0.05}
_PRE_MESH_TIMEOUT = None
ARMS = ("hyper_adaptive", "hyper_fixed", "dream")
SCHEDULE = {"e0": 4, **{f"{arm}_{kind}{cycle}": cap for arm in ARMS
                         for cycle in (1, 2) for kind, cap in (("update", 1), ("online", 2))}}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def load_dream():
    spec = importlib.util.spec_from_file_location("comparison_dream_trial", DREAM_ROOT / "scripts" / "run_blender_trial.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_evaluator():
    spec = importlib.util.spec_from_file_location(f"comparison_{TASK_NAME}_evaluator", TASK / "evaluator.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def select_task(task_name):
    """Select an immutable task bundle; never rewrite its files or manifest."""
    global TASK_NAME, TASK, TASK_MANIFEST, TASK_FILES
    if task_name not in ("blender_layout", "mesh_edit"):
        raise ValueError(f"Unknown task: {task_name}")
    TASK_NAME = task_name
    basename = "task" if task_name == "blender_layout" else "mesh_edit"
    TASK = ROOT / "portable" / basename
    TASK_MANIFEST = ROOT / "portable" / f"{basename}_manifest.json"
    manifest = read(TASK_MANIFEST)
    if not isinstance(manifest, dict) or not manifest:
        raise RuntimeError("Task manifest must be a nonempty relative-path to SHA-256 mapping")
    TASK_FILES = tuple(manifest)


def task_hashes():
    manifest = read(TASK_MANIFEST)
    if not isinstance(manifest, dict) or not manifest:
        raise RuntimeError("Task manifest must be a nonempty relative-path to SHA-256 mapping")
    hashes = {}
    for name, expected in manifest.items():
        path = Path(name)
        if (not name or "\\" in name or ":" in name or path.is_absolute()
                or ".." in path.parts or path.as_posix() != name):
            raise RuntimeError(f"Unsafe task manifest path: {name!r}")
        target = TASK / path
        target.resolve().relative_to(TASK.resolve())
        current = TASK
        for part in path.parts:
            current /= part
            if current.is_symlink() or (hasattr(current, "is_junction") and current.is_junction()):
                raise RuntimeError(f"Linked task file is not allowed: {name}")
        hashes[name] = sha(target)
        if hashes[name] != expected:
            raise RuntimeError(f"Vendored task differs from its frozen manifest: {name}")
    if TASK_NAME == "mesh_edit":
        required = {"problem.txt", "init_program.py", "evaluator.py", "worker.py", "scenes.py"}
        if not required.issubset(manifest):
            raise RuntimeError(f"Mesh manifest omits required task files: {sorted(required - manifest.keys())}")
        assets = {p.relative_to(TASK).as_posix() for p in (TASK / "assets").rglob("*") if p.is_file()}
        if not any(name.endswith(".npz") for name in assets):
            raise RuntimeError("Mesh task requires frozen assets/*.npz data")
        if assets != {name for name in manifest if name.startswith("assets/")}:
            raise RuntimeError("Mesh manifest must cover every asset file, including all NPZ bytes")
    else:
        other = {name: sha(DREAM_ROOT / "tasks" / "blender_layout" / name) for name in manifest}
        if hashes != other:
            raise RuntimeError("Vendored task differs from DREAM's frozen task files")
    return hashes


def setup(output, timeout=300, task_name="blender_layout"):
    global _PRE_MESH_TIMEOUT
    select_task(task_name)
    output = Path(output).resolve()
    # All paid artifacts must remain inside this repository's ignored workspace.
    output.relative_to((ROOT / "workspace").resolve())
    output.mkdir(parents=True, exist_ok=True)
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    os.environ.update({"CODEX_EXEC_TIMEOUT_SEC": str(timeout), "DRSI_REPLAY_MAX_WORKERS": "1",
                       "DRSI_HISTORY_SEED_MODE": "copy", "DRSI_HISTORY_ARCHIVE_FILTER": "all",
                       "DRSI_EVAL_PYTHON": sys.executable})
    mesh = TASK_NAME == "mesh_edit"
    if mesh:
        # One timeout for the complete dev/heldout split, not 120 seconds per mesh.
        if _PRE_MESH_TIMEOUT is None:
            _PRE_MESH_TIMEOUT = (os.environ.get("EVALUATOR_TIMEOUT_SECONDS"),)
        os.environ["EVALUATOR_TIMEOUT_SECONDS"] = "120"
    elif _PRE_MESH_TIMEOUT is not None:
        previous, = _PRE_MESH_TIMEOUT
        if previous is None:
            os.environ.pop("EVALUATOR_TIMEOUT_SECONDS", None)
        else:
            os.environ["EVALUATOR_TIMEOUT_SECONDS"] = previous
        _PRE_MESH_TIMEOUT = None
    dependencies = {name: importlib.metadata.version(name) for name in ("numpy", "scipy")} if mesh else {}
    bundle = runtime.initial_bundle()
    frozen = {
        "version": 2, "task_name": TASK_NAME, "task_dir": TASK.relative_to(ROOT).as_posix(),
        "task_manifest": TASK_MANIFEST.relative_to(ROOT).as_posix(),
        "task_sha256": task_hashes(), "timeout_sec": timeout,
        "dependency_versions": dependencies,
        "evaluation_limits": {"outer_timeout_sec": 180 if mesh else 30,
                              "split_timeout_sec": 120 if mesh else float(os.environ.get("EVALUATOR_TIMEOUT_SECONDS", "8"))},
        "calibration_gate": dict(CALIBRATION_THRESHOLDS) if mesh else None,
        "model_override": os.environ.get("CODEX_MODEL"),
        "codex_extra_args": os.environ.get("CODEX_EXTRA_ARGS", ""),
        "initial_agent_sha256": {k: runtime.digest(v) for k, v in bundle.items()},
        "host_sha256": {str(p.relative_to(ROOT)): sha(p) for p in
                         (ROOT / "portable" / "experiment.py", ROOT / "portable" / "runtime.py", ROOT / "portable" / "worker.py")},
        "dream_transport_sha256": {name: sha(DREAM_ROOT / "dream_rsi" / "agents" / name)
                                     for name in ("codex.py", "codex_json.py")},
        "dream_core_sha256": {name: sha(DREAM_ROOT / name) for name in (
            "scripts/run_blender_trial.py", "dream_rsi/orchestrate.py", "dream_rsi/env/runner.py",
            "dream_rsi/env/trace_env.py", "dream_rsi/learning/discover.py",
            "dream_rsi/learning/heuristic_search/workflow.py", "dream_rsi/learning/heuristic_search/eval.py")},
        "episode_caps": SCHEDULE, "total_episode_cap": 22, "max_workers": 1,
        "grid": {"max_width": 2, "max_refine": 1}, "e0_max_steps": 4, "online_max_steps": 2,
        "policy_rewrites_per_update": 1, "automatic_retries": 0,
        "backend": "shared Codex transport; structured, tool-forbidden output adapters",
        "solver_selection": "development combined_score only; ties select later candidate",
        "hyper_inheritance": "Every valid child bundle is inherited unconditionally, not quality-qualified promotion",
        "dream_selection": "Official frozen-trace replay reward; official newer-round tie selection",
        "heldout": "Evaluated only after all three arms finish; never model feedback",
        "comparison": "End-to-end framework comparison; Hyper mutates task/meta code, DREAM mutates scheduling policy",
        "budget": "Episode caps are not equal tokens or compute. All failed episodes count; no paid automatic retries.",
    }
    plan_path = output / "protocol.json"
    if plan_path.exists():
        if read(plan_path)["frozen"] != frozen:
            raise RuntimeError("Frozen protocol/environment/source changed; use a new output directory")
    else:
        if any(output.iterdir()):
            raise RuntimeError("New comparison requires an empty output directory")
        write(plan_path, {"created_utc": now(), "frozen": frozen})
        write(output / "initial_bundle.json", bundle)
    dream = load_dream()
    config = dream.ExploreConfig(problem=TASK / "problem.txt", evaluator=TASK / "evaluator.py",
                                init_program=TASK / "init_program.py", eval_timeout_sec=180 if mesh else 30,
                                eval_python=sys.executable)
    cfg = dream.PipelineConfig(
        workspace=output / "e0", agent="codex_json", cycles=2,
        discover_rounds=1, discover_retry_rounds=0, discover_max_retry_batches=0,
        evaluate_incumbent=True, eval_timeout_sec=120, resume=False, stream=False,
        archive_each_cycle=True, update_baseline_champion=False, explore_max_steps=4,
        explore_max_workers=1, explore_max_width=2, explore_max_refine=1,
        explore_config=config, direction_mode="off")
    return output, cfg, dream, load_evaluator()


def step(output, name, action):
    """Persist intent before execution. Never infer an interrupted call was free."""
    if name not in SCHEDULE:
        raise ValueError(f"Unknown paid step: {name}")
    verify_task_snapshot(output)
    journal = output / "journal" / f"{name}.json"
    if journal.exists():
        previous = read(journal)
        if previous["status"] == "completed":
            return previous["result"]
        raise RuntimeError(f"Refusing to repeat ambiguous/failed paid step {name}: {journal}")
    row = {"step": name, "status": "started", "started_utc": now(), "episode_cap": SCHEDULE[name]}
    write(journal, row)
    try:
        result = action()
        verify_task_snapshot(output)
    except BaseException as exc:
        row.update(status="failed", ended_utc=now(), error=f"{type(exc).__name__}: {exc}")
        write(journal, row)
        write(output / "usage.json", collect_usage(output))
        raise
    row.update(status="completed", ended_utc=now(), result=result)
    write(journal, row)
    write(output / "usage.json", collect_usage(output))
    return result


def verify_task_snapshot(output):
    frozen = read(output / "protocol.json")["frozen"]
    if "task_name" in frozen:
        if frozen["task_name"] != TASK_NAME or frozen["task_sha256"] != task_hashes():
            raise RuntimeError("Task changed during the experiment; refusing further execution")


def require_step(output, name):
    path = output / "journal" / f"{name}.json"
    if not path.exists() or read(path)["status"] != "completed":
        raise RuntimeError(f"Complete {name} first")
    return read(path)["result"]


def candidate(output, name, program, proposal, dev):
    directory = output / "candidates" / name
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "program.py"
    shutil.copyfile(program, destination)
    (directory / "proposal.md").write_text(proposal, encoding="utf-8")
    row = {"id": name, "program": str(destination), "source_sha256": runtime.digest(destination.read_text(encoding="utf-8")),
           "file_sha256": sha(destination), "proposal": proposal, "dev": dev}
    write(directory / "candidate.json", row)
    return row


def best_candidate(candidates):
    if not candidates:
        raise RuntimeError("No candidates available")
    # Explicit order wins ties, never heldout or lexical filenames.
    return max(enumerate(candidates), key=lambda item: (float(item[1]["dev"]["combined_score"]), item[0]))[1]


def dream_candidates(output, name, directory, evaluator, *, allow_empty=False):
    rows = []
    for score in sorted(directory.glob("explore_b*/attempt_*/eval/score.json")):
        node = score.parent.parent
        program = node / "artifact" / "program.py"
        if program.exists():
            proposal = node / "proposal.md"
            rows.append(candidate(output, f"{name}_{node.parent.name}_{node.name}", program,
                                  proposal.read_text(encoding="utf-8") if proposal.exists() else "",
                                  evaluator.evaluate_split(program, "dev")))
    if not rows and not allow_empty:
        raise RuntimeError(f"No evaluated candidate in {directory}")
    return rows


def phase_e0(output, cfg, dream, evaluator):
    def action():
        ws = output / "e0"
        if ws.exists():
            raise RuntimeError("E0 must be fresh, not copied/recovered/resumed")
        dream._bootstrap_workspace(ws, cfg.explore_config)
        if TASK_NAME == "mesh_edit":
            # A floor outcome is a calibration result, not permission to retry.
            # The old wrapper rejects all-zero traces before they can be reported.
            result = dream._run_explore_stage(cfg, ws, label="E0", cycle=0)
            write(ws / "round_0_result.json", result)
        else:
            result = dream.checked_round(cfg, ws, 0)
        initial = candidate(output, "initial", TASK / "init_program.py", "Frozen initial program",
                            evaluator.evaluate_split(TASK / "init_program.py", "dev"))
        rows = dream_candidates(output, "e0", ws, evaluator, allow_empty=TASK_NAME == "mesh_edit")
        raw = read(ws / "raw_scores.json")
        return {"round": result, "initial": initial, "candidates": rows,
                "evaluated_nodes": sum(bool(n.get("raw", {}).get("evaluated")) for n in raw.get("nodes", [])),
                "best": best_candidate([initial, *rows]), "policy": read(ws / "live_cycle_manifest.json")["policy"]}
    try:
        result = step(output, "e0", action)
    except BaseException:
        if TASK_NAME == "mesh_edit":
            write(output / "calibration_report.json", {
                "task_name": TASK_NAME, "split": "dev", "status": "incomplete",
                "eligible_for_comparison": False, "reason": "E0 did not complete; inspect journal/e0.json. No automatic retry.",
                "usage": collect_usage(output)})
        raise
    if TASK_NAME == "mesh_edit":
        write(output / "calibration_report.json", {
            **calibration_decision(result), "task_name": TASK_NAME, "split": "dev", "status": "completed",
            "task_sha256": read(output / "protocol.json")["frozen"]["task_sha256"],
            "usage": collect_usage(output),
            "note": "E0 is a development-set pipeline qualification pilot, not an untouched validation set. The separate calibration split is reserved for non-model geometric probes. No heldout was evaluated."})
    return result


def calibration_decision(e0):
    """Pure dev-only gate. Recomputed from the journal before either paid arm."""
    candidates = e0["candidates"]
    observed = [e0["initial"], *candidates]
    if any(row["dev"].get("split") != "dev" for row in observed):
        raise ValueError("Calibration accepts development outcomes only")
    scores = [float(row["dev"]["combined_score"]) for row in observed]
    strict_rates = [float(row["dev"].get("strict_success_rate", 0)) for row in observed]
    if not all(math.isfinite(value) for value in scores + strict_rates):
        raise ValueError("Calibration requires finite development metrics")
    score, strict = max(scores), max(strict_rates)
    saturated = score >= CALIBRATION_THRESHOLDS["saturation_score"] or strict >= CALIBRATION_THRESHOLDS["saturation_strict"]
    floor = score <= CALIBRATION_THRESHOLDS["floor_score"]
    errors = sum(bool(row["dev"].get("error")) for row in observed)
    usable = (not e0["round"].get("failed") and e0.get("evaluated_nodes", 0) >= 2
              and len(candidates) >= 2 and not errors
              and any(float(row["dev"]["combined_score"]) > 0 for row in candidates))
    reasons = []
    if saturated:
        reasons.append("development scores are saturated")
    if floor:
        reasons.append("development scores are at the floor")
    if not usable:
        reasons.append("E0 did not provide at least two evaluated candidates with a positive result and no execution/evaluator failures")
    return {"best_score": score, "best_strict_success_rate": strict,
            "saturated": saturated, "floor": floor, "pipeline_usable": bool(usable),
            "eligible_for_comparison": bool(usable and not saturated and not floor),
            "thresholds": dict(CALIBRATION_THRESHOLDS), "reasons": reasons,
            "candidate_count": len(candidates), "evaluated_nodes": e0.get("evaluated_nodes", 0),
            "evaluator_error_count": errors,
            "candidates": [{"id": r["id"], "combined_score": r["dev"]["combined_score"],
                            "strict_success_rate": r["dev"].get("strict_success_rate")} for r in observed]}


def require_calibration(output, e0):
    if TASK_NAME == "mesh_edit":
        decision = calibration_decision(e0)
        if not decision["eligible_for_comparison"]:
            raise RuntimeError("Mesh E0 calibration gate blocks the full comparison: " + "; ".join(decision["reasons"])
                               + ". No automatic task change or model retry; inspect calibration_report.json.")


def observed_context(output, history, iterations_left):
    initial = require_step(output, "e0")["initial"]
    best = best_candidate([initial, *history])
    def observed(row):
        if row["dev"].get("split") != "dev":
            raise ValueError("Only development outcomes may enter model context")
        path = Path(row["program"])
        if sha(path) != row["file_sha256"]:
            raise RuntimeError("Observed program changed after evaluation")
        return {"id": row["id"], "program": path.read_text(encoding="utf-8"),
                "proposal": row["proposal"], "dev": row["dev"]}
    return {"domain": TASK_NAME, "problem": (TASK / "problem.txt").read_text(encoding="utf-8"),
            "initial": observed(initial), "best": observed(best),
            "history": [observed(row) for row in history], "iterations_left": iterations_left}


def phase_hyper(output, cfg, dream, evaluator):
    e0 = require_step(output, "e0")
    require_calibration(output, e0)
    for arm in ("hyper_adaptive", "hyper_fixed"):
        bundle = read(output / "initial_bundle.json")
        history = list(e0["candidates"])
        for cycle in (1, 2):
            context = observed_context(output, history, 3-cycle)
            def update(bundle=bundle, context=context, arm=arm, cycle=cycle):
                result = runtime.improve(bundle, context, output / arm / "calls" / f"update{cycle}",
                                         freeze_meta=(arm == "hyper_fixed"))
                if result["executed_meta_sha256"] != runtime.digest(bundle["meta"]):
                    raise RuntimeError("Executed meta does not match inherited bundle")
                child = result["bundle"]
                if any(result[f"{role}_sha256"] != runtime.digest(child[role]) for role in ("task", "meta")):
                    raise RuntimeError("Child bundle hash mismatch")
                if arm == "hyper_fixed" and child["meta"] != bundle["meta"]:
                    raise RuntimeError("Fixed-meta control changed its meta source")
                return result
            improved = step(output, f"{arm}_update{cycle}", update)
            bundle = improved["bundle"]
            def online(bundle=bundle, history=list(history), arm=arm, cycle=cycle):
                rows = []
                for attempt in (1, 2):
                    call = output / arm / "calls" / f"online{cycle}_{attempt}"
                    context = observed_context(output, [*history, *rows], 3-cycle)
                    result = runtime.generate(bundle, context, call)
                    if any(result[f"{role}_sha256"] != runtime.digest(bundle[role]) for role in ("task", "meta")):
                        raise RuntimeError("Generated solver used the wrong task/meta bundle")
                    program = call / "program.py"
                    program.write_text(result["program"], encoding="utf-8")
                    rows.append(candidate(output, f"{arm}_g{cycle}_{attempt}", program, result["proposal"],
                                          evaluator.evaluate_split(program, "dev")))
                return {"candidates": rows, "round_best": best_candidate(rows),
                        "cumulative_best": best_candidate([e0["initial"], *history, *rows]),
                        "used_task_sha256": runtime.digest(bundle["task"]),
                        "used_meta_sha256": runtime.digest(bundle["meta"])}
            online_result = step(output, f"{arm}_online{cycle}", online)
            history.extend(online_result["candidates"])


def phase_dream(output, cfg, dream, evaluator):
    e0 = require_step(output, "e0")
    require_calibration(output, e0)
    ws = output / "dream"
    if not ws.exists():
        shutil.copytree(output / "e0", ws)
    history = list(e0["candidates"])
    previous_policy = e0["policy"]
    for cycle in (1, 2):
        def update(cycle=cycle, previous_policy=previous_policy):
            if (ws / "learning" / f"iter{cycle-1:03d}").exists():
                raise RuntimeError("Refusing to reuse an unjournaled learning job")
            result = dream.run_cycle(replace(cfg, workspace=ws, explore_max_steps=2),
                                     cycle=cycle-1, start_stage="learning")
            selected = (result.get("discover") or {}).get("best_round")
            if not selected:
                raise RuntimeError("DREAM did not select a replay-qualified policy")
            expected = previous_policy["sha256"] if selected.get("incumbent") else selected["method_sha256"]
            policy_path = ws / "policy" / "method.py"
            actual = sha(policy_path) if policy_path.exists() else e0["policy"]["sha256"]
            if actual != expected:
                raise RuntimeError("Selected-to-promoted DREAM policy mismatch")
            return {"learning": result, "selected": selected, "expected_policy_sha256": expected}
        learned = step(output, f"dream_update{cycle}", update)
        def online(cycle=cycle, learned=learned, history=list(history)):
            result = dream.checked_round(replace(cfg, workspace=ws, explore_max_steps=2), ws, cycle)
            manifest = read(ws / "live_cycle_manifest.json")
            if manifest["policy"]["sha256"] != learned["expected_policy_sha256"]:
                raise RuntimeError("Selected-to-used DREAM policy mismatch")
            rows = dream_candidates(output, f"dream_g{cycle}", ws, evaluator)
            # E1 stays live until L1 archives it; E2 is final and has no unused L2.
            if cycle == 2:
                dream._archive_and_reset_workspace(ws, cycle=cycle)
            return {"round": result, "candidates": rows, "round_best": best_candidate(rows),
                    "cumulative_best": best_candidate([e0["initial"], *history, *rows]),
                    "used_policy": manifest["policy"], "selected_used_verified": True}
        online_result = step(output, f"dream_online{cycle}", online)
        history.extend(online_result["candidates"])
        previous_policy = online_result["used_policy"]


def _explore_logs(directory):
    for path in directory.glob("explore_b*/attempt_*/agent_log/codex*/run.json"):
        yield path


def collect_usage(output):
    """Only canonical episode roots; never copied E0/history/baseline logs."""
    grouped = {"shared_e0": list(_explore_logs(output / "e0"))}
    for arm in ("hyper_adaptive", "hyper_fixed"):
        grouped[arm] = list((output / arm / "calls").glob("*/logs/run.json"))
    paths = list((output / "dream" / "learning").glob("iter*/logs/r*/run.json"))
    for cycle in (1, 2):
        paths.extend(_explore_logs(output / "dream" / "history" / f"iter{cycle:03d}"))
    # A failed live exploration may not have written its final manifest yet.
    active = any((p := output / "journal" / f"dream_online{c}.json").exists()
                 and not (output / "dream" / "history" / f"iter{c:03d}").exists() for c in (1, 2))
    if active:
        paths.extend(_explore_logs(output / "dream"))
    grouped["dream"] = paths
    totals, rows, seen = {}, [], set()
    by_arm = {}
    for arm, paths in grouped.items():
        arm_rows = []
        for path in sorted(paths):
            identity = str(path.resolve())
            if identity in seen:
                continue
            seen.add(identity)
            row = read(path)
            usage = path.with_name("usage.json")
            row = {**row, "log": identity, "arm": arm,
                   "usage": read(usage) if usage.exists() else {"available": False}}
            arm_rows.append(row)
            for key, value in (row["usage"].get("total") or {}).items():
                totals[key] = totals.get(key, 0) + value
        by_arm[arm] = {"episodes": len(arm_rows),
                       "reported_token_totals": {key: sum((r["usage"].get("total") or {}).get(key, 0) for r in arm_rows)
                                                 for key in {k for r in arm_rows for k in (r["usage"].get("total") or {})}}}
        rows.extend(arm_rows)
    ambiguous = [read(p)["step"] for p in (output / "journal").glob("*.json") if read(p)["status"] != "completed"]
    return {"episodes": len(rows), "max_episode_cap": 22, "by_arm": by_arm,
            "reported_token_totals": totals, "usage_available": sum(bool(r["usage"].get("available")) for r in rows),
            "reported_duration_sec": sum(r.get("duration_sec", 0) for r in rows),
            "ambiguous_or_failed_steps": ambiguous, "runs": rows,
            "note": "Shared E0 counted once. Missing logs/usage are unknown, not free calls. Durations include CLI overhead."}


def phase_report(output, cfg, dream, evaluator):
    e0 = require_step(output, "e0")
    verify_task_snapshot(output)
    require_calibration(output, e0)
    rounds = {arm: [require_step(output, f"{arm}_online{cycle}") for cycle in (1, 2)] for arm in ARMS}
    # Validate completed lifecycle before reading any heldout outcomes.
    lifecycle = {}
    for arm in ("hyper_adaptive", "hyper_fixed"):
        first, second = [require_step(output, f"{arm}_update{cycle}") for cycle in (1, 2)]
        initial = read(output / "initial_bundle.json")
        expected = runtime.digest(initial["meta"]) if arm == "hyper_fixed" else first["meta_sha256"]
        if second["executed_meta_sha256"] != expected:
            raise RuntimeError("Second meta update did not execute inherited code")
        lifecycle[arm] = {"g1_meta_sha256": first["meta_sha256"],
                          "g2_executed_meta_sha256": second["executed_meta_sha256"],
                          "inherited_meta_executed": True,
                          "changed_meta_executed": expected != runtime.digest(initial["meta"]),
                          "meta_promotion_quality_qualified": False}
    selected = {"initial": e0["initial"], "e0": e0["best"]}
    for arm, arm_rounds in rounds.items():
        for cycle, result in enumerate(arm_rounds, 1):
            selected[f"{arm}_g{cycle}_online"] = result["round_best"]
            selected[f"{arm}_g{cycle}_cumulative"] = result["cumulative_best"]
    scores, cache = {}, {}
    for name, row in selected.items():
        program = Path(row["program"])
        if sha(program) != row["file_sha256"]:
            raise RuntimeError(f"Evaluated candidate changed: {program}")
        if row["file_sha256"] not in cache:
            cache[row["file_sha256"]] = evaluator.evaluate_split(program, "heldout")
        scores[name] = {**row, "heldout": cache[row["file_sha256"]]}
    usage = collect_usage(output)
    if usage["episodes"] > 22:
        raise RuntimeError("Observed episode count exceeds preregistered cap")
    report = {"protocol": read(output / "protocol.json"), "results": scores, "rounds": rounds,
              "lifecycle": lifecycle, "usage": usage, "limitations": [
                  "Single same-task pilot; saturated scores cannot establish a quality improvement or general RSI capability.",
                  "Hyper inherits valid child agents unconditionally; this is a mechanism test, not quality-qualified meta selection.",
                  "Fixed-meta ablation isolates editable meta code; DREAM comparison also changes the editable object and prompt interface.",
                  "Equal episode caps do not equal tokens, wall time or compute. Shared bootstrap cost is listed separately.",
                  "DREAM selects on frozen trace replay; online generations are fresh and need not reproduce counterfactual replay.",
                  "Heldout solver evaluation does not demonstrate policy/meta transfer; local files are not a secure hidden benchmark."]}
    verify_task_snapshot(output)
    write(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--task", choices=("blender_layout", "mesh_edit"), default="blender_layout")
    parser.add_argument("--phase", choices=("all", "e0", "hyper", "dream", "report"), default="all")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    output = args.output or ROOT / "workspace" / ("fresh-mesh-edit-comparison" if args.task == "mesh_edit" else "fresh-comparison")
    values = setup(output, args.timeout, args.task)
    for name, function in (("e0", phase_e0), ("hyper", phase_hyper), ("dream", phase_dream), ("report", phase_report)):
        if args.phase in ("all", name):
            print(f"[comparison] {name}", flush=True)
            function(*values)


if __name__ == "__main__":
    main()
