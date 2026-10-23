"""Offline protocol tests: lifecycle, accounting and refusal of ambiguous calls."""
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
import shutil

import pytest

from portable import experiment as ex


@dataclass
class Config:
    workspace: Path
    explore_max_steps: int = 4
    explore_config: object = None


class Evaluator:
    def __init__(self):
        self.splits = []

    def evaluate_split(self, program, split):
        self.splits.append(split)
        # Deliberately give tied dev candidates different heldout outcomes.
        source = Path(program).read_text(encoding="utf-8")
        score = 0.2 if "INITIAL" in source else (0.1 if "low-heldout" in source and split == "heldout" else 1.0)
        return {"combined_score": score, "strict_success_rate": float(score == 1), "split": split}


def log_episode(path, status="ok", tokens=10, usage=True):
    ex.write(path / "run.json", {"status": status, "duration_sec": 2})
    if usage:
        ex.write(path / "usage.json", {"available": True, "total": {"input_tokens": tokens, "output_tokens": 2}})


@pytest.fixture
def trial(tmp_path, monkeypatch):
    output = tmp_path / "trial"
    output.mkdir()
    task = tmp_path / "task"
    task.mkdir()
    (task / "problem.txt").write_text("Fixed development geometry problem", encoding="utf-8")
    (task / "init_program.py").write_text("# INITIAL\ndef solve(scene): return {}\n", encoding="utf-8")
    monkeypatch.setattr(ex, "TASK", task)
    monkeypatch.setattr(ex, "TASK_NAME", "blender_layout")
    bundle = {"task": "T0", "meta": "M0"}
    ex.write(output / "initial_bundle.json", bundle)
    ex.write(output / "protocol.json", {"frozen": {"episode_caps": ex.SCHEDULE}})
    cfg = Config(output / "e0", explore_config=SimpleNamespace())
    events = []
    evaluator = Evaluator()

    def bootstrap(ws, config):
        events.append(("bootstrap",))
        ws.mkdir()

    def archive(ws, cycle):
        events.append(("archive", cycle))
        destination = ws / "history" / f"iter{cycle:03d}"
        destination.mkdir(parents=True)
        for p in [*ws.glob("explore_b*"), ws / "raw_scores.json", ws / "live_cycle_manifest.json"]:
            if p.exists():
                shutil.move(str(p), str(destination / p.name))
        return str(destination)

    def checked_round(config, ws, cycle):
        assert not (ws / "raw_scores.json").exists()
        events.append(("explore", cycle, config.explore_max_steps))
        for i in range(config.explore_max_steps):
            node = ws / f"explore_b{i//2}" / f"attempt_{i%2:02d}"
            program = node / "artifact" / "program.py"
            program.parent.mkdir(parents=True)
            program.write_text(f"# cycle{cycle} candidate{i}\ndef solve(scene): return {{}}\n", encoding="utf-8")
            ex.write(node / "eval" / "score.json", {"combined_score": 1})
            log_episode(node / "agent_log" / "codex_json")
        policy = ws / "policy" / "method.py"
        p = {"sha256": ex.sha(policy) if policy.exists() else "BASELINE", "source": str(policy) if policy.exists() else "baseline"}
        ex.write(ws / "raw_scores.json", {"nodes": []})
        ex.write(ws / "live_cycle_manifest.json", {"cycle": cycle, "policy": p})
        return {"failed": 0, "probe_budget": config.explore_max_steps}

    def run_cycle(config, cycle, start_stage):
        ws = config.workspace
        assert start_stage == "learning"
        assert (ws / "raw_scores.json").exists()
        events.append(("learn", cycle))
        log_episode(ws / "learning" / f"iter{cycle:03d}" / "logs" / "r0001")
        policy = ws / "policy" / "method.py"
        policy.parent.mkdir(exist_ok=True)
        policy.write_text(f"P{cycle+1}", encoding="utf-8")
        selected = {"method_sha256": ex.sha(policy), "incumbent": False}
        archive(ws, cycle)
        return {"discover": {"best_round": selected}}

    dream = SimpleNamespace(_bootstrap_workspace=bootstrap, checked_round=checked_round,
                            run_cycle=run_cycle, _archive_and_reset_workspace=archive)
    calls = []

    def improve(bundle, context, call_dir, freeze_meta=False):
        assert "heldout" not in str(context)
        calls.append(("meta", bundle.copy(), freeze_meta))
        log_episode(call_dir / "logs")
        proposed = bundle["meta"] + "+"
        child = {"task": bundle["task"] + "+", "meta": bundle["meta"] if freeze_meta else proposed}
        return {"bundle": child, "executed_meta_sha256": ex.runtime.digest(bundle["meta"]),
                "proposed_meta_sha256": ex.runtime.digest(proposed),
                **{f"{r}_sha256": ex.runtime.digest(child[r]) for r in ("task", "meta")}}

    def generate(bundle, context, call_dir):
        assert "heldout" not in str(context)
        calls.append(("task", bundle.copy(), None))
        log_episode(call_dir / "logs")
        return {"program": f"# {call_dir.name}\ndef solve(scene): return {{}}\n", "proposal": "Observed dev improvement",
                **{f"{r}_sha256": ex.runtime.digest(bundle[r]) for r in ("task", "meta")}}

    monkeypatch.setattr(ex.runtime, "improve", improve)
    monkeypatch.setattr(ex.runtime, "generate", generate)
    return SimpleNamespace(output=output, cfg=cfg, dream=dream, evaluator=evaluator, events=events, calls=calls,
                           args=(output, cfg, dream, evaluator))


def test_complete_22_episode_lifecycle_fresh_rounds_and_heldout_last(trial):
    ex.phase_e0(*trial.args)
    ex.phase_hyper(*trial.args)
    ex.phase_dream(*trial.args)
    assert trial.evaluator.splits and set(trial.evaluator.splits) == {"dev"}
    report = ex.phase_report(*trial.args)
    assert report["usage"]["episodes"] == 22
    assert report["usage"]["by_arm"]["shared_e0"]["episodes"] == 4
    assert [report["usage"]["by_arm"][a]["episodes"] for a in ex.ARMS] == [6, 6, 6]
    assert report["usage"]["reported_token_totals"]["input_tokens"] == 220
    assert report["lifecycle"]["hyper_adaptive"]["changed_meta_executed"] is True
    assert report["lifecycle"]["hyper_fixed"]["changed_meta_executed"] is False
    meta_calls = [r for r in trial.calls if r[0] == "meta"]
    assert [r[1]["meta"] for r in meta_calls] == ["M0", "M0+", "M0", "M0"]
    assert [e for e in trial.events if e[0] in ("learn", "archive", "explore")] == [
        ("explore", 0, 4), ("learn", 0), ("archive", 0), ("explore", 1, 2),
        ("learn", 1), ("archive", 1), ("explore", 2, 2), ("archive", 2)]
    assert not (trial.output / "dream" / "learning" / "iter002").exists()
    assert all(r["selected_used_verified"] for r in report["rounds"]["dream"])


def test_completed_phase_resume_makes_no_more_calls(trial):
    ex.phase_e0(*trial.args)
    ex.phase_hyper(*trial.args)
    ex.phase_dream(*trial.args)
    before = list(trial.events), list(trial.calls)
    ex.phase_e0(*trial.args)
    ex.phase_hyper(*trial.args)
    ex.phase_dream(*trial.args)
    assert (trial.events, trial.calls) == before
    assert ex.collect_usage(trial.output)["episodes"] == 22


@pytest.mark.parametrize("status", ["started", "failed"])
def test_ambiguous_paid_step_is_never_repeated(trial, status):
    ex.write(trial.output / "journal" / "e0.json", {"status": status, "step": "e0"})
    with pytest.raises(RuntimeError, match="Refusing to repeat"):
        ex.phase_e0(*trial.args)
    assert not trial.events


def test_failed_episode_counted_and_no_retry(trial, monkeypatch):
    ex.phase_e0(*trial.args)
    def fail(bundle, context, call_dir, freeze_meta):
        log_episode(call_dir / "logs", status="failed", usage=False)
        raise RuntimeError("model timeout")
    monkeypatch.setattr(ex.runtime, "improve", fail)
    with pytest.raises(RuntimeError, match="model timeout"):
        ex.phase_hyper(*trial.args)
    usage = ex.read(trial.output / "usage.json")
    assert usage["episodes"] == 5 and usage["usage_available"] == 4
    assert usage["ambiguous_or_failed_steps"] == ["hyper_adaptive_update1"]
    with pytest.raises(RuntimeError, match="Refusing to repeat"):
        ex.phase_hyper(*trial.args)


def test_usage_excludes_copied_e0_and_nested_histories(trial):
    ex.phase_e0(*trial.args)
    shutil.copytree(trial.output / "e0", trial.output / "dream")
    log_episode(trial.output / "e0" / "explore_b0" / "history" / "iter001" /
                "explore_b0" / "attempt_00" / "agent_log" / "codex_json", tokens=900)
    log_episode(trial.output / "dream" / "history" / "iter000" / "explore_b0" /
                "attempt_00" / "agent_log" / "codex_json", tokens=900)
    assert ex.collect_usage(trial.output)["episodes"] == 4
    assert ex.collect_usage(trial.output)["reported_token_totals"]["input_tokens"] == 40


def test_failed_unarchived_dream_online_logs_counted(trial):
    ex.phase_e0(*trial.args)
    node = trial.output / "dream" / "explore_b0" / "attempt_00"
    log_episode(node / "agent_log" / "codex_json", status="failed")
    ex.write(trial.output / "journal" / "dream_online1.json", {"status": "failed", "step": "dream_online1"})
    assert ex.collect_usage(trial.output)["episodes"] == 5


def test_dev_tie_selects_newest_even_if_heldout_worse(trial):
    e0 = ex.phase_e0(*trial.args)
    first, second = e0["candidates"][:2]
    a = {**first, "heldout": {"combined_score": 1}}
    b = {**second, "heldout": {"combined_score": 0}}
    assert ex.best_candidate([a, b])["id"] == b["id"]


def test_meta_execution_hash_mismatch_stops_before_online(trial, monkeypatch):
    ex.phase_e0(*trial.args)
    original = ex.runtime.improve
    def wrong(*args, **kwargs):
        return {**original(*args, **kwargs), "executed_meta_sha256": "WRONG"}
    monkeypatch.setattr(ex.runtime, "improve", wrong)
    with pytest.raises(RuntimeError, match="inherited bundle"):
        ex.phase_hyper(*trial.args)
    assert [c[0] for c in trial.calls] == ["meta"]


def test_dream_selected_used_hash_mismatch_stops(trial):
    ex.phase_e0(*trial.args)
    original = trial.dream.checked_round
    def wrong(config, ws, cycle):
        result = original(config, ws, cycle)
        manifest = ex.read(ws / "live_cycle_manifest.json")
        manifest["policy"]["sha256"] = "WRONG"
        ex.write(ws / "live_cycle_manifest.json", manifest)
        return result
    trial.dream.checked_round = wrong
    with pytest.raises(RuntimeError, match="Selected-to-used"):
        ex.phase_dream(*trial.args)
    assert not (trial.output / "journal" / "dream_update2.json").exists()


def test_report_cannot_touch_heldout_before_all_arms_finish(trial):
    ex.phase_e0(*trial.args)
    with pytest.raises(RuntimeError, match="Complete"):
        ex.phase_report(*trial.args)
    assert "heldout" not in trial.evaluator.splits


def test_changed_evaluated_source_rejected_before_model_context(trial):
    e0 = ex.phase_e0(*trial.args)
    Path(e0["candidates"][0]["program"]).write_text("tampered", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed after evaluation"):
        ex.phase_hyper(*trial.args)
    assert not trial.calls


def test_task_hashes_only_six_canonical_files(monkeypatch, tmp_path):
    task = tmp_path / "vendored"
    upstream = tmp_path / "dream" / "tasks" / "blender_layout"
    task.mkdir()
    upstream.mkdir(parents=True)
    for name in ex.TASK_FILES:
        (task / name).write_text(name, encoding="utf-8")
        (upstream / name).write_text(name, encoding="utf-8")
    (task / "LICENSE-APACHE-2.0.txt").write_text("license", encoding="utf-8")
    monkeypatch.setattr(ex, "TASK", task)
    monkeypatch.setattr(ex, "TASK_NAME", "blender_layout")
    monkeypatch.setattr(ex, "DREAM_ROOT", tmp_path / "dream")
    manifest = tmp_path / "task_manifest.json"
    ex.write(manifest, {name: ex.sha(task / name) for name in ex.TASK_FILES})
    monkeypatch.setattr(ex, "TASK_MANIFEST", manifest)
    assert set(ex.task_hashes()) == set(ex.TASK_FILES)
    (task / "evaluator.py").write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="differs"):
        ex.task_hashes()


@pytest.fixture
def mesh_trial(trial, monkeypatch):
    monkeypatch.setattr(ex, "TASK_NAME", "mesh_edit")
    ex.write(trial.output / "protocol.json", {"frozen": {"episode_caps": ex.SCHEDULE, "task_sha256": {}}})
    def explore(config, ws, label, cycle):
        result = trial.dream.checked_round(config, ws, cycle)
        ex.write(ws / "raw_scores.json", {"nodes": [{"raw": {"evaluated": True}} for _ in range(config.explore_max_steps)]})
        return result
    trial.dream._run_explore_stage = explore
    def set_metrics(score, strict):
        def evaluate(program, split):
            trial.evaluator.splits.append(split)
            return {"combined_score": score, "strict_success_rate": strict, "split": split}
        trial.evaluator.evaluate_split = evaluate
    trial.set_metrics = set_metrics
    set_metrics(0.6, 0.4)
    return trial


@pytest.mark.parametrize("score,strict,saturated,floor", [
    (0.95, 0.0, True, False), (0.5, 0.8, True, False), (0.05, 0.0, False, True),
    (0.0, 0.0, False, True), (0.949, 0.799, False, False), (0.051, 0.0, False, False)])
def test_mesh_dev_calibration_thresholds_and_no_automatic_paid_retry(mesh_trial, score, strict, saturated, floor):
    t = mesh_trial
    t.set_metrics(score, strict)
    ex.phase_e0(*t.args)
    report = ex.read(t.output / "calibration_report.json")
    assert report["split"] == "dev" and set(t.evaluator.splits) == {"dev"}
    assert report["saturated"] is saturated and report["floor"] is floor
    assert report["eligible_for_comparison"] is (not saturated and not floor)
    assert report["usage"]["episodes"] == 4
    before = list(t.events)
    ex.phase_e0(*t.args)
    assert t.events == before
    if saturated or floor:
        # An edited public summary cannot bypass the journal-derived decision.
        report["eligible_for_comparison"] = True
        ex.write(t.output / "calibration_report.json", report)
        for phase in (ex.phase_hyper, ex.phase_dream):
            with pytest.raises(RuntimeError, match="calibration gate blocks"):
                phase(*t.args)
        assert not t.calls
        assert not (t.output / "dream").exists()
        assert ex.collect_usage(t.output)["episodes"] == 4


def test_mesh_qualified_e0_continues_same_22_cap_with_mesh_domain(mesh_trial, monkeypatch):
    t = mesh_trial
    ex.phase_e0(*t.args)
    contexts = []
    original = ex.runtime.improve
    def inspect(bundle, context, call_dir, freeze_meta):
        contexts.append(context)
        assert context["domain"] == "mesh_edit"
        return original(bundle, context, call_dir, freeze_meta)
    monkeypatch.setattr(ex.runtime, "improve", inspect)
    ex.phase_hyper(*t.args)
    ex.phase_dream(*t.args)
    assert set(t.evaluator.splits) == {"dev"}
    report = ex.phase_report(*t.args)
    assert report["usage"]["episodes"] == 22
    assert len(contexts) == 4


def test_mesh_execution_failure_blocks_despite_midrange_scores(mesh_trial):
    t = mesh_trial
    original = t.dream._run_explore_stage
    def failed(*args, **kwargs):
        return {**original(*args, **kwargs), "failed": ["b0_a0"]}
    t.dream._run_explore_stage = failed
    ex.phase_e0(*t.args)
    report = ex.read(t.output / "calibration_report.json")
    assert not report["saturated"] and not report["floor"]
    assert not report["pipeline_usable"] and not report["eligible_for_comparison"]
    with pytest.raises(RuntimeError, match="calibration gate blocks"):
        ex.phase_hyper(*t.args)


def test_mesh_failed_phase_writes_incomplete_dev_report_and_refuses_retry(mesh_trial):
    t = mesh_trial
    def failed(*args, **kwargs):
        raise RuntimeError("transport stopped")
    t.dream._run_explore_stage = failed
    with pytest.raises(RuntimeError, match="transport stopped"):
        ex.phase_e0(*t.args)
    report = ex.read(t.output / "calibration_report.json")
    assert report["status"] == "incomplete" and report["split"] == "dev"
    assert not report["eligible_for_comparison"]
    with pytest.raises(RuntimeError, match="Refusing to repeat"):
        ex.phase_e0(*t.args)


@pytest.fixture
def setup_tree(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    dream = root / "dream-reference"
    # Register globals for restoration even though setup/select_task assign them.
    for name in ("TASK_NAME", "TASK", "TASK_MANIFEST", "TASK_FILES", "_PRE_MESH_TIMEOUT"):
        monkeypatch.setattr(ex, name, getattr(ex, name))
    monkeypatch.setattr(ex, "ROOT", root)
    monkeypatch.setattr(ex, "DREAM_ROOT", dream)
    source_paths = ["portable/experiment.py", "portable/runtime.py", "portable/worker.py"]
    dream_paths = ["dream_rsi/agents/codex.py", "dream_rsi/agents/codex_json.py", "scripts/run_blender_trial.py",
                   "dream_rsi/orchestrate.py", "dream_rsi/env/runner.py", "dream_rsi/env/trace_env.py",
                   "dream_rsi/learning/discover.py", "dream_rsi/learning/heuristic_search/workflow.py",
                   "dream_rsi/learning/heuristic_search/eval.py"]
    for base, names in ((root, source_paths), (dream, dream_paths)):
        for name in names:
            path = base / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# frozen host\n", encoding="utf-8")
    files = ("problem.txt", "init_program.py", "scenes.py", "worker.py", "evaluator.py", "export.py")
    for task_name in ("task", "mesh_edit"):
        directory = root / "portable" / task_name
        directory.mkdir()
        for name in files:
            (directory / name).write_text(name, encoding="utf-8")
        if task_name == "mesh_edit":
            (directory / "assets").mkdir()
            (directory / "assets" / "source.npz").write_bytes(b"frozen NPZ fixture")
        manifest = {p.relative_to(directory).as_posix(): ex.sha(p) for p in directory.rglob("*") if p.is_file()}
        ex.write(root / "portable" / f"{task_name}_manifest.json", manifest)
    shutil.copytree(root / "portable" / "task", dream / "tasks" / "blender_layout")
    versions = {"numpy": "2.3.0", "scipy": "1.16.0"}
    monkeypatch.setattr(ex.importlib.metadata, "version", lambda name: versions[name])
    monkeypatch.setattr(ex.runtime, "initial_bundle", lambda: {"task": "T0", "meta": "M0"})
    monkeypatch.setattr(ex, "load_evaluator", lambda: Evaluator())
    monkeypatch.setattr(ex, "load_dream", lambda: SimpleNamespace(ExploreConfig=lambda **kw: SimpleNamespace(**kw),
                                                                  PipelineConfig=lambda **kw: SimpleNamespace(**kw)))
    # Restore the environment changed by setup when this test ends.
    for name in ("EVALUATOR_TIMEOUT_SECONDS", "CODEX_EXEC_TIMEOUT_SEC", "DRSI_REPLAY_MAX_WORKERS",
                 "DRSI_HISTORY_SEED_MODE", "DRSI_HISTORY_ARCHIVE_FILTER", "DRSI_EVAL_PYTHON"):
        monkeypatch.delenv(name, raising=False)
    return SimpleNamespace(root=root, dream=dream, output=root / "workspace" / "new", versions=versions)


def test_mesh_setup_freezes_assets_dependencies_domain_and_limits(setup_tree):
    t = setup_tree
    output, cfg, _, _ = ex.setup(t.output, 300, "mesh_edit")
    frozen = ex.read(output / "protocol.json")["frozen"]
    assert frozen["task_name"] == "mesh_edit" and frozen["task_dir"] == "portable/mesh_edit"
    assert frozen["dependency_versions"] == t.versions
    assert frozen["evaluation_limits"] == {"outer_timeout_sec": 180, "split_timeout_sec": 120}
    assert "assets/source.npz" in frozen["task_sha256"]
    assert set(ex.TASK_FILES) == set(frozen["task_sha256"])
    assert cfg.explore_config.eval_timeout_sec == 180
    assert ex.os.environ["EVALUATOR_TIMEOUT_SECONDS"] == "120"
    assert not (t.dream / "tasks" / "mesh_edit").exists()
    # Completed-stage resumption must still use the originally frozen library versions.
    t.versions["numpy"] = "changed"
    with pytest.raises(RuntimeError, match="Frozen protocol"):
        ex.setup(t.output, 300, "mesh_edit")


def test_default_setup_keeps_blender_task_ungated_and_old_timeout(setup_tree):
    _, cfg, _, _ = ex.setup(setup_tree.output)
    frozen = ex.read(setup_tree.output / "protocol.json")["frozen"]
    assert frozen["task_name"] == "blender_layout"
    assert frozen["calibration_gate"] is None
    assert cfg.explore_config.eval_timeout_sec == 30
    assert "EVALUATOR_TIMEOUT_SECONDS" not in ex.os.environ
    assert frozen["total_episode_cap"] == 22


def test_switching_back_to_layout_restores_its_original_timeout(setup_tree, monkeypatch):
    monkeypatch.setenv("EVALUATOR_TIMEOUT_SECONDS", "17")
    ex.setup(setup_tree.output, task_name="mesh_edit")
    assert ex.os.environ["EVALUATOR_TIMEOUT_SECONDS"] == "120"
    ex.setup(setup_tree.output.with_name("layout"), task_name="blender_layout")
    assert ex.os.environ["EVALUATOR_TIMEOUT_SECONDS"] == "17"


@pytest.mark.parametrize("mutation", ["changed_asset", "new_unlisted_asset", "escape_manifest"])
def test_mesh_manifest_rejects_changed_or_unlisted_bytes_before_model_calls(setup_tree, mutation):
    t = setup_tree
    ex.setup(t.output, task_name="mesh_edit")
    directory = t.root / "portable" / "mesh_edit"
    if mutation == "changed_asset":
        (directory / "assets" / "source.npz").write_bytes(b"modified bytes")
    elif mutation == "new_unlisted_asset":
        (directory / "assets" / "unlisted.npz").write_bytes(b"extra")
    else:
        manifest = ex.read(ex.TASK_MANIFEST)
        manifest["../outside.npz"] = "0" * 64
        ex.write(ex.TASK_MANIFEST, manifest)
    with pytest.raises((RuntimeError, ValueError), match="differs|every asset|Unsafe"):
        ex.setup(t.output, task_name="mesh_edit")
    assert not (t.output / "journal").exists()


def test_task_drift_between_steps_is_caught_before_action(setup_tree):
    t = setup_tree
    ex.setup(t.output, task_name="mesh_edit")
    (ex.TASK / "assets" / "source.npz").write_bytes(b"changed during run")
    with pytest.raises(RuntimeError, match="differs"):
        ex.step(t.output, "e0", lambda: pytest.fail("Must refuse before calling action"))
    assert not (t.output / "journal").exists()
