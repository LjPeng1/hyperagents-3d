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
    monkeypatch.setattr(ex, "DREAM_ROOT", tmp_path / "dream")
    manifest = tmp_path / "task_manifest.json"
    ex.write(manifest, {name: ex.sha(task / name) for name in ex.TASK_FILES})
    monkeypatch.setattr(ex, "TASK_MANIFEST", manifest)
    assert set(ex.task_hashes()) == set(ex.TASK_FILES)
    (task / "evaluator.py").write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="differs"):
        ex.task_hashes()
