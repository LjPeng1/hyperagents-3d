"""No-model geometric sanity checks on the dedicated calibration split."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from portable.mesh_edit import evaluator, scenes


def preflight(output):
    started = time.monotonic()
    cases = scenes.get_cases("calibration")
    rows = []
    for case in cases:
        clean = case["reference"]
        identity = evaluator.score_case(case, {"vertices": case["public"]["vertices"]})
        oracle = evaluator.score_case(case, {"vertices": clean})
        illegal = np.asarray(clean).copy()
        protected = np.flatnonzero(~np.asarray(case["public"]["editable"]))
        illegal[protected[0], 0] += 0.01
        changed_protected = evaluator.score_case(case, {"vertices": illegal})
        assert identity["combined_score"] == 0, identity
        assert oracle["combined_score"] == 1 and oracle["strict_success"], oracle
        assert changed_protected["combined_score"] == 0, changed_protected
        rows.append({"id": case["public"]["id"], "metadata": case["metadata"],
                     "identity": identity, "oracle": oracle,
                     "protected_region_attack": changed_protected})
    baseline = evaluator.evaluate_split(scenes.ROOT / "init_program.py", "calibration")
    if baseline["valid_scene_rate"] != 1:
        raise RuntimeError("Initial solver/worker is not operational on calibration cases")
    report = {"status": "passed", "benchmark": scenes.VERSION, "split": "calibration",
              "model_episodes": 0, "development_or_heldout_evaluated": False,
              "thresholds": evaluator.THRESHOLDS, "sanity_checks": rows,
              "initial_heuristic": baseline, "elapsed_seconds": time.monotonic()-started,
              "note": "Oracle is an evaluator test, not an agent result. No thresholds were selected using heldout scores."}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    result = preflight(parser.parse_args().output)
    print(json.dumps({"status": result["status"], "cases": len(result["sanity_checks"]),
                      "initial_score": result["initial_heuristic"]["combined_score"],
                      "initial_strict": result["initial_heuristic"]["strict_success_rate"],
                      "seconds": result["elapsed_seconds"]}))
