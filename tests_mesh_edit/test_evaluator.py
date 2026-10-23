"""Offline contract, geometry, leakage-boundary and worker-quota checks."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from portable.mesh_edit import evaluator


@pytest.fixture
def case():
    # A small detailed cap: flattening its legitimate bump is over-smoothing.
    clean = [[x * 0.5, y * 0.5, 0.2 if (x, y) == (1, 1) else 0.0]
             for y in range(3) for x in range(3)]
    faces = []
    for y in range(2):
        for x in range(2):
            a = y * 3 + x
            faces.extend([[a, a+1, a+4], [a, a+4, a+3]])
    noisy = copy.deepcopy(clean)
    noisy[4][2] += 0.15
    return {"public": {"id": "cap", "task": "repair_surface", "vertices": noisy,
                       "faces": faces, "editable": [i == 4 for i in range(9)]},
            "reference": clean, "metadata": {"family": "localized_normal_noise", "seed": "HOST_ONLY"}}


def test_identity_zero_and_clean_oracle_one(case):
    identity = evaluator.score_case(case, {"vertices": case["public"]["vertices"]})
    oracle = evaluator.score_case(case, {"vertices": np.asarray(case["reference"], dtype=np.float64)})
    assert identity["valid"] and identity["combined_score"] == 0
    assert identity["rmse_ratio"] == pytest.approx(1)
    assert oracle["combined_score"] == pytest.approx(1)
    assert oracle["strict_success"]
    assert oracle["self_intersection_checked"] is False
    assert "reference" not in oracle and "seed" not in oracle


def test_over_smoothing_cannot_win_by_erasing_real_shape(case):
    flattened = copy.deepcopy(case["reference"])
    flattened[4][2] = 0
    scored = evaluator.score_case(case, {"vertices": flattened})
    assert scored["valid"]
    assert scored["rmse_ratio"] > 1
    assert scored["combined_score"] == 0
    assert scored["normal_error_degrees"] > 8


def test_protected_vertex_violation_overrides_perfect_repair(case):
    moved = copy.deepcopy(case["reference"])
    moved[0][0] += 2e-8
    scored = evaluator.score_case(case, {"vertices": moved})
    assert scored["components"]["restoration"] == 1
    assert scored["components"]["protection_gate"] == 0
    assert scored["combined_score"] == 0 and not scored["strict_success"]


@pytest.mark.parametrize("coordinate", [True, "0.2", float("nan"), float("inf"), 1e308])
def test_invalid_or_unrepresentable_coordinates_are_rejected(case, coordinate):
    output = copy.deepcopy(case["reference"])
    output[4][2] = coordinate
    score = evaluator.score_case(case, {"vertices": output})
    assert not score["valid"] and score["combined_score"] == 0
    json.dumps(score, allow_nan=False)


def test_output_schema_cannot_replace_faces_or_drop_vertices(case):
    for output in ({"vertices": case["reference"], "faces": []},
                   {"vertices": case["reference"][:-1]}, case["reference"]):
        assert not evaluator.score_case(case, output)["valid"]


def test_common_numpy_scalar_types_are_accepted(case):
    output = [[np.float64(value) for value in row] for row in case["reference"]]
    assert evaluator.score_case(case, {"vertices": output})["combined_score"] == 1


def test_area_weighting_does_not_equate_tiny_and_large_faces():
    clean = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0], [2.1, 0, 0], [2, .1, 0]]
    noisy = copy.deepcopy(clean)
    noisy[2][2] = noisy[5][2] = .1
    case = {"public": {"id": "areas", "vertices": noisy, "faces": [[0,1,2], [3,4,5]],
                       "editable": [False, False, True, False, False, True]}, "reference": clean}
    fixed_large, fixed_small = copy.deepcopy(noisy), copy.deepcopy(noisy)
    fixed_large[2] = clean[2]
    fixed_small[5] = clean[5]
    assert evaluator.score_case(case, {"vertices": fixed_large})["rmse_ratio"] < .2
    assert evaluator.score_case(case, {"vertices": fixed_small})["rmse_ratio"] > .9


@pytest.mark.parametrize("position, metric", [([0, 0, 0], "new_collapsed"), ([0, -1, 0], "new_reversed")])
def test_new_degeneracy_and_reference_reversal_are_reported(position, metric):
    case = {"public": {"id": "triangle", "vertices": [[0,0,0], [1,0,0], [0,1,.2]],
                       "faces": [[0,1,2]], "editable": [False,False,True]},
            "reference": [[0,0,0], [1,0,0], [0,1,0]]}
    result = evaluator.score_case(case, {"vertices": [[0,0,0], [1,0,0], position]})
    assert result["geometry"][metric] == 1
    assert result["components"]["geometry_gate"] == 0
    assert not result["strict_success"]


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("MESH_EDIT_TMPDIR", str(tmp_path / "workers"))
    return tmp_path


def program(workspace, source):
    target = workspace / "program.py"
    target.write_text(source, encoding="utf-8")
    return target


def test_worker_numeric_libraries_and_numpy_result(workspace, case):
    path = program(workspace, '''import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
def solve(case):
    vertices = np.array(case['vertices'], dtype=float)
    tree = cKDTree(vertices)
    tree.query(vertices, k=1)
    spsolve(csr_matrix(np.eye(2)), np.ones(2))
    return {'vertices': vertices}
''')
    output = evaluator.run_candidate(path, case["public"], timeout=20)
    assert output["vertices"] == case["public"]["vertices"]


def test_worker_public_boundary_and_fresh_process(workspace, case):
    path = program(workspace, '''counter = 0
def solve(case):
    global counter
    counter += 1
    if counter != 1 or 'reference' in case or 'metadata' in case:
        raise ValueError('private data or cross-case state')
    return {'vertices': case['vertices']}
''')
    for _ in range(2):
        assert evaluator.run_candidate(path, case["public"], timeout=20)["vertices"] == case["public"]["vertices"]
    with pytest.raises(ValueError, match="documented public"):
        evaluator.run_candidate(path, case, timeout=20)


@pytest.mark.parametrize("source", [
    "import os\ndef solve(case): return {'vertices':case['vertices']}\n",
    "import socket\ndef solve(case): return {'vertices':case['vertices']}\n",
    "import subprocess\ndef solve(case): return {'vertices':case['vertices']}\n",
    "def solve(case):\n    return case.__class__\n",
    "import numpy as np\ndef solve(case):\n    return {'vertices':np.loadtxt('private.txt')}\n",
    "import numpy as np\ndef solve(case):\n    np.save('written.npy',np.ones(2))\n    return {'vertices':case['vertices']}\n",
])
def test_worker_denies_process_import_reflection_and_data_io(workspace, case, source):
    path = program(workspace, source)
    with pytest.raises(ValueError, match="candidate exited"):
        evaluator.run_candidate(path, case["public"], timeout=20)
    assert not list((workspace / "workers").glob("mesh-repair-*"))


def test_worker_cannot_read_existing_host_data_via_numpy(workspace, case):
    private = workspace / "private.txt"
    np.savetxt(private, np.asarray(case["reference"]))
    path = program(workspace, "import numpy as np\ndef solve(case):\n"
                   + "    return {'vertices':np.loadtxt(" + repr(str(private)) + ")}\n")
    with pytest.raises(ValueError, match="only read-only numeric/standard-library imports"):
        evaluator.run_candidate(path, case["public"], timeout=20)


def test_worker_timeout_reaps_and_cleans_directory(workspace, case):
    path = program(workspace, "def solve(case):\n    while True: pass\n")
    with pytest.raises(TimeoutError, match="exceeded"):
        evaluator.run_candidate(path, case["public"], timeout=0.25)
    assert not list((workspace / "workers").iterdir())


def test_worker_output_and_diagnostic_quotas(workspace, case, monkeypatch):
    monkeypatch.setattr(evaluator, "OUTPUT_LIMIT_BYTES", 128)
    path = program(workspace, "def solve(case): return {'vertices': [[0,0,0]]*10000}\n")
    with pytest.raises(ValueError, match="output exceeds"):
        evaluator.run_candidate(path, case["public"], timeout=20)
    monkeypatch.setattr(evaluator, "STDERR_LIMIT_BYTES", 128)
    path = program(workspace, "def solve(case):\n    print('x'*10000)\n    return {'vertices':[]}\n")
    with pytest.raises(ValueError, match="diagnostics exceed"):
        evaluator.run_candidate(path, case["public"], timeout=20)


def test_split_loader_is_independent_of_cached_layout_scenes(tmp_path, monkeypatch):
    (tmp_path / "scenes.py").write_text("VERSION='mesh-fixture'\ndef get_cases(split): return []\n", encoding="utf-8")
    monkeypatch.setattr(evaluator, "TASK_DIR", tmp_path)
    monkeypatch.setitem(sys.modules, "scenes", SimpleNamespace(VERSION="wrong-layout"))
    assert evaluator._load_scenes().VERSION == "mesh-fixture"


def test_default_dev_scoring_aggregates_failure_without_private_feedback(case, tmp_path, monkeypatch):
    second = copy.deepcopy(case)
    second["public"]["id"] = "failure"
    seen = []
    def cases(split):
        seen.append(split)
        return [case, second]
    monkeypatch.setattr(evaluator, "_load_scenes", lambda: SimpleNamespace(VERSION="test-mesh", get_cases=cases))
    def worker(path, public, timeout=None):
        assert "reference" not in public and "metadata" not in public
        if public["id"] == "failure":
            raise TimeoutError("quota")
        return {"vertices": case["reference"]}
    monkeypatch.setattr(evaluator, "run_candidate", worker)
    result = evaluator.evaluate(tmp_path / "unused.py")
    assert seen == ["dev"]
    assert result["combined_score"] == .5
    assert result["valid_scene_rate"] == .5
    assert result["strict_success_rate"] == .5
    assert len(result["scenes"]) == 2 and result["split"] == "dev"
    assert result["execution_error_count"] == 1 and result["solver_processes"] == 2
    assert "HOST_ONLY" not in json.dumps(result)


def test_split_budget_stops_spawning_workers(case, tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(evaluator.time, "monotonic", lambda: clock[0])
    monkeypatch.setenv("EVALUATOR_TIMEOUT_SECONDS", "2")
    monkeypatch.setattr(evaluator, "_load_scenes", lambda: SimpleNamespace(VERSION="test", get_cases=lambda split: [case]*3))
    calls = []
    def worker(path, public, timeout=None):
        calls.append(timeout)
        clock[0] += 3
        return {"vertices": case["reference"]}
    monkeypatch.setattr(evaluator, "run_candidate", worker)
    result = evaluator.evaluate(tmp_path / "unused.py")
    assert calls == [2]
    assert result["solver_processes"] == 1 and result["execution_error_count"] == 2
    assert result["scenes"][1]["case_elapsed"] == 0
    assert "no worker started" in result["scenes"][1]["error"]
