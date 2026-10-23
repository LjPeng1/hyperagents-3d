"""Frozen local mesh-repair cases on CC0 Blender human base meshes.

Only ``public`` is passed to candidate processes. Reference geometry and random
seeds belong to the external evaluator. Split by anatomical asset family, not
by random perturbations of the same mesh. This is synthetic corruption of real
artist-authored geometry, not naturally defective production-asset ground truth.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

VERSION = "cc0-local-mesh-repair-v1"
ROOT = Path(__file__).resolve().parent
SPLITS = {
    "calibration": ("head_generic", "head_planar"),
    "dev": ("hand_realistic", "hand_stylized", "jaw_realistic", "jaw_stylized"),
    "heldout": ("foot_realistic", "foot_stylized", "skull_realistic", "skull_planar"),
}
FAMILIES = ("normal_noise", "sparse_spikes", "mixed_noise")
SEEDS = {"calibration": 67021, "dev": 83003, "heldout": 97159}


def _geometry(vertices, faces):
    edge = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edge = np.unique(np.sort(edge, axis=1), axis=0)
    lengths = np.linalg.norm(vertices[edge[:, 0]]-vertices[edge[:, 1]], axis=1)
    keep = lengths > 1e-12
    edge, lengths = edge[keep], lengths[keep]
    graph = coo_matrix((np.r_[lengths, lengths],
                       (np.r_[edge[:, 0], edge[:, 1]], np.r_[edge[:, 1], edge[:, 0]])),
                      shape=(len(vertices), len(vertices))).tocsr()
    cross = np.cross(vertices[faces[:, 1]]-vertices[faces[:, 0]],
                     vertices[faces[:, 2]]-vertices[faces[:, 0]])
    normals = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], cross)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    local_sum, degree = np.zeros(len(vertices)), np.zeros(len(vertices))
    for corner in (0, 1):
        np.add.at(local_sum, edge[:, corner], lengths)
        np.add.at(degree, edge[:, corner], 1)
    local_edge = local_sum/np.maximum(degree, 1)
    return graph, normals, local_edge


def make_case(asset_id, family, seed, case_id):
    asset = ROOT / "assets" / (asset_id + ".npz")
    with np.load(asset, allow_pickle=False) as data:
        clean = np.asarray(data["vertices"], dtype=np.float64).copy()
        faces = np.asarray(data["faces"], dtype=np.int64).copy()
    rng = np.random.default_rng(seed)
    # A proper random rotation and vertex permutation prevent index/axis shortcuts.
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    clean = clean @ rotation
    permutation = rng.permutation(len(clean))
    inverse = np.argsort(permutation)
    clean, faces = clean[permutation], inverse[faces]
    graph, normals, local_edge = _geometry(clean, faces)
    _, labels = connected_components(graph, directed=False)
    largest = np.flatnonzero(labels == np.argmax(np.bincount(labels)))
    center = int(rng.choice(largest))
    distances = dijkstra(graph, directed=False, indices=center)
    finite = distances[largest]
    radius = float(np.quantile(finite, 0.38))
    editable = np.isfinite(distances) & (distances <= radius)
    # Smooth taper leaves a narrow, clean transition near the ROI boundary.
    taper = np.zeros(len(clean))
    t = np.clip(distances[editable]/max(radius, 1e-12), 0, 1)
    taper[editable] = (1-t*t)**2
    scale = np.clip(local_edge, np.quantile(local_edge[largest], .2),
                    np.quantile(local_edge[largest], .8))
    displacement = np.zeros_like(clean)
    if family == "normal_noise":
        displacement = normals * (rng.normal(size=len(clean)) * scale * .65 * taper)[:, None]
    elif family == "sparse_spikes":
        active = editable & (rng.random(len(clean)) < .22)
        directions = rng.normal(size=clean.shape)
        directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
        displacement[active] = directions[active] * (scale[active]*2.8*taper[active])[:, None]
    elif family == "mixed_noise":
        gaussian = rng.normal(size=clean.shape)
        displacement = gaussian * (scale*.40*taper)[:, None]
        active = editable & (rng.random(len(clean)) < .07)
        displacement[active] += normals[active]*(scale[active]*1.6*taper[active])[:, None]
    else:
        raise ValueError("Unknown corruption family")
    displacement[~editable] = 0
    noisy = clean+displacement
    public = {
        "id": case_id, "task": "repair_surface", "vertices": noisy.tolist(),
        "faces": faces.tolist(), "editable": editable.tolist(),
        "noise_hint": {"family": family,
                       "local_edge_scale": float(np.median(local_edge[editable]))},
    }
    return {"public": public, "reference": clean.tolist(),
            "metadata": {"asset_id": asset_id, "family": family, "seed": seed,
                         "asset_sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
                         "vertex_count": len(clean), "triangle_count": len(faces),
                         "editable_count": int(editable.sum()),
                         "scale": "source maximum bbox side normalized to 1 before rotation",
                         "reference_kind": "uncorrupted artist base mesh, not ideal anatomy"}}


def get_cases(split="dev"):
    if split not in SPLITS:
        raise ValueError("Expected calibration, dev or heldout")
    return [make_case(asset, family, SEEDS[split]+101*ai+17*fi,
                      f"case_{ai*len(FAMILIES)+fi:02d}")
            for ai, asset in enumerate(SPLITS[split]) for fi, family in enumerate(FAMILIES)]


def get_scenes(split="dev"):
    """Public-only inputs; reference data must never enter solver context."""
    return [case["public"] for case in get_cases(split)]


def split_fingerprint(split="dev"):
    cases = get_cases(split)
    return hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest()
