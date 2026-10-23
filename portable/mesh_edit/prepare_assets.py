"""Export pinned CC0 base meshes without evaluating modifiers or source scripts.

Run with ordinary Python; it launches Blender with --disable-autoexec. The
exporter requires only Blender's bundled Python and NumPy. No model is called.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import zipfile


ASSETS = (
    ("head_generic", "Head - Generic Topology", "calibration", "head"),
    ("head_planar", "Head - Planar", "calibration", "head"),
    ("hand_realistic", "Hand  - Realistic", "dev", "hand"),
    ("hand_stylized", "Hand - Stylized ", "dev", "hand"),
    ("jaw_realistic", "Jaw - Realistic", "dev", "jaw"),
    ("jaw_stylized", "Jaw - Stylized ", "dev", "jaw"),
    ("foot_realistic", "Foot - Realistic", "heldout", "foot"),
    ("foot_stylized", "Foot - Stylized ", "heldout", "foot"),
    ("skull_realistic", "Skull - Realistic", "heldout", "skull"),
    ("skull_planar", "Skull - Planar", "heldout", "skull"),
)
SOURCE_URL = (
    "https://mirror.blender.org/demo/asset-bundles/human-base-meshes/"
    "human-base-meshes-bundle-v1.4.1.zip"
)
ARCHIVE_SHA256 = "811f43accbb31a88266d932f8f5563b2d13586fca0ba2693aad1f5fe582b3515"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def deterministic_npz(path: Path, **arrays) -> None:
    """Avoid wall-clock ZIP metadata so repeated extraction is byte-stable."""
    import numpy as np

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, array in arrays.items():
            data = io.BytesIO()
            np.lib.format.write_array(data, array, allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data.getvalue(), compresslevel=9)


def export_assets(args) -> dict:
    import bpy
    import numpy as np

    bundle = args.bundle.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Library loading only selected objects avoids evaluating the bundle's scene
    # drivers. No source text blocks or evaluated modifier geometry are used.
    bpy.ops.wm.read_factory_settings(use_empty=True)
    names = [entry[1] for entry in ASSETS]
    with bpy.data.libraries.load(str(bundle), link=False) as (available, loaded):
        missing = sorted(set(names) - set(available.objects))
        if missing:
            raise ValueError(f"Pinned source objects are missing: {missing}")
        loaded.objects = names
    objects = {obj.name: obj for obj in loaded.objects}
    rows = []
    for asset_id, object_name, split, family in ASSETS:
        obj = objects[object_name]
        metadata = obj.asset_data
        if obj.type != "MESH" or metadata is None or metadata.license != "CC0":
            raise ValueError(f"Expected mesh with explicit CC0 asset metadata: {object_name!r}")
        mesh = obj.data
        world = obj.matrix_world.copy()
        vertices = np.asarray([tuple(world @ v.co) for v in mesh.vertices], dtype="<f8")
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
            raise ValueError(f"Invalid source coordinates: {asset_id}")
        mesh.calc_loop_triangles()
        faces = np.asarray([tuple(t.vertices) for t in mesh.loop_triangles], dtype="<i8")
        original_triangles = len(faces)
        # Retain winding and original vertex order. Zero-area triangles carry no
        # surface and would make downstream normal/curvature calculations invalid.
        triangles = vertices[faces]
        area2 = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                       triangles[:, 2] - triangles[:, 0]), axis=1)
        faces = faces[area2 > 0]
        used = np.unique(faces)
        remap = np.full(len(vertices), -1, dtype="<i8")
        remap[used] = np.arange(len(used))
        vertices = vertices[used]
        faces = remap[faces].astype("<i8", copy=False)
        if len(vertices) < 3 or not len(faces):
            raise ValueError(f"Empty surface: {asset_id}")
        lower, upper = vertices.min(axis=0), vertices.max(axis=0)
        center = (lower + upper) / 2
        scale = float((upper - lower).max())
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid normalization scale: {asset_id}")
        vertices = np.asarray((vertices - center) / scale, dtype="<f8")
        if (not np.isfinite(vertices).all() or faces.min() < 0
                or faces.max() >= len(vertices) or np.max(np.abs(vertices)) > 0.500000000001):
            raise ValueError(f"Export validation failed: {asset_id}")
        if not np.isclose(np.ptp(vertices, axis=0).max(), 1.0, atol=1e-12, rtol=0):
            raise ValueError(f"Normalization validation failed: {asset_id}")
        destination = out / (asset_id + ".npz")
        deterministic_npz(destination, vertices=vertices, faces=faces)
        with np.load(destination, allow_pickle=False) as saved:
            if not np.array_equal(saved["vertices"], vertices) or not np.array_equal(saved["faces"], faces):
                raise ValueError(f"NPZ roundtrip failed: {asset_id}")
        rows.append({
            "id": asset_id, "split": split, "asset_family": family,
            "file": destination.name, "sha256": sha256(destination), "bytes": destination.stat().st_size,
            "source_object": object_name,
            "license_metadata": {"license": metadata.license, "author": metadata.author,
                                 "description": metadata.description,
                                 "copyright": getattr(metadata, "copyright", "")},
            "source_vertices": len(mesh.vertices), "source_polygons": len(mesh.polygons),
            "source_triangulated_faces": original_triangles,
            "export_vertices": len(vertices), "export_faces": len(faces),
            "ignored_modifiers": [{"name": modifier.name, "type": modifier.type}
                                  for modifier in obj.modifiers],
            "removed_zero_area_triangles": original_triangles - len(faces),
            "removed_loose_vertices": len(mesh.vertices) - len(vertices),
            "exact_duplicate_coordinate_vertices": len(vertices) - len(np.unique(vertices, axis=0)),
            "welded_vertices": 0,
            "source_world_matrix": [list(row) for row in world],
            "normalization": {"center_world": center.tolist(), "divide_by": scale,
                              "target_max_bbox_extent": 1.0},
            "bbox": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
        })
    manifest = {
        "schema_version": 1, "kind": "cc0_human_base_meshes_fixed_topology",
        "source_bundle": {"filename": bundle.name, "sha256": sha256(bundle),
                          "bytes": bundle.stat().st_size, "download_url": SOURCE_URL,
                          "archive_sha256": ARCHIVE_SHA256,
                          "license_url": "https://creativecommons.org/publicdomain/zero/1.0/"},
        "exporter": {"script_sha256": sha256(Path(__file__)), "blender_version": bpy.app.version_string,
                     "numpy_version": np.__version__, "source_auto_execution": False},
        "processing": [
            "Use original base mesh datablock; do not evaluate any modifiers or add subdivision.",
            "Apply stored object world transform; triangulate using Blender mesh.calc_loop_triangles.",
            "Remove exactly zero-area triangles and vertices unreferenced by remaining triangles.",
            "Do not weld vertices; report exact duplicate coordinates.",
            "Center source bounding box and divide by its maximum extent, preserving aspect ratio.",
            "Save deterministic NPZ with vertices float64 and faces int64; no textures or materials.",
        ],
        "split_note": "Names and split fixed before model outputs; related style variants stay together. "
                      "This is a small curated body-part collection, not ten independent data sources.",
        "assets": rows,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "completed", "assets": len(rows), "manifest": str(args.manifest),
                      "source_bundle_sha256": manifest["source_bundle"]["sha256"]}))
    return manifest


def main(argv=None):
    here = Path(__file__).resolve().parent
    workspace = here.parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=workspace / "benchmarks/hand_reference_sources_20261004/blender_official/source/human-base-meshes-bundle-v1.4.1/human_base_meshes_bundle.blend")
    parser.add_argument("--blender", type=Path, default=workspace / "runtime/blender-4.5.11-windows-x64/blender.exe")
    parser.add_argument("--output-dir", type=Path, default=here / "assets")
    parser.add_argument("--manifest", type=Path, default=here / "asset_manifest.json")
    parser.add_argument("--blender-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not args.bundle.is_file():
        parser.error(f"Source bundle does not exist: {args.bundle}; download {SOURCE_URL}")
    if args.blender_worker:
        export_assets(args)
        return
    if not args.blender.is_file():
        parser.error(f"Blender does not exist: {args.blender}; supply --blender")
    command = [str(args.blender.resolve()), "--background", "--factory-startup", "--disable-autoexec",
               "--python-exit-code", "1", "--python", str(Path(__file__).resolve()), "--", "--blender-worker",
               "--bundle", str(args.bundle.resolve()), "--output-dir", str(args.output_dir.resolve()),
               "--manifest", str(args.manifest.resolve())]
    subprocess.run(command, check=True, timeout=180,
                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else None)
