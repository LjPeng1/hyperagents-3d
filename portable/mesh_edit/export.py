"""Geometry-only repair previews and independently reimported GLB verification.

The clean reference is loaded only by this post-evaluation visualization host.
It is never passed to the candidate. This command deliberately excludes heldout.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys


def _local_module(name):
    path = Path(__file__).resolve().parent / (name + ".py")
    spec = importlib.util.spec_from_file_location("_mesh_export_" + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _camera_plan(reference, noisy, faces, editable):
    import numpy as np

    roi = reference[editable]
    center = roi.mean(axis=0)
    triangles = reference[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    affected = editable[faces].all(axis=1)
    direction = normals[affected].sum(axis=0)
    if np.linalg.norm(direction) < 1e-9:
        direction = center - reference.mean(axis=0)
    if np.linalg.norm(direction) < 1e-9:
        direction = np.array([0.0, -1.0, 0.0])
    direction /= np.linalg.norm(direction)
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(direction @ up)) > .9:
        up = np.array([0.0, 1.0, 0.0])
    tangent = np.cross(up, direction)
    tangent /= np.linalg.norm(tangent)
    all_source = np.concatenate((reference, noisy))
    full_center = (all_source.min(axis=0) + all_source.max(axis=0)) / 2
    full_scale = float(np.linalg.norm(np.ptp(all_source, axis=0)) * 1.12)
    roi_source = np.concatenate((reference[editable], noisy[editable]))
    roi_scale = max(.12, float(np.linalg.norm(np.ptp(roi_source, axis=0)) * 1.10))
    cameras = []
    for index, angle in enumerate((0.0, -.42, .42)):
        facing = direction * math.cos(angle) + tangent * math.sin(angle)
        for crop, target, scale in (("full", full_center, full_scale), ("roi", center, roi_scale)):
            cameras.append({"view": index, "crop": crop, "target": target.tolist(),
                            "direction": facing.tolist(), "up_hint": up.tolist(),
                            "ortho_scale": scale})
    return cameras


def _contact_sheet(output, metadata, metrics):
    from PIL import Image, ImageDraw, ImageFont

    width, height, header, label = 640, 560, 112, 38
    sheet = Image.new("RGB", (width * 3, header + (height + label) * 6), "#f0f2f5")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 25)
        small = ImageFont.truetype("arial.ttf", 19)
    except OSError:
        font, small = ImageFont.load_default(size=25), ImageFont.load_default(size=19)
    draw.text((20, 10), f"{metadata['case_id']} | {metadata['asset_id']} | {metadata['family']}", fill="#172533", font=font)
    draw.text((20, 44), "Geometry only. Reference = uncorrupted artist mesh (proxy GT), not anatomical ground truth.", fill="#334454", font=small)
    draw.text((20, 74), f"RMSE ratio {metrics['rmse_ratio']:.3f} | normal error {metrics['normal_error_degrees']:.2f} deg | "
              f"protected max displacement {metrics['protected_max_displacement']:.2e}", fill="#334454", font=small)
    names = (("noisy", "Noisy input"), ("repaired", "Candidate repair"), ("reference", "Uncorrupted reference (proxy GT)"))
    for view in range(3):
        for crop_index, crop in enumerate(("full", "roi")):
            y = header + (view * 2 + crop_index) * (height + label)
            for column, (name, title) in enumerate(names):
                x = column * width
                draw.text((x + 12, y + 7), f"{title} | view {view + 1} / {crop}", fill="#172533", font=small)
                with Image.open(output / f"{name}_{crop}_v{view}.png") as picture:
                    sheet.paste(picture.convert("RGB").resize((width, height)), (x, y + label))
    sheet.save(output / "contact_sheet.png")


def export_repair(program, output, blender, split="calibration", case_index=0):
    import numpy as np

    if split not in ("calibration", "dev"):
        raise ValueError("Visual export is restricted to calibration/dev; heldout stays sealed")
    scenes, evaluator = _local_module("scenes"), _local_module("evaluator")
    cases = scenes.get_cases(split)
    if not 0 <= case_index < len(cases):
        raise ValueError("case_index is outside the selected split")
    case = cases[case_index]
    candidate = evaluator.run_candidate(program, case["public"])
    metrics = evaluator.score_case(case, candidate)
    if not metrics.get("valid"):
        raise ValueError(f"Candidate cannot be exported: {metrics}")
    reference = np.asarray(case["reference"], dtype=np.float64)
    noisy = np.asarray(case["public"]["vertices"], dtype=np.float64)
    repaired = np.asarray(candidate["vertices"], dtype=np.float64)
    faces = np.asarray(case["public"]["faces"], dtype=np.int64)
    editable = np.asarray(case["public"]["editable"], dtype=bool)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "geometry.npz", noisy=noisy, repaired=repaired,
                        reference=reference, faces=faces, editable=editable)
    metadata = {"case_id": case["public"]["id"], **case["metadata"]}
    spec = {"output": str(output), "metadata": metadata, "split": split, "case_index": case_index,
            "program_sha256": hashlib.sha256(Path(program).read_bytes()).hexdigest(),
            "cameras": _camera_plan(reference, noisy, faces, editable), "evaluation": metrics,
            "reference_label": "Uncorrupted artist-authored mesh: proxy reference, not ideal anatomy",
            "camera_policy": "Fixed from reference/noisy input ROI; never fitted to repaired output",
            "geometry_only": True, "reference_sent_to_candidate": False}
    spec_path = output / "export_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    command = [str(Path(blender).resolve()), "--background", "--factory-startup", "--disable-autoexec",
               "--python-exit-code", "1", "--python", str(Path(__file__).resolve()),
               "--", "--worker", str(spec_path)]
    with (output / "blender.log").open("w", encoding="utf-8") as log:
        proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=300,
                              creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    if proc.returncode:
        raise RuntimeError(f"Blender exited {proc.returncode}; see {output / 'blender.log'}")
    report_path = output / "reimport_check.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("passed"):
        raise RuntimeError(f"GLB reimport verification failed: {report}")
    _contact_sheet(output, metadata, metrics)
    return {"output": str(output), "contact_sheet": str(output / "contact_sheet.png"),
            "reimport_check": report, "evaluation": metrics}


def _roundtrip(bpy, mesh_object, expected, faces, path):
    import numpy as np
    from mathutils import Vector
    from mathutils.kdtree import KDTree

    bpy.ops.object.select_all(action="DESELECT")
    mesh_object.hide_set(False)
    mesh_object.select_set(True)
    bpy.context.view_layer.objects.active = mesh_object
    bpy.ops.export_scene.gltf(filepath=str(path), export_format="GLB", use_selection=True,
                             export_materials="NONE", export_texcoords=False, export_normals=True,
                             export_animations=False, export_skins=False, export_yup=True)
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    imported = [obj for obj in bpy.data.objects if obj not in before]
    vertices, triangle_count = [], 0
    for obj in imported:
        if obj.type == "MESH":
            vertices.extend(tuple(obj.matrix_world @ v.co) for v in obj.data.vertices)
            obj.data.calc_loop_triangles()
            triangle_count += len(obj.data.loop_triangles)
    restored = np.asarray(vertices, dtype=np.float64)
    def maximum_nearest(a, b):
        tree = KDTree(len(b))
        for index, vertex in enumerate(b):
            tree.insert(Vector(vertex), index)
        tree.balance()
        # KDTree uses float32; use it only to find the index, then measure in
        # float64 so GLB float32 quantization is not rounded away in this report.
        return max(float(np.linalg.norm(vertex - b[tree.find(Vector(vertex))[1]])) for vertex in a)
    finite = restored.shape[0] > 0 and np.isfinite(restored).all()
    forward = maximum_nearest(expected, restored) if finite else math.inf
    backward = maximum_nearest(restored, expected) if finite else math.inf
    max_error = max(forward, backward)
    result = {"file": path.name, "input_vertices": len(expected), "reimport_vertices": len(restored),
              "input_triangles": len(faces), "reimport_triangles": triangle_count,
              "expected_to_import_max_nearest_error": forward,
              "import_to_expected_max_nearest_error": backward,
              "symmetric_max_nearest_error": max_error, "tolerance": 2e-6,
              "bbox_max_abs_error": float(np.max(np.abs(np.array([expected.min(0), expected.max(0)]) -
                                                            np.array([restored.min(0), restored.max(0)])))) if finite else None,
              "passed": bool(finite and triangle_count == len(faces) and max_error <= 2e-6)}
    for obj in imported:
        bpy.data.objects.remove(obj, do_unlink=True)
    return result


def blender_worker(spec_path):
    import bpy
    import numpy as np
    from mathutils import Vector

    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    output = Path(spec["output"])
    data = np.load(output / "geometry.npz", allow_pickle=False)
    faces = data["faces"]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    objects, checks = {}, []
    for name in ("noisy", "repaired", "reference"):
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(data[name].tolist(), [], faces.tolist())
        mesh.update()
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.collection.objects.link(obj)
        # Flat face normals expose the actual triangular surface and do not hide
        # corruption behind interpolation. This is identical for all conditions.
        for polygon in mesh.polygons:
            polygon.use_smooth = False
        checks.append(_roundtrip(bpy, obj, data[name], faces, output / (name + ".glb")))
        objects[name] = obj
    report = {"passed": all(item["passed"] for item in checks), "meshes": checks,
              "geometry_only": True, "materials_or_uv_preservation_guaranteed": False,
              "topology_isomorphism_checked": False,
              "scope": "Symmetric vertex-set nearest-neighbor error and triangle count after real GLB reimport. "
                       "Vertex ordering may change. This does not establish material/UV preservation or full topology equivalence.",
              "blender_version": bpy.app.version_string}
    (output / "reimport_check.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not report["passed"]:
        raise RuntimeError("GLB roundtrip failed")
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x, scene.render.resolution_y = 640, 560
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    shade = scene.display.shading
    shade.light = "STUDIO"
    shade.studiolight_rotate_z = .35
    shade.color_type = "SINGLE"
    shade.single_color = (.65, .70, .76)
    shade.show_shadows, shade.show_cavity = True, True
    shade.cavity_type = "BOTH"
    shade.curvature_ridge_factor, shade.curvature_valley_factor = 1.1, 1.0
    shade.background_type = "WORLD"
    if scene.world is None:
        scene.world = bpy.data.worlds.new("PreviewWorld")
    scene.world.color = (.075, .085, .10)
    camera = bpy.data.objects.new("CommonCamera", bpy.data.cameras.new("CommonCamera"))
    scene.collection.objects.link(camera)
    camera.data.type = "ORTHO"
    camera.data.clip_start, camera.data.clip_end = .001, 100
    scene.camera = camera
    for settings in spec["cameras"]:
        target = Vector(settings["target"])
        facing = Vector(settings["direction"])
        camera.location = target + facing * 4.0
        camera.rotation_euler = (-facing).to_track_quat("-Z", "Y").to_euler()
        camera.data.ortho_scale = settings["ortho_scale"]
        for name, obj in objects.items():
            for other in objects.values():
                other.hide_render = other != obj
                other.hide_set(other != obj)
            scene.render.filepath = str(output / f"{name}_{settings['crop']}_v{settings['view']}.png")
            bpy.ops.render.render(write_still=True)
    print(json.dumps({"rendered_pngs": 18, "roundtrip_passed": report["passed"]}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program")
    parser.add_argument("--output")
    parser.add_argument("--blender")
    parser.add_argument("--split", choices=("calibration", "dev"), default="calibration")
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        blender_worker(args.worker)
        return
    if not all((args.program, args.output, args.blender)):
        parser.error("--program, --output and --blender are required")
    print(json.dumps(export_repair(args.program, args.output, args.blender, args.split, args.case_index), indent=2))


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else None)
