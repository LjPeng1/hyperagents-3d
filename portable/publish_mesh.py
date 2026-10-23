"""Prepare a sanitized local mesh-trial example, without copying raw logs.

No model, network, Git or external publication calls are made. --dry-run validates
and describes the allowlisted files without creating the destination directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
RUN_FIELDS = ("arm", "returncode", "timed_out", "error", "duration_sec", "usage")
PRIVATE_KEYS = {"stdout", "stderr", "rawstdout", "rawstderr", "command", "commands", "cmd",
                "argv", "threadid", "sessionid", "conversationid", "accountid", "log", "logpath"}
RENDER_FILES = ("contact_sheet.png", "noisy.glb", "repaired.glb", "reference.glb")


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _inside(path, parent):
    resolved = Path(path).resolve()
    relative = resolved.relative_to(parent.resolve())
    if not relative.parts:
        raise ValueError("Expected a child directory, not the workspace/examples root")
    return resolved


def _cleaner(source):
    replacements = ((str(source), "."), (str(ROOT.parent / "dream-rsi-3d"), "<DREAM>"),
                    (str(ROOT), "<HYPER>"), (str(Path.home()), "<HOME>"))

    def clean(value):
        if isinstance(value, dict):
            result = {}
            for key, child in value.items():
                normalized = re.sub(r"[^a-z]", "", key.lower())
                if normalized in PRIVATE_KEYS:
                    continue
                if key == "runs" and isinstance(child, list):
                    result[key] = [clean({field: row.get(field) for field in RUN_FIELDS}) for row in child]
                else:
                    result[clean(key)] = clean(child)
            return result
        if isinstance(value, list):
            return [clean(child) for child in value]
        if isinstance(value, str):
            for prefix, replacement in replacements:
                for variant in {prefix, prefix.replace("\\", "/"), prefix.replace("/", "\\")}:
                    value = re.sub(re.escape(variant), lambda _: replacement, value, flags=re.IGNORECASE)
            # Catch other machine-local absolute paths without changing URLs or
            # repository-relative paths. Known paths with spaces were handled above.
            value = re.sub(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/][^\s\"'<>|]*", "<LOCAL_PATH>", value)
            value = re.sub(r"(?<![\\/])\\\\[^\\/\s\"'<>|]+\\[^\\/\s\"'<>|]+(?:\\[^\s\"'<>|]*)?",
                           "<LOCAL_PATH>", value)
            value = re.sub(r"(?<![:A-Za-z0-9])/(?:home|Users|tmp|var/tmp)/[^\s\"'<>|]*", "<LOCAL_PATH>", value)
            value = re.sub(r"(?i)((?:thread|session|conversation)[_-]?id\s*[:=]\s*)[A-Za-z0-9-]+",
                           r"\1<REDACTED>", value)
        return value
    return clean


def build_plan(source, destination):
    """Read and validate an allowlisted publication plan; never write files."""
    source = _inside(source, ROOT / "workspace")
    destination = _inside(destination, ROOT / "examples")
    if not source.is_dir():
        raise ValueError("Source trial directory does not exist")
    if destination.exists():
        raise RuntimeError("Publication directory exists; refusing overwrite")
    clean = _cleaner(source)
    plan = {}

    def checked(path):
        path = _inside(path, source)
        if not path.is_file():
            raise ValueError(f"Expected source artifact: {path}")
        return path

    def raw(path, target):
        plan[target] = checked(path).read_bytes()

    def data(value, target):
        plan[target] = (json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")

    def document(path, target):
        data(_read(checked(path)), target)

    protocol = _read(checked(source / "protocol.json"))
    if protocol.get("frozen", {}).get("task_name") != "mesh_edit":
        raise ValueError("This publisher requires a mesh_edit trial protocol")
    data(protocol, "protocol.json")
    for name in ("calibration_report.json", "report.json", "amendments.json"):
        path = source / name
        if path.exists():
            document(path, name)
    # This file contains the two original executable source strings, not a run
    # log. Preserve its bytes so the published initial source hashes still match.
    if (source / "initial_bundle.json").exists():
        raw(source / "initial_bundle.json", "initial_bundle.json")
    # Usage is copied from an existing host summary; publishing never reads model
    # stdout or tries to reconstruct ambiguous/unfinished episode costs.
    usage_path = source / "usage.json"
    if usage_path.exists():
        document(usage_path, "usage.json")
    else:
        usage = None
        for name in ("report.json", "calibration_report.json"):
            if (source / name).exists():
                usage = _read(checked(source / name)).get("usage")
                if usage is not None:
                    break
        data(usage if usage is not None else {"available": False, "note": "No host usage summary exists."}, "usage.json")
    candidate_count = 0
    for directory in sorted((source / "candidates").glob("*")):
        if not directory.is_dir():
            continue
        program = checked(directory / "program.py")
        row = _read(checked(directory / "candidate.json"))
        actual = hashlib.sha256(program.read_bytes()).hexdigest()
        if row.get("file_sha256") and row["file_sha256"] != actual:
            raise ValueError(f"Candidate bytes do not match host file_sha256: {directory.name}")
        prefix = f"candidates/{directory.name}"
        raw(program, prefix + "/program.py")
        data(row, prefix + "/candidate.json")
        proposal = directory / "proposal.md"
        if proposal.exists():
            plan[prefix + "/proposal.md"] = clean(checked(proposal).read_text(encoding="utf-8")).encode("utf-8")
        candidate_count += 1
    updates = 0
    for arm in ("hyper_adaptive", "hyper_fixed"):
        for call in sorted((source / arm / "calls").glob("update*")):
            match = re.fullmatch(r"update(\d+)", call.name)
            if match is None or not (call / "result.json").exists():
                continue
            result = _read(checked(call / "result.json"))
            prefix = f"agents/{arm}/g{match.group(1)}"
            for role in ("task", "meta"):
                # Source text is evidence: preserve it exactly, as for program.py.
                code = result["bundle"][role]
                if not isinstance(code, str):
                    raise ValueError(f"Invalid agent source: {arm}/{call.name}/{role}")
                plan[f"{prefix}/{role}_agent.py"] = code.encode("utf-8")
            data({key: value for key, value in result.items() if key != "bundle"}, prefix + "/lineage.json")
            request = call / "executed_request.json"
            if request.exists():
                document(request, prefix + "/executed_request.json")
            updates += 1
    policies = 0
    for journal in sorted((source / "journal").glob("dream_update*.json")):
        match = re.fullmatch(r"dream_update(\d+)\.json", journal.name)
        if match is None:
            continue
        entry = _read(checked(journal))
        if entry.get("status") != "completed":
            continue
        result = entry["result"]
        selected = result.get("selected")
        if not selected or not selected.get("method_src"):
            continue
        method = Path(selected["method_src"])
        if not method.is_absolute():
            method = source / method
        method = checked(method)
        expected = result.get("expected_policy_sha256") or selected.get("method_sha256")
        if expected and hashlib.sha256(method.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Selected DREAM policy hash mismatch: {journal.name}")
        prefix = f"dream_policies/update{match.group(1)}"
        raw(method, prefix + "/method.py")
        data({"selected": selected, "expected_policy_sha256": expected}, prefix + "/selection.json")
        policies += 1
    render_count = 0
    for render in sorted((source / "renders").glob("*")):
        if not render.is_dir():
            continue
        prefix = "renders/" + render.name
        for name in RENDER_FILES:
            if (render / name).exists():
                raw(render / name, prefix + "/" + name)
        for name in ("reimport_check.json", "source_provenance.json", "provenance.json"):
            if (render / name).exists():
                document(render / name, prefix + "/" + name)
        render_count += 1
    summary = {"task_name": "mesh_edit", "has_full_comparison_report": (source / "report.json").exists(),
               "candidate_count": candidate_count, "hyper_updates": updates,
               "selected_dream_policies": policies, "render_sets": render_count,
               "source_trial": source.name,
               "scope": "Allowlisted evidence only; raw logs, commands, session identifiers, export_spec, "
                        "geometry.npz and Blender project files are omitted. Executable sources remain unchanged.",
               "files": {name: {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
                         for name, content in sorted(plan.items())}}
    data(summary, "publication.json")
    return source, destination, plan


def publish(source, destination):
    _, destination, plan = build_plan(source, destination)
    # All source validation happens before creating a destination. mkdir also
    # rejects a concurrent publication rather than overwriting its artifacts.
    destination.mkdir(parents=True, exist_ok=False)
    for name, content in plan.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(content)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        _, _, plan = build_plan(args.source, args.destination)
        print(json.dumps({"dry_run": True, "files": sorted(plan), "bytes": sum(map(len, plan.values()))}, indent=2))
    else:
        print(publish(args.source, args.destination))


if __name__ == "__main__":
    main()
