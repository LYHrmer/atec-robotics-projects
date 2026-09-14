#!/usr/bin/env python3
"""Package a locally verified Task B positive run; never upload anything.

Requires the reviewed v2 auditor, supplied explicitly with --auditor. Run this
with a Python interpreter that has the auditor's NumPy dependency installed.
Cached reports are ignored. Audit rejection creates no output directory;
copy failures leave an INCOMPLETE marker and no valid package manifest.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath


AUDITOR_SHA256 = "f31f81582633b4fc53157b96e403a1fea4b1e571911a504c2923aeef555904ca"
REQUIRED = ("result.json", "environment_metadata.json", "source_manifest.json",
            "telemetry.npz", "trace.jsonl")
OPTIONAL = ("scoring_events.json", "public_cameras.mp4")


class Rejected(ValueError):
    """The input cannot be packaged as independently verified positive evidence."""


def require(condition, message):
    if not condition:
        raise Rejected(message)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_file(root, name):
    """Only regular files beneath root; reject every symlink in their path."""
    rel = PurePosixPath(name)
    require(bool(name) and not rel.is_absolute() and ".." not in rel.parts
            and "\\" not in name and rel.as_posix() == name,
            f"Unsafe artifact path: {name!r}")
    current = root
    for part in rel.parts:
        current = current / part
        require(not current.is_symlink(), f"Symlink artifact is forbidden: {name}")
    require(current.is_file() and current.resolve().is_relative_to(root),
            f"Missing or non-regular artifact: {name}")
    return current


def inventory(run):
    files = {name: safe_file(run, name) for name in REQUIRED}
    for name in OPTIONAL:
        if (run / name).exists() or (run / name).is_symlink():
            files[name] = safe_file(run, name)
    manifest = json.loads(files["source_manifest.json"].read_text())
    seen = set()
    for entry in manifest["hashed_files"].values():
        if entry["root"] != "project":
            continue
        rel = entry["relative_path"]
        require((PurePosixPath(rel).parts[0] in {"task_a", "task_b", "tools"}
                 and rel.endswith(".py")) or rel == "task_e_geometry.py",
                f"Unexpected project snapshot: {rel}")
        name = "source_snapshots/" + rel
        require(name not in seen, f"Duplicate project snapshot: {rel}")
        seen.add(name)
        path = safe_file(run, name)
        require(path.stat().st_size == entry["bytes"] and digest(path) == entry["sha256"],
                f"Snapshot does not match source manifest: {rel}")
        files[name] = path
    require(bool(seen), "No actual project source snapshots")
    return files


def verify_report(report, result, run):
    checks = report.get("checks")
    require(report.get("audit_schema_version") == 2
            and report.get("verified_positive_raw_score") is True
            and report.get("status") == "verified_positive_score"
            and report.get("run_directory") == str(run)
            and isinstance(checks, list) and bool(checks)
            and all(c.get("passed") is True for c in checks)
            and report.get("failed_checks") == [], "Fresh audit did not verify this run")
    score = result["score_raw_total"]
    require(type(score) in (int, float) and math.isfinite(score) and score > 0
            and report.get("score_raw_total") == score
            and report.get("steps") == result["steps"]
            and len(report.get("events", [])) == len(result.get("scoring_events", [])) > 0,
            "Positive audit/result evidence does not agree")


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def package(run_directory, output_directory, auditor):
    require(not run_directory.is_symlink(), "Run directory must not be a symlink")
    run = run_directory.resolve(strict=True)
    require(run.is_dir(), "Run directory must be a directory")
    output = output_directory.absolute()
    require(not output.exists() and not output.is_symlink(), "Output must be a new directory")
    require(output.parent.is_dir(), "Output parent directory must already exist")
    require(not output.resolve().is_relative_to(run), "Output must be outside the recorded run")
    require(not auditor.is_symlink() and auditor.is_file()
            and digest(auditor) == AUDITOR_SHA256,
            "Auditor is not the reviewed v2 script (SHA-256 mismatch)")
    require(not (run / "failure.txt").exists(), "Run contains an integration failure")
    require(not (run / "SYNTHETIC_ONLY.txt").exists()
            and not (run / "SYNTHETIC_ONLY.txt").is_symlink(),
            "Synthetic audit fixtures cannot be published as real runs")
    files = inventory(run)
    before = {name: digest(path) for name, path in files.items()}
    result = json.loads(files["result.json"].read_text())
    require("fixture_notice" not in result, "Synthetic fixture_notice is not real-run evidence")
    with tempfile.TemporaryDirectory(prefix="taskb-positive-audit-") as temporary:
        fresh = Path(temporary) / "audit.json"
        # Execute the pinned bytes, so changing the supplied script afterwards
        # cannot replace the verifier between its hash check and execution.
        script = Path(temporary) / "auditor.py"
        script.write_bytes(auditor.read_bytes())
        require(digest(script) == AUDITOR_SHA256, "Auditor changed while being copied")
        proc = subprocess.run([sys.executable, str(script), str(run), "--output", str(fresh)],
                              capture_output=True, text=True, check=False)
        report = json.loads(fresh.read_text()) if fresh.is_file() else {}
        require(proc.returncode == 0,
                f"Independent audit rejected run (exit {proc.returncode}, "
                f"status {report.get('status', 'no fresh report')})")
        verify_report(report, result, run)
    require("scoring_events.json" in files, "Positive run requires scoring_events.json")
    require({name: digest(safe_file(run, name)) for name in files} == before,
            "Recorded artifacts changed during independent audit")

    # Exclusive mkdir prevents accidental overwrite. If copying fails, keep
    # an explicit INCOMPLETE marker and never write the final package manifest.
    output.mkdir()
    marker = output / "INCOMPLETE"
    marker.write_text("Packaging unfinished; do not publish this directory.\n")
    entries = {}
    for name, source in files.items():
        source = safe_file(run, name)
        target_name = name + ".gz" if name == "trace.jsonl" else name
        target = output / target_name
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as incoming, target.open("xb") as outgoing:
            if name == "trace.jsonl":
                with gzip.GzipFile(filename="", mode="wb", fileobj=outgoing, mtime=0) as compressed:
                    shutil.copyfileobj(incoming, compressed)
            else:
                shutil.copyfileobj(incoming, outgoing)
        entry = {"bytes": target.stat().st_size, "sha256": digest(target)}
        if name == "trace.jsonl":
            h, size = hashlib.sha256(), 0
            with gzip.open(target, "rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(chunk)
                    size += len(chunk)
            entry.update(uncompressed_sha256=h.hexdigest(), uncompressed_bytes=size)
        copied_sha = entry.get("uncompressed_sha256", entry["sha256"])
        require(copied_sha == before[name], f"Copied artifact changed: {name}")
        entries[target_name] = entry
    write_json(output / "independent_positive_audit.json", report)
    audit_file = output / "independent_positive_audit.json"
    entries[audit_file.name] = {"bytes": audit_file.stat().st_size, "sha256": digest(audit_file)}
    record = {
        "package_schema_version": 1, "task": result["task"], "seed": result["seed"],
        "steps": result["steps"], "score_raw_total": result["score_raw_total"],
        "verified_positive_raw_score": True, "auditor_sha256": AUDITOR_SHA256,
        "audit_process_exit_code": 0, "competition_server_certified": False,
        "score_scope": "Local official-environment raw score; not competition-server certification.",
        "interpretation": "Proximity points do not prove grasp, lift, delivery or full Task B completion.",
        "full_task_pass_claimed": False, "video_included": "public_cameras.mp4" in files,
        "quality_limits": report.get("quality_limits", []),
        "audit_scope_limits": report.get("scope_limits", []),
        "files": entries,
    }
    write_json(output / "public_package.json", record)
    checksums = {name: item["sha256"] for name, item in entries.items()}
    checksums["public_package.json"] = digest(output / "public_package.json")
    with (output / "SHA256SUMS").open("x") as stream:
        stream.writelines(f"{sha}  {name}\n" for name, sha in sorted(checksums.items()))
    marker.unlink()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--auditor", type=Path, required=True,
                        help="Reviewed v2 script, e.g. task_b/audit_positive.py; SHA-256 is pinned")
    args = parser.parse_args()
    try:
        record = package(args.run_directory, args.output_directory, args.auditor)
    except (Rejected, OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        print(f"Package rejected: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output_directory),
                      "score_raw_total": record["score_raw_total"],
                      "video_included": record["video_included"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
