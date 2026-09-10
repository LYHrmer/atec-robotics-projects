#!/usr/bin/env python3
"""Check or import only the 18 audited DDT_Lab files; no downloads."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys
import tempfile
import zipfile


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PACKAGE_ROOT / "provenance/external_assets.json"


def load_manifest():
    entries = json.loads(MANIFEST.read_text())["files"]
    seen = set()
    for entry in entries:
        name = entry["path"]
        path = PurePosixPath(name)
        if (path.is_absolute() or ".." in path.parts or "\\" in name
                or str(path) != name or name in seen or name == "."):
            raise ValueError(f"Unsafe or duplicate manifest path: {name}")
        seen.add(name)
    return entries


def destination_file(root, name):
    path = root / name
    for part in (path, *path.parents):
        if part == root:
            break
        if part.is_symlink():
            raise ValueError(f"Destination contains a symlink: {part}")
    return path


def verify(path, entry):
    if not path.is_file() or path.stat().st_size != entry["bytes"]:
        raise ValueError(f"Missing file or size mismatch: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != entry["sha256"]:
        raise ValueError(f"SHA256 mismatch (file left unchanged): {path}")


def zip_members(archive, entries):
    # Require a single complete DDT_Lab tree. Nothing outside the manifest is read.
    names = [info.filename for info in archive.infolist()]
    first = entries[0]["path"]
    prefixes = set()
    for name in names:
        suffix = "DDT_Lab/" + first
        if name.endswith(suffix):
            prefix = name[:-len(first)]
            parts = PurePosixPath(prefix).parts
            if (parts and parts[-1] == "DDT_Lab" and not prefix.startswith("/")
                    and ".." not in parts and "\\" not in prefix):
                prefixes.add(prefix)
    complete = [prefix for prefix in prefixes
                if all(names.count(prefix + item["path"]) == 1 for item in entries)]
    if len(complete) != 1:
        raise ValueError("ZIP must contain exactly one complete, unambiguous audited DDT_Lab tree.")
    result = {}
    for entry in entries:
        info = archive.getinfo(complete[0] + entry["path"])
        mode = info.external_attr >> 16
        if info.is_dir() or stat.S_ISLNK(mode) or info.file_size != entry["bytes"]:
            raise ValueError(f"ZIP member type or size mismatch: {info.filename}")
        result[entry["path"]] = info
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--check", action="store_true", help="Only verify existing files (default).")
    source.add_argument("--from-directory", type=Path, metavar="DDT_Lab")
    source.add_argument("--from-zip", type=Path, metavar="workspace2.zip")
    parser.add_argument("--destination", type=Path, default=PACKAGE_ROOT / "assets/DDT_Lab")
    args = parser.parse_args()
    try:
        entries = load_manifest()
        destination = args.destination.expanduser().resolve()
        importing = args.from_directory is not None or args.from_zip is not None
        missing = []
        for entry in entries:
            path = destination_file(destination, entry["path"])
            if path.exists():
                verify(path, entry)  # All existing files checked before any writes.
            elif importing:
                missing.append(entry)
            else:
                raise ValueError(f"Required asset is missing: {path}")
        if missing:
            with ExitStack() as stack:
                if args.from_zip:
                    archive = stack.enter_context(zipfile.ZipFile(args.from_zip.expanduser()))
                    members = zip_members(archive, entries)
                    open_source = lambda name: archive.open(members[name])
                else:
                    source_root = args.from_directory.expanduser().resolve(strict=True)
                    def open_source(name):
                        path = (source_root / name).resolve(strict=True)
                        if not path.is_relative_to(source_root):
                            raise ValueError(f"Source path escapes DDT_Lab: {name}")
                        return path.open("rb")
                destination.parent.mkdir(parents=True, exist_ok=True)
                staging = Path(stack.enter_context(tempfile.TemporaryDirectory(
                    prefix=".d1g2-import-", dir=destination.parent)))
                for entry in missing:
                    temporary = staging / entry["path"]
                    temporary.parent.mkdir(parents=True, exist_ok=True)
                    size = 0
                    with open_source(entry["path"]) as src, temporary.open("wb") as dst:
                        for chunk in iter(lambda: src.read(1024 * 1024), b""):
                            size += len(chunk)
                            if size > entry["bytes"]:
                                raise ValueError(f"Source exceeds audited size: {entry['path']}")
                            dst.write(chunk)
                    verify(temporary, entry)
                # Publish only after every staged file passes. Hard-link publication is
                # atomic on the same filesystem and cannot overwrite an existing file.
                for entry in missing:
                    target = destination_file(destination, entry["path"])
                    target.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        os.link(staging / entry["path"], target)
                    except FileExistsError:
                        verify(target, entry)
        for entry in entries:
            verify(destination_file(destination, entry["path"]), entry)
        print(json.dumps({"status": "verified", "files": len(entries),
                          "imported": len(missing), "destination": str(destination)}, ensure_ascii=False))
        return 0
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        print(f"Asset check failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
