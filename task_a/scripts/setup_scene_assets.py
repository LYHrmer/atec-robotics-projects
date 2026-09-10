#!/usr/bin/env python3
"""Import or verify the original ATEC scene assets required for visual navigation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

from setup_assets import destination_file, verify

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'provenance/external_scene_assets.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--check', action='store_true', help='Verify existing files (default).')
    source.add_argument('--from-directory', type=Path, metavar='ATEC_ROBOT_MODEL')
    args = parser.parse_args()
    destination = ROOT / 'atec_robot_model'
    try:
        entries = json.loads(MANIFEST.read_text())['files']
        missing = []
        for entry in entries:
            name = Path(entry['path'])
            if name.is_absolute() or '..' in name.parts:
                raise ValueError(f'Unsafe manifest path: {name}')
            target = destination_file(destination, entry['path'])
            if target.exists():
                verify(target, entry)
            elif args.from_directory is not None:
                missing.append(entry)
            else:
                raise ValueError(f'Required scene asset missing: {target}')
        if missing:
            source_root = args.from_directory.expanduser().resolve(strict=True)
            with tempfile.TemporaryDirectory(prefix='.scene-import-', dir=ROOT) as tmp:
                staging = Path(tmp)
                for entry in missing:
                    original = (source_root / entry['path']).resolve(strict=True)
                    if not original.is_relative_to(source_root):
                        raise ValueError(f'Source escapes asset directory: {original}')
                    verify(original, entry)
                    staged = staging / entry['path']
                    staged.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(original, staged)
                    verify(staged, entry)
                for entry in missing:
                    target = destination_file(destination, entry['path'])
                    target.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        os.link(staging / entry['path'], target)
                    except FileExistsError:
                        verify(target, entry)
        for entry in entries:
            verify(destination_file(destination, entry['path']), entry)
        print(json.dumps({'status': 'verified', 'scene_files': len(entries),
                          'imported': len(missing), 'destination': str(destination)}))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f'Scene asset check failed: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
