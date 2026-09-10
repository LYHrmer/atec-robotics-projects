#!/usr/bin/env python3
"""Make a clearly labelled 10x preview from the completed landscape export."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont

from export_landscape import fresh, probe, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--font', type=Path, default=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'))
    parser.add_argument('--font-index', type=int, default=2)
    args = parser.parse_args()
    source_provenance = args.source.with_suffix('.provenance.json')
    original = json.loads(source_provenance.read_text())
    actual_source_sha = sha256(args.source)
    if original['output']['sha256'] != actual_source_sha:
        raise ValueError('Landscape source does not match its completed provenance.')
    if original['composition']['playback_speed'] != 1 or original['composition']['result_hold_frames'] != 20:
        raise ValueError('Expected the full 1x export with a separate 2s result page.')
    if original['source_video']['frames'] != 5057 or original['result']['simulation_seconds'] != 505.66:
        raise ValueError('This preview is only for the original 505.66s pass; do not mix it with the faster run.')
    fresh(args.output)
    badge_path = args.output.with_suffix('.badge.png')
    manifest = args.output.with_suffix('.provenance.json')
    fresh(badge_path)
    fresh(manifest)
    badge = Image.new('RGBA', (318, 60), (0, 0, 0, 0))
    d = ImageDraw.Draw(badge)
    d.rounded_rectangle((0, 0, 317, 59), radius=18, fill='#16382f')
    d.text((26, 8), '全程速览 · 10×', font=ImageFont.truetype(str(args.font), 28, index=args.font_index), fill='#55ddc0')
    badge.save(badge_path)
    filters = (
        '[0:v]split[run][result];'
        '[run]trim=end=505.7,setpts=(PTS-STARTPTS)/10,fps=30[fast];'
        '[result]trim=start=505.7:end=507.7,setpts=PTS-STARTPTS,fps=30[hold];'
        '[fast][hold]concat=n=2:v=1:a=0[base];'
        '[base][1:v]overlay=1570:24:shortest=1:eof_action=repeat[out]'
    )
    command = [
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-n', '-i', str(args.source),
        '-loop', '1', '-i', str(badge_path), '-filter_complex_threads', '2', '-filter_complex', filters, '-map', '[out]',
        '-an', '-c:v', 'libx264', '-preset', 'medium', '-crf', '20', '-threads', '8',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        '-metadata', 'title=D1+G2 Task A | 10x full-course preview',
        '-metadata', 'comment=Video playback accelerated 10x; original result remains 505.66 simulated seconds. Result page held 2s.',
        str(args.output),
    ]
    subprocess.run(command, check=True)
    info = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_landscape_sha256': actual_source_sha,
        'source_landscape_provenance_sha256': sha256(source_provenance),
        'source_original_video_sha256': original['source_video']['sha256'],
        'playback': {'run_speed_multiplier': 10, 'run_source_seconds': 505.7,
                     'result_hold_seconds': 2, 'output_fps': 30,
                     'frame_selection': 'Frames dropped by fps filter after 10x timestamp compression; no optical interpolation.',
                     'task_result_unchanged': {'simulation_seconds': 505.66, 'forward_distance_m': 286.0023498535156}},
        'command': command,
        'badge_sha256': sha256(badge_path),
        'script_sha256': sha256(Path(__file__)),
        'output': {'path': str(args.output.resolve()), 'sha256': sha256(args.output),
                   'bytes': args.output.stat().st_size, 'probe': probe(args.output)},
    }
    manifest.write_text(json.dumps(info, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'sha256': info['output']['sha256'], 'provenance': str(manifest)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
