#!/usr/bin/env python3
"""Compose the recorded Task A run for 16:9 viewing; never rerun simulation.

The 640x480 overview recording has a 40-pixel text header. Only that header is
cropped. Its 640x440 scene is scaled uniformly to 1280x880. The side camera is
the same run's RGB sensor, selected by the recorded simulation control step.
All original overview frames stay in order at their original 10 fps.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, __version__ as pillow_version


ROOT = Path(__file__).resolve().parents[2]
WIDTH, HEIGHT, FPS = 1920, 1080, 10
BG, PANEL, TEXT, MUTED = '#0b1321', '#142135', '#f3f6fb', '#a3b2c6'
ACCENT, TRACK = '#55ddc0', '#2b3c51'
COURSE = [
    ('平地起步', -141, -110, '建立方向，进入越障赛道'),
    ('碎石路面', -110, -30, '在凹凸路面连续保持平衡'),
    ('连续坡地', -30, 50, '走过交替起伏的坡面'),
    ('连续台阶', 50, 130, '跨越逐渐增高的上下台阶'),
    ('终点直道', 130, 145.00234985351562, '越过最后台阶，走向终点'),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def probe(path: Path) -> dict:
    return json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_format', '-show_streams', '-of', 'json', str(path)
    ]))


def fresh(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite existing file: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)


class Composition:
    def __init__(self, args, rows, result):
        self.args, self.rows, self.result = args, rows, result
        self.times = [float(r['sim_seconds']) for r in rows]
        self.start_x = -141.0
        self.goal_x = float(result['final']['xyz'][0])
        self.distance = self.goal_x - self.start_x
        self.seconds = float(result['simulation_seconds'])
        self.baseline = args.baseline_data
        self.without_front_camera = args.without_front_camera
        self.fonts = {}
        for size in [18, 20, 22, 24, 26, 30, 32, 36, 42, 46, 48, 62, 70]:
            self.fonts[size] = ImageFont.truetype(str(args.font), size, index=args.font_index)
        self.base = Image.new('RGB', (WIDTH, HEIGHT), BG)
        d = ImageDraw.Draw(self.base)
        self.text(d, (32, 17), 'D1 + G2  /  TASK A', 42)
        subtitle = '提速配置 · 同一赛道的另一次完整运行' if self.without_front_camera else '从碎石、坡地到连续台阶'
        self.text(d, (34, 73), subtitle, 24, MUTED)
        d.rounded_rectangle((1570, 24, 1888, 84), radius=18, fill='#16382f')
        self.text(d, (1604, 35), '完整原速 · 1×', 30, ACCENT)
        d.rounded_rectangle((1344, 124, 1888, 1004), radius=18, fill=PANEL)
        if self.without_front_camera:
            baseline_seconds = float(self.baseline['simulation_seconds'])
            difference = baseline_seconds - self.seconds
            percentage = difference / baseline_seconds * 100
            self.text(d, (1372, 146), '仿真用时对照', 26, MUTED)
            self.text(d, (1372, 209), '原通关配置', 24, MUTED)
            self.text(d, (1372, 247), f'{baseline_seconds:.2f} s', 42, MUTED)
            self.text(d, (1372, 326), '本次提速配置', 24)
            self.text(d, (1372, 361), f'{self.seconds:.2f} s', 62, ACCENT)
            d.rounded_rectangle((1372, 454, 1860, 518), radius=14, fill='#16382f')
            self.text(d, (1390, 464), f'减少 {difference:.2f} s（{percentage:.2f}%）', 26, ACCENT)
            self.text(d, (1372, 537), '两次实际运行 · 均按原速播放', 20, MUTED)
        else:
            self.text(d, (1364, 132), '前置相机 · 实际 RGB', 22, MUTED)
        self.text(d, (1372, 573), '赛道进度', 22, MUTED)
        self.text(d, (1372, 697), '仿真时间', 22, MUTED)
        d.line((1372, 883, 1860, 883), fill=TRACK, width=2)
        self.text(d, (1372, 904), '本次最终成绩', 20, MUTED)
        self.text(d, (1372, 938), f'{self.seconds:.2f} s  /  {self.distance:.3f} m', 26, ACCENT)
        self.text(d, (1344, 1022), '本机仿真 · 日志数据用于事后展示', 20, MUTED)

    def text(self, d, at, value, size, fill=TEXT):
        d.text(at, value, font=self.fonts[size], fill=fill)

    def row(self, index):
        # Original runner writes overview frame n after control step 5*n+1.
        t = 0.02 + index / FPS
        return self.rows[max(0, bisect_right(self.times, t + 1e-8) - 1)], t

    def render(self, bgr, rgb, index, final=False):
        row, t = self.row(index)
        x = self.goal_x if final else float(row['xyz'][0])
        progress = min(self.distance, max(0., x - self.start_x))
        stage = next((s for s in COURSE if x < s[2]), COURSE[-1])
        canvas = self.base.copy()
        overview = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        overview = overview.crop((0, 40, 640, 480)).resize((1280, 880), Image.Resampling.LANCZOS)
        canvas.paste(overview, (32, 124))
        if not self.without_front_camera:
            inset = Image.fromarray(rgb).resize((512, 384), Image.Resampling.LANCZOS)
            canvas.paste(inset, (1360, 170))
        d = ImageDraw.Draw(canvas)
        d.rounded_rectangle((48, 140, 266, 184), radius=10, fill='#172332')
        self.text(d, (60, 145), '第三人称观察', 24)
        self.text(d, (1372, 608), f'{progress:.1f} m', 46)
        self.text(d, (1710, 628), '/ 286 m', 22, MUTED)
        d.rounded_rectangle((1372, 676, 1860, 682), radius=3, fill=TRACK)
        if progress > 0:
            d.rounded_rectangle((1372, 676, 1372 + 488 * progress / self.distance, 682), radius=3, fill=ACCENT)
        time_text = f'{int(t // 60):02d}:{t % 60:04.1f}'
        final_time_text = f'{int(self.seconds // 60):02d}:{self.seconds % 60:05.2f}'
        self.text(d, (1600, 689), final_time_text if final else time_text, 36)
        self.text(d, (1372, 758), '终点已到达' if final else stage[0], 32, ACCENT)
        self.text(d, (1372, 811), '唯一终止项：reach_goal_x' if final else stage[3], 22, MUTED)

        # The stage ruler is a map reference; it does not alter or drive control.
        ruler_x, ruler_w = 32, 1280
        for name, a, b, _ in COURSE:
            left = ruler_x + ruler_w * (a - self.start_x) / self.distance
            right = ruler_x + ruler_w * (b - self.start_x) / self.distance
            d.rounded_rectangle((left, 1022, right - 4, 1028), radius=3,
                                fill=ACCENT if x >= a else TRACK)
            self.text(d, (left, 1037), name, 18, TEXT if stage[0] == name else MUTED)
        marker = ruler_x + ruler_w * progress / self.distance
        d.ellipse((marker - 5, 1020, marker + 5, 1030), fill=TEXT)

        if final:
            veil = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
            vd = ImageDraw.Draw(veil)
            vd.rounded_rectangle((170, 315, 1176, 814), radius=30, fill=(10, 20, 33, 238))
            self.text(vd, (230, 356), '原始 Task A · 完整通过', 46, ACCENT)
            self.text(vd, (230, 452), f'{self.seconds:.2f} s', 70)
            self.text(vd, (756, 452), f'{self.distance:.3f} m', 62)
            self.text(vd, (234, 552), '仿真用时', 26, MUTED)
            self.text(vd, (760, 552), '连续前进距离', 26, MUTED)
            self.text(vd, (232, 640), '到达终点，无同帧失败终止', 32)
            self.text(vd, (234, 709), '结果来自验收日志 · 此页定格 2 秒', 24, MUTED)
            canvas = Image.alpha_composite(canvas.convert('RGBA'), veil).convert('RGB')
        return canvas


def sensor_frame(folder, index):
    path = folder / f'frame_{index * 5:06d}.npz'
    with np.load(path, allow_pickle=False) as sample:
        rgb = np.asarray(sample['rgb'], dtype=np.uint8)
    if rgb.shape != (480, 640, 3):
        raise ValueError(f'Unexpected RGB dimensions: {path}: {rgb.shape}')
    return path, rgb


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-video', type=Path, required=True)
    p.add_argument('--sensors', type=Path)
    p.add_argument('--without-front-camera', action='store_true', help='Use a result comparison card; never read sensor frames.')
    p.add_argument('--baseline-result', type=Path, default=ROOT / 'task_a/evidence/full_course_02/result.json')
    p.add_argument('--trace', type=Path, default=ROOT / 'task_a/evidence/full_course_02/trace.jsonl')
    p.add_argument('--result', type=Path, default=ROOT / 'task_a/evidence/full_course_02/result.json')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--font', type=Path, default=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'))
    p.add_argument('--font-index', type=int, default=2, help='SC face in the Noto CJK TTC collection')
    p.add_argument('--crf', type=int, default=20)
    p.add_argument('--preset', default='medium')
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--preview-dir', type=Path)
    p.add_argument('--preview-only', action='store_true')
    args = p.parse_args()
    if not args.without_front_camera and args.sensors is None:
        p.error('--sensors is required unless --without-front-camera is selected')
    if args.without_front_camera and args.sensors is not None:
        p.error('Do not provide --sensors in the mode without a front camera.')
    args.baseline_data = None
    if args.without_front_camera:
        args.baseline_data = json.loads(args.baseline_result.read_text())
        if not args.baseline_data['reached_goal'] or args.baseline_data['reason'] != 'reach_goal_x':
            raise ValueError('Comparison baseline must be a verified complete pass.')
    cv2.setNumThreads(1)
    source = probe(args.source_video)
    stream = next(s for s in source['streams'] if s['codec_type'] == 'video')
    if (stream['width'], stream['height'], stream['avg_frame_rate']) != (640, 480, '10/1'):
        raise ValueError('This layout is audited only for the original 640x480, 10-fps recording.')
    frames = int(stream['nb_frames'])
    if not args.without_front_camera and frames != 5057:
        raise ValueError('Expected the complete 5057-frame source, not a clip.')
    result = json.loads(args.result.read_text())
    if (not result['reached_goal'] or result['reason'] != 'reach_goal_x'
            or set(result['final'].get('termination_terms', [])) != {'reach_goal_x'}):
        raise ValueError('Result does not identify the audited complete Task A pass.')
    if not args.without_front_camera and result['steps'] != 25283:
        raise ValueError('Front-camera layout is frozen to the original complete pass.')
    if frames != (int(result['steps']) - 2) // 5 + 1:
        raise ValueError('Video frame count does not match the original recording schedule and result steps.')
    rows = [json.loads(line) for line in args.trace.read_text().splitlines()]
    composition = Composition(args, rows, result)
    cap = cv2.VideoCapture(str(args.source_video))
    if not cap.isOpened():
        raise RuntimeError('Cannot decode original video.')

    if args.preview_dir:
        preview_indices = [0, 500, 1500, 3000, 4000, frames - 1] if args.without_front_camera else [0, 500, 1500, 3500, 4750, 5056]
        for index in preview_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, bgr = cap.read()
            if not ok:
                raise RuntimeError(f'Cannot read source frame {index}')
            rgb = None if args.without_front_camera else sensor_frame(args.sensors, index)[1]
            destination = args.preview_dir / f'frame_{index:06d}.jpg'
            fresh(destination)
            composition.render(bgr, rgb, index).save(destination, quality=94)
        destination = args.preview_dir / 'final_result.jpg'
        fresh(destination)
        composition.render(bgr, rgb, frames - 1, final=True).save(destination, quality=94)
    if args.preview_only:
        if not args.preview_dir:
            raise ValueError('--preview-only requires --preview-dir')
        cap.release()
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fresh(args.output)
    manifest_path = args.output.with_suffix('.provenance.json')
    sensor_manifest = None if args.without_front_camera else args.output.with_suffix('.sensor_rgb_sha256.jsonl')
    for destination in [manifest_path] + ([] if sensor_manifest is None else [sensor_manifest]):
        fresh(destination)
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-n', '-f', 'rawvideo',
               '-pixel_format', 'rgb24', '-video_size', '1920x1080', '-framerate', '10',
               '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', args.preset,
               '-crf', str(args.crf), '-threads', str(args.threads), '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', '-metadata', 'title=D1+G2 Task A | full run at 1x | landscape',
               '-metadata', f'comment={composition.seconds:.2f} simulated seconds; {frames/FPS:.1f}s source video plus 2s result hold.',
               str(args.output)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    sensor_hash = hashlib.sha256()
    try:
        with (sensor_manifest.open('x') if sensor_manifest else nullcontext(None)) as sensor_log:
            for index in range(frames):
                ok, bgr = cap.read()
                if not ok:
                    raise RuntimeError(f'Unexpected end of source at frame {index}/{frames}')
                rgb = None
                if not args.without_front_camera:
                    path, rgb = sensor_frame(args.sensors, index)
                    entry = {'overview_frame': index, 'sensor_step': index * 5,
                             'sensor_name': path.name, 'rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest()}
                    line = json.dumps(entry, sort_keys=True) + '\n'
                    sensor_log.write(line)
                    sensor_hash.update(line.encode())
                canvas = composition.render(bgr, rgb, index)
                encoder.stdin.write(canvas.tobytes())
                if index % 250 == 0:
                    print(json.dumps({'frame': index, 'total': frames, 'source_seconds': index / FPS}), flush=True)
            final = composition.render(bgr, rgb, frames - 1, final=True).tobytes()
            for _ in range(20):
                encoder.stdin.write(final)
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError('FFmpeg encoding failed.')
    except BaseException:
        encoder.terminate()
        encoder.wait()
        raise
    finally:
        cap.release()
    output = probe(args.output)
    provenance = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_video': {'path': str(args.source_video.resolve()), 'sha256': sha256(args.source_video),
                         'width': 640, 'height': 480, 'fps': 10, 'frames': frames, 'duration_seconds': frames / FPS},
        'source_result': {'path': str(args.result.resolve()), 'sha256': sha256(args.result)},
        'source_trace': {'path': str(args.trace.resolve()), 'sha256': sha256(args.trace)},
        'source_sensors': None if args.without_front_camera else {'path': str(args.sensors.resolve()), 'frames': frames,
                           'rgb_index_file': sensor_manifest.name, 'rgb_index_sha256': sensor_hash.hexdigest()},
        'source_baseline_result': None if not args.without_front_camera else {
            'path': str(args.baseline_result.resolve()), 'sha256': sha256(args.baseline_result),
            'simulation_seconds': float(args.baseline_data['simulation_seconds']),
            'difference_seconds': float(args.baseline_data['simulation_seconds']) - composition.seconds,
            'reduction_percent': (float(args.baseline_data['simulation_seconds']) - composition.seconds) / float(args.baseline_data['simulation_seconds']) * 100,
            'comparison_uses_simulation_time_not_wall_time': True},
        'composition': {'width': WIDTH, 'height': HEIGHT, 'fps': FPS, 'playback_speed': 1.0,
                        'all_source_frames_preserved_in_order': True,
                        'scene_crop_xywh': [0, 40, 640, 440], 'scene_scale': 2.0,
                        'crop_reason': 'Remove only the original black text header; remaining scene is not cropped or stretched. A small viewpoint label is overlaid.',
                        'overview_rect_xywh': [32, 124, 1280, 880],
                        'front_camera_rect_xywh': None if args.without_front_camera else [1360, 170, 512, 384],
                        'right_panel': 'Result comparison without camera footage' if args.without_front_camera else 'Same-run front RGB and recorded progress',
                        'image_color_adjustment': 'none', 'new_or_interpolated_scene_frames': False,
                        'result_hold_seconds': 2.0, 'result_hold_frames': 20,
                        'sensor_alignment': None if args.without_front_camera else 'Overview n: control step 5*n+1; front RGB: step 5*n, 0.02s earlier.',
                        'telemetry_alignment': 'Last recorded trace at/before overview time; sample-and-hold, no interpolation.',
                        'distance_metric': 'Saved diagnostic world x minus original start x; display only, never fed into control.',
                        'course_labels': 'Original fixed terrain x intervals; display only.'},
        'result': {'simulation_seconds': result['simulation_seconds'], 'reason': result['reason'],
                   'forward_progress_m': result['net_forward_progress_m'],
                   'video_duration_is_not_task_completion_time': True},
        'encoding': {'command': command, 'crf': args.crf, 'preset': args.preset, 'audio': 'none'},
        'environment': {'python': platform.python_version(), 'pillow': pillow_version,
                        'opencv': cv2.__version__, 'numpy': np.__version__,
                        'font': str(args.font), 'font_sha256': sha256(args.font), 'font_index': args.font_index,
                        'ffmpeg': subprocess.check_output(['ffmpeg', '-version'], text=True).splitlines()[0]},
        'script_sha256': sha256(Path(__file__)),
        'output': {'path': str(args.output.resolve()), 'sha256': sha256(args.output),
                   'bytes': args.output.stat().st_size, 'probe': output},
    }
    manifest_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': str(args.output), 'sha256': provenance['output']['sha256'],
                      'provenance': str(manifest_path)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
