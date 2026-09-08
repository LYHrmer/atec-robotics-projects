"""Encode a continuous Task E sensor recording with an evidence-only HUD."""
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class TaskEVideoRecorder:
    def __init__(self, path, *, step_dt, stride=2, seed=42):
        self.path = Path(path).resolve()
        if self.path.exists():
            raise FileExistsError(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.dt, self.stride, self.seed = step_dt, stride, seed
        self.fps = 1.0/(step_dt*stride)
        # Pillow also searches platform font directories by family filename.
        self.fonts = {}
        for size in (14, 16, 18, 22, 36):
            try:
                self.fonts[size] = ImageFont.truetype('DejaVuSans.ttf', size)
            except OSError:
                self.fonts[size] = ImageFont.load_default()
        self.temporary_path = self.path.with_name('.' + self.path.stem + '.recording.mp4')
        self.error_file = self.path.with_suffix('.ffmpeg.log').open('wb')
        self.process = subprocess.Popen([
            'ffmpeg', '-hide_banner', '-loglevel', 'warning', '-n',
            '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', '1280x720',
            '-framerate', str(self.fps), '-i', '-', '-an', '-c:v', 'libx264',
            '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', str(self.temporary_path)], stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=self.error_file)
        self.last_frame = None
        self.frames = 0

    @staticmethod
    def _rgb(value):
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        value = np.asarray(value)
        if value.ndim == 4:
            value = value[0]
        return Image.fromarray(value[..., :3].astype(np.uint8))

    def append(self, obs, *, step, score, state, object_id, completed=(), final=False):
        canvas = Image.new('RGB', (1280, 720), '#0e1826')
        canvas.paste(self._rgb(obs['image']['video_rgb']).resize((960, 720)), (0, 0))
        draw = ImageDraw.Draw(canvas)
        def text(x, y, value, size=18, color='#e6edf3'):
            draw.text((x, y), value, font=self.fonts[size], fill=color)
        draw.rectangle((12, 12, 264, 46), fill='#0e1826')
        text(23, 18, 'FIXED RGB-D CAMERA', 16)
        text(976, 20, 'ATEC 2026 | TASK E', 22)
        text(976, 54, 'RGB-D pick and place', 18, '#9cafc3')
        text(976, 91, f'{score:.0f} / 18', 36, '#49d4a3' if score >= 17.99 else '#e6edf3')
        text(976, 143, f'Seed {self.seed}  |  {step*self.dt:.2f} s', 16)
        text(976, 168, '1x simulation time | continuous', 14, '#9cafc3')
        if 'ee_rgb' in obs['image']:
            canvas.paste(self._rgb(obs['image']['ee_rgb']).resize((288, 216)), (976, 214))
            text(976, 190, 'WRIST CAMERA', 14, '#9cafc3')
        text(976, 451, 'STATE', 14, '#9cafc3')
        text(976, 476, 'TASK COMPLETE' if final and score >= 17.99 else str(state), 18)
        names = {1: 'Sugar box', 2: 'Mustard bottle', 3: 'Banana'}
        for index, object_key in enumerate((2, 1, 3)):
            done = object_key in completed or score >= 17.99
            label = 'DONE' if done else ('ACTIVE' if object_key == object_id else 'WAIT')
            text(976, 517+32*index, names[object_key], 16)
            text(1176, 517+32*index, label, 14, '#49d4a3' if done else '#9cafc3')
        text(976, 636, 'Public RGB-D + joint feedback', 14, '#9cafc3')
        text(976, 659, 'No scene-state input to policy', 14, '#9cafc3')
        text(976, 682, 'Official task scoring, unchanged', 14, '#9cafc3')
        self.last_frame = canvas.tobytes()
        self.process.stdin.write(self.last_frame)
        self.frames += 1

    def close(self):
        if self.process.stdin.closed:
            return
        if self.last_frame is not None:
            for _ in range(int(2*self.fps)):
                self.process.stdin.write(self.last_frame)
                self.frames += 1
        self.process.stdin.close()
        code = self.process.wait(timeout=30)
        self.error_file.close()
        if code:
            raise RuntimeError(f'ffmpeg exited with {code}; see {self.path.with_suffix(".ffmpeg.log")}')
        self.temporary_path.replace(self.path)
