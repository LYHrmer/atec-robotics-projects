"""Plot measured Task E tradeoffs from compare_motion.py output (Matplotlib)."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('comparison', type=Path)
    parser.add_argument('--output', type=Path, default=Path('media/optimization.png'))
    args = parser.parse_args()
    report = json.loads(args.comparison.read_text())
    panels = [
        ('duration_s', 'Simulation completion time', 'seconds'),
        ('wrist_joint6_path_rad', 'Wrist joint 6 total travel', 'radians'),
        ('arm_acceleration_rms_rad_s2', 'Arm acceleration RMS', 'rad/s²'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.0))
    for ax, (key, title, unit) in zip(axes, panels):
        values = [report[tag]['overall'][key] for tag in ('baseline', 'candidate')]
        ax.bar(['Baseline', 'Optimized'], values, width=.56, color=['#8292a8', '#087f8c'])
        ax.set(title=title, ylabel=unit, ylim=(0, max(values)*1.25))
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
        for index, value in enumerate(values):
            ax.text(index, value+max(values)*.025, f'{value:.2f}', ha='center', fontsize=12)
        change = (values[1]/values[0]-1)*100
        ax.text(.5, .93, f'{change:+.2f}%', transform=ax.transAxes, ha='center',
                fontsize=12, color='#a44900' if change > 0 else '#087f8c')
    fig.suptitle('Task E | seed 42 | both runs 18/18', fontsize=15, y=.99)
    fig.text(.5, .02, '50 Hz measured telemetry. Acceleration uses finite differences; no continuous jerk guarantee.',
             ha='center', fontsize=9, color='#445166')
    fig.tight_layout(rect=(0, .045, 1, .96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    svg_path = args.output.with_suffix('.svg')
    fig.savefig(svg_path)
    svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text().splitlines()) + '\n')
    plt.close(fig)


if __name__ == '__main__':
    main()
