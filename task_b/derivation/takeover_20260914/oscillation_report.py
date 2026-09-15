"""Offline oscillation report for one loaded payload run (read-only, no policy input).

Separates three things the arrival gate cares about:
  1. the structural mode amplitude over time (is it decaying or sustained?),
  2. the measured joint velocity against the unchanged .12 rad/s cap,
  3. the six-axis position span against the unchanged .002 rad window.

Nothing here feeds the controller. It describes recorded telemetry only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def band_amplitude(values, fs, low, high, window):
    """Peak amplitude of the strongest FFT bin inside [low, high] Hz."""
    segment = values[:window]-values[:window].mean()
    spectrum = np.abs(np.fft.rfft(segment*np.hanning(len(segment))))
    frequencies = np.fft.rfftfreq(len(segment), 1/fs)
    mask = (frequencies >= low) & (frequencies <= high)
    if not mask.any():
        return 0., 0.
    index = int(np.argmax(np.where(mask, spectrum, 0.)))
    return float(spectrum[index]/len(segment)*2), float(frequencies[index])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    run = args.run_directory
    telemetry = np.load(run/'telemetry.npz')
    fs = 1/float(telemetry['dt'])
    names = telemetry['joint_names'].tolist()
    phases = []
    with (run/'trace.jsonl').open() as handle:
        for line in handle:
            debug = (json.loads(line).get('policy_debug') or {})
            phases.append(str(debug.get('phase') or ''))
    payload = np.flatnonzero(np.char.startswith(np.array(phases), 'PAYLOAD'))
    if not len(payload):
        raise SystemExit('no PAYLOAD phase recorded in ' + run.name)

    report = {'run_id': run.name, 'payload_samples': int(len(payload)),
              'payload_seconds': float(len(payload)*float(telemetry['dt'])),
              'scope': 'OFFLINE description of recorded telemetry; no controller input',
              'is_success_evidence': False, 'axes': {}}
    window = int(round(4.*fs))
    for index, axis in enumerate(names[:6]):
        q = telemetry['q'][:, index][payload]
        qd = telemetry['qdot'][:, index][payload]
        trace = []
        for start in range(0, max(len(q)-window, 0)+1, window):
            amp, freq = band_amplitude(q[start:], fs, 1.2, 2.2, window)
            trace.append({'start_s': round(start/fs, 1), 'structural_amp_rad': round(amp, 5),
                          'freq_hz': round(freq, 2)})
        report['axes'][axis] = {
            'position_pkpk_rad': float(q.max()-q.min()),
            'velocity_pkpk_rad_s': float(qd.max()-qd.min()),
            'velocity_absmax_rad_s': float(np.max(np.abs(qd))),
            'velocity_cap_rad_s': .12,
            'seconds_over_velocity_cap': float(np.mean(np.abs(qd) > .12)*len(qd)/fs),
            'structural_trace': trace,
        }

    # The longest run of consecutive ticks that would satisfy the six-axis
    # .12 rad/s velocity gate, and the six-axis .002 rad position window.
    arm_qd = np.abs(telemetry['qdot'][:, :6][payload])
    arm_q = telemetry['q'][:, :6][payload]
    ok = np.all(arm_qd <= .12, axis=1)
    best = run_len = 0
    for flag in ok:
        run_len = run_len+1 if flag else 0
        best = max(best, run_len)
    report['longest_consecutive_ticks_under_velocity_cap'] = int(best)
    report['required_window_samples'] = 26
    spans = [float(np.max(np.ptp(arm_q[i:i+26], axis=0)))
             for i in range(0, len(arm_q)-26)]
    report['min_six_axis_span_rad_over_a_full_window'] = min(spans) if spans else None
    report['position_window_limit_rad'] = .002
    if args.output:
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in (
        'run_id', 'payload_seconds', 'longest_consecutive_ticks_under_velocity_cap',
        'required_window_samples', 'min_six_axis_span_rad_over_a_full_window',
        'position_window_limit_rad')}, indent=2))
    for axis, entry in report['axes'].items():
        print('%-14s pkpk_q=%.4f  |qdot|max=%.3f  over_cap=%.1fs  structural=%s'
              % (axis, entry['position_pkpk_rad'], entry['velocity_absmax_rad_s'],
                 entry['seconds_over_velocity_cap'],
                 ' '.join('%.0fs:%.4f' % (t['start_s'], t['structural_amp_rad'])
                          for t in entry['structural_trace'])))


if __name__ == '__main__':
    main()
