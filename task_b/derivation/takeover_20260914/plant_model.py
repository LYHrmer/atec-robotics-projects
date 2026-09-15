"""CPU model of the loaded joint + payload tracker, identified from public data.

MODEL, not the robot. Every constant below is either a recorded read-only drive
diagnostic (K=80, D=4) or a contract static bound (gravity sag), and the joint
inertia is the single value that reproduces the 1.67 Hz mode actually observed in
plan_p6/p7/p8 telemetry. Predictions from this file are HYPOTHESES to be verified
by a real run, never results.

Why it exists: the loaded arm shows a sustained 1.67 Hz oscillation that starts
when the scalar path finishes and never decays, and GPU runs cost ~8 minutes each.
This integrates the same tracker law against a second-order joint so candidate
fixes can be screened in seconds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# -- identified plant ---------------------------------------------------------
STIFFNESS = 80.0            # recorded original Piper arm_joint drive K
DRIVE_DAMPING = 4.0         # recorded original Piper arm_joint drive D
INERTIA = 0.727             # kg m^2: the value that puts the joint at 1.67 Hz
SAG_Q2_RAD = 13.60069/STIFFNESS    # contract static torque bound / K = .170009
SAG_Q3_RAD = 6.56302/STIFFNESS     # = .082038
#: Recorded public qdot carries ~.07 rad/s at exactly the 50 Hz Nyquist rate that
#: does NOT appear in the measured position; treated here as velocity-sense noise.
VELOCITY_NOISE_RAD_S = .07

DT = .02


def simulate(*, gain2, cap2, damp2, tau_damp, tau_filter, rate_limit,
             path_seconds, seconds, noise=VELOCITY_NOISE_RAD_S, reference_step=SAG_Q2_RAD,
             ff_fraction=0.):
    """Integrate the tracker law against the identified joint; return diagnostics.

    ``reference_step`` is how far the fixed goal sits above the pose the loaded
    joint would hang at with a zero command. With no feed-forward the tracker must
    supply that whole offset from its proportional term.
    """
    steps = int(round(seconds/DT))
    path_steps = int(round(path_seconds/DT))
    q, qd = 0., 0.
    command = 0.
    filtered_correction = 0.
    filtered_rate = 0.
    filter_weight = 1.-np.exp(-DT/tau_filter)
    damp_weight = 1.-np.exp(-DT/tau_damp)
    slew = rate_limit*DT
    trace = {'t': [], 'q': [], 'qd': [], 'cmd': [], 'correction': [], 'ref': [], 'meas_qd': []}
    for step in range(steps):
        alpha = min(1., step/path_steps) if path_steps else 1.
        reference = alpha*reference_step
        # measured velocity: physical plus Nyquist-rate sense noise
        # 25 Hz IS the Nyquist rate at the 50 Hz control rate, so a sampled
        # sinusoid there is identically zero: model it as the alternating
        # sequence it actually is in the recorded proprio.
        measured_rate = qd+noise*(1. if step % 2 == 0 else -1.)
        filtered_rate = filtered_rate+damp_weight*(measured_rate-filtered_rate)
        # raw correction against the CURRENT scalar reference
        raw = float(np.clip(gain2*(reference-q)-damp2*filtered_rate, -cap2, cap2))
        filtered_correction = filtered_correction+filter_weight*(raw-filtered_correction)
        candidate = reference+ff_fraction*SAG_Q2_RAD+filtered_correction
        low = max(command-slew, q-.18)          # command rate and measured tether
        high = min(command+slew, q+.18)
        command = float(np.clip(candidate, low, high))
        # second-order joint: J q'' = K(cmd - q) - D q' - tau_gravity
        gravity = SAG_Q2_RAD*STIFFNESS
        for _ in range(20):                     # fine sub-steps for accuracy
            h = DT/20
            qdd = (STIFFNESS*(command-q)-DRIVE_DAMPING*qd-gravity)/INERTIA
            qd += qdd*h
            q += qd*h
        trace['t'].append(step*DT)
        trace['q'].append(q)
        trace['qd'].append(qd)
        trace['cmd'].append(command)
        trace['correction'].append(filtered_correction)
        trace['ref'].append(reference)
        trace['meas_qd'].append(measured_rate)
    return {key: np.asarray(value) for key, value in trace.items()}


def report(trace, window_seconds=.5, cap=.12, span_limit=.002):
    """Amplitude of the sustained mode and whether the two real gates would pass."""
    fs = 1/DT
    tail = slice(int(len(trace['q'])*.6), len(trace['q']))
    q = trace['q'][tail]
    measured = trace['meas_qd'][tail]
    n = len(q)
    spectrum = np.abs(np.fft.rfft((q-q.mean())*np.hanning(n)))
    frequencies = np.fft.rfftfreq(n, fs)
    band = (frequencies > 1.2) & (frequencies < 2.2)
    amplitude = float(spectrum[band].max()/n*2) if band.any() else 0.
    samples = int(round(window_seconds/DT))+1
    ok = np.abs(measured) <= cap
    best = run = 0
    for flag in ok:
        run = run+1 if flag else 0
        best = max(best, run)
    spans = [float(np.ptp(q[i:i+samples])) for i in range(max(len(q)-samples, 1))]
    return {
        'sustained_mode_amplitude_rad': amplitude,
        'sustained_mode_pkpk_rad': float(q.max()-q.min()),
        'velocity_absmax_rad_s': float(np.abs(measured).max()),
        'longest_run_under_velocity_cap': int(best),
        'required_window_samples': samples,
        'min_position_span_rad': min(spans) if spans else None,
        'position_span_limit_rad': span_limit,
        'goal_error_limit_rad': .04,
        'would_pass_all_three_gates': bool(best >= samples and spans and min(spans) <= span_limit
                                           and abs(trace['ref'][-1]-trace['q'][-1]) < .04),
        'final_goal_error_rad': float(abs(trace['ref'][-1]-trace['q'][-1])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    # Candidate fixes, all screened against the SAME identified joint.
    candidates = [
        {'name': 'p8 as flown (gain 4, damp .05, tau_filt .10, no damping filter)',
         'kwargs': dict(gain2=4., cap2=.14, damp2=.05, tau_damp=1e6, tau_filter=.10, rate_limit=.10)},
        {'name': 'p9 damping-velocity filter only (tau .04)',
         'kwargs': dict(gain2=4., cap2=.14, damp2=.05, tau_damp=.04, tau_filter=.10, rate_limit=.10)},
        {'name': 'p9 + stronger damping .15',
         'kwargs': dict(gain2=4., cap2=.14, damp2=.15, tau_damp=.04, tau_filter=.10, rate_limit=.10)},
        {'name': 'p9 + stronger damping .40',
         'kwargs': dict(gain2=4., cap2=.14, damp2=.40, tau_damp=.04, tau_filter=.10, rate_limit=.10)},
        {'name': 'lower gain 2, higher cap .34, damp .15',
         'kwargs': dict(gain2=2., cap2=.34, damp2=.15, tau_damp=.04, tau_filter=.10, rate_limit=.10)},
        {'name': 'lower gain 1, higher cap .34, damp .15',
         'kwargs': dict(gain2=1., cap2=.34, damp2=.15, tau_damp=.04, tau_filter=.10, rate_limit=.10)},
    ]
    results = []
    for candidate in candidates:
        trace = simulate(path_seconds=15., seconds=40., **candidate['kwargs'])
        entry = {'name': candidate['name'], **candidate['kwargs'], **report(trace)}
        results.append(entry)
        print('%-52s mode=%.5f  |qdot|max=%.3f  run=%-4d  span=%.5f  err=%.4f  %s'
              % (entry['name'], entry['sustained_mode_amplitude_rad'],
                 entry['velocity_absmax_rad_s'], entry['longest_run_under_velocity_cap'],
                 entry['min_position_span_rad'] or -1, entry['final_goal_error_rad'],
                 'PASS' if entry['would_pass_both_gates'] else 'fail'))
    summary = {
        'model_not_the_robot': True,
        'prediction_is_a_hypothesis': True,
        'identified_from': {'K': STIFFNESS, 'D': DRIVE_DAMPING, 'inertia_kg_m2': INERTIA,
                            'mode_hz': 1/(2*np.pi)*np.sqrt(STIFFNESS/INERTIA),
                            'damping_ratio': DRIVE_DAMPING/(2*np.sqrt(STIFFNESS*INERTIA)),
                            'gravity_sag_q2_rad': SAG_Q2_RAD,
                            'velocity_noise_rad_s': VELOCITY_NOISE_RAD_S},
        'gates': {'velocity_cap_rad_s': .12, 'window_seconds': .5, 'position_span_limit_rad': .002},
        'results': results,
    }
    if args.output:
        args.output.write_text(json.dumps(summary, indent=2)+'\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
