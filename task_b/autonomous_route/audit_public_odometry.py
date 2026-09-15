"""Independent synthetic timing checks and offline public-odometry diagnostics.

Ground-truth pose is read only by this audit after public observations have been
replayed. It never enters the estimator and does not calibrate its constants.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from task_b.public_odometry import PublicPlanarOdometry
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run', action='append', type=Path, default=[])
    args = parser.parse_args()
    checks, evidence = {}, {}

    def check(name, callback):
        try:
            callback()
            checks[name] = True
        except Exception as error:
            checks[name] = False
            evidence[name] = repr(error)

    def obs(v=(0., 0., 0.), w=(0., 0., 0.), up=(0., 0., 1.)):
        a = np.zeros(84)
        a[:3], a[3:6], a[9:12] = v, w, -np.asarray(up)
        return a

    def close(a, b, atol=1e-10):
        assert np.allclose(a, b, rtol=0., atol=atol), (a, b)

    def timing():
        o = PublicPlanarOdometry(dt=.02)
        first = o.update(obs(v=(1., 0., 0.)))
        close(first['position_xy'], [-10., -10.])
        assert first['integrated_intervals'] == 0
        second = o.update(obs())
        close(second['position_xy'], [-9.98, -10.])
        close(o.update(obs())['position_xy'], [-9.98, -10.])
        assert o.integrated_intervals == 2 and o.update_calls == 3
        close(o.elapsed_s, .04)
    check('previous_observation_timing_without_double_integration', timing)

    def straight():
        o = PublicPlanarOdometry(.02)
        for _ in range(101):
            o.update(obs(v=(.3, -.1, 0.)))
        close(o.position_xy, [-9.4, -10.2])
        close(o.path_length_m, 2*np.hypot(.3, .1))
    check('constant_body_translation_and_path_length', straight)

    def arc():
        dt, speed, omega, intervals = .02, .2, .4, 200
        o = PublicPlanarOdometry(dt)
        for _ in range(intervals+1):
            o.update(obs(v=(speed, 0., 0.), w=(0., 0., omega)))
        angle = omega*dt
        amplitude = speed*dt*np.sin(intervals*angle/2)/np.sin(angle/2)
        expected = np.array([-10., -10.])+amplitude*np.array([
            np.cos(intervals*angle/2), np.sin(intervals*angle/2)])
        close(o.position_xy, expected)
        close(o.yaw_unwrapped_rad, intervals*angle)
    check('constant_turn_matches_independent_geometric_sum', arc)

    def rotation():
        o = PublicPlanarOdometry(1.)
        o.update(obs(w=(0., 0., np.pi/2)))
        o.update(obs(v=(.2, 0., 0.)))
        close(o.update(obs())['position_xy'], [-10., -9.8])
    check('body_forward_velocity_rotates_with_estimated_heading', rotation)

    def projection():
        up = np.array([.2, .1, 1.]); up /= np.linalg.norm(up)
        o = PublicPlanarOdometry(.02)
        for _ in range(101):
            o.update(obs(v=1.7*up, up=up))
        close(o.position_xy, [-10., -10.])
        close(o.path_length_m, 0.)
    check('gravity_projection_removes_pure_vertical_motion', projection)

    def wrapping():
        o = PublicPlanarOdometry(.02)
        for _ in range(101):
            o.update(obs(w=(0., 0., 4.)))
        close(o.yaw_unwrapped_rad, 8.)
        close(o.yaw_rad, (8.+np.pi) % (2*np.pi)-np.pi)
    check('wrapped_heading_preserves_unwrapped_continuity', wrapping)

    def invalid_inputs():
        bad = [np.zeros(83), np.r_[np.nan, np.zeros(83)], np.r_[np.inf, np.zeros(83)],
               obs(up=(0., 0., 0.)), obs(up=(1e308, 1e308, 1e308))]
        for value in bad:
            a, b = PublicPlanarOdometry(.02), PublicPlanarOdometry(.02)
            a.update(obs(v=(.3, .2, 0.))); b.update(obs(v=(.3, .2, 0.)))
            before = a.describe()
            try:
                with np.errstate(all='ignore'):
                    a.update(value)
            except ValueError:
                pass
            else:
                raise AssertionError('invalid or overflow-producing input was accepted')
            assert a.describe() == before
            assert a.update(obs()) == b.update(obs()), 'previous twist/history changed on rejection'
    check('invalid_inputs_are_atomic_including_history', invalid_inputs)

    def overflow():
        o = PublicPlanarOdometry(1e308)
        before = o.describe()
        try:
            with np.errstate(all='ignore'):
                o.update(obs(v=(1e308, 0., 0.)))
        except ValueError:
            assert o.describe() == before
            return
        before = o.describe()
        try:
            with np.errstate(all='ignore'):
                o.update(obs())
        except ValueError:
            assert o.describe() == before
        else:
            raise AssertionError('guaranteed integration overflow was accepted')
    check('overflow_integration_rejected_without_partial_commit', overflow)

    def elapsed_overflow():
        o = PublicPlanarOdometry(1e308)
        o.update(obs()); o.update(obs())
        before = o.describe()
        try:
            o.update(obs())
        except ValueError:
            assert o.describe() == before
        else:
            raise AssertionError('elapsed-time overflow was accepted')
    check('elapsed_time_overflow_preserves_state', elapsed_overflow)

    def copies_reset():
        o = PublicPlanarOdometry(.02)
        snap = o.update(obs(v=(.4, 0., 0.)))
        snap['position_xy'][0] = 123.
        p = o.position_xy; p[0] = 234.
        close(o.position_xy, [-10., -10.])
        o.update(obs()); o.reset()
        close(o.position_xy, [-10., -10.])
        close([o.yaw_rad, o.yaw_unwrapped_rad, o.elapsed_s, o.path_length_m], [0., 0., 0., 0.])
        assert o.update_calls == 0 and o.integrated_intervals == 0
        close(o.update(obs())['position_xy'], [-10., -10.])
    check('snapshot_copy_isolation_and_complete_reset', copies_reset)

    def target():
        o = PublicPlanarOdometry(1.)
        o.update(obs(w=(0., 0., np.pi/2))); o.update(obs())
        before = o.describe()
        close(o.target_body_xy([-3., -10.]), [0., -7.])
        for invalid in ([1.], [1., 2., 3.], [np.nan, 0.]):
            try:
                o.target_body_xy(invalid)
            except ValueError:
                pass
            else:
                raise AssertionError('invalid target accepted')
        assert o.describe() == before
        large = PublicPlanarOdometry(1e308)
        large.update(obs(v=(.5, 0., 0.))); large.update(obs())
        before = large.describe()
        try:
            with np.errstate(all='ignore'):
                large.target_body_xy([-1.4e308, 0.])
        except ValueError:
            assert large.describe() == before
        else:
            raise AssertionError('non-finite target offset was returned')
    check('known_waypoint_transform_is_stateless_and_validated', target)

    def dt_validation():
        for value in (0., -.02, np.nan, np.inf):
            try:
                PublicPlanarOdometry(value)
            except ValueError:
                pass
            else:
                raise AssertionError('invalid dt accepted')
    check('positive_finite_dt_required', dt_validation)

    replays = []
    for run in args.run:
        t = np.load(run/'telemetry.npz')
        o = PublicPlanarOdometry(float(t['dt']))
        xy, yaw = [], []
        for row in t['proprio']:
            state = o.update(row)
            xy.append(state['position_xy']); yaw.append(state['yaw_rad'])
        xy = np.asarray(xy); yaw = np.asarray(yaw)
        # Comparison begins only after estimates are complete. No GT update.
        q = np.asarray(t['base_quat']); w, x, y, z = q.T
        true_yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        err = np.linalg.norm(xy-t['base_xyz'][:, :2], axis=1)
        yawerr = (yaw-true_yaw+np.pi) % (2*np.pi)-np.pi
        replays.append({'run': run.name, 'observations': len(xy),
                        'final_xy_error_m': float(err[-1]), 'maximum_xy_error_m': float(err.max()),
                        'final_yaw_error_rad': float(yawerr[-1]),
                        'estimated_final_xy': xy[-1].tolist(), 'no_ground_truth_calibration': True})
    report = {'passed': all(checks.values()), 'checks': checks, 'evidence': evidence,
              'offline_replays': replays, 'scope': 'synthetic logic checks and offline finite-run error diagnostics; no carry accuracy guarantee',
              'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (Path(__file__), ROOT/'task_b/public_odometry.py')}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'passed': report['passed'], 'checks': len(checks),
                      'failed': [k for k,v in checks.items() if not v], 'offline_replays': replays}))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
