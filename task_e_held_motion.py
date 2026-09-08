"""Held-object IK paths generated with Claude Opus and reviewed locally.

This helper consumes measured robot joints and an observation-derived/declared
contact goal. It does not read scene state or alter physics/scoring. Validation
checks kinematics and the table floor only; basket/object collision clearance
must be provided by the caller's goal selection and physical evaluation.
"""
import numpy as np


class HeldMotionMixin:
    """Use with a controller providing GRASP_DEPTH and _log(event, **fields).

    Accepted paths contain six arm joints per waypoint and install the same
    fields as the Task E vision controller's existing _plan_motion method.
    _plan_held_motion returns bool and leaves the active path intact on failure.
    """

    def _accept_path(self, poses, goal):
        """Install a fully planned path; partial plans never reach here."""
        self._waypoints = poses
        self._waypoint_index = 0
        self._motion_goal = np.asarray(goal, dtype=float).copy()
        self._motion_settle = 0
        self._motion_steps = 0

    def _plan_held_motion(self, goal, q, *, spacing=.02, max_joint_step=.30,
                          minimum_floor_clearance=.002):
        """Interpolate contact position and orientation along complete IK paths.

        Search both jaw signs, several tilts and yaw offsets, including the
        measured rotation for short moves. Check each solved pose using FK:
        pinch error <= .015 m, orientation error <= .12 rad, finger clearance
        above the floor, and bounded joint steps inside the finite limits.
        Rank accepted paths by maximum step, total joint travel and rotation
        change. This avoids accepting abrupt wrist branch changes.
        """
        import numpy as np
        from scipy.spatial.transform import Rotation, Slerp
        try:
            from .task_e_geometry import (BASE_POSITION, JOINT_LOWER, JOINT_UPPER, fk,
                                          solve_ik, finger_floor_clearance)
        except ImportError:
            from task_e_geometry import (BASE_POSITION, JOINT_LOWER, JOINT_UPPER, fk,
                                         solve_ik, finger_floor_clearance)

        depth = float(self.GRASP_DEPTH)
        lo = np.asarray(JOINT_LOWER, dtype=float) + 1e-5
        hi = np.asarray(JOINT_UPPER, dtype=float) - 1e-5

        q0 = np.asarray(q, dtype=float).reshape(-1)
        arm0 = np.clip(q0[:6], lo, hi)
        fing = q0[6:8] if q0.size >= 8 else np.array([.02, -.02])
        fing = np.asarray(fing, dtype=float).reshape(-1)
        if fing.size < 2:
            fing = np.array([.02, -.02])
        jaw_open = float(np.clip(abs(fing[0]) + abs(fing[1]), .01, .09))

        pose0 = fk(arm0)
        rot0, pos0 = np.array(pose0[:3, :3]), np.array(pose0[:3, 3])
        contact0 = pos0 + depth * rot0[:, 2]
        target = np.asarray(goal, dtype=float).reshape(-1)[:3]

        clear0 = float(finger_floor_clearance(pos0, rot0, jaw_open))
        travel = float(np.linalg.norm(target - contact0))
        horiz = float(np.linalg.norm(target[:2] - contact0[:2]))
        lifting = bool(target[2] >= contact0[2] - 1e-4)
        short_lift = bool(horiz <= .06 and lifting and travel <= .22)
        floor_limit = float(minimum_floor_clearance)
        if short_lift and clear0 < minimum_floor_clearance:
            # Permit the measured initial clearance for a short lift only; the
            # monotone line below never descends past the initial height.
            floor_limit = clear0 - 1e-4

        # ---- bounded deterministic candidate goal orientations -------------
        base_xy = np.asarray(BASE_POSITION, dtype=float)[:2]
        radial = target[:2] - base_xy
        nrm = float(np.linalg.norm(radial))
        radial = radial / nrm if nrm > 1e-9 else np.array([1., 0.])
        jaw0 = np.array([rot0[0, 1], rot0[1, 1], 0.])
        if float(np.linalg.norm(jaw0)) < 1e-6:
            jaw0 = np.array([radial[1], -radial[0], 0.])
        jaw0 = jaw0 / float(np.linalg.norm(jaw0))

        raw = [('hold', np.array(rot0))]
        for yaw in (0., 25., -25.):
            ca, sa = np.cos(np.deg2rad(yaw)), np.sin(np.deg2rad(yaw))
            base = np.array([ca * jaw0[0] - sa * jaw0[1],
                             sa * jaw0[0] + ca * jaw0[1], 0.])
            for tilt in (0., 20., 30., 40., 45., 50., 60.):
                ang = np.deg2rad(tilt)
                approach = np.r_[radial * np.sin(ang), -np.cos(ang)]
                for sign in (1., -1.):
                    jaw = base * sign
                    jaw = jaw - approach * float(np.dot(jaw, approach))
                    jn = float(np.linalg.norm(jaw))
                    if jn < 1e-6:
                        continue
                    jaw = jaw / jn
                    rg = np.column_stack((np.cross(jaw, approach), jaw, approach))
                    raw.append(('yaw%+.0f/tilt%.0f/s%+.0f' % (yaw, tilt, sign), rg))

        def _ang(rg):
            return float(np.linalg.norm(
                Rotation.from_matrix(rg @ rot0.T).as_rotvec()))

        cands = []
        for name, rg in raw:
            if any(float(np.linalg.norm(rg - prev)) < 1e-6 for _, prev, _ in cands):
                continue
            cands.append((name, rg, _ang(rg)))
        cands.sort(key=lambda c: c[2])

        def _errors(sol, cw, rw):
            pose = fk(sol)
            ce = float(np.linalg.norm(pose[:3, 3] + depth * pose[:3, 2] - cw))
            oe = float(np.linalg.norm(
                Rotation.from_matrix(pose[:3, :3] @ np.asarray(rw).T).as_rotvec()))
            cl = float(finger_floor_clearance(pose[:3, 3], pose[:3, :3], jaw_open))
            return ce, oe, cl

        def _solve(cw, rw, seed, final):
            pw = cw - depth * np.asarray(rw)[:, 2]
            if float(finger_floor_clearance(pw, rw, jaw_open)) < floor_limit:
                return None
            guess = np.clip(np.asarray(seed, dtype=float)[:6], lo, hi)
            pe_tol, oe_tol = (.010, .10) if final else (.015, .12)
            for attempt in range(3):
                ik = solve_ik(pw, rw, guess, orientation_weight=.15,
                              multi_start=False,
                              max_nfev=70 if attempt == 0 else 130,
                              position_tolerance=.004,
                              orientation_tolerance=.06)
                sol = np.clip(np.asarray(ik.joints, dtype=float)[:6], lo, hi)
                ce, oe, cl = _errors(sol, cw, rw)
                if ce <= pe_tol and oe <= oe_tol and cl >= floor_limit:
                    return sol
                guess = sol if attempt == 0 else .5 * (sol + np.asarray(seed)[:6])
                guess = np.clip(guess, lo, hi)
            return None

        def _build(rg, factor):
            arc = _ang(rg)
            steps = max(travel / max(float(spacing), 1e-3), arc / .18, 1.)
            n = int(min(200, max(2, np.ceil(steps * factor) + 1)))
            slerp = Slerp([0., 1.], Rotation.from_matrix(np.stack((rot0, rg))))
            ts = np.linspace(0., 1., n)
            rots = slerp(ts).as_matrix()
            path, prev = [], arm0.copy()
            worst, total = 0., 0.
            for k in range(n):
                rw = rots[k]
                cw = contact0 + (target - contact0) * ts[k]
                if lifting and cw[2] < contact0[2] - 1e-3:
                    return None
                sol = _solve(cw, rw, prev, k == n - 1)
                if sol is None:
                    return None
                step = float(np.max(np.abs(sol - prev)))
                if step > float(max_joint_step) + 1e-9:
                    return None
                worst = max(worst, step)
                total += float(np.sum(np.abs(sol - prev)))
                prev = sol
                path.append(sol.copy())
            return path, worst, total

        best, checked, complete = None, 0, 0
        for name, rg, arc in cands:
            if checked >= 10 or complete >= 3:
                break
            # cheap goal-pose reachability screen on this orientation branch
            goal_pos = target - depth * rg[:, 2]
            if float(finger_floor_clearance(goal_pos, rg, jaw_open)) < floor_limit:
                continue
            reach = False
            for seed, multi in ((arm0, False), (arm0, True)):
                ik = solve_ik(goal_pos, rg, seed, orientation_weight=.15,
                              multi_start=multi, max_nfev=130,
                              position_tolerance=.004, orientation_tolerance=.06)
                sol = np.clip(np.asarray(ik.joints, dtype=float)[:6], lo, hi)
                ce, oe, cl = _errors(sol, target, rg)
                if ce <= .010 and oe <= .10 and cl >= floor_limit:
                    reach = True
                    break
            if not reach:
                continue
            checked += 1
            res = None
            for factor in (1., 2.):
                res = _build(rg, factor)
                if res is not None:
                    break
            if res is None:
                continue
            complete += 1
            path, worst, total = res
            key = (round(worst, 3), round(total, 3), round(arc, 3))
            if best is None or key < best[0]:
                best = (key, path, name, worst, total, arc, len(path))
            if worst <= .5 * float(max_joint_step) and arc <= .35:
                break

        if best is None:
            try:
                self._log('held motion: no complete path to %s (travel %.3f m, '
                          'clear0 %.4f, %d orientation branches screened)'
                          % (np.round(target, 3).tolist(), travel, clear0, checked))
            except Exception:
                pass
            return False
        try:
            self._log('held motion: %s waypoints=%d max_step=%.3f total=%.3f '
                      'rot_arc=%.3f rad -> %s'
                      % (best[2], best[6], best[3], best[4], best[5],
                         np.round(target, 3).tolist()))
        except Exception:
            pass
        self._accept_path(best[1], target)
        return True
