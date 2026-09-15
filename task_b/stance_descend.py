"""Bounded wheels-planted body descent, to measure the stance envelope.

Answers one question for M2: how far below its settled height can this chassis
carry its body before the official ``illegal_contact`` fires, and at what final
body height? It reaches no arm target, drives no wheel and reads no simulator
state -- root pose, contact forces and object poses stay diagnostics.

The descent is geometric, not empirical: :mod:`task_b.leg_kinematics` solves the
leg angles that raise every wheel by a requested amount in the body frame while
holding its (x, y), so the wheelbase and track are preserved and the body drops
by that amount. The solution is taken from the settled stance measured during the
settle window, not from the unloaded defaults, because the loaded legs sit well
below the default pose.

This policy only emits the leg delta. Pair it with ``--stance_hold`` so the
existing outer-loop tracking holds the commanded geometry; the achieved body
height is measured by the evaluator, never by this policy.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from task_b import leg_kinematics as legs
from task_b.control import LEG_TERM


class StanceDescendPolicy:
    def __init__(self, schema, observation_joint_names, defaults, dt=.02, *,
                 settle_calls=100, drop_max=.24, drop_rate=.03, hold_s=2., grid_step=.005):
        schema.validate()
        if dt <= 0 or not np.isfinite(dt):
            raise ValueError("Invalid timestep")
        if not np.isfinite(drop_max) or drop_max <= 0.:
            raise ValueError("drop_max must be finite and positive")
        if not np.isfinite(drop_rate) or drop_rate <= 0.:
            raise ValueError("drop_rate must be finite and positive")
        if not np.isfinite(hold_s) or hold_s < 0.:
            raise ValueError("hold_s must be finite and non-negative")
        if not 0. < grid_step <= drop_max:
            raise ValueError("grid_step must lie in (0, drop_max]")
        self.schema = schema
        self.names = tuple(observation_joint_names)
        self.term = schema.term(LEG_TERM)
        if not isinstance(defaults, Mapping):
            raise TypeError("defaults must be a joint-name -> position mapping, as the evaluator builds")
        missing = [name for name in self.term.joint_names if name not in defaults]
        if missing:
            raise ValueError(f"defaults is missing leg joints {missing}")
        self.defaults = np.asarray([defaults[name] for name in self.term.joint_names], dtype=np.float64)
        self.scale = float(self.term.scale)
        self.observation_ids = np.array([self.names.index(name) for name in self.term.joint_names])
        self.dt, self.settle_calls = float(dt), int(settle_calls)
        self.drop_max, self.drop_rate, self.hold_s = float(drop_max), float(drop_rate), float(hold_s)
        self.grid_step = float(grid_step)
        self.calls, self.alpha = 0, 0.
        self.state, self.state_reason, self.done_reason = 'SETTLE', 'measuring_settled_leg_geometry', None
        self.samples, self.settled_q, self.grid, self.grid_drop = [], None, None, None
        self.drop_cmd = 0.
        self.margin_calls = int(np.ceil(hold_s / dt))
        self.hold_start = None
        self.debug = {}

    def _leg_angles(self, obs):
        return obs[12 + self.observation_ids] + self.defaults

    def _build_grid(self, q_settled):
        """Solve the descent on a grid of drops, warm-starting along one branch.

        ``leg_kinematics.descend`` takes a rise *relative to the seed it is given*,
        so the grid is accumulated in increments and each point seeds the next.
        That keeps ``q(drop)`` on a single continuous branch instead of jumping
        between equivalent IK solutions. The achieved rise is then re-measured
        against the settled pose, so an increment/absolute mix-up cannot pass.
        """
        by_corner = {c: q_settled[[self.term.joint_names.index(n) for n in legs.leg_joint_names(c)]]
                     for c in legs.CORNERS}
        drops = list(np.arange(0., self.drop_max, self.grid_step))
        if not drops or abs(drops[-1] - self.drop_max) > 1e-9:
            drops.append(self.drop_max)
        grid, worst = [], 0.
        seed = {c: by_corner[c].copy() for c in legs.CORNERS}
        previous = 0.
        for drop in drops:
            solved, residual = legs.descend_all(seed, float(drop) - previous)
            if residual > 1e-6:
                raise RuntimeError(f"Leg descent IK residual {residual:.2e} m at drop {drop:.3f} m")
            for corner in legs.CORNERS:
                achieved = legs.foot_body_xyz(corner, *solved[corner])[2] \
                    - legs.foot_body_xyz(corner, *by_corner[corner])[2]
                # The grid is accumulated in increments, so the commanded drop and
                # the FK-measured rise drift apart by rounding over ~50 points
                # (observed 1e-6 m). 1e-4 m still catches any real scale error,
                # which would be off by centimetres.
                if abs(achieved - drop) > 1e-4:
                    raise RuntimeError(f"{corner} achieved a {achieved:.6f} m rise at commanded "
                                       f"drop {drop:.6f} m; the grid is not tracking the request")
            worst = max(worst, residual)
            seed, previous = solved, float(drop)
            grid.append(np.array([solved[self._corner_of(name)][self._link_of(name)]
                                  for name in self.term.joint_names]))
        self.grid, self.grid_drop, self.grid_residual = np.asarray(grid), np.asarray(drops), worst

    def _corner_of(self, joint_name):
        return joint_name.split("_")[0]

    def _link_of(self, joint_name):
        link = joint_name.split("_")[1]
        if link not in legs.LEG_LINKS:
            raise ValueError(f"Unrecognized leg joint '{joint_name}'")
        return legs.LEG_LINKS.index(link)

    def _target(self, drop):
        if self.grid is None:
            return self.settled_q
        clipped = float(np.clip(drop, 0., self.drop_max))
        return np.array([np.interp(clipped, self.grid_drop, self.grid[:, i])
                         for i in range(self.grid.shape[1])])

    def act(self, proprio):
        obs = np.asarray(proprio, dtype=float).reshape(-1)
        if obs.size != 84 or not np.isfinite(obs).all():
            raise ValueError(f"stance_descend requires finite proprio84, got {obs.size}")
        self.calls += 1
        action = np.zeros(self.schema.total_dim, dtype=np.float64)

        if self.calls <= self.settle_calls:
            self.samples.append(self._leg_angles(obs).copy())
            del self.samples[:-20]
            if self.calls == self.settle_calls:
                self.settled_q = np.mean(self.samples, axis=0)
                self._build_grid(self.settled_q)
            self.state = 'SETTLE'
            measured = self._leg_angles(obs)
            self.debug = {'drop_cmd': 0., 'grid_points': 0 if self.grid is None else len(self.grid_drop),
                          'leg_measured_rad': measured.tolist(),
                          'leg_joint_names': list(self.term.joint_names),
                          'leg_settle_samples': len(self.samples)}
            return action.astype(np.float32)

        if self.settled_q is None:
            self.settled_q = self._leg_angles(obs).copy()
            self._build_grid(self.settled_q)

        elapsed_s = (self.calls - self.settle_calls) * self.dt
        self.drop_cmd = min(self.drop_max, self.drop_rate * elapsed_s)
        target = self._target(self.drop_cmd)
        delta = target - self.settled_q
        action[self.term.start:self.term.stop] = delta / self.scale

        measured = self._leg_angles(obs)
        tracking = float(np.max(np.abs(measured - self.settled_q - delta)))
        at_max = self.drop_cmd >= self.drop_max
        if at_max:
            self.hold_start = self.calls if self.hold_start is None else self.hold_start
            self.state, self.state_reason = 'HOLD', 'holding_maximum_commanded_descent'
            if (self.calls - self.hold_start) * self.dt >= self.hold_s:
                self.done_reason = 'stance_descend_complete'
                self.state = 'STOPPED'
        else:
            self.state, self.state_reason = 'DESCEND', 'descending_at_fixed_geometric_rate'

        self.debug = {
            'drop_cmd_m': self.drop_cmd, 'drop_elapsed_s': elapsed_s,
            'grid_points': len(self.grid_drop), 'grid_ik_residual_max_m': self.grid_residual,
            'leg_target_rad': target.tolist(), 'leg_settled_rad': self.settled_q.tolist(),
            'leg_delta_rad': delta.tolist(), 'leg_measured_rad': measured.tolist(),
            'leg_tracking_error_rad': tracking, 'leg_joint_names': list(self.term.joint_names),
        }
        return action.astype(np.float32)

    def describe(self):
        return dict(mode='stance_descend', drop_max_m=self.drop_max, drop_rate_m_s=self.drop_rate,
                    hold_s=self.hold_s, settle_calls=self.settle_calls, calls=self.calls,
                    state=self.state, state_reason=self.state_reason, done_reason=self.done_reason,
                    grid_points=0 if self.grid is None else len(self.grid_drop),
                    grid_ik_residual_max_m=self.grid_residual if self.grid is not None else None,
                    source='task_b/leg_kinematics.py wheels-planted vertical descent',
                    inputs='public proprio and the static joint/action schema only',
                    claim='the commanded drop is geometric; whether the chassis can carry it, '
                          'keep its wheels down and stay clear of the official contact bodies is '
                          'measured by the evaluator, not asserted here')
