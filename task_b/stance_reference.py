"""Optional slow leg-reference change before :class:`StanceHold`.

``compact`` starts at the mean of the last 20 settling observations and adds
``(desired_actual_q - settled_q) / leg_scale`` to the incoming leg action.
StanceHold must receive the same observations/calls and settling duration. Its
existing ``reference + requested_delta`` then follows this reference without
confusing the loaded joint angles with the unloaded official defaults.

Only public joint observations and static robot/action geometry are used. No
physics, actuator gains, root pose, object pose or score are read or changed.
The compact target is an unvalidated motion candidate: completing the reference
trajectory does not prove that the robot attained a stable stance.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from task_b.control import LEG_TERM


# Desired ACTUAL leg angles [hip, thigh, calf], from static B2w joint-chain IK.
# Nominal wheelbase/track .76/.70 m at body height ~.519 m and pitch ~.064 rad.
# These design values are not runtime pose estimates or dynamic guarantees.
COMPACT_JOINT_REFERENCE = {
    'FR': (-.3023523168313238, .8130831661377943, -1.8754745945886573),
    'FL': (.30696657668610144, .8173372201608101, -1.8881043202266299),
    'RR': (-.27114297155288936, .9786392186138522, -1.729057945170937),
    'RL': (.2750100973730118, .9858544286612007, -1.7429508632418722),
}
COMPACT_SHORT_JOINT_REFERENCE = {
    'FR': (-.3228052193353918, .8989834889270468, -1.8522322288050448),
    'FL': (.3276493691378279, .9043952913485503, -1.8652912758550064),
    'RR': (-.2934330988364878, .8864735283244438, -1.7217835132468908),
    'RL': (.2975832097573324, .8930821152109296, -1.7360366617441942),
}
PROFILES = ('off', 'compact', 'compact_short')
_LINKS = ('hip', 'thigh', 'calf')
_COMPACT_BY_NAME = {f'{corner}_{link}_joint': q
                    for corner, values in COMPACT_JOINT_REFERENCE.items()
                    for link, q in zip(_LINKS, values)}
# Original B2w physical joint limits, exported from the official USD. A caller
# can supply the current articulation's static hard limits by joint name.
_ORIGINAL_HARD_LIMITS = {'hip': (-.87, .87), 'thigh': (-.94, 4.69),
                         'calf': (-2.82, -.43)}


class StanceReference:
    """Add an opt-in compact stance trajectory while preserving wheel/arm input.

    Typical order::

        action = reference.apply(policy_action, proprio)
        if reference.requires_wheel_stop:
            action[wheel.start:wheel.stop] = 0.
        action = stance_hold.apply(action, proprio)

    The wheel gate is deliberately explicit: this adapter preserves the caller's
    wheel and arm slices. For ``compact`` the caller must stop wheels throughout
    calibration and transition; after that, it must separately assess stability.
    Incoming leg actions must be zero during compact calibration, so StanceHold
    samples the original load-supporting equilibrium.
    """

    def __init__(self, schema, observation_joint_names, *, profile='off', dt=.02,
                 settle_calls=100, transition_seconds=5.,
                 hard_joint_pos_limits: Mapping | None = None):
        schema.validate()
        if profile not in PROFILES:
            raise ValueError(f'stance profile must be one of {PROFILES}')
        if not np.isfinite(dt) or dt <= 0.:
            raise ValueError('dt must be positive and finite')
        if isinstance(settle_calls, bool) or not isinstance(settle_calls, (int, np.integer)) or settle_calls < 0:
            raise ValueError('settle_calls must be a nonnegative integer')
        if profile != 'off' and settle_calls < 20:
            raise ValueError('compact requires at least 20 settling calls')
        if not np.isfinite(transition_seconds) or not 4. <= transition_seconds <= 6.:
            raise ValueError('transition_seconds must lie in [4, 6]')
        observed = tuple(observation_joint_names)
        if len(observed) != 24 or len(set(observed)) != 24 or set(observed) != set(schema.joint_names):
            raise ValueError('Expected each of the 24 observed joints exactly once')
        self.schema, self.term = schema, schema.term(LEG_TERM)
        if set(self.term.joint_names) != set(_COMPACT_BY_NAME):
            raise ValueError('Unexpected B2w leg names')
        if self.term.clip is not None:
            raise ValueError('StanceReference requires an unclipped leg position action term')
        self.ids = np.array([observed.index(n) for n in self.term.joint_names])
        jids = [schema.joint_index(n) for n in self.term.joint_names]
        self.defaults = schema.default_joint_pos[jids].copy()
        self.soft_limits = schema.soft_joint_pos_limits[jids].copy()
        targets = COMPACT_SHORT_JOINT_REFERENCE if profile == 'compact_short' else COMPACT_JOINT_REFERENCE
        target_by_name = {f'{corner}_{link}_joint': q for corner, values in targets.items()
                          for link, q in zip(_LINKS, values)}
        self.target = np.array([target_by_name[n] for n in self.term.joint_names])
        if hard_joint_pos_limits is None:
            bounds = [_ORIGINAL_HARD_LIMITS[n.split('_')[1]] for n in self.term.joint_names]
            self.hard_limit_source = 'original B2w USD static leg limits'
        else:
            if not isinstance(hard_joint_pos_limits, Mapping) or not set(self.term.joint_names).issubset(hard_joint_pos_limits):
                raise ValueError('hard_joint_pos_limits must map all 12 leg joint names to bounds')
            bounds = [hard_joint_pos_limits[n] for n in self.term.joint_names]
            self.hard_limit_source = 'caller supplied articulation hard limits by joint name'
        self.hard_limits = np.asarray(bounds, dtype=float)
        if (self.hard_limits.shape != (12, 2) or not np.isfinite(self.hard_limits).all()
                or np.any(self.hard_limits[:, 0] >= self.hard_limits[:, 1])):
            raise ValueError('Invalid hard leg limits')
        if (np.any(self.soft_limits[:, 0] < self.hard_limits[:, 0] - 1e-5)
                or np.any(self.soft_limits[:, 1] > self.hard_limits[:, 1] + 1e-5)):
            raise ValueError('Soft leg limits extend outside the supplied hard limits')
        if profile != 'off':
            self._check_bounds(self.target, 'compact actual joint reference')
        self.profile, self.dt = profile, float(dt)
        self.settle_calls, self.transition_seconds = int(settle_calls), float(transition_seconds)
        self.calls, self.samples, self.reference = 0, [], None
        self.alpha = 0.
        self.desired_delta = np.zeros(12)
        self.debug = {'status': 'off' if profile == 'off' else 'settle',
                      'requires_wheel_stop': profile != 'off'}

    def _check_bounds(self, q, label):
        if (not np.isfinite(q).all() or np.any(q < self.hard_limits[:, 0] - 1e-5)
                or np.any(q > self.hard_limits[:, 1] + 1e-5)):
            raise ValueError(f'{label} exceeds physical leg limits')

    @property
    def trajectory_complete(self):
        """True after the nominal ramp; this is not a dynamic stability flag."""
        return self.profile == 'off' or self.alpha >= 1.

    @property
    def requires_wheel_stop(self):
        """Caller must zero wheels through calibration and the nominal ramp."""
        return self.profile != 'off' and not self.trajectory_complete

    def apply(self, action, proprio):
        # Float64 preserves incoming float32/float64 wheel and arm values exactly.
        # The evaluator or StanceHold performs its normal float32 conversion later.
        out = np.array(action, dtype=float, copy=True)
        obs = np.asarray(proprio, dtype=float)
        if (out.shape != (24,) or obs.shape != (84,) or not np.isfinite(out).all()
                or not np.isfinite(obs).all()
                or np.any(np.abs(out) > np.finfo(np.float32).max)):
            raise ValueError('StanceReference requires finite action24 and public proprio84')
        self.calls += 1
        if self.profile == 'off':
            self.debug = {'status': 'off', 'requires_wheel_stop': False}
            return out
        leg = slice(self.term.start, self.term.stop)
        q = obs[12 + self.ids] + self.defaults
        self._check_bounds(q, 'observed leg positions')
        if self.calls <= self.settle_calls:
            if np.max(np.abs(out[leg])) > 1e-7:
                raise ValueError('compact calibration requires zero incoming leg action')
            self.samples.append(q.copy())
            self.samples = self.samples[-20:]
            if self.calls == self.settle_calls:
                self.reference = np.mean(self.samples, axis=0)
            self.debug = {'status': 'settle', 'calls': self.calls,
                          'requires_wheel_stop': True, 'sample_count': len(self.samples)}
            return out
        u = np.clip((self.calls - self.settle_calls) * self.dt / self.transition_seconds, 0., 1.)
        # Quintic interpolation has zero velocity and acceleration at both ends.
        self.alpha = float(u ** 3 * (10. - 15.*u + 6.*u*u))
        self.desired_delta = self.alpha * (self.target - self.reference)
        requested_delta = out[leg] * self.term.scale + self.desired_delta
        self._check_bounds(self.reference + requested_delta, 'combined actual joint reference')
        self._check_bounds(self.defaults + requested_delta, 'feedforward joint target')
        out[leg] = requested_delta / self.term.scale
        if not np.isfinite(out).all() or np.any(np.abs(out) > np.finfo(np.float32).max):
            raise ValueError('Normalized stance action is not representable as float32')
        self.debug = dict(status='holding_compact_reference' if self.trajectory_complete else 'transition',
                          calls=self.calls, alpha=self.alpha,
                          requires_wheel_stop=self.requires_wheel_stop,
                          joint_names=list(self.term.joint_names),
                          settled_reference_rad=self.reference.tolist(),
                          desired_delta_rad=self.desired_delta.tolist(),
                          desired_actual_q_rad=(self.reference+self.desired_delta).tolist(),
                          actual_reference_error_max_rad=float(np.max(np.abs(self.reference+self.desired_delta-q))))
        return out

    def describe(self):
        return dict(profile=self.profile, inputs='public leg q and static action/robot geometry',
                    settle_calls=self.settle_calls, reference_samples=20,
                    transition_seconds=self.transition_seconds, interpolation='quintic smoothstep',
                    joint_names=list(self.term.joint_names), compact_actual_q_rad=self.target.tolist(),
                    nominal_wheelbase_track_m=[.68, .72] if self.profile == 'compact_short' else [.76, .70],
                    hard_limit_source=self.hard_limit_source,
                    composition='add desired actual q minus settled q to incoming leg delta before StanceHold',
                    wheel_gate='caller zeros wheels while requires_wheel_stop is true',
                    claim='static geometry candidate; ramp completion is not dynamic stance success')
