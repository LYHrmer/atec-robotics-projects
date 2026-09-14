"""Hold the measured settled leg geometry through public joint feedback.

The official actuator gains are untouched. This controller adds bounded
position commands around the original command that supported the robot at
rest; it does not try to straighten legs to their unloaded default angles.
"""
import numpy as np
from task_b.control import LEG_TERM


class StanceHold:
    def __init__(self, schema, observation_joint_names, dt=.02, settle_calls=100,
                 gain=2., damping=.12, cap=.4):
        schema.validate()
        self.schema, self.term = schema, schema.term(LEG_TERM)
        self.ids = np.array([tuple(observation_joint_names).index(n) for n in self.term.joint_names])
        jids = [schema.joint_index(n) for n in self.term.joint_names]
        self.defaults = schema.default_joint_pos[jids]
        self.limits = schema.soft_joint_pos_limits[jids]
        self.dt, self.settle_calls = dt, settle_calls
        self.gain, self.damping, self.cap = gain, damping, cap
        self.calls, self.samples, self.reference = 0, [], None
        self.correction = np.zeros(12)
        self.debug = {}

    def apply(self, action, proprio):
        out = np.asarray(action, dtype=np.float32).copy()
        obs = np.asarray(proprio, dtype=float).reshape(-1)
        if out.shape != (24,) or obs.shape != (84,) or not np.isfinite(out).all() or not np.isfinite(obs).all():
            raise ValueError('StanceHold requires finite action24 and public proprio84')
        self.calls += 1
        q = obs[12+self.ids] + self.defaults
        qdot = obs[36+self.ids]
        if self.calls <= self.settle_calls:
            self.samples.append(q.copy())
            self.samples = self.samples[-20:]
            if self.calls == self.settle_calls:
                self.reference = np.mean(self.samples, axis=0)
            self.debug = {'status': 'settle', 'calls': self.calls}
            return out
        if self.reference is None:
            self.reference = q.copy()
        requested_delta = out[self.term.start:self.term.stop].astype(float)*self.term.scale
        error = self.reference + requested_delta - q
        desired = np.clip(self.gain*error-self.damping*qdot, -self.cap, self.cap)
        self.correction += np.clip(desired-self.correction, -self.dt, self.dt)
        target = np.clip(self.defaults+requested_delta+self.correction, self.limits[:, 0], self.limits[:, 1])
        out[self.term.start:self.term.stop] = (target-self.defaults)/self.term.scale
        self.debug = dict(status='hold_settled_joint_geometry', error_rad=error.tolist(),
                          correction_rad=self.correction.tolist(), reference_rad=self.reference.tolist())
        return out

    def describe(self):
        return dict(inputs='public q, qdot and static action schema', gain=self.gain,
                    damping_s=self.damping, correction_cap_rad=self.cap,
                    correction_slew_rad_s=1., settle_calls=self.settle_calls,
                    reference='mean of final 20 settling joint samples',
                    claim='experimental outer-loop joint holding; original physics unchanged')
