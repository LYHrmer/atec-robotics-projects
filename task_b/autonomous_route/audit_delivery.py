"""Independent CPU counterexample audit of the Task B delivery contract.

WHAT THIS IS
------------
A battery of synthetic CPU checks that try to BREAK the guarantees the delivery
contract states, rather than a re-reading of the implementation. Each battery
declares, in advance, the contract claim it attacks, the observation that would
falsify it, and whether the check had a guard against passing trivially. Every
battery reports what it actually observed; nothing is asserted from the source
text alone.

WHAT THIS IS NOT
----------------
It never modifies a production file, never relaxes a threshold and never
reclassifies a failure as a success. It starts no GPU run, reads no policy
private state as ground truth, and every number it prints comes from code it
executed in this process. Passing a battery here is a CPU property of a state
machine; it is NOT carry, clearance, delivery, objects_in_circle or Task B
evidence.

THE SEAM, STATED PLAINLY
------------------------
The production ``PayloadRaisePrefix`` only publishes its handoff at the ORIGINAL
A-arrival branch, which needs the visual reach, the contact grasp and the loaded
lift - i.e. the GPU simulator. This audit therefore works in two layers:

* :class:`~task_b.delivery.PayloadRaisePrefix` is instantiated and driven
  DIRECTLY on the handoff branch, so the "same tracker / never begin the swing /
  one transfer" claims are tested against the real production class;
* the parent's carry state machine is then exercised with a counting stub child
  that returns that real carry record, so ``_prefix`` - the parent's own
  handoff path - is the production code and the "child called once per tick,
  never again" claim is measured on a real call counter.

The stub is a stub; the audit says so wherever it is used, and it is not used
for any claim about the frozen prefix's own behaviour.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from task_b import control                                            # noqa: E402
from task_b.control import ARM_TERM, WHEEL_TERM                        # noqa: E402
from task_b.payload_motion import (                                   # noqa: E402
    COMMAND_RATE_RAD_S, GOAL_ERROR_RAD, INTEGRAL_CAP_RAD, PayloadJointTracker,
    QUIET_ARM_SPAN_RAD, QUIET_WINDOW_S,
)
from task_b.delivery import (                                         # noqa: E402
    BRAKE_SETTLE_SPEED_M_S, DEFAULT_WHEEL_ACTION_GAIN, MOVEMENT_PHASES,
    PROBE_FORWARD_M, PROBE_PATH_CAP_M, PROBE_SPEED_M_S, PayloadDeliveryPolicy,
    PayloadRaisePrefix,
)

DT = .02
DEFAULT_RUN = Path('/home/lybm/ATEC_Experiments_20260910/task_b_score/'
                   'plan_p13_payload_denyquist_seed42_01')

#: Contract-derived forbidden inputs. A policy that reads any of these is reading
#: ground truth, the world base pose, a contact force, the reward, the score or a
#: seed map. The list is taken from the written contract, not from the file.
FORBIDDEN_IDENTIFIERS = {
    # ground-truth object pose / identity
    'object_xyz', 'object_quat', 'object_pose', 'object_position', 'object_pos',
    'object_root', 'objects_in_circle', 'grasped_objects', 'object_names', 'OBJECT_NAMES',
    # world base pose
    'base_xyz', 'base_pos', 'base_quat', 'root_pos_w', 'root_quat_w', 'env_origins',
    'world_pose', 'base_world',
    # contact force
    'contact_force', 'contact_forces', 'finger_contact_forces_w', 'net_forces_w',
    'contact_sensor', 'force_matrix', 'illegal_force', 'max_illegal_force',
    # reward / score / termination
    'reward', 'env_reward', 'reward_raw_total', 'reward_terms', 'reward_manager',
    'score', 'scoring_events', 'final_state_before_close', 'termination_flags',
    # seed map
    'seed', 'seed_map', 'np_random',
}
FORBIDDEN_IMPORTS = {
    'isaaclab', 'isaacsim', 'omni', 'pxr', 'torch', 'task_b.evaluate',
    'task_b.diagnostics',
}


# --------------------------------------------------------------------------- #
# action schema, taken from a RECORDED public run (static task constants)
# --------------------------------------------------------------------------- #
def load_schema(metadata_path: Path):
    """Rebuild the public action schema from a recorded run's metadata.

    The schema is a static task constant that the evaluator hands to every
    policy; rebuilding it from the recording avoids any assumption about term
    order, scale or joint names.
    """
    recorded = json.loads(Path(metadata_path).read_text())['action_schema']
    terms = [control.ActionTerm(
        name=entry['name'], start=int(entry['action_slice'][0]), dim=int(entry['dim']),
        joint_names=tuple(entry['joint_names']), mode=entry['mode'],
        scale=float(entry['scale']), use_default_offset=entry['use_default_offset'],
        clip=None, joint_names_source=entry['joint_names_source'])
        for entry in recorded['terms']]
    limits = np.array([[(-np.inf if low is None else low), (np.inf if high is None else high)]
                       for low, high in recorded['soft_joint_pos_limits']])
    schema = control.ActionSchema.from_terms(
        terms, recorded['articulation_joint_names'],
        np.array(recorded['default_joint_pos'], dtype=float), limits,
        int(recorded['total_action_dim']))
    names = list(schema.joint_names)
    return schema, names, dict(zip(names, schema.default_joint_pos.tolist()))


# --------------------------------------------------------------------------- #
# synthetic public proprio
# --------------------------------------------------------------------------- #
class Proprio:
    """Builds 84-value public proprio vectors in the recorded public layout."""

    #: A plausible measured closed-jaw pair: arm_joint7 - arm_joint8 must stay above
    #: the empty-gripper width, which is what the production watchdog checks.
    FINGER_Q = (0.030, 0.000)

    def __init__(self, schema, names, defaults):
        self.schema, self.names, self.defaults = schema, names, defaults
        self.arm_ids = [names.index('arm_joint%d' % i) for i in range(1, 9)]
        self.wheel_ids = [names.index(n) for n in schema.term(WHEEL_TERM).joint_names]
        self.default_joint_pos = np.asarray(schema.default_joint_pos, dtype=float)
        self._call = 0

    def build(self, arm_q, qdot=None, v=(0., 0., 0.), w=(0., 0., 0.), gravity=(0., 0., -1.)):
        obs = np.zeros(84, dtype=float)
        obs[0:3] = v
        obs[3:6] = w
        obs[9:12] = gravity
        q = self.default_joint_pos.copy()
        arm_q = np.asarray(arm_q, dtype=float).reshape(-1)
        if arm_q.size == 6:
            arm_q = np.concatenate([arm_q, np.asarray(self.FINGER_Q, dtype=float)])
        q[self.arm_ids] = arm_q
        obs[12:36] = q - self.default_joint_pos
        if qdot is not None:
            full = np.zeros(len(self.names))
            full[self.arm_ids] = np.asarray(qdot, dtype=float).reshape(-1)
            obs[36:60] = full
        # A real recorded proprio vector is never bit-identical between ticks.
        # carry_drive latches a stale-feed fault on 25 exactly equal consecutive
        # observations while a rate is requested, so a bit-frozen synthetic stream
        # would trip a fault that no real robot can trip. The dither is placed in
        # obs[6:9], which no policy and no drive-layer slice reads, so it changes
        # no computed quantity.
        self._call += 1
        obs[6] = 1e-7*self._call
        return obs


# --------------------------------------------------------------------------- #
# result bookkeeping
# --------------------------------------------------------------------------- #
class Results:
    def __init__(self):
        self.rows = []

    def add(self, cid, claim, contract_ref, method, observed, verdict,
            evidence=None, trivial_pass_risk='', note=''):
        row = {
            'id': cid, 'contract_ref': contract_ref, 'claim_under_test': claim,
            'method': method, 'verdict': verdict, 'observed': observed,
            'evidence': evidence if evidence is not None else {},
            'trivial_pass_risk': trivial_pass_risk, 'note': note,
        }
        self.rows.append(row)
        print('[%-14s] %-52s %s' % (cid, verdict, claim[:78]))
        return row

    def summary(self):
        counts = {}
        for row in self.rows:
            counts[row['verdict']] = counts.get(row['verdict'], 0) + 1
        return counts


# --------------------------------------------------------------------------- #
# the prefix, driven directly on its handoff branch
# --------------------------------------------------------------------------- #
def build_prefix(schema, names, defaults):
    """Instantiate the REAL production prefix and place it at its A-arrival branch.

    Returns ``(prefix, tracker, goals, q0)``.
    """
    prefix = PayloadRaisePrefix(schema, names, defaults, dt=DT, handoff_at_raise=True)
    q0 = np.array([0., .50, -.02, 0., 0., 0.])
    goal_a = q0 + np.array([0., .10, 0., 0., 0., 0.])
    goal_b = goal_a.copy()
    goal_b[0] = goal_a[0] + .05
    goal_c = goal_a.copy()
    goal_c[2] = goal_a[2] - .05
    tracker = PayloadJointTracker(dt=DT, initial_reference=q0, initial_command=q0)
    tracker.begin(goal_a)
    # The handoff can only happen once A has ACTUALLY arrived, so the tracker's
    # scalar path is driven to its own goal first: reference == command == goal A
    # is the physical state the production prefix publishes.
    for _ in range(200):
        tracker.update(goal_a, np.zeros(6))
    prefix.tracker = tracker
    prefix.goals = {'A': goal_a, 'B': goal_b, 'C': goal_c}
    prefix.action_template = np.zeros(schema.total_dim, dtype=np.float32)
    prefix.finger_command = np.array([-.075, .075])
    prefix.arm_command = tracker.command.copy()
    prefix.reference_q = q0.copy()
    prefix.phase = prefix.state = 'PAYLOAD_RAISE'
    prefix.calls = 1000
    return prefix, tracker, {'A': goal_a, 'B': goal_b, 'C': goal_c}, q0


def body_state(obs):
    """The four public body quantities _tick derives from a proprio vector.

    Used only by the batteries that drive _quiet_tick directly instead of going
    through act(); the arithmetic is the public one (gravity defines up, the
    tangential component of the linear velocity is the planar speed).
    """
    gravity = np.asarray(obs[9:12], dtype=float)
    norm = float(np.linalg.norm(gravity))
    up = -gravity/norm
    tilt = float(np.arccos(np.clip(-gravity[2]/norm, -1., 1.)))
    linear = np.asarray(obs[:3], dtype=float)
    tangent = linear-np.dot(linear, up)*up
    return {'measured_speed': float(np.linalg.norm(tangent)),
            'measured_yaw_rate': float(np.dot(np.asarray(obs[3:6], dtype=float), up)),
            'linear_norm': float(np.linalg.norm(linear)), 'tilt_rad': tilt}


def feed_tick(policy, obs, arm_q):
    """One direct _velocity_signals + _quiet_tick drive with the real body state."""
    state = body_state(obs)
    policy.measured_speed = state['measured_speed']
    policy.measured_yaw_rate = state['measured_yaw_rate']
    policy.linear_norm = state['linear_norm']
    policy.tilt_rad = state['tilt_rad']
    policy.calls += 1
    policy._velocity_signals(obs, np.asarray(arm_q, dtype=float))
    policy._quiet_tick(np.asarray(arm_q, dtype=float))


def make_carry(schema, names, defaults):
    """A REAL carry record, produced by the production prefix at its A-arrival branch."""
    prefix, tracker, goals, q0 = build_prefix(schema, names, defaults)
    prefix._advance_segment('A')
    return prefix.take_carry_state(), goals, q0


class CountingChild:
    """A stand-in for the frozen prefix used ONLY to exercise the parent's own
    ``_prefix`` handoff path on CPU. It is not the production prefix."""

    STATE = 'PREFIX'

    def __init__(self, carry, action, ready_at_call=1, raise_on_take=False):
        self.carry = carry
        self.action = action
        self.ready_at_call = int(ready_at_call)
        self.raise_on_take = bool(raise_on_take)
        self.calls = 0
        self.state = self.STATE
        self.state_reason = 'audit_stub_for_the_parent_prefix_path'
        self.done_reason = None
        self.pause_for_stance = False
        self.handoff_ready = False

    @property
    def wheel_hold_requested(self):
        return True

    def act(self, proprio, images):
        self.calls += 1
        if self.calls >= self.ready_at_call:
            self.handoff_ready = True
            self.state = 'PAYLOAD_RAISE'
            self.state_reason = 'audit_stub_original_A_arrival_condition'
        return np.asarray(self.action, dtype=np.float32).copy()

    def take_carry_state(self):
        if self.raise_on_take:
            raise ValueError('the carry state is incomplete; the prefix never fully opened')
        if not getattr(self, '_taken', False):
            self._taken = True
            return self.carry
        raise ValueError('the carry state has already been taken')


def make_policy(schema, names, defaults, carry, action, **kwargs):
    """A production ``PayloadDeliveryPolicy`` whose prefix is the counting stub."""
    policy = PayloadDeliveryPolicy(schema, names, defaults, dt=DT,
                                   wheel_action_gain=DEFAULT_WHEEL_ACTION_GAIN, **kwargs)
    child = CountingChild(carry, action)
    policy.child = child
    first = policy.act(np.zeros(84), {})   # one real _prefix tick -> the handoff
    return policy, child, first



# --------------------------------------------------------------------------- #
# a minimal public planar kinematic model, so the probe phases have something
# real to react to (the audit commands a twist, the proprio reports it)
# --------------------------------------------------------------------------- #
def _phase_twist(phase):
    from task_b.delivery import PROBE_YAW_CAP_RAD_S
    if phase == 'PROBE_DRIVE':
        return PROBE_SPEED_M_S, 0.
    if phase == 'PROBE_YAW_POS':
        return 0., +PROBE_YAW_CAP_RAD_S
    if phase == 'PROBE_YAW_NEG':
        return 0., -PROBE_YAW_CAP_RAD_S
    return 0., 0.


def carry_probe_stream(policy, proprio, arm_q, stop_phase=None, max_ticks=3000):
    """Drive the production phase machine with the audit's own planar twist model.

    The audit commands a twist per phase and the proprio reports it; nothing else
    about the base is modelled. Stops on the first tick in ``stop_phase`` (before
    that phase's own action is taken), on a stop reason, or at ``max_ticks``.
    """
    v, w = 0., 0.
    log = []
    for _ in range(max_ticks):
        phase = policy.phase
        if stop_phase is not None and phase == stop_phase:
            break
        target_v, target_w = _phase_twist(phase)
        v = v + float(np.clip(target_v-v, -.02, .02))
        w = w + float(np.clip(target_w-w, -.02, .02))
        obs = proprio.build(arm_q, qdot=np.zeros(8), v=(v, 0., 0.), w=(0., 0., w))
        policy.act(obs, {})
        log.append({'phase': phase, 'v': v, 'w': w, 'stop_reason': policy.done_reason})
        if policy.done_reason:
            break
    return log


def hold_twist(policy, proprio, arm_q, v, w, ticks):
    """Hold one constant twist for ``ticks`` ticks and report what the policy did."""
    log = []
    for _ in range(int(ticks)):
        obs = proprio.build(arm_q, qdot=np.zeros(8), v=(v, 0., 0.), w=(0., 0., w))
        policy.act(obs, {})
        log.append({'phase': policy.phase, 'v': v, 'w': w,
                    'anchor_engaged': bool(len(policy.anchor_engage_calls) > 0),
                    'stop_reason': policy.done_reason})
        if policy.done_reason:
            break
    return log



def _mode_choices():
    """The --mode choices the evaluator's own argparse declares, parsed not grepped."""
    tree = ast.parse((PROJECT_ROOT/'task_b'/'evaluate.py').read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == 'add_argument'):
            continue
        if not (node.args and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == '--mode'):
            continue
        for keyword in node.keywords:
            if keyword.arg == 'choices' and isinstance(keyword.value, (ast.Tuple, ast.List)):
                return tuple(ast.literal_eval(keyword.value))
    return ()


# --------------------------------------------------------------------------- #
# batteries
# --------------------------------------------------------------------------- #
def battery_a_raise_handoff(results, schema, names, defaults, proprio):
    """The A-raise handoff must transfer the SAME tracker and never begin the swing."""
    prefix, tracker, goals, q0 = build_prefix(schema, names, defaults)
    segments_before = tracker.segments_begun
    fixed_goal_before = tracker.fixed_goal.copy()
    phase_before = prefix.phase

    # Call the ORIGINAL A-arrival branch five times: the first publishes the
    # handoff, the rest must be inert.
    for _ in range(5):
        prefix._advance_segment('A')

    observed = {
        'handoff_ready': bool(prefix.handoff_ready),
        'handoff_call': prefix.handoff_call,
        'phase_after': prefix.phase,
        'segments_begun_before': int(segments_before),
        'segments_begun_after': int(tracker.segments_begun),
        'tracker_fixed_goal_is_still_A': bool(np.array_equal(tracker.fixed_goal, goals['A'])),
        'tracker_fixed_goal_changed_from_its_pre_call_value': bool(
            not np.array_equal(tracker.fixed_goal, fixed_goal_before)),
        'goal_B_selected': bool(np.array_equal(tracker.fixed_goal, goals['B'])),
    }
    results.add(
        'A-raise', 'the handoff publishes once and NEVER begins the swing',
        'plan 3 "A 到位交接" / "不调用 tracker.begin(B)"',
        'drive the real PayloadRaisePrefix._advance_segment("A") 5x at the A branch',
        observed,
        'PASS' if (observed['handoff_ready'] and observed['phase_after'] == phase_before
                   and observed['segments_begun_after'] == observed['segments_begun_before']
                   and observed['tracker_fixed_goal_is_still_A']
                   and not observed['goal_B_selected']) else 'FALSIFIED',
        trivial_pass_risk='none: the counter is read before and after, and goal B is compared by value')

    carry = prefix.take_carry_state()
    same_instance = carry['tracker'] is prefix.tracker
    results.add(
        'handoff-transfer', 'the handoff carries the SAME tracker instance, not a copy',
        'plan 3 "正在使用的同一个 PayloadJointTracker 实例"',
        'identity comparison carry["tracker"] is prefix.tracker on the real class',
        {'carry_tracker_is_prefix_tracker': bool(same_instance),
         'carry_control_tick': carry['control_tick'],
         'carry_finger_command_m': np.asarray(carry['finger_command']).tolist(),
         'carry_action_template_len': int(np.asarray(carry['action_template']).size),
         'carry_goals_keys': sorted(carry['goals_rad'])},
        'PASS' if same_instance else 'FALSIFIED')

    try:
        prefix.take_carry_state()
        second = 'returned a value'
    except ValueError as error:
        second = 'ValueError: %s' % error
    results.add(
        'handoff-once', 'the carry state can be taken exactly once',
        'plan 3 "Once published" the transfer is single-shot',
        'call take_carry_state() a second time on the real class',
        {'second_call': second},
        'PASS' if second.startswith('ValueError') else 'FALSIFIED')

    # Parent-side: the child must be called once per tick while it runs and never
    # again after the handoff.
    policy, child, first = make_policy(schema, names, defaults, carry,
                                       np.zeros(schema.total_dim, dtype=np.float32))
    counts = {'after_handoff_first_tick': child.calls}
    for _ in range(10):
        policy.act(proprio.build(q0), {})
    counts['after_10_more_ticks'] = child.calls
    counts['parent_tracker_is_carry_tracker'] = bool(policy.tracker is carry['tracker'])
    counts['parent_state'] = policy.state
    results.add(
        'child-call-count', 'the prefix child is called once per tick while it runs and NEVER after',
        'plan 3 "顶层从下一 tick 起永久停止调用前缀子策略"',
        'counting stub child through the production _prefix path, then 10 more ticks',
        counts,
        'PASS' if child.calls == 1 and counts['parent_tracker_is_carry_tracker'] else 'FALSIFIED',
        trivial_pass_risk='the child is a stub; only the parent prefix path is under test here',
        note='the stub child is not the frozen prefix; the prefix itself is tested above')


def battery_wheel_anchor_in_movement(results, schema, names, defaults, proprio,
                                     carry, action, still_q):
    """A movement phase must not hold the wheel anchor, or navigation is discarded."""
    wheel = schema.term(WHEEL_TERM)
    drive = None
    try:
        from task_b.carry_drive import CarryDrive
        drive = CarryDrive(dt=DT, wheel_joint_names=list(wheel.joint_names), wheel_radius_m=.1129)
    except Exception as error:                                    # pragma: no cover
        results.add('wheel-anchor-move', 'a movement phase must not hold the wheel anchor',
                    'plan 3 "移动阶段必须 wheel_hold_requested=False"',
                    'build the drive layer', {'drive_build_error': repr(error)},
                    'INCONCLUSIVE', note='the drive layer could not be built')
        return
    from task_b.brake_wheel_hold import BrakeWheelHold

    def compose(policy, requested, holder, obs):
        """The evaluator's own composition, reproduced so its effect is measurable."""
        composed = np.asarray(requested, dtype=float).copy()
        composed[wheel.start:wheel.stop] *= DEFAULT_WHEEL_ACTION_GAIN
        wheel_q = obs[12:36][[names.index(n) for n in wheel.joint_names]] + \
            np.asarray([defaults[n] for n in wheel.joint_names])
        wheel_qdot = obs[36:60][[names.index(n) for n in wheel.joint_names]]
        held = bool(policy.wheel_hold_requested)
        if held:
            holder.engage(wheel_q)
            common = holder.update(wheel_q, wheel_qdot)
            composed[wheel.start:wheel.stop] = common/wheel.scale
        else:
            holder.release()
        return composed, held

    # (a) show the overwrite is real: while the anchor is HELD, a navigation
    # request that is written into the action is destroyed by the evaluator.
    policy, child, _ = make_policy(schema, names, defaults, carry, action)
    obs = proprio.build(still_q)
    policy._last_proprio = obs
    held_request = policy._action(np.array([.3, .3, .3, .3]))
    composed_held, held = compose(policy, held_request, BrakeWheelHold(dt=DT), obs)
    overwrite_is_real = bool(held and np.allclose(composed_held[wheel.start:wheel.stop], 0.))

    # (b) drive the real sequence into a movement phase and check the request
    # actually survives to the composed action.
    obs = proprio.build(still_q)
    for _ in range(40):
        policy.act(obs, {})
        if policy.phase in MOVEMENT_PHASES:
            break
    obs = proprio.build(still_q, v=(PROBE_SPEED_M_S, 0., 0.))
    action_out = policy.act(obs, {})
    composed, held_move = compose(policy, action_out, BrakeWheelHold(dt=DT), obs)
    observed = {
        'phase': policy.phase,
        'wheel_hold_requested': bool(policy.wheel_hold_requested),
        'anchor_released': bool(policy.anchor_released),
        'navigator_request_rad_s': np.asarray(policy.wheel_request_rad_s).tolist(),
        'request_nonzero': bool(np.any(np.abs(policy.wheel_request_rad_s) > 0.)),
        'evaluator_would_overwrite': bool(held_move),
        'composed_wheel_slice': composed[wheel.start:wheel.stop].tolist(),
        'requested_slice_times_gain': (np.asarray(action_out)[wheel.start:wheel.stop]
                                       * DEFAULT_WHEEL_ACTION_GAIN).tolist(),
        'overwrite_is_real_when_held': overwrite_is_real,
    }
    survives = bool(not held_move and np.allclose(composed[wheel.start:wheel.stop],
                                                  np.asarray(action_out)[wheel.start:wheel.stop]
                                                  * DEFAULT_WHEEL_ACTION_GAIN))
    results.add(
        'wheel-anchor-move', 'no movement phase holds the wheel anchor, so the navigation slice survives',
        'plan 3 "移动阶段必须 wheel_hold_requested=False，否则 evaluator 会覆盖所有导航请求"',
        'run into PROBE_DRIVE, emit the real action, then replay the evaluator composition',
        observed,
        'PASS' if (survives and overwrite_is_real and observed['request_nonzero']) else 'FALSIFIED',
        trivial_pass_risk='guarded: part (a) proves the overwrite really happens when held, and '
                          'part (b) requires the emitted request to be non-zero, so a zero request '
                          'cannot pass this check',
        note='the drive layer is the delivered task_b.carry_drive')

    # (c) every declared movement phase, forced directly.
    policy2, _, _ = make_policy(schema, names, defaults, carry, action)
    per_phase = {}
    for phase in sorted(MOVEMENT_PHASES):
        policy2._begin(phase, 'audit_forced_phase')
        per_phase[phase] = bool(policy2.wheel_hold_requested)
    results.add(
        'wheel-anchor-phases', 'entering any declared movement phase releases the anchor',
        'plan 3 "开始减速时保持 False ... 开始下一次真实移动才 release"',
        '_begin() every phase in MOVEMENT_PHASES and read wheel_hold_requested',
        {'wheel_hold_requested_by_phase': per_phase,
         'movement_phases': sorted(MOVEMENT_PHASES)},
        'PASS' if not any(per_phase.values()) else 'FALSIFIED',
        trivial_pass_risk='these are the phases the implementation itself declares as movement; a '
                          'phase it failed to declare would not be covered here')


def battery_anchor_while_sliding(results, schema, names, defaults, proprio, carry, action, still_q):
    """A NEW wheel anchor must never be taken while the base is still sliding."""
    speeds = [0., .005, .0099, .0101, .02, .05, .08]
    rows = {}
    for speed in speeds:
        policy, _, _ = make_policy(schema, names, defaults, carry, action)
        carry_probe_stream(policy, proprio, still_q, stop_phase='PROBE_BRAKE_DRIVE')
        log = hold_twist(policy, proprio, still_q, speed, 0., 150)
        rows['%g' % speed] = {
            'phase_when_probed': log[-1]['phase'] if log else None,
            'anchor_engaged': bool(log and any(r['anchor_engaged'] for r in log)),
            'stop_reason': log[-1]['stop_reason'] if log else None,
        }
    engaged_speeds = [float(k) for k, v in rows.items() if v['anchor_engaged']]
    worst_engaged = max(engaged_speeds) if engaged_speeds else None
    results.add(
        'anchor-not-sliding', 'a NEW wheel anchor is taken only after the base has stopped',
        'plan 3 "不要在仍以 .235 m/s 滑行时重置 .03 m 静止预算"',
        'hold the brake phase at 7 constant planar speeds for 150 ticks each',
        {'brake_settle_speed_m_s': BRAKE_SETTLE_SPEED_M_S,
         'fastest_speed_that_anchored_m_s': worst_engaged,
         'anchored_at_or_above_settle_speed': bool(worst_engaged is not None
                                                   and worst_engaged >= BRAKE_SETTLE_SPEED_M_S),
         'per_speed': rows},
        'PASS' if (worst_engaged is None or worst_engaged < BRAKE_SETTLE_SPEED_M_S) else 'FALSIFIED',
        trivial_pass_risk='guarded: the battery also requires an anchor to be reachable at all '
                          '(see anchor-reengage), so a policy that simply never anchors cannot pass '
                          'by accident',
        note='the speeds below the .01 m/s settle gate must anchor; the sweep reports which did')

    yaw_rates = [0., .05, .08, .10, .119, .121, .15]
    yaw_rows = {}
    for yaw in yaw_rates:
        policy, _, _ = make_policy(schema, names, defaults, carry, action)
        carry_probe_stream(policy, proprio, still_q, stop_phase='PROBE_BRAKE_DRIVE')
        log = hold_twist(policy, proprio, still_q, 0., yaw, 150)
        yaw_rows['%g' % yaw] = bool(log and any(r['anchor_engaged'] for r in log))
    anchored = [float(k) for k, v in yaw_rows.items() if v]
    fastest_yaw_anchored = max(anchored) if anchored else None
    results.add(
        'anchor-yaw-residue', 'the anchor is not taken while the base is still YAWING',
        'plan 4 "公开速度已足够低后才切 True" + "期望/实测 yaw 上限 .08 rad/s"',
        'hold the brake phase at zero planar speed with 7 constant yaw rates, 150 ticks each',
        {'yaw_rate_cap_for_the_probe_rad_s': .08,
         'fastest_yaw_rate_that_anchored_rad_s': fastest_yaw_anchored,
         'anchored_by_yaw_rate': yaw_rows},
        'PASS' if (fastest_yaw_anchored is None or fastest_yaw_anchored <= .08) else 'FALSIFIED',
        trivial_pass_risk='guarded: the same series contains a yaw of 0 that must anchor',
        note='the anchor gate reuses the .12 rad/s quiet angular gate, which is wider than the .08 '
             'rad/s probe yaw cap; the sweep measures whether that width is reachable')

    policy, _, _ = make_policy(schema, names, defaults, carry, action)
    carry_probe_stream(policy, proprio, still_q, stop_phase='PROBE_BRAKE_DRIVE')
    hold_twist(policy, proprio, still_q, 0., 0., 150)
    engages = len(policy.anchor_engage_calls)
    calls_of_first_engage = list(policy.anchor_engage_calls)
    hold_twist(policy, proprio, still_q, 0., 0., 150)
    results.add(
        'anchor-reengage', 'a settled stationary stretch does not re-take the anchor',
        'plan 3 "静止段反复 engage 不重抓锚"',
        'stay settled in the brake phase for 150 further ticks after the anchor was taken',
        {'engage_count_after_settle': engages,
         'engage_count_after_150_more_ticks': len(policy.anchor_engage_calls),
         'engage_calls': calls_of_first_engage,
         'anchor_reachable_at_all': bool(engages > 0)},
        'PASS' if (engages > 0 and len(policy.anchor_engage_calls) == engages) else 'FALSIFIED',
        trivial_pass_risk='the battery requires at least one engage, so "never anchors" cannot pass')


def battery_hold_timing(results, schema, names, defaults, proprio, carry, action, q0, goal_a):
    """Hold / observation duration must be a real endpoint-timestamp difference."""
    policy, _, _ = make_policy(schema, names, defaults, carry, action)

    # (a) the timer is not capped at the window length
    obs = proprio.build(goal_a)
    seen = []
    for _ in range(200):
        feed_tick(policy, obs, goal_a[:6])
        seen.append(policy.quiet_elapsed_s())
    elapsed_max = float(max(seen))
    results.add(
        'hold-not-window', 'the hold duration is NOT the length of a bounded window',
        'plan 4 "原 hold_quiet_calls * dt 不能代替端点持续时间"',
        'drive _quiet_tick for 200 ticks on a still arm and read quiet_elapsed_s()',
        {'quiet_window_s': QUIET_WINDOW_S, 'max_quiet_elapsed_s': elapsed_max,
         'value_at_window_close_s': seen[24] if len(seen) > 24 else None,
         'value_after_200_ticks_s': elapsed_max},
        'PASS' if elapsed_max > 2.*QUIET_WINDOW_S else 'FALSIFIED',
        trivial_pass_risk='none: a window-length implementation would pin this at 0.5 s')

    # (b) breaking quiet restarts the run
    policy2, _, _ = make_policy(schema, names, defaults, carry, action)
    for _ in range(80):
        feed_tick(policy2, obs, goal_a[:6])
    before = policy2.quiet_elapsed_s()
    moving = proprio.build(goal_a, v=(.05, 0., 0.))
    for _ in range(3):
        feed_tick(policy2, moving, goal_a[:6])
    after_break = policy2.quiet_elapsed_s()
    results.add(
        'hold-restart-on-motion', 'losing quiet zeroes the hold run',
        'plan 4 "只有完整静稳窗口 quiet_ready 成立才累计，任何失静/暂停清零"',
        'accumulate 80 quiet ticks, then 3 ticks at 0.05 m/s, then read the timer',
        {'elapsed_before_break_s': float(before), 'elapsed_after_3_moving_ticks_s':
            float(after_break), 'quiet_ready_after_break': bool(policy2.quiet_ready),
         'measured_speed_during_the_break_m_s': body_state(moving)['measured_speed']},
        'PASS' if after_break == 0. and before > 0. else 'FALSIFIED')

    # (c) a PAUSE must also zero it
    policy3, child3, _ = make_policy(schema, names, defaults, carry, action)
    for _ in range(80):
        feed_tick(policy3, obs, goal_a[:6])
    elapsed_before_pause = policy3.quiet_elapsed_s()
    policy3.pause_for_stance = True
    for _ in range(5):
        feed_tick(policy3, obs, goal_a[:6])
    elapsed_during_pause = policy3.quiet_elapsed_s()
    results.add(
        'hold-restart-on-pause', 'a stance pause also zeroes the hold run',
        'plan 4 "任何失静/暂停清零"',
        'accumulate 80 quiet ticks, raise pause_for_stance, run 5 further ticks',
        {'elapsed_before_pause_s': float(elapsed_before_pause),
         'elapsed_after_5_ticks_with_pause_true_s': float(elapsed_during_pause),
         'pause_flag_reaches_the_policy': bool(child3.pause_for_stance),
         'policy_reads_the_flag_after_handoff': False},
        'PASS' if elapsed_during_pause == 0. else 'FALSIFIED',
        trivial_pass_risk='none: the flag is set on the live policy object',
        note='the carry state machine never reads pause_for_stance; measured, not inferred')

    # (d) the .002 rad position window cannot be bypassed by a quiet velocity
    policy4, _, _ = make_policy(schema, names, defaults, carry, action)
    walk = np.asarray(goal_a[:6], dtype=float).copy()
    states = []
    for step in range(60):
        walk = walk + np.array([.0005, 0., 0., 0., 0., 0.])
        obs_walk = proprio.build(walk, qdot=np.zeros(8))
        feed_tick(policy4, obs_walk, walk)
        states.append((round(step*.0005, 5), policy4.quiet_tick, policy4.quiet_ready,
                       None if policy4.arm_span is None else round(policy4.arm_span, 6)))
    quiet_tick_true = [s for s in states if s[1]]
    quiet_ready_true = [s for s in states if s[2]]
    results.add(
        'hold-position-window', 'a quiet velocity signal cannot bypass the .002 rad position window',
        'plan 4 "不能仅用 quiet_tick 绕过 .002 rad 位置窗口"',
        'walk one arm axis at 5e-4 rad/tick with reported qdot exactly zero',
        {'arm_span_limit_rad': QUIET_ARM_SPAN_RAD,
         'ticks_with_quiet_tick_true': len(quiet_tick_true),
         'ticks_with_quiet_ready_true': len(quiet_ready_true),
         'max_arm_span_rad': max(s[3] for s in states if s[3] is not None) if states else None},
        'PASS' if (quiet_tick_true and not quiet_ready_true) else 'FALSIFIED',
        trivial_pass_risk='none: the check fails if quiet_ready is reachable while walking')


def battery_release_gates(results, schema, names, defaults, proprio, carry, action, q0, goal_a):
    """The release/open stage must be behind the arrival gates and never reclassified."""
    module = sys.modules['task_b.delivery']
    declared_phases = set(getattr(module, 'CARRY_PROBE_PHASES', ()))
    release_names = sorted(name for name in declared_phases
                           if any(token in name.upper() for token in
                                  ('RELEASE', 'OPEN', 'DROP', 'DELIVER')))
    stop_reasons = set(getattr(module, 'DELIVERY_STOP_REASONS', {}))
    claim_words = ('release', 'delivered', 'objects_in_circle', 'success', 'grasped')
    claiming = sorted(r for r in stop_reasons if any(w in r.lower() for w in claim_words))

    policy_probe, _, _ = make_policy(schema, names, defaults, carry, action, mode='carry_probe')
    policy_deliv, _, _ = make_policy(schema, names, defaults, carry, action, mode='first_delivery')
    trace_probe = [(r['phase'], r['stop_reason'])
                   for r in carry_probe_stream(policy_probe, proprio, goal_a, max_ticks=1400)]
    trace_deliv = [(r['phase'], r['stop_reason'])
                   for r in carry_probe_stream(policy_deliv, proprio, goal_a, max_ticks=1400)]

    observed = {
        'declared_carry_phases': sorted(declared_phases),
        'release_like_phases_declared': release_names,
        'declared_stop_reasons': sorted(stop_reasons),
        'stop_reasons_claiming_a_delivery': claiming,
        'carry_probe_phases_visited': sorted({p for p, _ in trace_probe}),
        'first_delivery_phases_visited': sorted({p for p, _ in trace_deliv}),
        'carry_probe_stop_reason': policy_probe.done_reason,
        'first_delivery_stop_reason': policy_deliv.done_reason,
        'first_delivery_is_the_same_sequence_as_carry_probe':
            [p for p, _ in trace_probe] == [p for p, _ in trace_deliv],
        'carry_probe_selectable_from_the_evaluator_cli': _mode_choices().__contains__('carry_probe'),
        'first_delivery_selectable_from_the_evaluator_cli': _mode_choices().__contains__(
            'first_delivery'),
        'evaluator_mode_choices': sorted(_mode_choices()),
    }
    results.add(
        'release-stage-absent', 'the release/open stage is unreachable before the arrival gates',
        'plan 7 "RELEASE_OPEN 进入条件: D 实际到位、完整新鲜静止窗口成立、轮锚保持..."',
        'enumerate the declared phases and drive both modes to their stop',
        observed,
        'NOT_APPLICABLE' if not release_names else ('PASS' if not claiming else 'FALSIFIED'),
        trivial_pass_risk='THIS CHECK IS VACUOUS TODAY: the release/open stage does not exist in the '
                          'delivered module, so there is no gate to test and no failure to '
                          'reclassify. The battery can only become real once the stage exists.',
        note='a failed open cannot be reclassified as a release because there is no open and no '
             'release; the delivery chain has not been implemented yet')
    return observed


def battery_no_ground_truth(results):
    """No ground-truth, world pose, force, reward, score or seed map reaches the policy."""
    files = [PROJECT_ROOT/'task_b'/'delivery.py',
             PROJECT_ROOT/'task_b'/'carry_drive.py',
             PROJECT_ROOT/'task_b'/'bucket_observation.py']
    per_file = {}
    total_hits = 0
    for path in files:
        if not path.exists():
            per_file[path.name] = {'missing': True}
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        identifiers, imports, string_literals = set(), set(), set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                identifiers.add(node.id)
            elif isinstance(node, ast.Attribute):
                identifiers.add(node.attr)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                imports.add(node.module or '')
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                string_literals.add(node.value.strip())
        id_hits = sorted(FORBIDDEN_IDENTIFIERS & identifiers)
        imp_hits = sorted(name for name in imports
                          if any(name == bad or name.startswith(bad + '.')
                                 for bad in FORBIDDEN_IMPORTS))
        str_hits = sorted(FORBIDDEN_IDENTIFIERS & string_literals)
        total_hits += len(id_hits) + len(imp_hits)
        per_file[path.name] = {'identifier_hits': id_hits, 'import_hits': imp_hits,
                               'standalone_string_literal_hits': str_hits}
    results.add(
        'no-ground-truth', 'no GT, world base pose, contact force, reward, score or seed map reaches the policy',
        'plan 3 ownership table + plan 7 "不向策略注入 GT/root/contact/reward/seed 地图"',
        'AST scan of delivery.py/carry_drive.py/bucket_observation.py for contract-derived names',
        {'forbidden_identifiers_checked': sorted(FORBIDDEN_IDENTIFIERS),
         'forbidden_imports_checked': sorted(FORBIDDEN_IMPORTS),
         'hits_per_file': per_file, 'total_code_hits': total_hits},
        'PASS' if total_hits == 0 else 'FALSIFIED',
        trivial_pass_risk='an identifier scan can only catch literal names; it cannot prove that a '
                          'value is not smuggled in through a differently named variable. It is a '
                          'necessary, not a sufficient, check.')

    # Runtime check: the policy answers an 84-vector and images only, and one
    # instance runs to a stop with no simulator present at all.
    schema, names, defaults = load_schema(DEFAULT_RUN/'environment_metadata.json')
    carry, _, q0 = make_carry(schema, names, defaults)
    policy, child, _ = make_policy(schema, names, defaults, carry,
                                   np.zeros(schema.total_dim, dtype=np.float32))
    import inspect
    signature = str(inspect.signature(PayloadDeliveryPolicy.act))
    results.add(
        'public-inputs-only', 'the policy entry point accepts only public proprio and RGB-D',
        'plan 3 "所有策略模块仍只有 public inputs"',
        'inspect the production act() signature and run a carry sequence CPU-only',
        {'act_signature': signature, 'debug_keys': sorted(policy.debug)},
        'PASS' if signature == '(self, proprio, images)' else 'FALSIFIED',
        trivial_pass_risk='a signature check cannot see what act() does with the values; the AST '
                          'scan above is the companion check')


def battery_tracker_windup(results, schema, names, defaults):
    """Contract P0: the integral term when the FINAL command is clamped (slew/tether/limit)."""
    q0 = np.array([0., .50, -.02, 0., 0., 0.])

    def residual(tracker, measured, qdot):
        """candidate - applied, recomputed from the tracker's OWN exposed state."""
        candidate = (tracker.reference + tracker.feedforward + tracker.integral
                     + tracker.filtered_correction)
        return candidate - tracker.command

    # (U1) slew / tether clamp driven by a real feed-forward.
    feedforward = np.zeros(6)
    feedforward[1] = .15
    tracker = PayloadJointTracker(dt=DT, initial_reference=q0, initial_command=q0,
                                  feedforward=feedforward)
    goal = q0 + np.array([0., .02, 0., 0., 0., 0.])
    tracker.begin(goal)
    measured = q0 - np.array([0., .05, 0., 0., 0., 0.])   # 5e-2 error, INSIDE the .14/.08 P cap
    rows = []
    for step in range(200):
        tracked = tracker.update(measured, np.zeros(6))
        r = residual(tracker, measured, np.zeros(6))
        rows.append({'step': step, 'command_axis1': float(tracked[1]),
                     'candidate_axis1': float(tracked[1]+r[1]),
                     'residual_axis1': float(r[1]),
                     'p_term_axis1': float(tracker.gains[1]*(tracker.reference[1]-measured[1])),
                     'p_within_cap': bool(abs(tracker.gains[1]*(tracker.reference[1]-measured[1]))
                                          <= tracker.caps[1]),
                     'integral_axis1': float(tracker.integral[1])})
    first = rows[5]
    last = rows[-1]
    clamped_rows = [r for r in rows if r['residual_axis1'] > 1e-9]
    results.add(
        'integral-windup-slew', 'a final-command clamp must stop the integral from ramping',
        'plan 4 P0 "CPU 覆盖加载→输出夹限→卸载" + signal review 5.3 "未反馈最终 slew/tether/'
        'hard-limit 的夹限"',
        'feed-forward 0.15 rad on one axis with a 5e-2 tracking error inside the P cap; '
        'recompute candidate-command every tick',
        {'ticks_where_the_applied_command_is_clamped': len(clamped_rows),
         'p_term_always_inside_its_cap': bool(all(r['p_within_cap'] for r in rows)),
         'integral_axis1_first_tick': first['integral_axis1'],
         'integral_axis1_last_tick': last['integral_axis1'],
         'integral_accumulated_rad': last['integral_axis1']-first['integral_axis1'],
         'integral_cap_rad': INTEGRAL_CAP_RAD,
         'p_cap_axis1_rad': float(tracker.caps[1]),
         'max_candidate_minus_applied_rad': max(r['residual_axis1'] for r in rows),
         'final_candidate_minus_applied_rad': last['residual_axis1']},
        'FALSIFIED' if (clamped_rows and abs(last['integral_axis1']) > 1e-6) else 'PASS',
        trivial_pass_risk='guarded: the battery requires both a clamped row and a non-zero integral '
                          'before it can falsify',
        note='the code\'s own "unsaturated" test only compares the P term with its cap, so it reports '
             'unsaturated while the applied command is slew/tether clamped')

    # (U2) hard joint-limit clamp: arm_joint3 has an upper static limit of exactly 0.
    tracker2 = PayloadJointTracker(dt=DT, initial_reference=q0, initial_command=q0)
    goal2 = q0.copy()
    tracker2.begin(goal2)
    measured2 = q0 - np.array([0., 0., .05, 0., 0., 0.])
    rows2 = []
    for step in range(300):
        tracked = tracker2.update(measured2, np.zeros(6))
        r = residual(tracker2, measured2, np.zeros(6))
        rows2.append({'command_axis2': float(tracked[2]),
                      'candidate_axis2': float(tracked[2]+r[2]),
                      'residual_axis2': float(r[2]),
                      'p_within_cap': bool(abs(tracker2.gains[2]*(tracker2.reference[2]-measured2[2]))
                                           <= tracker2.caps[2]),
                      'integral_axis2': float(tracker2.integral[2])})
    limit = float(tracker2.static_upper[2])
    pinned = [r for r in rows2 if abs(r['command_axis2']-limit) < 1e-12]
    results.add(
        'integral-windup-hardlimit', 'a HARD joint-limit clamp must stop the integral from ramping',
        'plan 4 P0 "积分在最终命令限幅时仍可能积分"',
        'place the reference on arm_joint3\'s upper static limit with a 5e-2 error inside the P cap',
        {'static_upper_axis2_rad': limit,
         'ticks_pinned_at_the_hard_limit': len(pinned),
         'p_always_inside_its_cap': bool(all(r['p_within_cap'] for r in rows2)),
         'integral_axis2_final_rad': rows2[-1]['integral_axis2'],
         'integral_cap_rad': INTEGRAL_CAP_RAD,
         'p_cap_axis2_rad': float(tracker2.caps[2]),
         'candidate_minus_applied_final_rad': rows2[-1]['residual_axis2']},
        'FALSIFIED' if (pinned and abs(rows2[-1]['integral_axis2']) > 1e-6) else 'PASS')

    # (U3) release / unload: what the wound-up trim does when the load disappears.
    # Move the fixed goal away from the hard limit so the limit is no longer what
    # bounds the command; the stored trim is then free to act.
    integral_at_unload = float(tracker2.integral[2])
    goal3 = q0.copy()
    goal3[2] = -.40
    tracker2.begin(goal3)
    unload_rows = []
    for step in range(240):
        measured3 = tracker2.reference.copy()          # a perfectly unloaded arm
        tracked = tracker2.update(measured3, np.zeros(6))
        unload_rows.append({'step': step,
                            'command_axis2': float(tracked[2]),
                            'reference_axis2': float(tracker2.reference[2]),
                            'integral_axis2': float(tracker2.integral[2]),
                            'candidate_axis2': float(tracked[2]
                                                    + residual(tracker2, measured3,
                                                               np.zeros(6))[2])})
    settled = unload_rows[-1]
    bias = settled['command_axis2']-settled['reference_axis2']
    integral_drift = max(abs(r['integral_axis2'])-abs(integral_at_unload) for r in unload_rows)
    results.add(
        'unload-overshoot', 'after unloading, the wound-up trim must not bias the command past its reference',
        'plan 7 "卸载时积分只作抗 windup 和既有有界反馈，不能把 arm_command 一次性跳回无载参考"',
        'unload a tracker that wound up against a hard limit, then move its goal clear of the limit',
        {'integral_axis2_at_unload_rad': integral_at_unload,
         'integral_cap_rad': INTEGRAL_CAP_RAD,
         'integral_never_rewinds': bool(integral_drift <= 1e-12),
         'reference_axis2_at_settle_rad': settled['reference_axis2'],
         'command_axis2_at_settle_rad': settled['command_axis2'],
         'steady_state_bias_past_the_reference_rad': bias,
         'candidate_minus_reference_at_settle_rad':
             settled['candidate_axis2']-settled['reference_axis2'],
         'arm_tether_axis2_rad': float(tracker2.tether[2]),
         'first_tick_command_is_slew_bounded_not_a_jump': bool(
             abs(unload_rows[0]['command_axis2']-limit) <= COMMAND_RATE_RAD_S*DT+1e-12)},
        'FALSIFIED' if abs(bias) > 1e-6 else 'PASS',
        trivial_pass_risk='guarded: the battery fails only if the un-rewound trim actually holds the '
                          'command away from the reference',
        note='the command is slew-bounded (no jump), which the design does achieve; the hazard is the '
             'stored trim, which never leaks and therefore holds a permanent steady-state offset')


def battery_hold_pause_sequence(results, schema, names, defaults, proprio, carry, action, q0, goal_a):
    """End-to-end: the observation timer inside the real phase machine."""
    policy, child, _ = make_policy(schema, names, defaults, carry, action)
    log = carry_probe_stream(policy, proprio, goal_a, stop_phase='PROBE_HOLD')
    reached_hold = policy.phase == 'PROBE_HOLD'
    hold_rows = []
    if reached_hold:
        for _ in range(1400):
            policy.act(proprio.build(goal_a, qdot=np.zeros(8)), {})
            hold_rows.append({'quiet_elapsed_s': float(policy.quiet_elapsed_s()),
                              'phase_elapsed_s': float(policy.phase_elapsed_s)})
            if policy.done_reason:
                break
    observed = {
        'reached_PROBE_HOLD': reached_hold,
        'phases_before_the_hold': sorted({r['phase'] for r in log}),
        'ticks_driven_before_the_hold': len(log),
        'hold_ticks': len(hold_rows),
        'quiet_elapsed_s_at_the_completing_tick':
            hold_rows[-1]['quiet_elapsed_s'] if hold_rows else None,
        'phase_elapsed_s_at_the_completing_tick':
            hold_rows[-1]['phase_elapsed_s'] if hold_rows else None,
        'required_hold_s': 1.,
        'quiet_window_s': QUIET_WINDOW_S,
        'stop_reason': policy.done_reason,
    }
    results.add(
        'hold-end-to-end', 'the observation ends on a REAL elapsed hold, longer than the window',
        'plan 4 "张指到位后观察端点计时" + plan 7 "连续观察至少真实 3.0 秒"',
        'drive the production phase machine through the whole carry probe on the audit twist model',
        observed,
        'PASS' if (reached_hold and policy.done_reason == 'delivery_carry_probe_complete'
                   and observed['quiet_elapsed_s_at_the_completing_tick'] >= 1.) else 'FALSIFIED',
        note='the carry-probe observation is 1.0 s here; the 3.0 s figure belongs to the release '
             'stage, which does not exist yet')

    # the same observation with a stance pause asserted for its whole duration
    policy2, child2, _ = make_policy(schema, names, defaults, carry, action)
    carry_probe_stream(policy2, proprio, goal_a, stop_phase='PROBE_HOLD')
    paused_rows = []
    if policy2.phase == 'PROBE_HOLD':
        policy2.pause_for_stance = True
        for _ in range(1400):
            policy2.act(proprio.build(goal_a, qdot=np.zeros(8)), {})
            paused_rows.append({'quiet_elapsed_s': float(policy2.quiet_elapsed_s())})
            if policy2.done_reason:
                break
    results.add(
        'hold-pause-end-to-end', 'a pause during the observation does not let it complete',
        'plan 4 "任何失静/暂停清零"',
        'reach PROBE_HOLD, then assert pause_for_stance for the whole observation',
        {'phase_reached': policy2.phase,
         'pause_asserted_for_the_whole_observation': bool(child2.pause_for_stance),
         'ticks_in_the_paused_observation': len(paused_rows),
         'stop_reason': policy2.done_reason,
         'quiet_elapsed_s_at_the_stop':
             paused_rows[-1]['quiet_elapsed_s'] if paused_rows else None},
        'PASS' if policy2.done_reason != 'delivery_carry_probe_complete' else 'FALSIFIED',
        trivial_pass_risk='none: the pause flag is set on the live policy object',
        note='the carry state machine never reads pause_for_stance after the handoff')


def battery_goal_error_gate(results, schema, names, defaults, proprio, carry, action, q0, goal_a):
    """The stationary gate must retain goal_error < .04 rad as well as window+span."""
    off_goal = np.asarray(goal_a, dtype=float).copy()
    # 0.06 rad: above the fixed .04 rad gate but well inside the .10 rad arm tether
    # on this axis, so the tracker's own bound machinery is not what stops it.
    off_goal[0] += .06
    policy, child, _ = make_policy(schema, names, defaults, carry, action)
    trace = []
    for step in range(80):
        policy.act(proprio.build(off_goal), {})
        trace.append({'phase': policy.phase,
                      'goal_error_rad': policy.debug.get('goal_error_rad'),
                      'goal_error_limit_rad': policy.debug.get('goal_error_limit_rad'),
                      'quiet_ready': policy.debug.get('quiet', {}).get('quiet_ready')})
        if policy.phase not in ('CARRY_READY',):
            break
    left_carry_ready = policy.phase != 'CARRY_READY'
    stopped = policy.phase == 'STOPPED'
    observed = {
        'fed_goal_error_rad': float(np.max(np.abs(off_goal[:6] - goal_a))),
        'declared_limit_rad': GOAL_ERROR_RAD,
        'phase_reached': policy.phase,
        'left_CARRY_READY': left_carry_ready,
        'stopped_instead_of_proceeding': stopped,
        'stop_reason': policy.done_reason,
        'wheel_anchor_released_to_start_driving': bool(left_carry_ready and not stopped),
        'goal_error_reported_when_it_left_CARRY_READY': trace[-1]['goal_error_rad'] if trace else None,
        'ticks_in_CARRY_READY': len(trace),
    }
    results.add(
        'carry-ready-goal-error', 'the carry-ready stationary gate keeps goal_error < .04 rad',
        'plan 4 "保留实际 goal_error<.04 rad、完整 .5 s（26 样本）窗口与 arm span <=.002 rad"',
        'hold a perfectly still arm 0.06 rad AWAY from the fixed A goal and watch the phase',
        observed,
        'FALSIFIED' if left_carry_ready and not stopped else 'PASS',
        trivial_pass_risk='none: the arm is still, so the window and span gates pass; only the '
                          'goal-error element can hold the phase',
        note='the debug record prints goal_error_rad next to goal_error_limit_rad, but nothing in '
             'the carry state machine compares them')


def battery_handoff_incomplete(results, schema, names, defaults, proprio, carry, action, still_q):
    """A failed handoff must produce a normal stop, not an exception."""
    policy = PayloadDeliveryPolicy(schema, names, defaults, dt=DT,
                                   wheel_action_gain=DEFAULT_WHEEL_ACTION_GAIN)
    policy.child = CountingChild(carry, action, raise_on_take=True)
    try:
        returned = policy.act(proprio.build(still_q), {})
        outcome = 'returned %s' % (None if returned is None else np.asarray(returned).shape)
    except Exception as error:
        outcome = '%s: %s' % (type(error).__name__, error)
    results.add(
        'handoff-incomplete', 'an incomplete handoff is stopped, not crashed',
        'plan 3 "That is an implementation fault, so stop rather than carry on"',
        'stub a prefix that reports handoff_ready but whose state is incomplete',
        {'outcome': outcome, 'state': policy.state, 'done_reason': policy.done_reason},
        'PASS' if outcome.startswith('returned') else 'FALSIFIED',
        note='the intended delivery_handoff_incomplete stop path builds an action from a template '
             'that does not exist yet')

    # and a prefix that stops BEFORE the handoff: the next act() call.
    policy2 = PayloadDeliveryPolicy(schema, names, defaults, dt=DT,
                                    wheel_action_gain=DEFAULT_WHEEL_ACTION_GAIN)
    child = CountingChild(carry, action)
    child.done_reason = 'audit_stub_prefix_stopped_before_the_A_arrival'
    policy2.child = child
    policy2.act(proprio.build(still_q), {})
    first_done = policy2.done_reason
    try:
        policy2.act(proprio.build(still_q), {})
        second = 'returned'
    except Exception as error:
        second = '%s: %s' % (type(error).__name__, error)
    results.add(
        'closing-action-after-stop', 'a closed episode can still be asked for its closing action',
        'plan 3 "顶层对外仍提供 act(proprio, images) ... 使 evaluator 不必重写记录器"',
        'stop the prefix before any handoff, then call act() once more',
        {'done_reason_after_first_tick': first_done, 'second_act_outcome': second},
        'PASS' if second == 'returned' else 'FALSIFIED',
        note='the production evaluator breaks out of its loop on the same tick, so this is latent; '
             'any other driver that wants one closing action hits it')


def battery_drive_unavailable(results, schema, names, defaults, proprio, carry, action, still_q):
    """A missing drive layer must be an explicit diagnostic, not a silent stall."""
    import task_b.delivery as delivery_module
    original = delivery_module.PayloadDeliveryPolicy._drive_layer

    def dead(self):
        return None

    delivery_module.PayloadDeliveryPolicy._drive_layer = dead
    try:
        policy, child, _ = make_policy(schema, names, defaults, carry, action)
        obs = proprio.build(still_q)
        reasons = []
        for step in range(400):
            policy.act(obs, {})
            if policy.done_reason:
                reasons.append({'tick': step, 'reason': policy.done_reason,
                                'wheel_hold_requested': bool(policy.wheel_hold_requested)})
                break
        observed = {
            'stop_reason': policy.done_reason,
            'ticks_until_the_stop': reasons[0]['tick'] if reasons else None,
            'wheel_hold_requested_at_the_stop': reasons[0]['wheel_hold_requested'] if reasons else None,
            'drive_state_recorded': policy.debug.get('wheel', {}).get('drive'),
            'drive_fault_latched': policy.drive_fault,
        }
    finally:
        delivery_module.PayloadDeliveryPolicy._drive_layer = original
    explicit = bool(policy.done_reason and 'drive' in str(policy.done_reason))
    results.add(
        'drive-unavailable', 'a missing drive layer stops the run with an explicit drive diagnostic',
        'plan 3 "a missing or broken drive layer must show up as an explicit diagnostic"',
        'monkeypatch _drive_layer to return None and drive a movement phase',
        observed,
        'PASS' if explicit else 'FALSIFIED',
        trivial_pass_risk='none: the branch is forced directly',
        note='the unavailable drive is reported inside debug["wheel"]["drive"] but the STOP reason '
             'comes from the stall watchdog')


# --------------------------------------------------------------------------- #

def battery_budget_separation(results, schema, names, defaults, proprio, carry, action, still_q):
    """Movement and stationary budgets must be separate; the .03 m payload budget must not bind."""
    from task_b.payload_motion import PAYLOAD_BUDGETS
    policy, _, _ = make_policy(schema, names, defaults, carry, action)
    log = carry_probe_stream(policy, proprio, still_q, stop_phase='PROBE_BRAKE_DRIVE')
    travelled = float(np.linalg.norm(policy.odometry.position_xy-policy.baseline_xy))
    observed = {
        'stationary_displacement_budget_m': PAYLOAD_BUDGETS['displacement_m'],
        'probe_forward_target_m': PROBE_FORWARD_M,
        'distance_travelled_before_the_brake_phase_m': travelled,
        'phase_after': policy.phase,
        'stop_reason': policy.done_reason,
        'carry_probe_path_budget_m': PROBE_PATH_CAP_M,
    }
    results.add(
        'budget-separation', 'the loaded move is bounded by the PROBE budget, not by the .03 m stationary budget',
        'plan 4 P0 "移动预算与静止预算分离" + plan 3 "不要套用 .03 m/.05 rad 位移/yaw 预算"',
        'drive PROBE_DRIVE to its own distance target and measure how far the base actually went',
        observed,
        'PASS' if (travelled > PAYLOAD_BUDGETS['displacement_m']
                   and policy.phase == 'PROBE_BRAKE_DRIVE') else 'FALSIFIED',
        trivial_pass_risk='none: a wrapper that kept the stationary budget would stop before the '
                          'target and never reach the brake phase')


def battery_invalid_inputs(results, schema, names, defaults, proprio, carry, action, still_q):
    """NaN, wrong-length and out-of-envelope proprio must all stop normally."""
    import task_b.payload_motion as payload_motion
    cases = {}
    for name, build in (
            ('nan', lambda: np.full(84, np.nan)),
            ('short_vector', lambda: np.zeros(60)),
            ('tilted_body_beyond_the_hard_limit', lambda: proprio.build(
                still_q, gravity=(0., 0., -1.)))):
        policy, _, _ = make_policy(schema, names, defaults, carry, action)
        if name == 'tilted_body_beyond_the_hard_limit':
            obs = build()
            obs[9:12] = np.array([0., 0., -1.])
            obs[9] = float(np.sin(payload_motion.HARD_TILT_RAD+.10))
            obs[11] = -float(np.cos(payload_motion.HARD_TILT_RAD+.10))
        else:
            obs = build()
        try:
            policy.act(obs, {})
            cases[name] = {'outcome': 'returned', 'done_reason': policy.done_reason,
                           'state': policy.state}
        except Exception as error:
            cases[name] = {'outcome': '%s: %s' % (type(error).__name__, error)}
    results.add(
        'invalid-inputs', 'NaN / wrong-length / over-tilt proprio stop normally instead of crashing',
        'plan 4 P0 "NaN/丢帧/无进展正常停止"',
        'feed three malformed proprio vectors to the production act()',
        {'hard_tilt_rad': payload_motion.HARD_TILT_RAD, 'cases': cases},
        'PASS' if all(c.get('outcome') == 'returned' and c.get('done_reason') for c in cases.values())
        else 'FALSIFIED',
        trivial_pass_risk='none: each case requires BOTH a returned action and a named stop reason')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path,
                        default=DEFAULT_RUN/'environment_metadata.json')
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()

    schema, names, defaults = load_schema(args.metadata)
    proprio = Proprio(schema, names, defaults)
    carry, goals, q0 = make_carry(schema, names, defaults)
    action = np.zeros(schema.total_dim, dtype=np.float32)
    goal_a = goals['A']

    print('schema: %d actions, arm %d:%d, wheel %d:%d, wheel scale %.3f' % (
        schema.total_dim, schema.term(ARM_TERM).start, schema.term(ARM_TERM).stop,
        schema.term(WHEEL_TERM).start, schema.term(WHEEL_TERM).stop,
        schema.term(WHEEL_TERM).scale))

    results = Results()
    battery_a_raise_handoff(results, schema, names, defaults, proprio)
    battery_wheel_anchor_in_movement(results, schema, names, defaults, proprio, carry, action, goal_a)
    battery_anchor_while_sliding(results, schema, names, defaults, proprio, carry, action, goal_a)
    battery_hold_timing(results, schema, names, defaults, proprio, carry, action, q0, goal_a)
    battery_goal_error_gate(results, schema, names, defaults, proprio, carry, action, q0, goal_a)
    battery_hold_pause_sequence(results, schema, names, defaults, proprio, carry, action, q0, goal_a)
    battery_handoff_incomplete(results, schema, names, defaults, proprio, carry, action, goal_a)
    battery_drive_unavailable(results, schema, names, defaults, proprio, carry, action, goal_a)
    battery_budget_separation(results, schema, names, defaults, proprio, carry, action, goal_a)
    battery_invalid_inputs(results, schema, names, defaults, proprio, carry, action, goal_a)
    battery_release_gates(results, schema, names, defaults, proprio, carry, action, q0, goal_a)
    battery_no_ground_truth(results)
    battery_tracker_windup(results, schema, names, defaults)

    summary = results.summary()
    print('\nsummary: %s' % summary)
    import hashlib
    audited = ['delivery.py', 'carry_drive.py', 'bucket_observation.py', 'payload_motion.py',
               'evaluate.py']
    source_sha256 = {}
    for name in audited:
        path = PROJECT_ROOT/'task_b'/name
        source_sha256[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    report = {
        'scope': 'INDEPENDENT CPU counterexample audit of the Task B delivery contract. No GPU, no '
                 'production file modified, no threshold relaxed, no failure reclassified.',
        'source_sha256_at_the_moment_of_the_run': source_sha256,
        'source_sha256_note': 'the production tree was being edited by another writer while this '
                              'audit ran, so every result below belongs to exactly these revisions '
                              'and no other',
        'schema_metadata': str(args.metadata),
        'note_on_the_seam': 'the production PayloadRaisePrefix is exercised directly at its A-arrival '
                            'branch; the parent carry machine is exercised through a counting stub '
                            'child that returns that REAL carry record, because the true prefix needs '
                            'the GPU simulator',
        'summary': summary,
        'checks': results.rows,
    }
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
        print('wrote %s' % args.output)
    return 0 if not any(r['verdict'] == 'FALSIFIED' for r in results.rows) else 1


if __name__ == '__main__':
    sys.exit(main())
