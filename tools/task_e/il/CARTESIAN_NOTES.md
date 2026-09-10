# Task E Cartesian-goal residual servo IL

Companion experiment to `README.md` (per-joint servo BC). Same demonstrations,
different learning problem. English notes; commands run from the repository
root inside the existing Isaac Lab environment.

## Why this variant exists

In the per-joint BC candidate the label is exactly `clip(feature_0, -1, 1)`:
the network input already contains `(target_j - q_j)/limit`, and the teacher
action is that value clipped. A network can reach near-zero validation error by
learning the identity, so the offline error says almost nothing about control.

Here the joint-space waypoint never reaches the network:

1. the IK path still produces a joint waypoint (hand-coded, unchanged);
2. `fk(waypoint)` converts it into a Cartesian gripper_base goal (position and
   rotation) — the only form the network sees;
3. `differential_ik` produces an analytic damped-least-squares joint step
   toward that goal, rate-limited by the same `limit` the teacher uses and
   clipped to `JOINT_LOWER/JOINT_UPPER`;
4. the network predicts a **bounded six-joint residual** added to that step;
5. physical guards clamp the total step to `±limit` and the result to the
   joint limits.

### Learned vs analytic (be precise when reporting)

Analytic / hand-coded: RGBD perception, IK path planning, the FK Cartesian
goal, the DLS step itself, waypoint acceptance and settling, the descent
contact stop, grasp/release FSM, gripper, homing, holding, both guards.

Learned: only `residual = tanh(MLP(features)) * residual_scale`, added to the
DLS step. What the residual can express is the mismatch between "move the
gripper pose toward the goal pose under DLS" and "move the joints toward the
waypoint at the teacher's rate": damping and orientation-weight effects,
behaviour near singularities and joint stops, and the phase/object dependent
conservatism the teacher applies through `motion_settings`.

Honest caveat: for a 6-DOF non-redundant arm the FK pose is locally invertible,
so the Cartesian goal is not information-theoretically less than the waypoint.
The claim is representational, not informational — the student must map a pose
error to joint motion instead of copying a joint error it was handed, and the
analytic DLS already does most of that mapping. The learned part is genuinely a
small correction, and the report always prints the DLS-only baseline next to
the learned one so the residual's contribution cannot be hidden.

## Files

| File | Role |
| --- | --- |
| `tools/task_e/il/cartesian.py` | Features, FK goal, DLS step, guards, residual scale, model |
| `tools/task_e/il/train_cartesian_bc.py` | CPU training + report |
| `tools/task_e/il/cartesian_policy.py` | `CartesianResidualSolution` deployment class |
| `solution_task_e_il_cartesian.py` | Evaluator entry point |

Data comes from the existing collector (`collect.py`, unmodified). Each stored
pre-action frame already has `q, qdot, target, delta, limit, phase, object_id`,
so features and residual labels are derived offline; no new collection is
needed. `derive_sample` is the same code path deployment uses, so training rows
and runtime rows cannot drift apart. Scaling is fixed physical constants
(metres, radians, `limit`); nothing is normalized per episode.

## CPU checks to run before training

These are property checks on the implementation, not restatements of it.

```bash
PYTHONNOUSERSITE=1 "$ATEC_PYTHON" - <<'PY'
import numpy as np, torch
from task_e_geometry import DEFAULT_JOINT_POS, JOINT_LOWER, JOINT_UPPER
from tools.task_e.il.cartesian import (analytic_step, cartesian_features,
    command_from_residual, goal_from_joint_target, make_cartesian_model)
q = DEFAULT_JOINT_POS.copy()
target = q[:6] + np.array([.05, -.04, .03, .02, -.06, .05])
position, rotation = goal_from_joint_target(target)
analytic = analytic_step(q, position, rotation, .085)
features = cartesian_features(q, np.zeros(6), position, rotation, analytic, .085, 'DESCEND', 2)
model = make_cartesian_model().eval()
with torch.inference_mode():
    output = model(torch.from_numpy(features)).numpy().astype(float)
print('zero_init_output_max', np.abs(output).max())                 # expect exactly 0.0
command, _, _ = command_from_residual(q, analytic, output, .085)
print('untrained_equals_dls', np.abs(command-(q[:6]+analytic)).max())  # expect 0.0
huge, rate, joint = command_from_residual(q, analytic, np.full(6, 5.), .085)
print('rate_guard', np.abs(huge-q[:6]).max() <= .085+1e-12, rate, joint,
      bool(np.all(huge >= JOINT_LOWER-1e-12) and np.all(huge <= JOINT_UPPER+1e-12)))
torch.nn.init.constant_(model[4].bias, .8)                          # stand-in for training
with torch.inference_mode():
    trained = model(torch.from_numpy(features)).numpy().astype(float)*np.full(6, .01)
moved, _, _ = command_from_residual(q, analytic, trained, .085)
print('residual_changes_command', float(np.abs(moved-command).max()))  # expect > 0
# Does the analytic step alone track the waypoint? This is the baseline the
# residual has to improve on; a large remaining joint error means real work.
walk = q.copy()
for _ in range(200):
    walk[:6] += analytic_step(walk, position, rotation, .085)
print('dls_joint_error_after_200_steps', float(np.max(np.abs(walk[:6]-target))))
PY
```

Once a dataset directory is finalized, check that the residual labels are not
degenerate before spending epochs on them:

```bash
PYTHONNOUSERSITE=1 "$ATEC_PYTHON" - <<'PY'
import numpy as np
from pathlib import Path
from tools.task_e.il.train_cartesian_bc import derive, load_dataset
from tools.task_e.il.cartesian import residual_scale_from_labels
data, provenance = load_dataset([Path('runs/task_e_il/data_seed42_complete')])
features, labels, analytic = derive(data)
print('frames', len(labels))
print('teacher_delta_mae', float(np.mean(np.abs(data['delta']))))
print('residual_label_mae', float(np.mean(np.abs(labels))))          # DLS-only error
print('residual_over_teacher', float(np.mean(np.abs(labels))/max(1e-9, np.mean(np.abs(data['delta'])))))
scale = residual_scale_from_labels(labels)
print('residual_scale', scale.tolist(), 'coverage', float(np.mean(np.abs(labels) <= scale)))
PY
```

`residual_over_teacher` near 0 means DLS already reproduces the teacher and
there is little to learn; near or above 1 means the Cartesian goal and the
teacher's joint move disagree a lot and the residual bound must be checked.
Report the number either way — it is the honest measure of how much of this
variant is learned.

## Training

Whole disjoint seeds only; the loader rejects unfinalized datasets, chunk-count
mismatches, teacher deltas above their declared limit, and any overlap of
episode id, directory or seed between train and validation.

```bash
PYTHONNOUSERSITE=1 "$ATEC_PYTHON" tools/task_e/il/train_cartesian_bc.py \
  --train runs/task_e_il/data_seed42_complete \
  --validation runs/task_e_il/data_seed0_complete \
  --output runs/task_e_il/cartesian_01 --epochs 200
```

`--residual-quantile` (default 0.995) sets the residual bound from training
label coverage; `training_report.json` records the chosen scale, the fraction
of labels inside it, per-phase learned vs DLS-only commanded-delta error,
guard clamp counts, the fraction of frames where the residual actually changed
the command, the marginal input-range shift between the two seeds, raw chunk
and metadata SHA-256 for both splits, the collector's source hashes and the
checkpoint SHA-256.

## Closed-loop evaluation

Ablation first (network bypassed, everything else identical), then the learned
student, on an independent seed:

```bash
ATEC_IL_CHECKPOINT=runs/task_e_il/cartesian_01/cartesian_bc.pt \
ATEC_IL_METRICS=runs/task_e_il/cartesian_dls_seed1_metrics.json ATEC_IL_DLS_ONLY=1 \
  bash run.sh task-e --headless --solution solution_task_e_il_cartesian.py --seed 1 \
  --max_steps 6000 --output runs/task_e_il/cartesian_dls_seed1 --snapshot_interval 600

ATEC_IL_CHECKPOINT=runs/task_e_il/cartesian_01/cartesian_bc.pt \
ATEC_IL_METRICS=runs/task_e_il/cartesian_seed1_metrics.json \
  bash run.sh task-e --headless --solution solution_task_e_il_cartesian.py --seed 1 \
  --max_steps 6000 --output runs/task_e_il/cartesian_seed1 --snapshot_interval 600
```

Both runs still load and validate the checkpoint, so the ablation differs only
in whether the residual is applied. Metrics are written at startup, every 250
policy calls, on giveup, from `finalize()` (the evaluator calls it before Isaac
shutdown) and from an `atexit` backstop; `finalized_before_sim_shutdown` in the
JSON says which runs ended cleanly. `<metrics>.servo.npz` holds the per-call
features, analytic step, residual, commanded delta, `q`, goal position, limit,
phase and object for offline analysis.

Report from `result.json`: score, termination reason, steps and wall time. From
the metrics JSON: `learned_fraction_all_calls` **and**
`learned_fraction_moving_calls` (the first includes INIT/gripper/hold calls,
which are not learned — do not quote only the second),
`residual_changed_command_calls`, `teacher_action_fallback_calls` (0 by
construction), `rule_contact_stop_calls`, and the two guard clamp counts.

## Limitations

- This is hierarchical Cartesian servo IL, not end-to-end visual IL and not a
  learned Task E policy. Perception, planning, the FSM and the gripper are
  hand-coded, and the analytic DLS still produces most of every command.
- Offline residual error is not a score. No pass/score claim is made anywhere
  in the checkpoint, the report or the metrics until closed-loop evaluator
  results exist, and one seed is one sample, not a result.
- Supervision is two teacher episodes from one rule-based controller. The
  residual is only defined where that teacher went; the input-range shift block
  is a marginal diagnostic, not a coverage guarantee.
- The guards clamp physics, not mistakes. A clamped call is not corrected
  toward the teacher, and a high clamp count means the learned residual is
  fighting the rate limit and should be treated as a failure signal.
- `derive` recomputes FK/DLS per frame in Python; expect roughly a minute per
  10k frames on CPU, once, before the training loop.
- The rotation error uses the same `orientation_weight` convention as
  `differential_ik`. Changing `DLS_DAMPING`, `DLS_ORIENTATION_WEIGHT` or the
  fixed feature scales invalidates existing checkpoints; bump
  `CARTESIAN_INTERFACE_VERSION` if any of them changes.
