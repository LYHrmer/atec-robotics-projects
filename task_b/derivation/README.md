# Derivation studies (archival)

These are the one-off CPU studies that produced the numbers the M2/M3 design is
built on. They are kept because the reasoning behind the constants in
`task_b/arm_kinematics.py`, `task_b/leg_kinematics.py` and the probe parameters is
otherwise only in the commit messages.

They are **not** part of the runnable pipeline and nothing imports them. They were
written to answer one question each, against the local USD assets and a recorded
run, and several print rather than assert. Run them by hand when you need to
re-derive a constant:

## Subdirectories

Four further archives sit alongside the studies above. They are larger and more
mixed, and like the studies nothing in the shipped path imports them.

| directory | what it holds |
| --- | --- |
| `score_cpu/` | the CPU audit scripts and their JSON output that decided the docking, release, finger-sweep and payload-raise geometry — the numbers the delivery design rests on. Includes `D3_YAW_VERDICT.md`, `CARRY_PROBE_RESULT_d1_d2.md` and the fixture-free audit records. |
| `takeover_20260914/` | the control and audit source written during the day the delivery was pushed (`audit_carry_probe.py`, `audit_first_delivery.py`, `plant_model.py`, follow-up verification and smoke scripts). |
| `astra_review_20260914/` | contact and geometry studies from the review round, including the 200 KB `contact_force75_wrist_depth.json` and `next_delivery_plan_astra_2006.md`. |
| `plan_opus/` | provenance of the designated control modules: the prompts, the answers, and the versioned `.py` files as they were handed over, named with a content hash. Several are near-duplicates of each other and of files elsewhere in the repo; they are kept as the record of what was delivered, not as working code. |

Two things were deliberately left out: `__pycache__` directories, and about 9 MB
of synthetic audit fixtures under `task_b_score_cpu/*_fixtures*`, which are
regenerable and contain no recorded evidence.

## Studies

| script | question it answered |
| --- | --- |
| `bbox.py` | world-space bounding box of each Task B object asset, from USD only |
| `limits.py` | the Piper's real joint limits straight out of the robot USD |
| `legframes.py`, `legmodel.py`, `analyze_stance.py` | leg joint frames and the wheels-planted body-drop model |
| `default_solve.py`, `lowest.py`, `pitch.py`, `pitchclear.py` | the default stance, its lowest reachable body height and its pitch clearance |
| `reach_study.py`, `reach_recheck.py` | can the finger slab straddle an object, and at what park |
| `ik_proto.py`, `clearance.py` | the IK and approach-clearance prototypes behind `solve_ik` |
| `fingers.py`, `fingers2.py`, `width.py` | jaw travel and the object widths the jaws must straddle |
| `frames.py`, `frames2.py`, `jaw_at_m1.py`, `mount_check.py` | checking the model's frames against a recorded run's own gripper record |
| `tops.py`, `wheel.py` | object top heights and wheel geometry |

The path insertions were rewritten from the original scratch directory to resolve
relative to this file, so they run from the repository. A few still read absolute
paths under `ATEC2026_Simulation_Challenge/atec_robot_model`, which is where the
official assets live on the machine these were run on.

The evidence those studies feed is re-derived, not remembered:
`task_b/grasp_evidence.py` recomputes the M2 grip claim from a run's telemetry,
`task_b/audit_delivery_score.py` recomputes the official score from the same
telemetry, and `task_b/results/grasp_evidence/` holds the per-run output.
