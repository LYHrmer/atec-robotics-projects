Write `/home/lybm/ATEC_Robotics_Projects_20260910/task_b/public_odometry.py` ONLY. No memory, other files, tests, Git, simulator/GPU or network. Keep this pure NumPy module small, preferably under150 lines. This is an independent actual-Claude implementation task. Existing FirstReach/MultiReach drift budgets are local guards, not full-episode navigation odometry; do not modify or import them.

Class `PublicPlanarOdometry(dt=.02)`:

- `dt` must be finite and positive. `reset()` has no pose arguments: use only the original Task B public static spawn XY=(-10.,-10.), yaw=0. This is official task configuration, not a seed-specific object map. Clear all previous-twist/history/counters on reset.
- `update(proprio)` reads one original finite84-element public observation per environment control tick. It may read only base linear velocity[0:3], base angular velocity[3:6] and projected gravity[9:12] for estimation. Validate observation shape/finiteness and gravity norm>=.5 before mutating state. Invalid input raises ValueError without changing pose/history/counters. No simulator state, pose snapshots, images, reward, score, object identifiers or seeded object coordinates.
- Timing must be explicit. The first update returns the fixed-spawn estimate at observation time0 and stores that observation's public twist. Every subsequent update integrates the PREVIOUS stored twist over exactlydt to advance to the current observation time, then stores the new twist. Thus rowN's estimate has consumed rows1..N-1, matching pre-step telemetry. Never integrate a current observation twice. There is no automatic reset or failure recovery.
- Reproduce this already-offline-tested planar approximation exactly, not a new estimator:
  `up = -gravity / norm(gravity)`;
  `v_t = linear_velocity - dot(linear_velocity,up)*up`;
  `w = dot(angular_velocity,up)`;
  `mid_yaw = yaw_unwrapped + .5*w*dt`;
  `xy += dt * Rz2(mid_yaw) @ v_t[:2]`;
  `yaw_unwrapped += w*dt`.
  Use the previous observation's storedv_t,w during each integration. Do not integrate vertical drift or treat public base linear velocity as already world-frame. Do not add filtering, bias correction, clipping, world-pose calibration or learned constants.
- Maintain `position_xy` (return a copy), `yaw_rad` wrapped to[-pi,pi), `yaw_unwrapped_rad`, `elapsed_s`, `update_calls`, `integrated_intervals`, and `path_length_m` (sum norm(previousv_t)*dt). Keep all floating state finite; malformed or overflow-producing input must fail without partial state mutation.
- `update()` returns a dictionary snapshot containing those fields, with XY serialized as a list. `describe()` returns the same snapshot plus fixed initialization source, input indices, timing and approximation claims. Use explicit wording: public-twist planar dead reckoning, no world-pose measurement, no uncertainty guarantee, no navigation/action logic.
- A small `target_body_xy(target_xy)` helper may transform any finite known static task waypoint by `Rz2(-yaw)@(target_xy-position_xy)`; it must neither change state nor retain a world/object map. The future delivery caller will use original static deliverycenter(-3,-10) and derived approach waypoints. Reject invalid targets without mutation.

Independent tests are owned by root/another agent: first-call timing; constant straight and constant yaw midpoint motion; gravity projection removes pure vertical input; body-to-planar rotation; wraparound/unwrapped continuity; reset; invalid-input atomicity; returned snapshots cannot mutate internal state; no object/reward source dependencies; replay completed original observations against recorded GT only as OFFLINE error diagnostics. Previous exact-formula replay on two completed runs gave final XY errors16.7/19.8 mm and yaw error≈.008 rad, without GT calibration. These are finite-run measurements, not promised accuracy for carrying.
