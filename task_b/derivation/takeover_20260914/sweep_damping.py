"""CPU stability sweep of the payload damping coefficient (SYNTHETIC, no physics).

The plant is a zero-lag position follower, which is the WORST case for a discrete
derivative term and is not the original robot. A value that settles here is a
necessary sanity bound, not a guarantee about the real loaded joint.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from unittest.mock import patch
import numpy as np

sys.path.insert(0, '/home/lybm/ATEC_Experiments_20260910/task_b_takeover_20260914')
ROOT = Path('/home/lybm/ATEC_Robotics_Projects_20260910')
sys.path.insert(0, str(ROOT))

import smoke_payload_motion as smoke
import task_b.payload_motion as pm

results = []
for q2_damp, q3_damp in ((0., 0.), (.02, .012), (.05, .03), (.08, .05), (.11, .07), (.20, .12), (.35, .20)):
    axes = {1: (4., .14, q2_damp), 2: (2., .08, q3_damp)}
    with patch.object(pm, 'FEEDBACK_AXES', axes):
        buffer = []
        original = print
        try:
            import io, contextlib
            with contextlib.redirect_stdout(io.StringIO()) as sink:
                smoke.main()
            text = sink.getvalue()
            payload = json.loads(text[:text.rindex('}', 0, text.index('\n{'))+1]
                                 if '\n{' in text else text)
        except Exception as error:
            results.append({'q2_damp': q2_damp, 'q3_damp': q3_damp, 'error': repr(error)})
            continue
    debug = payload.get('final_debug') or {}
    qdot = debug.get('arm_qdot_rad_s') or []
    results.append({
        'q2_damp': q2_damp, 'q3_damp': q3_damp,
        'done_reason': payload.get('done_reason'),
        'phases': payload.get('payload_phases_reached'),
        'final_goal_error_rad': debug.get('actual_goal_error_rad'),
        'final_max_abs_qdot_rad_s': (float(np.max(np.abs(qdot))) if qdot else None),
        'quiet_ready': debug.get('quiet_ready'),
        'completed_all_segments': payload.get('done_reason') == 'payload_motion_observation_complete',
    })
    print(json.dumps(results[-1]))

out = Path('/home/lybm/ATEC_Experiments_20260910/task_b_takeover_20260914/sweep_damping.json')
out.write_text(json.dumps({'synthetic_zero_lag_plant': True,
                           'is_not_a_real_stability_guarantee': True,
                           'results': results}, indent=2)+'\n')
