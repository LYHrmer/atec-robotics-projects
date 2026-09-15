import numpy as np, sys
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from task_b.arm_kinematics import fk, pinch_position, GRASP_DEPTH

# From the M1 run's own policy debug (result.json).
reach_goal_q = np.array([0.5495857422686986, 3.1399988169418163, -1.676926793594718,
                         0.0031833506576024596, -0.08839255959722687, 0.0])
predicted_gripper_body = np.array([0.5223120423867686, 0.19739333364016873, -0.14430601805855275])
reach_target_body      = np.array([0.5397154678108251, 0.20387822389278645, -0.3009624417189533])

pose = fk(reach_goal_q)
print("fk(reach_goal_q)[:3,3] =", np.round(pose[:3,3], 5), " (run recorded", np.round(predicted_gripper_body,5), ")")
print("max |fk - recorded| =", float(np.max(np.abs(pose[:3,3]-predicted_gripper_body))))

zaxis = pose[:3,2]; xaxis = pose[:3,0]; yaxis = pose[:3,1]
print("gripper local +Z (approach) in body frame =", np.round(zaxis,4))
print("gripper local +X in body frame            =", np.round(xaxis,4))
print("gripper local +Y (jaw opening axis)       =", np.round(yaxis,4))
print("angle of +Z from straight-down (0,0,-1)   = %.1f deg" % np.degrees(np.arccos(np.clip(-zaxis[2],-1,1))))

mid = pinch_position(reach_goal_q)
print("\njaw midpoint (0.115 m along +Z) body-frame =", np.round(mid,5))
d = mid - reach_target_body
print("jaw_mid - object_root =", np.round(d,5), " |d| = %.4f m" % np.linalg.norm(d))
print("  horizontal = %.4f   vertical = %+.4f" % (np.linalg.norm(d[:2]), d[2]))
print("fingers reach to 0.1358 along +Z:  body-frame =", np.round(fk(reach_goal_q)[:3,3]+0.1358*zaxis,5))
