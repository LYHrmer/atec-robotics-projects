import json, numpy as np, sys
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from scipy.spatial.transform import Rotation
from task_b.arm_kinematics import fk

r = json.load(open("results/task_b_positive/plan_p2_lower02_seed42_01/result.json"))
fs = r["final_state_before_close"]
base = np.array(fs["base_xyz"]); quat = np.array(fs["base_quat"])   # wxyz
gripper_meas = np.array(fs["gripper_xyz"])
print("base_quat_wxyz", np.round(quat,5))
R = Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
print("body axes in world: x",np.round(R[:,0],3)," y",np.round(R[:,1],3)," z",np.round(R[:,2],3))
print("tilt from vertical (deg):", round(float(np.degrees(np.arccos(np.clip(R[2,2],-1,1)))),2))

goal = np.array([0.5495857422686986, 3.1399988169418163, -1.676926793594718,
                 0.0031833506576024596, -0.08839255959722687, 0.0])
pred_body = fk(goal)[:3,3]
pred_world = base + R @ pred_body
print("\npredicted gripper world =", np.round(pred_world,4))
print("measured  gripper world =", np.round(gripper_meas,4))
print("error (m) = %.4f" % float(np.linalg.norm(pred_world-gripper_meas)))

# object_10: model body-frame estimate vs measured world
obj_meas = np.array(fs["object_xyz"])[9]
obj_body_from_meas = R.T @ (obj_meas - base)
print("\nobject_10 measured world =", np.round(obj_meas,4))
print("object_10 body frame (from measured) =", np.round(obj_body_from_meas,4))
print("object_10 body frame (visual estimate) = [0.5397 0.2039 -0.301]", )
vis = np.array([0.5397154678108251, 0.20387822389278645, -0.3009624417189533])
print("visual/localization error (m) = %.4f" % float(np.linalg.norm(obj_body_from_meas-vis)))
print("  horizontal err %.4f  vertical err %+.4f" % (np.linalg.norm(obj_body_from_meas[:2]-vis[:2]), obj_body_from_meas[2]-vis[2]))
