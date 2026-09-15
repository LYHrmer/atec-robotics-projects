"""Independently read every Piper joint frame from the USD and compare to the model."""
import numpy as np, sys
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from pxr import Usd, UsdPhysics
import task_e_geometry as G

p="/home/lybm/ATEC2026_Simulation_Challenge/atec_robot_model/robot/b2w/b2w_piper.usda"
st=Usd.Stage.Open(p)
usd_pos, usd_quat, usd_axis = [], [], []
for i in range(1,7):
    pr=st.GetPrimAtPath(f"/b2w_description/arm_joints/arm_joint{i}")
    usd_pos.append(np.array(pr.GetAttribute("physics:localPos0").Get()))
    usd_quat.append(np.array(pr.GetAttribute("physics:localRot0").Get()))  # wxyz
    usd_axis.append(pr.GetAttribute("physics:axis").Get())
usd_pos=np.array(usd_pos); usd_quat=np.array(usd_quat)
print("USD localPos0 (x,y,z) per joint:")
for i,(a,b) in enumerate(zip(usd_pos, G._JOINT_POS), 1):
    print(f"  joint{i}: usd={np.round(a,8)}  model={np.round(b,8)}  d={np.abs(a-b).max():.2e}")
print("max |USD localPos0 - model _JOINT_POS| = %.3e" % np.abs(usd_pos-G._JOINT_POS).max())
print("axes:", usd_axis)
# quaternion convention: model _JOINT_QUAT is wxyz
dq = np.abs(usd_quat - G._JOINT_QUAT).max()
print("max |USD localRot0 - model _JOINT_QUAT| (wxyz) = %.3e" % dq)
print("=> frames identical" if dq < 1e-6 and np.abs(usd_pos-G._JOINT_POS).max() < 1e-6 else "=> MISMATCH")
