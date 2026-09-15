import numpy as np, sys
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from pxr import Usd, Gf
import task_e_geometry as G
p="/home/lybm/ATEC2026_Simulation_Challenge/atec_robot_model/robot/b2w/b2w_piper.usda"
st=Usd.Stage.Open(p)
rows=[]
for i in range(1,7):
    pr=st.GetPrimAtPath(f"/b2w_description/arm_joints/arm_joint{i}")
    q=pr.GetAttribute("physics:localRot0").Get()
    r=q.GetReal(); im=q.GetImaginary()
    rows.append([r, im[0], im[1], im[2]])
usd=np.array(rows); model=np.array(G._JOINT_QUAT)
for i,(a,b) in enumerate(zip(usd,model),1):
    print(f"joint{i}: usd={np.round(a,7)}  model={np.round(b,7)}  d={np.abs(a-b).max():.2e}")
print("max |USD localRot0 - model| = %.3e" % np.abs(usd-model).max())
