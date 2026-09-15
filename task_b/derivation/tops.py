"""Object top height = rotated bbox, using the exact quaternions from env_cfg."""
import numpy as np
from pxr import Usd, UsdGeom
from scipy.spatial.transform import Rotation

D="/home/lybm/ATEC2026_Simulation_Challenge/atec_robot_model/objects/task_b/"
ASSETS={"sugar":"004_sugar_box.usd","mustard":"006_mustard_bottle.usd","banana":"011_banana.usd"}
# From task_b/env_cfg.py, as authored there (w, x, y, z).
QUAT={"sugar":[0.0,0.707,0.0,0.707], "mustard":[0.0,0.0,-0.707,0.707], "banana":[0.0,0.0,-0.707,0.707]}
MEASURED_ROOT={"sugar":0.0902064,"mustard":0.1402247,"banana":0.0600874}

xc=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_,UsdGeom.Tokens.render,UsdGeom.Tokens.proxy])
for kind,usd in ASSETS.items():
    st=Usd.Stage.Open(D+usd)
    r=xc.ComputeWorldBound(st.GetPseudoRoot()).ComputeAlignedRange()
    lo,hi=np.array(r.GetMin()),np.array(r.GetMax())
    corners=np.array([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])
    R=Rotation.from_quat(np.array(QUAT[kind])[[1,2,3,0]]).as_matrix()
    w=(R@corners.T).T
    zlo,zhi=w[:,2].min(),w[:,2].max()
    print(f"{kind:8s} asset bbox size={np.round(hi-lo,4)}  centre={np.round((hi+lo)/2,5)}")
    print(f"         after spawn quat: z range [{zlo:+.4f},{zhi:+.4f}]  vertical extent {zhi-zlo:.4f}")
    print(f"         resting on ground -> root z would be {-zlo:.4f}; MEASURED {MEASURED_ROOT[kind]:.4f}"
          f"  (diff {MEASURED_ROOT[kind]+zlo:+.4f})")
    print(f"         top above ground = root + (zhi) = {MEASURED_ROOT[kind]+zhi:.4f}")
