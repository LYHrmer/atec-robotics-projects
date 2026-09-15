"""Width of each Task B object along the jaw-closing axis, vs height above the ground."""
import numpy as np
from pxr import Usd, UsdGeom
from scipy.spatial.transform import Rotation

D="/home/lybm/ATEC2026_Simulation_Challenge/atec_robot_model/objects/task_b/"
ASSETS={"sugar":"004_sugar_box.usd","mustard":"006_mustard_bottle.usd","banana":"011_banana.usd"}
QUAT={"sugar":[0.0,0.707,0.0,0.707],"mustard":[0.0,0.0,-0.707,0.707],"banana":[0.0,0.0,-0.707,0.707]}
GROUND=0.0449   # measured terrain top surface

for kind,usd in ASSETS.items():
    st=Usd.Stage.Open(D+usd)
    pts=[]
    for prim in st.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            m=UsdGeom.Mesh(prim)
            p=m.GetPointsAttr().Get()
            if p: pts.append(np.array([list(v) for v in p]))
    P=np.vstack(pts)
    R=Rotation.from_quat(np.array(QUAT[kind])[[1,2,3,0]]).as_matrix()
    W=(R@P.T).T
    z0=W[:,2].min()
    print(f"\n{kind}: {len(P)} verts, world-y extent (jaw closing axis) by height above ground")
    print(f"  {'h_above_ground':>15} {'width_x':>9} {'width_y':>9}  {'fits 7cm jaw?':>14}")
    for h in np.arange(0.0, W[:,2].max()-z0+0.001, 0.01):
        m=(W[:,2]-z0 >= h) & (W[:,2]-z0 < h+0.01)
        if m.sum() < 3: continue
        wx=W[m,0].max()-W[m,0].min(); wy=W[m,1].max()-W[m,1].min()
        top = W[:,2].max()-z0
        print(f"  {h:15.3f} {wx:9.4f} {wy:9.4f}  {'YES' if wy<=0.07 else 'NO':>14}   (from top: {top-h:.3f})")
