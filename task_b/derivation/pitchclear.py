"""Crude pitch bound: lowest world z of the official illegal-contact bodies vs pitch."""
import numpy as np, re
from pxr import Usd, UsdGeom
from scipy.spatial.transform import Rotation
USD="/home/lybm/ATEC2026_Simulation_Challenge/atec_robot_model/robot/b2w/b2w_piper.usda"
st=Usd.Stage.Open(USD)
xc=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_,UsdGeom.Tokens.render])
PAT=re.compile(r"^(base_link|.*_hip.*|.*_thigh.*)$")
pts=[]
for prim in st.Traverse():
    n=prim.GetName()
    if prim.GetTypeName()=="Xform" and PAT.match(n) and "/visuals/" not in str(prim.GetPath()) and "/collisions/" not in str(prim.GetPath()):
        r=xc.ComputeWorldBound(prim).ComputeAlignedRange()
        lo,hi=r.GetMin(),r.GetMax()
        if not np.isfinite([lo[0],hi[0]]).all(): continue
        print(f"{n:28s} body-frame z range [{lo[2]:+.4f}, {hi[2]:+.4f}]  x [{lo[0]:+.4f}, {hi[0]:+.4f}]")
        for x in (lo[0],hi[0]):
            for y in (lo[1],hi[1]):
                for z in (lo[2],hi[2]):
                    pts.append((n,(x,y,z)))
P=np.array([p for _,p in pts])
BASE_H=0.5148
print("\npitch_deg  lowest illegal-body world z   clearance_to_ground")
for deg in (0,5,10,15,20,25,30):
    R=Rotation.from_euler('y', np.deg2rad(deg)).as_matrix()
    z=(R@P.T).T[:,2]+BASE_H
    k=int(np.argmin(z))
    print(f"{deg:9.0f}  {z[k]:24.4f}   {z[k]:18.4f}   ({pts[k][0]})")
