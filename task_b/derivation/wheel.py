from pxr import Usd, UsdGeom
import numpy as np
p="/home/lybm/ATEC2026_Simulation_Challenge/atec_robot_model/robot/b2w/b2w_piper.usda"
st=Usd.Stage.Open(p)
xc=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_,UsdGeom.Tokens.render])
for n in ("FR_foot","FL_foot","RR_foot","RL_foot","FR_calf","FR_thigh"):
    prim=st.GetPrimAtPath(f"/b2w_description/{n}")
    if not prim or not prim.IsValid(): 
        cands=[q.GetPath() for q in st.Traverse() if q.GetName()==n and q.GetTypeName()=="Xform"]
        print(n,"->",cands[:2]); 
        if cands: prim=st.GetPrimAtPath(cands[0])
        else: continue
    r=xc.ComputeWorldBound(prim).ComputeAlignedRange(); lo,hi=r.GetMin(),r.GetMax()
    size=np.array(hi)-np.array(lo)
    print(f"{n:10s} size=({size[0]:.4f},{size[1]:.4f},{size[2]:.4f})  min=({lo[0]:+.4f},{lo[1]:+.4f},{lo[2]:+.4f}) max=({hi[0]:+.4f},{hi[1]:+.4f},{hi[2]:+.4f})")
