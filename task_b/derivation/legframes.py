from pxr import Usd, UsdPhysics
import numpy as np
p="/home/lybm/ATEC2026_Simulation_Challenge/atec_robot_model/robot/b2w/b2w_piper.usda"
st=Usd.Stage.Open(p)
names=[]
for prim in st.Traverse():
    if prim.IsA(UsdPhysics.RevoluteJoint):
        n=prim.GetName()
        if "_joint" in n and "arm_" not in n:
            names.append((n, prim))
for n, prim in names:
    q=prim.GetAttribute("physics:localRot0").Get()
    pos=prim.GetAttribute("physics:localPos0").Get()
    ax=prim.GetAttribute("physics:axis").Get()
    lo=prim.GetAttribute("physics:lowerLimit").Get(); hi=prim.GetAttribute("physics:upperLimit").Get()
    body0=prim.GetRelationship("physics:body0").GetTargets(); body1=prim.GetRelationship("physics:body1").GetTargets()
    print(f"{n:22s} axis={ax} limits=({lo:.4f},{hi:.4f})")
    print(f"    localPos0={tuple(round(v,8) for v in pos)}")
    print(f"    localRot0(wxyz)=({q.GetReal():.8f},{q.GetImaginary()[0]:.8f},{q.GetImaginary()[1]:.8f},{q.GetImaginary()[2]:.8f})")
    print(f"    body0={body0[0].name if body0 else None}  body1={body1[0].name if body1 else None}")
