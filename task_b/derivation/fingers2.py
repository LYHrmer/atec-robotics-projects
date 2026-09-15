from pxr import Usd, UsdGeom, UsdPhysics
p="/home/lybm/ATEC2026_Simulation_Challenge/atec_robot_model/robot/b2w/b2w_piper.usda"
st=Usd.Stage.Open(p)
for prim in st.Traverse():
    n=prim.GetName()
    if n in ("arm_joint7","arm_joint8"):
        print("PATH", prim.GetPath())
        for attr in ("physics:localPos0","physics:localRot0","physics:localPos1","physics:localRot1"):
            a=prim.GetAttribute(attr)
            print(f"  {attr:18s} = {a.Get() if a and a.IsValid() else None}")
xc=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_,UsdGeom.Tokens.render,UsdGeom.Tokens.proxy])
for prim in st.Traverse():
    if prim.GetName() in ("gripper_base","arm_link7","arm_link8") and prim.GetTypeName()=="Xform":
        r=xc.ComputeWorldBound(prim).ComputeAlignedRange()
        lo,hi=r.GetMin(),r.GetMax()
        print(f"{prim.GetName():14s} {str(prim.GetPath()):40s}")
        print(f"    min=({lo[0]:+.4f},{lo[1]:+.4f},{lo[2]:+.4f}) max=({hi[0]:+.4f},{hi[1]:+.4f},{hi[2]:+.4f})")
