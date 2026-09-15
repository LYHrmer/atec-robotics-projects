from pxr import Usd, UsdGeom, UsdPhysics, Gf
p="/home/lybm/ATEC2026_Simulation_Challenge/atec_robot_model/robot/b2w/b2w_piper.usda"
st=Usd.Stage.Open(p)
for n in ("arm_joint7","arm_joint8"):
    pr=st.GetPrimAtPath(f"/b2w_description/{n}")
    for attr in ("physics:localPos0","physics:localRot0","physics:localPos1","physics:localRot1"):
        a=pr.GetAttribute(attr)
        print(f"{n:12s} {attr:18s} = {a.Get() if a and a.IsValid() else None}")
    print()
xc=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_,UsdGeom.Tokens.render,UsdGeom.Tokens.proxy])
for link in ("gripper_base","arm_link7","arm_link8"):
    r=xc.ComputeWorldBound(st.GetPrimAtPath(f"/b2w_description/{link}")).ComputeAlignedRange()
    lo,hi=r.GetMin(),r.GetMax()
    print(f"{link:14s} world min=({lo[0]:+.4f},{lo[1]:+.4f},{lo[2]:+.4f}) max=({hi[0]:+.4f},{hi[1]:+.4f},{hi[2]:+.4f})")
