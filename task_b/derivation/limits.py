from pxr import Usd, UsdPhysics, UsdGeom
p = "/home/lybm/ATEC2026_Simulation_Challenge/atec_robot_model/robot/b2w/b2w_piper.usda"
stage = Usd.Stage.Open(p)
print("defaultPrim:", stage.GetDefaultPrim().GetName())
for prim in stage.Traverse():
    if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
        j = UsdPhysics.RevoluteJoint(prim) if prim.IsA(UsdPhysics.RevoluteJoint) else UsdPhysics.PrismaticJoint(prim)
        name = prim.GetName()
        if "arm_joint" not in name:
            continue
        lo = j.GetLowerLimitAttr().Get()
        hi = j.GetUpperLimitAttr().Get()
        body0 = prim.GetRelationship("physics:body0").GetTargets()
        body1 = prim.GetRelationship("physics:body1").GetTargets()
        axis = "?"
        a = prim.GetAttribute("physics:axis")
        if a and a.IsValid(): axis = a.Get()
        print(f"{name:12s} {prim.GetTypeName():24s} limits=({lo}, {hi})  axis={axis}")
