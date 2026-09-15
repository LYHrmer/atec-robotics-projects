"""World-space bounding box of each Task B object asset, via USD only (no Kit/GPU)."""
import sys
from pxr import Usd, UsdGeom, Gf

paths = {
    "cracker_box":  "/home/lybm/ATEC2026_Simulation_Challenge/atec_robot_model/objects/task_b/003_cracker_box.usd",
    "sugar_box":    "/home/lybm/ATEC2026_Simulation_Challenge/atec_robot_model/objects/task_b/004_sugar_box.usd",
    "mustard_bottle":"/home/lybm/ATEC2026_Simulation_Challenge/atec_robot_model/objects/task_b/006_mustard_bottle.usd",
    "banana":       "/home/lybm/ATEC2026_Simulation_Challenge/atec_robot_model/objects/task_b/011_banana.usd",
}
for name, path in paths.items():
    stage = Usd.Stage.Open(path)
    xc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy])
    rng = xc.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
    lo, hi = rng.GetMin(), rng.GetMax()
    size = hi - lo
    center = (hi + lo) / 2.0
    print(f"{name:16s} size(x,y,z)=({size[0]:.4f}, {size[1]:.4f}, {size[2]:.4f})  "
          f"min=({lo[0]:+.4f},{lo[1]:+.4f},{lo[2]:+.4f})  center=({center[0]:+.4f},{center[1]:+.4f},{center[2]:+.4f})")
