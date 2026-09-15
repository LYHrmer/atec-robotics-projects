"""Predict which official contact body reaches the ground first, vs commanded drop.

Uses the leg FK plus each link's local USD bound. Body height is taken relative to
the MEASURED settled stance (0.5344 m, task_b/results/piper_gripper_and_reach.json),
not from the model, because the loaded legs sag well below the nominal default pose.
"""
import numpy as np, sys
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from task_b import leg_kinematics as legs
from pxr import Usd, UsdGeom
from scipy.spatial.transform import Rotation as Rot

USD="/home/lybm/ATEC2026_Simulation_Challenge/atec_robot_model/robot/b2w/b2w_piper.usda"
st=Usd.Stage.Open(USD)
xc=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_,UsdGeom.Tokens.render])
LOCAL={}
for c in legs.CORNERS:
    for link in ("thigh","calf","foot"):
        prim=st.GetPrimAtPath(f"/b2w_description/{c}_{link}")
        if not prim or not prim.IsValid():
            cands=[q for q in st.Traverse() if q.GetName()==f"{c}_{link}" and q.GetTypeName()=="Xform"]
            prim=cands[0] if cands else None
        if prim is None: continue
        r=xc.ComputeLocalBound(prim).ComputeAlignedRange(); lo,hi=r.GetMin(),r.GetMax()
        corners=np.array([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])
        LOCAL[(c,link)]=corners
base=st.GetPrimAtPath("/b2w_description/base_link")
_r=xc.ComputeLocalBound(base).ComputeAlignedRange()
BASE_LOCAL=np.array([[x,y,z] for x in (_r.GetMin()[0],_r.GetMax()[0])
                     for y in (_r.GetMin()[1],_r.GetMax()[1]) for z in (_r.GetMin()[2],_r.GetMax()[2])])

def Rx(a): return Rot.from_rotvec([a,0,0]).as_matrix()
def Ry(a): return Rot.from_rotvec([0,a,0]).as_matrix()

def frames(c, hip, thigh, calf):
    """Body-frame transforms of the thigh / calf / foot link origins."""
    T_hip = np.eye(3), legs._HIP[c][0]
    R1, p1 = Rx(hip), np.zeros(3)
    T_thigh = R1, legs._HIP[c][0] + R1 @ legs._THIGH[c][0]
    T_calf  = R1 @ Ry(thigh), T_thigh[1] + (R1 @ Ry(thigh)) @ legs._CALF[c][0]
    T_foot  = T_calf[0] @ Ry(calf), T_calf[1] + (T_calf[0] @ Ry(calf)) @ legs._FOOT[c][0]
    return {"thigh":T_thigh, "calf":T_calf, "foot":T_foot}

SETTLED_H = 0.5344   # measured natural stance, piper_gripper_and_reach.json
POSE = {c:(0.0,0.8,-1.5) if c in ("FR","FL") else (0.0,1.0,-1.5) for c in legs.CORNERS}
# The loaded legs sit below the unloaded default pose, so the settled geometry is
# the default pose with every wheel raised by the measured sag. At the nominal
# default the FR/FL wheel axle sits 0.4969 m below the body origin, but a wheel of
# radius 0.1129 supporting a body at 0.5344 must sit 0.4215 m below it.
SAG = 0.4969 - (SETTLED_H - legs.WHEEL_RADIUS_M)
print(f"modelled sag from the unloaded default pose: {SAG*1000:.1f} mm")

print(f"{'drop':>5} {'body_h':>7} | " + " ".join(f"{c+'_'+l:>13}" for c in legs.CORNERS for l in ("thigh","calf","foot")) )
for drop in np.arange(0., 0.301, 0.02):
    H = SETTLED_H - drop
    row={}
    for c in legs.CORNERS:
        q = np.array(POSE[c])
        # descend solves from the nominal pose; use it, then evaluate the real FK
        sol,_ = legs.descend(c, q, SAG + drop)
        F = frames(c, *sol)
        for link in ("thigh","calf","foot"):
            R,p = F[link]
            pts = (R @ LOCAL[(c,link)].T).T + p
            row[f"{c}_{link}"] = H + pts[:,2].min()
    k=min(row, key=row.get)
    if drop % 0.04 < 1e-9 or row[k] < 0.05:
        print(f"{drop:5.2f} {H:7.4f} | " + " ".join(f"{row[f'{c}_{l}']:13.4f}" for c in legs.CORNERS for l in ("thigh","calf","foot")) + f"   <- lowest {k}")
    if row[k] <= 0.0:
        print(f"\nPREDICTED FIRST CONTACT: {k} at commanded drop ~{drop:.2f} m (body height {H:.3f} m)")
        break
