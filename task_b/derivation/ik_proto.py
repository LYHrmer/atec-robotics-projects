"""Prototype: shorten each leg vertically by Delta while keeping the wheel (x,y)."""
import numpy as np, sys
sys.path.insert(0, str(_Path(__file__).resolve().parent))
from legmodel import foot_body_xyz, CORNERS, LIMITS
from scipy.optimize import least_squares

COMPACT = {
 "FR": (-.3023523168313238, .8130831661377943, -1.8754745945886573),
 "FL": ( .30696657668610144, .8173372201608101, -1.8881043202266299),
 "RR": (-.27114297155288936, .9786392186138522, -1.729057945170937),
 "RL": ( .2750100973730118, .9858544286612007, -1.7429508632418722)}
lo = np.array([LIMITS["hip"][0], LIMITS["thigh"][0], LIMITS["calf"][0]])
hi = np.array([LIMITS["hip"][1], LIMITS["thigh"][1], LIMITS["calf"][1]])

def solve(corner, q0, drop):
    p0 = foot_body_xyz(corner, *q0)
    target = p0 + np.array([0., 0., drop])          # wheel rises in the body frame = body drops
    def res(q): return foot_body_xyz(corner, *q) - target
    f = least_squares(res, np.clip(q0, lo+1e-6, hi-1e-6), bounds=(lo+1e-6, hi-1e-6),
                      max_nfev=500, ftol=1e-12, xtol=1e-12)
    return f.x, float(np.linalg.norm(res(f.x)))

print(f"{'drop':>6} | " + " | ".join(f"{c:^26}" for c in CORNERS))
print(f"{'m':>6} | " + " | ".join(f"{'q':^26}" for c in CORNERS) + "   max|err|")
for drop in (0.0, .03, .06, .09, .12, .15, .18, .21):
    row, errs = [], []
    for c in CORNERS:
        q, e = solve(c, COMPACT[c], drop); row.append(q); errs.append(e)
    print(f"{drop:6.2f} | " + " | ".join(f"{np.round(q,3)}" for q in row) + f"   {max(errs):.2e}")
print("\njoint limits: hip +/-0.87  thigh -0.94..4.69  calf -2.82..-0.43 rad")
