"""Test killing_2d / gunther17_2d on the Rotation-Four-Center (RFC) flow.

RFC's exact observer is a constant rotation with angular velocity c = -1, i.e. the
velocity field  u = (y, -x) = constant_rotation.  So:
  * killing_2d(rfc).params  ->  (a,b,c) ~= (0, 0, -1)  at every timestep
  * both methods' recovered observer u  ~=  constant_rotation field
"""
import sys
import numpy as np

sys.path.insert(0, r"C:\Users\xingdi\sources\PyflowVis")

from FLowUtils.AnalyticalFlowCreator import rotation_four_center, constant_rotation
from FLowUtils.ReferenceFrameOptimization import (
    killing_optimization_2d, gunther17_optimization_2d,
)

GRID = (64, 64)
NT = 64

rfc = rotation_four_center(GRID, NT)
cr = constant_rotation(GRID, NT)

rfc_data = np.asarray(rfc.field if not hasattr(rfc.field, "detach") else rfc.field.detach().cpu().numpy(), np.float64)
cr_data = np.asarray(cr.field if not hasattr(cr.field, "detach") else cr.field.detach().cpu().numpy(), np.float64)

print("rfc.field shape:", rfc_data.shape, " cr.field shape:", cr_data.shape)
print("domain:", rfc.domainMinBoundary, "->", rfc.domainMaxBoundary, " tmin,tmax:", rfc.getMinTime(), rfc.getMaxTime())

# sanity: constant_rotation at a few points should equal (y, -x)
ix, iy = 48, 32  # some grid point
px, py = rfc.convert_grid_pos_2_physical_pos(ix, iy)
print(f"constant_rotation at (x={px:.2f},y={py:.2f}) = {cr_data[0, iy, ix]}  expect (y,-x)=({py:.2f},{-px:.2f})")


def rel_err(u, uref, mask=None):
    """||u - uref|| / ||uref||, optionally in a region mask (spatial, (Y,X))."""
    T = u.shape[0]
    num = den = 0.0
    for t in range(1, T - 1):
        d = u[t] - uref[t]
        if mask is not None:
            d = d[mask]; r = uref[t][mask]
        else:
            r = uref[t]
        num += np.sum(d * d); den += np.sum(r * r)
    return np.sqrt(num / den)


# ── killing_2d ──
rk = killing_optimization_2d(rfc)
p = rk.params  # (T,3)=(a,b,c)
print("\n[killing_2d] params (a,b,c) mean over t:", np.round(p[1:-1].mean(0), 4), " expect (0,0,-1)")
print("             params (a,b,c)  std over t:", np.round(p[1:-1].std(0), 4))
uk = np.asarray(rk.u_field.field, np.float64)
ek = rel_err(uk, cr_data)
print(f"[killing_2d] observer u vs constant_rotation: relative L2 error = {ek:.4e}")

# ── gunther17_2d ──
rg = gunther17_optimization_2d(rfc, neighborhood=4)
ug = np.asarray(rg.u_field.field, np.float64)
eg_all = rel_err(ug, cr_data)
# central region (avoid the very boundary where finite neighborhood clamps)
Y, X = GRID[1], GRID[0]
m = np.zeros((Y, X), bool); m[6:-6, 6:-6] = True
eg_core = rel_err(ug, cr_data, m)
print(f"\n[gunther17_2d] observer u vs constant_rotation: relative L2 error = {eg_all:.4e} (all), {eg_core:.4e} (interior)")

# report angular velocity recovered by gunther pointwise: u = (y,-x) => omega_z = dv/dx - du/dy = -2
def omega_z(u_slice, dx, dy):
    dv_dx = np.gradient(u_slice[..., 1], dx, axis=1)
    du_dy = np.gradient(u_slice[..., 0], dy, axis=0)
    return dv_dx - du_dy  # vorticity; /2 = angular velocity
dx = rfc.gridInterval[0]; dy = rfc.gridInterval[1]
wg = omega_z(ug[NT // 2], dx, dy)[m]
print(f"[gunther17_2d] recovered omega (vorticity/2) interior mean = {0.5*wg.mean():.4f}  expect -1")

print("\n--- verdict ---")
ok_k = abs(p[1:-1, 2].mean() + 1.0) < 0.02 and ek < 0.02
ok_g = eg_core < 0.10
print("killing_2d   recovers constant_rotation:", "PASS" if ok_k else "FAIL")
print("gunther17_2d recovers constant_rotation:", "PASS" if ok_g else "FAIL")
