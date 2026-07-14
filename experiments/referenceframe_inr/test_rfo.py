"""Self-test for FLowUtils/ReferenceFrameOptimization.py

Construction: a TRANSLATING vortex  v(x,t) = g(x - v0*t, y) + (v0,0,..).
Its EXACT optimal observer is the pure translation u_exact = (v0, 0[, 0]).
So all four optimizers should recover  u ≈ u_exact  wherever the vortex has signal.

Metrics:
  * killing_* : per-timestep params should be (a,b,c)=(v0,0,0) / (t,w)=((v0,0,0),0).
  * signal-weighted recovery error: ||u - u_exact|| weighted by vortex magnitude
    ||v - u_exact||  (so the undefined far-field where the vortex ~ 0 is ignored).
"""
import sys
import numpy as np

sys.path.insert(0, r"C:\Users\xingdi\sources\PyflowVis")

from FLowUtils.VectorField2d import UnsteadyVectorField2D
from FLowUtils.VectorField3d import UnsteadyVectorField3D
from FLowUtils.ReferenceFrameOptimization import (
    killing_optimization_2d, killing_optimization_3d,
    gunther17_optimization_2d, gunther17_optimization_3d,
)


def recovery_error(v, u, u_exact, margin, dim):
    """Relative observer-recovery error in the vortex CORE (strong signal, away from
    boundary). Local methods legitimately degrade at the vortex periphery / where the
    finite neighborhood is clamped at the domain boundary, so the core is the fair test.
    Returns mean over interior timesteps of  mean(||u - u_exact||) / |v0|  in the core."""
    v = v.astype(np.float64); u = u.astype(np.float64)
    ue = np.array(u_exact, np.float64)
    v0 = max(np.linalg.norm(ue), 1e-9)
    T = v.shape[0]
    out = []
    for t in range(1, T - 1):
        g = np.linalg.norm(v[t] - ue, axis=-1)  # vortex magnitude
        core = g >= 0.4 * g.max()               # strong-signal core
        if dim == 2:
            m = np.zeros_like(core); m[margin:-margin, margin:-margin] = True
        else:
            m = np.zeros_like(core); m[margin:-margin, margin:-margin, margin:-margin] = True
        core &= m
        if core.sum() == 0:
            continue
        e = np.linalg.norm(u[t] - ue, axis=-1)
        out.append(e[core].mean() / v0)
    return float(np.mean(out)) if out else 0.0


# ───────────────────────── 2D ─────────────────────────
def build_translating_vortex_2d(X=48, Y=48, T=9, v0=0.35):
    xmin, xmax, ymin, ymax = -2.0, 2.0, -2.0, 2.0
    tmin, tmax = 0.0, 2.0
    f = UnsteadyVectorField2D(X, Y, T, [xmin, ymin, tmin], [xmax, ymax, tmax])
    data = np.zeros((T, Y, X, 2), np.float32)
    xs = np.linspace(xmin, xmax, X)
    ys = np.linspace(ymin, ymax, Y)
    Yg, Xg = np.meshgrid(ys, xs, indexing="ij")
    for it in range(T):
        t = tmin + (tmax - tmin) * it / (T - 1)
        xi = Xg - v0 * t
        env = np.exp(-(xi**2 + Yg**2))
        data[it, :, :, 0] = -Yg * env + v0     # lab field = swirl(translated) + translation
        data[it, :, :, 1] = xi * env
    f.field = data
    return f, v0


def test_2d():
    print("=== 2D ===")
    f, v0 = build_translating_vortex_2d()
    data = f.field.astype(np.float64)
    ue = (v0, 0.0)

    res_k = killing_optimization_2d(f)
    p = res_k.params
    print("killing params (a,b,c) interior mean:", np.round(p[1:-1].mean(0), 4),
          " expected (", v0, ", 0, 0)")
    ek = recovery_error(data, res_k.u_field.field, ue, 4, 2)
    print(f"killing  recovery error (rel) = {ek:.4f}")
    assert abs(p[1:-1, 0].mean() - v0) < 0.05, "killing_2d translation a wrong"
    assert abs(p[1:-1, 1].mean()) < 0.05 and abs(p[1:-1, 2].mean()) < 0.05, "killing_2d b/c not ~0"
    assert ek < 0.02, "killing_2d observer not recovered"

    res_g = gunther17_optimization_2d(f, neighborhood=3)
    eg = recovery_error(data, res_g.u_field.field, ue, 4, 2)
    print(f"gunther  recovery error (rel) = {eg:.4f}")
    assert np.all(np.isfinite(res_g.u_field.field))
    assert eg < 0.05, "gunther17_2d observer not recovered"

    # reconstruction identity u + (v-u) == v
    assert np.allclose(res_k.u_field.field + res_k.v_minus_u_field.field, f.field, atol=1e-4)
    assert np.allclose(res_g.u_field.field + res_g.v_minus_u_field.field, f.field, atol=1e-3)
    print("2D OK\n")


# ───────────────────────── 3D ─────────────────────────
def build_translating_vortex_3d(X=40, Y=40, Z=40, T=7, v0=0.3):
    bmin = [-2.0, -2.0, -2.0]; bmax = [2.0, 2.0, 2.0]
    tmin, tmax = 0.0, 2.0
    f = UnsteadyVectorField3D(X, Y, Z, T, bmin, bmax, tmin, tmax)
    data = np.zeros((T, Z, Y, X, 3), np.float32)
    xs = np.linspace(bmin[0], bmax[0], X)
    ys = np.linspace(bmin[1], bmax[1], Y)
    zs = np.linspace(bmin[2], bmax[2], Z)
    Zg, Yg, Xg = np.meshgrid(zs, ys, xs, indexing="ij")
    for it in range(T):
        t = tmin + (tmax - tmin) * it / (T - 1)
        xi = Xg - v0 * t
        env = np.exp(-(xi**2 + Yg**2))          # vortex tube along z
        data[it, :, :, :, 0] = -Yg * env + v0
        data[it, :, :, :, 1] = xi * env
        data[it, :, :, :, 2] = 0.0
    f.field = data
    return f, v0


def test_3d():
    print("=== 3D ===")
    f, v0 = build_translating_vortex_3d()
    data = f.field.astype(np.float64)
    ue = (v0, 0.0, 0.0)

    res_k = killing_optimization_3d(f)
    p = res_k.params
    print("killing params (tx,ty,tz,wx,wy,wz) interior mean:", np.round(p[1:-1].mean(0), 4),
          " expected t~(", v0, ",0,0), w~0")
    ek = recovery_error(data, res_k.u_field.field, ue, 3, 3)
    print(f"killing3d recovery error (rel) = {ek:.4f}")
    assert abs(p[1:-1, 0].mean() - v0) < 0.06, "killing_3d translation tx wrong"
    assert np.linalg.norm(p[1:-1, 3:].mean(0)) < 0.06, "killing_3d omega not ~0"
    assert ek < 0.05, "killing_3d observer not recovered"

    res_g = gunther17_optimization_3d(f, neighborhood=3)
    eg = recovery_error(data, res_g.u_field.field, ue, 3, 3)
    print(f"gunther3d recovery error (rel) = {eg:.4f}")
    assert np.all(np.isfinite(res_g.u_field.field))
    assert eg < 0.08, "gunther17_3d observer not recovered"

    assert np.allclose(res_k.u_field.field + res_k.v_minus_u_field.field, f.field, atol=1e-3)
    print("3D OK\n")


if __name__ == "__main__":
    test_2d()
    test_3d()
    print("ALL TESTS PASSED")
