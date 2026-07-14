"""
STRICT, DETERMINISTIC, controlled-variable experiment (first-order killing, 3-DOF).

Fixed INR: CoordNet m=64 d=4, deterministic (see rfo_decompose_inr module top).
Compares, per field:
  baseline : INR(v)                          -- direct fit of the raw field
  global   : whole-field 3-DOF killing observer (NO tree) -> pushforward observed field -> INR
  cut=1    : referenceFrameDecompose.cut(1) observer (WITH tree) -> pushforward -> INR
global and cut=1 use the IDENTICAL pushforward+INR code path (fit_region), differing only in where
the observer comes from. They MUST give the same PSNR (proving cut=1 reproduces the no-tree global).
"""
import sys, math, argparse
sys.path.insert(0, r"C:\Users\xingdi\sources\PyflowVis")
sys.path.insert(0, r"C:\Users\xingdi\AppData\Local\Temp\claude\C--Users-xingdi-sources-PyflowVis\5b7dbb49-2a88-462c-b42d-687217cb2b30\scratchpad")
import numpy as np
# rfo_decompose_inr sets determinism at import (cudnn.deterministic, use_deterministic_algorithms, CUBLAS cfg)
from rfo_decompose_inr import fit_inr, fit_region, _np, vpsnr, DEV
from FLowUtils.ReferenceFrameOptimization import _as_numpy, _dvdt, _coords_2d
from FLowUtils.referenceFrameDecompose import decompose_reference_frame_2d
from FLowUtils.AnalyticalFlowCreator import rotation_four_center
from FLowUtils.flowDatasetUtils.NetCDF_AmiraLoader import NetCDFLoader

M, D, LR, BATCH = 64, 4, 3e-4, 200000   # fixed INR


def global_killing_observer_2d(field):
    """Whole-field first-order (3-DOF) killing observer, per frame. NO tree. Returns (T,3)."""
    data = _as_numpy(field); T, Y, X, _ = data.shape
    dx, dy = float(field.gridInterval[0]), float(field.gridInterval[1]); dt = float(field.timeInterval)
    Xg, Yg = _coords_2d(field); Xp = np.stack([-Yg, Xg], -1)
    vt = _dvdt(data, dt)
    obs = np.zeros((T, 3))
    for t in range(T):
        sl = data[t]
        dv_dx = np.gradient(sl, dx, axis=1); dv_dy = np.gradient(sl, dy, axis=0)
        J = np.stack([dv_dx, dv_dy], -1)
        Vp = np.stack([-sl[:, :, 1], sl[:, :, 0]], -1)
        A = np.empty((Y, X, 2, 3)); A[..., 0] = dv_dx; A[..., 1] = dv_dy
        A[..., 2] = np.einsum("yxik,yxk->yxi", J, Xp) - Vp
        b = -vt[t]
        S = np.einsum("yxki,yxkj->ij", A, A); r = np.einsum("yxki,yxk->i", A, b)
        obs[t] = np.linalg.solve(S + 1e-9 * np.trace(S) / 3 * np.eye(3), r)
    return obs


def baseline_inr(field, epochs):
    v = _np(field); T, Y, X, C = v.shape
    ax = lambda n: np.linspace(-1, 1, n, np.float32)
    gt, gy, gx = np.meshgrid(ax(T), ax(Y), ax(X), indexing="ij")
    coords = np.stack([gx.ravel(), gy.ravel(), gt.ravel()], -1).astype(np.float32)
    flat = v.reshape(-1, C); vmin = flat.min(0); vmax = flat.max(0); sc = np.maximum(vmax - vmin, 1e-8)
    vals = (2 * (flat - vmin) / sc - 1).astype(np.float32)
    rec, tt, npar = fit_inr(coords, vals, epochs, M, D, LR, BATCH)
    recon = ((rec + 1) * 0.5 * sc + vmin).reshape(T, Y, X, C)
    return vpsnr(recon, v), npar, tt


def pushforward_inr_whole(field, observer, epochs):
    """Whole-field pushforward with given observer -> INR (uses the SAME fit_region as decomposition)."""
    v = _np(field); T, Y, X, _ = v.shape
    xph = field.domainMinBoundary[0] + field.gridInterval[0] * np.arange(X)
    yph = field.domainMinBoundary[1] + field.gridInterval[1] * np.arange(Y)
    times = np.array([field.getPhysicalTime(i) for i in range(T)], np.float64)
    mask = np.ones((Y, X), bool)
    ys, xs, vx, vy, tt, npar = fit_region(v, mask, observer, times, xph, yph, epochs, M, D, LR, BATCH)
    v_recon = np.zeros_like(v); v_recon[:, ys, xs, 0] = vx; v_recon[:, ys, xs, 1] = vy
    return vpsnr(v_recon, v), npar, tt


def run(name, field, epochs):
    print(f"\n===== {name}  shape={_np(field).shape}  (INR m={M} d={D}, {epochs}ep, deterministic) =====", flush=True)
    pb, nb, tb = baseline_inr(field, epochs)
    print(f"  baseline INR(v)          : PSNR={pb:6.2f}dB  params={nb/1e6:.2f}M  ({tb:.0f}s)", flush=True)

    obs_g = global_killing_observer_2d(field)
    pg, ng, tg = pushforward_inr_whole(field, obs_g, epochs)
    print(f"  global (no tree)         : PSNR={pg:6.2f}dB  params={ng/1e6:.2f}M  ({tg:.0f}s)", flush=True)

    dec = decompose_reference_frame_2d(field, k=2)
    obs_c1 = list(dec.region_observers(dec.cut(n_regions=1)).values())[0]
    pc, nc, tc = pushforward_inr_whole(field, obs_c1, epochs)
    print(f"  cut=1 (with tree)        : PSNR={pc:6.2f}dB  params={nc/1e6:.2f}M  ({tc:.0f}s)", flush=True)

    print(f"  >> observer max|global - cut1| = {np.abs(obs_g - obs_c1).max():.2e}   PSNR |global-cut1| = {abs(pg-pc):.4f} dB", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="rfc", choices=["rfc", "cylinder", "both"])
    ap.add_argument("--epochs", type=int, default=600)
    a = ap.parse_args()
    print("device:", DEV, flush=True)
    if a.field in ("rfc", "both"):
        run("rfc", rotation_four_center((64, 64), 64), a.epochs)
    if a.field in ("cylinder", "both"):
        cf = NetCDFLoader.load_vector_field2d(r"C:\Users\xingdi\OneDrive - KAUST\WorkingInProcess\FLowVisAssets\flowData2D\cylinder2d.nc", 800, 960)
        cf.resample2UnsteadyField((128, 320, 80))
        run("cylinder2d", cf, a.epochs)
