"""
Hierarchical reference-frame decomposition  x  per-region INR of the (pushforward) observed field.

Pipeline:
  1. decompose_reference_frame_2d(field) -> merge tree.
  2. cut(N) -> N regions, each with its own killing observer (a,b,c)_t.
  3. Per region: integrate its observer motion (theta_t=∫c, D_t=∫R(-theta)v0), warp region pixels to the
     CO-MOVING frame, and fit ONE INR to that region's pushforward observed field  R(-theta)(v-u).
  4. Reconstruct each region (rotate back + u) -> full v_recon.  PSNR in original v space.
  cut=1 == whole-field pushforward (the earlier "warp" experiment).

This pass: "big enough" INR per region -> PSNR ceiling. (Future: constrain INR size.)
"""
import sys, time, math, argparse, copy
sys.path.insert(0, r"C:\Users\xingdi\sources\PyflowVis")
import numpy as np, torch
from CoordNetCompression import CoordNet
from FLowUtils.AnalyticalFlowCreator import rotation_four_center
from FLowUtils.flowDatasetUtils.NetCDF_AmiraLoader import NetCDFLoader
from FLowUtils.referenceFrameDecompose import decompose_reference_frame_2d

# ── determinism: same inputs+seed -> same output (cuda matmul is non-deterministic by default) ──
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _np(f):
    d = f.field
    if hasattr(d, "detach"): d = d.detach().cpu().numpy()
    return np.asarray(d, np.float64)

def integrate_frame_motion(coeff, times):
    """coeff (T,3)=(a,b,c); times (T,). Returns theta(T,), D(T,2)=∫R(-theta)v0 (trapezoid)."""
    T = len(times); a, b, c = coeff[:, 0], coeff[:, 1], coeff[:, 2]
    theta = np.zeros(T); D = np.zeros((T, 2))
    def Rm(th, vx, vy):
        ct, st = math.cos(-th), math.sin(-th); return ct * vx - st * vy, st * vx + ct * vy
    for k in range(1, T):
        dt = times[k] - times[k - 1]
        theta[k] = theta[k - 1] + 0.5 * (c[k] + c[k - 1]) * dt
        f0 = Rm(theta[k - 1], a[k - 1], b[k - 1]); f1 = Rm(theta[k], a[k], b[k])
        D[k, 0] = D[k - 1, 0] + 0.5 * (f0[0] + f1[0]) * dt
        D[k, 1] = D[k - 1, 1] + 0.5 * (f0[1] + f1[1]) * dt
    return theta, D


def fit_inr(coords, values, epochs, m, d, lr, batch, seed=0, omega_0=30.0):
    """coords (N,3) in [-1,1], values (N,2) in [-1,1]. Returns recon_values (N,2) norm, train_time, n_params."""
    torch.manual_seed(seed); np.random.seed(seed)
    N, C = values.shape
    ct = torch.from_numpy(coords).to(DEV); vt = torch.from_numpy(values).to(DEV)
    model = CoordNet(3, C, m=m, d=d, omega_0=omega_0, final_activation="sine").to(DEV)
    npar = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bs = min(batch, N); steps = max(1, N // bs); lr_f = lr * 0.1
    best = float("inf"); best_state = copy.deepcopy(model.state_dict()); t0 = time.time()
    eval_every = max(1, epochs // 50)     # full-data best-snapshot check ~50x (cheap, robust)
    for ep in range(epochs):
        cur = lr_f + 0.5 * (lr - lr_f) * (1 + math.cos(math.pi * ep / max(1, epochs - 1)))
        for g in opt.param_groups: g["lr"] = cur
        perm = torch.randperm(N, device=DEV)
        for s in range(steps):
            idx = perm[s * bs:(s + 1) * bs]
            loss = ((model(ct[idx]) - vt[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # stabilize high-omega SIREN
            opt.step()
        if (ep + 1) % eval_every == 0 or ep == epochs - 1:
            with torch.no_grad():   # best snapshot on full-data eval (mini-batch loss too noisy)
                fl = sum(((model(ct[s:s+400000]) - vt[s:s+400000]) ** 2).sum().item() for s in range(0, N, 400000)) / N
            if fl < best: best = fl; best_state = copy.deepcopy(model.state_dict())
    tt = time.time() - t0; model.load_state_dict(best_state); model.eval()
    rec = np.empty((N, C), np.float32)
    with torch.no_grad():
        for s in range(0, N, 200000): rec[s:s + 200000] = model(ct[s:s + 200000]).clamp(-1, 1).cpu().numpy()
    return rec, tt, npar


def fit_region(data, mask, coeff, times, xph, yph, epochs, m, d, lr, batch, omega_0=30.0):
    """Fit one region's pushforward observed field; write reconstruction into a full-size v_recon (masked pixels)."""
    T = data.shape[0]
    ys_idx, xs_idx = np.where(mask)                     # region pixel indices (npix,)
    xr = xph[xs_idx]; yr = yph[ys_idx]                  # physical coords (npix,)
    theta, D = integrate_frame_motion(coeff, times)
    cth = np.cos(-theta)[:, None]; sth = np.sin(-theta)[:, None]     # (T,1)  R(-theta)
    # warp coords ξ = R(-theta)(x,y) - D
    xi_x = cth * xr[None] + (-sth) * yr[None] - D[:, 0:1]            # (T,npix)
    xi_y = sth * xr[None] + cth * yr[None] - D[:, 1:2]
    # relative velocity v-u_r, then rotate to co-moving: v_co = R(-theta)(v-u)
    vreg = data[:, ys_idx, xs_idx, :]                               # (T,npix,2)
    a = coeff[:, 0][:, None]; b = coeff[:, 1][:, None]; c = coeff[:, 2][:, None]
    ux = a - c * yr[None]; uy = b + c * xr[None]                    # (T,npix)
    mx = vreg[..., 0] - ux; my = vreg[..., 1] - uy
    vco_x = cth * mx + (-sth) * my; vco_y = sth * mx + cth * my     # (T,npix)
    # coords & values (T*npix, .)
    tnorm = np.linspace(-1, 1, T, dtype=np.float32)[:, None] * np.ones((1, xr.size), np.float32)
    xi = np.stack([xi_x.ravel(), xi_y.ravel(), tnorm.ravel()], -1).astype(np.float32)
    # normalize ξ (first 2 dims) to [-1,1]
    lo = xi[:, :2].min(0); hi = xi[:, :2].max(0); sc = np.maximum(hi - lo, 1e-8)
    xi[:, :2] = (2 * (xi[:, :2] - lo) / sc - 1)
    vco = np.stack([vco_x.ravel(), vco_y.ravel()], -1).astype(np.float32)
    vmin = vco.min(0); vmax = vco.max(0); vsc = np.maximum(vmax - vmin, 1e-8)
    vco_n = (2 * (vco - vmin) / vsc - 1).astype(np.float32)
    rec_n, tt, npar = fit_inr(xi, vco_n, epochs, m, d, lr, batch, omega_0=omega_0)
    vco_r = (rec_n + 1) * 0.5 * vsc + vmin                          # (T*npix,2) -> physical co-moving
    vco_r = vco_r.reshape(T, xr.size, 2)
    # rotate back R(theta) + u_r
    cB = np.cos(theta)[:, None]; sB = np.sin(theta)[:, None]
    vx = cB * vco_r[..., 0] + (-sB) * vco_r[..., 1] + ux
    vy = sB * vco_r[..., 0] + cB * vco_r[..., 1] + uy
    return ys_idx, xs_idx, vx, vy, tt, npar


def vpsnr(recon, gt):
    rng = float(gt.max() - gt.min()); mse = float(np.mean((recon - gt) ** 2))
    return float("inf") if mse <= 0 else 20 * math.log10(rng) - 10 * math.log10(mse)


def run_field(name, vfield, cuts, epochs, m, d, lr, batch):
    v = _np(vfield); T, Y, X, _ = v.shape
    xph = vfield.domainMinBoundary[0] + vfield.gridInterval[0] * np.arange(X)
    yph = vfield.domainMinBoundary[1] + vfield.gridInterval[1] * np.arange(Y)
    times = np.array([vfield.getPhysicalTime(i) for i in range(T)], np.float64)
    print(f"\n===== {name}  shape={v.shape} =====", flush=True)
    dec = decompose_reference_frame_2d(vfield, k=2)
    print(f"  decompose diag: benefit={dec.diag['decomposition_benefit']:.3f}", flush=True)
    for N in cuts:
        labels = dec.cut(n_regions=N); nreg = len(np.unique(labels))
        obs = dec.region_observers(labels)
        v_recon = np.zeros_like(v); tot_par = 0; tot_t = 0.0; ksc = 0
        for rid, coeff in obs.items():
            mask = labels == rid
            ys_idx, xs_idx, vx, vy, tt, npar = fit_region(v, mask, coeff, times, xph, yph, epochs, m, d, lr, batch)
            v_recon[:, ys_idx, xs_idx, 0] = vx; v_recon[:, ys_idx, xs_idx, 1] = vy
            tot_par += npar; tot_t += tt; ksc += coeff.size
        p = vpsnr(v_recon, v)
        print(f"  cut={N} (regions={nreg}): PSNR={p:6.2f}dB  INR_params={tot_par/1e6:.2f}M "
              f"(+{ksc} killing)  train={tot_t:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", required=True, choices=["rfc", "cylinder", "boussinesq"])
    ap.add_argument("--cuts", default="1,3,6")
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--m", type=int, default=64); ap.add_argument("--d", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4); ap.add_argument("--batch", type=int, default=200000)
    a = ap.parse_args(); print("device:", DEV, flush=True)
    cuts = [int(x) for x in a.cuts.split(",")]
    if a.field == "rfc":
        vf = rotation_four_center((64, 64), 64)
    elif a.field == "cylinder":
        vf = NetCDFLoader.load_vector_field2d(r"C:\Users\xingdi\OneDrive - KAUST\WorkingInProcess\FLowVisAssets\flowData2D\cylinder2d.nc", 800, 960)
        vf.resample2UnsteadyField((128, 320, 80))
    else:
        vf = NetCDFLoader.load_vector_field2d(r"C:\Users\xingdi\OneDrive - KAUST\WorkingInProcess\FLowVisAssets\flowData2D\boussinesq.nc", 800, 960)
        vf.resample2UnsteadyField((128, 75, 225))
    run_field(a.field, vf, cuts, a.epochs, a.m, a.d, a.lr, a.batch)
