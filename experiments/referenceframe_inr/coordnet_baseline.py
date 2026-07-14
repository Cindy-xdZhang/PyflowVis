"""Faithful CoordNet (Han & Wang, IEEE TVCG 2023) baseline — SIREN, paper hyperparameters.

Per user's choice (2026-07): baseline = faithful CoordNet at PAPER scale + PAPER training recipe.
  network : CoordNet SIREN residual net, m=64, d=10, omega_0=30, final=sine   (~1.487M params)
  train   : Adam(betas=0.9/0.999), batch=32000, lr=1e-5 (FIXED, no decay), weight_decay=1e-6,
            300 epochs, coords & values normalized to [-1,1], pure MSE.   (docs/coordnet.md sec 5)

Deliberately NO cosine decay / grad clip / best-snapshot tricks — this is the literal paper recipe,
so the baseline is faithful (not something I tuned). We still PRINT the best-so-far PSNR next to the
final one, purely to DIAGNOSE SIREN training stability (does lr=1e-5/300ep converge on our fields?).
The reported baseline number is PSNR_last (training-end weights = paper default).

Smoke-test first (boussinesq) before committing to full runs.
"""
import os, sys, time, json, argparse, warnings
warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
sys.path.insert(0, _ROOT); sys.path.insert(0, _HERE)

import numpy as np, torch
from CoordNetCompression import CoordNet
from rfo_decompose_inr import DEV, _np, vpsnr            # importing sets CUDA determinism + seeds infra
from FLowUtils.flowDatasetUtils.NetCDF_AmiraLoader import NetCDFLoader
from FLowUtils.AnalyticalFlowCreator import rotation_four_center

# ---- paper hyperparameters (docs/coordnet.md sec 5), do not tune ----
M, D, OMEGA = 64, 10, 30.0
LR, BATCH, WD, BETAS, EPOCHS = 1e-5, 32000, 1e-6, (0.9, 0.999), 300
DATA = r"C:\Users\xingdi\OneDrive - KAUST\WorkingInProcess\FLowVisAssets\flowData2D"


def load_field(name):
    if name == "rfc":
        return rotation_four_center((64, 64), 64)
    fn = {"cylinder2d": "cylinder2d.nc", "boussinesq": "boussinesq.nc"}[name]
    shp = {"cylinder2d": (128, 320, 80), "boussinesq": (128, 75, 225)}[name]
    f = NetCDFLoader.load_vector_field2d(rf"{DATA}\{fn}", 800, 960); f.resample2UnsteadyField(shp)
    return f


def coordnet_baseline(name, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    field = load_field(name); v = _np(field); T, Y, X, C = v.shape
    ax = lambda n: np.linspace(-1, 1, n, np.float32)
    gt, gy, gx = np.meshgrid(ax(T), ax(Y), ax(X), indexing="ij")
    coords = np.stack([gx.ravel(), gy.ravel(), gt.ravel()], -1).astype(np.float32)
    flat = v.reshape(-1, C); vmin = flat.min(0); vmax = flat.max(0); sc = np.maximum(vmax - vmin, 1e-8)
    vals = (2 * (flat - vmin) / sc - 1).astype(np.float32)               # values -> [-1,1]
    ct = torch.from_numpy(coords).to(DEV); vt = torch.from_numpy(vals).to(DEV)
    N = ct.shape[0]

    model = CoordNet(3, C, m=M, d=D, omega_0=OMEGA, final_activation="sine").to(DEV)
    npar = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR, betas=BETAS, weight_decay=WD)   # FIXED lr
    bs = min(BATCH, N); steps = max(1, N // bs)
    print(f"\n===== {name} shape={v.shape}  CoordNet m={M} d={D} omega={OMEGA}  params={npar/1e6:.3f}M =====", flush=True)
    print(f"      PAPER recipe: lr={LR} (fixed) batch={BATCH} wd={WD} epochs={EPOCHS}  "
          f"N={N/1e6:.2f}M pts, {steps} steps/epoch", flush=True)

    def full_psnr():
        model.eval()
        with torch.no_grad():
            rec = np.empty((N, C), np.float32)
            for s in range(0, N, 200000):
                rec[s:s + 200000] = model(ct[s:s + 200000]).clamp(-1, 1).cpu().numpy()
        model.train()
        return vpsnr(((rec + 1) * 0.5 * sc + vmin).reshape(T, Y, X, C), v)

    best = -1.0; t0 = time.time()
    for ep in range(EPOCHS):
        perm = torch.randperm(N, device=DEV)
        for s in range(steps):
            idx = perm[s * bs:(s + 1) * bs]
            loss = ((model(ct[idx]) - vt[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 20 == 0 or ep == EPOCHS - 1:
            p = full_psnr(); best = max(best, p)
            print(f"  ep {ep+1:3d}/{EPOCHS}  PSNR(last)={p:6.2f}  best={best:6.2f}  ({time.time()-t0:.0f}s)", flush=True)
    p_last = full_psnr()
    print(f"  >>> {name} CoordNet baseline: PSNR_last={p_last:.2f}dB  PSNR_best={best:.2f}dB  "
          f"params={npar/1e6:.3f}M", flush=True)
    return dict(field=name, psnr_last=float(p_last), psnr_best=float(best), params=int(npar),
                m=M, d=D, omega=OMEGA, lr=LR, batch=BATCH, weight_decay=WD, epochs=EPOCHS)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", default="boussinesq")
    ap.add_argument("--out", default="coordnet_baseline.json")
    a = ap.parse_args()
    res = []
    for n in [x.strip() for x in a.fields.split(",") if x.strip()]:
        res.append(coordnet_baseline(n))
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)
    print(f"\nwrote {a.out}", flush=True)
