"""CoordNet-based proposed: partition (+ optional observer), per-region small CoordNet, paper recipe.

Fair rematch against the FAITHFUL CoordNet baseline (boussinesq 52.37, cylinder 50.68 best-snapshot).
Everything is CoordNet(SIREN) now, same paper recipe as the baseline:
  per-region net : CoordNet, boussinesq N=3 m=34 d=12 (~0.496M x3),  cylinder N=5 m=30 d=9 (~0.30M x5)
                   -> total ~= 1.487M == baseline (fair).
  train          : Adam(0.9/0.999), lr=1e-5 FIXED, batch=32000, wd=1e-6, 300 epochs, best-snapshot
                   (best matches the baseline's best取值 the user chose).
  reconstruct    : overlap(margin=4)+gaussian blend(sigma=4), same as the earlier FF experiments.

Two configs per field, to isolate the observer:
  * pure partition (use_observer=False): each region fits its RAW v, no pushforward.
  * partition+observer (use_observer=True): each region pushforward to its moving frame, then fit.

Prints each config's PSNR and the gap vs the faithful CoordNet baseline.
"""
import os, sys, copy, json, argparse, warnings
warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
sys.path.insert(0, _ROOT); sys.path.insert(0, _HERE)

import numpy as np, torch
from scipy.ndimage import binary_dilation, gaussian_filter
import rfo_decompose_inr as DEC
from CoordNetCompression import CoordNet
from rfo_decompose_inr import DEV, _np, vpsnr, fit_region
from FLowUtils.referenceFrameDecompose import decompose_reference_frame_2d
from FLowUtils.flowDatasetUtils.NetCDF_AmiraLoader import NetCDFLoader
from FLowUtils.AnalyticalFlowCreator import rotation_four_center

# paper recipe (same as coordnet_baseline.py)
LR, BATCH, WD, EPOCHS, OMEGA = 1e-5, 32000, 1e-6, 300, 30.0
PER_REGION = {"boussinesq": (3, 34, 12), "cylinder2d": (5, 30, 9)}   # N, m, d  (~1.487M total)
BASELINE_BEST = {"boussinesq": 52.37, "cylinder2d": 50.68}           # faithful CoordNet, best-snapshot
DATA = r"C:\Users\xingdi\OneDrive - KAUST\WorkingInProcess\FLowVisAssets\flowData2D"


def load_field(name):
    if name == "rfc":
        return rotation_four_center((64, 64), 64)
    fn = {"cylinder2d": "cylinder2d.nc", "boussinesq": "boussinesq.nc"}[name]
    shp = {"cylinder2d": (128, 320, 80), "boussinesq": (128, 75, 225)}[name]
    f = NetCDFLoader.load_vector_field2d(rf"{DATA}\{fn}", 800, 960); f.resample2UnsteadyField(shp)
    return f


def fit_coordnet_paper(coords, values, epochs, m, d, lr, batch, seed=0, omega_0=30.0):
    """CoordNet, paper recipe (fixed lr, wd, no decay/clip) + best-snapshot on normalized full-data loss.
    Signature matches rfo_decompose_inr.fit_inr so fit_region can call it via monkeypatch."""
    torch.manual_seed(seed); np.random.seed(seed)
    N, C = values.shape
    ct = torch.from_numpy(coords).to(DEV); vt = torch.from_numpy(values).to(DEV)
    model = CoordNet(3, C, m=m, d=d, omega_0=omega_0, final_activation="sine").to(DEV)
    npar = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=WD)  # fixed lr
    bs = min(batch, N); steps = max(1, N // bs)
    best = float("inf"); bstate = copy.deepcopy(model.state_dict()); ev = max(1, epochs // 30)
    for ep in range(epochs):
        perm = torch.randperm(N, device=DEV)
        for s in range(steps):
            idx = perm[s * bs:(s + 1) * bs]
            loss = ((model(ct[idx]) - vt[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()          # no clip, no decay (paper)
        if (ep + 1) % ev == 0 or ep == epochs - 1:
            with torch.no_grad():
                fl = sum(((model(ct[s:s+400000]) - vt[s:s+400000]) ** 2).sum().item()
                         for s in range(0, N, 400000)) / N
            if fl < best: best = fl; bstate = copy.deepcopy(model.state_dict())
    model.load_state_dict(bstate); model.eval()
    rec = np.empty((N, C), np.float32)
    with torch.no_grad():
        for s in range(0, N, 200000):
            rec[s:s + 200000] = model(ct[s:s + 200000]).clamp(-1, 1).cpu().numpy()
    return rec, 0.0, npar


def proposed(field, dec, N, m, d, use_observer, margin=4, sigma=4.0):
    DEC.fit_inr = fit_coordnet_paper                                # route fit_region through CoordNet
    v = _np(field); T, Y, X, _ = v.shape
    xph = field.domainMinBoundary[0] + field.gridInterval[0] * np.arange(X)
    yph = field.domainMinBoundary[1] + field.gridInterval[1] * np.arange(Y)
    times = np.array([field.getPhysicalTime(i) for i in range(T)], np.float64)
    labels = dec.cut(n_regions=N); obs = dec.region_observers(labels)
    accum_v = np.zeros_like(v); accum_w = np.zeros((Y, X)); tot = 0; ksc = 0
    for rid, coeff in obs.items():
        coeff_use = coeff if use_observer else np.zeros_like(coeff)
        core = labels == rid
        tmask = binary_dilation(core, iterations=margin)
        wgt = gaussian_filter(core.astype(np.float64), sigma) * tmask
        ys, xs, vx, vy, _, npar = fit_region(v, tmask, coeff_use, times, xph, yph, EPOCHS, m, d, LR, BATCH)
        wp = wgt[ys, xs]
        accum_v[:, ys, xs, 0] += wp[None] * vx; accum_v[:, ys, xs, 1] += wp[None] * vy
        accum_w[ys, xs] += wp; tot += npar; ksc += (coeff.size if use_observer else 0)
    v_recon = accum_v / np.clip(accum_w, 1e-8, None)[None, :, :, None]
    return vpsnr(v_recon, v), tot, ksc


def run(name, results):
    N, m, d = PER_REGION[name]
    field = load_field(name); dec = decompose_reference_frame_2d(field, k=2)
    base = BASELINE_BEST[name]
    print(f"\n===== {name}  N={N} per-region CoordNet m={m}/d={d}  vs faithful baseline {base:.2f}dB "
          f"(benefit={dec.diag['decomposition_benefit']:.3f}) =====", flush=True)
    for use_obs in (False, True):
        p, tot, ksc = proposed(field, dec, N, m, d, use_obs)
        tag = "partition+observer" if use_obs else "pure partition    "
        print(f"  {tag}: PSNR={p:6.2f}dB  params={tot/1e6:.3f}M (+{ksc} k)  "
              f"vsBaseline={p-base:+.2f}dB", flush=True)
        results.append(dict(field=name, N=N, m=m, d=d, observer=use_obs, epochs=EPOCHS,
                            psnr=float(p), baseline=base, delta=float(p - base),
                            params=int(tot), killing=int(ksc)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", default="boussinesq")
    ap.add_argument("--out", default="coordnet_proposed.json")
    a = ap.parse_args()
    results = []
    for name in [x.strip() for x in a.fields.split(",") if x.strip()]:
        run(name, results)
        with open(a.out, "w") as f:
            json.dump(results, f, indent=2)
    print("\n" + "=" * 78, flush=True)
    for r in results:
        obs = "obs " if r["observer"] else "noobs"
        print(f"  {r['field']:>11} {obs}: PSNR={r['psnr']:.2f}  vsBaseline={r['delta']:+.2f}dB", flush=True)
    print(f"wrote {a.out}", flush=True)
