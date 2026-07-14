"""FINAL8: ADAPTIVE per-region parameter allocation (docs section 7, the top pivot).

rfo_final5/6/7 split a fixed budget P EQUALLY over N regions (every region-INR gets hidden=hidden_for(P/N)).
That wastes capacity on trivial regions (e.g. boussinesq's near-constant background) and starves the hard
ones (the plume). Here each region r instead gets a share  P_r = P * w_r / sum(w)  by a complexity weight
w_r, then hidden_r = hidden_for(P_r). Same TOTAL budget P as the baseline -> still a fair comparison.

Weight options (--weight):
  uniform   w_r = 1                      (== rfo_final5, the equal-split control)
  pixels    w_r = |region|              (bigger area -> more params)
  residual  w_r = observed-td residual  (harder-to-explain region -> more params)
  respix    w_r = residual * |region|
A --gamma exponent smooths/sharpens weights: w_r <- w_r**gamma (gamma<1 flattens, avoids starving small
regions to hidden<32; gamma=0 == uniform).

Constraints honored: epochs <= 1000 (user hard rule, 2026-07-10); FF+ReLU fixed (scale=8,n_freq=256,
layers=4); deterministic cuda. Baseline is RETRAINED in-process at the SAME epochs (the 2500ep cache from
final6 is void under the <=1000ep rule).

Usage:
    python rfo_final8.py --dry-run                      # CPU: print per-region alloc, no training
    python rfo_final8.py --fields boussinesq --weight residual --gamma 0.5
"""
import os, sys, json, math, argparse, warnings
warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
sys.path.insert(0, _ROOT); sys.path.insert(0, _HERE)

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter
import rfo_decompose_inr as DEC
from ff_inr import fit_ffrelu, FFReLU
from rfo_decompose_inr import fit_region, vpsnr, _np
from FLowUtils.referenceFrameDecompose import decompose_reference_frame_2d
from FLowUtils.AnalyticalFlowCreator import rotation_four_center
from FLowUtils.flowDatasetUtils.NetCDF_AmiraLoader import NetCDFLoader

EP, LR, BATCH, NFREQ, LAYERS, SCALE = 1000, 1e-3, 200000, 256, 4, 8.0    # EP<=1000 hard rule
DATA = r"C:\Users\xingdi\OneDrive - KAUST\WorkingInProcess\FLowVisAssets\flowData2D"
BEST_N = {"rfc": 1, "cylinder2d": 5, "boussinesq": 3}                    # each field's best equal-split N so far


def params_of(hidden):
    return LAYERS * hidden * hidden + (2 * NFREQ + 1 + LAYERS + 2) * hidden + 2


def hidden_for_params(target):
    a, b, c = LAYERS, 2 * NFREQ + 1 + LAYERS + 2, 2 - target
    return max(16, int(round((-b + math.sqrt(b * b - 4 * a * c)) / (2 * a))))


def load_field(name):
    if name == "rfc":
        return rotation_four_center((64, 64), 64)
    fn = {"cylinder2d": "cylinder2d.nc", "boussinesq": "boussinesq.nc"}[name]
    shp = {"cylinder2d": (128, 320, 80), "boussinesq": (128, 75, 225)}[name]
    f = NetCDFLoader.load_vector_field2d(rf"{DATA}\{fn}", 800, 960); f.resample2UnsteadyField(shp)
    return f


def region_weights(dec, labels, kind, gamma):
    """Return {rid: weight}, normalized-agnostic (only ratios matter). gamma exponent smooths."""
    rids = [int(r) for r in np.unique(dec._leaf_labels(labels))]
    if kind == "uniform":
        w = {r: 1.0 for r in rids}
    elif kind == "pixels":
        w = {r: float((labels == r).sum()) for r in rids}
    elif kind == "residual":
        w = {r: float(v) for r, v in dec.region_residuals(labels).items()}
    elif kind == "respix":
        res = dec.region_residuals(labels)
        w = {r: float(res[r]) * float((labels == r).sum()) for r in rids}
    else:
        raise ValueError(kind)
    # guard: residual can be ~0 for a rigid region; floor so gamma/log stay finite
    mx = max(w.values()) or 1.0
    w = {r: max(v, 1e-6 * mx) for r, v in w.items()}
    if gamma != 1.0:
        w = {r: v ** gamma for r, v in w.items()}
    return w


def allocate(dec, labels, budget, kind, gamma):
    """Split `budget` params over regions by weight; return {rid: hidden}, total_params."""
    w = region_weights(dec, labels, kind, gamma)
    sw = sum(w.values())
    alloc = {r: hidden_for_params(budget * w[r] / sw) for r in w}
    tot = sum(params_of(h) for h in alloc.values())
    return alloc, tot, w


def proposed_adaptive(field, dec, N, alloc, margin=4, sigma=4.0, use_observer=True):
    """Like rfo_final5.proposed_overlap but each region uses its OWN hidden from `alloc`.

    use_observer=False is the ABLATION: pass a zero killing coeff so fit_region does no pushforward
    (theta=0, D=0, u=0 -> it fits the region's RAW v directly). Same partition/alloc/overlap otherwise,
    so proposed(use_observer=True) vs proposed(use_observer=False) isolates the observer's contribution
    from the pure "split into easier-to-train sub-INRs" effect.
    """
    v = _np(field); T, Y, X, _ = v.shape
    xph = field.domainMinBoundary[0] + field.gridInterval[0] * np.arange(X)
    yph = field.domainMinBoundary[1] + field.gridInterval[1] * np.arange(Y)
    times = np.array([field.getPhysicalTime(i) for i in range(T)], np.float64)
    labels = dec.cut(n_regions=N); obs = dec.region_observers(labels)
    accum_v = np.zeros_like(v); accum_w = np.zeros((Y, X)); tot = 0; ksc = 0
    for rid, coeff in obs.items():
        coeff_use = coeff if use_observer else np.zeros_like(coeff)
        h = alloc[int(rid)]
        DEC.fit_inr = (lambda hh: (lambda coords, values, epochs, m, d, lr, batch, seed=0, omega_0=None:
                                   fit_ffrelu(coords, values, epochs, lr, batch, hh, LAYERS, NFREQ, SCALE, seed)))(h)
        core = labels == rid
        tmask = binary_dilation(core, iterations=margin)
        wgt = gaussian_filter(core.astype(np.float64), sigma) * tmask
        ys, xs, vx, vy, _, npar = fit_region(v, tmask, coeff_use, times, xph, yph, EP, 0, 0, LR, BATCH)
        wp = wgt[ys, xs]
        accum_v[:, ys, xs, 0] += wp[None] * vx; accum_v[:, ys, xs, 1] += wp[None] * vy
        accum_w[ys, xs] += wp; tot += npar; ksc += (coeff.size if use_observer else 0)
    v_recon = accum_v / np.clip(accum_w, 1e-8, None)[None, :, :, None]
    return vpsnr(v_recon, v), tot, ksc


def baseline_ff(field, hidden):
    v = _np(field); T, Y, X, C = v.shape
    ax = lambda n: np.linspace(-1, 1, n, np.float32)
    gt, gy, gx = np.meshgrid(ax(T), ax(Y), ax(X), indexing="ij")
    coords = np.stack([gx.ravel(), gy.ravel(), gt.ravel()], -1).astype(np.float32)
    flat = v.reshape(-1, C); vmin = flat.min(0); vmax = flat.max(0); sc = np.maximum(vmax - vmin, 1e-8)
    rec, _, npar = fit_ffrelu(coords, (2 * (flat - vmin) / sc - 1).astype(np.float32),
                              EP, LR, BATCH, hidden, LAYERS, NFREQ, SCALE)
    return vpsnr(((rec + 1) * 0.5 * sc + vmin).reshape(T, Y, X, C), v), npar


def run(name, budget, N, weights, gamma, margin, sigma, dry, use_observer, skip_baseline, results):
    field = load_field(name); v = _np(field)
    dec = decompose_reference_frame_2d(field, k=2)
    labels = dec.cut(n_regions=N)
    obs_tag = "obs" if use_observer else "NOobs(ablation)"
    print(f"\n===== {name} shape={v.shape}  budget={budget/1e6:.2f}M  N={N}  EP={EP}  observer={use_observer}  "
          f"(benefit={dec.diag['decomposition_benefit']:.3f}) =====", flush=True)

    # show the allocation for every weighting (cheap, CPU) so we can sanity-check before training
    for kind in weights:
        alloc, tot, w = allocate(dec, labels, budget, kind, gamma)
        sw = sum(w.values())
        detail = "  ".join(f"r{r}:h{alloc[r]}({100*w[r]/sw:.0f}%,{(labels==r).sum()}px)" for r in sorted(alloc))
        print(f"  [alloc {kind:8s} gamma={gamma}] total={tot/1e6:.2f}M  {detail}", flush=True)
    if dry:
        return

    if skip_baseline:
        pb = float("nan")
        print(f"  baseline INR(v)               : skipped (--skip-baseline; compare against the cached value)", flush=True)
    else:
        hb = hidden_for_params(budget)
        pb, nb = baseline_ff(field, hb)
        print(f"  baseline INR(v)               h={hb:4d}: PSNR={pb:6.2f}dB  params={nb/1e6:.2f}M", flush=True)

    for kind in weights:
        alloc, tot, _ = allocate(dec, labels, budget, kind, gamma)
        pp, ptot, ksc = proposed_adaptive(field, dec, N, alloc, margin, sigma, use_observer)
        base = "uniform (=equal split)" if kind == "uniform" else f"adaptive:{kind}"
        tag = f"{base} [{obs_tag}]"
        print(f"  proposed {tag:40s}: PSNR={pp:6.2f}dB  params={ptot/1e6:.2f}M (+{ksc} k)  "
              f"dVsBaseline={pp-pb:+.2f}dB", flush=True)
        results.append(dict(field=name, N=N, weight=kind, gamma=gamma, epochs=EP, budget=budget,
                            observer=use_observer, baseline_psnr=pb, proposed_psnr=pp, delta=pp - pb,
                            proposed_params=int(ptot), killing=int(ksc)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", default="cylinder2d,boussinesq")
    ap.add_argument("--N", type=int, default=0, help="regions; 0 = each field's BEST_N")
    ap.add_argument("--weights", default="uniform,pixels,residual,respix")
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--budget", type=float, default=3.94e6)
    ap.add_argument("--margin", type=int, default=4); ap.add_argument("--sigma", type=float, default=4.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-observer", action="store_true", help="ABLATION: pure spatial partition, no observer pushforward")
    ap.add_argument("--skip-baseline", action="store_true", help="don't retrain baseline (compare against cached value)")
    ap.add_argument("--out", default="final8_results.json")
    a = ap.parse_args()

    weights = [w.strip() for w in a.weights.split(",") if w.strip()]
    results = []
    for name in [f.strip() for f in a.fields.split(",") if f.strip()]:
        N = a.N or BEST_N[name]
        run(name, a.budget, N, weights, a.gamma, a.margin, a.sigma, a.dry_run,
            not a.no_observer, a.skip_baseline, results)
        if not a.dry_run:
            with open(a.out, "w") as f:
                json.dump(results, f, indent=2)

    if results:
        print("\n" + "=" * 86, flush=True)
        print(f"{'field':>12} | {'N':>2} | {'weight':>9} | {'baseline':>8} | {'proposed':>8} | {'delta':>7}", flush=True)
        print("-" * 86, flush=True)
        for r in results:
            print(f"{r['field']:>12} | {r['N']:>2} | {r['weight']:>9} | {r['baseline_psnr']:>8.2f} | "
                  f"{r['proposed_psnr']:>8.2f} | {r['delta']:>+7.2f}", flush=True)
        print("=" * 86, flush=True)
        print(f"wrote {a.out}", flush=True)
