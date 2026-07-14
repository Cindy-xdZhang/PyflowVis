"""FINAL7: proposed(observer pushforward, N regions, overlap+blend) vs baseline at a FIXED param budget.

Supersedes rfo_final6 by decoupling the two things final6 conflated:
  * WHICH N to evaluate  -- give --regions N,... directly (final6 could only reach N via --tau, and the
    threshold-prefix cut cannot reach every N; e.g. cylinder N=5 is unreachable by ANY tau).
  * tau -> N             -- still available via --taus, resolved through dec.cut_adaptive().

Decisive experiment this was written for:  --regions 1  == global observer pushforward, no partition,
one INR with EXACTLY the baseline's parameter count. Never run at the 3.94M budget for cylinder/boussinesq.

Baselines are reused from rfo_final6 (identical INR/epochs/budget, determinism verified: two runs of the
same config agree to 0.00 dB, and final4 vs final6 both give cylinder 49.17 / boussinesq 48.93).
Pass --recompute-baseline to train it in-process instead.

Everything else is held fixed exactly as in rfo_final5/6: FF+ReLU (scale=8, n_freq=256, layers=4),
epochs=2500, lr=1e-3, deterministic cuda, overlap margin=4 / sigma=4.
"""
import os, sys, json, math, argparse, warnings
warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

import numpy as np
from ff_inr import fit_ffrelu, FFReLU                       # noqa: E402  (sets determinism via rfo_decompose_inr)
from rfo_decompose_inr import vpsnr, _np                    # noqa: E402
from rfo_final5 import proposed_overlap                     # noqa: E402
from FLowUtils.referenceFrameDecompose import decompose_reference_frame_2d   # noqa: E402
from FLowUtils.AnalyticalFlowCreator import rotation_four_center             # noqa: E402
from FLowUtils.flowDatasetUtils.NetCDF_AmiraLoader import NetCDFLoader       # noqa: E402

EP, LR, BATCH, NFREQ, LAYERS, SCALE = 2500, 1e-3, 200000, 256, 4, 8.0
DATA = r"C:\Users\xingdi\OneDrive - KAUST\WorkingInProcess\FLowVisAssets\flowData2D"

# baseline PSNR at budget 3.94M / EP=2500 / h=930, from rfo_final6 (in-process, deterministic).
BASELINE_CACHE = {(3.94e6, 2500): {"rfc": 35.24, "cylinder2d": 49.17, "boussinesq": 48.93}}


def hidden_for_params(target):
    """Invert params(FFReLU) = LAYERS*h^2 + (2*NFREQ+1+LAYERS+2)*h + 2  (B is a buffer, not a parameter)."""
    a, b, c = LAYERS, 2 * NFREQ + 1 + LAYERS + 2, 2 - target
    return max(16, int(round((-b + math.sqrt(b * b - 4 * a * c)) / (2 * a))))


def load_field(name):
    if name == "rfc":
        return rotation_four_center((64, 64), 64)
    if name == "cylinder2d":
        f = NetCDFLoader.load_vector_field2d(rf"{DATA}\cylinder2d.nc", 800, 960)
        f.resample2UnsteadyField((128, 320, 80))
        return f
    if name == "boussinesq":
        f = NetCDFLoader.load_vector_field2d(rf"{DATA}\boussinesq.nc", 800, 960)
        f.resample2UnsteadyField((128, 75, 225))
        return f
    raise ValueError(name)


def baseline_ff(field, hidden):
    """One INR fitted directly to v(x,y,t) -- no observer, no partition."""
    v = _np(field); T, Y, X, C = v.shape
    ax = lambda n: np.linspace(-1, 1, n, np.float32)
    gt, gy, gx = np.meshgrid(ax(T), ax(Y), ax(X), indexing="ij")
    coords = np.stack([gx.ravel(), gy.ravel(), gt.ravel()], -1).astype(np.float32)
    flat = v.reshape(-1, C); vmin = flat.min(0); vmax = flat.max(0); sc = np.maximum(vmax - vmin, 1e-8)
    rec, _, npar = fit_ffrelu(coords, (2 * (flat - vmin) / sc - 1).astype(np.float32),
                              EP, LR, BATCH, hidden, LAYERS, NFREQ, SCALE)
    return vpsnr(((rec + 1) * 0.5 * sc + vmin).reshape(T, Y, X, C), v), npar


def run(name, budget, regions, taus, margin, sigma, recompute_baseline, results):
    field = load_field(name)
    dec = decompose_reference_frame_2d(field, k=2)
    print(f"\n===== {name} shape={_np(field).shape}  budget={budget/1e6:.2f}M  margin={margin} sigma={sigma}  "
          f"(benefit={dec.diag['decomposition_benefit']:.3f}) =====", flush=True)

    hb = hidden_for_params(budget)
    cached = BASELINE_CACHE.get((budget, EP), {}).get(name)
    if cached is not None and not recompute_baseline:
        pb, nb, how = cached, None, "cached from rfo_final6"
        print(f"  baseline INR(v)            : PSNR={pb:6.2f}dB  (h={hb})  [{how}]", flush=True)
    else:
        pb, nb = baseline_ff(field, hb)
        print(f"  baseline INR(v)            : PSNR={pb:6.2f}dB  params={nb/1e6:.2f}M (h={hb})  [in-process]", flush=True)

    # resolve the N list: explicit --regions, plus whatever --taus map to (deduped, order preserved)
    Ns = list(regions)
    for tau in taus:
        _, n = dec.cut_adaptive(tau)
        print(f"  [tau={tau}] cut_adaptive -> N={n}", flush=True)
        if n not in Ns:
            Ns.append(n)

    for N in Ns:
        hr = hidden_for_params(budget / max(N, 1))
        pp, tot, ksc = proposed_overlap(field, dec, N, hr, margin=margin, sigma=sigma)
        label = "N=1 (global pushforward, no partition)" if N == 1 else f"N={N}"
        print(f"  proposed {label:38s} h={hr:4d}: PSNR={pp:6.2f}dB  params={tot/1e6:.2f}M "
              f"(+{ksc} killing)  dVsBaseline={pp-pb:+.2f}dB", flush=True)
        results.append(dict(field=name, budget=budget, N=N, hidden=hr, margin=margin, sigma=sigma,
                            baseline_psnr=pb, proposed_psnr=pp, delta=pp - pb,
                            proposed_params=int(tot), killing_scalars=int(ksc), epochs=EP))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", default="cylinder2d,boussinesq")
    ap.add_argument("--regions", default="1", help="comma list of N to evaluate directly (1 = global pushforward)")
    ap.add_argument("--taus", default="", help="comma list of tau, each resolved to N via cut_adaptive")
    ap.add_argument("--budget", type=float, default=3.94e6)
    ap.add_argument("--margin", type=int, default=4)
    ap.add_argument("--sigma", type=float, default=4.0)
    ap.add_argument("--recompute-baseline", action="store_true")
    ap.add_argument("--out", default="final7_results.json")
    a = ap.parse_args()

    regions = [int(x) for x in a.regions.split(",") if x.strip()]
    taus = [float(x) for x in a.taus.split(",") if x.strip()]
    results = []
    for name in [f.strip() for f in a.fields.split(",") if f.strip()]:
        run(name, a.budget, regions, taus, a.margin, a.sigma, a.recompute_baseline, results)
        with open(a.out, "w") as f:                      # write after each field (long run, survive a kill)
            json.dump(results, f, indent=2)

    print("\n" + "=" * 78, flush=True)
    print(f"{'field':>12} | {'N':>3} | {'baseline':>8} | {'proposed':>8} | {'delta':>7}", flush=True)
    print("-" * 78, flush=True)
    for r in results:
        print(f"{r['field']:>12} | {r['N']:>3} | {r['baseline_psnr']:>8.2f} | {r['proposed_psnr']:>8.2f} | "
              f"{r['delta']:>+7.2f}", flush=True)
    print("=" * 78, flush=True)
    print(f"wrote {a.out}", flush=True)
