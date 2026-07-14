"""FINAL: adaptive #regions (cut_adaptive) + fixed budget, baseline RECOMPUTED in-process (no hardcode).
FF+ReLU fixed (scale=8,n_freq=256,layers=4), deterministic, EP=2500. Same budget P on both sides:
  baseline = 1 INR hidden=hidden_for(P);  proposed = N adaptive regions, each INR hidden=hidden_for(P/N),
  overlap+blend reconstruction. rfc / cylinder / boussinesq."""
import sys, argparse, math, numpy as np, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, r"C:\Users\xingdi\sources\PyflowVis")
sys.path.insert(0, r"C:\Users\xingdi\AppData\Local\Temp\claude\C--Users-xingdi-sources-PyflowVis\5b7dbb49-2a88-462c-b42d-687217cb2b30\scratchpad")
from ff_inr import fit_ffrelu, FFReLU
from rfo_decompose_inr import vpsnr, _np
from rfo_final5 import proposed_overlap
from FLowUtils.referenceFrameDecompose import decompose_reference_frame_2d
from FLowUtils.AnalyticalFlowCreator import rotation_four_center
from FLowUtils.flowDatasetUtils.NetCDF_AmiraLoader import NetCDFLoader
EP, LR, BATCH, NFREQ, LAYERS = 2500, 1e-3, 200000, 256, 4

def hidden_for_params(target):
    # params(FFReLU) = LAYERS*h² + (2*NFREQ+1+LAYERS+2)*h + 2  (B is a buffer, excluded)
    a, b, c = LAYERS, 2*NFREQ + 1 + LAYERS + 2, 2 - target
    return max(16, int(round((-b + math.sqrt(b*b - 4*a*c)) / (2*a))))

def baseline_ff(field, hidden):
    v = _np(field); T, Y, X, C = v.shape
    ax = lambda n: np.linspace(-1, 1, n, np.float32)
    gt, gy, gx = np.meshgrid(ax(T), ax(Y), ax(X), indexing="ij")
    coords = np.stack([gx.ravel(), gy.ravel(), gt.ravel()], -1).astype(np.float32)
    flat = v.reshape(-1, C); vmin = flat.min(0); vmax = flat.max(0); sc = np.maximum(vmax-vmin, 1e-8)
    rec, _, npar = fit_ffrelu(coords, (2*(flat-vmin)/sc-1).astype(np.float32), EP, LR, BATCH, hidden, LAYERS, NFREQ, 8.0)
    return vpsnr(((rec+1)*0.5*sc+vmin).reshape(T,Y,X,C), v), npar

def run(name, field, budget, tau):
    dec = decompose_reference_frame_2d(field, k=2)
    labels, N = dec.cut_adaptive(tau)
    print(f"\n===== {name} shape={_np(field).shape}  budget={budget/1e6:.2f}M  tau={tau}  "
          f"(benefit={dec.diag['decomposition_benefit']:.3f})  adaptive N={N} =====", flush=True)
    hb = hidden_for_params(budget)
    pb, nb = baseline_ff(field, hb)                                   # RECOMPUTED in-process
    print(f"  baseline INR(v)            : PSNR={pb:6.2f}dB  params={nb/1e6:.2f}M (h={hb})", flush=True)
    hr = hidden_for_params(budget / max(N, 1))
    pp, tot, ksc = proposed_overlap(field, dec, N, hr)               # N adaptive regions, overlap+blend
    print(f"  proposed adaptive N={N} (h={hr}): PSNR={pp:6.2f}dB  params={tot/1e6:.2f}M (+{ksc} killing)  "
          f"ΔvsBaseline={pp-pb:+.2f}dB", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="all", choices=["rfc","cylinder","boussinesq","all"])
    ap.add_argument("--budget", type=float, default=3.94e6); ap.add_argument("--tau", type=float, default=0.02)
    a = ap.parse_args()
    if a.field in ("rfc","all"): run("rfc", rotation_four_center((64,64),64), a.budget, a.tau)
    if a.field in ("cylinder","all"):
        cf=NetCDFLoader.load_vector_field2d(r"C:\Users\xingdi\OneDrive - KAUST\WorkingInProcess\FLowVisAssets\flowData2D\cylinder2d.nc",800,960); cf.resample2UnsteadyField((128,320,80)); run("cylinder2d", cf, a.budget, a.tau)
    if a.field in ("boussinesq","all"):
        bf=NetCDFLoader.load_vector_field2d(r"C:\Users\xingdi\OneDrive - KAUST\WorkingInProcess\FLowVisAssets\flowData2D\boussinesq.nc",800,960); bf.resample2UnsteadyField((128,75,225)); run("boussinesq", bf, a.budget, a.tau)
