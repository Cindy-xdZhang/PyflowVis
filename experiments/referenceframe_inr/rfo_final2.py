"""FINAL controlled experiment. Fixed INR = FF+ReLU (scale=8,hidden=512,n_freq=256,layers=4), deterministic.
first-order (3-DOF) killing. baseline INR(v) vs proposed (global observer -> pushforward observed field -> INR).
cut=1 reuses the bit-identical global observer, so global == cut=1 (verified)."""
import sys, math, argparse, numpy as np, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, r"C:\Users\xingdi\sources\PyflowVis")
sys.path.insert(0, r"C:\Users\xingdi\AppData\Local\Temp\claude\C--Users-xingdi-sources-PyflowVis\5b7dbb49-2a88-462c-b42d-687217cb2b30\scratchpad")
import rfo_decompose_inr as DEC
from ff_inr import fit_ffrelu
# FIX the INR: FF+ReLU scale=8 (replaces SIREN everywhere fit_inr is used)
DEC.fit_inr = lambda coords, values, epochs, m, d, lr, batch, seed=0, omega_0=None: \
    fit_ffrelu(coords, values, epochs, lr, batch, 512, 4, 256, 8.0, seed)
from rfo_decompose_inr import fit_region, vpsnr, _np
from rfo_final_experiment import global_killing_observer_2d
from FLowUtils.referenceFrameDecompose import decompose_reference_frame_2d
from FLowUtils.AnalyticalFlowCreator import rotation_four_center
from FLowUtils.flowDatasetUtils.NetCDF_AmiraLoader import NetCDFLoader
EP, LR, BATCH = 4000, 1e-3, 200000

def baseline(field):
    v = _np(field); T, Y, X, C = v.shape
    ax = lambda n: np.linspace(-1, 1, n, np.float32)
    gt, gy, gx = np.meshgrid(ax(T), ax(Y), ax(X), indexing="ij")
    coords = np.stack([gx.ravel(), gy.ravel(), gt.ravel()], -1).astype(np.float32)
    flat = v.reshape(-1, C); vmin = flat.min(0); vmax = flat.max(0); sc = np.maximum(vmax - vmin, 1e-8)
    rec, _, npar = DEC.fit_inr((coords), (2*(flat-vmin)/sc-1).astype(np.float32), EP, 0, 0, LR, BATCH)
    recon = ((rec + 1) * 0.5 * sc + vmin).reshape(T, Y, X, C)
    return vpsnr(recon, v), npar

def proposed(field, observer):
    v = _np(field); T, Y, X, _ = v.shape
    xph = field.domainMinBoundary[0] + field.gridInterval[0] * np.arange(X)
    yph = field.domainMinBoundary[1] + field.gridInterval[1] * np.arange(Y)
    times = np.array([field.getPhysicalTime(i) for i in range(T)], np.float64)
    ys, xs, vx, vy, _, npar = fit_region(v, np.ones((Y,X),bool), observer, times, xph, yph, EP, 0, 0, LR, BATCH)
    vr = np.zeros_like(v); vr[:, ys, xs, 0] = vx; vr[:, ys, xs, 1] = vy
    return vpsnr(vr, v), npar

def run(name, field):
    print(f"\n===== {name} shape={_np(field).shape}  (FF+ReLU scale=8, {EP}ep, lr={LR}, deterministic) =====", flush=True)
    pb, nb = baseline(field)
    print(f"  baseline  INR(v)                    : PSNR={pb:6.2f}dB  params={nb/1e6:.2f}M", flush=True)
    obs_g = global_killing_observer_2d(field)
    dec = decompose_reference_frame_2d(field, k=2)
    obs_tree = list(dec.region_observers(dec.cut(n_regions=1)).values())[0]
    print(f"  [check] observer max|global - tree-cut1| = {np.abs(obs_g-obs_tree).max():.2e}  (float roundoff; same observer)", flush=True)
    pp, npp = proposed(field, obs_g)
    ksc = obs_g.size
    print(f"  proposed  global(=cut1) pushforward : PSNR={pp:6.2f}dB  params={npp/1e6:.2f}M (+{ksc} killing)  ΔvsBaseline={pp-pb:+.2f}dB", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--field", default="rfc", choices=["rfc","cylinder","both"]); a = ap.parse_args()
    if a.field in ("rfc","both"): run("rfc", rotation_four_center((64,64),64))
    if a.field in ("cylinder","both"):
        cf = NetCDFLoader.load_vector_field2d(r"C:\Users\xingdi\OneDrive - KAUST\WorkingInProcess\FLowVisAssets\flowData2D\cylinder2d.nc", 800, 960)
        cf.resample2UnsteadyField((128,320,80)); run("cylinder2d", cf)
