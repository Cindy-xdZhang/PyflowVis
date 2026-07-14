"""Improved proposed: region OVERLAP training + distance-weighted BLEND reconstruction (removes hard-boundary loss).
Fixed budget 3.94M: each cut=N uses N INRs of hidden st N*params=3.94M. Compare vs baseline h=930 (48.93 bouss / 49.17 cyl).
epochs=2500, FF+ReLU scale=8, deterministic."""
import sys, argparse, numpy as np, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, r"C:\Users\xingdi\sources\PyflowVis")
sys.path.insert(0, r"C:\Users\xingdi\AppData\Local\Temp\claude\C--Users-xingdi-sources-PyflowVis\5b7dbb49-2a88-462c-b42d-687217cb2b30\scratchpad")
from scipy.ndimage import binary_dilation, gaussian_filter
import rfo_decompose_inr as DEC
from ff_inr import fit_ffrelu, FFReLU
from rfo_decompose_inr import fit_region, vpsnr, _np
from FLowUtils.referenceFrameDecompose import decompose_reference_frame_2d
from FLowUtils.flowDatasetUtils.NetCDF_AmiraLoader import NetCDFLoader
EP, LR, BATCH = 2500, 1e-3, 200000
BASELINE = {"cylinder2d": 49.17, "boussinesq": 48.93}  # baseline h=930 (3.94M), from rfo_final4

def set_ff(hidden):
    DEC.fit_inr = lambda coords, values, epochs, m, d, lr, batch, seed=0, omega_0=None: \
        fit_ffrelu(coords, values, epochs, lr, batch, hidden, 4, 256, 8.0, seed)

def proposed_overlap(field, dec, N, hidden, margin=4, sigma=4.0):
    set_ff(hidden)
    v = _np(field); T, Y, X, _ = v.shape
    xph = field.domainMinBoundary[0] + field.gridInterval[0]*np.arange(X)
    yph = field.domainMinBoundary[1] + field.gridInterval[1]*np.arange(Y)
    times = np.array([field.getPhysicalTime(i) for i in range(T)], np.float64)
    labels = dec.cut(n_regions=N); obs = dec.region_observers(labels)
    accum_v = np.zeros_like(v); accum_w = np.zeros((Y, X)); tot = 0; ksc = 0
    for rid, coeff in obs.items():
        core = labels == rid
        tmask = binary_dilation(core, iterations=margin)                 # overlap band
        w = gaussian_filter(core.astype(np.float64), sigma) * tmask      # soft weight, limited to trained region
        ys, xs, vx, vy, _, npar = fit_region(v, tmask, coeff, times, xph, yph, EP, 0, 0, LR, BATCH)
        wp = w[ys, xs]
        accum_v[:, ys, xs, 0] += wp[None] * vx; accum_v[:, ys, xs, 1] += wp[None] * vy
        accum_w[ys, xs] += wp; tot += npar; ksc += coeff.size
    v_recon = accum_v / np.clip(accum_w, 1e-8, None)[None, :, :, None]
    return vpsnr(v_recon, v), tot, ksc

def run(name, field):
    print(f"\n===== {name}  (overlap+blend, fixed budget ~3.94M, {EP}ep)  baseline(h930)={BASELINE[name]}dB =====", flush=True)
    dec = decompose_reference_frame_2d(field, k=2)
    for N, h in [(3,512),(5,384),(10,256)]:
        p, tot, ksc = proposed_overlap(field, dec, N, h)
        print(f"  cut={N:2d} (h={h}, {N}INR): PSNR={p:6.2f}dB  params={tot/1e6:.2f}M (+{ksc} killing)  vsBaseline={p-BASELINE[name]:+.2f}dB", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--field", default="boussinesq", choices=["cylinder","boussinesq","both"]); a = ap.parse_args()
    if a.field in ("boussinesq","both"):
        bf = NetCDFLoader.load_vector_field2d(r"C:\Users\xingdi\OneDrive - KAUST\WorkingInProcess\FLowVisAssets\flowData2D\boussinesq.nc", 800, 960); bf.resample2UnsteadyField((128,75,225)); run("boussinesq", bf)
    if a.field in ("cylinder","both"):
        cf = NetCDFLoader.load_vector_field2d(r"C:\Users\xingdi\OneDrive - KAUST\WorkingInProcess\FLowVisAssets\flowData2D\cylinder2d.nc", 800, 960); cf.resample2UnsteadyField((128,320,80)); run("cylinder2d", cf)
