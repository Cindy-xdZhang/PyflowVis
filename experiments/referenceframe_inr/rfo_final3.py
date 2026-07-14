"""cut=3/6 baseline vs proposed, fixed FF+ReLU (scale=8), deterministic. cylinder + boussinesq.
proposed cut=N: decompose -> N regions, each with its own 3-DOF killing observer -> pushforward observed
field -> ONE FF+ReLU per region. baseline: one FF+ReLU on raw v."""
import sys, argparse, numpy as np, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, r"C:\Users\xingdi\sources\PyflowVis")
sys.path.insert(0, r"C:\Users\xingdi\AppData\Local\Temp\claude\C--Users-xingdi-sources-PyflowVis\5b7dbb49-2a88-462c-b42d-687217cb2b30\scratchpad")
import rfo_decompose_inr as DEC
from ff_inr import fit_ffrelu
DEC.fit_inr = lambda coords, values, epochs, m, d, lr, batch, seed=0, omega_0=None: \
    fit_ffrelu(coords, values, epochs, lr, batch, 512, 4, 256, 8.0, seed)
from rfo_decompose_inr import fit_region, vpsnr, _np
from FLowUtils.referenceFrameDecompose import decompose_reference_frame_2d
from FLowUtils.flowDatasetUtils.NetCDF_AmiraLoader import NetCDFLoader
EP, LR, BATCH = 2500, 1e-3, 200000

def baseline(field):
    v = _np(field); T, Y, X, C = v.shape
    ax = lambda n: np.linspace(-1, 1, n, np.float32)
    gt, gy, gx = np.meshgrid(ax(T), ax(Y), ax(X), indexing="ij")
    coords = np.stack([gx.ravel(), gy.ravel(), gt.ravel()], -1).astype(np.float32)
    flat = v.reshape(-1, C); vmin = flat.min(0); vmax = flat.max(0); sc = np.maximum(vmax - vmin, 1e-8)
    rec, _, npar = DEC.fit_inr(coords, (2*(flat-vmin)/sc-1).astype(np.float32), EP, 0, 0, LR, BATCH)
    return vpsnr(((rec+1)*0.5*sc+vmin).reshape(T,Y,X,C), v), npar

def proposed_cut(field, dec, N):
    v = _np(field); T, Y, X, _ = v.shape
    xph = field.domainMinBoundary[0] + field.gridInterval[0]*np.arange(X)
    yph = field.domainMinBoundary[1] + field.gridInterval[1]*np.arange(Y)
    times = np.array([field.getPhysicalTime(i) for i in range(T)], np.float64)
    labels = dec.cut(n_regions=N); obs = dec.region_observers(labels)
    v_recon = np.zeros_like(v); tot = 0; ksc = 0
    for rid, coeff in obs.items():
        mask = labels == rid
        ys, xs, vx, vy, _, npar = fit_region(v, mask, coeff, times, xph, yph, EP, 0, 0, LR, BATCH)
        v_recon[:, ys, xs, 0] = vx; v_recon[:, ys, xs, 1] = vy; tot += npar; ksc += coeff.size
    return vpsnr(v_recon, v), tot, ksc, len(obs)

def run(name, field):
    print(f"\n===== {name} shape={_np(field).shape} (FF+ReLU scale=8, {EP}ep, deterministic) =====", flush=True)
    pb, nb = baseline(field)
    print(f"  baseline INR(v)    : PSNR={pb:6.2f}dB  params={nb/1e6:.2f}M", flush=True)
    dec = decompose_reference_frame_2d(field, k=2)
    for N in [1, 3, 6]:
        p, tot, ksc, nreg = proposed_cut(field, dec, N)
        print(f"  proposed cut={N} (regions={nreg}): PSNR={p:6.2f}dB  params={tot/1e6:.2f}M (+{ksc} killing)  ΔvsBaseline={p-pb:+.2f}dB", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--field", default="both", choices=["cylinder","boussinesq","both"]); a = ap.parse_args()
    if a.field in ("cylinder","both"):
        cf = NetCDFLoader.load_vector_field2d(r"C:\Users\xingdi\OneDrive - KAUST\WorkingInProcess\FLowVisAssets\flowData2D\cylinder2d.nc", 800, 960)
        cf.resample2UnsteadyField((128,320,80)); run("cylinder2d", cf)
    if a.field in ("boussinesq","both"):
        bf = NetCDFLoader.load_vector_field2d(r"C:\Users\xingdi\OneDrive - KAUST\WorkingInProcess\FLowVisAssets\flowData2D\boussinesq.nc", 800, 960)
        bf.resample2UnsteadyField((128,75,225)); run("boussinesq", bf)
