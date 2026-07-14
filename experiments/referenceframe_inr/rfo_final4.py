"""FIXED PARAMETER BUDGET comparison. Same total params: baseline(1 big INR) vs proposed(N small INRs).
Here: budget P=3.94M -> baseline hidden=930 (~3.94M, 1 INR) vs proposed cut=3 (3x hidden=512, total 3.94M).
Same FF+ReLU family, scale=8, epochs=2500, deterministic. proposed cut=3 numbers from rfo_final3 (same config)."""
import sys, argparse, numpy as np, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, r"C:\Users\xingdi\sources\PyflowVis")
sys.path.insert(0, r"C:\Users\xingdi\AppData\Local\Temp\claude\C--Users-xingdi-sources-PyflowVis\5b7dbb49-2a88-462c-b42d-687217cb2b30\scratchpad")
from ff_inr import fit_ffrelu, FFReLU
from rfo_decompose_inr import vpsnr, _np
from FLowUtils.flowDatasetUtils.NetCDF_AmiraLoader import NetCDFLoader
EP, LR, BATCH = 2500, 1e-3, 200000

def baseline_H(field, hidden):
    v = _np(field); T, Y, X, C = v.shape
    ax = lambda n: np.linspace(-1, 1, n, np.float32)
    gt, gy, gx = np.meshgrid(ax(T), ax(Y), ax(X), indexing="ij")
    coords = np.stack([gx.ravel(), gy.ravel(), gt.ravel()], -1).astype(np.float32)
    flat = v.reshape(-1, C); vmin = flat.min(0); vmax = flat.max(0); sc = np.maximum(vmax - vmin, 1e-8)
    rec, _, npar = fit_ffrelu(coords, (2*(flat-vmin)/sc-1).astype(np.float32), EP, LR, BATCH, hidden, 4, 256, 8.0)
    return vpsnr(((rec+1)*0.5*sc+vmin).reshape(T,Y,X,C), v), npar

def run(name, field):
    for H in [1343]:
        npar = sum(p.numel() for p in FFReLU(3,2,H,4,256,8.0).parameters())
        p, _ = baseline_H(field, H)
        print(f"  {name}: baseline hidden={H} : PSNR={p:6.2f}dB  params={npar/1e6:.2f}M", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--field", default="both", choices=["cylinder","boussinesq","both"]); a = ap.parse_args()
    if a.field in ("cylinder","both"):
        cf = NetCDFLoader.load_vector_field2d(r"C:\Users\xingdi\OneDrive - KAUST\WorkingInProcess\FLowVisAssets\flowData2D\cylinder2d.nc", 800, 960); cf.resample2UnsteadyField((128,320,80)); run("cylinder2d", cf)
    if a.field in ("boussinesq","both"):
        bf = NetCDFLoader.load_vector_field2d(r"C:\Users\xingdi\OneDrive - KAUST\WorkingInProcess\FLowVisAssets\flowData2D\boussinesq.nc", 800, 960); bf.resample2UnsteadyField((128,75,225)); run("boussinesq", bf)
