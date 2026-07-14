"""
Experiment: does fitting the KILLING-observed field v_hat (+ storing 3 killing
coeffs/timestep) beat fitting the raw field v with the same INR?

baseline : INR(v)                         -> recon_v
proposed : killing -> (c_t, v_hat=v-u);  INR(v_hat) -> recon_vhat;  v_recon = recon_vhat + u
metrics  : train time, PSNR & rel-L2 in the ORIGINAL v space, #params (+ 3T killing).
Same net config on both sides (only the training data differs) => fair.
"""
import sys, time, math, argparse
sys.path.insert(0, r"C:\Users\xingdi\sources\PyflowVis")
import numpy as np
import torch
import torch.nn as nn

from CoordNetCompression import CoordNet
from FLowUtils.ReferenceFrameOptimization import killing_optimization_2d
from FLowUtils.AnalyticalFlowCreator import rotation_four_center
from FLowUtils.flowDatasetUtils.NetCDF_AmiraLoader import NetCDFLoader

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _field_np(f):
    d = f.field
    if hasattr(d, "detach"): d = d.detach().cpu().numpy()
    return np.asarray(d, np.float32)


def make_coords(T, Y, X):
    ax = lambda n: (np.linspace(-1,1,n,dtype=np.float32) if n>1 else np.zeros(1,np.float32))
    gt,gy,gx = np.meshgrid(ax(T), ax(Y), ax(X), indexing="ij")
    return np.stack([gx.ravel(), gy.ravel(), gt.ravel()], -1)  # (N,3) = (x,y,t)


def fit_vector_inr(data, epochs, m, d, lr, batch, seed=0, log_tag=""):
    """data: (T,Y,X,2) physical. Returns (recon (T,Y,X,2), train_time, n_params, final_mse_phys)."""
    torch.manual_seed(seed); np.random.seed(seed)
    T,Y,X,C = data.shape
    vmin = data.reshape(-1,C).min(0); vmax = data.reshape(-1,C).max(0)
    scale = np.maximum(vmax-vmin, 1e-8)
    norm = (2*(data.reshape(-1,C)-vmin)/scale - 1).astype(np.float32)     # (N,2) in [-1,1]
    coords = make_coords(T,Y,X)                                           # (N,3)
    coords_t = torch.from_numpy(coords).to(DEV)
    values_t = torch.from_numpy(norm).to(DEV)
    N = coords_t.shape[0]

    model = CoordNet(3, C, m=m, d=d, omega_0=30.0, final_activation="sine").to(DEV)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bs = min(batch, N); steps = max(1, N//bs)
    lr_f = lr*0.1
    import copy
    best=float("inf"); best_state=copy.deepcopy(model.state_dict())
    t0=time.time()
    for ep in range(epochs):
        cur = lr_f + 0.5*(lr-lr_f)*(1+math.cos(math.pi*ep/max(1,epochs-1)))
        for g in opt.param_groups: g["lr"]=cur
        perm = torch.randperm(N, device=DEV)
        eloss=0.0
        for s in range(steps):
            idx = perm[s*bs:(s+1)*bs]
            pred = model(coords_t[idx])
            loss = ((pred - values_t[idx])**2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            eloss += loss.item()
        eloss/=steps
        if eloss<best: best=eloss; best_state=copy.deepcopy(model.state_dict())
        if log_tag and ((ep+1)%max(1,epochs//5)==0 or ep==epochs-1):
            print(f"    [{log_tag}] ep {ep+1}/{epochs} lr={cur:.1e} mse(norm)={eloss:.3e} best={best:.3e}")
    train_time=time.time()-t0
    model.load_state_dict(best_state)
    # reconstruct
    model.eval()
    recon_n = np.empty((N,C), np.float32)
    with torch.no_grad():
        for s in range(0,N,200000):
            recon_n[s:s+200000] = model(coords_t[s:s+200000]).clamp(-1,1).cpu().numpy()
    recon = ((recon_n+1)*0.5*scale + vmin).reshape(T,Y,X,C).astype(np.float32)
    mse_phys = float(np.mean((recon-data)**2))
    return recon, train_time, n_params, mse_phys


def vpsnr(recon, gt):
    rng = float(gt.max()-gt.min())
    mse = float(np.mean((recon.astype(np.float64)-gt.astype(np.float64))**2))
    return float("inf") if mse<=0 else 20*math.log10(rng)-10*math.log10(mse)

def rel_l2(recon, gt):
    return float(np.linalg.norm(recon-gt)/max(np.linalg.norm(gt),1e-12))


def run_field(name, vfield, epochs, m, d, lr, batch):
    v = _field_np(vfield)                       # (T,Y,X,2)
    T = v.shape[0]
    print(f"\n===== {name}  v shape={v.shape}  |v| range [{v.min():.3f},{v.max():.3f}] =====")

    # ---- baseline: INR on v ----
    print("  [baseline] fitting INR on raw v ...")
    recon_v, t_b, p_b, mse_b = fit_vector_inr(v, epochs, m, d, lr, batch, log_tag="baseline")
    psnr_b, rl_b = vpsnr(recon_v, v), rel_l2(recon_v, v)

    # ---- proposed: killing -> vhat ; INR on vhat ; v_recon = recon_vhat + u ----
    print("  [proposed] killing optimization ...")
    res = killing_optimization_2d(vfield)
    u = _field_np(res.u_field); vhat = _field_np(res.v_minus_u_field)   # v = u + vhat
    print(f"    killing c(angular vel) mean={res.params[1:-1,2].mean():+.4f}  "
          f"|vhat| range [{vhat.min():.3f},{vhat.max():.3f}]  (raw |v| [{v.min():.3f},{v.max():.3f}])")
    print("  [proposed] fitting INR on observed vhat ...")
    recon_vhat, t_p, p_p, mse_p = fit_vector_inr(vhat, epochs, m, d, lr, batch, log_tag="proposed")
    v_recon = recon_vhat + u
    psnr_p, rl_p = vpsnr(v_recon, v), rel_l2(v_recon, v)
    killing_scalars = res.params.size                                    # T*3

    print(f"\n  --- {name} results (same net m={m} d={d}, {epochs} ep) ---")
    print(f"  baseline : PSNR={psnr_b:5.2f}dB  relL2={rl_b:.4f}  time={t_b:5.1f}s  params={p_b/1e3:.1f}K")
    print(f"  proposed : PSNR={psnr_p:5.2f}dB  relL2={rl_p:.4f}  time={t_p:5.1f}s  params={p_p/1e3:.1f}K + {killing_scalars} killing")
    print(f"  fit-MSE(phys) on data: baseline(v)={mse_b:.3e}  proposed(vhat)={mse_p:.3e}  ratio={mse_b/max(mse_p,1e-20):.2f}x")
    return dict(name=name, psnr_b=psnr_b, psnr_p=psnr_p, rl_b=rl_b, rl_p=rl_p,
                t_b=t_b, t_p=t_p, p_b=p_b, p_p=p_p, killing=killing_scalars,
                v=v, vhat=vhat, u=u, recon_v=recon_v, v_recon=v_recon)


def load_rfc():
    return rotation_four_center((64,64), 64)

def load_cylinder():
    vf = NetCDFLoader.load_vector_field2d(
        r"C:\Users\xingdi\OneDrive - KAUST\WorkingInProcess\FLowVisAssets\flowData2D\cylinder2d.nc",
        800, 960)
    vf.resample2UnsteadyField((128, 320, 80))   # (newT, newX, newY)
    return vf


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="rfc", choices=["rfc","cylinder","both"])
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--d", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch", type=int, default=200000)
    a = ap.parse_args()
    print("device:", DEV)
    fields = []
    if a.field in ("rfc","both"):      fields.append(("rfc2d", load_rfc()))
    if a.field in ("cylinder","both"): fields.append(("cylinder2d", load_cylinder()))
    results = [run_field(n, f, a.epochs, a.m, a.d, a.lr, a.batch) for n,f in fields]
    np.save(r"C:\Users\xingdi\AppData\Local\Temp\claude\C--Users-xingdi-sources-PyflowVis\5b7dbb49-2a88-462c-b42d-687217cb2b30\scratchpad\rfo_inr_results.npy",
            {r["name"]: {k:v for k,v in r.items() if not isinstance(v,np.ndarray)} for r in results}, allow_pickle=True)
    print("\n================ SUMMARY ================")
    for r in results:
        print(f"{r['name']:12s} baseline PSNR {r['psnr_b']:5.2f}dB / proposed PSNR {r['psnr_p']:5.2f}dB "
              f"(Δ{r['psnr_p']-r['psnr_b']:+.2f})  | time {r['t_b']:.0f}s vs {r['t_p']:.0f}s")
