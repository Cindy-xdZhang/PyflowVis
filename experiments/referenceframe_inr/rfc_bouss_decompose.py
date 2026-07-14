import sys, time; sys.path.insert(0, r"C:\Users\xingdi\sources\PyflowVis")
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from FLowUtils.AnalyticalFlowCreator import rotation_four_center
from FLowUtils.flowDatasetUtils.NetCDF_AmiraLoader import NetCDFLoader
from FLowUtils.referenceFrameDecompose import decompose_reference_frame_2d

OUT = r"C:\Users\xingdi\AppData\Local\Temp\claude\C--Users-xingdi-sources-PyflowVis\5b7dbb49-2a88-462c-b42d-687217cb2b30\scratchpad"

def make_fig(name, dec, v, cuts, save):
    T,Y,X,_ = v.shape
    speed = np.hypot(v[T//2,:,:,0], v[T//2,:,:,1])
    fig, axes = plt.subplots(len(cuts)+1, 1, figsize=(max(6,11*X/max(X,Y)*1.0), 2.2*(len(cuts)+1)))
    if len(cuts)+1==1: axes=[axes]
    axes[0].imshow(speed, origin="lower", cmap="viridis", aspect="auto")
    axes[0].set_title(f"{name}  |v| (t=mid)   [diag: global_res_ratio={dec.diag['global_residual_ratio']:.3f}, "
                      f"benefit={dec.diag['decomposition_benefit']:.3f}]  {dec.diag['interpretation'][:34]}", fontsize=8)
    for ax, n in zip(axes[1:], cuts):
        labels = dec.cut(n_regions=n)
        nreg = len(np.unique(labels))
        totres = sum(dec.region_residuals(labels).values())
        ax.imshow(speed, origin="lower", cmap="gray", aspect="auto")
        ax.imshow(labels, origin="lower", cmap="tab10", alpha=0.45, aspect="auto", vmin=0, vmax=9)
        obs = dec.region_observers(labels)
        txt = " ".join(f"R{r}:({o[2:-2,0].mean():.2f},{o[2:-2,1].mean():.2f},{o[2:-2,2].mean():.2f})" for r,o in list(obs.items())[:5])
        ax.set_title(f"cut n={n} (regions={nreg})  total_residual={totres:.3g}   {txt}", fontsize=7)
    for ax in axes: ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(save, dpi=110); print("saved", save)

# ---- RFC: expect ONE region ----
print("=== RFC ===")
f = rotation_four_center((64,64), 64)
dec = decompose_reference_frame_2d(f, k=2)
d=dec.diag
print("diag: global_res_ratio=%.4f finest=%.4f benefit=%.4f -> %s"%(d['global_residual_ratio'],d['finest_residual_ratio'],d['decomposition_benefit'],d['interpretation']))
ns,res = dec.residual_curve(max_regions=8)
print("residual curve (n,res):", list(zip(ns.tolist(), np.round(res,3).tolist())))
# auto-cut by cost_threshold = small fraction of raw energy => RFC should give 1 region
thr = 0.01*d['raw_td_energy']
auto = dec.cut(cost_threshold=thr)
print("auto cut (cost_threshold=1%% raw) -> #regions =", len(np.unique(auto)))
make_fig("RFC (rotating four-center)", dec, np.asarray(f.field,np.float64), [1,2,3], OUT+r"\rfc_decompose.png")

# ---- boussinesq ----
print("\n=== boussinesq ===")
vf = NetCDFLoader.load_vector_field2d(r"C:\Users\xingdi\OneDrive - KAUST\WorkingInProcess\FLowVisAssets\flowData2D\boussinesq.nc", 800, 960)
vf.resample2UnsteadyField((128, 75, 225))   # (newT,newX,newY) -> field (128,225,75,2)
print("boussinesq resampled", vf.field.shape, "domain", vf.domainMinBoundary, vf.domainMaxBoundary)
t0=time.time(); dec2 = decompose_reference_frame_2d(vf, k=2); print("decompose %.1fs"%(time.time()-t0))
d2=dec2.diag
print("diag: global_res_ratio=%.4f finest=%.4f benefit=%.4f n_leaves=%d -> %s"%(d2['global_residual_ratio'],d2['finest_residual_ratio'],d2['decomposition_benefit'],d2['n_leaves'],d2['interpretation']))
ns2,res2 = dec2.residual_curve(max_regions=12)
print("residual curve:", list(zip(ns2.tolist(), np.round(res2,1).tolist())))
make_fig("boussinesq", dec2, np.asarray(vf.field,np.float64), [2,3,4,6], OUT+r"\bouss_decompose.png")
