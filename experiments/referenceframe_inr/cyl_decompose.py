import sys, time; sys.path.insert(0, r"C:\Users\xingdi\sources\PyflowVis")
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from FLowUtils.flowDatasetUtils.NetCDF_AmiraLoader import NetCDFLoader
from FLowUtils.referenceFrameDecompose import decompose_reference_frame_2d

vf = NetCDFLoader.load_vector_field2d(r"C:\Users\xingdi\OneDrive - KAUST\WorkingInProcess\FLowVisAssets\flowData2D\cylinder2d.nc", 800, 960)
vf.resample2UnsteadyField((128, 320, 80))
t0=time.time(); dec = decompose_reference_frame_2d(vf, k=2, verbose=False); print("decompose %.1fs"%(time.time()-t0))
d = dec.diag
print("diag: global_res_ratio=%.3f finest_res_ratio=%.3f benefit=%.3f n_leaves=%d grid=%s"%(
    d["global_residual_ratio"], d["finest_residual_ratio"], d["decomposition_benefit"], d["n_leaves"], d["leaf_grid"]))
print("->", d["interpretation"])
ns, res = dec.residual_curve(max_regions=12)
print("residual curve:", list(zip(ns.tolist(), np.round(res,2).tolist())))

v = np.asarray(vf.field, np.float64); T,Y,X,_ = v.shape
speed = np.hypot(v[T//2,:,:,0], v[T//2,:,:,1])
fig, axes = plt.subplots(5,1, figsize=(11,11))
axes[0].imshow(speed, origin="lower", cmap="viridis", aspect="auto"); axes[0].set_title("cylinder2d |v| (t=mid)")
for ax, n in zip(axes[1:], [2,3,4,6]):
    labels = dec.cut(n_regions=n)
    ax.imshow(speed, origin="lower", cmap="gray", aspect="auto")
    ax.imshow(labels, origin="lower", cmap="tab10", alpha=0.45, aspect="auto")
    obs = dec.region_observers(labels)
    txt = "  ".join(f"R{r}:(a,b,c)=({o[2:-2,0].mean():.2f},{o[2:-2,1].mean():.2f},{o[2:-2,2].mean():.2f})" for r,o in list(obs.items())[:4])
    ax.set_title(f"cut n={n}   {txt}", fontsize=8)
for ax in axes: ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
out=r"C:\Users\xingdi\AppData\Local\Temp\claude\C--Users-xingdi-sources-PyflowVis\5b7dbb49-2a88-462c-b42d-687217cb2b30\scratchpad\cyl_decompose.png"
fig.savefig(out, dpi=110); print("saved", out)
