import sys; sys.path.insert(0, r"C:\Users\xingdi\sources\PyflowVis")
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from FLowUtils.ReferenceFrameOptimization import killing_optimization_2d, gunther17_optimization_2d
from FLowUtils.AnalyticalFlowCreator import rotation_four_center, constant_rotation

rfc = rotation_four_center((64,64),64)
cr  = constant_rotation((64,64),64)
v  = np.asarray(rfc.field, np.float64)
cr = np.asarray(cr.field,  np.float64)
uk = np.asarray(killing_optimization_2d(rfc).u_field.field, np.float64)
ug = np.asarray(gunther17_optimization_2d(rfc, neighborhood=4).u_field.field, np.float64)

x = np.linspace(-2,2,64); y = np.linspace(-2,2,64)
ts = [0, 21, 42]
rows = [("input  v(rfc)", v),
        ("ground-truth observer\n= constant_rotation", cr),
        ("killing  recovered u", uk),
        ("gunther17 recovered u", ug)]
fig, axes = plt.subplots(4, 3, figsize=(11, 13.5))
for r,(label,data) in enumerate(rows):
    for c,t in enumerate(ts):
        ax = axes[r,c]; U=data[t,:,:,0]; V=data[t,:,:,1]; spd=np.sqrt(U*U+V*V)
        ax.streamplot(x,y,U,V,color=spd,cmap="viridis",density=1.1,linewidth=0.7,arrowsize=0.7)
        ax.set_xlim(-2,2); ax.set_ylim(-2,2); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        if r==0: ax.set_title(f"t = {t/63*2*np.pi:.2f}", fontsize=10)
        if c==0: ax.set_ylabel(label, fontsize=9)
fig.suptitle("RFC verification: rows 2-4 are IDENTICAL across all columns (a CONSTANT rotation)\n"
             "killing & gunther17 both recover u = constant_rotation (the known observer, angular velocity -1)", fontsize=10)
fig.tight_layout(rect=[0,0,1,0.965])
out = r"C:\Users\xingdi\AppData\Local\Temp\claude\C--Users-xingdi-sources-PyflowVis\5b7dbb49-2a88-462c-b42d-687217cb2b30\scratchpad\rfc_verify.png"
fig.savefig(out, dpi=110); print("saved", out)

def tvar(d): return float(np.mean(np.std(d[1:-1],axis=0)))   # temporal variation (0 => constant in time)
def relerr(a,b,m=None):
    if m is None: m=np.ones(a.shape[1:3],bool)
    d=a[1:-1][:,m]-b[1:-1][:,m]; r=b[1:-1][:,m]; return float(np.sqrt((d*d).sum()/(r*r).sum()))
mi=np.zeros((64,64),bool); mi[6:-6,6:-6]=True
print(f"temporal variation (0=constant):  constant_rotation={tvar(cr):.4f}  killing_u={tvar(uk):.4f}  gunther_u={tvar(ug):.4f}")
print(f"observer vs constant_rotation rel-err:  killing={relerr(uk,cr):.4f} (all)   gunther={relerr(ug,cr,mi):.4f} (interior)")
