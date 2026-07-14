"""
3-way comparison of INR compression under different reference frames:
  baseline : INR(v)               coords (x,y,t)                      -> recon_v
  observed : INR(v_hat=v-u)       coords (x,y,t)                      -> +u -> recon_v      [user's original idea]
  warp     : INR(v_hat in co-moving frame)  coords (xi,t)=warp(x,t)   -> rotate back +u     [the fix]

The 'warp' variant integrates the killing observer motion (theta_t=∫c, D_t=∫R(-theta)v0)
and feeds CO-MOVING coordinates to the INR, so a field that is steady in the moving frame
(e.g. RFC four-center) becomes ~t-independent and trivial to fit.  This is the CoordNet
'coordinate transform' seam used for real.
Same net config across variants (only coords/targets differ) => fair.
"""
import sys, time, math, argparse, copy
sys.path.insert(0, r"C:\Users\xingdi\sources\PyflowVis")
import numpy as np, torch
from CoordNetCompression import CoordNet
from FLowUtils.ReferenceFrameOptimization import killing_optimization_2d
from FLowUtils.AnalyticalFlowCreator import rotation_four_center
from FLowUtils.flowDatasetUtils.NetCDF_AmiraLoader import NetCDFLoader

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _np(f):
    d = f.field
    if hasattr(d, "detach"): d = d.detach().cpu().numpy()
    return np.asarray(d, np.float32)

def _norm_axis(n): return np.linspace(-1,1,n,np.float32) if n>1 else np.zeros(1,np.float32)

def norm_pc(vals):                      # per-channel minmax -> [-1,1]
    vmin = vals.min(0); vmax = vals.max(0); scale = np.maximum(vmax-vmin,1e-8)
    return (2*(vals-vmin)/scale-1).astype(np.float32), vmin, scale
def denorm_pc(vn, vmin, scale): return ((vn+1)*0.5*scale+vmin).astype(np.float32)


def fit_inr_coords(coords, values, epochs, m, d, lr, batch, seed=0, tag=""):
    torch.manual_seed(seed); np.random.seed(seed)
    N,C = values.shape
    ct = torch.from_numpy(coords).to(DEV); vt = torch.from_numpy(values).to(DEV)
    model = CoordNet(coords.shape[1], C, m=m, d=d, omega_0=30.0, final_activation="sine").to(DEV)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bs = min(batch,N); steps=max(1,N//bs); lr_f=lr*0.1
    best=float("inf"); best_state=copy.deepcopy(model.state_dict()); t0=time.time()
    for ep in range(epochs):
        cur=lr_f+0.5*(lr-lr_f)*(1+math.cos(math.pi*ep/max(1,epochs-1)))
        for g in opt.param_groups: g["lr"]=cur
        perm=torch.randperm(N,device=DEV); el=0.0
        for s in range(steps):
            idx=perm[s*bs:(s+1)*bs]
            loss=((model(ct[idx])-vt[idx])**2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); el+=loss.item()
        el/=steps
        if el<best: best=el; best_state=copy.deepcopy(model.state_dict())
        if tag and ((ep+1)%max(1,epochs//4)==0 or ep==epochs-1):
            print(f"    [{tag}] ep{ep+1}/{epochs} mse(norm)={el:.3e} best={best:.3e}")
    tt=time.time()-t0; model.load_state_dict(best_state); model.eval()
    rec=np.empty((N,C),np.float32)
    with torch.no_grad():
        for s in range(0,N,200000): rec[s:s+200000]=model(ct[s:s+200000]).clamp(-1,1).cpu().numpy()
    return rec, tt, n_params


def integrate_frame_motion(params, times):
    """params (T,3)=(a,b,c); times (T,). Returns theta(T,), D(T,2)=∫ R(-theta) v0 ds (trapezoid)."""
    T=len(times); a,b,c=params[:,0],params[:,1],params[:,2]
    theta=np.zeros(T); D=np.zeros((T,2))
    def Rm(th,vx,vy):
        ct,st=math.cos(-th),math.sin(-th); return ct*vx-st*vy, st*vx+ct*vy
    for k in range(1,T):
        dt=times[k]-times[k-1]
        theta[k]=theta[k-1]+0.5*(c[k]+c[k-1])*dt
        f0=Rm(theta[k-1],a[k-1],b[k-1]); f1=Rm(theta[k],a[k],b[k])
        D[k,0]=D[k-1,0]+0.5*(f0[0]+f1[0])*dt; D[k,1]=D[k-1,1]+0.5*(f0[1]+f1[1])*dt
    return theta, D


def vpsnr(recon, gt):
    rng=float(gt.max()-gt.min()); mse=float(np.mean((recon.astype(np.float64)-gt.astype(np.float64))**2))
    return float("inf") if mse<=0 else 20*math.log10(rng)-10*math.log10(mse)
def rel_l2(recon, gt): return float(np.linalg.norm(recon-gt)/max(np.linalg.norm(gt),1e-12))


def run(name, vfield, variants, epochs, m, d, lr, batch):
    v=_np(vfield); T,Y,X,C=v.shape
    xs,ys,ts=_norm_axis(X),_norm_axis(Y),_norm_axis(T)
    gt_,gy_,gx_=np.meshgrid(ts,ys,xs,indexing="ij")
    coords_xyt=np.stack([gx_.ravel(),gy_.ravel(),gt_.ravel()],-1).astype(np.float32)
    res=killing_optimization_2d(vfield); u=_np(res.u_field); vhat=_np(res.v_minus_u_field)
    print(f"\n===== {name}  shape={v.shape}  killing (a,b,c)mean={np.round(res.params[1:-1].mean(0),3)} =====")
    out={}

    if "baseline" in variants:
        vals,vmin,sc=norm_pc(v.reshape(-1,C))
        rec,tt,npar=fit_inr_coords(coords_xyt,vals,epochs,m,d,lr,batch,tag="base")
        vr=denorm_pc(rec,vmin,sc).reshape(T,Y,X,C)
        out["baseline"]=dict(psnr=vpsnr(vr,v),rl=rel_l2(vr,v),t=tt,p=npar,extra=0)

    if "observed" in variants:
        vals,vmin,sc=norm_pc(vhat.reshape(-1,C))
        rec,tt,npar=fit_inr_coords(coords_xyt,vals,epochs,m,d,lr,batch,tag="obs")
        vr=denorm_pc(rec,vmin,sc).reshape(T,Y,X,C)+u
        out["observed"]=dict(psnr=vpsnr(vr,v),rl=rel_l2(vr,v),t=tt,p=npar,extra=res.params.size)

    if "warp" in variants:
        times=np.array([vfield.getPhysicalTime(i) for i in range(T)],np.float64)
        theta,D=integrate_frame_motion(res.params,times)
        # physical euler grid
        xph=vfield.domainMinBoundary[0]+vfield.gridInterval[0]*np.arange(X)
        yph=vfield.domainMinBoundary[1]+vfield.gridInterval[1]*np.arange(Y)
        Yg,Xg=np.meshgrid(yph,xph,indexing="ij")   # (Y,X)
        xi=np.empty((T,Y,X,2),np.float32); vco=np.empty((T,Y,X,2),np.float32)
        for ti in range(T):
            cm,sm=math.cos(-theta[ti]),math.sin(-theta[ti])          # R(-theta)
            xi[ti,...,0]=cm*Xg-sm*Yg-D[ti,0]; xi[ti,...,1]=sm*Xg+cm*Yg-D[ti,1]
            vco[ti,...,0]=cm*vhat[ti,...,0]-sm*vhat[ti,...,1]
            vco[ti,...,1]=sm*vhat[ti,...,0]+cm*vhat[ti,...,1]
        # normalize xi globally to [-1,1], t to [-1,1]
        ximin=xi.reshape(-1,2).min(0); ximax=xi.reshape(-1,2).max(0); xisc=np.maximum(ximax-ximin,1e-8)
        xin=(2*(xi.reshape(-1,2)-ximin)/xisc-1).astype(np.float32)
        coords_warp=np.concatenate([xin, coords_xyt[:,2:3]],axis=1)   # (xi_x,xi_y,t) all [-1,1]
        vals,vmin,sc=norm_pc(vco.reshape(-1,C))
        rec,tt,npar=fit_inr_coords(coords_warp,vals,epochs,m,d,lr,batch,tag="warp")
        vco_r=denorm_pc(rec,vmin,sc).reshape(T,Y,X,C)
        vr=np.empty_like(vco_r)
        for ti in range(T):                                          # rotate back R(theta)+u
            cth,sth=math.cos(theta[ti]),math.sin(theta[ti])
            vr[ti,...,0]=cth*vco_r[ti,...,0]-sth*vco_r[ti,...,1]+u[ti,...,0]
            vr[ti,...,1]=sth*vco_r[ti,...,0]+cth*vco_r[ti,...,1]+u[ti,...,1]
        out["warp"]=dict(psnr=vpsnr(vr,v),rl=rel_l2(vr,v),t=tt,p=npar,extra=res.params.size)

    print(f"\n  --- {name} (net m={m} d={d}, {epochs}ep) ---")
    for k in ["baseline","observed","warp"]:
        if k in out:
            o=out[k]; print(f"  {k:9s}: PSNR={o['psnr']:6.2f}dB  relL2={o['rl']:.4f}  time={o['t']:5.1f}s  "
                             f"params={o['p']/1e3:.1f}K"+(f" +{o['extra']} killing" if o['extra'] else ""))
    return out


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--field",default="rfc",choices=["rfc","cylinder"])
    ap.add_argument("--variants",default="baseline,observed,warp")
    ap.add_argument("--epochs",type=int,default=1000)
    ap.add_argument("--m",type=int,default=64); ap.add_argument("--d",type=int,default=4)
    ap.add_argument("--lr",type=float,default=5e-4); ap.add_argument("--batch",type=int,default=200000)
    a=ap.parse_args(); print("device:",DEV)
    if a.field=="rfc": vf=rotation_four_center((64,64),64)
    else:
        vf=NetCDFLoader.load_vector_field2d(r"C:\Users\xingdi\OneDrive - KAUST\WorkingInProcess\FLowVisAssets\flowData2D\cylinder2d.nc",800,960)
        vf.resample2UnsteadyField((128,320,80))
    run(a.field, vf, a.variants.split(","), a.epochs, a.m, a.d, a.lr, a.batch)
