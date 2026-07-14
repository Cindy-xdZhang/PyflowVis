import sys, numpy as np, math, copy, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, r"C:\Users\xingdi\sources\PyflowVis")
sys.path.insert(0, r"C:\Users\xingdi\AppData\Local\Temp\claude\C--Users-xingdi-sources-PyflowVis\5b7dbb49-2a88-462c-b42d-687217cb2b30\scratchpad")
import torch, torch.nn as nn
import rfo_decompose_inr as DEC
from rfo_decompose_inr import vpsnr, _np, fit_region, DEV
from rfo_final_experiment import global_killing_observer_2d
from FLowUtils.AnalyticalFlowCreator import rotation_four_center

class FFReLU(nn.Module):
    def __init__(self, in_dim=3, out_dim=2, hidden=256, layers=4, n_freq=128, scale=8.0, seed=0):
        super().__init__()
        g=torch.Generator().manual_seed(seed)
        self.register_buffer("B", torch.randn(in_dim, n_freq, generator=g)*scale)
        L=[nn.Linear(2*n_freq,hidden), nn.ReLU()]
        for _ in range(layers): L+=[nn.Linear(hidden,hidden), nn.ReLU()]
        L+=[nn.Linear(hidden,out_dim)]
        self.net=nn.Sequential(*L)
    def forward(self,x):
        p=2*math.pi*(x@self.B); ff=torch.cat([torch.sin(p),torch.cos(p)],-1); return self.net(ff)

def fit_inr_ff(coords, values, epochs, m, d, lr, batch, seed=0, omega_0=None):
    torch.manual_seed(seed); np.random.seed(seed)
    N,C=values.shape; ct=torch.from_numpy(coords).to(DEV); vt=torch.from_numpy(values).to(DEV)
    model=FFReLU(3,C,seed=seed).to(DEV); npar=sum(p.numel() for p in model.parameters())
    opt=torch.optim.Adam(model.parameters(),lr=lr); bs=min(batch,N); steps=max(1,N//bs); lr_f=lr*0.1
    best=float("inf"); bs_state=copy.deepcopy(model.state_dict()); ev=max(1,epochs//50)
    for ep in range(epochs):
        cur=lr_f+0.5*(lr-lr_f)*(1+math.cos(math.pi*ep/max(1,epochs-1)))
        for g in opt.param_groups: g["lr"]=cur
        perm=torch.randperm(N,device=DEV)
        for s in range(steps):
            idx=perm[s*bs:(s+1)*bs]; loss=((model(ct[idx])-vt[idx])**2).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if (ep+1)%ev==0 or ep==epochs-1:
            with torch.no_grad(): fl=sum(((model(ct[s:s+400000])-vt[s:s+400000])**2).sum().item() for s in range(0,N,400000))/N
            if fl<best: best=fl; bs_state=copy.deepcopy(model.state_dict())
    model.load_state_dict(bs_state); model.eval()
    rec=np.empty((N,C),np.float32)
    with torch.no_grad():
        for s in range(0,N,200000): rec[s:s+200000]=model(ct[s:s+200000]).clamp(-1,1).cpu().numpy()
    return rec,0.0,npar

DEC.fit_inr=fit_inr_ff
f=rotation_four_center((64,64),64); v=_np(f); T,Y,X,_=v.shape
xph=f.domainMinBoundary[0]+f.gridInterval[0]*np.arange(X); yph=f.domainMinBoundary[1]+f.gridInterval[1]*np.arange(Y)
times=np.array([f.getPhysicalTime(i) for i in range(T)]); mask=np.ones((Y,X),bool)
obs=global_killing_observer_2d(f); obs_p=obs+np.random.default_rng(0).normal(0,1e-9,obs.shape)
def run(observer,ep):
    ys,xs,vx,vy,_,_=fit_region(v,mask,observer,times,xph,yph,ep,64,4,1e-3,200000)
    vr=np.zeros_like(v); vr[:,ys,xs,0]=vx; vr[:,ys,xs,1]=vy; return vpsnr(vr,v)
print("FF+ReLU params ~%.2fM  epochs | PSNR(obs) | PSNR(obs+1e-9) | sensitivity(dB)"%(sum(p.numel() for p in FFReLU().parameters())/1e6))
for ep in [1000,2000]:
    p0=run(obs,ep); p1=run(obs_p,ep); print("%6d | %9.4f | %13.4f | %.4f"%(ep,p0,p1,abs(p0-p1)),flush=True)
