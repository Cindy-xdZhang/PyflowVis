import sys, numpy as np, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, r"C:\Users\xingdi\sources\PyflowVis")
sys.path.insert(0, r"C:\Users\xingdi\AppData\Local\Temp\claude\C--Users-xingdi-sources-PyflowVis\5b7dbb49-2a88-462c-b42d-687217cb2b30\scratchpad")
import rfo_decompose_inr as DEC
from ff_inr import fit_ffrelu
from rfo_decompose_inr import vpsnr, _np, fit_region
from rfo_final_experiment import global_killing_observer_2d
from FLowUtils.AnalyticalFlowCreator import rotation_four_center
SCALE=[16.0]
def mk_fit(scale):
    return lambda coords,values,epochs,m,d,lr,batch,seed=0,omega_0=None: fit_ffrelu(coords,values,epochs,lr,batch,512,4,256,scale,seed)
f=rotation_four_center((64,64),64); v=_np(f); T,Y,X,_=v.shape
xph=f.domainMinBoundary[0]+f.gridInterval[0]*np.arange(X); yph=f.domainMinBoundary[1]+f.gridInterval[1]*np.arange(Y)
times=np.array([f.getPhysicalTime(i) for i in range(T)]); mask=np.ones((Y,X),bool)
obs=global_killing_observer_2d(f); obs_p=obs+np.random.default_rng(0).normal(0,1e-9,obs.shape)
def run(observer,ep):
    ys,xs,vx,vy,_,_=fit_region(v,mask,observer,times,xph,yph,ep,64,4,1e-3,200000)
    vr=np.zeros_like(v); vr[:,ys,xs,0]=vx; vr[:,ys,xs,1]=vy; return vpsnr(vr,v)
print("scale | epochs | PSNR(obs) | sensitivity(dB) | params")
for sc in [8.0,16.0,30.0]:
    DEC.fit_inr=mk_fit(sc)
    for ep in [4000]:
        p0=run(obs,ep); p1=run(obs_p,ep)
        import ff_inr; npar=sum(p.numel() for p in ff_inr.FFReLU(3,2,512,4,256,sc).parameters())
        print("%5.0f | %6d | %9.4f | %.4f | %.2fM"%(sc,ep,p0,abs(p0-p1),npar/1e6),flush=True)
