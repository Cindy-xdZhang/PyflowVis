import sys, numpy as np, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, r"C:\Users\xingdi\sources\PyflowVis")
sys.path.insert(0, r"C:\Users\xingdi\AppData\Local\Temp\claude\C--Users-xingdi-sources-PyflowVis\5b7dbb49-2a88-462c-b42d-687217cb2b30\scratchpad")
from rfo_decompose_inr import fit_region, vpsnr, _np
from rfo_final_experiment import global_killing_observer_2d
from FLowUtils.AnalyticalFlowCreator import rotation_four_center
f=rotation_four_center((64,64),64); v=_np(f); T,Y,X,_=v.shape
xph=f.domainMinBoundary[0]+f.gridInterval[0]*np.arange(X); yph=f.domainMinBoundary[1]+f.gridInterval[1]*np.arange(Y)
times=np.array([f.getPhysicalTime(i) for i in range(T)]); mask=np.ones((Y,X),bool)
obs=global_killing_observer_2d(f); obs_p=obs+np.random.default_rng(0).normal(0,1e-9,obs.shape)
def run(observer,ep,lr):
    ys,xs,vx,vy,_,_=fit_region(v,mask,observer,times,xph,yph,ep,64,4,lr,200000,omega_0=30.0)
    vr=np.zeros_like(v); vr[:,ys,xs,0]=vx; vr[:,ys,xs,1]=vy; return vpsnr(vr,v)
print("lr | epochs | PSNR(obs) | PSNR(obs+1e-9) | sensitivity(dB)")
for lr in [1e-4]:
    for ep in [2000, 4000]:
        p0=run(obs,ep,lr); p1=run(obs_p,ep,lr)
        print("%.0e | %6d | %9.4f | %13.4f | %.4f"%(lr,ep,p0,p1,abs(p0-p1)), flush=True)
