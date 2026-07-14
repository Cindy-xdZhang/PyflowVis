import sys, numpy as np, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, r"C:\Users\xingdi\sources\PyflowVis")
sys.path.insert(0, r"C:\Users\xingdi\AppData\Local\Temp\claude\C--Users-xingdi-sources-PyflowVis\5b7dbb49-2a88-462c-b42d-687217cb2b30\scratchpad")
from rfo_final_experiment import global_killing_observer_2d, pushforward_inr_whole
from FLowUtils.AnalyticalFlowCreator import rotation_four_center
f=rotation_four_center((64,64),64); obs=global_killing_observer_2d(f)
obs_p=obs+np.random.default_rng(0).normal(0,1e-9,obs.shape)   # 1e-9 perturbation
print("epochs | PSNR(obs) | PSNR(obs+1e-9) | sensitivity(dB)")
for ep in [300, 800, 1500, 3000]:
    p0,_,_=pushforward_inr_whole(f,obs,ep)
    p1,_,_=pushforward_inr_whole(f,obs_p,ep)
    print("%6d | %9.4f | %13.4f | %.4f"%(ep,p0,p1,abs(p0-p1)), flush=True)
