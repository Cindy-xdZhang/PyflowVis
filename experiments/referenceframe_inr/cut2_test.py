import sys, numpy as np, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, r"C:\Users\xingdi\sources\PyflowVis")
sys.path.insert(0, r"C:\Users\xingdi\AppData\Local\Temp\claude\C--Users-xingdi-sources-PyflowVis\5b7dbb49-2a88-462c-b42d-687217cb2b30\scratchpad")
from rfo_final5 import proposed_overlap, BASELINE
from FLowUtils.referenceFrameDecompose import decompose_reference_frame_2d
from FLowUtils.flowDatasetUtils.NetCDF_AmiraLoader import NetCDFLoader
from ff_inr import FFReLU
bf = NetCDFLoader.load_vector_field2d(r"C:\Users\xingdi\OneDrive - KAUST\WorkingInProcess\FLowVisAssets\flowData2D\boussinesq.nc", 800, 960); bf.resample2UnsteadyField((128,75,225))
dec = decompose_reference_frame_2d(bf, k=2)
print(f"boussinesq baseline(h930)={BASELINE['boussinesq']}dB")
for N,h in [(2,640)]:
    p,tot,ksc = proposed_overlap(bf, dec, N, h)
    npar1 = sum(pp.numel() for pp in FFReLU(3,2,h,4,256,8.0).parameters())
    print(f"  cut={N} (h={h}, per-INR={npar1/1e6:.2f}M): PSNR={p:.2f}dB  total={tot/1e6:.2f}M  vsBaseline={p-BASELINE['boussinesq']:+.2f}dB", flush=True)
