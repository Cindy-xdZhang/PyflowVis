import sys; sys.path.insert(0, r"C:\Users\xingdi\sources\PyflowVis")
import numpy as np
from FLowUtils.AnalyticalFlowCreator import rotation_four_center
from FLowUtils.flowDatasetUtils.NetCDF_AmiraLoader import NetCDFLoader
from FLowUtils.referenceFrameDecompose import decompose_reference_frame_2d
from FLowUtils.ReferenceFrameViz import plot_decomposition
OUT=r"C:\Users\xingdi\AppData\Local\Temp\claude\C--Users-xingdi-sources-PyflowVis\5b7dbb49-2a88-462c-b42d-687217cb2b30\scratchpad"

# RFC (square 64x64)
f=rotation_four_center((64,64),64); dec=decompose_reference_frame_2d(f,k=2)
plot_decomposition(dec, cuts=(1,2,3), save_path=OUT+r"\viz_rfc.png")

# boussinesq (tall 225x75)
vf=NetCDFLoader.load_vector_field2d(r"C:\Users\xingdi\OneDrive - KAUST\WorkingInProcess\FLowVisAssets\flowData2D\boussinesq.nc",800,960)
vf.resample2UnsteadyField((128,75,225)); dec2=decompose_reference_frame_2d(vf,k=2)
plot_decomposition(dec2, cuts=(2,3,4,6), save_path=OUT+r"\viz_bouss.png")

# cylinder (wide 320x80)
cf=NetCDFLoader.load_vector_field2d(r"C:\Users\xingdi\OneDrive - KAUST\WorkingInProcess\FLowVisAssets\flowData2D\cylinder2d.nc",800,960)
cf.resample2UnsteadyField((128,320,80)); dec3=decompose_reference_frame_2d(cf,k=2)
plot_decomposition(dec3, cuts=(2,3,4,6), save_path=OUT+r"\viz_cyl.png")
print("done")
