import matplotlib.pyplot as plt
import torch,random
import numpy as np
import os
import datetime
from DeepUtils.dataset import build_dataloader_from_cfg
from FLowUtils.VectorField2d import UnsteadyVectorField2D
from DeepUtils.dataset import UnsteadyVastisDataset
from PIL import Image
from FLowUtils.vortexCriteria import *
from FLowUtils.LicRenderer import *
from DeepUtils.MiscFunctions import argParseAndPrepareConfig,readDataSetRelatedConfig
from DeepUtils.models import build_model_from_cfg
def draw_Curve():
    # Example data
    epochs = np.arange(1, 601)
    scratch_mIoU = 20 + 40 * (1 - np.exp(-0.03 * (epochs-10)))
    point_mae_mIoU = 20 + 45 * (1 - np.exp(-0.035 * (epochs-10)))
    pix4point_mIoU = 20 + 50 * (1 - np.exp(-0.04 * (epochs-10)))

    # Plotting the data
    plt.figure(figsize=(6, 5))

    plt.plot(epochs, scratch_mIoU, 'k--', label='scratch', linewidth=2)
    plt.plot(epochs, point_mae_mIoU, 'b--', label='Point-MAE', linewidth=2)
    plt.plot(epochs, pix4point_mIoU, 'r--', label='Pix4Point', linewidth=2)

    # Labeling the plot
    plt.xlabel('finetuning epochs', fontsize=14)
    plt.ylabel('mIoU', fontsize=14)

    # Set limits and ticks
    plt.xlim([10, 600])
    plt.ylim([20, 70])

    # Adding a legend
    plt.legend(loc='lower right', fontsize=12)

    # Display the plot
    plt.tight_layout()
    plt.show()
    
class MyModule(torch.nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.weight = torch.nn.Linear(3,3)

    def forward(self, input):
        output = self.weight (input)
        return output   
    
    
test_model_path="outputModels\\PT_bs_100_ep_200_lr_0.0001_20240928_202906_seed_3481_FullRealD\\best_checkpoint.pth.tar"


# test_model_path= "outputModels\\DeSilvaVortexViz\\bs_256_ep_40_lr_0.0005_20240929_155247_seed_3716\\epoch_31.pth.tar"  
# test_model_path="outputModels\\TobiasVortexBoundaryUnet\\bs_256_ep_100_lr_0.0001_20240924_170417_seed_4462\\best_checkpoint.pth.tar"    
# test_model_path="outputModels\\bs_100_ep_200_lr_0.0001_20240926_123336_seed_3097\\best_checkpoint.pth.tar"    
def save_model_as_type_script(model_path):
    cfg=argParseAndPrepareConfig()
    readDataSetRelatedConfig(cfg)
    model = build_model_from_cfg(cfg.model)
 
    if model_path is not None and os.path.exists(model_path):
        checkpoint=torch.load(model_path) 
        model.load_state_dict(checkpoint['state_dict'])
        inputPIc =( torch.rand(1, 1,65,65), torch.rand(1, 1000))

        # traced_script_module = torch.jit.trace(model,inputPIc)
        traced_script_module =torch.jit.script(model)
        print(traced_script_module.encoder.code)
        outputPath=model_path.replace(".tar","traced.pt")
        traced_script_module.save(outputPath)
        
# save_model_as_type_script(test_model_path)



# Function to load the model and perform inference
# def load_model_and_infer(model_path):
#     cfg=argParseAndPrepareConfig()
#     readDataSetRelatedConfig(cfg)
#     model = build_model_from_cfg(cfg.model)
#     if model_path is not None and os.path.exists(model_path):
#         checkpoint=torch.load(model_path) 
#         model.load_state_dict(checkpoint['state_dict'])
#     model.eval()
#     return model



# def pybind_api_inter(input_tensor):
#     model=load_model_and_infer()
#     with torch.no_grad():
#         output = model(input_tensor)
#         return output
        


def remove_raw_bin_files():
    for root, dirs, files in os.walk('CppProjects\\data\\RealDataCross4_256samplesPerGrid'):
        for file in files:
            if file.endswith('.bin') and not file.endswith('_pathline.bin'):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
remove_raw_bin_files()