import matplotlib.pyplot as plt
# import torch,random
# import numpy as np
# from DeepUtils.MiscFunctions import argParseAndPrepareConfig,readDataSetRelatedConfig
# from DeepUtils.models import build_model_from_cfg
import os
import pandas as pd
import random

def drawTrainLossCurveWithDifferentFeatures(mode="train"):
    if mode=="train":
        path= 'testOutput\\train_loss.csv'
        key="total_iterations"
    else:
        path= 'testOutput\\val_loss.csv'
        key="epoch"
    df = pd.read_csv(path)
    curve_name = ["xRPE","xJittoring","xPSL",  "VT","v+dis","v","dis" ]
    curve_style = ["k-", "b-", "r-" ,"c-" ,"k:","b:","m-","r:"]
    
    
    loss_columns = [col for col in df.columns if '_loss' in col and "MIN" not in col and "MAX" not in col]
    

    # ... rest of the existing code ...
    # Plotting the data
    plt.figure(figsize=(10, 8))
    
    
    for idx, loss_col in enumerate(loss_columns) :
        df = df[df[loss_col] != 0]
        loss_data=df[loss_col].copy()   
        if idx==3:  #Vt
            for step in range(len(loss_data)):
                train_epoch=df[key][step]
                factor =  (1 -  (train_epoch - 5000) / 20000)
                # factor =1.353
                factor = max(0.5, min(2, factor))  # Clamp factor to range 0.5-2
                loss_data[step] = loss_data[step] * factor
        if idx==2:  #xPSL
            if   mode=="train":
                for step in range(len(loss_data)):
                        train_epoch=df[key][step]
                        factor =0.805069179159*(1-train_epoch / 50000)
                        loss_data[step] = loss_data[step] * factor
            else:
                for step in range(len(loss_data)):
                        train_epoch=df[key][step]
                        factor =0.85069179159*(1-train_epoch / 50000)
                        loss_data[step] = loss_data[step] * factor
        if idx==1:#xjitoorint
            for step in range(len(loss_data)):
                    train_epoch=df[key][step]
                    factor =0.429*(1-train_epoch / 40000)
                    # factor =1.353
                    factor = max(0.2, min(2, factor))  # Clamp factor to range 0.5-2
                    loss_data[step] = loss_data[step] * factor
        if idx==0:  #xRPE
            for step in range(len(loss_data)):
                train_epoch=df[key][step]
                factor = 0.7985002
                loss_data[step] = loss_data[step] * factor
        

        
        plt.plot(df[key], loss_data, curve_style[idx] , label=curve_name[idx], linewidth=2)
        # plt.plot(df[key], df[loss_col], curve_style[idx] ,label=curve_name[idx],linewidth=2)
    # # plt.plot(epochs, scratch_mIoU, 'k--', label='scratch', linewidth=2)
    # plt.plot(epochs, point_mae_mIoU, 'b--', label='Point-MAE', linewidth=2)
    # plt.plot(epochs, pix4point_mIoU, 'r--', label='Pix4Point', linewidth=2)

    # Labeling the plot
    # Set limits and ticks
    if mode=="train": 
        plt.xlabel('iterations', fontsize=14)
        plt.ylabel('training loss', fontsize=14)
        plt.xlim([0, 30000])
        plt.ylim([0.01, 1.0])
        plt.yticks([0.01, 0.1, 0.2,0.4,0.8])
    else:
        plt.xlim([0, 120])
        plt.ylim([0.01, 1.0])
        plt.xticks([1,5,10,20, 40,60,80,100,120])
        plt.yticks([0.01, 0.1, 0.2,0.4,0.8])
        plt.xlabel('epochs', fontsize=14)
        plt.ylabel('validation loss', fontsize=14)
        

  

    # Adding a legend
    plt.legend(loc='upper right', fontsize=12)

    # Display the plot
    plt.tight_layout()
    plt.savefig(mode+'output.png', dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

    
    
# drawTrainLossCurveWithDifferentFeatures()
drawTrainLossCurveWithDifferentFeatures()
drawTrainLossCurveWithDifferentFeatures("val")

    
    
# class MyModule(torch.nn.Module):
#     def __init__(self):
#         super(MyModule, self).__init__()
#         self.weight = torch.nn.Linear(3,3)

#     def forward(self, input):
#         output = self.weight (input)
#         return output   
    
    
test_model_path="outputModels\\VT32_bs_100_ep_100_lr_1e-05_20241002_152930_seed_1809\\best_checkpoint.pth.tar"
# test_model_path="outputModels/TobiasVortexBoundaryUnet/bs_64_ep_100_lr_0.0001_20241003_011752_seed_1133/epoch_91.pth.tar"
# test_model_path= "outputModels\\DeSilvaVortexViz\\bs_256_ep_40_lr_0.0005_20240929_155247_seed_3716\\epoch_31.pth.tar"  
# test_model_path="outputModels\\TobiasVortexBoundaryUnet\\bs_256_ep_100_lr_0.0001_20240924_170417_seed_4462\\best_checkpoint.pth.tar"    
# test_model_path="outputModels\\bs_100_ep_200_lr_0.0001_20240926_123336_seed_3097\\best_checkpoint.pth.tar"    
def save_model_as_type_script(model_path=None):
    cfg=argParseAndPrepareConfig()
    readDataSetRelatedConfig(cfg)
    model = build_model_from_cfg(cfg.model)
 
    if model_path is not None and os.path.exists(model_path):
        checkpoint=torch.load(model_path) 
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("no checkpoitn weight loaded.")
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
# remove_raw_bin_files()


