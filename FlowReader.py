import json
import numpy as np
import os
from flowCreator import LICAlgorithm
from VectorField2d import SteadyVectorField2D
import matplotlib.pyplot as plt

def read_json_file(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

def read_binary_file(filepath, dtype=np.float32):
    with open(filepath, 'rb') as file:
        data = np.fromfile(file, dtype=dtype)
    return data

def load_results(directory_path):
    # Load main JSON file
    # main_json_file = os.path.join(directory_path, 'OptConnection-resampled_gn_lscg_VinitPad1stage_DiC_step1_kcrp_1_0.4_0.1_0_mi5_fi5_si5000.json')
    # main_data = read_json_file(main_json_file)
    # file="OptConnection-resampled_gn_lscg_VinitPad1stage_DiC_step1_kcrp_1_0.4_0.1_0_mi5_fi5_si5000.bin"
    results = {}
    # Load binary data
    # binary_path = os.path.join(directory_path, file)
    # res_name = os.path.splitext(file)[0]
    data = read_binary_file(directory_path)[2:]
    return data

# Example usage
directory_path = 'CppProjects/data/64_64/velocity_field_ 0rc_1.000000n_2.000000S[1]_velocity.bin'
results = load_results(directory_path)
steadyVectorField2D = SteadyVectorField2D( 64, 64, [-2.0, -2.0], [2.0, 2.0])
steadyVectorField2D.field = results.reshape(64, 64, 2)


texture = np.random.rand(64, 64)
lic_result=LICAlgorithm(  texture  ,steadyVectorField2D ,128,128,0.01,128)
lic_normalized =255* (lic_result - np.min(lic_result)) / (np.max(lic_result) - np.min(lic_result))
    
# Step 4: Convert to an image and save
plt.imshow(lic_normalized, cmap='gray')
plt.axis('off')  # Optional: Remove axis for a cleaner image
plt.savefig("vector_field_lic3.png", bbox_inches='tight', pad_inches=0)

# print("Main JSON Data:", results)
# for key, value in results.items():
#     print(f"Result Name: {key}")
#     print("Data:", value)


import torch
# create torch dataset using the load result function:
class VectorFieldDataset(torch.utils.data.Dataset):
    def __init__(self, directory_path):
        self.data = load_results(directory_path)
        self.steadyVectorField2D = SteadyVectorField2D( 64, 64, [-2.0, -2.0], [2.0, 2.0])
        self.steadyVectorField2D.field = self.data.reshape(64, 64, 2)
        self.texture = np.random.rand(64, 64)
        self.lic_result=LICAlgorithm(  self.texture  ,self.steadyVectorField2D ,128,128,0.01,128)
        self.lic_normalized =255* (self.lic_result - np.min(self.lic_result)) / (np.max(self.lic_result) - np.min(self.lic_result))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.lic_normalized


