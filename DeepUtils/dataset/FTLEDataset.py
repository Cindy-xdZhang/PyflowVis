import numpy as np
import os
from FLowUtils.VectorField2d import *
import torch,time,tqdm
from .build import DATASETS
from .data_utils import *
import re,glob



def extract_number_fromFileName(filename):
    # Extract time from FTLE_DoubleGyre2D10.bin format
    match = re.search(r'DoubleGyre2D(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0

# Function to load a flow map from a file
#the first 3 int +2 float are meta inforamtion
#pathlineDataType = np.dtype([('x', np.float32), ('y', np.float32), ('timestep', np.int32)])    
def readPathlineDataFromBinaryFileWithMetaHeading(filename):
    with open(filename, 'rb') as file:
        # Read the width, height, depth, and time (each as 4-byte integers)
        width = np.fromfile(file, dtype=np.int32, count=1)[0]
        height = np.fromfile(file, dtype=np.int32, count=1)[0]
        numberOfTimeSteps = np.fromfile(file, dtype=np.int32, count=1)[0]
        
        minTime = np.fromfile(file, dtype=np.float32, count=1)[0]
        maxTime = np.fromfile(file, dtype=np.float32, count=1)[0]
        dt = np.fromfile(file, dtype=np.float32, count=1)[0]
        data = np.fromfile(file, dtype=np.float32)
    steps_per_pathline = numberOfTimeSteps
    num_pathlines = width * height
    values_per_step = 3
    pathlinesData = data.reshape(num_pathlines, steps_per_pathline, values_per_step)
    return pathlinesData

def load_ftle_data(filename):
    with open(filename, 'rb') as file:
        # Read the width, height, depth, and time (each as 4-byte integers)
        width = np.fromfile(file, dtype=np.int32, count=1)[0]
        height = np.fromfile(file, dtype=np.int32, count=1)[0]
        depth = np.fromfile(file, dtype=np.int32, count=1)[0]
        time = np.fromfile(file, dtype=np.int32, count=1)[0]
    
        print(f"Width: {width}, Height: {height}, Depth: {depth}, Time: {time}")

        # Read and discard the 8-byte header (saved by Cereal)
        file.read(8)
        data = np.fromfile(file, dtype=np.float32)
        # print("Values:")
        # print(data)
        
    # Reshape the data into a 2D grid (height x width)
    ftle_data = data.reshape((height, width))
    return ftle_data, time



DoubleGryle_Xmin=0.0
DoubleGryle_Xmax=2.0
DoubleGryle_Ymin=0.0
DoubleGryle_Ymax=1.0

#scalarFieldData is np.array((height, width)) Pathline shape is [num_pathlines, steps_per_pathline, values_per_step]
#use the first timestep's position of pathline as its coordinate and bilinear interpolate scalar value for every pathline.
def resample_scalar_field_2_pathline(scalarFieldData, Pathlines, stepToIdentifyPathline=0):
    """
    Resample scalar field data at pathline initial positions using bilinear interpolation
    Args:
        scalarFieldData: np.array of shape (height, width) containing scalar values
        Pathlines: np.array of shape (num_pathlines, steps_per_pathline, values_per_step)
    Returns:
        np.array of shape (num_pathlines,) containing interpolated scalar values
    """
    height, width = scalarFieldData.shape    
    # Get initial positions of pathlines (x, y coordinates at t=0)
    initial_positions = Pathlines[:, stepToIdentifyPathline, :2]  # Shape: (num_pathlines, 2)
    
    # Convert to grid coordinates (assuming normalized coordinates)
    x = initial_positions[:, 0] *(1.0/DoubleGryle_Xmax)* (width - 1)
    y = initial_positions[:, 1] * (height - 1)
    
    # Get the four nearest neighbor indices
    x0 = np.floor(x).astype(int)
    x1 = np.minimum(x0 + 1, width - 1)
    y0 = np.floor(y).astype(int)
    y1 = np.minimum(y0 + 1, height - 1)
    
    # Calculate interpolation weights
    wx = x - x0
    wy = y - y0
    
    # Perform bilinear interpolation
    c00 = scalarFieldData[y0, x0]
    c10 = scalarFieldData[y1, x0]
    c01 = scalarFieldData[y0, x1]
    c11 = scalarFieldData[y1, x1]
    
    # Interpolate along x direction
    c0 = c00 * (1 - wx) + c01 * wx
    c1 = c10 * (1 - wx) + c11 * wx
    
    # Interpolate along y direction
    interpolated_values = c0 * (1 - wy) + c1 * wy
    
    return interpolated_values
    
    
    




@DATASETS.register_module()
class Pathline2FTLEDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir,split, transform,**kwargs):
        self.directory_path=data_dir
        self.dataName=[]
        self.data=[]
        self.label=[]
        self.dastasetMetaInfo={}
        self.split=split
        self.preLoading(split)
        self.transform = transform

    def getSampleName(self,sampleIdx):        
        return self.dataName[sampleIdx]    
    
    def preLoading(self,mode):                
        print(f"Preloading [{mode}] data......")
        start = time.time()         
        split_str =mode if mode!="val" else "validation"
        target_directory_path=os.path.join(self.directory_path,split_str) 
        # If the target directory doesn't exist, try alternatives
        if not os.path.exists(target_directory_path):
            if mode == "val":
                # For validation, try using the test directory
                target_directory_path = os.path.join(self.directory_path, "test")
            elif mode == "test":
                # For test, try using the validation directory
                target_directory_path = os.path.join(self.directory_path, "validation")
                
                
        if not os.path.exists(target_directory_path):
            raise ValueError(f"Could not find a valid directory for mode: {mode}")
        
        print(f"using folder {target_directory_path} for mode:{mode}")
        
        rc_n_subfoders=[os.path.join(target_directory_path,folder) for folder in os.listdir(target_directory_path) if os.path.isdir(os.path.join(target_directory_path, folder))]
        for folder_name in tqdm.tqdm(rc_n_subfoders) :           
                self.loadOneTaskFolder(folder_name)          
        #logging dataset information    and time consuming of preloading data
        elapsed = time.time()
        elapsed = elapsed - start
        print(f"Preloading  [{mode}] data......done")
        print(f"Total number of [{mode}] data:{len(self.data)}, took {elapsed} seconds")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data=self.data[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data , self.label[idx]
    

    def loadOneTaskFolder(self,sub_folder:str):
        """son class need to overwrite this function
        """
        LowResPathlineDataFile=os.path.join(sub_folder,"LowRes_Pathlines","PATHLINES_DoubleGyre2D.bin")
        pathlinesData=readPathlineDataFromBinaryFileWithMetaHeading(LowResPathlineDataFile)
        FTLEFiles= list(glob.glob(os.path.join(sub_folder ,"HighRes_FTLE",'*.bin')))
        replicate=10
        for FTLE_file in FTLEFiles:
            time_step=extract_number_fromFileName(FTLE_file)
            pathlinesData=pathlinesData[:,0:time_step,:]
            ftle_scalar_field, time=load_ftle_data(FTLE_file)
            pathline_label=resample_scalar_field_2_pathline(ftle_scalar_field,pathlinesData)
            
            for i in range(replicate):
                theta = np.random.uniform(0, 2*np.pi)
                tx = np.random.uniform(-0.05, 0.05)
                ty = np.random.uniform(-0.05, 0.05)
                rotation_matrix = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
                ])
                # transformed_pathlines = pathlinesData.copy()
                # for t in range(pathlinesData.shape[1]):
                #     positions = pathlinesData[:, t, :2]
                #     transformed_positions = np.dot(positions, rotation_matrix.T) + np.array([tx, ty])
                #     transformed_pathlines[:, t, :2] = transformed_positions
                transformed_pathlines=pathlinesData.copy()
                transformed_pathlines[:,:,:2] = np.einsum('ij,ntj->nti', rotation_matrix, pathlinesData[:, :, :2]) + np.array([tx, ty])
                transformed_pathlines[:,:,2]=pathlinesData[:,:,2]
            
            
                PathlineDataShapeLKC_float32=torch.tensor(transformed_pathlines).permute(1,0,2)            
                Label_float32= torch.tensor(pathline_label.astype(np.float32))            
                self.data.append(PathlineDataShapeLKC_float32)
                self.label.append(Label_float32)
            
            
        
        
            
