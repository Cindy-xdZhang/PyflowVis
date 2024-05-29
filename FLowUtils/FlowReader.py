import json
import numpy as np
import os
from typeguard import typechecked
from VectorField2d import UnsteadyVectorField2DNp,SteadyVectorField2D,UnsteadyVectorField2D
import matplotlib.pyplot as plt


def bilinear_interpolate(vector_field, x, y):
    """
    Perform bilinear interpolation for a 2D vector field.

    Parameters:
    - vector_field: np.ndarray of shape (Ydim, Xdim, 2), the 2D vector field.
    - x, y: float, the fractional coordinates at which to interpolate the vector.

    Returns:
    - interpolated_vector: The interpolated vector at position (x, y).
    """
    
    # Ensure x, y are within the bounds of the vector field
    x = np.clip(x, 0, vector_field.shape[1] - 1)
    y = np.clip(y, 0, vector_field.shape[0] - 1)

    # Get the integer parts of x, y
    x0 = int(x)
    y0 = int(y)

    # Ensure that we don't go out of bounds in the interpolation
    x1 = min(x0 + 1, vector_field.shape[1] - 1)
    y1 = min(y0 + 1, vector_field.shape[0] - 1)

    # Calculate the fractional parts of x, y
    tx = x - x0
    ty = y - y0

    # Get the vectors at the corner points
    v00 = vector_field[y0, x0,:]
    v01 = vector_field[y0, x1,:]
    v10 = vector_field[y1, x0,:]
    v11 = vector_field[y1, x1,:]

    # Perform bilinear interpolation
    a = v00 * (1 - tx) + v01 * tx
    b = v10 * (1 - tx) + v11 * tx
    interpolated_vector = a * (1 - ty) + b * ty

    return interpolated_vector

def LICAlgorithm(texture:np.ndarray, vecfield: SteadyVectorField2D, resultImageSizeX, resultImageSizeY,stepSize:float, MaxIntegrationSteps:int):
    """
    A simplified LIC algorithm to visualize the flow of a 2D vector field slice. 
    texture as same size as the vector field slice, vecfield is the vector field slice,
    resultImageSizeX and resultImageSizeY are the size of the output image, 
    stepSize is the step size for integration, and MaxIntegrationSteps is the maximum number of integration steps to take.
    """
    if texture.ndim == 2:
        Ydim, Xdim = texture.shape
        texture = texture[:, :, np.newaxis]  # Add a dummy channel dimension
    else:
        Ydim, Xdim, _ = texture.shape

    output_texture = np.zeros((resultImageSizeY, resultImageSizeX), dtype=np.float32)
    vecfieldData=vecfield.field
    domainRangeX=vecfield.domainMaxBoundary[0]-vecfield.domainMinBoundary[0]
    domainRangeY=vecfield.domainMaxBoundary[1]-vecfield.domainMinBoundary[1]

    inverse_grid_interval_x=1/float(vecfield.gridInterval[0])
    inverse_grid_interval_y=1/float(vecfield.gridInterval[1])
    for y in range(resultImageSizeY):
        for x in range(resultImageSizeX):

            ratioX=float(x)/float(resultImageSizeX)
            ratioY=float(y)/float(resultImageSizeY)
            accum_value = 0.0
            accum_count = 0
            
            # Trace forward
            #pos (x,y)
            pos = np.array([ratioX* domainRangeX+vecfield.domainMinBoundary[0], ratioY* domainRangeY+vecfield.domainMinBoundary[1]], dtype=np.float32)

            for _ in range(MaxIntegrationSteps):
                floatIndexX=(pos[0]-vecfield.domainMinBoundary[0])*inverse_grid_interval_x
                floatIndexY=(pos[1]-vecfield.domainMinBoundary[1])*inverse_grid_interval_y
                if not (0 <= floatIndexX < Xdim and 0 <= floatIndexY < Ydim):
                    break  # Stop if we move outside the texture bounds

                accum_value += bilinear_interpolate(texture, floatIndexX, floatIndexY)
                accum_count += 1
                vec =bilinear_interpolate(vecfieldData,  floatIndexX, floatIndexY)
                pos += vec * stepSize
                
            # Trace backward
            pos = np.array([ratioX* domainRangeX+vecfield.domainMinBoundary[0], ratioY* domainRangeY+vecfield.domainMinBoundary[1]], dtype=np.float32)
            for _ in range(MaxIntegrationSteps):
                floatIndexX=(pos[0]-vecfield.domainMinBoundary[0])*inverse_grid_interval_x
                floatIndexY=(pos[1]-vecfield.domainMinBoundary[1])*inverse_grid_interval_y
                if not (0 <= floatIndexX < Xdim and 0 <= floatIndexY < Ydim):
                    break  # Stop if we move outside the texture bounds

                accum_value += bilinear_interpolate(texture, floatIndexX, floatIndexY)
                accum_count += 1
                vec =bilinear_interpolate(vecfieldData,  floatIndexX, floatIndexY)

                pos -= vec * stepSize
            
            # Compute the average value along the path
            if accum_count > 0:
                output_texture[y, x] = accum_value / accum_count
    
    return output_texture


def LICImage_OFFLINE_RENDERING(vecfield: UnsteadyVectorField2DNp|UnsteadyVectorField2D, timeSlice=0,stepSize=0.01, MaxIntegrationSteps=128):
    """
    Render a steady 2D vector field as an LIC image and save to a PNG file.
    """
    # Step 1: Initialize a texture for the LIC process, often random noise
    texture = np.random.rand(vecfield.Ydim, vecfield.Xdim)
    # Detach the tensor, move it to CPU, and convert to NumPy
    VecFieldSlice=vecfield.getSlice(timeSlice)
    # Step 2: Prepare your LIC implementation here. This is a placeholder for
    # the process of integrating along the vector field to modify the texture.
    # You'll need to replace this with your actual LIC algorithm.
    lic_result = LICAlgorithm(texture, VecFieldSlice, 128,128,stepSize, MaxIntegrationSteps)
    
    # Step 3: Normalize the LIC result for visualization
    lic_normalized = (lic_result - np.min(lic_result)) / (np.max(lic_result) - np.min(lic_result))
    
    # Step 4: Convert to an image and save
    plt.imshow(lic_normalized, cmap='gray')
    plt.axis('off')  # Optional: Remove axis for a cleaner image
    plt.savefig("vector_field_lic.png", bbox_inches='tight', pad_inches=0)


def read_json_file(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

def read_binary_file(filepath, dtype=np.float32) -> np.ndarray:
    with open(filepath, 'rb') as file:
        data = np.fromfile(file, dtype=dtype)
        if dtype == np.float32:
            data=data[2:]
        elif dtype == np.float64:
            data=data[1:]        
    return data

def load_results(directory_path) -> np.ndarray:
    # Load main JSON file
    # main_json_file = os.path.join(directory_path, 'OptConnection-resampled_gn_lscg_VinitPad1stage_DiC_step1_kcrp_1_0.4_0.1_0_mi5_fi5_si5000.json')
    # main_data = read_json_file(main_json_file)
    # file="OptConnection-resampled_gn_lscg_VinitPad1stage_DiC_step1_kcrp_1_0.4_0.1_0_mi5_fi5_si5000.bin"
    results = {}
    # Load binary data
    # binary_path = os.path.join(directory_path, file)
    # res_name = os.path.splitext(file)[0]
    data = read_binary_file(directory_path)
    return data

def loadOneFlowEntry(binPath) -> tuple[UnsteadyVectorField2DNp, list]:
    raw_Binary = load_results(binPath)
    

    #get meta information
    meta_file = binPath.replace('.bin', 'meta.json')
    metaINFo=read_json_file(meta_file)
    Xdim=metaINFo['Xdim']
    Ydim=metaINFo['Ydim']
    #observe and unsteady info 
    time_steps=metaINFo['observerfieldDeform']['timeSteps']
    tmin=metaINFo['observerfieldDeform']['tmin']
    tmax=metaINFo['observerfieldDeform']['tmax']
    dominMinBoundary=[metaINFo['domainMinBoundary']["value0"],metaINFo['domainMinBoundary']["value1"],tmin] 
    dominMaxBoundary=[metaINFo['domainMaxBoundary']["value0"],metaINFo['domainMaxBoundary']["value1"],tmax]
    # print(metaINFo)
    labelReferenceFrameABC=metaINFo['observerfieldDeform']["abcs_"]
    abc_t=[]
    for abcDict_of_time_step_t in labelReferenceFrameABC:
        abs_slice=[abcDict_of_time_step_t["value0"],abcDict_of_time_step_t["value1"],abcDict_of_time_step_t["value2"]] 
        abc_t.append(abs_slice)
        

                            
    n,rc,si=metaINFo['n_rc_Si']["value0"],metaINFo['n_rc_Si']["value1"],metaINFo['n_rc_Si']["value2"]
    tx,ty=metaINFo['txy']["value0"],metaINFo['txy']["value1"]

    # print (f"Xdim:{Xdim},Ydim:{Ydim},dominMinBoundary:{dominMinBoundary},dominMaxBoundary:{dominMaxBoundary},time_steps:{time_steps},tmin:{tmin},tmax:{tmax}")
    #check raw_Binary.size
    if raw_Binary.size!=time_steps*Ydim*Xdim* 2:
        raise ValueError(f"Binary data size is not correct, expected {time_steps*Ydim*Xdim* 2}, got {raw_Binary.size}")
    fieldData = raw_Binary.reshape( time_steps,Ydim,Xdim, 2)
    loadField = UnsteadyVectorField2DNp(Xdim, Ydim, time_steps, dominMinBoundary, dominMaxBoundary)
    loadField.field =fieldData
    vortexLableData=[tx,ty,n,rc] 
    return loadField, abc_t,vortexLableData

@typechecked
def LicRenderingUnsteady(field:UnsteadyVectorField2DNp,licImageSize:int,timeStepSKip:int=2,saveFolder:str="./"):
    #typecheck field type and field is not None    
    Xdim,Ydim,time_steps=field.Xdim,field.Ydim,field.time_steps
    texture = np.random.rand(Xdim, Ydim)    
    for i in range(0, time_steps, timeStepSKip):
        print(f"Processing time step {i}")
        steadyVectorField2D = field.getSlice(i)
        lic_result=LICAlgorithm(  texture  ,steadyVectorField2D ,licImageSize,licImageSize,0.005,256)
        lic_normalized =255* (lic_result - np.min(lic_result)) / (np.max(lic_result) - np.min(lic_result))
        #  Convert to an image and save
        plt.imshow(lic_normalized, cmap='gray')
        plt.axis('off')  # Optional: Remove axis for a cleaner image
        save_name=f"vector_field_lic_{i}.png"
        savePath=os.path.join(saveFolder,save_name)
        plt.savefig(savePath, bbox_inches='tight', pad_inches=0)




def test_load_results(): 
    directory_path = 'C:\\Users\\zhanx0o\\Documents\\sources\\PyflowVis\\CppProjects\\data\\unsteady\\64_64\\velocity_rc_1n_2\\rc_1_n_2_sample_0Si_0observer_0type_0.bin'
    loadField, labelReferenceFrameABC=loadOneFlowEntry(directory_path)    
    LicRenderingUnsteady(loadField,64,4)

    




import torch,time,tqdm
# create torch dataset using the load result function:
class UnsteadyVastisDataset(torch.utils.data.Dataset):
    def __init__(self, directory_path):
        self.directory_path = directory_path
        self.data=[]
        self.labelReferenceFrame=[]
        self.labelVortex=[]
        self.preLoading()
    def preLoading(self,slicePerData:int=7, TrainValidationSplitRatio:float=0.8):        
        print("Preloading data......")
        start = time.time() 
        rc_n_subfoders=os.listdir(self.directory_path)
        for folder_name in tqdm.tqdm(rc_n_subfoders) :
            sub_folder=os.path.join(self.directory_path,folder_name)
            #find all *.bin data in this subfoder
            binFiles = [f for f in os.listdir(sub_folder) if f.endswith('.bin')]
            for binFile in binFiles:
                binPath=os.path.join(sub_folder,binFile)
                #split according to slicePerData 
                loadField, labelReferenceFrameABC,vortexlabel=loadOneFlowEntry(binPath)
                TimeWindowCount=loadField.time_steps//slicePerData
                for i in range(0,TimeWindowCount):
                    sliceStart=i*slicePerData
                    sliceEnd=(i+1)*slicePerData
                    dataSlice=loadField.getSlice(sliceStart,sliceEnd)
                    self.data.append(dataSlice)
                    self.labelReferenceFrame.append(labelReferenceFrameABC[sliceStart:sliceEnd])
                    self.labelVortex.append(vortexlabel)#vortexlabel=[tx,ty,n,rc] 
                    
        #logging dataset information    and time consuming of preloading data
        elapsed = time.time()
        elapsed = elapsed - start
        print("Preloading data......done")
        print(f"Total number of data:{len(self.data)}, took {elapsed} seconds")
        
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labelReferenceFrame[idx]




if __name__ == '__main__':
   test_load_results()