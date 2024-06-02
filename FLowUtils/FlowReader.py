import json
import numpy as np
import os
# from typeguard import typechecked





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

def loadOneFlowEntry(binPath):
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

    #create instance of UnsteadyVectorField2DNp is time consuming
    # loadField = UnsteadyVectorField2DNp(Xdim, Ydim, time_steps, dominMinBoundary, dominMaxBoundary)
    # loadField.field =fieldData
    vortexLableData= np.array([tx,ty,n,rc],dtype=np.float32) 
    return fieldData, abc_t,vortexLableData



# def test_load_results(): 
#     directory_path = 'C:\\Users\\zhanx0o\\Documents\\sources\\PyflowVis\\CppProjects\\data\\unsteady\\64_64\\velocity_rc_1n_2\\rc_1_n_2_sample_0Si_0observer_0type_0.bin'
#     loadField, labelReferenceFrameABC,votexInfo=loadOneFlowEntry(directory_path)    
#     LicRenderingUnsteady(loadField,64,4)

# if __name__ == '__main__':
#    test_load_results()