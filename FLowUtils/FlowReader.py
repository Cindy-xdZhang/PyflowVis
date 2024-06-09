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

def read_rootMetaGridresolution(meta_file):
    metaINFo = read_json_file(meta_file)
    Xdim=metaINFo['Xdim']
    Ydim=metaINFo['Ydim']
    time_steps=metaINFo['unsteadyFieldTimeStep']
    tmin=metaINFo['tmin']
    tmax=metaINFo['tmax']    
    dominMinBoundary=[metaINFo['domainMinBoundary']["value0"],metaINFo['domainMinBoundary']["value1"],tmin] 
    dominMaxBoundary=[metaINFo['domainMaxBoundary']["value0"],metaINFo['domainMaxBoundary']["value1"],tmax]
    return Xdim,Ydim,time_steps,dominMinBoundary,dominMaxBoundary

def read_binary_file(filepath, dtype=np.float32) -> np.ndarray:
    with open(filepath, 'rb') as file:
        data = np.fromfile(file, dtype=dtype)
        if dtype == np.float32:
            data=data[2:]
        elif dtype == np.float64:
            data=data[1:]        
    return data



def loadOneFlowEntryRawData(binPath,Xdim,Ydim,time_steps):
    raw_Binary = read_binary_file(binPath)

    #get meta information
    meta_file = binPath.replace('.bin', 'meta.json')
    metaINFo=read_json_file(meta_file)
    # Xdim=metaINFo['Xdim']
    # Ydim=metaINFo['Ydim']
    #observe and unsteady info 
    
    labelReferenceFrameABC=metaINFo['observerfieldDeform']["abcs_"]
    abc_t=[]
    for abcDict_of_time_step_t in labelReferenceFrameABC:
        abs_slice=[abcDict_of_time_step_t["value0"],abcDict_of_time_step_t["value1"],abcDict_of_time_step_t["value2"]] 
        abc_t.append(abs_slice)
        
                            
    n,rc,si=metaINFo['n_rc_Si']["value0"],metaINFo['n_rc_Si']["value1"],metaINFo['n_rc_Si']["value2"]
    tx,ty=metaINFo['txy']["value0"],metaINFo['txy']["value1"]

    #check raw_Binary.size
    if raw_Binary.size!=time_steps*Ydim*Xdim* 2:
        raise ValueError(f"Binary data size is not correct, expected {time_steps*Ydim*Xdim* 2}, got {raw_Binary.size}")
    fieldData = raw_Binary.reshape( time_steps,Ydim,Xdim, 2)
    vortexLableData= np.array([tx,ty,n,rc],dtype=np.float32) 
    return fieldData, abc_t,vortexLableData


def test_load_results(): 
    Xdim,Ydim,time_steps,dominMinBoundary,dominMaxBoundary=read_rootMetaGridresolution('C:\\Users\\zhanx0o\\Documents\\sources\\PyflowVis\\CppProjects\\data\\unsteady\\64_64\\velocity_rc_1n_2\\meta.json')
    directory_path = 'C:\\Users\\zhanx0o\\Documents\\sources\\PyflowVis\\CppProjects\\data\\unsteady\\64_64\\velocity_rc_1n_2\\rc_1_n_2_sample_0Si_1observer_0type_3.bin'
    loadField, labelReferenceFrameABC,votexInfo=loadOneFlowEntryRawData(directory_path,Xdim,Ydim,time_steps)    
    

if __name__ == '__main__':
   test_load_results()