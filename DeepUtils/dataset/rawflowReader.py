import json
import numpy as np
import os

def read_json_file(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

def read_rootMetaGridresolution(meta_file):
    metaINFo = read_json_file(meta_file)
    if 'tmin' in metaINFo:          
        tmin=metaINFo['tmin']
        dominMinBoundary=[metaINFo['domainMinBoundary']["value0"],metaINFo['domainMinBoundary']["value1"],tmin] 
    else:
        dominMinBoundary=[metaINFo['domainMinBoundary']["value0"],metaINFo['domainMinBoundary']["value1"]]
    if 'tmax' in metaINFo:
        tmax=metaINFo['tmax']    
        dominMaxBoundary=[metaINFo['domainMaxBoundary']["value0"],metaINFo['domainMaxBoundary']["value1"],tmax]
    else:
        dominMaxBoundary=[metaINFo['domainMaxBoundary']["value0"],metaINFo['domainMaxBoundary']["value1"]]
    metaINFo['domainMinBoundary']=dominMinBoundary
    metaINFo['domainMaxBoundary']=dominMaxBoundary
    return metaINFo


def read_binary_file(filepath, dtype=np.float32) -> np.ndarray:
    with open(filepath, 'rb') as file:
        data = np.fromfile(file, dtype=dtype)
        if dtype == np.float32:
            data=data[2:]
        elif dtype == np.float64:
            data=data[1:]        
    return data



def loadOneFlowEntryRawDataSteady(binPath,Xdim,Ydim):
    raw_Binary = read_binary_file(binPath)
    #get meta information
    meta_file = binPath.replace('.bin', 'meta.json')
    metaINFo=read_json_file(meta_file)
    n,rc,si=metaINFo['n_rc_Si']["value0"],metaINFo['n_rc_Si']["value1"],metaINFo['n_rc_Si']["value2"]
    fieldData = raw_Binary.reshape( Ydim,Xdim, 2)
    #si=0 means this is saddle not vortex, logits 0 means non-votex class, logits 1 means vortex class
    if si==0.0:
        vortexLabelOneHot= np.array([1.0,0.0],dtype=np.float32)
    else:
        vortexLabelOneHot=np.array([0.0,1.0],dtype=np.float32)
    return fieldData,vortexLabelOneHot


def loadOneFlowEntryRawData(binPath,Xdim,Ydim,time_steps):
    raw_Binary = read_binary_file(binPath)
    #get meta information
    meta_file = binPath.replace('.bin', 'meta.json')
    metaINFo=read_json_file(meta_file)

    abc_tInfo=[value for value in metaINFo['observer_abc'].values()]
    abcdot_t=[value for value in metaINFo['observer_abc_dot'].values()] 
    referenceLabel=abc_tInfo+abcdot_t
                            
    # n,rc,si=metaINFo['n_rc_Si']["value0"],metaINFo['n_rc_Si']["value1"],metaINFo['n_rc_Si']["value2"]
    # tx,ty=metaINFo['txy']["value0"],metaINFo['txy']["value1"]
    # vortexLableData= np.array([tx,ty,n,rc],dtype=np.float32) 
    #check raw_Binary.size
    assert (raw_Binary.size==time_steps*Ydim*Xdim* 2)
    fieldData = raw_Binary.reshape( time_steps,Ydim,Xdim, 2)
    referenceLabel= np.array(referenceLabel,dtype=np.float32)
    return fieldData,referenceLabel



