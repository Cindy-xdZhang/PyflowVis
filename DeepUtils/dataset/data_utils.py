import os
import json
import numpy as np

def keep_path_last_n_names(path,n):
    """
    Keep only the last two levels of the given path.
    
    :param path: Original path
    :return: Path with only the last two levels
    """
    # Normalize the path to remove any redundant separators or up-level references
    normalized_path = os.path.normpath(path)
    
    # Split the path into parts
    path_parts = normalized_path.split(os.sep)
    
    # Keep only the last two levels
    last_two_levels = os.sep.join(path_parts[-n:])
    last_two_levels=last_two_levels.replace("/","_")
    last_two_levels=last_two_levels.replace("\\","_")
    return last_two_levels


def read_binary_file(filepath, dtype=np.float32) -> np.ndarray:
    with open(filepath, 'rb') as file:
        data = np.fromfile(file, dtype=dtype)
        if dtype == np.float32:
            data=data[2:]
        elif dtype == np.float64:
            data=data[1:]        
    return data


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

def getDatasetRootaMeta(root_directory):
    return read_rootMetaGridresolution(os.path.join(root_directory, 'meta.json'))



def loadOneFlowEntrySteadySegmentation(binPath,Xdim,Ydim,domainMinBoundary,dominMaxBoundary):
    raw_Binary = read_binary_file(binPath)
    #get meta information
    meta_file = binPath.replace('.bin', 'meta.json')
    metaINFo=read_json_file(meta_file)
    n,rc,si=metaINFo['n_rc_Si']["value0"],metaINFo['n_rc_Si']["value1"],metaINFo['n_rc_Si']["value2"]
    txy=np.array([metaINFo['txy']["value0"],metaINFo['txy']["value1"]]  )  
    theta,sx,sy=metaINFo['deform_theta_sx_sy']["value0"],metaINFo['deform_theta_sx_sy']["value1"],metaINFo['deform_theta_sx_sy']["value2"]
    deformMatA=np.array([
                    [sx*np.cos(theta), -sy*np.sin(theta)],
                    [sx*np.sin(theta), sy*np.cos(theta)]])
    InvA= np.linalg.inv(deformMatA)

    
    fieldData = raw_Binary.reshape( Ydim,Xdim, 2)
    gridInterval=[ float(dominMaxBoundary[0]-domainMinBoundary[0])/float(Xdim-1),
                  float(dominMaxBoundary[0]-domainMinBoundary[0])/float(Ydim-1)    ]
    vortexsegmentationLabel = np.zeros((Ydim, Xdim,2),dtype=np.float32)
    if si==0.0:
        vortexsegmentationLabel[:, :] = np.array([1.0, 0.0], dtype=np.float32)
    else:
        for y in range(Ydim):
            for x in range(Xdim):
                # Extract position vector from fieldData
                pos=np.array([domainMinBoundary[0]+x*gridInterval[0],domainMinBoundary[1]+y*gridInterval[1]])
                transformed_pos=np.dot(InvA, (pos-txy))
                dx=rc- np.linalg.norm(transformed_pos)  
                #dx>0 ->rc >distance->inside vortex->label to one()[0,1]
                vortexsegmentationLabel[y, x] = np.array([0.0,1.0],dtype=np.float32) if dx>0 else np.array([1.0,0.0],dtype=np.float32) 
    return fieldData,vortexsegmentationLabel

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



