import os
import json
import numpy as np
from scipy.interpolate import CubicSpline

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
    # if not os.path.exists(filepath):
    #     raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data




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



def pad_or_truncate_pathlines(pathlineClusters, L):
    K = len(pathlineClusters)  
    C = len(pathlineClusters[0][0]) if pathlineClusters and pathlineClusters[0] else 0  

    result = np.zeros((L, K , C), dtype=np.float32)
    for i, pathline in enumerate(pathlineClusters):
        pathline_length = min(len(pathline), L)  
        for step in range(pathline_length):
            result[step, i,:] = np.array(pathline[step])
    result[0,:,-1]=0.0
    return result


def getClassificationOfPatch(si):
    #classfication of this patch
    vortexsegmentationLabel = np.zeros((1),dtype=np.float32)
    if si ==1.0 or si == 2.0 :
        vortexsegmentationLabel[0] = 1.0        
    else:
        # vortexsegmentationLabel[0] = 0.0 
        vortexsegmentationLabel[0] =0.0
    return vortexsegmentationLabel


def getSegmentationofPathlines(pathlineClusters,si):
    Klines,PointsPerline,featurePerPoint=pathlineClusters.shape
    vortexsegmentationLabel = np.zeros((Klines),dtype=np.float32)
    if si ==1.0 or si == 2.0 :
        for lineId in range(Klines):
            #point features: px, py, time, ivd, distance,v_0, v_1([nablaV...])
            vortexsegmentationLabel[lineId] =pathlineClusters[lineId,0,4]
            #remove the label 
            pathlineClusters[lineId,0,4]=0

    return vortexsegmentationLabel

    
def loadUnsteadyFlowPathlineSegmentation(metaPath,Xdim,Ydim,time_steps,PathlineLength,PathlineCount,PathlineFeature,mode="train"):
    #get meta information
    # metaINFo=read_json_file(metaPath)
    # n,rc,si=metaINFo['rc_n_si']["value0"],metaINFo['rc_n_si']["value1"],metaINFo['rc_n_si']["value2"]
    if "saddle" in metaPath or "zero" in metaPath :
        si=0.0
    else:
        si=1.0
    
    fieldData=None
    if mode=="test":#during test, we need data for visualization, for train and validation, only pathline are need
        rawBinaryPath= metaPath.replace('meta.json', '.bin')
        raw_Binary = read_binary_file(rawBinaryPath)
        fieldData = raw_Binary.reshape(time_steps, Ydim,Xdim, 2)
    else:
        fieldData=np.zeros([1,1,1,1],dtype=np.float32)
    # pathlineClusters= np.array(metaINFo["ClusterPathlines"],dtype=np.float32)
    
    pathlineBinarypath= metaPath.replace('meta.json', '_pathline.bin')
    pathlineClusters=read_binary_file(pathlineBinarypath)
    pathlineClusters=pathlineClusters.reshape(PathlineCount,PathlineLength,PathlineFeature)
    
    vortexsegmentationLabel=getSegmentationofPathlines(pathlineClusters,si)
    
    # Permute pathline clusters first and second axis

    pathlineClusters = np.transpose(pathlineClusters, (1, 0, 2))
    #In spatical, the range is [-2,2] generate 8 cross, seed poistion distance is sampleGrid_dx / 3.0, that is 4*0.8/((8-1)*3)=0.15237
    #normalize time then time is range (0,1), we will need to compute distance (for knn query),
    #then the range of time will influence distance in space and distance in time, and dt is 1/15=0.06666666 vs. not normalize : pi/(4*15)=0.0523598
    pathlineClusters[:,:,2]=pathlineClusters[:,:,2]/(0.25*np.pi)     
    
    
    return (fieldData,pathlineClusters),vortexsegmentationLabel



















# Generate grid of positions
    # x_coords = np.linspace(domainMinBoundary[0], dominMaxBoundary[0], Xdim)
    # y_coords = np.linspace(domainMinBoundary[1], dominMaxBoundary[1], Ydim)
    # X, Y = np.meshgrid(x_coords, y_coords)

    # # Create the position vectors
    # pos = np.stack([X, Y], axis=-1)  # Shape: (Ydim, Xdim, 2)
    
    # # Transform positions
    # transformed_pos = np.dot(pos - txy, InvA.T)  # Shape: (Ydim, Xdim, 2)

    # # Compute distances from the vortex center
    # distances = np.linalg.norm(transformed_pos, axis=-1)  # Shape: (Ydim, Xdim)

    # # Determine labels based on distance
    # inside_vortex = (rc > distances)
    # vortexsegmentationLabel[:, :, 1] = inside_vortex.astype(np.float32)
    # vortexsegmentationLabel[:, :, 0] = (~inside_vortex).astype(np.float32)
    
# vortexsegmentationLabel2 = np.zeros((Ydim, Xdim,2),dtype=np.float32)       
# gridInterval=[ float(dominMaxBoundary[0]-domainMinBoundary[0])/float(Xdim-1),
#               float(dominMaxBoundary[0]-domainMinBoundary[0])/float(Ydim-1)    ]
# if si==0.0:
#     vortexsegmentationLabel[:, :, 0] = 1.0
# else:
#     for y in range(Ydim):
#         for x in range(Xdim):
#             # Extract position vector from fieldData
#             pos=np.array([domainMinBoundary[0]+x*gridInterval[0],domainMinBoundary[1]+y*gridInterval[1]])
#             transformed_pos=np.dot(InvA, (pos-txy))
#             dx=rc- np.linalg.norm(transformed_pos)  
#             #dx>0 ->rc >distance->inside vortex->label to one()[0,1]
#             vortexsegmentationLabel[y, x] = np.array([0.0,1.0],dtype=np.float32) if dx>0.0 else np.array([1.0,0.0],dtype=np.float32) 