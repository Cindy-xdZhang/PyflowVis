import numpy as np
import os,logging
from FLowUtils.VectorField2d import *
from  .data_utils import loadOneFlowEntryRawData
import torch,time,tqdm
from .build import DATASETS
from .data_utils import *
from .SteadyVastisDataset import VastisDataset
from .transforms.basic_transform import PathlineJittorCubic

@DATASETS.register_module()
class UnsteadyVastisDataset(VastisDataset):
    def __init__(self, data_dir,split, transform,**kwargs):
        super().__init__( data_dir,split, transform,**kwargs)

    def loadOneTaskFolder(self,sub_folder:str):
        Xdim,Ydim,time_steps=self.dastasetMetaInfo["Xdim"],self.dastasetMetaInfo["Ydim"],self.dastasetMetaInfo["unsteadyFieldTimeStep"]
        #find all *.bin data in this subfoder
        binFiles = [f for f in os.listdir(sub_folder) if f.endswith('.bin')]
        minV,maxV=   self.dastasetMetaInfo['minV'],self.dastasetMetaInfo['maxV']
        for binFile in binFiles:
            binPath=os.path.join(sub_folder,binFile)
            loadField, labelReferenceFrame=loadOneFlowEntryRawData(binPath, Xdim,Ydim,time_steps) 
            # timesteps=loadField.shape[0]
            # dataSlice shape is [ depth(timsteps), W, H, chanel=2]
            # need  transpose to [ chanel=2, W, H, depth(timsteps)] to feed conv3D
            dataSlice=torch.tensor(loadField).transpose(0, -1)
            self.data.append(dataSlice)
            self.label.append(torch.tensor(labelReferenceFrame))
            if self.split=="test":
                self.dataName.append(keep_path_last_n_names(binPath,2))    
  


@DATASETS.register_module()
class UnsteadyVastisPathlineSeg(VastisDataset):
    def __init__(self, data_dir,split, transform, mask_out_feature=None,downSampleRatio=1.0,**kwargs):
        self.mask_out_feature=mask_out_feature if mask_out_feature  else None
        self.downSampleRatio=downSampleRatio
        super().__init__( data_dir,split, transform,**kwargs)
        # self.lineTransform=PathlineJittorCubic()
        self.jitter_epsilon=0.0001
        
    
    def __getitem__(self, idx):
        data,pathlines=self.data[idx]
        if self.transform is not None:
            data = self.transform(data)   
        # pathlines = self.lineTransform(pathlines)

        #shape of pathline is Lsequence,Klines,Cfeatures
        #now jittor the pathlines: the first [0:2]channel of last dim is position
        #every pathline points compute its vector to its previous point, and then move along that direction vector with epsilo
        jittered_pathlines = self.jitter_pathlines(pathlines)
        return (data,jittered_pathlines) , self.label[idx]
    
    def jitter_pathlines(self, pathlines):
        # pathlines shape: [Lsequence, Klines, Cfeatures]
        jittered = pathlines.clone()
        # Compute vectors between consecutive points
        vectors = jittered[:, :, :2] - torch.roll(jittered[:, :, :2], shifts=1, dims=0)
        # Set the first row of vectors to zero to avoid wrapping
        vectors[0] = 0
        
        # Normalize vectors
        magnitudes = torch.norm(vectors, dim=-1, keepdim=True)
        normalized_vectors = vectors / (magnitudes + 1e-8)  # Add small epsilon to avoid division by zero
        
        # Generate random jitter
        random_jitter = torch.randn_like(normalized_vectors) * self.jitter_epsilon
        # Apply jitter along the direction vectors
        jittered[:, :, :2] += normalized_vectors * random_jitter
        return jittered


    def loadOneTaskFolder(self,sub_folder:str):
        print(f"loading {sub_folder}...")
        Xdim,Ydim,time_steps=getattr(self.dastasetMetaInfo,"Xdim",32),getattr(self.dastasetMetaInfo,"Ydim",32) ,getattr(self.dastasetMetaInfo,"unsteadyFieldTimeStep",5) 
        #find all *.bin data in this subfoder
        metaFiles = [f for f in os.listdir(sub_folder) if f.endswith('.json') and f!="meta.json"]
        minV,maxV=   self.dastasetMetaInfo['minV'],self.dastasetMetaInfo['maxV']
        PathlineCountK=16
        PathlineLength=16
        PathlineFeature=7
        if "outputPathlineLength" not in self.dastasetMetaInfo:
            logging.warning("outputPathlineLength not in self.dastasetMetaInfo,assume 16" )
        else:
            PathlineLength=self.dastasetMetaInfo["outputPathlineLength"]
        if "outputPathlinesCountK" not in self.dastasetMetaInfo:
            logging.warning("outputPathlinesCountK not in self.dastasetMetaInfo,assume 16" )
        else:
            PathlineCountK= self.dastasetMetaInfo["outputPathlinesCountK"]
        if "PathlineFeature" not in self.dastasetMetaInfo:
            logging.warning("PathlineFeature not in self.dastasetMetaInfo,assume 10" )
        else:
            PathlineFeature= self.dastasetMetaInfo["PathlineFeature"]

        PathlineCount=int(PathlineCountK/2)*int(PathlineCountK/2)*4
        oneFolderData=[None]*len(metaFiles)
        oneFolderLabel=[None]*len(metaFiles)
        for idx,metaFile in enumerate(metaFiles) :
            metaPath=os.path.join(sub_folder,metaFile)
            data, label=loadUnsteadyFlowPathlineSegmentation(metaPath, PathlineLength=PathlineLength,PathlineCount=PathlineCount,   PathlineFeature=PathlineFeature, downSampleRatio=self.downSampleRatio,
                                                             mask_out_feature= self.mask_out_feature,mode=self.split) 
            fieldData,pathlineData=data
            fieldData=torch.tensor(fieldData).transpose(0, -1)
            pathlineData=torch.tensor(pathlineData)
            oneFolderData[idx]=(fieldData,pathlineData)
            oneFolderLabel[idx]=torch.tensor(label)
            
            if self.split=="test":
                self.dataName.append(keep_path_last_n_names(metaPath,2))    
                
        self.data.extend(oneFolderData)
        self.label.extend(oneFolderLabel)