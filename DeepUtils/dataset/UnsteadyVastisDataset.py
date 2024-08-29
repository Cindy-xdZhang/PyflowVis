import numpy as np
import os
from FLowUtils.VectorField2d import *
from  .data_utils import loadOneFlowEntryRawData,read_rootMetaGridresolution
import torch,time,tqdm
from .build import DATASETS
from .data_utils import *


@DATASETS.register_module()
class UnsteadyVastisDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir,split, transform,**kwargs):
        self.directory_path=data_dir
        self.dataName=[]
        self.data=[]
        self.label=[]
        self.transform=transform
        self.dastasetMetaInfo={}
        self.mode=split
        self.preLoading(split)


    def getSampleName(self,sampleIdx):        
        return self.dataName[sampleIdx]

    def readRootMetaInfo(self,rootFolder):
        if self.dastasetMetaInfo=={}:
            folder_meta_file = os.path.join(rootFolder, 'meta.json')
            #   {"Xdim":Xdim,"Ydim":Ydim,"time_steps":time_steps,"dominMinBoundary":dominMinBoundary,"dominMaxBoundary":dominMaxBoundary,"tmin":tmin,"tmax":tmax}
            self.dastasetMetaInfo=read_rootMetaGridresolution(folder_meta_file)          
        
    def preLoading(self,mode):                
        print(f"Preloading [{mode}] data......")
        start = time.time()         
        #self.directory_path should have meta.json
        self.dastasetMetaInfo=read_rootMetaGridresolution(os.path.join(self.directory_path, 'meta.json'))
        #train_directory_path should be  the direct parent folder of all "rc_xxxx_n_xxx"  folders
        split_str =mode if mode!="val" else "validation"
        target_directory_path=os.path.join(self.directory_path,split_str) 
        rc_n_subfoders=[os.path.join(target_directory_path,folder) for folder in os.listdir(target_directory_path) if os.path.isdir(os.path.join(target_directory_path, folder))]
        for folder_name in tqdm.tqdm(rc_n_subfoders) :           
                self.loadOneTaskFolder(folder_name)          
        #logging dataset information    and time consuming of preloading data
        elapsed = time.time()
        elapsed = elapsed - start
        print(f"Preloading  [{mode}] data......done")
        print(f"Total number of [{mode}] data:{len(self.data)}, took {elapsed} seconds")

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
            if self.mode=="test":
                self.dataName.append(keep_path_last_n_names(binPath,2))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data=self.data[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data , self.label[idx]
  


