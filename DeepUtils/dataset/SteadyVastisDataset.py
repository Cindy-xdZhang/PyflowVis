import numpy as np
import os
from FLowUtils.VectorField2d import *
import torch,time,tqdm
from .build import DATASETS
from .data_utils import *

class SteadyVastis(torch.utils.data.Dataset):
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
        pass
            
            
        
@DATASETS.register_module()
class SteadyVastisClassification(SteadyVastis):
    def __init__(self, data_dir,split, transform,**kwargs):
        super().__init__( data_dir,split, transform,**kwargs)

    def loadOneTaskFolder(self,sub_folder:str):
        Xdim,Ydim=self.dastasetMetaInfo["Xdim"],self.dastasetMetaInfo["Ydim"]
        #find all *.bin data in this subfoder
        binFiles = [f for f in os.listdir(sub_folder) if f.endswith('.bin')]
        for binFile in binFiles:
            binPath=os.path.join(sub_folder,binFile)
            loadField,label=loadOneFlowEntryRawDataSteady(binPath, Xdim,Ydim)
            dataSlice=torch.tensor(loadField).transpose(0, -1)
            self.data.append(dataSlice)
            self.label.append(torch.tensor(label))
            if self.split=="test":
                self.dataName.append(keep_path_last_n_names(binPath,2))


        
@DATASETS.register_module()
class SteadyVastisSegmentation(SteadyVastis):
    def __init__(self, data_dir,split, transform,**kwargs):
        super().__init__( data_dir,split, transform,**kwargs)

    def loadOneTaskFolder(self,sub_folder:str):
        Xdim,Ydim=self.dastasetMetaInfo["Xdim"],self.dastasetMetaInfo["Ydim"]
        dm_min,dm_max=self.dastasetMetaInfo["domainMinBoundary"],self.dastasetMetaInfo["domainMaxBoundary"]
        #find all *.bin data in this subfoder
        binFiles = [f for f in os.listdir(sub_folder) if f.endswith('.bin')]
        for binFile in binFiles:
            binPath=os.path.join(sub_folder,binFile)
            loadField,label=loadOneFlowEntrySteadySegmentation(binPath, Xdim,Ydim,domainMinBoundary=dm_min,dominMaxBoundary=dm_max)
            dataSlice=torch.tensor(loadField).transpose(0, -1)
            self.data.append(dataSlice)
            self.label.append(torch.tensor(label))
            if self.split=="test":
                self.dataName.append(keep_path_last_n_names(binPath,2))                