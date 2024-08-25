import numpy as np
import os
from FLowUtils.VectorField2d import *
import torch,time,tqdm
from .build import DATASETS
from .data_utils import *


@DATASETS.register_module()
class SteadyVastisDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir,split, transform,**kwargs):
        self.directory_path=data_dir
        self.dataName=[]
        self.data=[]
        self.labelVortex=[]
        self.transform=[]
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
        
        rc_n_subfoders=os.listdir(target_directory_path)
        for folder_name in tqdm.tqdm(rc_n_subfoders) :           
                sub_folder=os.path.join(target_directory_path,folder_name)
                self.loadOneTaskFolder(sub_folder)          
        #logging dataset information    and time consuming of preloading data
        elapsed = time.time()
        elapsed = elapsed - start
        print(f"Preloading  [{mode}] data......done")
        print(f"Total number of [{mode}] data:{len(self.data)}, took {elapsed} seconds")

    def loadOneTaskFolder(self,sub_folder:str):
        Xdim,Ydim=self.dastasetMetaInfo["Xdim"],self.dastasetMetaInfo["Ydim"]
        #find all *.bin data in this subfoder
        binFiles = [f for f in os.listdir(sub_folder) if f.endswith('.bin')]
        for binFile in binFiles:
            binPath=os.path.join(sub_folder,binFile)
            loadField,label=loadOneFlowEntryRawDataSteady(binPath, Xdim,Ydim)
            dataSlice=torch.tensor(loadField).transpose(0, -1)
            self.data.append(dataSlice)
            self.labelVortex.append(torch.tensor(label))
            if self.split=="test":
                self.dataName.append(keep_path_last_n_names(binPath,2))
            
        
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        data=self.data[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data , self.labelVortex[idx]
