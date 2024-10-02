import numpy as np
import os
from FLowUtils.VectorField2d import *
import torch,time,tqdm
from .build import DATASETS
from .data_utils import *

def getDatasetRootaMeta(root_directory):
    try_path0=os.path.join(root_directory, 'meta.json')
    try_path1=os.path.join(root_directory, 'train/meta.json')
    if os.path.exists(try_path0):
        return read_rootMetaGridresolution(try_path0)
    elif os.path.exists(try_path1):
        return read_rootMetaGridresolution(try_path1)
    else:
        raise ValueError("no root meta file found.")


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

class VastisDataset(torch.utils.data.Dataset):
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
        self.dastasetMetaInfo=getDatasetRootaMeta(self.directory_path)
        #train_directory_path should be  the direct parent folder of all "rc_xxxx_n_xxx"  folders
        split_str =mode if mode!="val" else "validation"
        target_directory_path=os.path.join(self.directory_path,split_str) 
        # If the target directory doesn't exist, try alternatives
        if not os.path.exists(target_directory_path):
            if mode == "val":
                # For validation, try using the test directory
                target_directory_path = os.path.join(self.directory_path, "test")
            elif mode == "test":
                # For test, try using the validation directory
                target_directory_path = os.path.join(self.directory_path, "validation")
                
        if not os.path.exists(target_directory_path):
            raise ValueError(f"Could not find a valid directory for mode: {mode}")
        
        print(f"using folder {target_directory_path} for mode:{mode}")
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
class SteadyVastisClassification(VastisDataset):
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
class SteadyVastisSegmentation(VastisDataset):
    def __init__(self, data_dir,split, transform,**kwargs):
        super().__init__( data_dir,split, transform,**kwargs)

    def loadOneTaskFolder(self,sub_folder:str):
        Xdim,Ydim=self.dastasetMetaInfo["Xdim"],self.dastasetMetaInfo["Ydim"]
        dm_min,dm_max=self.dastasetMetaInfo["domainMinBoundary"],self.dastasetMetaInfo["domainMaxBoundary"]
        #find all *.bin data in this subfoder
        metaFiles = [f for f in os.listdir(sub_folder) if f.endswith('.json')]
        for File in metaFiles:
            metaPath=os.path.join(sub_folder,File)
            loadField,label=loadOneFlowEntrySteadySegmentation(metaPath, Xdim,Ydim,domainMinBoundary=dm_min,dominMaxBoundary=dm_max)
            # dataSlice=torch.tensor(loadField).permute(2,0, 1)
            #I think this is a bug, when there is no time axis, transpose will switch yx->xy,should use .permute(2,0, 1) instead.
            dataSlice=torch.tensor(loadField).transpose(0, -1)
            self.data.append(dataSlice)
            self.label.append(torch.tensor(label))
            if self.split=="test":
                self.dataName.append(keep_path_last_n_names(metaPath,2))                