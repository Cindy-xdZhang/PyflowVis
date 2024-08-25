import numpy as np
import os
from FLowUtils.VectorField2d import *
from DeepUtils.dataset.rawflowReader import loadOneFlowEntryRawData,loadOneFlowEntryRawDataSteady,read_rootMetaGridresolution
import torch,time,tqdm
from .build import DATASETS

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


@DATASETS.register_module()
class UnsteadyVastisDataset(torch.utils.data.Dataset):
    def __init__(self, directory_path,mode):
        self.directory_path=directory_path
        self.dataName=[]
        self.data=[]
        self.labelReferenceFrame=[]
        self.labelVortex=[]
        self.transform=[]
        self.dastasetMetaInfo={}
        self.mode=mode
        self.steady= directory_path.find("Steady")>0 or directory_path.find("steady")>0 
        self.preLoading(mode)

    def setTransform(self,transform):
        self.transform=transform
    def getSampleName(self,sampleIdx):        
        return self.dataName[sampleIdx]

    def readRootMetaInfo(self,rootFolder):
        if self.dastasetMetaInfo=={}:
            folder_meta_file = os.path.join(rootFolder, 'meta.json')
            #   {"Xdim":Xdim,"Ydim":Ydim,"time_steps":time_steps,"dominMinBoundary":dominMinBoundary,"dominMaxBoundary":dominMaxBoundary,"tmin":tmin,"tmax":tmax}
            self.dastasetMetaInfo=read_rootMetaGridresolution(folder_meta_file)
    
    def loadOneTaskFolder(self,sub_folder:str):
        Xdim,Ydim,time_steps=self.dastasetMetaInfo["Xdim"],self.dastasetMetaInfo["Ydim"],self.dastasetMetaInfo["unsteadyFieldTimeStep"]
        #find all *.bin data in this subfoder
        binFiles = [f for f in os.listdir(sub_folder) if f.endswith('.bin')]
        minV,maxV=   self.dastasetMetaInfo['minV'],self.dastasetMetaInfo['maxV']
        for binFile in binFiles:
            binPath=os.path.join(sub_folder,binFile)
            if  self.steady:
                loadField, labelReferenceFrame,vortexlabel=loadOneFlowEntryRawDataSteady(binPath, Xdim,Ydim)
                dataSlice=torch.tensor(loadField).transpose(0, -1)
            else:
                loadField, labelReferenceFrame,vortexlabel=loadOneFlowEntryRawData(binPath, Xdim,Ydim,time_steps) 
                # timesteps=loadField.shape[0]
                # dataSlice shape is [ depth(timsteps), W, H, chanel=2]
                # need  transpose to [ chanel=2, W, H, depth(timsteps)=7] to feed conv3D
                dataSlice=torch.tensor(loadField).transpose(0, -1)

            dataSlice = (dataSlice -minV) / ( maxV-minV)
            self.data.append(dataSlice)
            self.labelReferenceFrame.append(torch.tensor(labelReferenceFrame))
            self.labelVortex.append(torch.tensor(vortexlabel))

            if self.mode=="test":
                self.dataName.append(keep_last_n_levels(binPath,2))
            
        

    def preLoading(self,mode):                
        start = time.time()         
        print(f"Preloading [{mode}] data......")
        #train_directory_path should be  the direct parent folder of all "rc_xxxx_n_xxx"  folders
        train_directory_path=os.path.join(self.directory_path,mode)
        self.readRootMetaInfo(train_directory_path)
        rc_n_subfoders=os.listdir(train_directory_path)
        for folder_name in tqdm.tqdm(rc_n_subfoders) :    
                if folder_name    == "meta.json":
                    continue            
                sub_folder=os.path.join(train_directory_path,folder_name)
                self.loadOneTaskFolder(sub_folder)          
        #logging dataset information    and time consuming of preloading data
        elapsed = time.time()
        elapsed = elapsed - start
        print(f"Preloading  [{mode}] data......done")
        print(f"Total number of [{mode}] data:{len(self.data)}, took {elapsed} seconds")
        
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sample=self.data[idx]
        # if self.transform:
        #     sample=self.transform(self.data[idx])
        #  # Generate white noise
        noise = np.random.uniform(-0.02, 0.02, sample.shape).astype(np.float32)  # Generate noise within the range [-0.01, 0.01]
        # Add the white noise to the sample
        sample_with_noise = sample + noise
        return sample_with_noise , self.labelReferenceFrame[idx],self.labelVortex[idx]
  


def buildDataset(args,mode="train"):
    ds=UnsteadyVastisDataset(args['root'],mode)
    return ds




if __name__ == '__main__':
    pass