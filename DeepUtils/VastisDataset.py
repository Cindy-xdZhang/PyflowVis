import numpy as np
import os
from FLowUtils.VectorField2d import *
from FLowUtils.FlowReader import loadOneFlowEntryRawData,read_rootMetaGridresolution
import torch,time,tqdm
import torchvision.transforms as transforms
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    
def keep_last_n_levels(path,n):
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

# create torch dataset using the load result function:
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
            loadField, labelReferenceFrame,vortexlabel=loadOneFlowEntryRawData(binPath, Xdim,Ydim,time_steps)

            # timesteps=loadField.shape[0]
            # dataSlice shape is [ depth(timsteps), W, H, chanel=2]
            # need  transpose to [ chanel=2, W, H, depth(timsteps)=7] to feed conv3D
            dataSlice=torch.tensor(loadField).transpose(0, 3)
            # if ForceNormalization:
            dataSlice = (dataSlice -minV) / ( maxV-minV)
            self.data.append(dataSlice)
            self.labelReferenceFrame.append(torch.tensor(labelReferenceFrame))
            self.labelVortex.append(torch.tensor(vortexlabel))#vortexlabel=[tx,ty,n,rc,minv,maxv] 

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
        print("Preloading data......done")
        print(f"Total number of f{mode} data:{len(self.data)}, took {elapsed} seconds")
        
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