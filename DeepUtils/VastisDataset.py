import numpy as np
import os
from FLowUtils.VectorField2d import *
from FLowUtils.FlowReader import read_rootMetaGridresolution,loadOneFlowEntryRawData
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
    return last_two_levels

# create torch dataset using the load result function:
class UnsteadyVastisDataset(torch.utils.data.Dataset):
    def __init__(self, directory_path,mode,transform=None):
        fx_directory_path=self.FixDataFolder(directory_path)
        self.directory_path=os.path.join(fx_directory_path,"X64_Y64_T16_no_mixture")
        self.dataName=[]
        self.data=[]
        self.labelReferenceFrame=[]
        self.labelVortex=[]
        self.transform=transform
        self.dastasetMetaInfo={}
        self.preLoading(mode)
    def getBinaryName(self,sampleIdx):        
        return self.dataName[sampleIdx]
    def FixDataFolder(self,directory_path):
        """try directory_path, directory_path's parent, directory_path's grad parent to find the folder with name "unsteady"
        """
        if os.path.isdir(directory_path) and "X64_Y64_T16_no_mixture" in os.listdir(directory_path):
            return directory_path
        parent=os.path.dirname(directory_path)
        if os.path.isdir(parent) and "X64_Y64_T16_no_mixture" in os.listdir(parent):
            return parent
        gradParent=os.path.dirname(parent)
        if os.path.isdir(gradParent) and "X64_Y64_T16_no_mixture" in os.listdir(gradParent):
            return gradParent
        raise ValueError(f"Can't find the folder with name 'unsteady' in {directory_path}, {parent}, {gradParent}")
    
    def loadOneTaskFolder(self,sub_folder:str):
        if self.dastasetMetaInfo=={}:
            folder_meta_file = os.path.join(sub_folder, 'meta.json')
            Xdim,Ydim,time_steps,dominMinBoundary,dominMaxBoundary,tmin,tmax=read_rootMetaGridresolution(folder_meta_file)
            self.dastasetMetaInfo={"Xdim":Xdim,"Ydim":Ydim,"time_steps":time_steps,"dominMinBoundary":dominMinBoundary,"dominMaxBoundary":dominMaxBoundary,"tmin":tmin,"tmax":tmax}

        Xdim,Ydim,time_steps=self.dastasetMetaInfo["Xdim"],self.dastasetMetaInfo["Ydim"],self.dastasetMetaInfo["time_steps"]
        #find all *.bin data in this subfoder
        binFiles = [f for f in os.listdir(sub_folder) if f.endswith('.bin')]
        for binFile in binFiles:
            binPath=os.path.join(sub_folder,binFile)
            loadField, labelReferenceFrame,vortexlabel=loadOneFlowEntryRawData(binPath, Xdim,Ydim,time_steps)
            timesteps=loadField.shape[0]
            # dataSlice shape is [(batch_size,) depth(timsteps)=7, W=64, H=64, chanel=2]
            # need  transpose to [(batch_size,)  chanel=2, W=64, H=64, depth(timsteps)=7] to feed conv3D
            dataSlice=torch.tensor(loadField)
            vectorFieldDataSlice = dataSlice.transpose(0, 3)
            self.data.append(vectorFieldDataSlice)
            self.dataName.append(keep_last_n_levels(binPath,3))
            Qt, tc=torch.tensor( labelReferenceFrame[0]),torch.tensor(labelReferenceFrame[1])
            #Qt shape is [ time_steps,4], ct shape is [ time_steps,2],concat to [time_steps,6]->reshape to [6*time_steps]
            labelQtctSlice = torch.concat((Qt, tc), dim=1).reshape(-1)
            self.labelReferenceFrame.append(labelQtctSlice)
            self.labelVortex.append(torch.tensor(vortexlabel))#vortexlabel=[tx,ty,n,rc] 
            
            #uncomment code if you need  split vectorfield according to slicePerData             
            # TimeWindowCount=timesteps//self.slicePerData
            # for i in range(0,TimeWindowCount):
            #     sliceStart=i*self.slicePerData
            #     sliceEnd=(i+1)*self.slicePerData
            #     dataSlice=torch.tensor(loadField[sliceStart:sliceEnd])
            #     # dataSlice shape is [(batch_size,) depth(timsteps)=7, W=64, H=64, chanel=2]
            #     # need  transpose to [(batch_size,)  chanel=2, W=64, H=64, depth(timsteps)=7] to feed conv3D
            #     vectorFieldDataSlice = dataSlice.transpose(0, 3)
            #     self.data.append(vectorFieldDataSlice)
            #     Qt, tc=torch.tensor( labelReferenceFrame[0][sliceStart:sliceEnd,:]),torch.tensor(labelReferenceFrame[1][sliceStart:sliceEnd,:])
            #     #Qt shape is [  time_steps,4], ct shape is [ time_steps,2],concat to [time_steps,6]->reshape to [6*time_steps]
            #     labelQtctSlice = torch.concat((Qt, tc), dim=1).reshape(-1)
            #     self.labelReferenceFrame.append(labelQtctSlice)
            #     self.labelVortex.append(vortexlabel)#vortexlabel=[tx,ty,n,rc] 

    def preLoading(self,mode):                
        start = time.time()         
        if mode=="train":
            print("Preloading training data......")
            #train_directory_path should be  the direct parent folder of all "rc_xxxx_n_xxx"  folders
            train_directory_path=os.path.join(self.directory_path,"train")
            rc_n_subfoders=os.listdir(train_directory_path)
            for folder_name in tqdm.tqdm(rc_n_subfoders) :                
                if "test" not in folder_name and "val" not in folder_name:                
                    sub_folder=os.path.join(train_directory_path,folder_name)
                    self.loadOneTaskFolder(sub_folder)          

        elif mode=="val" or mode=="validation": 
            print("Preloading validation  data......")
            val_directory_path=os.path.join(self.directory_path,"validation")
            rc_n_subfoders=os.listdir(val_directory_path)
            for folder_name in tqdm.tqdm(rc_n_subfoders) :
                val_splict_sub_folder=os.path.join(val_directory_path,folder_name)    
                self.loadOneTaskFolder(val_splict_sub_folder)            

        elif mode=="test":
            print("Preloading test data......")
            test_directory_path=os.path.join(self.directory_path,"test")
            rc_n_subfoders=os.listdir(test_directory_path)
            for folder_name in tqdm.tqdm(rc_n_subfoders) :
                test_splict_sub_folder=os.path.join(test_directory_path,folder_name)    
                self.loadOneTaskFolder(test_splict_sub_folder)                  


        #logging dataset information    and time consuming of preloading data
        elapsed = time.time()
        elapsed = elapsed - start
        print("Preloading data......done")
        print(f"Total number of data:{len(self.data)}, took {elapsed} seconds")
        
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sample=self.data[idx]
        if self.transform:
            sample=self.transform(self.data[idx])            
        return sample, self.labelReferenceFrame[idx],self.labelVortex[idx]

def buildDataset(args,mode="train"):
    ds=UnsteadyVastisDataset(args['root'],mode)
    
    return ds




if __name__ == '__main__':
    pass