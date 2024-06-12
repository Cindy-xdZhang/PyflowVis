import numpy as np
import os
from .VectorField2d import *
from .FlowReader import read_rootMetaGridresolution,loadOneFlowEntryRawData
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
    

# create torch dataset using the load result function:
class UnsteadyVastisDataset(torch.utils.data.Dataset):
    def __init__(self, directory_path,slicePerdata,mode,transform=None):
        directory_path=self.FixDataFolder(directory_path)
        #directory_path should be  the direct parent folder of all "rc_xxxx_n_xxx"  folders
        self.directory_path=os.path.join( os.path.join(directory_path,"unsteady"),"64_64_nomix")
        self.data=[]
        self.labelReferenceFrame=[]
        self.labelVortex=[]
        self.slicePerData=slicePerdata
        self.transform=transform
        self.preLoading(mode)

    def FixDataFolder(self,directory_path):
        """try directory_path, directory_path's parent, directory_path's grad parent to find the folder with name "unsteady"
        """
        if os.path.isdir(directory_path) and "unsteady" in os.listdir(directory_path):
            return directory_path
        parent=os.path.dirname(directory_path)
        if os.path.isdir(parent) and "unsteady" in os.listdir(parent):
            return parent
        gradParent=os.path.dirname(parent)
        if os.path.isdir(gradParent) and "unsteady" in os.listdir(gradParent):
            return gradParent
        raise ValueError(f"Can't find the folder with name 'unsteady' in {directory_path}, {parent}, {gradParent}")
    
    def loadOneTaskFolder(self,sub_folder:str):
        folder_meta_file = os.path.join(sub_folder, 'meta.json')
        Xdim,Ydim,time_steps,dominMinBoundary,dominMaxBoundary=read_rootMetaGridresolution(folder_meta_file)
        #find all *.bin data in this subfoder
        binFiles = [f for f in os.listdir(sub_folder) if f.endswith('.bin')]
        for binFile in binFiles:
            binPath=os.path.join(sub_folder,binFile)
            #split according to slicePerData 
            loadField, labelReferenceFrameABC,vortexlabel=loadOneFlowEntryRawData(binPath, Xdim,Ydim,time_steps)
            timesteps=loadField.shape[0]
            TimeWindowCount=timesteps//self.slicePerData
            for i in range(0,TimeWindowCount):
                sliceStart=i*self.slicePerData
                sliceEnd=(i+1)*self.slicePerData
                # dataSlice=loadField.getSlice(sliceStart,sliceEnd)
                dataSlice=loadField[sliceStart:sliceEnd]
                self.data.append(dataSlice)
                self.labelReferenceFrame.append( labelReferenceFrameABC[sliceStart:sliceEnd,:] )
                self.labelVortex.append(vortexlabel)#vortexlabel=[tx,ty,n,rc] 

    def preLoading(self,mode):        

        start = time.time()         
        if mode=="train":
            print("Preloading training data......")
            rc_n_subfoders=os.listdir(self.directory_path)
            for folder_name in tqdm.tqdm(rc_n_subfoders) :
                sub_folder=os.path.join(self.directory_path,folder_name)
                #if subfoder name doesnt have "test"
                if "test" not in sub_folder:                
                    self.loadOneTaskFolder(sub_folder)          

        elif mode=="test":
            print("Preloading test data......")
            rc_n_subfoders=os.listdir(self.directory_path)
            for folder_name in tqdm.tqdm(rc_n_subfoders) :
                if "test_split"  == folder_name:                
                    test_splict_folder=os.path.join(self.directory_path,folder_name)
                    test_sub_folders=os.listdir(test_splict_folder)
                    for test_folder_name in test_sub_folders:
                        test_sub_folder=os.path.join(test_splict_folder,test_folder_name)              
                        self.loadOneTaskFolder(test_sub_folder)            


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
    ds=UnsteadyVastisDataset(args['root'], args['time_steps'],mode)
    return ds




if __name__ == '__main__':
    pass