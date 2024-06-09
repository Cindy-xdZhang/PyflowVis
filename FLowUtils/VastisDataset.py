import numpy as np
import os
from .VectorField2d import *
from .FlowReader import read_rootMetaGridresolution,loadOneFlowEntryRawData
import torch,time,tqdm

# create torch dataset using the load result function:
class UnsteadyVastisDataset(torch.utils.data.Dataset):
    def __init__(self, directory_path):
        directory_path=self.FixDataFolder(directory_path)
        #directory_path should be  the direct parent folder of all "rc_xxxx_n_xxx"  folders
        self.directory_path=os.path.join( os.path.join(directory_path,"unsteady"),"64_64")
        self.data=[]
        self.labelReferenceFrame=[]
        self.labelVortex=[]
        self.preLoading()

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
    

    def preLoading(self,slicePerData:int=7, TrainValidationSplitRatio:float=0.8):        
        print("Preloading data......")
        start = time.time()         
        rc_n_subfoders=os.listdir(self.directory_path)
        for folder_name in tqdm.tqdm(rc_n_subfoders) :
            sub_folder=os.path.join(self.directory_path,folder_name)
            folder_meta_file = os.path.join(sub_folder, 'meta.json')
            Xdim,Ydim,time_steps,dominMinBoundary,dominMaxBoundary=read_rootMetaGridresolution(folder_meta_file)
            #find all *.bin data in this subfoder
            binFiles = [f for f in os.listdir(sub_folder) if f.endswith('.bin')]
            for binFile in binFiles:
                binPath=os.path.join(sub_folder,binFile)
                #split according to slicePerData 
                loadField, labelReferenceFrameABC,vortexlabel=loadOneFlowEntryRawData(binPath, Xdim,Ydim,time_steps)
                timesteps=loadField.shape[0]
                TimeWindowCount=timesteps//slicePerData
                for i in range(0,TimeWindowCount):
                    sliceStart=i*slicePerData
                    sliceEnd=(i+1)*slicePerData
                    # dataSlice=loadField.getSlice(sliceStart,sliceEnd)
                    dataSlice=loadField[sliceStart:sliceEnd]
                    self.data.append(dataSlice)
                    self.labelReferenceFrame.append(labelReferenceFrameABC[sliceStart:sliceEnd])
                    self.labelVortex.append(vortexlabel)#vortexlabel=[tx,ty,n,rc] 
                    
        #logging dataset information    and time consuming of preloading data
        elapsed = time.time()
        elapsed = elapsed - start
        print("Preloading data......done")
        print(f"Total number of data:{len(self.data)}, took {elapsed} seconds")
        
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labelReferenceFrame[idx],self.labelVortex[idx]

def buildDataset(args):
    ds=UnsteadyVastisDataset(args['root'])
    return ds




if __name__ == '__main__':
    pass