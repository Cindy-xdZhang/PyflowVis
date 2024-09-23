import torch,random
import numpy as np
import os
import datetime
from DeepUtils.dataset import build_dataloader_from_cfg
from FLowUtils.VectorField2d import UnsteadyVectorField2D
from DeepUtils.dataset import UnsteadyVastisDataset
from PIL import Image
from FLowUtils.vortexCriteria import *
from FLowUtils.LicRenderer import *
from DeepUtils.MiscFunctions import argParseAndPrepareConfig,readDataSetRelatedConfig
from DeepUtils.models import build_model_from_cfg
from DeepUtils.dataset.data_utils import read_binary_file


class TestReconstructSteadyField(object):
    def __init__(self, device, data_dir,**kwargs):
        self.device=device
        bs=1
        #deformingZerofieldData
        self.dataset= UnsteadyVastisDataset(data_dir,"test",None)

    def __call__(self,model,test_data_loader) -> None:
        from FLowUtils.LicRenderer import LicRenderingUnsteadyCpp
        # from FLowUtils.GlyphRenderer import glyphsRenderUnsteadyField
        device=self.device
        minV=-3.699822425842285
        maxV= 3.9069676399230959
        model.eval()
        total_error=0
        for sample in range(len( self.dataset)):
            vectorFieldImage, label=self.dataset[sample]
            UnsteadyField=  UnsteadyVectorField2D(16,16,5,[-2,-2],[2,2],tmin=0,tmax=0.7853981633974483)
            UnsteadyField.field=vectorFieldImage.transpose(0,-1).cpu().numpy()
            #feed network
            vectorFieldImage=(vectorFieldImage-minV)/(maxV-minV)
            vectorFieldImage = vectorFieldImage.unsqueeze(0).to(device)
            predictition= model(vectorFieldImage)
            predictition=predictition[0].cpu().numpy()
            name=self.dataset.getSampleName(sample)
            abc,abc_dot=predictition[0:3],predictition[3:6]
            recRes=referenceFrameReconstruct(abc,abc_dot,UnsteadyField)
            LicRenderingUnsteadyCpp(recRes,128,timeStepSKip=2,saveFolder="./testOutput",saveName=f"lic__{name}_rec")
            print(f"reconstruct task {name}, predicts {predictition}, vs label ={label}")
            diff= (recRes.field-UnsteadyField.field)*(recRes.field-UnsteadyField.field)
            print(f"reconstructdiff max={diff.max()}, min={diff.min()}, mean={diff.mean()}")
            total_error+=diff.mean()
        print(f"reconstruct  totaldiff ={total_error}")
        return None    
    
def save_segmentation_as_png(vortexsegmentationLabel, filename, upSample=1.0):

    """
    Saves a 2D binary segmentation as a PNG file.

    Parameters:
        vortexsegmentationLabel (numpy.ndarray): The segmentation array of shape (Ydim, Xdim, 2).
        filename (str): The filename to save the PNG image.
        upSample (float): Upsampling factor to resize the image. Default is 1.0 (no scaling).
    """
    # Create the directory if it does not exist
    folder = os.path.dirname(filename)  # Extract the folder path from the filename
    if folder and not os.path.exists(folder):  # Ensure folder is non-empty and doesn't exist
        os.makedirs(folder)
        print(f"Created directory: {folder}")
    
    # Convert the segmentation to a binary mask
    if len(vortexsegmentationLabel.shape)==3:
        binary_mask = np.where(vortexsegmentationLabel[..., 1] > 0.5, 255, 0).astype(np.uint8)
        
    binary_mask = np.where(vortexsegmentationLabel > 0.5, 255, 0).astype(np.uint8)
    
    # Create an image from the binary mask
    image = Image.fromarray(binary_mask, mode='L')  # 'L' mode for (8-bit pixels, black and white)
    
    # Apply upsampling if needed
    if upSample != 1.0:
        new_size = (int(image.width * upSample), int(image.height * upSample))
        image = image.resize(new_size, Image.NEAREST)  # Use NEAREST for upsampling binary images
    
    # Save the image
    image.save(filename)

def segmentationCriteria(pred, gt):
    """
    Computes precision, recall, F1 score, and Intersection over Union (IoU) for segmentation.
    Parameters:
        pred (numpy.ndarray): Predicted segmentation mask, shape [batch_size, width, height, 2].
        gt (numpy.ndarray): Ground truth segmentation mask, shape [batch_size, width, height, 2].
    Returns:
        np.array( [TP,FP,FN, precision, recall, F1, IoU],dtype=np.float32) 
    """
    # Extract the binary segmentation mask (second channel)
    pred_mask = pred[..., 0]  # shape [batch_size, width, height]
    gt_mask = gt[..., 0]      # shape [batch_size, width, height]
    
    # Flatten the masks to compute metrics for the entire batch
    pred_flat = pred_mask.flatten()
    gt_flat = gt_mask.flatten()
    total_samples=gt_flat.shape[0]

    # True positives, False positives, False negatives
    TP = np.sum((pred_flat > 0.5) & (gt_flat == 1))  # True Positive
    FP = np.sum((pred_flat > 0.5) & (gt_flat == 0))  # False Positive
    FN = np.sum((pred_flat < 0.5) & (gt_flat == 1))  # False Negative

    # Precision: TP / (TP + FP)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0

    # Recall: TP / (TP + FN)
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    # F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
    F1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Intersection over Union (IoU): TP / (TP + FP + FN)
    IoU = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0

    TP=TP/float(total_samples)
    FP=FP/float(total_samples)
    FN=FN/float(total_samples)
    return np.array( [TP,FP,FN, precision, recall, F1, IoU],dtype=np.float32) 





class TestSegmentation(object):
    """ TestSegmentation  is the default test task for Segmentation tasks """
    def __init__(self, device,config,samples=10,**kwargs):
          self.device=device
          self.samples=samples
          self.runName=config["run_name"]


    def __call__(self, model,test_data_loader):
        device=self.device
        segError=0.0
        meta_=test_data_loader.dataset.dastasetMetaInfo
        Xdim,Ydim=meta_["Xdim"],meta_["Ydim"]
        dm_min,dm_max=meta_["domainMinBoundary"],meta_["domainMaxBoundary"]
        grid_dx,grid_dy=(dm_max[0]-dm_min[0])/float( Xdim-1),(dm_max[1]-dm_min[1])/float( Ydim-1)
        test_loss=0
     
        for batch_idx, (data, label) in enumerate(test_data_loader):
            if isinstance(data, list):
                # Unpack the tuple
                vectorFieldImage, pathlines = data
                # Move each element to the device
                vectorFieldImage = vectorFieldImage.to(device)
                pathlines = pathlines.to(device)
                label = label.to(device)
                # Repack into a tuple if needed
                data = (vectorFieldImage, pathlines)
            else:
                # If data is not a tuple, directly move to the device
                data = data.to(device)
                label = label.to(device)    
            predictition= model(data)
            loss=model.get_loss(predictition,label)
            test_loss += loss.item()                
            segError_=segmentationCriteria(predictition.cpu().numpy(),label.cpu().numpy())
            segError+=segError_

        segError /= len(test_data_loader)
        TP,FP,FN, precision, recall, F1, IoU=segError[0],segError[1],segError[2],segError[3],segError[4],segError[5],segError[6]
        print(f"TP,FP,FN={TP},{FP},{FN}")
        print(f"precision, recall, F1, IoU={precision},{recall},{F1},{IoU}")
        
        result = {
            "precision": precision,
            "F1": F1,
            "IoU": IoU
        }
        return result
    
        
        # #random select  samples to visualize
        # for i in range(self.samples):
        #     sample=random.randint(0,len(test_data_loader.dataset)-1)
        #     data, label=test_data_loader.dataset[sample]
        #     if isinstance(data, list) or isinstance(data, tuple) :
        #         # Unpack the tuple
        #         vectorFieldImage, pathlines = data
        #         # Move each element to the device
        #         batch_vectorFieldImage = vectorFieldImage.unsqueeze(0).to(device)
        #         pathlines = pathlines.unsqueeze(0).to(device)
        #         label = label.to(device)
        #         # Repack into a tuple if needed
        #         data = (batch_vectorFieldImage, pathlines)
        #     else:
        #         # If data is not a tuple, directly move to the device
        #         data = data.to(device)
        #         label = label.to(device)    

            
        #     predictition= model(data)
        #     predictition=predictition[0].cpu().numpy()
        #     label=label.cpu().numpy()
        #     name=test_data_loader.dataset.getSampleName(sample)
        #     save_segmentation_as_png(predictition,f"./testOutput/{self.runName}/{name}_pred.png",upSample=10.0)
        #     save_segmentation_as_png(label,f"./testOutput/{self.runName}/{name}_gt.png",upSample=10.0)
        #     rawVectorField=vectorFieldImage.transpose(0,-1).cpu().numpy()
            
            # qCriterion=computeQcriterion(rawVectorField,grid_dx,grid_dy)
            # ivd=computeIVD(rawVectorField,grid_dx,grid_dy)
            # saveCriteriaPicture(qCriterion,f"./testOutput/{run_name}/{name}_q_cri.png",upSample=10.0)
            # saveCriteriaPicture(ivd,f"./testOutput/{run_name}/{name}_ivd.png",upSample=10.0)
            # precision, recall, F1, IoU=segmentationCriteria(predictition,labelVortex)


def pathlineSegToFieldSeg(Pathlines,PathlineSeg,Xdim,Ydim,DominMin,DominMax):
    L,K,C=Pathlines.shape
    # Extract seeding points from Pathlines
    seeding_points = Pathlines[ 0,:, :2]  # Assuming first two dimensions are x, y coordinates
    dx,dy=(DominMax[0]-DominMin[0])/float(Xdim-1),(DominMax[1]-DominMin[1])/float(Ydim-1)
    # Compute the segmentation for each grid point
    grid_segmentation = np.zeros((Ydim, Xdim), dtype=np.float32)
    for (idx) in range(K):
        this_pathline_seeding_pos=seeding_points[idx]
        corresponding_discrete_gridx=int((this_pathline_seeding_pos[0]-DominMin[0])/(dx))
        corresponding_discrete_gridy=int((this_pathline_seeding_pos[1]-DominMin[1])/(dy))
        corresponding_discrete_gridx = max(0, min(corresponding_discrete_gridx, Xdim-1))
        corresponding_discrete_gridy = max(0, min(corresponding_discrete_gridy, Ydim-1))
        grid_segmentation[corresponding_discrete_gridy,corresponding_discrete_gridx]+= PathlineSeg[idx]
        
    return grid_segmentation


class TestPathlineSeg(object):
    def __init__(self, device, config,samples=30, **kwargs):
          self.device=device
          self.samples=samples
          self.runName=config["run_name"]
          self.data_dir=config["dataset"]["data_dir"]
          self.outputPathlineLength=config["outputPathlineLength"]
          self.outputPathlinesCountK=config["outputPathlinesCountK"]        
          self.pathlineFeatures=config["PathlineFeature"]
          self.config=config
               
    
    def __call__(self, model,test_data_loader) -> torch.Any:
        device=self.device
        # first random select  samples to visualize
        out_folder=f"./testOutput/{self.runName}"
        print(f"TestPathlineSeg save to {out_folder}")
        Xdim=test_data_loader.dataset.dastasetMetaInfo["Xdim"]
        Ydim=test_data_loader.dataset.dastasetMetaInfo["Ydim"]
        Tdim=getattr(test_data_loader.dataset.dastasetMetaInfo,"unsteadyFieldTimeStep",5)
        DomainMin=test_data_loader.dataset.dastasetMetaInfo["domainMinBoundary"]
        DomainMax=test_data_loader.dataset.dastasetMetaInfo["domainMaxBoundary"]

        
        for i in range(self.samples):
            # sample=random.randint(0,len(test_data_loader.dataset)-1)
            sample=i
            data, label=test_data_loader.dataset[sample]
            assert(isinstance(data, list) or isinstance(data, tuple) ) 
            # Unpack the tuple
            vectorFieldImage, pathlines = data
            # Repack into a tuple if needed
            data = (None, pathlines.unsqueeze(0).to(device))
            predictition= model(data)
            predictition=predictition[0].cpu().numpy()
            label=label.cpu().numpy()
            name=test_data_loader.dataset.getSampleName(sample)
            
            UnsteadyField=  UnsteadyVectorField2D(Xdim,Ydim,Tdim,DomainMin ,DomainMax)
            UnsteadyField.field=vectorFieldImage.transpose(0,-1).cpu().numpy()
            label_seg=pathlineSegToFieldSeg(pathlines,label,Xdim,Ydim,DomainMin,DomainMax)
            predictition_seg=pathlineSegToFieldSeg(pathlines,predictition,Xdim,Ydim,DomainMin,DomainMax)
            LicRenderingPathlineSegmentation(UnsteadyField,predictition_seg,4.0,saveFolder=out_folder,saveName=f"{name}__pred")
            LicRenderingPathlineSegmentation(UnsteadyField,label_seg,4.0,saveFolder=out_folder,saveName=f"{name}__gt")
                
        #then   visualize resulst on analytical field
        analytical_field_Folder= os.path.join(os.path.dirname(self.data_dir),"analytical")
        
        outputPathlinesCount=int(self.outputPathlinesCountK//2) *int(self.outputPathlinesCountK//2) *5
        # if os.path.exists(analytical_field_Folder):
        #       pathline_Files = [f  for f in os.listdir(analytical_field_Folder) if f.endswith('_pathline.bin')]
        #       for pathline_file in pathline_Files:
        #             name=pathline_file.replace("_pathline.bin","")
                  
        #             pathline_file_dir= os.path.join(analytical_field_Folder,pathline_file) 
        #             pathlineClusters = read_binary_file(pathline_file_dir).reshape(outputPathlinesCount, self.outputPathlineLength,self.pathlineFeatures)
        #             raw_data_file=pathline_file_dir.replace("_pathline.bin",".bin")
        #             raw_Binary = read_binary_file(raw_data_file).reshape(self.config["unsteadyFieldTimeStep"],self.config["Ydim"],self.config["Xdim"], 2)
                    
        #             pathlineClusters = np.transpose(pathlineClusters, (1, 0, 2))[:,:,:self.pathlineFeatures]#L,K,C
        #             pathlines_tensor=torch.tensor(pathlineClusters)
        #             data = (None, pathlines_tensor.unsqueeze(0).to(device))
        #             predictition= model(data)
        #             predictition=predictition[0].cpu().numpy()
        #             UnsteadyField=  UnsteadyVectorField2D(32,32,5,[-2,-2],[2,2],np.pi * 0.25)
        #             UnsteadyField.field=raw_Binary
        #             predictition_seg=pathlineSegToFieldSeg(pathlineClusters,predictition,Xdim=32,Ydim=32,DominMin=[-2,-2],DominMax=[2,2])
        #             LicRenderingPathlineSegmentation(UnsteadyField,predictition_seg,4.0,saveFolder=out_folder,saveName=f"{name}__pred")
        # else:
        #     print(f"{analytical_field_Folder} doesnt exist.") 
                    
                    
                    
        
        
            

   
      

def test_model(model,cfg):
    device = cfg['device']
    test_data_loader = build_dataloader_from_cfg(cfg.batch_size,
                                        cfg.dataset,
                                        cfg.dataloader,
                                        datatransforms_cfg=cfg.datatransforms,
                                        split='test'                                             
                                        )
    print(f"length of test dataset: {len(test_data_loader.dataset)}")
    model.to(cfg['device'])
    model.eval()
    #building test tasks
    test_cfg=cfg['test_tasks']
    kwagrs=test_cfg["kwargs"] if "kwargs" in test_cfg else {}
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    cfg['run_name']=getattr(cfg,'run_name',f"default/{timestamp}")
    
    test_tasks=[]
    for  cfg_task_name in test_cfg['tasks']:
        task_init_fn=eval(cfg_task_name)
        t=task_init_fn(device=device,config=cfg,**kwagrs)
        test_tasks.append(t)
        
    model.eval()
    retValues={}
    with torch.no_grad():
        for t in test_tasks:
            key=str(t.__class__.__name__)
            print(f"run test task: [{key}]")
            retValues[key]=t(model,test_data_loader)
    if "TestLoss" in retValues:
        retLoss=retValues["TestLoss"]
    elif "TestClassification" in retValues:
        retLoss=retValues["TestClassification"]
    elif "TestSegmentation" in retValues:
        retLoss=retValues["TestSegmentation"]
    else :
        retLoss=None
    return retLoss

    

def test_pipeline(model_path=None):
    cfg=argParseAndPrepareConfig()
    readDataSetRelatedConfig(cfg)
    model = build_model_from_cfg(cfg.model)
    if model_path is not None and os.path.exists(model_path):
        checkpoint=torch.load(model_path) 
        model.load_state_dict(checkpoint['state_dict'])
    
    model.to(cfg['device'])
    test_model(model,cfg)
  



if __name__ == '__main__':
    test_pipeline("outputModels\\bs_48_ep_240_lr_0.0001_20240922_212715_seed_3337\\epoch_211.pth.tar")



