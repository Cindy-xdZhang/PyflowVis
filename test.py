import torch,random
import logging
from DeepUtils.dataset import build_dataloader_from_cfg
import numpy as np
from FLowUtils.VectorField2d import UnsteadyVectorField2D
from DeepUtils.dataset import UnsteadyVastisDataset
import os
from PIL import Image
from FLowUtils.vortexCriteria import *


class TestLoss(object):
    """ TestLoss is the default test task.  """
    def __init__(self, device,**kwargs):
          self.device=device

    def __call__(self, model,test_data_loader):
        device=self.device
        test_loss = 0
        test_loss_records=[]
        model.to(device)
        for batch_idx, (data, label) in enumerate(test_data_loader):
            data,label = data.to(device), label.to(device)
            predictition= model(data)                
            loss=model.get_loss(predictition,label)
            test_loss += loss.item()
            test_loss_records.append(loss.item())
        test_loss /= len(test_data_loader)
        min_loss,max_loss=min(test_loss_records),max(test_loss_records)
        logging.info(f'Avg test loss: {test_loss}, min test loss: {min_loss}, max test loss: {max_loss}')
        return (test_loss,min_loss,max_loss) 
    
class TestRandomSamples(object):
    """ TestRandomSamples random pick n samples and print the prediction and label"""
    def __init__(self, device,samples=20,**kwargs):
          self.device=device
          self.samples=samples

    def __call__(self, model,test_data_loader):
        device=self.device
        #random select  samples to visualize
        for i in range(self.samples):
            sample=random.randint(0,len(test_data_loader.dataset)-1)
            vectorFieldImage, labelVortex=test_data_loader.dataset[sample]
            vectorFieldImage = vectorFieldImage.unsqueeze(0).to(device)
            predictition= model(vectorFieldImage)
            predictition=predictition[0].cpu().numpy()
            logging.info(f"testSample{sample}_predict  {predictition}, vs label { labelVortex}")

class TestClassification(object):
    """ TestClassification  is the default test task for classification tasks """
    def __init__(self, device,**kwargs):
          self.device=device

    def __call__(self, model,test_data_loader):
        device=self.device
        test_loss = 0
        test_loss_records=[]
        correct=0
        model.to(device)
        for batch_idx, (data, label) in enumerate(test_data_loader):
            data,label = data.to(device), label.to(device)
            predictition= model(data)                
            loss=model.get_loss(predictition,label)
            test_loss += loss.item()
            test_loss_records.append(loss.item())
            predicted_classes = torch.argmax(predictition, dim=1)
            true_classes = torch.argmax(label, dim=1)
            # Compare and count the number of correct predictions
            correct += (predicted_classes == true_classes).sum().item()

        precision=float(correct)/float(len(test_data_loader.dataset)-1)
        logging.info(f"cls correctly predicts {correct} out of {len(test_data_loader.dataset)-1}, precision={precision*100}%.")
        test_loss /= len(test_data_loader)
        min_loss,max_loss=min(test_loss_records),max(test_loss_records)
        logging.info(f'Avg test loss: {test_loss}, min test loss: {min_loss}, max test loss: {max_loss}')
        return (test_loss,min_loss,max_loss) 
    
def read_binary_file(filepath, dtype=np.float32) -> np.ndarray:
    with open(filepath, 'rb') as file:
        data = np.fromfile(file, dtype=dtype)
        if dtype == np.float32:
            data=data[2:]
        elif dtype == np.float64:
            data=data[1:]        
    return data

class TestRotatingZeroField(object):

    def __init__(self, device,**kwargs):
          self.device=device
    def __call__(self,model,test_data_loader) -> None:                
        from FLowUtils.LicRenderer import LicRenderingUnsteadyCpp
        from FLowUtils.GlyphRenderer import glyphsRenderUnsteadyField
        device=self.device
        minV= -3.8220109939575197
        maxV= 3.5120744705200197
        def testOneSample(raw_data_file,correctlabel):
            raw_Binary = read_binary_file(raw_data_file).reshape(5,16,16, 2)
            name=raw_data_file.split("\\")[-1]
            # UnsteadyField=  UnsteadyVectorField2D(16,16,5,[-2,-2],[2,2],np.pi * 0.25)
            # UnsteadyField.field=raw_Binary
            # glyphsRenderUnsteadyField(UnsteadyField,800,timeStepSKip=1,saveFolder="./testOutput",saveName=f"glyph__{name}")
            # LicRenderingUnsteadyCpp(UnsteadyField,800,timeStepSKip=1,saveFolder="./testOutput",saveName=f"lic__{name}")
            model.eval()
            with torch.no_grad():
                for step in range(5):
                    slice_data=raw_Binary[step]
                    fieldData = torch.tensor(slice_data).transpose(0, -1).unsqueeze(0)
                    vectorFieldImage=(fieldData-minV)/(maxV-minV)
                    vectorFieldImage = vectorFieldImage.to(device)
                    predictition= model(vectorFieldImage) 
                    predictition=predictition[0].cpu()
                    logging.info(f"testRotatingZeroField {name} step {step}, network predicts {predictition}, vs label ={correctlabel}")

        testOneSample("CppProjects\\data\\rotatingZeroField\\sample_0saddle.bin",0)
        testOneSample("CppProjects\\data\\rotatingZeroField\\sample_6center_ccw.bin",0)
        testOneSample("CppProjects\\data\\rotatingZeroField\\sample_1NotZeroFieldSaddlemeta.bin",0)
        testOneSample("CppProjects\\data\\rotatingZeroField\\sample_1977center_cw.bin",1)
        return None


def referenceFrameReconstruct(abc,abcDot,inputfield:UnsteadyVectorField2D):
    """
    referenceFrameReconstruct is suffer from inputfield has limited domain size, and doesn't have analytical expression for point out side of its domain.
    """
    dt=inputfield.timeInterval
    # Initial values
    theta=0.0        
    theta_t = [0.0]
    angular_velocity = abc[2]  # abc is a numpy array of shape (3,)
    angular_velocities = [angular_velocity]
    

    translation_c=np.array([0.0, 0.0])
    translation_c_t = [np.array([0.0, 0.0])]
    translation_cdot = np.array([abc[0], abc[1]])  # translation velocity
    velocities = [translation_cdot]
    translation_cdotdot = np.array([abcDot[0], abcDot[1]])  # acceleration
    Q_tlist= [ np.array([
            [1.0, 0],
            [0, 1.0]
        ])]
    # Integrate rotation and translation
    for i in range(1, inputfield.time_steps):
        theta += dt * angular_velocity
        theta_t.append(theta)
        angular_velocity += dt * abcDot[2]
        angular_velocities.append(angular_velocity)
        Q_tlist.append( np.array([
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]]) )
        translation_c += dt * translation_cdot
        translation_c_t.append(translation_c)
        translation_cdot += dt * translation_cdotdot
        velocities.append(translation_cdot)

    #reconstruct:

    reconstructField=UnsteadyVectorField2D(16,16,5,[-2,-2],[2,2],tmin=0,tmax=0.7853981633974483)
    reconstructField.field=np.zeros([5,16,16, 2],dtype=np.float32)
    
    for t in range(0, inputfield.time_steps):
        # Rotation matrix Q_t based on theta
        theta=theta_t[t]
        Q_t = Q_tlist[t]
        Q_t_transpose = Q_t.T
        angular_velocity=angular_velocities[t]
        # Compute spin tensor (anti-symmetric matrix of angular velocity)
        spin_tensor = np.array([
            [0.0, angular_velocity],
            [-angular_velocity, 0.0]
        ])
        # Compute Q_dot
        Q_dot = np.dot(Q_t, spin_tensor)
        # Translation velocity at this time step
        translation_velocity = velocities[t]
        for y in range(0, inputfield.Ydim):
            for x in range(0, inputfield.Xdim):
                pos_x=np.array([inputfield.domainMinBoundary[0]+x*inputfield.gridInterval[0],inputfield.domainMinBoundary[1]+y*inputfield.gridInterval[1]])
                # Transformed position xStar
                x_star = np.dot(Q_t, pos_x) + translation_c_t[t]
                # Get the analytical vector from the input field at xStar and time t
                #x_star is physical coordinate, need convert to floating indices
                x_star_floatIndex_x=float(x_star[0]-inputfield.domainMinBoundary[0])/inputfield.gridInterval[0]
                x_star_floatIndex_y=float(x_star[1]-inputfield.domainMinBoundary[1])/inputfield.gridInterval[1]
                v_star_xstar = inputfield.getBilinearInterpolateVector(x_star_floatIndex_x,x_star_floatIndex_y,t)

                # Compute the velocity at the original position
                v_at_pos = np.dot(Q_t_transpose, (v_star_xstar - np.dot(Q_dot, pos_x) - translation_velocity))
                reconstructField.field[t][y][x]=v_at_pos
    return reconstructField


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
    binary_mask = np.where(vortexsegmentationLabel[..., 1] > 0.5, 255, 0).astype(np.uint8)
    
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
    def __init__(self, device,run_name,samples=10,**kwargs):
          self.device=device
          self.samples=samples
          self.runName=run_name


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
    test_cfg["run_name"]=cfg['run_name'] if "run_name" in cfg else "default"
    test_tasks=[]
    for  cfg_task_name in test_cfg['tasks']:
        task_init_fn=eval(cfg_task_name)
        kwagrs=test_cfg["kwargs"] if "kwargs" in test_cfg else {}
        t=task_init_fn(device,**test_cfg,**kwagrs)
        test_tasks.append(t)
    #if no specified test tasks, then append the default one:    
    if len(test_tasks)<1:
        test_tasks.append(TestLoss(device))

    model.eval()
    retValues={}
    with torch.no_grad():
        for t in test_tasks:
            key=str(t.__class__.__name__)
            retValues[key]=t(model,test_data_loader)
    

    if "TestLoss" in retValues:
        retLoss=retValues["TestLoss"]
    elif "TestClassification" in retValues:
        retLoss=retValues["TestClassification"]
    else :
        retLoss=(None,None,None)

    test_loss,min_loss,max_loss=retLoss
    return test_loss,min_loss,max_loss



if __name__ == '__main__':
    from DeepUtils.MiscFunctions import argParseAndPrepareConfig
    from DeepUtils.models import build_model_from_cfg
    cfg=argParseAndPrepareConfig()
    model = build_model_from_cfg(cfg.model)
    test_model(model,cfg)



