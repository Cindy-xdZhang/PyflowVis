import torch,random
import logging
import numpy as np
import os
from .dataset.data_utils import  read_binary_file

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
        result = {
            "avg_test_loss": test_loss,
            "min_test_loss": min_loss,
            "max_test_loss": max_loss
        }
        return result
    
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
        result = {
            "avg_test_loss": test_loss,
            "min_test_loss": min_loss,
            "max_test_loss": max_loss
        }
        return result
    
    


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




