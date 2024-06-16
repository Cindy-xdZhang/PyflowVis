import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from FLowUtils.LicRenderer import LicRenderingUnsteady
from FLowUtils.FlowReader import read_rootMetaGridresolution,loadOneFlowEntryRawData
from FLowUtils.VectorField2d import UnsteadyVectorField2D
from DeepUtils.VastisDataset import buildDataset
from config.LoadConfig import load_config
from DeepUtils.NetworkFactory import *
from DeepUtils.MiscFunctions import *
import wandb
GLOBAL_WANDB_PROJECT_NAME="DeepVortexExtraction"


#! TODO (s) of PyFLowVis
#! todo: load unsteady data from cpp generated binary file [DONE]
#! todo: create torch dataset   [DONE]
#! todo: what is the label? ->fist stage the reference frame: a(t),b(t),c(t)  [DONE]
#! todo: DEFINE THE cnn MODEL:   NET0: VORETXNET, NET1: VORETXsegNET, NET3: RESNET , net4 vortexViz
#! todo: what is the label? ->second stage the segmentation of as steady as possible (asap) field.
#! todo: visualize the model's output: classification of vortex bondary+ coreline  in 3d space (2d space+ 1D time)
#! todo:test the model's with RFC, bossineq, helix, spriral motion
#? - [ ]  TODO: draw pathline?
#? - [ ]  TODO: train vector field for killing +regular+ Di term 
#? - [ ]  TODO: curvature gradient
#? - [ ]  TODO: implement hanger of imgui widget

def validate(model, data_loader, device,config) -> float:
    slicePerdata = config["dataset"]['time_steps']
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_idx, (vectorFieldImage, labelferenceFrame, labelVortex) in enumerate(data_loader):
            # vectorFieldImage shape is [batch_size, depth(timsteps)=7, W=64, H=64, chanel=2]
            # transpose to [batch_size, chanel=2, W=64, H=64, depth(timsteps)=7]
            vectorFieldImage = vectorFieldImage.transpose(1, 4)
            steadyField3D = vectorFieldImage[:, :, :, :, 0]
            # tx, ty, n, rc = labelVortex[0], labelVortex[1], labelVortex[2], labelVortex[3]
            vectorFieldImage, label = vectorFieldImage.to(device), labelVortex.to(device)
            steadyField3D = steadyField3D.unsqueeze(-1).repeat(1, 1, 1, 1, slicePerdata).to(device)
            Qt, tc = labelferenceFrame
            labelQtct = torch.concat((Qt, tc), dim=2)
            labelQtct = labelQtct.reshape(vectorFieldImage.shape[0], -1).to(device)

            predictQtCt, rec = model(vectorFieldImage)
            lossRec = F.mse_loss(rec, steadyField3D)
            lossRef = F.mse_loss(predictQtCt, labelQtct)
            loss = lossRec + lossRef

            val_loss += loss.item()

    val_loss /= len(data_loader)
    model.train()
    return val_loss


def train_pipeline():
    config=load_config("config\\cfgs\\config.yaml")
    training_args=config['training']
    # Initialize training parameters from args
    epochs = training_args['epochs']
    device = training_args['device']
    # Initialize wandb
    if config['wandb']:
        try:
            wandb.init(project=GLOBAL_WANDB_PROJECT_NAME, config=config)
        except:
            config['wandb'] = False
            print("wandb init failed, reset config to disable wandb")        

    #build data loader using the dataset and args
    train_VastisDataset=buildDataset(config["dataset"],mode="train")   
    validationDataset=buildDataset(config["dataset"],mode="val")   
    data_loader = torch.utils.data.DataLoader(train_VastisDataset, batch_size=training_args['batch_size'], shuffle=training_args['shuffle'], num_workers=training_args['num_workers'], pin_memory=training_args['pin_memory'])
    validation_data_loader= torch.utils.data.DataLoader(validationDataset, batch_size=training_args['batch_size'], shuffle=False)
    
 

    # Initialize the model
    slicePerdata = config["dataset"]['time_steps']
    model = ReferenceFrameExtractor(2, 64, 64, slicePerdata, ouptputDim=6*slicePerdata, hiddenSize=64)
    model.apply(init_weights)
    model.to(device)

    # Initialize the optimizer with L2 regularization
    optimizer = optim.Adam(model.parameters(), lr= training_args['learning_rate'], weight_decay=training_args['weight_decay'])
    best_val_loss = float('inf')

    # Training
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, (vectorFieldImage, labelferenceFrame, labelVortex) in enumerate(data_loader):
            # vectorFieldImage shape is [batch_size, depth(timsteps)=7, W=64, H=64, chanel=2]
            # transpose to [batch_size, chanel=2, W=64, H=64, depth(timsteps)=7]
            vectorFieldImage = vectorFieldImage.transpose(1, 4)
            steadyField3D = vectorFieldImage[:, :, :, :, 0]
            # tx, ty, n, rc = labelVortex[0], labelVortex[1], labelVortex[2], labelVortex[3]
            vectorFieldImage, label = vectorFieldImage.to(device), labelVortex.to(device)
            steadyField3D = steadyField3D.unsqueeze(-1).repeat(1, 1, 1, 1, slicePerdata).to(device)
            Qt, tc = labelferenceFrame
            labelQtct = torch.concat((Qt, tc), dim=2)
            labelQtct = labelQtct.reshape(vectorFieldImage.shape[0], -1).to(device)

            predictQtCt, rec = model(vectorFieldImage)
            lossRec = F.mse_loss(rec, steadyField3D)
            lossRef = F.mse_loss(predictQtCt, labelQtct)
            loss = lossRec + lossRef

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if batch_idx % training_args['print_frequency'] == 0:
                print(f'Epoch {epoch+1}, iter {batch_idx+1},  Loss: {loss.item()}')
                if config['wandb']:
                    wandb.log({"train_loss": loss.item(),  "epoch": epoch+1, "iteration": batch_idx})
            
        epoch_loss /= len(data_loader)
        # Validate the model
        val_loss = validate(model, validation_data_loader, device,config=config)  
        print(f'Epoch {epoch+1}, Loss: {epoch_loss}, Val Loss: {val_loss}')
        if config['wandb']:
            wandb.log({"epoch": epoch+1,  "Loss": {epoch_loss}, "val_loss": val_loss})
        if val_loss < best_val_loss and training_args['save_model'] and epoch % training_args["save_model_frequency"]== 0:
            best_val_loss = val_loss
            save_checkpoint({
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, folder_name= training_args['save_model_path'], checkpoint_name= f'best_checkpoint_{epoch+1}.pth.tar')

    if config['wandb']:
        wandb.finish()

 

def test_model(model,data_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (vectorFieldImage, labelferenceFrame, labelVortex) in enumerate(data_loader):
            vectorFieldImage = vectorFieldImage.transpose(1, 3)
            vectorFieldImage, label = vectorFieldImage, labelVortex
            output = model(vectorFieldImage)
            test_loss += F.mse_loss(output, labelVortex, reduction='sum').item()  # sum up batch loss

    test_loss /= len(data_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}\n')



def test(checkpoint_path):
    # config = load_config("config\\cfgs\\config.yaml")
    # test_args = config['testing']
    # unsteadyVastisDataset = buildDataset(test_args["dataset"])

    # data_loader = torch.utils.data.DataLoader(unsteadyVastisDataset, batch_size=test_args['batch_size'], shuffle=False, num_workers=test_args['num_workers'], pin_memory=test_args['pin_memory'])

    # device = test_args['device']
    # model = VortexClassifier()
    # model.to(device)

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # # Load the last checkpoint
    # load_checkpoint(torch.load(checkpoint_path), model, optimizer)

    # model.eval()
    # test_loss = 0
    # correct = 0

    # with torch.no_grad():
    #     for batch_idx, (vectorFieldImage, labelferenceFrame, labelVortex) in enumerate(data_loader):
    #         vectorFieldImage = vectorFieldImage[:,0,:,:]
    #         vectorFieldImage = vectorFieldImage.transpose(1, 3)
    #         vectorFieldImage, label = vectorFieldImage.to(device), labelVortex.to(device)
    #         output = model(vectorFieldImage)
    #         test_loss += F.mse_loss(output, labelVortex, reduction='sum').item()  # sum up batch loss

    # test_loss /= len(data_loader.dataset)
    # print(f'\nTest set: Average loss: {test_loss:.4f}\n')

    return None




def test_load_results(): 
    Xdim,Ydim,time_steps,dominMinBoundary,dominMaxBoundary=read_rootMetaGridresolution('C:\\Users\\zhanx0o\\Documents\\sources\\PyflowVis\\CppProjects\\data\\unsteady\\64_64_nomix\\velocity_rc_1n_2\\meta.json')
    directory_path = 'C:\\Users\\zhanx0o\\Documents\\sources\\PyflowVis\\CppProjects\\data\\unsteady\\64_64_nomix\\velocity_rc_1n_2\\rc_1_n_2_sample_0Si_1observer_1type_4.bin'
    loadField, labelReferenceFrameABC,votexInfo=loadOneFlowEntryRawData(directory_path,Xdim,Ydim,time_steps)
    dataSlice=loadField[0:7]    
    unsteadyField=UnsteadyVectorField2D(Xdim,Ydim,7,dominMinBoundary,dominMaxBoundary) 
    LicRenderingUnsteady(unsteadyField,128,1)


    

# if __name__ == '__main__':
#    test_load_results()

if __name__ == '__main__':
    train_pipeline()