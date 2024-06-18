import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from FLowUtils.LicRenderer import LicRenderingUnsteady,LicRenderingSteady
from FLowUtils.FlowReader import read_rootMetaGridresolution,loadOneFlowEntryRawData
from FLowUtils.VectorField2d import UnsteadyVectorField2D,SteadyVectorField2D
from DeepUtils.VastisDataset import buildDataset
from config.LoadConfig import load_config
from DeepUtils.ModelZoo import *
from DeepUtils.MiscFunctions import *
import logging,random,wandb

GLOBAL_WANDB_PROJECT_NAME="DeepVortexExtraction"
initLogging()

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

def runNameTagGenerator(config, tagattach) ->(str,list[:str]):
    seed=config['training']['random_seed']
    runName=f"{config['network']['family_name']}_{config['dataset']['name']}_{config['training']['epochs']}_{config['training']['learning_rate']}_seed_{seed}"
    runTags= [config['network']['family_name'],config['dataset']['name']]+tagattach
    return runName,runTags
    

def validate(model, data_loader, device,config) -> float:
    slicePerdata = config["dataset"]['time_steps']
    model.eval()
    val_loss = 0
    val_rec_loss = 0
    with torch.no_grad():
         for batch_idx, (vectorFieldImage, labelQtct, labelVortex) in enumerate(data_loader):
            # tx, ty, n, rc = labelVortex[0], labelVortex[1], labelVortex[2], labelVortex[3]
            steadyField3D = vectorFieldImage[:, :, :, :, 0]
            steadyField3D = steadyField3D.unsqueeze(-1).repeat(1, 1, 1, 1, slicePerdata).to(device)
            vectorFieldImage, labelQtct = vectorFieldImage.to(device), labelQtct.to(device)
            predictQtCt, rec = model(vectorFieldImage)

            lossRec = F.mse_loss(rec, steadyField3D)
            lossRef = F.mse_loss(predictQtCt, labelQtct)
            loss = lossRec + lossRef
            val_loss += loss.item()
            val_rec_loss += lossRec.item()

    val_loss /= len(data_loader)
    val_rec_loss /= len(data_loader)
    model.train()
    return val_loss, val_rec_loss

 


def test_model(model,config):
    device = config['training']['device']
    testDataset=buildDataset(config["dataset"],mode="test")
    test_data_loader= torch.utils.data.DataLoader(testDataset, batch_size=config['training']['batch_size'], shuffle=False)
    slicePerdata = config["dataset"]["time_steps"]
    model.eval()
    reconstruction_error = 0
    test_loss = 0
    test_loss_records=[]
    save_folder=f"./testOutput/{config['run_name']}/"
    with torch.no_grad():
        for batch_idx, (vectorFieldImage, labelQtct, labelVortex) in enumerate(test_data_loader):
            # tx, ty, n, rc = labelVortex[0], labelVortex[1], labelVortex[2], labelVortex[3]
            steadyField3D = vectorFieldImage[:, :, :, :, 0]
            steadyField3D = steadyField3D.unsqueeze(-1).repeat(1, 1, 1, 1, slicePerdata).to(device)
            vectorFieldImage, labelQtct = vectorFieldImage.to(device), labelQtct.to(device)
            predictQtCt, rec = model(vectorFieldImage)
            lossRec = F.mse_loss(rec, steadyField3D)
            lossRef = F.mse_loss(predictQtCt, labelQtct)
            loss = lossRec + lossRef

            reconstruction_error += lossRec.item()
            test_loss += loss.item()
            test_loss_records.append(loss.item())

        #random select 10 samples to visualize
        for i in range(5):
            sample=random.randint(0,len(test_data_loader))
            vectorFieldImage, labelQtct, labelVortex=testDataset[sample]
            binaryName=testDataset.getBinaryName(sample)

            vectorFieldImage=vectorFieldImage.unsqueeze(0)
            #vectorFieldImage shape is [bs=1,  chanel=2,W=64, H=64, depth(timsteps)]
            steadyField3D = vectorFieldImage[0, :, :, :, 0].transpose(0, 2).cpu()
 

            vectorFieldImage, labelQtct = vectorFieldImage.to(device), labelQtct.to(device)
            predictQtCt, recField = model(vectorFieldImage)
            #recField [batch_size, chanel=2,W=64, H=64, depth(timsteps)]
            recField=recField[0,:,:,:,:] .cpu()
            recField=recField.transpose(0,3)
            recUnsteadyField=  UnsteadyVectorField2D(64,64,slicePerdata)
            recUnsteadyField.field=recField
            steadyField=  SteadyVectorField2D(64,64)
            steadyField.field=steadyField3D.numpy()
            LicRenderingUnsteady(recUnsteadyField,64,2,save_folder,f"sample{sample}_{binaryName}__rec")
            LicRenderingSteady(steadyField,64,2,save_folder,f"sample{sample}_{binaryName}__gt")





    reconstruction_error /= len(test_data_loader)
    test_loss /= len(test_data_loader)
    min_loss,max_loss=min(test_loss_records),max(test_loss_records)
    model.train()
    return test_loss,min_loss,max_loss,reconstruction_error


def train_pipeline():
    config=load_config("config\\cfgs\\config.yaml")
    training_args=config['training']
    # Initialize training parameters from args
    epochs = training_args['epochs']
    device = training_args['device']
    #generate radom seed and record to config
    config['training']['random_seed']=torch.seed()
    run_Name,runTags=runNameTagGenerator(config,tagattach=["Q(t)c(t)","referenceFrameReconstruct"])

    config['wandb']=False
    # Initialize wandb
    if config['wandb']:
        try:
            wandb.init(project=GLOBAL_WANDB_PROJECT_NAME,
                       name=run_Name,
                       tags=runTags,
                       config=config)
        except:
            config['wandb'] = False
            logging.error("wandb init failed, reset config to disable wandb")        
            
    logging.info(config)
    logging.info(f"run name: {run_Name}, run tags: {runTags}")
    config['run_name']=run_Name

    #build data loader using the dataset and args
    train_VastisDataset=buildDataset(config["dataset"],mode="train")   
    config["dataset"].update(train_VastisDataset.dastasetMetaInfo)  
    xdim, ydim,timesteps= config["dataset"]["Xdim"],config["dataset"]["Ydim"],config["dataset"]["time_steps"]
    validationDataset=buildDataset(config["dataset"],mode="val")   
    
    data_loader = torch.utils.data.DataLoader(train_VastisDataset, batch_size=training_args['batch_size'], shuffle=training_args['shuffle'], num_workers=training_args['num_workers'], pin_memory=training_args['pin_memory'])
    validation_data_loader= torch.utils.data.DataLoader(validationDataset, batch_size=training_args['batch_size'], shuffle=False)
    
    
 

    # Initialize the model
    model = ReferenceFrameExtractor(2,  xdim, ydim, timesteps, ouptputDim=6*timesteps, hiddenSize=64)
    model.apply(init_weights)
    model.to(device)

    # Initialize the optimizer with L2 regularization
    optimizer = optim.Adam(model.parameters(), lr= training_args['learning_rate'], weight_decay=training_args['weight_decay'])
    best_val_loss = float('inf')

    # Training
    total_iterations = 0
    for epoch in range(epochs):
        epoch_loss = 0
        
        for batch_idx, (vectorFieldImage, labelQtct, labelVortex) in enumerate(data_loader):

            # tx, ty, n, rc = labelVortex[0], labelVortex[1], labelVortex[2], labelVortex[3]
            steadyField3D = vectorFieldImage[:, :, :, :, 0]
            steadyField3D = steadyField3D.unsqueeze(-1).repeat(1, 1, 1, 1, timesteps).to(device)
            vectorFieldImage, labelQtct = vectorFieldImage.to(device), labelQtct.to(device)
            predictQtCt, rec = model(vectorFieldImage)

            lossRec = F.mse_loss(rec, steadyField3D)
            lossRef = F.mse_loss(predictQtCt, labelQtct)
            loss = lossRec + lossRef

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            total_iterations += 1
            if batch_idx % training_args['print_frequency'] == 0:
                stepRecError = lossRec.item()
                logging.info(f'Epoch {epoch+1}, iter {batch_idx+1}, total_iterations: {total_iterations}. Loss: {loss.item()}.RecLoss: {stepRecError}')
                if config['wandb']:
                    wandb.log({"train_loss": loss.item(),  "step_rec_error":stepRecError,"total_iterations": total_iterations})
            
        epoch_loss /= len(data_loader)
        
        # Validate the model
        val_loss,val_rec_error = validate(model, validation_data_loader, device,config=config)  
        logging.info(f'Epoch {epoch+1}, Loss: {epoch_loss}, Val Loss: {val_loss}, Val Rec Error: {val_rec_error}')
        if config['wandb']:
            wandb.log({"epoch": epoch+1,  "epoch_Loss": {epoch_loss}, "val_loss": val_loss, "val_rec_error": val_rec_error})

        #save best model
        if val_loss < best_val_loss and training_args['save_model'] and epoch % training_args["save_model_frequency"]== 0:
            best_val_loss = val_loss
            saving_path= os.path.join(training_args['save_model_path'],run_Name) 
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, folder_name=saving_path, checkpoint_name= f'best_checkpoint.pth.tar')

    #test 
    avgtest_loss,mintest_error,maxtest_error,reconstruction_error=test_model(model,config=config)


    if config['wandb']:
        wandb.summary.update({"best_val_loss": best_val_loss})
        wandb.summary.update({"avg_test_loss": avgtest_loss})
        wandb.summary.update({"min_test_error": mintest_error})
        wandb.summary.update({"max_test_error": maxtest_error})
        wandb.summary.update({"test_rec_error": reconstruction_error})
        wandb.finish()



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


def testVastisDataset():
    config=load_config("config\\cfgs\\config.yaml")
    train_VastisDataset=buildDataset(config["dataset"],mode="train")   
    validationDataset=buildDataset(config["dataset"],mode="val")   
    testDataset=buildDataset(config["dataset"],mode="test")
    print(train_VastisDataset[0])    

# if __name__ == '__main__':
#    test_load_results()

if __name__ == '__main__':
    train_pipeline()