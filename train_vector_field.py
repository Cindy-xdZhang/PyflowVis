import torch
import torch.nn.functional as F
import torch.optim as optim
from FLowUtils.LicRenderer import *
from FLowUtils.VectorField2d import UnsteadyVectorField2D,SteadyVectorField2D
from FLowUtils.GlyphRenderer import *
from DeepUtils.VastisDataset import buildDataset
from DeepUtils.ModelZoo import *
from DeepUtils.MiscFunctions import *
import logging,random,wandb
import torchsummary

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
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    runName = f"{config['network']['family_name']}_{config['training']['epochs']}_{config['training']['learning_rate']}_{current_time}_seed_{seed}"
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

 


def test_model(model,config,testDataset=None):
    device = config['training']['device']
    if testDataset is None:
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

        #random select  samples to visualize
        for i in range(2):
            sample=random.randint(0,len(testDataset))
            vectorFieldImage, labelQtct, labelVortex=testDataset[sample]
            minv,maxv=labelVortex[4],labelVortex[5]
            binaryName=testDataset.getBinaryName(sample)

            vectorFieldImage=vectorFieldImage.unsqueeze(0)
            #vectorFieldImage shape is [bs=1,  chanel=2,W=Ydim, Xdim, depth(timsteps)]
            tmp =vectorFieldImage
            tmp =tmp.transpose(1, 4).cpu()
            steadyField3D = tmp[0, 0, :, :, :].cpu()
 

            vectorFieldImage, labelQtct = vectorFieldImage.to(device), labelQtct.to(device)
            predictQtCt, recField = model(vectorFieldImage)
            #recField [batch_size, chanel=2,W=64, H=64, depth(timsteps)]
            recField=recField[0,:,:,:,:] .cpu()
            recField=recField.transpose(0,3)
            recField=recField*(maxv-minv)+minv
            recUnsteadyField=  UnsteadyVectorField2D(64,64,slicePerdata)            
            recUnsteadyField.field=recField

            steadyField=  SteadyVectorField2D(64,64)
            steadyField3D=steadyField3D*(maxv-minv)+minv
            steadyField.field=steadyField3D.numpy()
            predictTransformation=predictQtCt[0].cpu().numpy()
            logging.info(f"testSample{sample}_predict Q(t)c(t){predictTransformation}, vs labelQtct{  labelQtct.cpu()}__rec")
            # LicRenderingUnsteadyCpp(recUnsteadyField,licImageSize=800,timeStepSKip=10,saveFolder=save_folder,saveName=f"testSample_{binaryName}__rec",stepSize=0.005,MaxIntegrationSteps=128)
            # glyphsRenderUnsteadyField(recUnsteadyField,ImageSize=800,timeStepSKip=10,saveFolder=save_folder,saveName=f"testSample_{binaryName}__rec_glyph",ColorCodingFn=lambda u, v: math.sqrt(u*u + v*v))
            LicGlyphMixRenderingUnsteady(recUnsteadyField,licImageSize=800,timeStepSKip=4,saveFolder=save_folder,saveName=f"testSample_{binaryName}_rec__licglyph",stepSize=0.005,MaxIntegrationSteps=128)

            LicRenderingSteadyCpp(steadyField,licImageSize=800,saveFolder=save_folder,saveName=f"testSample_{binaryName}__gt",stepSize=0.005,MaxIntegrationSteps=256)
            ouputSteadyPath=os.path.join(save_folder,f"testSample_{binaryName}__gt_glyph.png")
            glyphsRenderSteadyField(steadyField,ouputSteadyPath,(800,800),ColorCodingFn=lambda u, v: math.sqrt(u*u + v*v),gridSkip=1)






    reconstruction_error /= len(test_data_loader)
    test_loss /= len(test_data_loader)
    min_loss,max_loss=min(test_loss_records),max(test_loss_records)
    model.train()
    return test_loss,min_loss,max_loss,reconstruction_error



    

def train_pipeline():
    # config=load_config("config\\cfgs\\config.yaml")
    config=argParse()
    config["gitInfo"]=get_git_commit_id()
    training_args=config['training']
    # Initialize training parameters from args
    epochs = training_args['epochs']
    device = training_args['device']
    #generate radom seed and record to config
    config['training']['random_seed']=torch.seed()
    run_Name,runTags=runNameTagGenerator(config,tagattach=["theta(t)c(t)","referenceFrame"])


    # Initialize wandb
    if config['wandb']:
        run=wandb.init(project=GLOBAL_WANDB_PROJECT_NAME,
                    name=run_Name,
                    tags=runTags,
                    config=config)
        arti_code=wandb.Artifact("code", type="code")
        arti_code=CollectWandbLogfiles(arti_code)
        wandb.log_artifact(arti_code) 

                
    logging.info(config)
    logging.info(f"run name: {run_Name}, run tags: {runTags}")
    config['run_name']=run_Name

    #build data loader using the dataset and args
    #don't forget there is a built-in "force positive" transformation defined in the datset load function
    train_VastisDataset=buildDataset(config["dataset"],mode="train")   
    config["dataset"].update(train_VastisDataset.dastasetMetaInfo)  
    xdim, ydim,timesteps= config["dataset"]["Xdim"],config["dataset"]["Ydim"],config["dataset"]["time_steps"]
    validationDataset=buildDataset(config["dataset"],mode="val")   
    
    data_loader = torch.utils.data.DataLoader(train_VastisDataset, batch_size=training_args['batch_size'], shuffle=training_args['shuffle'], num_workers=training_args['num_workers'], pin_memory=training_args['pin_memory'])
    validation_data_loader= torch.utils.data.DataLoader(validationDataset, batch_size=training_args['batch_size'], shuffle=False)
    
    
 

    # Initialize the model
    model = ReferenceFrameExtractor(2,  xdim, ydim, timesteps, ouptputDim=3*timesteps, hiddenSize=config['network']['hidden_size'])
    model.apply(init_weights)
    model.to(device)
    torchsummary.summary(model, (2, xdim, ydim, timesteps))

    # Initialize the optimizer with L2 regularization
    optimizer = optim.Adam(model.parameters(), lr= training_args['learning_rate'], weight_decay=training_args['weight_decay'])
    best_val_loss = float('inf')

    # Training
    total_iterations = 0
    oom_time=0
    for epoch in range(epochs):
        epoch_loss = 0
        try:
            for batch_idx, (vectorFieldImage, labelQtct, labelVortex) in enumerate(data_loader):

                # tx, ty, n, rc = labelVortex[0], labelVortex[1], labelVortex[2], labelVortex[3]
                #vectorFieldImage'shape is [bs,chanel=2,W, H, depth(timsteps)]
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
            logging.info(f'Epoch {epoch+1}, epoch_Loss: {epoch_loss}, Val Loss: {val_loss}, Val Rec Error: {val_rec_error}')
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
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                oom_time += 1
                logging.info("WARNING: ran out of memory,times: {}".format(oom_time))
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                bs= config['training']['batch_size']=2
            else:
                logging.error(str(exception))
                raise exception

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
    config = load_config("config\\cfgs\\config.yaml")
    device = config['training']['device']
    test_dataset=buildDataset(config["dataset"],mode="test")   
    config["dataset"].update(test_dataset.dastasetMetaInfo)  
    config['run_name']=checkpoint_path.split(".pth.tar")[0]

    xdim, ydim,timesteps= config["dataset"]["Xdim"],config["dataset"]["Ydim"],config["dataset"]["time_steps"]
    model = ReferenceFrameExtractor(2,  xdim, ydim, timesteps, ouptputDim=6*timesteps, hiddenSize=config['network']['hidden_size'])
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Load the last checkpoint
    load_checkpoint(torch.load(checkpoint_path), model, optimizer)

    model.eval()
    test_model(model,config,test_dataset)













# if __name__ == '__main__':
#    test("models\\CNN_200_0.001_20240629_013617_seed_623621614793900\\20240629_153130_best_checkpoint.pth.tar")

if __name__ == '__main__':
    train_pipeline()

  
# if __name__ == "__main__":
#     # Define the vector field dimensions and boundaries
#     Xdim, Ydim = 20, 20
#     from FLowUtils.AnalyticalFlowCreator import rotation_four_center
#     vecfield=rotation_four_center((64,64),16)
#     # LicRenderingUnsteadyCpp(vecfield,licImageSize=800,timeStepSKip=10,saveFolder="./testOutput",saveName=f"test__rfc0.000001",stepSize=0.000001,MaxIntegrationSteps=128)
#     # LicRenderingUnsteadyCpp(vecfield,licImageSize=800,timeStepSKip=10,saveFolder="./testOutput",saveName=f"test__rfc0.0005",stepSize=0.0005,MaxIntegrationSteps=128)
#     # LicRenderingUnsteadyCpp(vecfield,licImageSize=800,timeStepSKip=10,saveFolder="./testOutput",saveName=f"test__rfc0.005",stepSize=0.005,MaxIntegrationSteps=128)
#     # LicRenderingUnsteadyCpp(vecfield,licImageSize=800,timeStepSKip=10,saveFolder="./testOutput",saveName=f"test__rfc0.01",stepSize=0.01,MaxIntegrationSteps=128)
#     # LicRenderingUnsteadyCpp(vecfield,licImageSize=800,timeStepSKip=10,saveFolder="./testOutput",saveName=f"test__rfc0.1",stepSize=0.1,MaxIntegrationSteps=128)
#     # LicRenderingUnsteadyCpp(vecfield,licImageSize=800,timeStepSKip=10,saveFolder="./testOutput",saveName=f"test__rfc0.2",stepSize=0.2,MaxIntegrationSteps=128)


#     # LicRenderingUnsteadyCpp(vecfield,licImageSize=800,timeStepSKip=10,saveFolder="./testOutput",saveName=f"test__rfc256",stepSize=0.005,MaxIntegrationSteps=256)
#     # LicRenderingUnsteadyCpp(vecfield,licImageSize=800,timeStepSKip=10,saveFolder="./testOutput",saveName=f"test__rfc512",stepSize=0.005,MaxIntegrationSteps=32)
#     # glyphsRenderUnsteadyField(vecfield,ImageSize=800,timeStepSKip=10,saveFolder="./testOutput",saveName=f"test__rfc_glyph")
#     # LicRenderingUnsteadyCpp(vecfield,licImageSize=800,timeStepSKip=10,saveFolder="./testOutput",saveName=f"test__rfc",stepSize=0.0005,MaxIntegrationSteps=128)
#     LicGlyphMixRenderingUnsteady(vecfield,licImageSize=800,timeStepSKip=10,saveFolder="./testOutput",saveName=f"test__rfc_licglyph",stepSize=0.0005,MaxIntegrationSteps=128)




 

