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
#! todo: DEFINE THE cnn MODEL:   NET0: VORETXNET, NET1: VORETXsegNET, NET3: RESNET , net4 vortexViz
#! todo: what is the label? ->second stage the segmentation of as steady as possible (asap) field.
#! todo: visualize the model's output: classification of vortex bondary+ coreline  in 3d space (2d space+ 1D time)
#! todo:test the model's with RFC, bossineq, helix, spriral motion
#? - [ ]  TODO: draw pathline?
#? - [ ]  TODO: train vector field for killing +regular+ Di term 
#? - [ ]  TODO: curvature gradient
#? - [ ]  TODO: implement hanger of imgui widget

def runNameTagGenerator(config) ->(str,list[:str]):
    seed=config['training']['random_seed']
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    runName = f"{config['network']['family_name']}_{config['training']['epochs']}_{config['training']['learning_rate']}_{current_time}_seed_{seed}"
    
    tagGen0=config['dataset']['root'].split("\\")[-1]
    tagGen1=config['network']['family_name']
    runTags= [tagGen0,tagGen1]
    return runName,runTags
    

def validate(model, data_loader, device) -> float:
    model.eval()
    val_loss = 0
    # val_rec_loss = 0
    with torch.no_grad():
         for batch_idx, (vectorFieldImage, labelQtct, labelVortex) in enumerate(data_loader):
            vectorFieldImage, labelQtct = vectorFieldImage.to(device), labelQtct.to(device)
            predictQtCt= model(vectorFieldImage)
            loss = F.mse_loss(predictQtCt, labelQtct)
            val_loss += loss.item()
            # val_rec_loss += lossRec.item()

    val_loss /= len(data_loader)
    # val_rec_loss /= len(data_loader)
    model.train()
    return val_loss


def test_model(model,config,testDataset=None):
    device = config['training']['device']
    if testDataset is None:
        testDataset=buildDataset(config["dataset"],mode="test")

    test_data_loader= torch.utils.data.DataLoader(testDataset, batch_size=config['training']['batch_size'], shuffle=False)
    model.eval()
    test_loss = 0
    test_loss_records=[]
    save_folder=f"./testOutput/{config['run_name']}/"
    minv,maxv=testDataset.dastasetMetaInfo["minV"],testDataset.dastasetMetaInfo["maxV"]
    with torch.no_grad():
        for batch_idx, (vectorFieldImage, labelReferenceF, labelVortex) in enumerate(test_data_loader):
            vectorFieldImage, labelReferenceF = vectorFieldImage.to(device), labelReferenceF.to(device)
            predictReferenceF= model(vectorFieldImage)
            loss = F.mse_loss(predictReferenceF, labelReferenceF)
            test_loss += loss.item()
            test_loss_records.append(loss.item())

        #random select  samples to visualize
        for i in range(5):
            sample=random.randint(0,len(testDataset)-1)
            vectorFieldImage, labelReferenceF, labelVortex=testDataset[sample]
            vectorFieldImage = vectorFieldImage.unsqueeze(0).to(device)
            predictReferenceF= model(vectorFieldImage)
            predictReferenceF=predictReferenceF[0].cpu().numpy()
            logging.info(f"testSample{sample}_predict ReferenceFrame {predictReferenceF}, vs label Ref{  labelReferenceF}__rec")
     
    test_loss /= len(test_data_loader)
    min_loss,max_loss=min(test_loss_records),max(test_loss_records)
    model.train()
    logging.info(f'Avg test loss: {test_loss}, min test loss: {min_loss}, max test loss: {max_loss}')
    return test_loss,min_loss,max_loss


#MAKE THIS FUNCIION INDENPENT OF THE MODEL
def train_model(model, data_loader, validation_data_loader, optimizer,config):
    # Initialize training parameters from args
    training_args=config['training']
    epochs = training_args['epochs']
    device = training_args['device']
    valiate_every=training_args['validateion_frequency']
    run_Name=config['run_name']
    model.to(device)
    total_iterations = 0
    oom_time=0
    best_val_loss = float('inf')
    for epoch in range(epochs):
        epoch_loss = 0
        try:
            for batch_idx, (vectorFieldImage, labelQtct, labelVortex) in enumerate(data_loader):
                vectorFieldImage, labelQtct = vectorFieldImage.to(device), labelQtct.to(device)
                predictQtCt= model(vectorFieldImage)
                loss = F.mse_loss(predictQtCt, labelQtct)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                total_iterations += 1
                if batch_idx % training_args['print_frequency'] == 0:
                    logging.info(f'Epoch {epoch+1}, iter {batch_idx+1}, total_iterations: {total_iterations}. Loss: {loss.item()}.')
                    if config['wandb']:
                        wandb.log({"train_loss": loss.item(),  "total_iterations": total_iterations})
                
            epoch_loss /= len(data_loader)


            # Validate the model
            if (epoch+1) % valiate_every == 0:
                val_loss = validate(model, validation_data_loader, device)  
                logging.info(f'Epoch {epoch+1}, epoch_Loss: {epoch_loss}, Val Loss: {val_loss}')
                if config['wandb']:
                    wandb.log({"epoch": epoch+1,  "epoch_Loss": epoch_loss, "val_loss": val_loss})
                #save best model 
                if val_loss < best_val_loss and training_args['save_model'] :
                    best_val_loss = val_loss
                    saving_path= os.path.join(training_args['save_model_path'],run_Name) 
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, folder_name=saving_path, checkpoint_name= f'best_checkpoint.pth.tar')
                #periodically save model
                if  training_args['save_model'] and epoch % training_args["save_model_frequency"]== 0 and epoch>0:
                    saving_path= os.path.join(training_args['save_model_path'],run_Name) 
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, folder_name=saving_path, checkpoint_name= f'epoch_{epoch+ 1}.pth.tar')
            else:
                logging.info(f'Epoch {epoch+1}, epoch_Loss: {epoch_loss}')
                if config['wandb']:
                    wandb.log({"epoch": epoch+1,  "epoch_Loss": {epoch_loss}})


        except RuntimeError as exception:
            if "out of memory" in str(exception):
                oom_time += 1
                logging.info("WARNING: ran out of memory,times: {}".format(oom_time))
                if oom_time > 3:
                    logging.info("WARNING: ran out of memory 3 times, stopping training")
                    break
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                logging.error(str(exception))
                raise exception
            
    return best_val_loss


def train_pipeline():
    config=argParse()
    config['training']['random_seed']=torch.seed()
    config["gitInfo"]=get_git_commit_id()
    run_Name,runTags=runNameTagGenerator(config)
    config['run_name']=run_Name
    # Initialize wandb
    if config['wandb']:
        run=wandb.init(project=GLOBAL_WANDB_PROJECT_NAME,
                    name=run_Name,
                    tags=runTags,
                    config=config)
        arti_code=wandb.Artifact("code", type="code")
        arti_code=CollectWandbLogfiles(arti_code)
        wandb.log_artifact(arti_code) 
    logging.info(config,f"run name: {run_Name}, run tags: {runTags}")



    #build data loader using the dataset and args
    training_args=config['training']
    train_VastisDataset=buildDataset(config["dataset"],mode="train")   
    config["dataset"].update(train_VastisDataset.dastasetMetaInfo)  
    xdim, ydim,timesteps= config["dataset"]["Xdim"],config["dataset"]["Ydim"],config["dataset"]["unsteadyFieldTimeStep"]
    validationDataset=buildDataset(config["dataset"],mode="validation")       
    data_loader = torch.utils.data.DataLoader(train_VastisDataset, batch_size=training_args['batch_size'], shuffle=training_args['shuffle'], num_workers=training_args['num_workers'], pin_memory=training_args['pin_memory'])
    validation_data_loader= torch.utils.data.DataLoader(validationDataset, batch_size=training_args['batch_size'], shuffle=False)
    


    # Initialize the model
    model = TobiasReferenceFrameCNN(2,  xdim, ydim, timesteps, ouptputDim=6, hiddenSize=config['network']['hidden_size'],dropout=config['network']['dropout'])
    model.apply(init_weights)  
    model.to(training_args['device'])
    torchsummary.summary(model, (2, xdim, ydim, timesteps))

    # Initialize the optimizer with L2 regularization
    optimizer = optim.Adam(model.parameters(), lr= training_args['learning_rate'], weight_decay=training_args['weight_decay'])

    # Training
    best_val_loss=train_model(model, data_loader, validation_data_loader, optimizer,config)
    #final test 
    avgtest_loss,mintest_error,maxtest_error=test_model(model,config=config)
    if config['wandb']:
        wandb.summary.update({"best_val_loss": best_val_loss})
        wandb.summary.update({"avg_test_loss": avgtest_loss})
        wandb.summary.update({"min_test_error": mintest_error})
        wandb.summary.update({"max_test_error": maxtest_error})
        wandb.finish()





def test(checkpoint_path):
    config = load_config("config\\cfgs\\config.yaml")
    device = config['training']['device']
    test_dataset=buildDataset(config["dataset"],mode="test")   
    config["dataset"].update(test_dataset.dastasetMetaInfo)  
    config['run_name']=checkpoint_path.split(".pth.tar")[0]

    xdim, ydim,timesteps= config["dataset"]["Xdim"],config["dataset"]["Ydim"],config["dataset"]["unsteadyFieldTimeStep"]
    model = TobiasReferenceFrameCNN(2,  xdim, ydim, timesteps, ouptputDim=6, hiddenSize=config['network']['hidden_size'])
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Load the last checkpoint
    load_checkpoint(torch.load(checkpoint_path), model, optimizer)
    test_model(model,config,test_dataset)












if __name__ == '__main__':
    train_pipeline()

# if __name__ == '__main__':
#    test("models\\CNN_200_0.001_20240818_154340_seed_4993847091818000\\epoch_101.pth.tar")

# if __name__ == '__main__':
#     testCppGeneratedData()




def testCppGeneratedData(testDataset=None):
    """
    test cpp generated flow data is correctly loaded by the torch dataset 
    """
    config=argParse()
    if testDataset is None:
        testDataset=buildDataset(config["dataset"],mode="test")
    bs=1
    minv,maxv=testDataset.dastasetMetaInfo["minV"],testDataset.dastasetMetaInfo["maxV"]
    with torch.no_grad():
        for i in range(10):
            sample=i
            loadField, _, labelVortex=testDataset[sample]
            loadFieldDATA=loadField*(maxv-minv)+minv
            loadFieldDATA=loadFieldDATA.transpose(0,3)
            UnsteadyField=  UnsteadyVectorField2D(16,16,5,[-2,-2],[2,2],np.pi * 0.25)
            UnsteadyField.field=loadFieldDATA
            name=testDataset.getSampleName(sample)
            LicRenderingUnsteadyCpp(UnsteadyField,licImageSize=256,timeStepSKip=1,saveFolder="./testOutput",saveName=f"test__{name}",stepSize=0.005,MaxIntegrationSteps=256)
   

