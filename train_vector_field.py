import torch
import torch.nn.functional as F
import torch.optim as optim
from DeepUtils.MiscFunctions import *
import logging,random,wandb
import torchsummary

from DeepUtils.models import build_model_from_cfg
from DeepUtils.optim import build_optimizer_from_cfg
from DeepUtils.dataset import build_dataloader_from_cfg,getDatasetRootaMeta
from DeepUtils.scheduler import build_scheduler_from_cfg
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

    



def validate(model, val_data_loader, device) -> float:
    model.eval()
    val_loss = 0
    # val_rec_loss = 0
    with torch.no_grad():
        for batch_idx, (vectorFieldImage, label) in enumerate(val_data_loader):
                vectorFieldImage,label = vectorFieldImage.to(device), label.to(device)
                predictition= model(vectorFieldImage)                
                loss=model.get_loss(predictition,label)
                val_loss += loss.item()
    val_loss /= len(val_data_loader)
    model.train()
    return val_loss

def test_spiral_rotating_field(model,device) -> None:
    from FLowUtils.LicRenderer import LicRenderingUnsteadyCpp
    from FLowUtils.GlyphRenderer import glyphsRenderUnsteadyField
    from FLowUtils.VectorField2d import UnsteadyVectorField2D
    def read_binary_file(filepath, dtype=np.float32) -> np.ndarray:
        with open(filepath, 'rb') as file:
            data = np.fromfile(file, dtype=dtype)
            if dtype == np.float32:
                data=data[2:]
            elif dtype == np.float64:
                data=data[1:]        
        return data
    def testOneSample(raw_data_file,correctlabel):
        raw_Binary = read_binary_file(raw_data_file).reshape(5,16,16, 2)
        raw_Binary=raw_Binary
        name=raw_data_file.split("\\")[-1]
        # UnsteadyField=  UnsteadyVectorField2D(16,16,5,[-2,-2],[2,2],np.pi * 0.25)
        # UnsteadyField.field=raw_Binary
        # glyphsRenderUnsteadyField(UnsteadyField,800,timeStepSKip=1,saveFolder="./testOutput",saveName=f"glyph__{name}")
        # LicRenderingUnsteadyCpp(UnsteadyField,800,timeStepSKip=1,saveFolder="./testOutput",saveName=f"lic__{name}")
        minV= -3.8220109939575197
        maxV= 3.5120744705200197
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
        model.train()
    testOneSample("CppProjects\\data\\rotatingZeroField\\sample_0saddle.bin",0)
    testOneSample("CppProjects\\data\\rotatingZeroField\\sample_6center_ccw.bin",0)
    testOneSample("CppProjects\\data\\rotatingZeroField\\sample_1NotZeroFieldSaddlemeta.bin",0)
    testOneSample("CppProjects\\data\\rotatingZeroField\\sample_1977center_cw.bin",1)



def test_model(model,cfg):
    device = cfg['device']
    test_loss = 0
    test_loss_records=[]
    save_folder=f"./testOutput/{cfg['run_name']}/"
    test_data_loader = build_dataloader_from_cfg(cfg.batch_size,
                                        cfg.dataset,
                                        cfg.dataloader,
                                        datatransforms_cfg=cfg.datatransforms,
                                        split='test'                                             
                                        )
    print(f"length of test dataset: {len(test_data_loader.dataset)}")
    model.eval()
    with torch.no_grad():
        correct=0
        for batch_idx, (vectorFieldImage, label) in enumerate(test_data_loader):
            vectorFieldImage,label = vectorFieldImage.to(device), label.to(device)
            predictition= model(vectorFieldImage)                
            loss=model.get_loss(predictition,label)
            test_loss += loss.item()
            test_loss_records.append(loss.item())

            predicted_classes = torch.argmax(predictition, dim=1)
            true_classes = torch.argmax(label, dim=1)
            # Compare and count the number of correct predictions
            correct += (predicted_classes == true_classes).sum().item()

        precision=float(correct)/float(len(test_data_loader.dataset)-1)
        logging.info(f"correctly predict {correct} out of {len(test_data_loader.dataset)-1}, precision={precision*100}%.")

        #random select  samples to visualize
        for i in range(20):
            sample=random.randint(0,len(test_data_loader.dataset)-1)
            vectorFieldImage, labelVortex=test_data_loader.dataset[sample]
            vectorFieldImage = vectorFieldImage.unsqueeze(0).to(device)
            predictition= model(vectorFieldImage)
            predictition=predictition[0].cpu().numpy()
            logging.info(f"testSample{sample}_predict  {predictition}, vs label { labelVortex}")
 


     
    test_loss /= len(test_data_loader)
    min_loss,max_loss=min(test_loss_records),max(test_loss_records)
    model.train()
    logging.info(f'Avg test loss: {test_loss}, min test loss: {min_loss}, max test loss: {max_loss}')
    return test_loss,min_loss,max_loss


#MAKE THIS FUNCIION INDENPENT OF THE MODEL
def train_model(model, data_loader, validation_data_loader, optimizer,scheduler,config):
    # Initialize training parameters from args
    epochs,device,valiate_every,run_Name=config['epochs'],config['device'],config['val_freq'],config['run_name']
    total_iterations,oom_time,best_val_loss=0,0,float('inf')
    model.to(device)
    for epoch in range(epochs):
        epoch_loss = 0
        try:
            for batch_idx, (vectorFieldImage, label) in enumerate(data_loader):
                vectorFieldImage,label = vectorFieldImage.to(device), label.to(device)
                predictition= model(vectorFieldImage)                
                loss=model.get_loss(predictition,label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                total_iterations += 1
                if batch_idx % config['print_freq'] == 0:
                    lr=optimizer.param_groups[0]["lr"]
                    logging.info(f'Epoch {epoch+1}, iter {batch_idx+1}, total_iterations: {total_iterations}. Loss: {loss.item()},lr:{lr}.')
                    if config['wandb']:
                        wandb.log({"train_loss": loss.item(),  "total_iterations": total_iterations})
                
            if scheduler is not None:
                scheduler.step(epoch)               
            epoch_loss /= len(data_loader)


            # Validate the model
            if (epoch+1) % valiate_every == 0:
                val_loss = validate(model, validation_data_loader, device)  
                logging.info(f'Epoch {epoch+1}, epoch_Loss: {epoch_loss}, Val Loss: {val_loss}')
                if config['wandb']:
                    wandb.log({"epoch": epoch+1,  "epoch_Loss": epoch_loss, "val_loss": val_loss})
                #save best model 
                if val_loss < best_val_loss and config['save_model'] :
                    best_val_loss = val_loss
                    saving_path= os.path.join(config['save_model_path'],run_Name) 
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }, folder_name=saving_path, checkpoint_name= f'best_checkpoint.pth.tar')
                #periodically save model
                if  config['save_model'] and epoch % config["save_freq"]== 0 and epoch>0 and config["save_freq"]>0:
                    saving_path= os.path.join(config['save_model_path'],run_Name) 
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
    try:
        cfg=argParseAndPrepareConfig()
        cfg["gitInfo"]=get_git_commit_id()
        run_Name,runTags=runNameTagGenerator(cfg)
        cfg['run_name']=run_Name
        # Initialize wandb
        if cfg['wandb']:
            run=wandb.init(project=GLOBAL_WANDB_PROJECT_NAME,
                        name=run_Name,
                        tags=runTags,
                        config=cfg)
            arti_code=wandb.Artifact("code", type="code")
            arti_code=CollectWandbLogfiles(arti_code)
            wandb.log_artifact(arti_code) 
        print_args(cfg)
        logging.info(f"run name: {run_Name}, run tags: {runTags}")

        model = build_model_from_cfg(cfg.model)
        model.to(cfg['device'])
        # build dataset        
        rootInfo=getDatasetRootaMeta(cfg.dataset['data_dir'])
        if "unsteadyFieldTimeStep" not in rootInfo:            
            torchsummary.summary(model, (2, rootInfo["Xdim"], rootInfo["Ydim"]))
        else:
            torchsummary.summary(model, (2, rootInfo["Xdim"], rootInfo["Ydim"],rootInfo["unsteadyFieldTimeStep"]))


        if isinstance(cfg.datatransforms['kwargs'], dict):   
            cfg.datatransforms['kwargs'].update(rootInfo) 
        else: 
            cfg.datatransforms['kwargs']= rootInfo

        train_loader = build_dataloader_from_cfg(cfg.batch_size,
                                                cfg.dataset,
                                                cfg.dataloader,
                                                datatransforms_cfg=cfg.datatransforms,
                                                split='train'                                             
                                                )
        print(f"length of training dataset: {len(train_loader.dataset)}")

        val_loader = build_dataloader_from_cfg(cfg.batch_size,
                                                cfg.dataset,
                                                cfg.dataloader,
                                                datatransforms_cfg=cfg.datatransforms,
                                                split='val'                                             
                                                )
        print(f"length of val dataset: {len(val_loader.dataset)}")

      
       # optimizer & scheduler
        optimizer = build_optimizer_from_cfg(model, lr=cfg.lr, **cfg.optimizer)
        scheduler = build_scheduler_from_cfg(cfg, optimizer) if "scheduler" in cfg else None

        # Training
        best_val_loss=train_model(model, train_loader, val_loader, optimizer,scheduler,cfg)
        #final test 
        avgtest_loss,mintest_error,maxtest_error=test_model(model,cfg)

        test_spiral_rotating_field(model,device=cfg['device'])

        if cfg['wandb']:
            wandb.summary.update({"best_val_loss": best_val_loss})
            wandb.summary.update({"avg_test_loss": avgtest_loss})
            wandb.summary.update({"min_test_error": mintest_error})
            wandb.summary.update({"max_test_error": maxtest_error})
            wandb.finish()

    except Exception as e:
        print(e)
        print("Error loading config file")
        return
   







def test_pipeline():
    model_path="models/bs_512_ep_100_lr_0.0005_20240826_152302_seed_00041698924902/best_checkpoint.pth.tar"
    cfg=argParseAndPrepareConfig()
    model = build_model_from_cfg(cfg.model)
    checkpoint=torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(cfg['device'])
    test_spiral_rotating_field(model,device=cfg['device'])





if __name__ == '__main__':
    train_pipeline()
    # test_pipeline()

# if __name__ == '__main__':
#    test("models\\CNN_200_0.001_20240818_154340_seed_4993847091818000\\epoch_101.pth.tar")

# if __name__ == '__main__':
#     testCppGeneratedData()




# def testCppGeneratedData(testDataset=None):
#     """
#     test cpp generated flow data is correctly loaded by the torch dataset 
#     """
#     from FLowUtils.LicRenderer import *
#     from FLowUtils.VectorField2d import UnsteadyVectorField2D,SteadyVectorField2D
#     from FLowUtils.GlyphRenderer import *
#     config=argParse()
#     if testDataset is None:
#         testDataset=buildDataset(config["dataset"],mode="test")
#     bs=1
#     minv,maxv=testDataset.dastasetMetaInfo["minV"],testDataset.dastasetMetaInfo["maxV"]
#     with torch.no_grad():
#         for i in range(10):
#             sample=i
#             loadField, _, labelVortex=testDataset[sample]
#             loadFieldDATA=loadField*(maxv-minv)+minv
#             loadFieldDATA=loadFieldDATA.transpose(0,3)
#             UnsteadyField=  UnsteadyVectorField2D(16,16,5,[-2,-2],[2,2],np.pi * 0.25)
#             UnsteadyField.field=loadFieldDATA
#             name=testDataset.getSampleName(sample)
#             LicRenderingUnsteadyCpp(UnsteadyField,licImageSize=256,timeStepSKip=1,saveFolder="./testOutput",saveName=f"test__{name}",stepSize=0.005,MaxIntegrationSteps=256)
   

