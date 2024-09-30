import torch
import torch.nn.functional as F
import torch.optim as optim
from DeepUtils.MiscFunctions import *
import logging,random,wandb
# import torchsummary

from DeepUtils.models import build_model_from_cfg
from DeepUtils.optim import build_optimizer_from_cfg
from DeepUtils.dataset import build_dataloader_from_cfg
from DeepUtils.scheduler import build_scheduler_from_cfg
from test import test_model

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
         for batch_idx, (data, label) in enumerate(val_data_loader):
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
                val_loss += loss.item()
    val_loss /= len(val_data_loader)
    model.train()
    return val_loss



#MAKE THIS FUNCIION INDENPENT OF THE MODEL
def train_model(model, data_loader, validation_data_loader, optimizer,scheduler,config):
    # Initialize training parameters from args
    epochs,device,valiate_every,run_Name,lossName=config['epochs'],config['device'],config['val_freq'],config['run_name'],config["model"]["criterion_args"]["NAME"]
    total_iterations,oom_time,best_val_loss=0,0,float('inf')
    model.to(device)
    sched_on_epoch=getattr(config,"sched_on_epoch",True)
    accumulation_steps=getattr(config,'gradient_accumulate', 1) 
    max_grad_norm = getattr(config,'gradient_norm_clip', 1.0)   # Maximum norm for gradient clipping

    for epoch in range(epochs):
        epoch_loss = 0
        running_accumulate_loss=0
        try:
            for batch_idx, (data, label) in enumerate(data_loader):
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
                loss = loss / accumulation_steps  # Normalize the loss
                loss.backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)                    
                    optimizer.step()
                    optimizer.zero_grad()
                    if scheduler is not None and not sched_on_epoch:
                        scheduler.step(epoch)         

                 
        
                epoch_loss += loss.item()* accumulation_steps
                running_accumulate_loss+=loss.item()* accumulation_steps
                total_iterations += 1
                if batch_idx % config['print_freq'] == 0:
                    lr=optimizer.param_groups[0]["lr"]
                    print_freq=config['print_freq'] if config['print_freq'] >0 and batch_idx>1 else 1
                    running_accumulate_loss/=float(print_freq)
                    logging.info(f'Epoch {epoch+1}, iter {batch_idx+1}, total_iterations: {total_iterations}. {lossName}: {running_accumulate_loss},lr:{lr}.')                    
                    if config['wandb']:
                        wandb.log({"train_loss": running_accumulate_loss,  "total_iterations": total_iterations,"lr":lr})
                    running_accumulate_loss=0
                               
            if scheduler is not None and sched_on_epoch:
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
            if  config['save_model'] and config["save_freq"]>0 and epoch % config["save_freq"]== 0 and epoch>0 :
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
    print("torch.cuda.is_available():",torch.cuda.is_available())  # Should return True
    print("torch.version.cuda:",torch.version.cuda)         # Should print the CUDA version PyTorch was built with
    try:
        cfg=argParseAndPrepareConfig()
        cfg["gitInfo"]=get_git_commit_id()
        run_Name,runTags=runNameTagGenerator(cfg)
        cfg['run_name']=run_Name
        logging.info(f"run name: {run_Name}, run tags: {runTags}")
        #get dataset realted config paramters,like PathlineGroupsCount, minV,maxV, etc.
        readDataSetRelatedConfig(cfg)
        model = build_model_from_cfg(cfg.model)
        model.to(cfg['device'])
    
        # if "unsteadyFieldTimeStep" not in rootInfo:            
        #     torchsummary.summary(model, (2, rootInfo["Xdim"], rootInfo["Ydim"]))
        # else:
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

        # Initialize wandb
        if cfg['wandb']:
            run=wandb.init(project=GLOBAL_WANDB_PROJECT_NAME,
                        name=run_Name,
                        tags=runTags,
                        config=cfg)
            arti_code=wandb.Artifact("code", type="code")
            arti_code=CollectFiles4Backup(config=cfg,arti_code=arti_code)
            wandb.log_artifact(arti_code) 
        print_args(cfg)
        
        # Training
        best_val_loss=train_model(model, train_loader, val_loader, optimizer,scheduler,cfg)
        #final test 
        test_result:dict=test_model(model,cfg)

        if cfg['wandb'] and  isinstance(test_result,dict):            
            wandb.summary.update({"best_val_loss": best_val_loss})
            for key, value in test_result.items():
                wandb.summary.update({key: value})
            wandb.finish()

    except Exception as e:
        print(e)
        return
   











if __name__ == '__main__':
    train_pipeline()
    
