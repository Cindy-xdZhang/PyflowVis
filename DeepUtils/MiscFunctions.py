import torch
from datetime import datetime
import os
import logging
import argparse
import yaml
import numpy as np
import random
from .utils import EasyConfig
from typing import List, Tuple
from .dataset.SteadyVastisDataset import getDatasetRootaMeta
import shutil

def print_args(args, printer=print):
    printer("==========       args      =============")
    for arg, content in args.items():
        printer("{}:{}".format(arg, content))
    printer("==========     args END    =============")

def set_seed(seed,force_determinsitic=False):
    torch.manual_seed(seed)  # Sets the seed for CPU
    torch.cuda.manual_seed(seed)  # Sets the seed for all GPU devices
    torch.cuda.manual_seed_all(seed)  # If you’re using multiple GPUs
    np.random.seed(seed)
    random.seed(seed)
    if force_determinsitic:
        # Ensures that the CUDA algorithms are deterministic, potentially at the cost of performance
        torch.backends.cudnn.deterministic=True


def runNameTagGenerator(config)->Tuple[str, List[str]]:
    seed=config['random_seed']
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    runName = f"bs_{config['batch_size']}_ep_{config['epochs']}_lr_{config['lr']}_{current_time}_seed_{seed}"
    tagGen0=config['dataset']['NAME']
    tagGen1=config['model']['encoder_args']['NAME']
    runTags= [tagGen0,tagGen1]
    return runName,runTags

def collectPyFilesOfAFolder(saveFolder):
    fileList=[os.path.join( saveFolder,f)  for f in os.listdir(saveFolder) if f.endswith(".py") and "__init__" not in f ]            
    return fileList

def CollectFiles4Backup(config,arti_code):
    if "run_name" in  config:
        run_name=getattr(config,"run_name")         
        fileList=[]    
        saveFolder0=collectPyFilesOfAFolder("./DeepUtils/models/segmentation")
        fileList.extend(saveFolder0)
        configFile=getattr(config,"config_yaml",None) 
        fileList.append("train.py")
        fileList.append(configFile)
        #copy file to backup folder 
        backup_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "./outputModels/backUPFiles/")
        backup_folder= os.path.join(backup_folder,run_name)
        
        print(f"backup Files {fileList} to:{backup_folder}.." )
        # Create backup folder if it doesn't exist
        if not os.path.exists(backup_folder):
            os.makedirs(backup_folder)

        # Copy the collected files to the backup folder
        for file in fileList:
            if file and os.path.exists(file):
                shutil.copy(file, backup_folder)
                print(f"Copied {file} to {backup_folder}")
                
        if config["wandb"]:
            for file in fileList:
                if os.path.isfile(file):
                    arti_code.add_file(file, name= file)      
        return arti_code
    
    return None

def readDataSetRelatedConfig(cfg):
    # some config paramters  need to rewrite by dataset meta file 
    rootInfo=getDatasetRootaMeta(cfg.dataset['data_dir'])
    if isinstance(cfg.datatransforms['kwargs'], dict):   
        cfg.datatransforms['kwargs']["minV"]=rootInfo["minV"]
        cfg.datatransforms['kwargs']["maxV"]=rootInfo["maxV"]
    else: 
        cfg.datatransforms['kwargs']= rootInfo
    PathlineCountK=16
    outputPathlineLength=16
    PathlineFeature=7
    if "outputPathlinesCountK" not in rootInfo:
        logging.warning("outputPathlinesCountK not in self.dastasetMetaInfo,assume 16" )
    else:
        PathlineCountK= rootInfo["outputPathlinesCountK"]
        
    if "PathlineFeature" not in rootInfo:
        logging.warning("PathlineFeature not in self.dastasetMetaInfo,assume 7" )
    else:
        PathlineFeature= rootInfo["PathlineFeature"]
        
    if "outputPathlineLength" not in rootInfo:
        logging.warning("outputPathlineLength  not in self.dastasetMetaInfo,assume 16" )
    else:
        outputPathlineLength= rootInfo["outputPathlineLength"] 
        
    PathlineGroupsCount=int(PathlineCountK/2)*int(PathlineCountK/2)
    
    cfg["model"]["encoder_args"]["PathlineGroups"]=PathlineGroupsCount
    cfg["PathlineFeature"]=PathlineFeature
    cfg["outputPathlineLength"]=outputPathlineLength
    cfg["outputPathlinesCountK"]=PathlineCountK
    if "in_channels" not in cfg["model"]["encoder_args"]:
        cfg["model"]["encoder_args"]["in_channels"]=PathlineFeature
    
        
def load_config(path):
    with open(path, 'r') as file:
        args = yaml.safe_load(file)
    # ==> Device
    num_gpus = torch.cuda.device_count()
    args['num_gpus'] = num_gpus
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    args['device'] = device
     # ==> Logger
    # set_logger(result_dir)
    # logging.info(args)
    # if  args['wandb']==True:
        # wandb init
        # pass
    return args

def initLogging():
    class CustomFormatter(logging.Formatter):
        grey = "\x1b[38;21m"
        yellow = "\x1b[33;21m"
        red = "\x1b[31;21m"
        bold_red = "\x1b[31;1m"
        reset = "\x1b[0m"
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

        FORMATS = {
            logging.DEBUG: grey + format + reset,
            logging.INFO: grey + format + reset,
            logging.WARNING: yellow + format + reset,
            logging.ERROR: red + format + reset,
            logging.CRITICAL: bold_red + format + reset
        }
        def format(self, record):
            log_fmt = self.FORMATS.get(record.levelno)
            formatter = logging.Formatter(log_fmt)
            return formatter.format(record)

    ch = logging.StreamHandler()
    ch.setFormatter(CustomFormatter())
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(ch)



def save_checkpoint(state, folder_name="./", checkpoint_name="checkpoint.pth.tar"):
    # Create the directory if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    # Get the current date and time
    # current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # file_path = os.path.join(folder_name, f"{current_time}_{checkpoint_name}")
    file_path =os.path.join(folder_name, checkpoint_name)

    # Save the checkpoint
    print(f"=> Saving checkpoint to {file_path}")
    torch.save(state, file_path)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

def get_git_commit_id():
    import subprocess
    try:
        commit_id = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
        is_dirty = subprocess.check_output(["git", "diff"]).decode("utf-8")
        dirty_suffix = "-dirty" if is_dirty else ""
        return f"GitCommit-{commit_id}{dirty_suffix}"
    except subprocess.CalledProcessError:
        return "Not a git repository"
    
# Function to parse command line arguments and update config
def argParseAndPrepareConfig():
    parser = argparse.ArgumentParser(description="Train pipeline parameters")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--device", type=str, help="Device to be used for training (e.g., 'cpu', 'cuda')")
    parser.add_argument("--batch_size", type=int, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, help="Learning rate for the optimizer")
    parser.add_argument("--wandb", action='store_true', help="Enable wandb logging")
    parser.add_argument("--num_workers", type=int, help="Number of data loading workers")

    args = parser.parse_args()
    cfg = EasyConfig()
    cfg.load(args.config, recursive=True)
    cfg["config_yaml"]=args.config
    # Update config with command line arguments
    if args.epochs is not None:
        cfg['epochs'] = args.epochs
    if args.device is not None:
        cfg['device'] = args.device
    if args.batch_size is not None:
        cfg['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        cfg['lr'] = args.learning_rate
    if args.wandb:
        cfg['wandb'] = True
    if args.num_workers is not None:
        cfg['dataloader']['num_workers'] = args.num_workers

    if 'random_seed' not in cfg:
        cfg['random_seed']=np.random.randint(0,5000)
        set_seed(cfg['random_seed'])
    else:
        set_seed(cfg['random_seed'],force_determinsitic=True)

    return cfg


