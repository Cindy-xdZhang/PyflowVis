import torch
from datetime import datetime
import os
import logging
import argparse
import yaml
def CollectWandbLogfiles(arti_code):
    def match_file(path):    
        return path.endswith(".py") 
    saveFolder="./DeepUtils/"
    fileList0=[os.path.join( saveFolder,f)  for f in os.listdir(saveFolder) if match_file(f)]
    saveFolder="./FlowUtils/"
    fileList1=[os.path.join( saveFolder,f)  for f in os.listdir(saveFolder) if match_file(f)]    
    fileList0.append("train_vector_field.py")

    for file in fileList0:
        arti_code.add_file(file, name= file)
    for file in fileList1:
        arti_code.add_file(file, name=file)
    return arti_code


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
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Construct the file path
    file_path = os.path.join(folder_name, f"{current_time}_{checkpoint_name}")

    # Save the checkpoint
    print(f"=> Saving checkpoint to {file_path}")
    torch.save(state, file_path)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


# Function to parse command line arguments and update config
def argParse():
    parser = argparse.ArgumentParser(description="Train pipeline parameters")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--device", type=str, help="Device to be used for training (e.g., 'cpu', 'cuda')")
    parser.add_argument("--batch_size", type=int, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, help="Learning rate for the optimizer")
    parser.add_argument("--wandb", action='store_true', help="Enable wandb logging")
    parser.add_argument("--num_workers", type=int, help="Number of data loading workers")
    parser.add_argument("--pin_memory", action='store_true', help="Pin memory for data loading")

    args = parser.parse_args()
    config = load_config(args.config)

    # Update config with command line arguments
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.device is not None:
        config['training']['device'] = args.device
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        config['training']['learning_rate'] = args.learning_rate
    if args.wandb:
        config['wandb'] = True
    if args.num_workers is not None:
        config['training']['num_workers'] = args.num_workers
    if args.pin_memory:
        config['training']['pin_memory'] = True

    return config