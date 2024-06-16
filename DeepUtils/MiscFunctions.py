import torch
from datetime import datetime
import os

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