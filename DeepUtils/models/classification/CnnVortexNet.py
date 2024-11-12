import torch
import torch.nn as nn
import torch.nn.functional as F
from ..build import MODELS
# Every Network is split into two parts: the encoder and the classification head(predictor);
# predictor could be indentity for simple classification networks, if we put the whole network into "encoder"

#REPRODUCE OF PREVIOUS PAPERS
@MODELS.register_module()
class LiuVortexNet(nn.Module):
    def __init__(self, in_channels, DataSizeX,DataSizeY,out_channels=1, **kwargs):
        super(LiuVortexNet, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        
        # Calculate the size of the flattened features after convolutions
        self.flatten_size = 64 * (DataSizeX ) * (DataSizeY )
        self.fc1 = nn.Linear(self.flatten_size,128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, out_channels)

    def forward(self, x):
        # Convolutional layers with ReLU activation
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        # Flatten the output
        x = x.view(-1, self.flatten_size)
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x)) 
        x = self.relu(self.fc3(x))
        x=F.sigmoid(x).squeeze()
        return x
    






