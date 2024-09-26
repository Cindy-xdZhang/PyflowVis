import torch
import torch.nn as nn
import torch.nn.functional as F
from ..build import MODELS

#REPRODUCE OF PREVIOUS PAPERS
@MODELS.register_module()
class DeSilvaVortexViz(nn.Module):
    def __init__(self, DataSizeX,DataSizeY,pathlineStep,in_channels=2, out_channels=1, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1,stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1,stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1,stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.FCN=nn.Sequential(
             nn.Linear(pathlineStep,64),
             nn.BatchNorm1d(64),
             nn.ReLU(),
             nn.Linear(64,64),
             nn.BatchNorm1d(64),
             nn.ReLU(),
        )
        # Calculate the size of the flattened features after convolutions
        self.flatten_size = 64 * (DataSizeX//8 ) * (DataSizeY//8 )
        self.fc1 = nn.Linear(self.flatten_size+64,128)
        self.outPutLayer =nn.Sequential(
            nn.Linear(128, out_channels),
            nn.Sigmoid()                                  
            )

    def forward(self, data):
        binaryImage,informationVector=data
        # Convolutional layers for binary iamgebranch
        x = self.relu(self.bn1 (self.conv1(binaryImage)))
        x = self.relu(self.bn2 (self.conv2(x)))
        x = self.relu(self.bn2 (self.conv3(x)))
        # Flatten the output
        PathlineBinaryhImageFeature = x.view(-1, self.flatten_size)
        
        #expect shape of informationVector [B,L(pathline steps),C=1(pathline point Cumulative Absolute Curl)]
        #->[B,64]
        inforFeature=self.FCN(informationVector)
        concatFeature=torch.concat([PathlineBinaryhImageFeature,inforFeature],dim=-1)
        
        
        # Fully connected layers
        x = self.relu(self.fc1(concatFeature))
        x = self.outPutLayer(x)
        return x
    