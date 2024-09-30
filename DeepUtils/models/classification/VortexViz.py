import torch
import torch.nn as nn
import torch.nn.functional as F
from ..build import MODELS
from typing import Tuple
#REPRODUCE OF PREVIOUS PAPERS
@MODELS.register_module()
class DeSilvaVortexViz(nn.Module):
    def __init__(self, DataSizeX, **kwargs):
        super().__init__()
        out_channels=1
        in_channels=1
        pathlineStep=1024
        self.DataSizeX =DataSizeX
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1,stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1,stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1,stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.dropout=nn.Dropout(0.2)
        # Output size=floor( (input size-kernel+2*padding) /stride  ) +1
        lineLengthAfterConv1d=((pathlineStep-5+2)//2)+1
        self.convseq1=nn.Conv1d(1,1,kernel_size=5,padding=1,stride=2)
        self.FCN=nn.Sequential(
             nn.Linear(pathlineStep,128),
             nn.BatchNorm1d(128),
             nn.ReLU(),
             nn.Linear(128,256),
             nn.BatchNorm1d(256),
             nn.ReLU(),
        )
        # Calculate the size of the flattened features after convolutions
        # Output size=floor( (input size-kernel+2*padding) /stride  ) +1
        dx1=(DataSizeX-3+2)//2+1
        dx2=(dx1-3+2)//2+1
        dx3=(dx2-3+2)//2+1
        self.flatten_size = 64 * dx3*dx3
        self.fc1 = nn.Linear(self.flatten_size+256,256)
        self.outPutLayer =nn.Sequential(
            nn.Linear(256, out_channels),
            nn.Sigmoid()                                  
            )

    def forward(self, data:torch.Tensor) -> torch.Tensor:
        # binaryImage,informationVector=data
        # Split the merged data into informationVector and binaryImage
        binaryImage ,informationVector =data[:,0:self.DataSizeX*self.DataSizeX], data[:,self.DataSizeX*self.DataSizeX:],
        # Reshape binaryImage to 2D
        binaryImage = binaryImage.view(-1, self.DataSizeX, self.DataSizeX).unsqueeze(1)

        # Convolutional layers for binary iamgebranch
        x = self.relu((self.bn1 (self.conv1(binaryImage))))
        x = self.relu(self.bn2 (self.conv2(x)))
        x = self.relu(self.bn3 (self.conv3(x)))
        # Flatten the output
        PathlineBinaryhImageFeature = x.view(-1, self.flatten_size)
        
        #expect shape of informationVector [B,L(pathline steps),C=1(pathline point Cumulative Absolute Curl)]
        #->[B,64]
        # informationVector=F.relu(self.convseq1(informationVector.unsqueeze(1)))
        inforFeature=self.FCN(informationVector.squeeze(1))
        concatFeature=torch.concat([PathlineBinaryhImageFeature,inforFeature],dim=-1)
        
        
        # Fully connected layers
        x = self.relu(self.fc1(concatFeature))
        x = self.outPutLayer(x).squeeze()
        return x
    