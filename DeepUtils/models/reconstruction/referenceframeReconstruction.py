import torch
import torch.nn as nn
import torch.nn.functional as F
from ..build import MODELS

@MODELS.register_module()
class TobiasReferenceFrameCNN(nn.Module):
    """ RoboustReferenceFrameCNN is the CNN model from paper: Robust Reference Frame Extraction from Unsteady 2D Vector Fields with Convolutional Neural Networks
    """
    def __init__(self,in_channels, DataSizeX,DataSizeY,TimeSteps,out_channels, hiddenSize=64, dropout=0.1, **kwargs):
        super(TobiasReferenceFrameCNN, self).__init__()
        # the input tensor of Conv3d should be in the shape of[batch_size, chanel=2,W=16, H=16, depth(timsteps)]
        self.conv1_1 = nn.Conv3d(in_channels=in_channels, out_channels=hiddenSize, kernel_size=3, stride=2,padding=1)
        self.bn1_1 = nn.BatchNorm3d(hiddenSize)
        # self.CNNlayerCountN=4-2 #n=log2(max(H,W))-2
        self.conv2_1 = nn.Conv3d(in_channels=hiddenSize, out_channels=hiddenSize*2, kernel_size=3, stride=2,padding=1)
        self.bn2_1 = nn.BatchNorm3d(hiddenSize*2)
        lastCnnOuputChannel=1024


        self.flatten = nn.Flatten()
        DataSizeX = DataSizeX // 4
        DataSizeY = DataSizeY // 4
        TimeSteps = 2#(5+1/2=3,3+1/2=2)
        # Fully connected layer
        self.fc1 = nn.Linear(hiddenSize*2 * DataSizeX * DataSizeY * TimeSteps, 1024)
        self.bn_fc_1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn_fc_2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, out_channels)
        self.dropout_fc = nn.Dropout(dropout)
        self.outputDim=out_channels
        

    def forward(self, x):
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        # x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = self.flatten(x)
        x = self.dropout_fc(F.relu(self.bn_fc_1(self.fc1(x))))
        x = self.dropout_fc(F.relu(self.bn_fc_2(self.fc2(x)))) 
        x =self.fc3(x)
        return x


