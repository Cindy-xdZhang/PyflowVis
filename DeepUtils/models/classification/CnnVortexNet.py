import torch
import torch.nn as nn
import torch.nn.functional as F
from ..build import MODELS
# Every Network is split into two parts: the encoder and the classification head(predictor);
# predictor could be indentity for simple classification networks, if we put the whole network into "encoder"

#REPRODUCE OF PREVIOUS PAPERS
@MODELS.register_module()
class LiuVortexNet(nn.Module):
    def __init__(self, in_channels, DataSizeX,DataSizeY,out_channels=2, **kwargs):
        super(LiuVortexNet, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # self.bn1 = nn.BatchNorm2d(16)
        # self.bn2 = nn.BatchNorm2d(32)
        # self.bn3 = nn.BatchNorm2d(64)
        # self.bn4 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        
        # Calculate the size of the flattened features after convolutions
        self.flatten_size = 64 * (DataSizeX ) * (DataSizeY )
        self.fc1 = nn.Linear(self.flatten_size,128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, out_channels)

    def forward(self, x):
        # Convolutional layers with ReLU activation
        x = self.relu((self.conv1(x)))
        x = self.relu((self.conv2(x)))
        x = self.relu((self.conv3(x)))
        x = self.relu((self.conv4(x)))
        # Flatten the output
        x = x.view(-1, self.flatten_size)
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x)) 
        x = self.relu(self.fc3(x))
        x=F.softmax(x, dim=1)
        return x
    





@MODELS.register_module()
class TobiasVortexBoundaryCNN(nn.Module):
    """ RoboustReferenceFrameCNN is the CNN model from paper: Robust Reference Frame Extraction from Unsteady 2D Vector Fields with Convolutional Neural Networks
    """
    def __init__(self,in_channels, DataSizeX,DataSizeY,out_channels=2, dropout= 0.005,**kwargs):
        super(TobiasVortexBoundaryCNN, self).__init__()
        # the input tensor of Conv3d should be in the shape of[batch_size, chanel=2,W=16, H=16]
        self.conv1_1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=2,padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2,padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)

        self.flatten = nn.Flatten()
        DataSizeX = DataSizeX // 4
        DataSizeY = DataSizeY // 4
        # Fully connected layer
        self.fc1 = nn.Linear(128 * DataSizeX * DataSizeY , 128)
        self.bn_fc_1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 2)
        self.bn_fc_2 = nn.BatchNorm1d(2)
        self.dropout = nn.Dropout(dropout)

        

    def forward(self, x):
        x = self.dropout( F.relu(self.bn1_1(self.conv1_1(x))))
        x = self.dropout(F.relu(self.bn2_1(self.conv2_1(x))))
        # x = F.relu(self.bn3_1(self.conv3_1(x)))

        x = self.flatten(x)
        x = self.dropout(F.relu(self.bn_fc_1(self.fc1(x))))
        x = F.relu(self.bn_fc_2(self.fc2(x)))
        x= F.softmax(x, dim=1)
        return x
        
