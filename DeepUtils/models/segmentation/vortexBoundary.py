import torch
import torch.nn as nn
import torch.nn.functional as F
from ..build import MODELS

@MODELS.register_module()
class TobiasVortexBoundaryCNN(nn.Module):

    def __init__(self,in_channels, DataSizeX,DataSizeY,out_channels=1, dropout= 0.005,**kwargs):
        super(TobiasVortexBoundaryCNN, self).__init__()
        # the input tensor of Conv3d should be in the shape of[batch_size, chanel=2,W=16, H=16]
        self.conv1_1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=2,padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2,padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)

        self.flatten = nn.Flatten()
        
        featureDataSizeX = DataSizeX // 4
        featureDataSizeY= DataSizeY // 4
        # Fully connected layer
        self.fc1 = nn.Linear(128 *featureDataSizeX *featureDataSizeY , 128)
        self.bn_fc_1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, DataSizeX*DataSizeY)
        self.bn_fc_2 = nn.BatchNorm1d(DataSizeX*DataSizeY)
        self.dropout = nn.Dropout(dropout)

        

    def forward(self, x):
        B,Xdim,Ydim,C=x.shape
        x = self.dropout( F.relu(self.bn1_1(self.conv1_1(x))))
        x = self.dropout(F.relu(self.bn2_1(self.conv2_1(x))))
        # x = F.relu(self.bn3_1(self.conv3_1(x)))

        x = self.flatten(x)
        x = self.dropout(F.relu(self.bn_fc_1(self.fc1(x))))
        x = F.relu(self.bn_fc_2(self.fc2(x)))
        x=x.reshape(B,Xdim,Ydim)
        x= F.sigmoid(x)
        return x
        
        

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels,dropout=0.0):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.dropout=nn.Dropout(dropout)

    def forward(self, x):
        return  self.dropout(self.conv(x)) 

@MODELS.register_module()
class TobiasVortexBoundaryUnet(nn.Module):
    def __init__(self,  in_channels=2,  features=64, dropout= 0.01,**kwargs):
        super(TobiasVortexBoundaryUnet, self).__init__()
        self.n = 3
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.out_channels=1
        # Down part of U-Net
        in_features = in_channels
        for _ in range( self.n ):
            self.downs.append(DoubleConv(in_features, features,dropout))
            in_features = features
            features *= 2

        # Bottom part of U-Net
        self.bottleneck = DoubleConv(features // 2, features,dropout)

        # Up part of U-Net
        for _ in range( self.n ):
            self.ups.append(
                nn.ConvTranspose2d(features, features // 2, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(features, features // 2,dropout))
            features //= 2
        
        self.final_conv = nn.Conv2d(features, self.out_channels, kernel_size=1)
 

    def forward(self, x:torch.Tensor):
        skip_connections = []

        # Downsampling
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # Reverse the list
        # Upsampling
        # for idx in range(0,   self.n *2, 2):
        #     x = self.ups[idx](x)
        #     skip_connection = skip_connections[idx // 2]

        #     if x.shape != skip_connection.shape:
        #         x = nn.functional.resize(x, size=skip_connection.shape[2:])

        #     concat_skip = torch.cat((skip_connection, x), dim=1)
        #     x = self.ups[idx + 1](concat_skip)
        # Upsampling manually unloop for torch.jit.script
        # idx = 0
        x = self.ups[0](x)
        skip_connection = skip_connections[0]
        # if x.shape != skip_connection.shape:
        #     x = nn.functional.resize(x, size=skip_connection.shape[2:])
        concat_skip = torch.cat((skip_connection, x), dim=1)
        x = self.ups[1](concat_skip)
        # idx = 2
        x = self.ups[2](x)
        skip_connection = skip_connections[1]
        # if x.shape != skip_connection.shape:
        #     x = nn.functional.resize(x, size=skip_connection.shape[2:])
        concat_skip = torch.cat((skip_connection, x), dim=1)
        x = self.ups[3](concat_skip)
        # idx = 4
        x = self.ups[4](x)
        skip_connection = skip_connections[2]
        # if x.shape != skip_connection.shape:
        #     x = nn.functional.resize(x, size=skip_connection.shape[2:])
        concat_skip = torch.cat((skip_connection, x), dim=1)
        x = self.ups[5](concat_skip)
        
        
        #bs,c,w,h ->reshape to bs ,w,h,c            
        x=self.final_conv(x)
        x=  F.sigmoid(x.permute(0, 2, 3, 1))
        x=x.squeeze(-1)
        return x


    # def forward(self, x):
    #     skip_connections = []

    #     # Downsampling
    #     for down in self.downs:
    #         x = down(x)
    #         skip_connections.append(x)
    #         x = self.pool(x)

    #     x = self.bottleneck(x)
    #     skip_connections = skip_connections[::-1]  # Reverse the list

    #     # Upsampling
    #     for idx in range(0,   self.n *2, 2):
    #         x = self.ups[idx](x)
    #         skip_connection = skip_connections[idx // 2]
    #         if x.shape != skip_connection.shape:
    #             x = nn.functional.resize(x, size=skip_connection.shape[2:])
    #         concat_skip = torch.cat((skip_connection, x), dim=1)
    #         x = self.ups[idx + 1](concat_skip)  
    #     #bs,c,w,h ->reshape to bs ,w,h,c            
    #     x=self.final_conv(x)
    #     x=  F.sigmoid(x.permute(0, 2, 3, 1))
    #     x=x.squeeze(-1)
    #     return x

