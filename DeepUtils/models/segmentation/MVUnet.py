import torch
import torch.nn as nn
import torch.nn.functional as F
from ..build import MODELS

@MODELS.register_module()
class DengMVUnet(nn.Module):
    def __init__(self, in_channels, out_channels, n=3, features=64, dropout= 0.005,**kwargs):
        super().__init__()
        self.n = n
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
       

        # Down part of U-Net
        in_features = in_channels
        for _ in range(n):
            self.downs.append(DoubleConv(in_features, features,dropout))
            in_features = features
            features *= 2

        # Bottom part of U-Net
        self.bottleneck = DoubleConv(features // 2, features,dropout)

        # Up part of U-Net
        for _ in range(n):
            self.ups.append(
                nn.ConvTranspose2d(features, features // 2, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(features, features // 2,dropout))
            features //= 2

        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)
 

    def forward(self, x):
        skip_connections = []

        # Downsampling
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # Reverse the list

        # Upsampling
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = nn.functional.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        #bs,c,w,h ->reshape to bs ,w,h,c            
        x=self.final_conv(x)
        x=  F.softmax(x.permute(0, 2, 3, 1),dim=-1)
        return x

