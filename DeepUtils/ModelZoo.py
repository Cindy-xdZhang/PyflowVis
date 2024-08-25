import torch
import torch.nn as nn
import torch.nn.functional as F
def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

class CNN3DBNBLock(nn.Module):
    def __init__(self, inputChannels, hiddenSize=64, dropout_prob=0.5):
        self.conv1_1 = nn.Conv3d(in_channels=inputChannels, out_channels=hiddenSize, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm3d(hiddenSize)
        self.dropout1 = nn.Dropout3d(dropout_prob)

    def forward(self, x):
        # Conv Block 1
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = self.dropout1(x)


class CNN3D(nn.Module):
    def __init__(self, inputChannels, DataSizeX, DataSizeY, TimeSteps, outputDim, hiddenSize=64, dropout_prob=0.5):
        super(CNN3D, self).__init__()
        # the input tensor of Conv3d should be in the shape of[batch_size, chanel=2,W=64, H=64, depth(timsteps)=7]
        self.conv1_1 = nn.Conv3d(in_channels=inputChannels, out_channels=hiddenSize, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm3d(hiddenSize)
        self.conv1_2 = nn.Conv3d(in_channels=hiddenSize, out_channels=hiddenSize, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm3d(hiddenSize)
        self.conv1_3 = nn.Conv3d(in_channels=hiddenSize, out_channels=hiddenSize, kernel_size=3, padding=1)
        self.bn1_3 = nn.BatchNorm3d(hiddenSize)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout3d(dropout_prob)

        self.conv2_1 = nn.Conv3d(in_channels=hiddenSize, out_channels=hiddenSize*2, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm3d(hiddenSize*2)
        self.conv2_2 = nn.Conv3d(in_channels=hiddenSize*2, out_channels=hiddenSize*2, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm3d(hiddenSize*2)
        self.conv2_3 = nn.Conv3d(in_channels=hiddenSize*2, out_channels=hiddenSize*2, kernel_size=3, padding=1)
        self.bn2_3 = nn.BatchNorm3d(hiddenSize*2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv3d(in_channels=hiddenSize*2, out_channels=hiddenSize*4, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm3d(hiddenSize*4)
        self.conv3_2 = nn.Conv3d(in_channels=hiddenSize*4, out_channels=hiddenSize*4, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm3d(hiddenSize*4)
        self.conv3_3 = nn.Conv3d(in_channels=hiddenSize*4, out_channels=hiddenSize*4, kernel_size=3, padding=1)
        self.bn3_3 = nn.BatchNorm3d(hiddenSize*4)

        # self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout3d(dropout_prob)

        # Flatten for fully connected layer
        self.flatten = nn.Flatten()
        # Fully connected layer
        DataSizeX = DataSizeX // 4
        DataSizeY = DataSizeY // 4
        TimeSteps = TimeSteps // 4
        self.fc1 = nn.Linear(hiddenSize*4 * DataSizeX * DataSizeY * TimeSteps, hiddenSize*4)
        self.fc2 = nn.Linear(hiddenSize*4, hiddenSize)
        self.fc3 = nn.Linear(hiddenSize, outputDim)
        self.dropout_fc = nn.Dropout(dropout_prob)

    def forward(self, x):
        # Conv Block 1
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = F.relu(self.bn1_3(self.conv1_3(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        # Conv Block 2
        residual0 = x.repeat(1, 2, 1, 1, 1)
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = F.relu(self.bn2_3(self.conv2_3(x)))
        x = x + residual0
        x = self.pool2(x)

        # Conv Block 3
        residual =  x.repeat(1, 2, 1, 1, 1)
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = F.relu(self.bn3_3(self.conv3_3(x)))
        x = x + residual
        x = self.dropout3(x)

        # Flatten and fully connected layer
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = F.relu(self.fc2(x))
        x = self.dropout_fc(x)
        x = F.relu(self.fc3(x))
        return x


class ReferenceFrameExtractor(nn.Module):
    def __init__(self,inputChannels, DataSizeX,DataSizeY,TimeSteps,ouptputDim, hiddenSize=64):
        super(ReferenceFrameExtractor, self).__init__()
        self.cnn = CNN3D(inputChannels, DataSizeX,DataSizeY,TimeSteps,ouptputDim, hiddenSize)
        # self.reconstructor = nn.Sequential(
        #     nn.Conv3d(inputChannels+6, hiddenSize, kernel_size=3, padding=1),
        #     nn.BatchNorm3d(hiddenSize),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool3d(kernel_size=2, stride=2),
        #     nn.Conv3d(hiddenSize, hiddenSize * 2, kernel_size=3, padding=1),
        #     nn.BatchNorm3d(hiddenSize * 2),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool3d(kernel_size=2, stride=2),
        #     nn.ConvTranspose3d(hiddenSize * 2, hiddenSize, kernel_size=2, stride=2),
        #     nn.BatchNorm3d(hiddenSize),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose3d(hiddenSize, inputChannels, kernel_size=2, stride=2),
        #     nn.ReLU()  
        # )
        self.outputDim=ouptputDim
        self.referenceFrameDim=ouptputDim//TimeSteps

    def forward(self, image):
        #image [batch_size, chanel=2,W=64, H=64, depth(timsteps)=7]
        bs,vecComponnetChannel,height,width,depth=image.shape
        #abc_t [batchsize, self.ouptputDim*time_steps]
        abc_t_abcdot_t = self.cnn(image)
        return abc_t_abcdot_t





class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, n=4, features=64):
        super(UNet, self).__init__()
        self.n = n
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of U-Net
        in_features = in_channels
        for _ in range(n):
            self.downs.append(DoubleConv(in_features, features))
            in_features = features
            features *= 2

        # Bottom part of U-Net
        self.bottleneck = DoubleConv(features // 2, features)

        # Up part of U-Net
        for _ in range(n):
            self.ups.append(
                nn.ConvTranspose2d(features, features // 2, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(features, features // 2))
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

        return self.final_conv(x)




class TobiasReferenceFrameCNN(nn.Module):
    """ RoboustReferenceFrameCNN is the CNN model from paper: Robust Reference Frame Extraction from Unsteady 2D Vector Fields with Convolutional Neural Networks
    """
    def __init__(self,inputChannels, DataSizeX,DataSizeY,TimeSteps,ouptputDim, hiddenSize=64, dropout=0.1):
        super(TobiasReferenceFrameCNN, self).__init__()
        # the input tensor of Conv3d should be in the shape of[batch_size, chanel=2,W=16, H=16, depth(timsteps)]
        self.conv1_1 = nn.Conv3d(in_channels=inputChannels, out_channels=hiddenSize, kernel_size=3, stride=2,padding=1)
        self.bn1_1 = nn.BatchNorm3d(hiddenSize)
        # self.CNNlayerCountN=4-2 #n=log2(max(H,W))-2
        self.conv2_1 = nn.Conv3d(in_channels=hiddenSize, out_channels=hiddenSize*2, kernel_size=3, stride=2,padding=1)
        self.bn2_1 = nn.BatchNorm3d(hiddenSize*2)

        lastCnnOuputChannel=1024
        # self.conv3_1 = nn.Conv3d(in_channels=hiddenSize*2, out_channels=lastCnnOuputChannel, kernel_size=3, stride=2,padding=1)
        # self.bn3_1 = nn.BatchNorm3d(lastCnnOuputChannel)

        self.flatten = nn.Flatten()
        DataSizeX = DataSizeX // 4
        DataSizeY = DataSizeY // 4
        TimeSteps = 2#(5+1/2=3,3+1/2=2)
        # Fully connected layer
        self.fc1 = nn.Linear(hiddenSize*2 * DataSizeX * DataSizeY * TimeSteps, 1024)
        self.bn_fc_1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn_fc_2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, ouptputDim)
        # self.bn_fc_3 = nn.BatchNorm1d(ouptputDim)
        self.dropout_fc = nn.Dropout(dropout)
        self.outputDim=ouptputDim
        

    def forward(self, x):
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        # x = F.relu(self.bn3_1(self.conv3_1(x)))


        x = self.flatten(x)
        x = self.dropout_fc(F.relu(self.bn_fc_1(self.fc1(x))))
        x = self.dropout_fc(F.relu(self.bn_fc_2(self.fc2(x)))) 
        x =self.fc3(x)
        return x

    def get_loss(self, ret, labelRef,labelVortex) -> torch.Tensor:
        loss = F.mse_loss(ret, labelVortex)
        return loss

