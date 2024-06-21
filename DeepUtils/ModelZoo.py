import torch
import torch.nn as nn
import torch.nn.functional as F
def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)



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

        self.conv2_1 = nn.Conv3d(in_channels=hiddenSize, out_channels=hiddenSize, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm3d(hiddenSize)
        self.conv2_2 = nn.Conv3d(in_channels=hiddenSize, out_channels=hiddenSize, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm3d(hiddenSize)
        self.conv2_3 = nn.Conv3d(in_channels=hiddenSize, out_channels=hiddenSize, kernel_size=3, padding=1)
        self.bn2_3 = nn.BatchNorm3d(hiddenSize)

        self.conv2b_1 = nn.Conv3d(in_channels=hiddenSize, out_channels=hiddenSize, kernel_size=3, padding=1)
        self.bn2b_1 = nn.BatchNorm3d(hiddenSize)
        self.conv2b_2 = nn.Conv3d(in_channels=hiddenSize, out_channels=hiddenSize, kernel_size=3, padding=1)
        self.bn2b_2 = nn.BatchNorm3d(hiddenSize)
        self.conv2b_3 = nn.Conv3d(in_channels=hiddenSize, out_channels=hiddenSize, kernel_size=3, padding=1)
        self.bn2b_3 = nn.BatchNorm3d(hiddenSize)

        self.conv3_1 = nn.Conv3d(in_channels=hiddenSize, out_channels=hiddenSize, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm3d(hiddenSize)
        self.conv3_2 = nn.Conv3d(in_channels=hiddenSize, out_channels=hiddenSize, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm3d(hiddenSize)
        self.conv3_3 = nn.Conv3d(in_channels=hiddenSize, out_channels=hiddenSize, kernel_size=3, padding=1)
        self.bn3_3 = nn.BatchNorm3d(hiddenSize)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout3d(dropout_prob)

        # Flatten for fully connected layer
        self.flatten = nn.Flatten()
        
        # Fully connected layer
        DataSizeX = DataSizeX // 4
        DataSizeY = DataSizeY // 4
        TimeSteps = TimeSteps // 4
        self.fc1 = nn.Linear(hiddenSize * DataSizeX * DataSizeY * TimeSteps, hiddenSize)
        self.fc2 = nn.Linear(hiddenSize, hiddenSize)
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
        residual0 = x
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = F.relu(self.bn2_3(self.conv2_3(x)))
        x = x + residual0

        residual0 = x
        x = F.relu(self.bn2b_1(self.conv2b_1(x)))
        x = F.relu(self.bn2b_2(self.conv2b_2(x)))
        x = F.relu(self.bn2b_3(self.conv2b_3(x)))
        x = x + residual0

        # Conv Block 3
        residual = x
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = F.relu(self.bn3_3(self.conv3_3(x)))
        x = x + residual
        x = self.pool3(x)
        x = self.dropout3(x)

        # Flatten and fully connected layer
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = F.relu(self.fc2(x))
        x = self.dropout_fc(x)
        x = self.fc3(x)
        return x

#! todo: add vector field reconstruction loss into  ReferenceFrameExtractor
class ReferenceFrameExtractor(nn.Module):
    def __init__(self,inputChannels, DataSizeX,DataSizeY,TimeSteps,ouptputDim, hiddenSize=64):
        super(ReferenceFrameExtractor, self).__init__()
        self.cnn = CNN3D(inputChannels, DataSizeX,DataSizeY,TimeSteps,ouptputDim, hiddenSize)
        self.reconstructor = nn.Sequential(
            nn.Conv3d(inputChannels+6, hiddenSize, kernel_size=3, padding=1),
            nn.BatchNorm3d(hiddenSize),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.Conv3d(hiddenSize, hiddenSize * 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(hiddenSize * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.ConvTranspose3d(hiddenSize * 2, hiddenSize, kernel_size=2, stride=2),
            nn.BatchNorm3d(hiddenSize),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(hiddenSize, inputChannels, kernel_size=2, stride=2),
            nn.ReLU()  # Assuming the reconstructed field is normalized to [0, 1]
        )

    def forward(self, image):
        #image [batch_size, chanel=2,W=64, H=64, depth(timsteps)=7]
        bs,vecComponnetChannel,height,width,depth=image.shape
        #abc_t [batchsize, self.ouptputDim*time_steps]
        abc_t = self.cnn(image)
        #generate reconstruct steady field
        abc_t_reshape = abc_t.reshape(bs, 6, depth).unsqueeze(-2).unsqueeze(-2)
        #abc_t_reshape [batchsize, 6,  height, width,time_steps]
        abc_t_reshape_expanded = abc_t_reshape.repeat(1, 1, height,width,1)
        vectorFieldwithTransforamtion=torch.concat((image, abc_t_reshape_expanded), dim=1)
        rec = self.reconstructor(vectorFieldwithTransforamtion)
        return abc_t, rec 
    


        


# class VortexClassifier(nn.Module):
#     def __init__(self):
#         super(VortexClassifier, self).__init__()
#         self.reference_frame_extractor = ReferenceFrameExtractor()
#         self.cnn = CNN()

#     def forward(self, image):
#         reference_frame_abc= self.reference_frame_extractor(image)
#         image_features = self.cnn(image)
#         return image_features



