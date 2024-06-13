import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from FLowUtils.LicRenderer import LicRenderingUnsteady
from FLowUtils.FlowReader import read_rootMetaGridresolution,loadOneFlowEntryRawData
from FLowUtils.VectorField2d import UnsteadyVectorField2D
from FLowUtils.VastisDataset import buildDataset
from config.LoadConfig import load_config
from datetime import datetime
import os
#! TODO (s) of PyFLowVis
#! TODO (s) of PyFLowVis
#! todo: load unsteady data from cpp generated binary file [DONE]
#! todo: create torch dataset   [DONE]
#! todo: what is the label? ->fist stage the reference frame: a(t),b(t),c(t)  [DONE]
#! todo: DEFINE THE cnn MODEL:   NET0: VORETXNET, NET1: VORETXsegNET, NET3: RESNET
#! todo: what is the label? ->second stage the segmentation of as steady as possible (asap) field.
#! todo: visualize the model's output: classification of vortex bondary+ coreline  in 3d space (2d space+ 1D time)
#! todo:test the model's with RFC, bossineq, helix, spriral motion
#? - [ ]  TODO: draw pathline?
#? - [ ]  TODO: train vector field for killing +regular+ Di term 
#? - [ ]  TODO: curvature gradient
#? - [ ]  TODO: implement hanger of imgui widget

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

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)



class CNN3D(nn.Module):
    def __init__(self, inputChannels, DataSizeX, DataSizeY, TimeSteps, outputDim, hiddenSize=64, dropout_prob=0.2):
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




def train_pipeline():

    config=load_config("config\\cfgs\\config.yaml")
    train_VastisDataset=buildDataset(config["dataset"],mode="train")

    training_args=config['training']

    #build data loader using the dataset and args
    bacth_size=training_args['batch_size']
    data_loader = torch.utils.data.DataLoader(train_VastisDataset, batch_size=bacth_size, shuffle=training_args['shuffle'], num_workers=training_args['num_workers'], pin_memory=training_args['pin_memory'])
    #initialize training paramters from args
    epochs=training_args['epochs']
    device=training_args['device']
    #initialize the model
    slicePerdata=config["dataset"]['time_steps']
    model=ReferenceFrameExtractor(2, 64,64,slicePerdata,ouptputDim=6*slicePerdata, hiddenSize=64)
    model.apply(init_weights)
    model.to(device)
    #initialize the  optimizer with L2 regularization
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    #training 
    for epoch in range(epochs):
        epochLoss=0
        for batch_idx,(vectorFieldImage, labelferenceFrame,labelVortex) in enumerate(data_loader):
            #vectorFieldImage shape is [batch_size, depth(timsteps)=7, W=64, H=64,chanel=2]
            #transpose to [batch_size, chanel=2,W=64, H=64, depth(timsteps)=7]
            vectorFieldImage=vectorFieldImage.transpose(1,4)
            steadyField3D=vectorFieldImage[:,:,:,:,0]
            steadyField3D=steadyField3D.unsqueeze(-1).repeat(1,1,1,1,slicePerdata).to(device)
            vectorFieldImage, label = vectorFieldImage.to(device),  labelVortex.to(device)

            Qt,tc=labelferenceFrame
            labelQtct=torch.concat((Qt,tc),dim=2)
            labelQtct=labelQtct.to(device)

            # tx,ty,n,rc =labelVortex[0],labelVortex[1],labelVortex[2],labelVortex[3]
            labelQtct=labelQtct.reshape(vectorFieldImage.shape[0],-1).to(device)
            optimizer.zero_grad()
          
            predictQtCt, rec  = model(vectorFieldImage) 
            lossRec = F.mse_loss(rec, steadyField3D)
            lossRef = F.mse_loss(predictQtCt, labelQtct)
            loss=lossRec+lossRef

            loss.backward() 
            optimizer.step()
            epochLoss+=loss.item()
            #print loss in every 50 epoch
            if batch_idx % 20 == 0:
                print(f'Epoch {epoch+1}, iter {batch_idx+1},  Loss: {loss.item()}')

        epochLoss/=len(data_loader)
        print(f'Epoch {epoch+1}, Loss: {epochLoss}')

    #testing        
    # test_VastisDataset=buildDataset(config["dataset"],mode="test")
    # test_data_loader= torch.utils.data.DataLoader(test_VastisDataset, batch_size=training_args['batch_size'], shuffle=training_args['shuffle'], num_workers=training_args['num_workers'], pin_memory=training_args['pin_memory'])
    # test_model(model,test_data_loader)
    # Save checkpoint
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    save_checkpoint(checkpoint, folder_name="models/", checkpoint_name=f'checkpoint_epoch_{epochs}.pth.tar')
    return None

def test_model(model,data_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (vectorFieldImage, labelferenceFrame, labelVortex) in enumerate(data_loader):
            vectorFieldImage = vectorFieldImage.transpose(1, 3)
            vectorFieldImage, label = vectorFieldImage, labelVortex
            output = model(vectorFieldImage)
            test_loss += F.mse_loss(output, labelVortex, reduction='sum').item()  # sum up batch loss

    test_loss /= len(data_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}\n')



def test(checkpoint_path):
    # config = load_config("config\\cfgs\\config.yaml")
    # test_args = config['testing']
    # unsteadyVastisDataset = buildDataset(test_args["dataset"])

    # data_loader = torch.utils.data.DataLoader(unsteadyVastisDataset, batch_size=test_args['batch_size'], shuffle=False, num_workers=test_args['num_workers'], pin_memory=test_args['pin_memory'])

    # device = test_args['device']
    # model = VortexClassifier()
    # model.to(device)

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # # Load the last checkpoint
    # load_checkpoint(torch.load(checkpoint_path), model, optimizer)

    # model.eval()
    # test_loss = 0
    # correct = 0

    # with torch.no_grad():
    #     for batch_idx, (vectorFieldImage, labelferenceFrame, labelVortex) in enumerate(data_loader):
    #         vectorFieldImage = vectorFieldImage[:,0,:,:]
    #         vectorFieldImage = vectorFieldImage.transpose(1, 3)
    #         vectorFieldImage, label = vectorFieldImage.to(device), labelVortex.to(device)
    #         output = model(vectorFieldImage)
    #         test_loss += F.mse_loss(output, labelVortex, reduction='sum').item()  # sum up batch loss

    # test_loss /= len(data_loader.dataset)
    # print(f'\nTest set: Average loss: {test_loss:.4f}\n')

    return None




def test_load_results(): 
    Xdim,Ydim,time_steps,dominMinBoundary,dominMaxBoundary=read_rootMetaGridresolution('C:\\Users\\zhanx0o\\Documents\\sources\\PyflowVis\\CppProjects\\data\\unsteady\\64_64_nomix\\velocity_rc_1n_2\\meta.json')
    directory_path = 'C:\\Users\\zhanx0o\\Documents\\sources\\PyflowVis\\CppProjects\\data\\unsteady\\64_64_nomix\\velocity_rc_1n_2\\rc_1_n_2_sample_0Si_1observer_1type_4.bin'
    loadField, labelReferenceFrameABC,votexInfo=loadOneFlowEntryRawData(directory_path,Xdim,Ydim,time_steps)
    dataSlice=loadField[0:7]    
    unsteadyField=UnsteadyVectorField2D(Xdim,Ydim,7,dominMinBoundary,dominMaxBoundary) 
    LicRenderingUnsteady(unsteadyField,128,1)


    

# if __name__ == '__main__':
#    test_load_results()

if __name__ == '__main__':
    train_pipeline()