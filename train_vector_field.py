import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from FLowUtils.VectorField2d import UnsteadyVectorField2D
from FLowUtils.VastisDataset import buildDataset
from config.LoadConfig import load_config
#! TODO (s) of PyFLowVis
#? - [x]  create time-dependent 2d vector field save data as np.ndarray
#? - [ ]  TODO: create linear operations for 2d vector field
#? - [ ]  TODO: train a vector field for killing energy +regular
#? - [ ]  TODO: create analytical flow 
#? - [ ]  TODO: draw LIC and vector glyph 
#? - [ ]  TODO: draw pathline?
#? - [ ]  TODO: train vector field for killing +regular+ Di term 
#? - [ ]  TODO: curvature gradient
#? - [ ]  TODO: implement hanger of imgui widget


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Block 1
        self.conv1_1 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv1_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Flatten for fully connected layer
        self.flatten = nn.Flatten()
        
        # Fully connected layer
        self.fc1 = nn.Linear(64 * 32 * 32, 20)
        self.fc2 = nn.Linear(20, 4)

    def forward(self, x):
        # Block 1
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.relu(self.conv1_3(x))
        x = self.pool1(x)  # Downsample 1
        
        # Flatten and fully connected layer
        x = self.flatten(x)
        x = F.relu( self.fc1(x))
        x = F.relu( self.fc2(x))

        return x



# class TransformerEncoder(nn.Module):
#     def __init__(self, input_dim, model_dim, num_heads, num_layers):
#         super(TransformerEncoder, self).__init__()
#         self.embedding = nn.Linear(input_dim, model_dim)
#         encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

#     def forward(self, x):
#         x = self.embedding(x)  # Embedding the input
#         x = self.transformer(x)  # (seq_len, batch_size, model_dim)
#         x = x.mean(dim=0)  # Global average pooling over sequence
#         return x

class VortexClassifier(nn.Module):
    def __init__(self):
        super(VortexClassifier, self).__init__()
        self.cnn = CNN()
        # self.transformer = TransformerEncoder(input_dim=10, model_dim=128, num_heads=4, num_layers=2)
        self.classifier = nn.Linear(512 + 128, 1)  # Assuming output of CNN is 512, Transformer is 128

    def forward(self, image):
        image_features = self.cnn(image)
        return image_features
        # info_features = self.transformer(information)
        # combined_features = torch.cat((image_features, info_features), dim=1)
        # output = self.classifier(image_features)
        # return torch.sigmoid(output)
        # return output







def train_pipeline():
    #! TODO (s) of PyFLowVis
    #! TODO (s) of PyFLowVis
    #! TODO (s) of PyFLowVis    
    #! TODO (s) of PyFLowVis    
    #! todo: load unsteady data from cpp generated binary file [DONE]
    #! todo: create torch dataset   [DONE]
    #! todo: what is the label? ->fist stage the reference frame: a(t),b(t),c(t)  [DONE]
    #! todo: DEFINE THE cnn MODEL:   NET0: VORETXNET, NET1: VORETXsegNET, NET3: RESNET
    #! todo: what is the label? ->second stage the segmentation of as steady as possible (asap) field.
    #! todo: visualize the model's output: classification of vortex bondary+ coreline  in 3d space (2d space+ 1D time)
    #! todo:test the model's with RFC, bossineq, helix, spriral motion
    config=load_config("config\\cfgs\\config.yaml")
    training_args=config['training']
    unsteadyVastisDataset=buildDataset(training_args["dataset"])
    #build data loader using the dataset and args
    data_loader = torch.utils.data.DataLoader(unsteadyVastisDataset, batch_size=training_args['batch_size'], shuffle=training_args['shuffle'], num_workers=training_args['num_workers'], pin_memory=training_args['pin_memory'])


    #initialize training paramters from args
    epochs=training_args['epochs']
    device=training_args['device']
    #initialize the model
    model=VortexClassifier()
    model.to(device)
    #initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #training 
    for epoch in range(epochs):
        # for vectorFieldImage, information, label in unsteadyVastisDataset:
        for batch_idx,(vectorFieldImage, labelferenceFrame,labelVortex) in enumerate(data_loader):
            vectorFieldImage=vectorFieldImage[:,0,:,:]
            vectorFieldImage=vectorFieldImage.transpose(1,3)
            vectorFieldImage, label = vectorFieldImage.to(device),  labelVortex.to(device)
            # tx,ty,n,rc =labelVortex[0],labelVortex[1],labelVortex[2],labelVortex[3]
            optimizer.zero_grad()
            output = model(vectorFieldImage)            
            loss = F.mse_loss(output, labelVortex)
            loss.backward()
            optimizer.step()
            #print loss in every 50 epoch
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch+1}, iter {batch_idx+1},  Loss: {loss.item()}')

        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    return None




def ObserverFieldOptimization(INputFieldV,args):

    time_steps,Ydim,Xdim = INputFieldV.time_steps,INputFieldV.Ydim,INputFieldV.Xdim
    # Create an instance of UnsteadyVectorField2D
    vector_field = UnsteadyVectorField2D(Xdim, Ydim, time_steps)
    vector_field.to(args['device'])
    INputFieldV.to(args['device'])
    # Training setup
    epochs = args["epochs"]
    optimizer = torch.optim.Adam(vector_field.parameters(), lr=0.1)
    
    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = vector_field(INputFieldV.field) 
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    vector_field.to('cpu')
    INputFieldV.to('cpu')
    return vector_field














if __name__ == '__main__':
    train_pipeline()