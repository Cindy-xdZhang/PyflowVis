import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from VectorField2d import VectorField2D
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


class ImageCNN(nn.Module):
    def __init__(self):
        super(ImageCNN, self).__init__()
        # Load a pre-trained ResNet and remove the last layer
        #! todo:
        # base_model = models.resnet18(pretrained=True)1
        self.features = nn.Sequential(*list(base_model.children())[:-1])  # Remove last layer

    def forward(self, x):
        x = self.features(x)  # Output shape will be (batch_size, 512, 1, 1)
        x = torch.flatten(x, 1)  # Flatten the features
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.embedding(x)  # Embedding the input
        x = self.transformer(x)  # (seq_len, batch_size, model_dim)
        x = x.mean(dim=0)  # Global average pooling over sequence
        return x

class VortexClassifier(nn.Module):
    def __init__(self):
        super(VortexClassifier, self).__init__()
        self.cnn = ImageCNN()
        self.transformer = TransformerEncoder(input_dim=10, model_dim=128, num_heads=4, num_layers=2)
        self.classifier = nn.Linear(512 + 128, 1)  # Assuming output of CNN is 512, Transformer is 128

    def forward(self, image, information):
        image_features = self.cnn(image)
        info_features = self.transformer(information)
        combined_features = torch.cat((image_features, info_features), dim=1)
        output = self.classifier(combined_features)
        return torch.sigmoid(output)



def train_pipeline(INputFieldV,args):

    time_steps,Ydim,Xdim = INputFieldV.time_steps,INputFieldV.Ydim,INputFieldV.Xdim
    # Create an instance of VectorField2D
    vector_field = VectorField2D(Xdim, Ydim, time_steps)
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
    train_pipeline(16,16,16)