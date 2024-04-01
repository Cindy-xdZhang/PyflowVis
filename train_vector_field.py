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


    



def train_pipeline(INputFieldV):

    time_steps,Ydim,Xdim = INputFieldV.time_steps,INputFieldV.Ydim,INputFieldV.Xdim
    # Create an instance of VectorField2D
    vector_field = VectorField2D(Xdim, Ydim, time_steps)

    # Training setup
    epochs = 5000
    optimizer = torch.optim.Adam(vector_field.parameters(), lr=0.005)
    
    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = vector_field(INputFieldV.field) 
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    
    return vector_field



if __name__ == '__main__':
    train_pipeline(16,16,16)