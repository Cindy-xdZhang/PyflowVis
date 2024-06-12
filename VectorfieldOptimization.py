import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from FLowUtils.VectorField2d import UnsteadyVectorField2D

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


