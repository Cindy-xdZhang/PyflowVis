import torch
import torch.nn as nn
import numpy as np
import cereal
import os

class VectorFieldLinearOperation():
    """ the VectorFieldLinearOperation class implements linear operations on vector fields.
    """ 
    def __init__(self):
        super(VectorFieldLinearOperation, self).__init__()
    @staticmethod  
    def magnitude(v):
        """Compute the magnitude scalar field of vector field v."""
        return torch.sum(v ** 2)
    @staticmethod
    def difference( v, u):
        """Compute the difference vector field (v - u) and its magnitude scalar field."""
        diff = v - u
        magnitudeF = VectorFieldLinearOperation.magnitude(diff)
        return diff, magnitudeF
    @staticmethod
    def sum(v, u):
        """Compute the sum vector field (v + u)."""
        return v + u

    @staticmethod
    def compute_killing_energy(v):
        # Calculate the Killing energy for the vector field
        energy =None
        energyTimeSlice = []
        for t in range(v.time_steps):
            field_t = v.field[t]

            # at position(x,y) of matrix field_t we have vector2d U(x,y),
            # let field_Xminus = torch.roll(field_t, shifts=-1, dims=1) then at position(x,y) matrix  field_Xminus 
            # it is the  vector2d U(x+1,y), so the difference  is the forward difference in x direction.
            # but at the last column the difference is between the last column and the first column.
            dx_forward_difference = torch.roll(field_t, shifts=-1, dims=1) - field_t#Ux+1-Ux
            dx_forward_difference[:, -1,:] = 0#last column is the difference between the last column and the first column
            dx_backward_difference = field_t-torch.roll(field_t, shifts=1, dims=1) #Ux-Ux-1
            dx_backward_difference[:, 0,:] = 0 #first column is the difference between the first column and the last column
            dudx = (dx_forward_difference + dx_backward_difference) / (2*v.gridInterval[0])

            dy_forward_difference = torch.roll(field_t, shifts=-1, dims=0) - field_t#Uy+1-Uy
            dy_forward_difference[-1,:,:] = 0
            dy_backward_difference = field_t-torch.roll(field_t, shifts=1, dims=0) #Uy-Uy-1
            dy_backward_difference[0,:,:] = 0

            dudy = (dy_forward_difference + dy_backward_difference) / (2*v.gridInterval[1])
            dudx[:, -1,:] *= 2.0            
            dudx[:, -1,:] *= 2.0
            dudy[-1, :,:] *= 2.0
            dudy[0, :,:] *= 2.0

            
            # Correcting dimensions to match for addition
            dudx = dudx.unsqueeze(-1)#(ydim,xdim,2,1)
            dudy = dudy.unsqueeze(-1)

            gradient = torch.cat((dudx, dudy), dim=-1)#gradient shape is (Ydim,Xdim,2,2)
            transposed_gradient = gradient.permute(0, 1, 3, 2)  # Adjust dimensions as  transpose operation

            # Ensure dimensions match and compute the Killing energy
            nablaUPlus_nablauT=gradient + transposed_gradient
            killing_energy = (nablaUPlus_nablauT) ** 2
            Ke=killing_energy.sum()
            energy =Ke if energy is None else energy+Ke
            # energyTimeSlice.append(Ke)

        return energy 


    def lie_derivative(self, L, v):
        """Compute the Lie derivative (Lv) of vector field v."""
        pass
        # # Assuming L is another vector field acting on v
        # # Here we use a simplified approximation for demonstration purposes
        # dxL = torch.roll(L, shifts=-1, dims=2) - L
        # dyL = torch.roll(L, shifts=-1, dims=1) - L
        # # Compute cross product as a placeholder for the lie derivative operation
        # lie_derivative = dxL * v[..., 1] - dyL * v[..., 0]
        # return lie_derivative.unsqueeze(-1)



class VectorField2D(nn.Module):
    def __init__(self, Xdim, Ydim, time_steps,domainMinBoundary:list=[-2.0,-2.0,0.0],domainMaxBoundary:list=[2.0,2.0,2*np.pi]):
        super(VectorField2D, self).__init__()
        self.Xdim= Xdim
        self.Ydim = Ydim
        self.time_steps = time_steps
        # Initialize the vector field parameters with random values, considering the time dimension
        self.field = nn.Parameter(torch.randn(time_steps, Ydim,Xdim, 2))
        self.domainMinBoundary=domainMinBoundary
        self.domainMaxBoundary=domainMaxBoundary
        self.gridInterval = [(domainMaxBoundary[0]-domainMinBoundary[0])/(Xdim-1),(domainMaxBoundary[1]-domainMinBoundary[1])/(Ydim-1)]
        self.timeInterval = (domainMaxBoundary[2]-domainMinBoundary[2])/(time_steps-1)
     
    def setInitialVectorField(self, vector_field):
        self.field = nn.Parameter(torch.tensor(vector_field))
    
    def forward(self,inputFieldV):
        diff, magnitudeR=VectorFieldLinearOperation.difference(inputFieldV,self.field)
        killingEnergy=VectorFieldLinearOperation.compute_killing_energy(self)
        return killingEnergy+magnitudeR
    
    # Cereal serialization and deserialization functions
    def serialize(self, archive):
        archive(self.field, self.domainMinBoundary, self.domainMaxBoundary, self.gridInterval, self.timeInterval)

    def deserialize(self, archive):
        archive(self.field, self.domainMinBoundary, self.domainMaxBoundary, self.gridInterval, self.timeInterval)
        


# def save_vector_field(scene, file_path):
#     actWidget=scene.getObject("ActiveField")    if scene is not None else None
#     vec2d=actWidget.getActiveField() if actWidget is not None else None
#     if vec2d is not None:
#         directory = os.path.dirname(file_path)
#         if not os.path.exists(directory):
#             os.makedirs(directory)
#         # Check if the file exists, and create it if not
#         if not os.path.exists(file_path):
#             with open(file_path, 'wb') as f:
#                 self.serialize
#         else:
#             with open(file_path, 'wb') as f:
#                 cereal.serialize_state(vec2d, f)   

# def load_vector_field(scene,file_path):
#     actWidget=scene.getObject("ActiveField")    if scene is not None else None
#     if actWidget is not None:
#         vec2d=VectorField2D(-1,-1,-1)
#         with open(file_path, 'rb') as f:
#             cereal.deserialize_state(vec2d, f) 
#             actWidget.insertField("loaded field"+file_path,vec2d)
     