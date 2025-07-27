import torch
import torch.nn as nn
import numpy as np
# abstract base class work
from abc import ABC, abstractmethod
from .interpolation import bilinear_interpolate
from numba import njit 

@njit(cache=True)
def bilinear_interpolate_numpy(field_slice, x, y):
    # field_slice 应该是 numpy 数组
    # Get grid coordinates
    x0 = int(np.floor(x))
    x1 = int(np.ceil(x))
    y0 = int(np.floor(y))
    y1 = int(np.ceil(y))

    # Clamp to grid boundaries (assuming field_slice dimensions are known or passed)
    # This requires passing field_slice.shape or similar
    Ydim, Xdim, _ = field_slice.shape # Assuming shape (Y, X, 2)

    x0 = max(0, min(x0, Xdim - 1))
    x1 = max(0, min(x1, Xdim - 1))
    y0 = max(0, min(y0, Ydim - 1))
    y1 = max(0, min(y1, Ydim - 1))

    # Get interpolation weights
    wx = x - x0
    wy = y - y0

    # Get vectors at grid points
    v00 = field_slice[y0, x0]
    v10 = field_slice[y0, x1]
    v01 = field_slice[y1, x0]
    v11 = field_slice[y1, x1]

    # Bilinear interpolation
    v0 = v00 * (1.0 - wx) + v10 * wx
    v1 = v01 * (1.0 - wx) + v11 * wx
    return v0 * (1.0 - wy) + v1 * wy

# 可以为 get_vector_at_grid 也创建一个 njit 函数
@njit(cache=True)
def _get_vector_unsteady_numba(field_data, x, y, time):
    return field_data[time, y, x] # field_data 应该是一个 numpy 数组









class IDiscreteField2D(ABC):
    """IDiscreteField2D is an abstract base class for 2D vector/scalar fields with grid discretization, it provideds necessary api and grid information.

    Args:
        Xdim (int): x  dimension of the vector field
        Ydim (int):  y dimension of the vector field
        time_steps (int): for steady vector field time_steps is -1/1
        domainMinBoundary (list, optional): [xmin, ymin]. Defaults to [-2.0,-2.0,].
        domainMaxBoundary (list, optional): [xmax, ymax]. Defaults to [2.0,2.0].
    """        
    def __init__(self,Xdim:int, Ydim:int,domainMinBoundary:list=[-2.0,-2.0,0.0],domainMaxBoundary:list=[2.0,2.0,2.0],timsteps:int=1):
        self.Xdim= Xdim
        self.Ydim = Ydim
        self.time_steps = timsteps
        self.domainMinBoundary=domainMinBoundary
        self.domainMaxBoundary=domainMaxBoundary
        assert len(domainMinBoundary) == 3, "domainMinBoundary must be a list of length 3"
        assert len(domainMaxBoundary) == 3, "domainMaxBoundary must be a list of length 3"
        self.gridInterval = [(domainMaxBoundary[0]-domainMinBoundary[0])/(Xdim-1),(domainMaxBoundary[1]-domainMinBoundary[1])/(Ydim-1)]
        self.tmin=domainMinBoundary[2]
        self.tmax=domainMaxBoundary[2]
        self.timeInterval = (self.tmax-self.tmin)/(timsteps-1)
        self.valid=(self.domainMinBoundary[0]  <= self.domainMaxBoundary[0] and self.domainMinBoundary[1]  <= self.domainMaxBoundary[1])and (1 <= Xdim  and 1 <= Ydim)  and (timsteps>=1) and (self.timeInterval>=0)
        assert self.valid
        self.__name = f"Unnamed_Field_{Xdim}x{Ydim}_{timsteps}t_ID{np.random.randint(0, 10000)}"

    def getName(self):
        return self.__name
    def setName(self, name):
        self.__name=name

    @abstractmethod
    def getSlice(self, timeSlice):
        pass
     
    def getMinTime(self):
        return self.tmin
    def getMaxTime(self):
        return self.tmax
    
    def getPhysicalTime(self,idt:int)->float:
        return self.timeInterval*idt+self.tmin
    def getFloatGridTime(self,time:float)->float:
        return float((time - self.tmin) / self.timeInterval)
    def getIntGridTime(self,time:float)->int:
        return int((time - self.tmin) / self.timeInterval)
    def convert_physical_pos_2_grid_pos(self, posX:float,posY:float):
        # Convert physical coordinates to grid indices
        float_grid_x = (posX - self.domainMinBoundary[0]) / self.gridInterval[0]
        float_grid_y = (posY - self.domainMinBoundary[1]) / self.gridInterval[1]
        return float_grid_x,float_grid_y 
    def convert_grid_pos_2_physical_pos(self, grid_x:float,grid_y:float):
        physical_x = grid_x * self.gridInterval[0] + self.domainMinBoundary[0]
        physical_y = grid_y * self.gridInterval[1] + self.domainMinBoundary[1]
        return physical_x,physical_y
    

class SteadyVectorField2D(IDiscreteField2D):
    def __init__(self, Xdim:int, Ydim:int,domainMinBoundary:list=[-2.0,-2.0,0.0],domainMaxBoundary:list=[2.0,2.0,0.0]):
        super(SteadyVectorField2D, self).__init__(Xdim, Ydim,domainMinBoundary,domainMaxBoundary,1)
        self.field = np.zeros( (Ydim,Xdim,2),np.float32)
    def getSlice(self, timeSlice):
        return  self.field
    
    def get_vector(self, posX: float, posY: float, time: float) -> np.ndarray:
        """Get interpolated vector at arbitrary position.
        
        Args:
            posX (float): X coordinate
            posY (float): Y coordinate 
            time (float): Time step
            
        Returns:
            np.ndarray: 2D vector at specified position using bilinear interpolation
        """
        float_grid_x,float_grid_y=self.convert_physical_pos_2_grid_pos(posX,posY)
        # Get vectors at surrounding grid points
        x0 = int(np.floor(float_grid_x))
        x1 = int(np.ceil(float_grid_x))
        y0 = int(np.floor(float_grid_y))
        y1 = int(np.ceil(float_grid_y))
        
        # Clamp to grid boundaries
        x0 = max(0, min(x0, self.Xdim-1))
        x1 = max(0, min(x1, self.Xdim-1))
        y0 = max(0, min(y0, self.Ydim-1))
        y1 = max(0, min(y1, self.Ydim-1))
        # Get interpolation weights
        wx = posX - x0
        wy = posY - y0
        # Get vectors at grid points
        v00 = self.get_vector_at_grid(x0, y0)
        v10 = self.get_vector_at_grid(x1, y0)
        v01 = self.get_vector_at_grid(x0, y1)
        v11 = self.get_vector_at_grid(x1, y1)
        
        # Bilinear interpolation
        v0 = v00 * (1-wx) + v10 * wx
        v1 = v01 * (1-wx) + v11 * wx
        return v0 * (1-wy) + v1 * wy
    
    def get_vector_at_grid(self, x: int, y: int, time: int) -> np.ndarray:
        """Get vector at grid point.
        
        Args:
            x (int): Grid X index
            y (int): Grid Y index
            time (int): Time step index
            
        Returns:
            np.ndarray: 3D vector at grid point
        """
        return self.field[ y, x, :]
    

class UnsteadyVectorField2D(IDiscreteField2D):
    def __init__(self, Xdim:int, Ydim:int,time_steps:int,domainMinBoundary:list=[-2.0,-2.0,0.0],domainMaxBoundary:list=[2.0,2.0,2.0]):
        super(UnsteadyVectorField2D, self).__init__(Xdim, Ydim,domainMinBoundary,domainMaxBoundary,time_steps)
        # Initialize the vector field parameters with random values, considering the time dimension
        self.field = torch.randn(time_steps, Ydim,Xdim, 2)
        
    def get_vector_at_float_pos_int_slice(self, posX:float,posY:float,time:int):
        float_grid_x,float_grid_y=self.convert_physical_pos_2_grid_pos(posX,posY)
        vec =bilinear_interpolate(self.field[time],  posX,posY)
        return vec
    

    def get_vector(self, posX: float, posY: float, time: float) -> np.ndarray:
        return _get_vector_unsteady_numba(
            self.field.numpy() if isinstance(self.field, torch.Tensor) else self.field, 
            posX, posY, time,
            self.domainMinBoundary[0], self.domainMinBoundary[1],
            self.gridInterval[0], self.gridInterval[1],
            self.tmin, self.timeInterval,
            self.Xdim, self.Ydim, self.time_steps
        )


    def get_vector(self, posX: float, posY: float, time: float) -> np.ndarray:
        """Get interpolated vector at arbitrary position using trilinear interpolation.
        
        Args:
            posX (float): X coordinate
            posY (float): Y coordinate 
            time (float): physcial Time 
            
        Returns:
            np.ndarray: 2D vector at specified position using trilinear interpolation
        """
        # Convert physical coordinates to grid coordinates
        float_grid_x, float_grid_y = self.convert_physical_pos_2_grid_pos(posX, posY)
        float_grid_time = self.getFloatGridTime(time)
        
        # Get surrounding time indices
        t0 = int(np.floor(float_grid_time))
        t1 = int(np.ceil(float_grid_time))
        t0 = max(0, min(t0, self.time_steps-1))
        t1 = max(0, min(t1, self.time_steps-1))
        
        # Get vectors at surrounding grid points
        x0 = int(np.floor(float_grid_x))
        x1 = int(np.ceil(float_grid_x))
        y0 = int(np.floor(float_grid_y))
        y1 = int(np.ceil(float_grid_y))
        
        # Clamp to grid boundaries
        x0 = max(0, min(x0, self.Xdim-1))
        x1 = max(0, min(x1, self.Xdim-1))
        y0 = max(0, min(y0, self.Ydim-1))
        y1 = max(0, min(y1, self.Ydim-1))
        
        # Get interpolation weights
        wx = float_grid_x - x0
        wy = float_grid_y - y0
        wt = float_grid_time - t0
        
        # Get vectors at grid points for both time steps
        v000 = self.get_vector_at_grid(x0, y0, t0)
        v100 = self.get_vector_at_grid(x1, y0, t0)
        v010 = self.get_vector_at_grid(x0, y1, t0)
        v110 = self.get_vector_at_grid(x1, y1, t0)
        
        v001 = self.get_vector_at_grid(x0, y0, t1)
        v101 = self.get_vector_at_grid(x1, y0, t1)
        v011 = self.get_vector_at_grid(x0, y1, t1)
        v111 = self.get_vector_at_grid(x1, y1, t1)
        
        # Bilinear interpolation for t0
        v0_t0 = v000 * (1-wx) + v100 * wx
        v1_t0 = v010 * (1-wx) + v110 * wx
        v_t0 = v0_t0 * (1-wy) + v1_t0 * wy
        
        # Bilinear interpolation for t1
        v0_t1 = v001 * (1-wx) + v101 * wx
        v1_t1 = v011 * (1-wx) + v111 * wx
        v_t1 = v0_t1 * (1-wy) + v1_t1 * wy
        
        # Linear interpolation in time
        return v_t0 * (1-wt) + v_t1 * wt
    
    def get_vector_at_grid(self, x: int, y: int, time: int) -> np.ndarray:
        """Get vector at grid point.
        
        Args:
            x (int): Grid X index
            y (int): Grid Y index
            time (int): Time step index
            
        Returns:
            np.ndarray: 3D vector at grid point
        """
        return self.field[time, y, x, :]
       
    
    def getSlice(self, timeSlice) -> SteadyVectorField2D:
        steadyVectorField2D = SteadyVectorField2D(self.Xdim, self.Ydim,self.domainMinBoundary,self.domainMaxBoundary)
        if isinstance(self.field, torch.Tensor):
            steadyVectorField2D.field=self.field.cpu().numpy()[timeSlice,:,:,:]
        elif isinstance(self.field, np.ndarray):
            steadyVectorField2D.field=self.field[timeSlice,:,:,:]
        return steadyVectorField2D
    
    def getDataAsNumpy(self):
        # if self.field is torch tensor
        if isinstance(self.field, torch.Tensor) :
            data= self.field.detach().cpu().numpy()
            return data
        elif isinstance(self.field, np.ndarray):
            return self.field
        
    def getDataAsTensor(self):
        if isinstance(self.field, torch.Tensor) or  isinstance(self.field, nn.Parameter):
            data= self.field.detach().cpu()
            return data
        elif isinstance(self.field, np.ndarray):
            return torch.tensor(self.field)
        
    def numpy2torch(self):
        """Convert field data from  numpy array to torch tensor for the field parameter.
        """
        self.field = torch.tensor(self.field)

    def torch2numpy(self):
        """Convert field data from torch tensor to numpy array for the field parameter.
        """
        self.field = self.field.detach().cpu().numpy()
        


class LinearOperationONTrainableVectorField():
    """ the VectorFieldLinearOperation class implements linear operations on vector fields.
    """ 
    def __init__(self):
        super(LinearOperationONTrainableVectorField, self).__init__()
    @staticmethod  
    def magnitude(v):
        """Compute the magnitude scalar field of vector field v."""
        return torch.sum(v ** 2)
    @staticmethod
    def difference( v, u):
        """Compute the difference vector field (v - u) and its magnitude scalar field."""
        diff = v - u
        magnitudeF = LinearOperationONTrainableVectorField.magnitude(diff)
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
      

class UnsteadyVectorField2DTrainable(nn.Module,UnsteadyVectorField2D):
    def __init__(self, Xdim:int, Ydim:int,time_steps:int,domainMinBoundary:list=[-2.0,-2.0],domainMaxBoundary:list=[2.0,2.0], tmin=0.0,tmax=2*np.pi):
        nn.Module.__init__(self)
        UnsteadyVectorField2D.__init__(self,Xdim, Ydim,domainMinBoundary,domainMaxBoundary,time_steps,tmin,tmax)
        # Initialize the vector field parameters with random values, considering the time dimension
        self.field = nn.Parameter(torch.randn(time_steps, Ydim,Xdim, 2))
        self.gridInterval = [(domainMaxBoundary[0]-domainMinBoundary[0])/(Xdim-1),(domainMaxBoundary[1]-domainMinBoundary[1])/(Ydim-1)]
        self.timeInterval = (tmax-tmin)/(time_steps-1)

    def getSlice(self, timeSlice) -> SteadyVectorField2D:
        steadyVectorField2D = SteadyVectorField2D(self.Xdim, self.Ydim,self.domainMinBoundary,self.domainMaxBoundary)
        steadyVectorField2D.field=self.field.detach().cpu().numpy()[timeSlice,:,:,:]
        return steadyVectorField2D

    def forward(self,inputFieldV):
        diff, magnitudeR=VectorFieldLinearOperation.difference(inputFieldV,self.field)
        killingEnergy=VectorFieldLinearOperation.compute_killing_energy(self)
        return killingEnergy+magnitudeR
        




