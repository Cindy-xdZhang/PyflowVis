import torch
import torch.nn as nn
import numpy as np
from numba import njit
# abstract base class work
from abc import ABC, abstractmethod

@njit(cache=True)
def _quadrilinear_interpolate_njit(
    field_data: np.ndarray,
    posX: float, posY: float, posZ: float, time: float,
    Xdim: int, Ydim: int, Zdim: int, time_steps: int,
    domainMinBoundary: np.ndarray, gridInterval: np.ndarray,
    tmin: float, timeInterval: float
) -> np.ndarray:
    """
    Perform quadrilinear interpolation on a 4D vector field using Numba for acceleration.
    """
    # 1. convert_physical_pos_2_grid_pos
    float_grid_x = (posX - domainMinBoundary[0]) / gridInterval[0] if gridInterval[0] != 0 else 0.0
    float_grid_y = (posY - domainMinBoundary[1]) / gridInterval[1] if gridInterval[1] != 0 else 0.0
    float_grid_z = (posZ - domainMinBoundary[2]) / gridInterval[2] if gridInterval[2] != 0 else 0.0

    # 2. getFloatGridTime
    if timeInterval == 0:
        float_grid_time = 0.0
    else:
        float_grid_time = (time - tmin) / timeInterval

    # Get surrounding time indices
    t0 = int(np.floor(float_grid_time))
    t1 = int(np.ceil(float_grid_time))
    t0 = max(0, min(t0, time_steps - 1))
    t1 = max(0, min(t1, time_steps - 1))

    # Get vectors at surrounding grid points
    x0 = int(np.floor(float_grid_x))
    x1 = int(np.ceil(float_grid_x))
    y0 = int(np.floor(float_grid_y))
    y1 = int(np.ceil(float_grid_y))
    z0 = int(np.floor(float_grid_z))
    z1 = int(np.ceil(float_grid_z))

    # Clamp to grid boundaries
    x0 = max(0, min(x0, Xdim - 1))
    x1 = max(0, min(x1, Xdim - 1))
    y0 = max(0, min(y0, Ydim - 1))
    y1 = max(0, min(y1, Ydim - 1))
    z0 = max(0, min(z0, Zdim - 1))
    z1 = max(0, min(z1, Zdim - 1))

    # Get interpolation weights
    wx = float_grid_x - x0
    wy = float_grid_y - y0
    wz = float_grid_z - z0
    wt = float_grid_time - t0
    
    # Get vectors at grid points for both time steps, inlined from get_vector_at_grid
    # field access is field[t, z, y, x, :]
    v0000 = field_data[t0, z0, y0, x0]
    v1000 = field_data[t0, z0, y0, x1]
    v0100 = field_data[t0, z0, y1, x0]
    v1100 = field_data[t0, z0, y1, x1]
    v0010 = field_data[t0, z1, y0, x0]
    v1010 = field_data[t0, z1, y0, x1]
    v0110 = field_data[t0, z1, y1, x0]
    v1110 = field_data[t0, z1, y1, x1]

    v0001 = field_data[t1, z0, y0, x0]
    v1001 = field_data[t1, z0, y0, x1]
    v0101 = field_data[t1, z0, y1, x0]
    v1101 = field_data[t1, z0, y1, x1]
    v0011 = field_data[t1, z1, y0, x0]
    v1011 = field_data[t1, z1, y0, x1]
    v0111 = field_data[t1, z1, y1, x0]
    v1111 = field_data[t1, z1, y1, x1]

    # Trilinear interpolation for t0
    v00_t0 = v0000 * (1 - wx) + v1000 * wx
    v10_t0 = v0100 * (1 - wx) + v1100 * wx
    v01_t0 = v0010 * (1 - wx) + v1010 * wx
    v11_t0 = v0110 * (1 - wx) + v1110 * wx
    v0_t0 = v00_t0 * (1 - wy) + v10_t0 * wy
    v1_t0 = v01_t0 * (1 - wy) + v11_t0 * wy
    v_t0 = v0_t0 * (1 - wz) + v1_t0 * wz

    # Trilinear interpolation for t1
    v00_t1 = v0001 * (1 - wx) + v1001 * wx
    v10_t1 = v0101 * (1 - wx) + v1101 * wx
    v01_t1 = v0011 * (1 - wx) + v1011 * wx
    v11_t1 = v0111 * (1 - wx) + v1111 * wx
    v0_t1 = v00_t1 * (1 - wy) + v10_t1 * wy
    v1_t1 = v01_t1 * (1 - wy) + v11_t1 * wy
    v_t1 = v0_t1 * (1 - wz) + v1_t1 * wz
    
    # Linear interpolation in time
    return v_t0 * (1 - wt) + v_t1 * wt

class IVectorField3D(ABC):
    """IVectorField3D is an abstract base class for 3D vector fields with grid discretization.
    It provides necessary API and grid information.

    Args:
        Xdim (int): x dimension of the vector field
        Ydim (int): y dimension of the vector field
        Zdim (int): z dimension of the vector field
        domainMinBoundary (list, optional): [xmin, ymin, zmin, tmin]. Defaults to [-2.0, -2.0, -2.0, 0.0].
        domainMaxBoundary (list, optional): [xmax, ymax, zmax, tmax]. Defaults to [2.0, 2.0, 2.0, 0.0].
        timesteps (int): Number of time steps. Defaults to 1.
    """
    def __init__(self, Xdim: int, Ydim: int, Zdim: int, timesteps: int = 1,domainMinBoundary: list = [-2.0, -2.0, -2.0], domainMaxBoundary: list = [2.0, 2.0, 2.0], tmin:float=0.0,tmax:float=0.0):
        self.Xdim = Xdim
        self.Ydim = Ydim
        self.Zdim = Zdim
        self.time_steps = timesteps
        self.domainMinBoundary = np.array(domainMinBoundary, dtype=np.float32)
        self.domainMaxBoundary = np.array(domainMaxBoundary, dtype=np.float32)
        
        assert len(domainMinBoundary) == 3, "domainMinBoundary must be a list of length 3"
        assert len(domainMaxBoundary) == 3, "domainMaxBoundary must be a list of length 3"

        self.gridInterval = np.array([
            (self.domainMaxBoundary[0] - self.domainMinBoundary[0]) / (Xdim - 1) if Xdim > 1 else 0,
            (self.domainMaxBoundary[1] - self.domainMinBoundary[1]) / (Ydim - 1) if Ydim > 1 else 0,
            (self.domainMaxBoundary[2] - self.domainMinBoundary[2]) / (Zdim - 1) if Zdim > 1 else 0
        ], dtype=np.float32)
        
        self.tmin = tmin
        self.tmax = tmax
        
        if timesteps > 1:
            self.timeInterval = (self.tmax - self.tmin) / (timesteps - 1)
        else:
            self.timeInterval = 0

        self.valid = (self.domainMinBoundary[0] <= self.domainMaxBoundary[0] and
                      self.domainMinBoundary[1] <= self.domainMaxBoundary[1] and
                      self.domainMinBoundary[2] <= self.domainMaxBoundary[2]) and \
                     (1 <= Xdim and 1 <= Ydim and 1 <= Zdim) and (timesteps >= 1)
        assert self.valid
        
    def getDim(self):
        return 3
    def IsInside(self,pos_3d):
        if pos_3d[0] < self.domainMinBoundary[0] or pos_3d[0] > self.domainMaxBoundary[0] or \
            pos_3d[1] < self.domainMinBoundary[1] or pos_3d[1] > self.domainMaxBoundary[1] or pos_3d[2] < self.domainMinBoundary[2] or pos_3d[2] > self.domainMaxBoundary[2]:
            return False
        return True
    
    # def getName(self):
    #     return self.__name

    # def setName(self, name):
    #     self.__name = name

    @abstractmethod
    def getSlice(self, timeSlice):
        pass

    def getMinTime(self):
        return self.tmin

    def getMaxTime(self):
        return self.tmax

    def getPhysicalTime(self, idt: int) -> float:
        return self.timeInterval * idt + self.tmin

    def getFloatGridTime(self, time: float) -> float:
        if self.timeInterval == 0:
            return 0.0
        return float((time - self.tmin) / self.timeInterval)

    def getIntGridTime(self, time: float) -> int:
        if self.timeInterval == 0:
            return 0
        return int((time - self.tmin) / self.timeInterval)

    def convert_physical_pos_2_grid_pos(self, posX: float, posY: float, posZ: float):
        # Convert physical coordinates to grid indices
        float_grid_x = (posX - self.domainMinBoundary[0]) / self.gridInterval[0] if self.gridInterval[0] != 0 else 0
        float_grid_y = (posY - self.domainMinBoundary[1]) / self.gridInterval[1] if self.gridInterval[1] != 0 else 0
        float_grid_z = (posZ - self.domainMinBoundary[2]) / self.gridInterval[2] if self.gridInterval[2] != 0 else 0
        return float_grid_x, float_grid_y, float_grid_z

    def convert_grid_pos_2_physical_pos(self, grid_x: float, grid_y: float, grid_z: float):
        physical_x = grid_x * self.gridInterval[0] + self.domainMinBoundary[0]
        physical_y = grid_y * self.gridInterval[1] + self.domainMinBoundary[1]
        physical_z = grid_z * self.gridInterval[2] + self.domainMinBoundary[2]
        return physical_x, physical_y, physical_z
    
class SteadyVectorField3D(IVectorField3D):
    def __init__(self, Xdim: int, Ydim: int, Zdim: int, domainMinBoundary: list = [-2.0, -2.0, -2.0, 0.0], domainMaxBoundary: list = [2.0, 2.0, 2.0, 0.0]):
        super(SteadyVectorField3D, self).__init__(Xdim, Ydim, Zdim, domainMinBoundary, domainMaxBoundary, timesteps=1)
        self.field = np.zeros((Zdim, Ydim, Xdim, 3), np.float32)

    def getSlice(self, timeSlice):
        return self.field

class UnsteadyVectorField3D(IVectorField3D):
    def __init__(self, Xdim: int, Ydim: int, Zdim: int, time_steps: int, 
                 domainMinBoundary: list = [-2.0, -2.0, -2.0], domainMaxBoundary: list = [2.0, 2.0, 2.0],tmin:float=0.0,tmax:float=0.0):
        super(UnsteadyVectorField3D, self).__init__(Xdim, Ydim, Zdim, time_steps,domainMinBoundary, domainMaxBoundary,tmin,tmax)
        self.field = None
        # self.field = torch.zeros(time_steps, Zdim, Ydim, Xdim, 3)
        assert(time_steps >= 1)

    def getSlice(self, timeSlice) -> SteadyVectorField3D:
        steady_min_b = self.domainMinBoundary[:3] + [self.getPhysicalTime(timeSlice)]
        steady_max_b = self.domainMaxBoundary[:3] + [self.getPhysicalTime(timeSlice)]
        steadyVectorField3D = SteadyVectorField3D(self.Xdim, self.Ydim, self.Zdim, steady_min_b, steady_max_b)
        
        if self.field is not None:
            if isinstance(self.field, torch.Tensor):
                steadyVectorField3D.field = self.field[timeSlice, :, :, :, :].detach().cpu().numpy()
            else:
                steadyVectorField3D.field = self.field[timeSlice, :, :, :, :]
        return steadyVectorField3D
    
    def getDataAsNumpy(self):
        if isinstance(self.field, torch.Tensor):
            fielddata = self.field.detach().cpu().numpy()
        else:
            fielddata = self.field
        return fielddata    
    
    def get_vector(self, posX: float, posY: float, posZ: float, time: float) -> np.ndarray:
        """Get interpolated vector at arbitrary position using quadrilinear interpolation.
        
        Args:
            posX (float): X coordinate
            posY (float): Y coordinate 
            posZ (float): Z coordinate
            time (float): physical Time 
            
        Returns:
            np.ndarray: 3D vector at specified position
        """
        field_data = self.field
        if isinstance(field_data, torch.Tensor):
            field_data = field_data.detach().cpu().numpy()
        return _quadrilinear_interpolate_njit(
            field_data,
            posX, posY, posZ, time,
            self.Xdim, self.Ydim, self.Zdim, self.time_steps,
            self.domainMinBoundary, self.gridInterval,
            self.tmin, self.timeInterval
        )
    
    def get_vector_at_grid(self, x: int, y: int, z: int, time: int) -> np.ndarray:
        """Get vector at grid point.
        
        Args:
            x (int): Grid X index
            y (int): Grid Y index
            z (int): Grid Z index
            time (int): Time step index
            
        Returns:
            np.ndarray: 3D vector at grid point
        """
        if self.field is None:
            return np.zeros(3, dtype=np.float32)

        # Clamping
        x = max(0, min(x, self.Xdim-1))
        y = max(0, min(y, self.Ydim-1))
        z = max(0, min(z, self.Zdim-1))
        time = max(0, min(time, self.time_steps-1))
        
        if isinstance(self.field, torch.Tensor):
            return self.field[time, z, y, x, :].detach().cpu().numpy()
        return self.field[time, z, y, x, :]
