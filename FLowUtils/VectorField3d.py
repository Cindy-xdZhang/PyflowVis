import torch
import torch.nn as nn
import numpy as np
# abstract base class work
from abc import ABC, abstractmethod
from .interpolation import bilinear_interpolate
class IVectorField3D:
    def __init__(self, Xdim: int, Ydim: int, Zdim: int, domainMinBoundary: list = [-2.0, -2.0, -2.0], domainMaxBoundary: list = [2.0, 2.0, 2.0], time_steps: int = 1, tmin: float = 0.0, tmax: float = 2 * np.pi):
        self.Xdim = Xdim
        self.Ydim = Ydim
        self.Zdim = Zdim
        self.domainMinBoundary = domainMinBoundary
        self.domainMaxBoundary = domainMaxBoundary
        self.time_steps = time_steps
        self.tmin = tmin
        self.tmax = tmax

class SteadyVectorField3D(IVectorField3D):
    def __init__(self, Xdim: int, Ydim: int, Zdim: int, domainMinBoundary: list = [-2.0, -2.0, -2.0], domainMaxBoundary: list = [2.0, 2.0, 2.0]):
        super(SteadyVectorField3D, self).__init__(Xdim, Ydim, Zdim, domainMinBoundary, domainMaxBoundary)
        self.field = np.zeros((Zdim, Ydim, Xdim, 3), np.float32)

    def getSlice(self, timeSlice):
        return self.field

class UnsteadyVectorField3D(IVectorField3D):
    def __init__(self, Xdim: int, Ydim: int, Zdim: int, time_steps: int, domainMinBoundary: list = [-2.0, -2.0, -2.0], domainMaxBoundary: list = [2.0, 2.0, 2.0], tmin: float = 0.0, tmax: float = 2 * np.pi):
        super(UnsteadyVectorField3D, self).__init__(Xdim, Ydim, Zdim, domainMinBoundary, domainMaxBoundary, time_steps, tmin, tmax)
        self.field = None
        # self.field = torch.zeros(time_steps, Zdim, Ydim, Xdim, 3)
        self.gridInterval = [
            (domainMaxBoundary[0] - domainMinBoundary[0]) / (Xdim - 1),
            (domainMaxBoundary[1] - domainMinBoundary[1]) / (Ydim - 1),
            (domainMaxBoundary[2] - domainMinBoundary[2]) / (Zdim - 1)
        ]
        assert(time_steps > 1)
        self.timeInterval = (tmax - tmin) / (time_steps - 1)

    def getSlice(self, timeSlice) -> SteadyVectorField3D:
        steadyVectorField3D = SteadyVectorField3D(self.Xdim, self.Ydim, self.Zdim, self.domainMinBoundary, self.domainMaxBoundary)
        steadyVectorField3D.field = self.field[timeSlice, :, :, :, :]
        return steadyVectorField3D

    # def getDataAsNumpy(self):
    #     if isinstance(self.field, torch.Tensor):
    #         return self.field.detach().cpu().numpy()
    #     elif isinstance(self.field, np.ndarray):
    #         return self.field

    # def getDataAsTensor(self):
    #     if isinstance(self.field, torch.Tensor):
    #         return self.field.detach().cpu()
    #     elif isinstance(self.field, np.ndarray):
    #         return torch.tensor(self.field)
