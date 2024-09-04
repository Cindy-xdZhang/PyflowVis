import numpy as np
import netCDF4 as nc
from .VectorField2d import UnsteadyVectorField2D, SteadyVectorField2D
import os
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
        # self.field = torch.randn(time_steps, Zdim, Ydim, Xdim, 3)
        self.gridInterval = [
            (domainMaxBoundary[0] - domainMinBoundary[0]) / (Xdim - 1),
            (domainMaxBoundary[1] - domainMinBoundary[1]) / (Ydim - 1),
            (domainMaxBoundary[2] - domainMinBoundary[2]) / (Zdim - 1)
        ]
        assert(time_steps > 1)
        self.timeInterval = (tmax - tmin) / (time_steps - 1)

    def getSlice(self, timeSlice) -> SteadyVectorField3D:
        steadyVectorField3D = SteadyVectorField3D(self.Xdim, self.Ydim, self.Zdim, self.domainMinBoundary, self.domainMaxBoundary)
        if isinstance(self.field, torch.Tensor):
            steadyVectorField3D.field = self.field.cpu().numpy()[timeSlice, :, :, :, :]
        elif isinstance(self.field, np.ndarray):
            steadyVectorField3D.field = self.field[timeSlice, :, :, :, :]
        return steadyVectorField3D

    def getDataAsNumpy(self):
        if isinstance(self.field, torch.Tensor):
            return self.field.detach().cpu().numpy()
        elif isinstance(self.field, np.ndarray):
            return self.field

    def getDataAsTensor(self):
        if isinstance(self.field, torch.Tensor):
            return self.field.detach().cpu()
        elif isinstance(self.field, np.ndarray):
            return torch.tensor(self.field)

class NetCDFLoader:
    @staticmethod
    def load_vector_field2d(file_path: str) -> UnsteadyVectorField2D|SteadyVectorField2D:
        if not os.path.exists(file_path)  or not os.path.isfile(file_path):
            raise ValueError(f"url wrong")
            
        with nc.Dataset(file_path, 'r') as dataset:
            # Check dimensions
            dimensions = list(dataset.dimensions.keys())
            time_axis_name=None
            for dim in dimensions:
                if str(dim).lower()  in ["time", "tdim"]:
                    time_axis_name=dim

            spatial_dims = [dim for dim in dimensions if str(dim).lower() not in ['time',"const","tdim"]]

            if len(spatial_dims) != 2:
                raise ValueError(f"Expected 2 spatial dimensions, found {len(spatial_dims)}")

            Ydim, Xdim = [len(dataset.dimensions[dim]) for dim in spatial_dims]
            time_steps = len(dataset.dimensions[time_axis_name]) if time_axis_name is not None else 1

            # Extract domain boundaries
            x = dataset.variables[spatial_dims[1]][:]
            y = dataset.variables[spatial_dims[0]][:]
            domainMinBoundary = [x.min(), y.min()]
            domainMaxBoundary = [x.max(), y.max()]

            # Extract time information
            if time_axis_name is not None:
                time = dataset.variables[time_axis_name][:]
                tmin, tmax = time.min(), time.max()
            else:
                tmin, tmax = 0, 1

            # Create UnsteadyVectorField2D instance
            vector_field = UnsteadyVectorField2D(Xdim, Ydim, time_steps, domainMinBoundary, domainMaxBoundary, tmin, tmax)

            # Try different naming conventions for vector components
            component_names = [
                ['u', 'v'],
                ['x', 'y'],
                ['a', 'b'],
                ['Component1', 'Component2'],
                ['velocity_x', 'velocity_y']
            ]

            field_data = None
            for names in component_names:
                if all(name in dataset.variables for name in names):
                    field_data = np.zeros((time_steps,  Xdim,Ydim, 2))
                    for i, var_name in enumerate(names):
                        if time_axis_name is not None:
                            field_data[:, :, :, i] = dataset.variables[var_name][:]
                        else:
                            field_data[0, :, :, i] = dataset.variables[var_name][:]
                    break

            if field_data is None:
                raise ValueError("Could not find vector components in the NetCDF file")

            vector_field.field = field_data.transpose(1,2)

        return vector_field
    
    @staticmethod
    def load_vector_field3d(file_path: str,variable_names) -> UnsteadyVectorField3D|SteadyVectorField3D:
        """Load a 3D vector field from a NetCDF file. It tries multiple naming conventions for the vector components, including:
            u, v, w;x, y, z;a, b, c;Component1, Component2, Component3;velocity_x, velocity_y, velocity_z
        Args:
            file_path (str): Path to the NetCDF file
        Returns:
            UnsteadyVectorField3D: The loaded vector field
        """
        with nc.Dataset(file_path, 'r') as dataset:
            # Check dimensions
            dimensions = list(dataset.dimensions.keys())
            has_time = 'time' in dimensions
            spatial_dims = [dim for dim in dimensions if dim not in ['time','const']]

            if len(spatial_dims) != 3:
                raise ValueError(f"Expected 3 spatial dimensions, found {len(spatial_dims)}")

            Zdim, Ydim, Xdim = [len(dataset.dimensions[dim]) for dim in spatial_dims]
            time_steps = len(dataset.dimensions['time']) if has_time else 1

            # Extract domain boundaries
            x = dataset.variables[spatial_dims[2]][:]
            y = dataset.variables[spatial_dims[1]][:]
            z = dataset.variables[spatial_dims[0]][:]
            domainMinBoundary = [x.min(), y.min(), z.min()]
            domainMaxBoundary = [x.max(), y.max(), z.max()]


            vector_field = None
            # Extract time information
            if has_time:
                time = dataset.variables['time'][:]
                tmin, tmax = time.min(), time.max()
                # Create UnsteadyVectorField3D instance
                vector_field = UnsteadyVectorField3D(Xdim, Ydim, Zdim, time_steps, domainMinBoundary, domainMaxBoundary, tmin, tmax)
                
            else:
                tmin, tmax = 0, 1
                vector_field = SteadyVectorField3D(Xdim, Ydim, Zdim, domainMinBoundary, domainMaxBoundary)

            

            # Try different naming conventions for vector components
            component_names = [
                ['u', 'v', 'w'],
                ['x', 'y', 'z'],
                ['a', 'b', 'c'],
                ['Component1', 'Component2', 'Component3'],
                ['velocity_x', 'velocity_y', 'velocity_z']
            ]
            if variable_names is not None:
                component_names.insert(0, variable_names)
                
            field_data = None
            for names in component_names:
                if all(name in dataset.variables for name in names):
                    field_data = np.zeros((time_steps, Zdim, Ydim, Xdim, 3))
                    for i, var_name in enumerate(names):
                        if has_time:
                            field_data[:, :, :, :, i] = dataset.variables[var_name][:]
                        else:
                            field_data[0, :, :, :, i] = dataset.variables[var_name][:]
                    break

            if field_data is None:
                raise ValueError("Could not find vector components in the NetCDF file")

            vector_field.field = torch.tensor(field_data)

        return vector_field
    

