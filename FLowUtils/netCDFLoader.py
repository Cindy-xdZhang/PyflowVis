import numpy as np
import netCDF4 as nc
from .VectorField2d import UnsteadyVectorField2D, SteadyVectorField2D
from .VectorField3d import *
import os,time,logging

def measure_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time} seconds")
        return result
    return wrapper



class NetCDFLoader:
    @staticmethod
    def load_vector_field2d(file_path: str,timestep_begin=-1,timestep_end=-1) -> UnsteadyVectorField2D|SteadyVectorField2D:
        if not os.path.exists(file_path)  or not os.path.isfile(file_path):
            logging.error(f"url wrong")
            return
            
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
            xdim_axis=None
            for dim in spatial_dims:
                if str(dim).lower()  in ["xdim", "X"]:
                    xdim_axis=dim
            ydim_axis=None
            for dim in spatial_dims:
                if str(dim).lower()  in ["ydim", "Y"]:
                    ydim_axis=dim
                    
            Xdim ,  Ydim = len(dataset.dimensions[xdim_axis]) ,len(dataset.dimensions[ydim_axis]) 
            
            

            # Adjust time steps based on input parameters
            if time_axis_name is not None:
                total_timesteps =len(dataset.dimensions[time_axis_name]) if time_axis_name is not None else 1
                # Handle negative indices and validate range
                if timestep_begin < 0:
                    timestep_begin = 0
                if timestep_end < 0:
                    timestep_end = total_timesteps
                if timestep_begin >= total_timesteps:
                    raise ValueError(f"timestep_begin ({timestep_begin}) exceeds available timesteps ({total_timesteps})")
                if timestep_end > total_timesteps:
                    timestep_end = total_timesteps
                if timestep_begin >= timestep_end:
                    raise ValueError(f"Invalid time range: begin={timestep_begin}, end={timestep_end}")
                time_steps = timestep_end - timestep_begin
                time = dataset.variables[time_axis_name][timestep_begin:timestep_end]
                tmin, tmax = time.min(), time.max()
            else:
                 raise ValueError("Could not find time_axis_name in the NetCDF file")

            # Extract domain boundaries
            x = dataset.variables[xdim_axis][:]
            y = dataset.variables[ydim_axis][:]
            domainMinBoundary = [x.min(), y.min()]
            domainMaxBoundary = [x.max(), y.max()]

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
                    field_data = np.zeros((time_steps, Ydim,Xdim, 2))
                    for i, var_name in enumerate(names):
                        field_data[:, :, :, i] =dataset.variables[var_name][timestep_begin:timestep_end]
                    break
            # field_data is numpy array shape [t,x,y,w] ->permute axis to [t,y,x ,w]
            # vector_field.field = np.transpose(field_data, (0, 2, 1, 3))
            vector_field.field =field_data
        return vector_field
    
    @staticmethod
 

    @staticmethod
    @measure_execution_time
    def load_vector_field3d(file_path: str,timestep_begin=-1,timestep_end=-1) -> UnsteadyVectorField3D:
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
            time_axis_name=None
            for dim in dimensions:
                if str(dim).lower()  in ["time", "tdim"]:
                    time_axis_name=dim

            spatial_dims = [dim for dim in dimensions if dim not in ['time','const',"tdim"]]

            if len(spatial_dims) != 3:
                raise ValueError(f"Expected 3 spatial dimensions, found {len(spatial_dims)}")

            xdim_axis=None
            for dim in spatial_dims:
                if str(dim).lower()  in ["xdim", "X"]:
                    xdim_axis=dim
            ydim_axis=None
            for dim in spatial_dims:
                if str(dim).lower()  in ["ydim", "Y"]:
                    ydim_axis=dim
            zdim_axis=None
            for dim in spatial_dims:
                if str(dim).lower()  in ["zdim", "Z"]:
                    zdim_axis=dim
                    
            Xdim ,  Ydim, Zdim = len(dataset.dimensions[xdim_axis]) ,len(dataset.dimensions[ydim_axis]) ,len(dataset.dimensions[zdim_axis]) 
            
            # Adjust time steps based on input parameters
            if time_axis_name is not None:
                total_timesteps =len(dataset.dimensions[time_axis_name]) if time_axis_name is not None else 1
                if timestep_begin >= timestep_end:
                    raise ValueError(f"Invalid time range: begin={timestep_begin}, end={timestep_end}")
                # Handle negative indices and validate range
                if timestep_begin < 0:
                    timestep_begin = 0
                if timestep_end < 0:
                    timestep_end = total_timesteps
                if timestep_begin >= total_timesteps:
                   timestep_begin = 0
                if timestep_end > total_timesteps:
                    timestep_end = total_timesteps
                time_steps = timestep_end - timestep_begin
                time = dataset.variables[time_axis_name][timestep_begin:timestep_end]
                tmin, tmax = time.min(), time.max()
            else:
                 raise ValueError("Could not find time_axis_name in the NetCDF file")


            # Extract domain boundaries
            x = dataset.variables[xdim_axis][:]
            y = dataset.variables[ydim_axis][:]
            z = dataset.variables[zdim_axis][:]
            domainMinBoundary = [x.min(), y.min(), z.min()]
            domainMaxBoundary = [x.max(), y.max(), z.max()]


            vector_field = None
  
            time = dataset.variables[time_axis_name][:]
            tmin, tmax = time.min(), time.max()
            # Create UnsteadyVectorField3D instance
            vector_field = UnsteadyVectorField3D(Xdim, Ydim, Zdim, time_steps, domainMinBoundary, domainMaxBoundary, tmin, tmax)
    
            # Try different naming conventions for vector components
            component_names = [
                ['u', 'v', 'w'],
                ['x', 'y', 'z'],
                ['a', 'b', 'c'],
                ['Component1', 'Component2', 'Component3'],
                ['velocity_x', 'velocity_y', 'velocity_z']
            ]
   
            field_data = None
            for names in component_names:
                if all(name in dataset.variables for name in names):
                    field_data = np.zeros((time_steps, Zdim, Ydim, Xdim, 3))
                    for i, var_name in enumerate(names):
                        field_data[:, :, :, :,i] =dataset.variables[var_name][timestep_begin:timestep_end]
                    break

            if field_data is None:
                raise ValueError("Could not find vector components in the NetCDF file")

            vector_field.field = field_data

        return vector_field

