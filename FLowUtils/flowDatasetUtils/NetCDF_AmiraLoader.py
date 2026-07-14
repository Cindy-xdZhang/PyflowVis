import numpy as np
import netCDF4 as nc
import os,time,logging
import re
from ..VectorField2d import UnsteadyVectorField2D, SteadyVectorField2D
from ..VectorField3d import *
from ..AnalyticalFlowCreator import *


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
            dimensions = list(dataset.variables.keys())
            time_axis_name=None
            for dim in dimensions:
                if str(dim).lower()  in ["time", "tdim","t"]:
                    time_axis_name=dim

            spatial_dims = [dim for dim in dimensions if str(dim).lower()  in ["z","y","x",'xdim',"ydim","zdim"]]
            if len(spatial_dims) != 2:
                raise ValueError(f"Expected 2 spatial dimensions, found {len(spatial_dims)}")
            xdim_axis=None
            for dim in spatial_dims:
                if str(dim).lower()  in ["xdim", "x"]:
                    xdim_axis=dim
            ydim_axis=None
            for dim in spatial_dims:
                if str(dim).lower()  in ["ydim", "y"]:
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
            domainMinBoundary = [x.min(), y.min(),tmin]
            domainMaxBoundary = [x.max(), y.max(),tmax]

            # Create UnsteadyVectorField2D instance
            vector_field = UnsteadyVectorField2D(Xdim, Ydim, time_steps, domainMinBoundary, domainMaxBoundary)

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
    def save_vector_field2d(file_path: str,vector_field: UnsteadyVectorField2D,timestep_begin=-1,timestep_end=-1):
        if not isinstance(vector_field, UnsteadyVectorField2D):
            raise TypeError("Expected UnsteadyVectorField2D instance.")

        # Ensure field data is numpy array
        field_data = vector_field.getDataAsNumpy()

        # Adjust time steps based on input parameters
        total_timesteps = vector_field.time_steps
        if timestep_begin < 0:
            timestep_begin = 0
        if timestep_end < 0 or timestep_end > total_timesteps:
            timestep_end = total_timesteps
        if timestep_begin >= timestep_end:
            raise ValueError(f"Invalid time range: begin={timestep_begin}, end={timestep_end}")

        # Get sliced data
        time_slice = slice(timestep_begin, timestep_end)
        sliced_field_data = field_data[time_slice, :, :, :]
        vector_field_time = np.linspace(vector_field.tmin, vector_field.tmax, vector_field.time_steps)
        sliced_time = vector_field_time[time_slice]

        with nc.Dataset(file_path, 'w', format='NETCDF4') as dataset:
            dataset.createDimension('time', sliced_time.shape[0])
            dataset.createDimension('x', vector_field.Xdim)
            dataset.createDimension('y', vector_field.Ydim)

            # Create variables
            times = dataset.createVariable('time', 'f8', ('time',))
            xs = dataset.createVariable('x', 'f8', ('x',))
            ys = dataset.createVariable('y', 'f8', ('y',))
            us = dataset.createVariable('u', 'f8', ('time', 'y', 'x'))
            vs = dataset.createVariable('v', 'f8', ('time', 'y', 'x'))

            # Write data
            times[:] = sliced_time
            # Calculate x and y coordinates
            vector_field_x = np.linspace(vector_field.domainMinBoundary[0], vector_field.domainMaxBoundary[0], vector_field.Xdim)
            vector_field_y = np.linspace(vector_field.domainMinBoundary[1], vector_field.domainMaxBoundary[1], vector_field.Ydim)
     

            xs[:] = vector_field_x
            ys[:] = vector_field_y
            us[:] = sliced_field_data[:, :, :, 0]
            vs[:] = sliced_field_data[:, :, :, 1]

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
            dimensions_variables = list(dataset.variables.keys())
            dimensions_names = list(dataset.dimensions.keys())
            time_axis_name=None
            for dim in dimensions_names:
                if str(dim).lower()  in ["time", "tdim","t"]:
                    time_axis_name=dim
            time_variable_name=None
            for dim in dimensions_variables:
                if str(dim).lower()  in ["time", "tdim","t"]:
                    time_variable_name=dim

            spatial_dims_variable_names = [dim for dim in dimensions_variables if str(dim).lower()  in ["z","y","x",'xdim',"ydim","zdim"]]
            spatial_dims_axis_names = [dim for dim in dimensions_names if str(dim).lower()  in ["z","y","x",'xdim',"ydim","zdim"]]

            if len(spatial_dims_variable_names) != 3 and len(spatial_dims_axis_names) != 3:
                raise ValueError(f"Expected 3 spatial dimensions, found {len(spatial_dims_variable_names)} or {len(spatial_dims_axis_names)}")

            xdim_axis=None
            for dim in spatial_dims_axis_names:
                if str(dim).lower()  in ["xdim", "x"]:
                    xdim_axis=dim
            ydim_axis=None
            for dim in spatial_dims_axis_names:
                if str(dim).lower()  in ["ydim", "y"]:
                    ydim_axis=dim
            zdim_axis=None
            for dim in spatial_dims_axis_names:
                if str(dim).lower()  in ["zdim", "z"]:
                    zdim_axis=dim
            xdim_variable_name=None
            for dim in spatial_dims_variable_names:
                if str(dim).lower()  in ["xdim", "x"]:
                    xdim_variable_name=dim
            ydim_variable_name=None
            for dim in spatial_dims_variable_names:
                if str(dim).lower()  in ["ydim", "y"]:
                    ydim_variable_name=dim
            zdim_variable_name=None
            for dim in spatial_dims_variable_names:
                if str(dim).lower()  in ["zdim", "z"]:
                    zdim_variable_name=dim
                    


            Xdim ,  Ydim, Zdim = len(dataset.dimensions[xdim_axis]) ,len(dataset.dimensions[ydim_axis]) ,len(dataset.dimensions[zdim_axis]) 
            
            # Adjust time steps based on input parameters
            if time_axis_name is not None and time_variable_name is not None:
                total_timesteps =len(dataset.dimensions[time_axis_name]) if time_axis_name is not None else 1
                # Handle negative indices and validate range
                # Clamp timestep_begin and timestep_end to valid range
                timestep_begin = max(0, min(timestep_begin, total_timesteps - 1))
                timestep_end = max(0, min(timestep_end, total_timesteps - 1))
                if timestep_begin > timestep_end:
                    raise ValueError(f"Invalid time range: begin={timestep_begin}, end={timestep_end}")
                time_steps = timestep_end - timestep_begin
                time = dataset.variables[time_variable_name][timestep_begin:timestep_end]
                tmin, tmax = time.min(), time.max()
            else:
                #steady field
                time_steps=1
                tmin,tmax=0,0


            # Extract domain boundaries
            x = dataset.variables[xdim_variable_name][:]
            y = dataset.variables[ydim_variable_name][:]
            z = dataset.variables[zdim_variable_name][:]
            domainMinBoundary = [x.min(), y.min(), z.min()]
            domainMaxBoundary = [x.max(), y.max(), z.max()]

            # Create UnsteadyVectorField3D instance
            vector_field = UnsteadyVectorField3D(Xdim, Ydim, Zdim, time_steps, domainMinBoundary, domainMaxBoundary,tmin,tmax)
    
            # Try different naming conventions for vector components
            component_names = [
                ['u', 'v', 'w'],
                ['x', 'y', 'z'],
                ['a', 'b', 'c'],
                ['Component1', 'Component2', 'Component3'],
                ['velocity_x', 'velocity_y', 'velocity_z']
            ]
   
            field_data = None
            if time_steps > 1:
                for names in component_names:
                    if all(name in dataset.variables for name in names):
                        field_data = np.zeros((time_steps, Zdim, Ydim, Xdim, 3))
                        for i, var_name in enumerate(names):
                            field_data[:, :, :, :,i] =dataset.variables[var_name][timestep_begin:timestep_end]
                        break
            else:
                for names in component_names:
                    if all(name in dataset.variables for name in names):
                        field_data = np.zeros((1,Zdim, Ydim, Xdim, 3))
                        for i, var_name in enumerate(names):
                            field_data[0,  :, :, :,i] =dataset.variables[var_name][:]
                        break

            if field_data is None:
                raise ValueError("Could not find vector components in the NetCDF file")

            vector_field.field = field_data

        return vector_field


class AmiraLoader:
    """
    Special loader for Amira Mesh 2.1 (*.am) unsteady 2D vector fields.
    Works with datasets like: https://www.research-collection.ethz.ch/entities/researchdata/52b7a501-c669-411b-9265-f3f039a28dbe

    Assumptions:
      - Header line like: 'define Lattice NX NY NT'
      - Parameters contain: 'BoundingBox xmin xmax ymin ymax tmin tmax'
      - Lattice declaration: 'Lattice { float[2] Velocity } @<tag>' (two components interleaved)
      - Binary block starts after a line '@<tag>' and is little-endian float32 in x-fastest order, then y, then t.

    Returns UnsteadyVectorField2D with field shaped [T, Y, X, 2].
    """

  
    @staticmethod
    def _parse_header(lines):
        import re
        nx = ny = nt = None
        bbox = None
        tag = None
        comp = 2
        joined = "\n".join(lines)
        m = re.search(r"define\s+Lattice\s+(\d+)\s+(\d+)\s+(\d+)", joined, re.IGNORECASE)
        if m:
            nx, ny, nt = int(m.group(1)), int(m.group(2)), int(m.group(3))
        bb = re.search(r"BoundingBox\s+([-+eE0-9\.]+)\s+([-+eE0-9\.]+)\s+([-+eE0-9\.]+)\s+([-+eE0-9\.]+)\s+([-+eE0-9\.]+)\s+([-+eE0-9\.]+)", joined)
        if bb:
            bbox = [float(bb.group(i)) for i in range(1, 7)]
        for ln in lines:
            s = ln.strip()
            if s.lower().startswith("lattice {") and "float[" in s:
                try:
                    inside = s[s.index("float[") + 6:]
                    comp = int(inside.split("]")[0])
                except Exception:
                    comp = 2
                if "@" in s:
                    tag = s.split("@")[-1].strip()
            if s.startswith("@") and tag is None:
                tag = s[1:].strip()
        if nx is None or ny is None or nt is None:
            raise ValueError("AmiraLoader: cannot parse 'define Lattice NX NY NT'")
        if bbox is None:
            bbox = [0.0, float(nx - 1), 0.0, float(ny - 1), 0.0, float(nt - 1)]
        if comp != 2:
            raise ValueError(f"AmiraLoader: expected float[2], got float[{comp}]")
        return nx, ny, nt, bbox, (tag or "1")

    @staticmethod
    def load_vector_field2d(file_path: str) -> UnsteadyVectorField2D:
        if not os.path.exists(file_path) or not os.path.isfile(file_path):
            raise FileNotFoundError(file_path)
        with open(file_path, 'rb') as f:
            raw = f.read()

        marker = re.search(br'(?m)^#\s*Data section follows\r?\n?', raw)
        if not marker:
            #privous I write bug code generated  file with this header:
            # header_lines = [
            #     f"define Lattice {nx} {ny} {nt}",
            #     f"BoundingBox {xmin:.8g} {xmax:.8g} {ymin:.8g} {ymax:.8g} {tmin:.8g} {tmax:.8g}",
            #     f"Lattice {{ float[2] Velocity }} @1\n@1\n"
            # ]
            marker = re.search(br'@1\n@1\n', raw)
            if not marker:
                raise ValueError(b'No "Data section follows" marker found')

        header_bytes = raw[:marker.start()]
        data_binary   = raw[marker.end():]
        header_text = header_bytes.decode('ascii', errors='ignore')
        lines = header_text.splitlines()
        nx, ny, nt, bbox, tag = AmiraLoader._parse_header(lines)
        # read binary payload
        count = int(nx * ny * nt * 2)
        arr = np.frombuffer(data_binary, dtype='<f4', count=count)
        if arr.size != count:
            raise RuntimeError(f"AmiraLoader: unexpected EOF, need {count}, got {arr.size}")
        field = arr.reshape(nt, ny, nx, 2).astype(np.float32)
        xmin, xmax, ymin, ymax, tmin, tmax = bbox
        vf = UnsteadyVectorField2D(nx, ny, nt, [xmin, ymin, tmin], [xmax, ymax, tmax])
        vf.field = field
        return vf
        
    @staticmethod
    def save_vector_field2d(file_path: str,vector_field: UnsteadyVectorField2D):
        if not isinstance(vector_field, UnsteadyVectorField2D):
            raise TypeError("Expected UnsteadyVectorField2D instance.")

        # Extract dims and bbox
        nx = int(vector_field.Xdim)
        ny = int(vector_field.Ydim)
        nt = int(vector_field.time_steps)
        xmin, ymin, tmin = [float(x) for x in vector_field.domainMinBoundary]
        xmax, ymax, tmax = [float(x) for x in vector_field.domainMaxBoundary]

        # Prepare data as numpy float32 shaped [T, Y, X, 2]
        field = vector_field.getDataAsNumpy().astype(np.float32, copy=False)
        if field.ndim != 4 or field.shape[-1] != 2:
            raise ValueError("Expected field shape [T, Y, X, 2]")
        if field.shape[0] != nt or field.shape[1] != ny or field.shape[2] != nx:
            raise ValueError("Field shape does not match vector field dimensions")

        # Compose minimal AmiraMesh-like header compatible with our loader
        tag = "1"
        header_lines = [
            f"define Lattice {nx} {ny} {nt}",
            f"BoundingBox {xmin:.8g} {xmax:.8g} {ymin:.8g} {ymax:.8g} {tmin:.8g} {tmax:.8g}",
            f"Lattice {{ float[2] Velocity }} @{tag}",
            f"@{tag}",
        ]
        header = ("\n".join(header_lines) + "\n").encode("ascii")

        # Binary payload: little-endian float32, order x-fastest, then y, then t
        payload = field.astype('<f4', copy=False).tobytes(order='C')

        # Write file
        with open(file_path, 'wb') as f:
            f.write(header)
            f.write(payload)


def relocate_flow2d_dataset_folder(config):
    import platform

    if platform.system() == "Windows":
        # config.dataset.dat_dir="C:\\Users\\xingdi\\OneDrive - KAUST\\WorkingInProcess\\FLowVisAssets\\flowData2d"->move to config file
        if config.dataset.dat_dir is None:
            config.dataset.dat_dir="C:\\Users\\xingdi\\OneDrive - KAUST\\WorkingInProcess\\FLowVisAssets\\flowData2d"
        config.cache_dir="./outputs/"

    elif platform.system() == "Linux":
        if config.dataset.dat_dir is None:
            config.dataset.dat_dir="/ibex/user/zhanx0o/FLowDataFolder/"
        config.cache_dir="/ibex/user/zhanx0o/outputs/"
    else:
        raise ValueError(f"Unknown system: {platform.system()}")


def load_UnsteadyVectorFields_general(flowDataFolder,vector_field_names: list[str]|str):
    netCDF = NetCDFLoader()
    amiraLoader=AmiraLoader()
    UnsteadyVectorFields=[]
    vector_field_names = vector_field_names if isinstance(vector_field_names, list) else [vector_field_names]
    for name in vector_field_names:
        if name == "beads2d":
            UnsteadyVectorFields.append(beadsFLow([128,128],32))
        elif name == "doublegyre2d":
            UnsteadyVectorFields.append(double_gyre_2D([256,128],64))
        elif name == "rfc2d":
            UnsteadyVectorFields.append(rotation_four_center([128,128],32))
        elif os.path.isdir(os.path.join(flowDataFolder, name)) and \
                any(f.endswith(".am") for f in os.listdir(os.path.join(flowDataFolder, name))):
            # Generic Amira folder: flowDataFolder/<name>/ holds one or more .am files.
            # Each .am is a complete unsteady 2D field (e.g. GerrisTinySet: 8 files -> 8 fields).
            # Loaded in sorted filename order so field order is deterministic and matches the files.
            amira_folder = os.path.join(flowDataFolder, name)
            logging.info(f"[load_UnsteadyVectorFields_general] load amira files from {amira_folder}")
            am_files_list = []
            for file in sorted(os.listdir(amira_folder)):
                if not file.endswith(".am"):
                    continue
                am_file_path = os.path.join(amira_folder, file)
                try:
                    load_vector_field2d = amiraLoader.load_vector_field2d(am_file_path)
                except Exception as e:
                    print(f"[load_UnsteadyVectorFields_general] load {file} failed: {e}. Skip this field.")
                    continue
                if load_vector_field2d is None:
                    print(f"[load_UnsteadyVectorFields_general] load {file} failed. Skip this field.")
                    continue
                UnsteadyVectorFields.append(load_vector_field2d)
                am_files_list.append(file)
            if len(am_files_list) == 0:
                logging.warning(f"[load_UnsteadyVectorFields_general] no loadable .am files in {amira_folder}. Skip this field.")
            else:
                logging.info(f"[load_UnsteadyVectorFields_general] loaded {len(am_files_list)} .am files from {amira_folder}: {am_files_list}")
        else:
            try:
                vectorfield_datapath=os.path.join(flowDataFolder, f"{name}.nc")
                load_vector_field2d=netCDF.load_vector_field2d(vectorfield_datapath)
                if load_vector_field2d is not None:
                    UnsteadyVectorFields.append(load_vector_field2d)
            except Exception as e:
                print(f"[load_UnsteadyVectorFields_general] load {name} failed: {e}. Skip this field.")
    return UnsteadyVectorFields
