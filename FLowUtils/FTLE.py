from .VectorField2d import UnsteadyVectorField2D
from .ScalarField2d import ScalarField2D
from .VectorField3d import UnsteadyVectorField3D
from .flowlineIntegral import *
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

CUDA_KERNEL_MODULE=None
CUDA_KERNEL_MODULE_SM=None

def get_or_compile_FTLE_CUDA_kernel(USE_SM:bool=False):
    if USE_SM==False:
        global CUDA_KERNEL_MODULE
        if CUDA_KERNEL_MODULE is not None:
            return CUDA_KERNEL_MODULE   
        with open("assets/cuda_kernal/FTLE_2D_CUDA.cu", "r") as file:
            kernel_src = file.read()
        try:
            CUDA_KERNEL_MODULE = SourceModule(kernel_src,
                                                options=["-O3","-use_fast_math"])
            print("✅ FTLE CUDA kernel compiled successfully.")
        except cuda.CompileError as e:
            print("❌ FTLE CUDA kernel compilation failed:")
            print(e.stdout)
        except Exception as e:
            print(f"❌ An unexpected error occurred during CUDA initialization: {e}")
        return CUDA_KERNEL_MODULE
    else:
        global CUDA_KERNEL_MODULE_SM
        if CUDA_KERNEL_MODULE_SM is not None:
            return CUDA_KERNEL_MODULE_SM   
        with open("assets/cuda_kernal/FTLE_2D_CUDA_sharemem.cu", "r") as file:
            kernel_src = file.read()
        try:
            CUDA_KERNEL_MODULE_SM = SourceModule(kernel_src,
                                                options=["-O3","-use_fast_math"])
            print("✅ FTLE CUDA SM kernel compiled successfully.")
        except cuda.CompileError as e:
            print("❌ FTLE CUDA SM kernel compilation failed:")
            print(e.stdout)
        except Exception as e:
            print(f"❌ An unexpected error occurred during FTLE CUDA SM  initialization: {e}")
        return CUDA_KERNEL_MODULE_SM



def compute_FTLE_2D_CUDA_oneSlice(vector_field:UnsteadyVectorField2D,  time:float, step_size:float, max_iteration:int,upSampling:int=2):
    """Compute 2D FTLE image on GPU using a CUDA kernel.

    Args:
        vector_field: UnsteadyVectorField2D with shape (T, Y, X, 2)
        pos3d: unused placeholder to keep signature compatibility (ignored)
        time: physical time at which to evaluate FTLE (will be shifted by tmin)
        step_size: FTLE integration dt (use negative for backward time or set method="backward")
        max_iteration: number of RK4 steps for flow map integration
        method: "forward" or "backward" to set advection direction (overrides sign of step_size)

    Returns:
        np.ndarray (H, W) of FTLE values as float64
    """

    # Normalize integration direction
    FTLE_dt = float(step_size)

    # Pull data to numpy (T, H, W, 2) float32
    data = vector_field.getDataAsNumpy()
    assert data is not None and data.ndim == 4 and data.shape[-1] == 2, "vector_field data must be (T, H, W, 2)"
    TotalTimeSteps, v_height, v_width, _ = data.shape

    # Split components, ensure contiguous float32
    field_u = np.ascontiguousarray(data[..., 0].astype(np.float32))
    field_v = np.ascontiguousarray(data[..., 1].astype(np.float32))

    # Grid spacings and time settings
    v_dx = float(vector_field.gridInterval[0])
    v_dy = float(vector_field.gridInterval[1])
    v_dt = float(vector_field.timeInterval if vector_field.time_steps > 1 else 1.0)
    # shift physical time to start from zero for device code (expects t in [0, (T-1)*dt])
    t_i = float(time)

    # Output grid (use same resolution as the field)
    FTLE_size_x = int(v_width*upSampling)
    FTLE_size_y = int(v_height*upSampling)
    bmax_x = (v_width - 1) * v_dx
    bmax_y = (v_height - 1) * v_dy
    FTLE_dx = bmax_x / (FTLE_size_x - 1) if FTLE_size_x > 1 else 1.0
    FTLE_dy = bmax_y / (FTLE_size_y - 1) if FTLE_size_y > 1 else 1.0

    
    #laod from .cu file
    module=get_or_compile_FTLE_CUDA_kernel(False)
    compute_ftle_kernel = module.get_function("compute_FTLE_image_kernel")
    if compute_ftle_kernel is None:
        raise Exception("Failed to get symbol compute_FTLE_image_kernel from module")

    # Allocate device memory
    u_gpu = cuda.mem_alloc(field_u.nbytes)
    v_gpu = cuda.mem_alloc(field_v.nbytes)
    FTLE_host = np.zeros((FTLE_size_y, FTLE_size_x), dtype=np.float64)
    FTLE_gpu = cuda.mem_alloc(FTLE_host.nbytes)

    # Copy inputs to device
    cuda.memcpy_htod(u_gpu, field_u)
    cuda.memcpy_htod(v_gpu, field_v)
    cuda.memcpy_htod(FTLE_gpu, FTLE_host)

    # Launch configuration
    block = (16, 16, 1)
    grid = ((FTLE_size_x + block[0] - 1) // block[0], (FTLE_size_y + block[1] - 1) // block[1], 1)


    # Launch kernel
    compute_ftle_kernel(
        u_gpu, v_gpu,
        np.int32(v_width), np.int32(v_height), np.int32(TotalTimeSteps), np.float64(v_dx), np.float64(v_dy), np.float64(v_dt),
        FTLE_gpu, np.int32(FTLE_size_x), np.int32(FTLE_size_y), np.float64(FTLE_dx), np.float64(FTLE_dy), np.float64(t_i), np.float64(FTLE_dt), np.int32(max_iteration),
        block=block, grid=grid
    )

    # Copy result back
    cuda.memcpy_dtoh(FTLE_host, FTLE_gpu)

    # Free device memory
    u_gpu.free()
    v_gpu.free()
    FTLE_gpu.free()
    # result= ScalarField2D(FTLE_size_x, FTLE_size_y, 1, vector_field.domainMinBoundary, vector_field.domainMaxBoundary)
    # result.set_discrete_data(FTLE_host)
    return FTLE_host




def compute_FTLE_2D_CUDA_SM_oneSlice(vector_field:UnsteadyVectorField2D,  time:float, step_size:float, max_iteration:int,upSampling:int=2):
    """Compute 2D FTLE image on GPU using a CUDA kernel (Shared Memory).
    """

    # Normalize integration direction
    FTLE_dt = float(step_size)

    # Pull data to numpy (T, H, W, 2) float32
    data = vector_field.getDataAsNumpy()
    assert data is not None and data.ndim == 4 and data.shape[-1] == 2, "vector_field data must be (T, H, W, 2)"
    TotalTimeSteps, v_height, v_width, _ = data.shape

    # Split components, ensure contiguous float32
    field_u = np.ascontiguousarray(data[..., 0].astype(np.float32))
    field_v = np.ascontiguousarray(data[..., 1].astype(np.float32))

    # Grid spacings and time settings
    v_dx = float(vector_field.gridInterval[0])
    v_dy = float(vector_field.gridInterval[1])
    v_dt = float(vector_field.timeInterval if vector_field.time_steps > 1 else 1.0)
    # shift physical time to start from zero for device code (expects t in [0, (T-1)*dt])
    t_i = float(time)

    # Output grid (use same resolution as the field)
    FTLE_size_x = int(v_width*upSampling)
    FTLE_size_y = int(v_height*upSampling)
    bmax_x = (v_width - 1) * v_dx
    bmax_y = (v_height - 1) * v_dy
    FTLE_dx = bmax_x / (FTLE_size_x - 1) if FTLE_size_x > 1 else 1.0
    FTLE_dy = bmax_y / (FTLE_size_y - 1) if FTLE_size_y > 1 else 1.0

    
    #laod from .cu file
    module=get_or_compile_FTLE_CUDA_kernel(USE_SM=True)
    compute_ftle_kernel_tiled = module.get_function("compute_FTLE_image_kernel_tiled")
    if compute_ftle_kernel_tiled is None:
        raise Exception("Failed to get symbol compute_FTLE_image_kernel_tiled from module")

    # Allocate device memory
    u_gpu = cuda.mem_alloc(field_u.nbytes)
    v_gpu = cuda.mem_alloc(field_v.nbytes)
    FTLE_host = np.zeros((FTLE_size_y, FTLE_size_x), dtype=np.float64)
    FTLE_gpu = cuda.mem_alloc(FTLE_host.nbytes)

    # Copy inputs to device
    cuda.memcpy_htod(u_gpu, field_u)
    cuda.memcpy_htod(v_gpu, field_v)
    cuda.memcpy_htod(FTLE_gpu, FTLE_host)

    # Launch configuration
    block = (16, 16, 1)
    grid = ((FTLE_size_x + block[0] - 1) // block[0], (FTLE_size_y + block[1] - 1) // block[1], 1)
    
    # Calculate shared memory requirement
    # max_tile_w = ceil(blockDim.x * FTLE_dx / v_dx) + 2*halo
    # FTLE_dx / v_dx is approx 1/upSampling
    halo = 2
    # Conservative estimate:
    # block cover in physical: blockDim * FTLE_dx
    # block cover in vector index: ceil(blockDim * FTLE_dx / v_dx) + 1 (for interpolation interp)
    # Actually floor(end)+1 - floor(start)
    # Approx: ceil(block.x / upSampling) + 2 or so.
    # To be safe and since upSampling can be < 1 (super resolution FTLE), 
    # v_width_covered = ceil(32 * FTLE_dx / v_dx) + 1
    # Let's compute exact max possible tile width for this grid setup
    
    ratio_x = FTLE_dx / v_dx
    ratio_y = FTLE_dy / v_dy
    
    # Max possible width in vector field indices for a block of size 32
    # 32 * ratio_x + small_eps
    max_v_w = int(np.ceil(block[0] * ratio_x)) + 2 # +1 for interpolation right edge, +1 for safety?
    max_v_h = int(np.ceil(block[1] * ratio_y)) + 2
    
    tile_w_shared = max_v_w + 2*halo
    tile_h_shared = max_v_h + 2*halo
    
    # Time tile: load 2 frames (current and next) to handle initial steps.
    # If t_i is between t_idx and t_idx+1, we need both.
    tile_t_start = int(np.floor(t_i / v_dt))
    tile_t_count = 2
    
    # Ensure tile_t_start is valid
    if tile_t_start < 0: tile_t_start = 0
    if tile_t_start >= TotalTimeSteps - 1: tile_t_start = TotalTimeSteps - 2 # if possible
    # if TotalTimeSteps is 1, tile_t_count=1? logic handles bounds.
    
    shared_mem_size = tile_w_shared * tile_h_shared * tile_t_count * 4 * 2 # 4 bytes float, 2 fields U,V
    
    # print(f"Shared Mem Launch: tile_w={tile_w_shared}, tile_h={tile_h_shared}, size={shared_mem_size} bytes")

    # Launch kernel
    compute_ftle_kernel_tiled(
        u_gpu, v_gpu,
        np.int32(v_width), np.int32(v_height), np.int32(TotalTimeSteps), np.float64(v_dx), np.float64(v_dy), np.float64(v_dt),
        FTLE_gpu, np.int32(FTLE_size_x), np.int32(FTLE_size_y), np.float64(FTLE_dx), np.float64(FTLE_dy), np.float64(t_i), np.float64(FTLE_dt), np.int32(max_iteration),
        np.int32(halo), np.int32(tile_t_start), np.int32(tile_t_count),
        block=block, grid=grid, shared=shared_mem_size
    )

    # Copy result back
    cuda.memcpy_dtoh(FTLE_host, FTLE_gpu)

    # Free device memory
    u_gpu.free()
    v_gpu.free()
    FTLE_gpu.free()
    # result= ScalarField2D(FTLE_size_x, FTLE_size_y, 1, vector_field.domainMinBoundary, vector_field.domainMaxBoundary)
    # result.set_discrete_data(FTLE_host)
    return FTLE_host


def compute_FTLE_2D_field_CUDA(vector_field:UnsteadyVectorField2D, step_size:float, max_iteration:int,upSampling:int=2,temporalUpSampling:int=1):
    resultSlice=[]
    #geneate time to calculate FTLE
    time_list = np.linspace(0, vector_field.tmax-vector_field.tmin, int((vector_field.tmax - vector_field.tmin) / (vector_field.timeInterval / temporalUpSampling)) + 1)

    for time_slice in time_list:
        print(f"Computing FTLE at time {time_slice},progress {time_slice/time_list[-1]*100}%...")
        resultSlice.append(compute_FTLE_2D_CUDA_oneSlice(vector_field, time_slice, step_size, max_iteration,upSampling))

    result= ScalarField2D(int(vector_field.Xdim*upSampling), int(vector_field.Ydim*upSampling), len(time_list), vector_field.domainMinBoundary, vector_field.domainMaxBoundary)
    result.set_discrete_data(resultSlice)
    return result

def compute_FTLE_2D_field_CUDA_SM(vector_field:UnsteadyVectorField2D, step_size:float, max_iteration:int,upSampling:int=2,temporalUpSampling:int=1):
    resultSlice=[]
    #geneate time to calculate FTLE
    time_list = np.linspace(0, vector_field.tmax-vector_field.tmin, int((vector_field.tmax - vector_field.tmin) / (vector_field.timeInterval / temporalUpSampling)) + 1)

    for time_slice in time_list:
        print(f"Computing FTLE at time {time_slice},progress {time_slice/time_list[-1]*100}%...")
        resultSlice.append(compute_FTLE_2D_CUDA_SM_oneSlice(vector_field, time_slice, step_size, max_iteration,upSampling,USE_SM=True))

    result= ScalarField2D(int(vector_field.Xdim*upSampling), int(vector_field.Ydim*upSampling), len(time_list), vector_field.domainMinBoundary, vector_field.domainMaxBoundary)
    result.set_discrete_data(resultSlice)
    return result
