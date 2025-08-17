from .VectorField2d import UnsteadyVectorField2D
from .ScalarField2d import ScalarField2D
from .VectorField3d import UnsteadyVectorField3D
from .flowlineIntegral import *
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

CUDA_KERNEL_MODULE=None
def get_or_compile_FTLE_CUDA_kernel():
    global CUDA_KERNEL_MODULE
    if CUDA_KERNEL_MODULE is not None:
        return CUDA_KERNEL_MODULE   
    with open("assets/cuda_kernal/FTLE_CUDA.cu", "r") as file:
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
    module=get_or_compile_FTLE_CUDA_kernel()
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
    block = (32, 16, 1)
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



def compute_FTLE_2D_field_CUDA(vector_field:UnsteadyVectorField2D, step_size:float, max_iteration:int,upSampling:int=2,temporalUpSampling:int=1):
    resultSlice=[]
    #geneate time to calculate FTLE
    time_list = np.linspace(vector_field.tmin, vector_field.tmax, int((vector_field.tmax - vector_field.tmin) / (vector_field.timeInterval / temporalUpSampling)) + 1)
    for time_slice in time_list:
        print(f"Computing FTLE at time {time_slice},progress {time_slice/time_list[-1]*100}%...")
        resultSlice.append(compute_FTLE_2D_CUDA_oneSlice(vector_field, time_slice, step_size, max_iteration,upSampling))

    result= ScalarField2D(vector_field.Xdim*upSampling, vector_field.Ydim*upSampling, len(time_list), vector_field.domainMinBoundary, vector_field.domainMaxBoundary)
    result.set_discrete_data(resultSlice)
    return result




