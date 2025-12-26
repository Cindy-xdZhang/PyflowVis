
import time
import numpy as np
from .FTLE import compute_FTLE_2D_CUDA_oneSlice, compute_FTLE_2D_CUDA_SM_oneSlice
from .VectorField2d import UnsteadyVectorField2D

def benchMarkFTLE_CompuationPerformance():
    print("Initializing Benchmark...")
    
    # 1. Create Synthetic Data
    # Dimensions
    T, H, W = 10, 256, 512
    # T, H, W = 10, 1024, 2048 # Larger for better GPU saturation check?
    
    print(f"Vector Field Size: T={T}, H={H}, W={W}")
    
    # Random vector field
    # smooth it a bit so lookups aren't random noise 
    data = np.random.rand(T, H, W, 2).astype(np.float32)
    
    # Create VectorField object
    # Constructor: Xdim, Ydim, time_steps, minB, maxB
    vf = UnsteadyVectorField2D(
        Xdim=W, Ydim=H, time_steps=T,
        domainMinBoundary=[0.0, 0.0, 0.0], domainMaxBoundary=[float(W), float(H), float(T-1)]
    )
    # Overwrite field with numpy data
    vf.field = data
    vf.u_field = data[...,0] # Is this needed? FTLE functions use getDataAsNumpy which returns data.
    # FTLE implementation calls getDataAsNumpy() -> returns vf.field (if numpy).
    
    # Params
    ftle_t = 0.5
    step_size = 0.1
    max_iter = 100
    upSampling = 2
    
    # Warmup
    print("Warming up CUDA...")
    _ = compute_FTLE_2D_CUDA_oneSlice(vf, ftle_t, step_size, 10, upSampling=upSampling)
    
    # Helper to measure time
    def measure(func, name):
        start = time.time()
        res = func(vf, ftle_t, step_size, max_iter, upSampling=upSampling)
        cuda.Context.synchronize()
        end = time.time()
        print(f"{name} Time: {end-start:.4f} s")
        return res, end-start
    
    import pycuda.driver as cuda
    
    # Measure Global
    print("Running Global Memory Kernel...")
    res_global, t_global = measure(lambda v, t, s, m, upSampling: compute_FTLE_2D_CUDA_oneSlice(v, t, s, m, upSampling=upSampling), "Global Mem")
    
    # Measure Shared
    print("Running Shared Memory Kernel...")
    res_sm, t_sm = measure(lambda v, t, s, m, upSampling: compute_FTLE_2D_CUDA_SM_oneSlice(v, t, s, m, upSampling=upSampling, USE_SM=True), "Shared Mem")
    
    # Compare correctness
    diff = np.abs(res_global - res_sm)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"Max Diff: {max_diff:.6e}, Mean Diff: {mean_diff:.6e}")
    if np.allclose(res_global, res_sm, atol=1e-4):
        print("✅ Results Match!")
    else:
        print("❌ Results Do Not Match (Check logic)")
        
    print(f"Speedup: {t_global / t_sm:.2f}x")
