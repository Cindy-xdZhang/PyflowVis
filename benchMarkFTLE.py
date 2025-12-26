
import time
import numpy as np
from FLowUtils.FTLE import compute_FTLE_2D_CUDA_oneSlice, compute_FTLE_2D_CUDA_SM_oneSlice
from FLowUtils.VectorField2d import UnsteadyVectorField2D
from FLowUtils.netCDFLoader import load_UnsteadyVectorFields_netCDFOrAnalytical
import pycuda.driver as cuda

def benchMarkFTLE_CompuationPerformance():
    print("Initializing Benchmark...")
    
    # 1. Create Synthetic Data
    # Dimensions
    T, H, W = 8, 256, 256
    # T, H, W = 10, 1024, 2048 # Larger for better GPU saturation check?
    
    print(f"Vector Field Size: T={T}, H={H}, W={W}")
    
    # Random vector field
    # smooth it a bit so lookups aren't random noise (though doesn't matter for perf)
    data = np.random.rand(T, H, W, 2).astype(np.float32)*0.1
    
    # Create VectorField object
    vf = UnsteadyVectorField2D(W, H, T, [0,0,0], [W*0.1,H*0.1,T*0.05])   
    vf.field = data

    # Params
    ftle_t = 0.5
    step_size = 0.005
    max_iter = 1000
    upSampling = 8
    
    # Warmup
    print("Warming up CUDA...")
    _ = compute_FTLE_2D_CUDA_oneSlice(vf, ftle_t, step_size, 10, upSampling=upSampling)
    _ = compute_FTLE_2D_CUDA_SM_oneSlice(vf, ftle_t, step_size, 10, upSampling=upSampling)

    # Helper to measure time
    def measure(func, name):
        start = time.time()
        res = func(vf, ftle_t, step_size, max_iter, upSampling)
        cuda.Context.synchronize()
        end = time.time()
        print(f"{name} Time: {end-start:.4f} s")
        return res, end-start
    
    
    # Measure Global
    print("Running Global Memory Kernel...")
    res_global, t_global = measure(lambda v, t, s, m, u: compute_FTLE_2D_CUDA_oneSlice(v, t, s, m, u), "Global Mem")
    
    # Measure Shared
    print("Running Shared Memory Kernel...")
    res_sm, t_sm = measure(lambda v, t, s, m, u: compute_FTLE_2D_CUDA_SM_oneSlice(v, t, s, m, u), "Shared Mem")
    
    # Compare correctness
    # Note: FP precision diffs might occur, but should be small.
    # Shared mem uses same float precision.
    diff = np.abs(res_global - res_sm)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"Max Diff: {max_diff:.6e}, Mean Diff: {mean_diff:.6e}")
    if np.allclose(res_global, res_sm, atol=1e-4):
        print("✅ Results Match!")
    else:
        print("❌ Results Do Not Match (Check logic)")
        
    print(f"Speedup: {t_global / t_sm:.3f}x")
    
if __name__ == "__main__":
    benchMarkFTLE_CompuationPerformance()