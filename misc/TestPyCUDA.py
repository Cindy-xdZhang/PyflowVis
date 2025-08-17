
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


def test_kernel():
    print("test_kernel")
    minimal_kernel = """ __global__ void test_kernel() {  } """
    with open("assets/cuda_kernal/FTLE_CUDA.cu", "r") as file:
        kernel_src = file.read()
    try:
        module_minimal = SourceModule(minimal_kernel)
        module = SourceModule(kernel_src)
        print("compile done")
        minimal_kernel = module_minimal.get_function("test_kernel")
        compute_ftle_kernel = module.get_function("compute_FTLE_image_kernel")
        print("✅ FTLE CUDA kernel compiled successfully.")
    except cuda.CompileError as e:
        print("❌ FTLE CUDA kernel compilation failed:")
        print(e.stdout)
    except Exception as e:
        print(f"❌ An unexpected error occurred during CUDA initialization: {e}")


if __name__ == "__main__":
    test_kernel()