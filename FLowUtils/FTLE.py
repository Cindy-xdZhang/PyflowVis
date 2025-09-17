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
    time_list = np.linspace(0, vector_field.tmax-vector_field.tmin, int((vector_field.tmax - vector_field.tmin) / (vector_field.timeInterval / temporalUpSampling)) + 1)

    for time_slice in time_list:
        print(f"Computing FTLE at time {time_slice},progress {time_slice/time_list[-1]*100}%...")
        resultSlice.append(compute_FTLE_2D_CUDA_oneSlice(vector_field, time_slice, step_size, max_iteration,upSampling))

    result= ScalarField2D(int(vector_field.Xdim*upSampling), int(vector_field.Ydim*upSampling), len(time_list), vector_field.domainMinBoundary, vector_field.domainMaxBoundary)
    result.set_discrete_data(resultSlice)
    return result


# extract ftle ridge and valley as mask (1: ridge, -1: valley, 0: background)
def ridge_extraction(FTLE_field: ScalarField2D,
                     grad_tol_ratio: float = 0.02,
                     curv_percentile: float = 70.0,
                     grad_min_percentile: float = 30.0,
                     nms_offset_px: float = 1.0) -> np.ndarray:
    """
    Extract LCS ridges (maxima) and valleys (minima) from an FTLE scalar field using
    gradient–Hessian ridge criteria on each time slice.

    For 2D slice s(y,x):
      - Compute gradient g and Hessian H
      - Eigendecompose H → (λ_min, v_min), (λ_max, v_max)
      - Ridge condition (maximal): v_min^T g ≈ 0 and λ_min < 0 and s is local max along v_min
      - Valley condition (minimal): v_max^T g ≈ 0 and λ_max > 0 and s is local min along v_max

    Args:
      FTLE_field: ScalarField2D with data shape (T, Y, X)
      grad_tol_ratio: tolerance factor for |v^T g| relative to median |g|
      curv_percentile: percentile threshold on |λ| to reject weak curvature noise

    Returns:
      mask: np.ndarray (T, Y, X) with values {1: ridge, -1: valley, 0: none}
    """
    data = FTLE_field.getDataAsNumpy().astype(np.float32)  # (T, Y, X)
    T, Y, X = data.shape
    xmin, ymin, _ = FTLE_field.domainMinBoundary
    xmax, ymax, _ = FTLE_field.domainMaxBoundary
    dx = (xmax - xmin) / max(1, (X - 1))
    dy = (ymax - ymin) / max(1, (Y - 1))
    dx = float(dx if np.isfinite(dx) and dx > 0 else 1.0)
    dy = float(dy if np.isfinite(dy) and dy > 0 else 1.0)

    out = np.zeros((T, Y, X), dtype=np.int8)

    def bilinear_sample(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        H, W = img.shape
        x0 = np.floor(xs).astype(np.int32)
        y0 = np.floor(ys).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, W - 1)
        y1 = np.clip(y0 + 1, 0, H - 1)
        x0 = np.clip(x0, 0, W - 1)
        y0 = np.clip(y0, 0, H - 1)
        wx = xs - x0
        wy = ys - y0
        v00 = img[y0, x0]
        v10 = img[y0, x1]
        v01 = img[y1, x0]
        v11 = img[y1, x1]
        v0 = v00 * (1.0 - wx) + v10 * wx
        v1 = v01 * (1.0 - wx) + v11 * wx
        return v0 * (1.0 - wy) + v1 * wy

    for t in range(T):
        s = data[t]
        # First derivatives
        gy, gx = np.gradient(s, dy, dx)  # gy: d/dy, gx: d/dx
        # Second derivatives
        dyy, dyx = np.gradient(gy, dy, dx)
        dxy, dxx = np.gradient(gx, dy, dx)
        # Symmetrize H
        dxy = 0.5 * (dxy + dyx)

        # Gradient magnitude and tolerance
        gmag = np.sqrt(gx * gx + gy * gy)
        med_g = np.median(gmag)
        tol = float(grad_tol_ratio) * float(med_g + 1e-12)
        gmin = np.percentile(gmag, grad_min_percentile)

        # Eigen decomposition for 2x2 symmetric Hessian via closed form
        trace = dxx + dyy
        det = dxx * dyy - dxy * dxy
        disc = np.maximum(trace * trace * 0.25 - det, 0.0)
        root = np.sqrt(disc)
        lam_min = trace * 0.5 - root
        lam_max = trace * 0.5 + root

        # Eigenvectors (avoid degeneracy): for λ use vector (dxy, λ - dxx)
        # min eigenvector
        vmx = dxy
        vmy = lam_min - dxx
        nrm = np.sqrt(vmx * vmx + vmy * vmy) + 1e-20
        vmx /= nrm; vmy /= nrm
        # max eigenvector
        vMx = dxy
        vMy = lam_max - dxx
        nrmM = np.sqrt(vMx * vMx + vMy * vMy) + 1e-20
        vMx /= nrmM; vMy /= nrmM

        # Projection of gradient onto eigenvectors
        proj_min = vmx * gx + vmy * gy
        proj_max = vMx * gx + vMy * gy

        # Curvature strength thresholds (percentile of absolute lambda)
        lam_min_abs = np.abs(lam_min)
        lam_max_abs = np.abs(lam_max)
        th_min = np.percentile(lam_min_abs, curv_percentile)
        th_max = np.percentile(lam_max_abs, curv_percentile)

        # Candidate ridge/valley with stronger constraints
        ridge_cand = (np.abs(proj_min) <= tol) & (lam_min < -1e-12) & (lam_min_abs >= th_min) & (gmag >= gmin)
        valley_cand = (np.abs(proj_max) <= tol) & (lam_max > +1e-12) & (lam_max_abs >= th_max) & (gmag >= gmin)

        # Directional NMS along eigen directions using sub-pixel bilinear sampling (±nms_offset_px)
        yy, xx = np.meshgrid(np.arange(Y, dtype=np.float32), np.arange(X, dtype=np.float32), indexing='ij')
        # ridge: compare along v_min (normal)
        xn_p = xx + vmx * nms_offset_px
        yn_p = yy + vmy * nms_offset_px
        xn_m = xx - vmx * nms_offset_px
        yn_m = yy - vmy * nms_offset_px
        sp = bilinear_sample(s, xn_p, yn_p)
        sm = bilinear_sample(s, xn_m, yn_m)
        ridge_keep = (s >= sp) & (s >= sm)
        # valley: compare along v_max (normal to valley)
        xv_p = xx + vMx * nms_offset_px
        yv_p = yy + vMy * nms_offset_px
        xv_m = xx - vMx * nms_offset_px
        yv_m = yy - vMy * nms_offset_px
        spv = bilinear_sample(s, xv_p, yv_p)
        smv = bilinear_sample(s, xv_m, yv_m)
        valley_keep = (s <= spv) & (s <= smv)


        ridge = ridge_cand & ridge_keep
        valley = valley_cand & valley_keep

        # Write to output (-1 for valley, 1 for ridge; ridge takes precedence if overlap)
        mask = np.zeros((Y, X), dtype=np.int8)
        mask[valley] = -1
        mask[ridge] = 1
        out[t] = mask

    return out