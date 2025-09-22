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
    block = (32, 32, 1)
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


# # extract ftle ridge and valley as mask (1: ridge, -1: valley, 0: background)
# def ridge_extraction(FTLE_field: ScalarField2D,
#                      grad_tol_ratio: float = 0.02,
#                      curv_percentile: float = 70.0,
#                      grad_min_percentile: float = 30.0,
#                      nms_offset_px: float = 1.0) -> np.ndarray:
#     """
#     Extract LCS ridges (maxima) and valleys (minima) from an FTLE scalar field using
#     gradient–Hessian ridge criteria on each time slice.

#     For 2D slice s(y,x):
#       - Compute gradient g and Hessian H
#       - Eigendecompose H → (λ_min, v_min), (λ_max, v_max)
#       - Ridge condition (maximal): v_min^T g ≈ 0 and λ_min < 0 and s is local max along v_min
#       - Valley condition (minimal): v_max^T g ≈ 0 and λ_max > 0 and s is local min along v_max

#     Args:
#       FTLE_field: ScalarField2D with data shape (T, Y, X)
#       grad_tol_ratio: tolerance factor for |v^T g| relative to median |g|
#       curv_percentile: percentile threshold on |λ| to reject weak curvature noise

#     Returns:
#       mask: np.ndarray (T, Y, X) with values {1: ridge, -1: valley, 0: none}
#     """
#     data = FTLE_field.getDataAsNumpy().astype(np.float32)  # (T, Y, X)
#     T, Y, X = data.shape
#     xmin, ymin, _ = FTLE_field.domainMinBoundary
#     xmax, ymax, _ = FTLE_field.domainMaxBoundary
#     dx = (xmax - xmin) / max(1, (X - 1))
#     dy = (ymax - ymin) / max(1, (Y - 1))
#     dx = float(dx if np.isfinite(dx) and dx > 0 else 1.0)
#     dy = float(dy if np.isfinite(dy) and dy > 0 else 1.0)

#     out = np.zeros((T, Y, X), dtype=np.int8)

#     def bilinear_sample(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
#         H, W = img.shape
#         x0 = np.floor(xs).astype(np.int32)
#         y0 = np.floor(ys).astype(np.int32)
#         x1 = np.clip(x0 + 1, 0, W - 1)
#         y1 = np.clip(y0 + 1, 0, H - 1)
#         x0 = np.clip(x0, 0, W - 1)
#         y0 = np.clip(y0, 0, H - 1)
#         wx = xs - x0
#         wy = ys - y0
#         v00 = img[y0, x0]
#         v10 = img[y0, x1]
#         v01 = img[y1, x0]
#         v11 = img[y1, x1]
#         v0 = v00 * (1.0 - wx) + v10 * wx
#         v1 = v01 * (1.0 - wx) + v11 * wx
#         return v0 * (1.0 - wy) + v1 * wy

#     for t in range(T):
#         s = data[t]
#         # First derivatives
#         gy, gx = np.gradient(s, dy, dx)  # gy: d/dy, gx: d/dx
#         # Second derivatives
#         dyy, dyx = np.gradient(gy, dy, dx)
#         dxy, dxx = np.gradient(gx, dy, dx)
#         # Symmetrize H
#         dxy = 0.5 * (dxy + dyx)

#         # Gradient magnitude and tolerance
#         gmag = np.sqrt(gx * gx + gy * gy)
#         med_g = np.median(gmag)
#         tol = float(grad_tol_ratio) * float(med_g + 1e-12)
#         gmin = np.percentile(gmag, grad_min_percentile)

#         # Eigen decomposition for 2x2 symmetric Hessian via closed form
#         trace = dxx + dyy
#         det = dxx * dyy - dxy * dxy
#         disc = np.maximum(trace * trace * 0.25 - det, 0.0)
#         root = np.sqrt(disc)
#         lam_min = trace * 0.5 - root
#         lam_max = trace * 0.5 + root

#         # Eigenvectors (avoid degeneracy): for λ use vector (dxy, λ - dxx)
#         # min eigenvector
#         vmx = dxy
#         vmy = lam_min - dxx
#         nrm = np.sqrt(vmx * vmx + vmy * vmy) + 1e-20
#         vmx /= nrm; vmy /= nrm
#         # max eigenvector
#         vMx = dxy
#         vMy = lam_max - dxx
#         nrmM = np.sqrt(vMx * vMx + vMy * vMy) + 1e-20
#         vMx /= nrmM; vMy /= nrmM

#         # Projection of gradient onto eigenvectors
#         proj_min = vmx * gx + vmy * gy
#         proj_max = vMx * gx + vMy * gy

#         # Curvature strength thresholds (percentile of absolute lambda)
#         lam_min_abs = np.abs(lam_min)
#         lam_max_abs = np.abs(lam_max)
#         th_min = np.percentile(lam_min_abs, curv_percentile)
#         th_max = np.percentile(lam_max_abs, curv_percentile)

#         # Candidate ridge/valley with stronger constraints
#         ridge_cand = (np.abs(proj_min) <= tol) & (lam_min < -1e-12) & (lam_min_abs >= th_min) & (gmag >= gmin)
#         valley_cand = (np.abs(proj_max) <= tol) & (lam_max > +1e-12) & (lam_max_abs >= th_max) & (gmag >= gmin)

#         # Directional NMS along eigen directions using sub-pixel bilinear sampling (±nms_offset_px)
#         yy, xx = np.meshgrid(np.arange(Y, dtype=np.float32), np.arange(X, dtype=np.float32), indexing='ij')
#         # ridge: compare along v_min (normal)
#         xn_p = xx + vmx * nms_offset_px
#         yn_p = yy + vmy * nms_offset_px
#         xn_m = xx - vmx * nms_offset_px
#         yn_m = yy - vmy * nms_offset_px
#         sp = bilinear_sample(s, xn_p, yn_p)
#         sm = bilinear_sample(s, xn_m, yn_m)
#         ridge_keep = (s >= sp) & (s >= sm)
#         # valley: compare along v_max (normal to valley)
#         xv_p = xx + vMx * nms_offset_px
#         yv_p = yy + vMy * nms_offset_px
#         xv_m = xx - vMx * nms_offset_px
#         yv_m = yy - vMy * nms_offset_px
#         spv = bilinear_sample(s, xv_p, yv_p)
#         smv = bilinear_sample(s, xv_m, yv_m)
#         valley_keep = (s <= spv) & (s <= smv)


#         ridge = ridge_cand & ridge_keep
#         valley = valley_cand & valley_keep

#         # Write to output (-1 for valley, 1 for ridge; ridge takes precedence if overlap)
#         mask = np.zeros((Y, X), dtype=np.int8)
#         mask[valley] = -1
#         mask[ridge] = 1
#         out[t] = mask

#     return out




import numpy as np
# If VTK is needed for data structures or interpolation:
import vtk



def _bilinear_sample(arr, ys, xs):
    H, W = arr.shape
    ys = np.asarray(ys, dtype=np.float64)
    xs = np.asarray(xs, dtype=np.float64)

    y0 = np.floor(ys).astype(np.int32)
    x0 = np.floor(xs).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, H - 1)
    x1 = np.clip(x0 + 1, 0, W - 1)
    y0 = np.clip(y0, 0, H - 1)
    x0 = np.clip(x0, 0, W - 1)

    wy = ys - y0
    wx = xs - x0

    v00 = arr[y0, x0]
    v01 = arr[y1, x0]
    v10 = arr[y0, x1]
    v11 = arr[y1, x1]

    v0 = v00 * (1 - wx) + v10 * wx
    v1 = v01 * (1 - wx) + v11 * wx
    return v0 * (1 - wy) + v1 * wy


# ------------ 从整格数据构建 ∇s 与 H，并能在浮点坐标采样它们 ------------

def _precompute_derivatives_2d(s):
    """在整格上用中心差分计算 ∇s 与 Hessian H 的分量。"""
    Hh, Wh = s.shape
    # 一阶
    gy = np.zeros_like(s, dtype=np.float64)
    gx = np.zeros_like(s, dtype=np.float64)
    gy[1:-1, :] = 0.5 * (s[2:, :] - s[:-2, :])
    gx[:, 1:-1] = 0.5 * (s[:, 2:] - s[:, :-2])
    gy[0, :] = s[1, :] - s[0, :]
    gy[-1, :] = s[-1, :] - s[-2, :]
    gx[:, 0] = s[:, 1] - s[:, 0]
    gx[:, -1] = s[:, -1] - s[:, -2]

    # 二阶
    dyy = np.zeros_like(s, dtype=np.float64)
    dxx = np.zeros_like(s, dtype=np.float64)
    dxy = np.zeros_like(s, dtype=np.float64)

    dyy[1:-1, :] = s[2:, :] - 2 * s[1:-1, :] + s[:-2, :]
    dyy[0, :] = s[1, :] - 2 * s[0, :] + s[0, :]
    dyy[-1, :] = s[-1, :] - 2 * s[-1, :] + s[-2, :]

    dxx[:, 1:-1] = s[:, 2:] - 2 * s[:, 1:-1] + s[:, :-2]
    dxx[:, 0] = s[:, 1] - 2 * s[:, 0] + s[:, 0]
    dxx[:, -1] = s[:, -1] - 2 * s[:, -1] + s[:, -2]

    # f_xy 用九点模板近似
    dxy[1:-1, 1:-1] = 0.25 * (s[2:, 2:] - s[2:, :-2] - s[:-2, 2:] + s[:-2, :-2])
    dxy[0, :] = 0.0
    dxy[-1, :] = 0.0
    dxy[:, 0] = 0.0
    dxy[:, -1] = 0.0
    return gx, gy, dxx, dyy, dxy


def _sample_grad_hess(s, gx, gy, dxx, dyy, dxy, ys, xs):
    """在浮点 (ys, xs) 位置采样 s, ∇s, H（用双线性对每个分量）。"""
    vals = _bilinear_sample(s, ys, xs)
    gxs = _bilinear_sample(gx, ys, xs)
    gys = _bilinear_sample(gy, ys, xs)
    dxxs = _bilinear_sample(dxx, ys, xs)
    dyys = _bilinear_sample(dyy, ys, xs)
    dxys = _bilinear_sample(dxy, ys, xs)
    return vals, gxs, gys, dxxs, dyys, dxys


# ------------ 核心：一次在给定“层级网格”上做 ridge 边检测 ------------

def _detect_ridge_edges_on_level(s, gx, gy, dxx, dyy, dxy,
                                 grid_y_idx, grid_x_idx,
                                 s_min=None, lambda_max=None):
    """
    在一张层级网格（由 grid_*_idx 指定的整格索引点）上，检测每个单元四条边的 ridge 交点。
    返回：list[(y0,x0,y1,x1,  level_tag)] —— 边两端（fine-grid 像素坐标），用于写回 mask。
    """
    Htot, Wtot = s.shape
    hits = []

    # 每个单元：四顶点索引（以 fine-grid 像素坐标计）
    for i in range(len(grid_y_idx) - 1):
        for j in range(len(grid_x_idx) - 1):
            y00, x00 = grid_y_idx[i],   grid_x_idx[j]
            y01, x01 = grid_y_idx[i+1], grid_x_idx[j]
            y10, x10 = grid_y_idx[i],   grid_x_idx[j+1]
            y11, x11 = grid_y_idx[i+1], grid_x_idx[j+1]

            # 四条边端点（A->B），以及中点（用于统一法向）
            edges = [
                ((y00, x00), (y10, x10)),  # top  边（x 方向）
                ((y01, x01), (y11, x11)),  # bottom 边
                ((y00, x00), (y01, x01)),  # left 边（y 方向）
                ((y10, x10), (y11, x11)),  # right 边
            ]

            for (Ay, Ax), (By, Bx) in edges:
                # 用边中点处的 Hessian 求最小特征向量（ridge 法向）
                my = 0.5 * (Ay + By)
                mx = 0.5 * (Ax + Bx)
                s_m, gx_m, gy_m, dxx_m, dyy_m, dxy_m = _sample_grad_hess(
                    s, gx, gy, dxx, dyy, dxy, np.array([my]), np.array([mx])
                )
                dxx_m, dyy_m, dxy_m = dxx_m[0], dyy_m[0], dxy_m[0]
                Hm = np.array([[dyy_m, dxy_m], [dxy_m, dxx_m]], dtype=np.float64)
                # eigh: 特征值从小到大
                lam, vec = np.linalg.eigh(Hm)
                lam_min = lam[0]
                n = vec[:, 0]  # 最小特征向量：ridge 法向（在像素坐标的 (dy, dx) 顺序）

                # 在边两端点上，投影梯度到这同一个 n，检测零交叉
                s_A, gx_A, gy_A, *_ = _sample_grad_hess(s, gx, gy, dxx, dyy, dxy,
                                                         np.array([Ay]), np.array([Ax]))
                s_B, gx_B, gy_B, *_ = _sample_grad_hess(s, gx, gy, dxx, dyy, dxy,
                                                         np.array([By]), np.array([Bx]))
                gA = gx_A[0] * n[1] + gy_A[0] * n[0]
                gB = gx_B[0] * n[1] + gy_B[0] * n[0]

                if gA * gB > 0:
                    continue  # 没有符号变化

                # 线性内插求根位置
                denom = (abs(gA) + abs(gB))
                t = 0.5 if denom < 1e-14 else (abs(gA) / denom)
                iy = Ay + t * (By - Ay)
                ix = Ax + t * (Bx - Ax)

                # 在交点重新计算 Hessian 最小特征值与 s（过滤）
                s_I, gx_I, gy_I, dxx_I, dyy_I, dxy_I = _sample_grad_hess(
                    s, gx, gy, dxx, dyy, dxy, np.array([iy]), np.array([ix])
                )
                H_I = np.array([[dyy_I[0], dxy_I[0]], [dxy_I[0], dxx_I[0]]], dtype=np.float64)
                lam_I, _ = np.linalg.eigh(H_I)
                lam_min_I = lam_I[0]

                if lam_min_I >= 0:  # 必须是沿法向的负曲率（高度脊）
                    continue
                if (s_min is not None) and (s_I[0] < s_min):
                    continue
                if (lambda_max is not None) and (lam_min_I > lambda_max):
                    continue

                # 记录该边（用端点像素坐标）供 rasterize；层级标签由上层函数附加
                hits.append((Ay, Ax, By, Bx))

    return hits


# ------------ AMR 主流程（2D，逐 time-slice） ------------

def extract_ridges_2d_amr_mask(
    FTLE_field,
    max_level=3,
    neighbor_range=1,
    s_min=None,
    lambda_max=None,
    lookahead_cells=0,
    lookahead_by='height'  # 'height' or 'curvature'
):
    """
    2D Filtered AMR Ridge Extraction（返回 (T,Y,X) 的多标签 ridge mask）。
    标签为该像素最后确认的细化层级（1..max_level），0=非脊线。
    """
    data = FTLE_field.getDataAsNumpy().astype(np.float64)  # (T, Y, X)
    T, H, W = data.shape
    mask = np.zeros((T, H, W), dtype=np.int32)

    # 初始层级的抽样步长（像素单位）。假定 finest 是输入分辨率，每层细化 x2。
    base_step = 2 ** max_level

    for t in range(T):
        s = data[t]
        gx, gy, dxx, dyy, dxy = _precompute_derivatives_2d(s)

        # level 0 网格索引（像素坐标）
        grid_y = np.arange(0, H, base_step, dtype=np.int32)
        grid_x = np.arange(0, W, base_step, dtype=np.int32)
        if grid_y[-1] != H - 1:
            grid_y = np.append(grid_y, H - 1)
        if grid_x[-1] != W - 1:
            grid_x = np.append(grid_x, W - 1)

        # 每层维护：网格索引、当前层检测到的 ridge 边集合
        level_info = []
        level_info.append({'gy': grid_y, 'gx': grid_x, 'hits': []})

        # 初始检测
        hits0 = _detect_ridge_edges_on_level(s, gx, gy, dxx, dyy, dxy, grid_y, grid_x,
                                             s_min=s_min, lambda_max=lambda_max)
        level_info[0]['hits'] = hits0

        # 自 level 1 到 max_level 迭代
        for L in range(1, max_level + 1):
            # 上一层信息
            prev = level_info[L - 1]
            prev_hits = prev['hits']
            prev_gY = prev['gy']
            prev_gX = prev['gx']

            if len(prev_hits) == 0:
                # 没有脊线候选，仍然构造更密网格以保证邻域扩展逻辑一致
                step = max(1, base_step // (2 ** L))
                gY = np.arange(0, H, step, dtype=np.int32)
                gX = np.arange(0, W, step, dtype=np.int32)
                if gY[-1] != H - 1: gY = np.append(gY, H - 1)
                if gX[-1] != W - 1: gX = np.append(gX, W - 1)
                level_info.append({'gy': gY, 'gx': gX, 'hits': []})
                continue

            # —— 邻域带细化：确定需要细化的 coarse cell 范围（上一层的网格单元坐标系）——
            # 先把上一层命中的边转成“单元索引”，然后膨胀邻域。
            cell_marks = np.zeros((len(prev_gY) - 1, len(prev_gX) - 1), dtype=bool)

            def _mark_cell_by_edge(y0, x0, y1, x1):
                # 边端点属于哪个 cell：找其较小的端点落在的 cell
                # 找行列索引
                i0 = np.searchsorted(prev_gY, min(y0, y1), side='right') - 1
                j0 = np.searchsorted(prev_gX, min(x0, x1), side='right') - 1
                if 0 <= i0 < cell_marks.shape[0] and 0 <= j0 < cell_marks.shape[1]:
                    cell_marks[i0, j0] = True

            for (Ay, Ax, By, Bx) in prev_hits:
                _mark_cell_by_edge(Ay, Ax, By, Bx)

            if neighbor_range > 0:
                from scipy.ndimage import binary_dilation  # 若不想依赖 scipy，可自己写 8 邻域膨胀
                se = np.ones((2 * neighbor_range + 1, 2 * neighbor_range + 1), dtype=bool)
                cell_marks = binary_dilation(cell_marks, structure=se)

            # —— 构造当前层网格：上一层每个被标记 cell 切分为 2×2 子单元 —— 
            step = max(1, base_step // (2 ** L))
            gY = np.arange(0, H, step, dtype=np.int32)
            gX = np.arange(0, W, step, dtype=np.int32)
            if gY[-1] != H - 1: gY = np.append(gY, H - 1)
            if gX[-1] != W - 1: gX = np.append(gX, W - 1)

            # —— 在新层网格上检测 ridge 边（用一致法向 + 子像素）——
            hitsL = _detect_ridge_edges_on_level(s, gx, gy, dxx, dyy, dxy, gY, gX,
                                                 s_min=s_min, lambda_max=lambda_max)

            # —— 可选 look-ahead：对未命中但高值/高曲率的 coarse cell 再细化（简化实现：放到下一层自然覆盖）——
            # 这里不追加额外格，因为我们已经按全局 step 构建了 L 层网格。

            level_info.append({'gy': gY, 'gx': gX, 'hits': hitsL})

        # —— 将所有层的命中的边，按层级写回 mask（层级越高覆盖越前）——
        # 使用离散线段插值把边写成连续像素。
        for L in range(1, max_level + 1):  # 从粗到细，最后由细层覆盖
            hitsL = level_info[L]['hits']
            if not hitsL:
                continue
            tag = L  # 标签=层级
            for (Ay, Ax, By, Bx) in hitsL:
                # 插值绘线（保证连续）
                dy = By - Ay
                dx = Bx - Ax
                n = int(max(abs(dy), abs(dx))) + 1
                if n <= 1:
                    iy = int(np.clip(round(Ay), 0, H - 1))
                    ix = int(np.clip(round(Ax), 0, W - 1))
                    mask[t, iy, ix] = max(mask[t, iy, ix], tag)
                else:
                    ys = Ay + (np.arange(n) * dy / (n - 1))
                    xs = Ax + (np.arange(n) * dx / (n - 1))
                    ys = np.clip(np.round(ys).astype(int), 0, H - 1)
                    xs = np.clip(np.round(xs).astype(int), 0, W - 1)
                    mask[t, ys, xs] = np.maximum(mask[t, ys, xs], tag)

    return mask