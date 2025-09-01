import numpy as np
import torch
from .VectorField2d import UnsteadyVectorField2D
from .VectorField3d import UnsteadyVectorField3D
import importlib


def pathline_integration_one_direction_2D(
    vectorField: UnsteadyVectorField2D,
    start_pos:np.ndarray[np.float32,3],
    timeStart:float,
    timeEnd:float,
    stepSize:float=0.01,
    maxIterations:int=5000,
    NumericalMethod:str="RK4"
):
    """
    Integrate a pathline in one direction through an unsteady vector field.
    Args:
        vectorField: UnsteadyVectorField2D 
        start_pos: [x, y,z=0] initial position
        timeStart: float, start time
        timeEnd: float, end time
        stepSize: float, integration time step
        maxIterations: int, maximum number of steps
        NumericalMethod: str, "RK4" or "Euler"
    Returns:
        List of (position, time) tuples
    """
    pos_3d = np.array(start_pos, dtype=np.float32)
    t = timeStart
    path = [(pos_3d.copy(), t)]
    direction = 1 if timeEnd > timeStart else -1
    stepSize = abs(stepSize) * direction
    for i in range(maxIterations):
        if (direction > 0 and t >= timeEnd) or (direction < 0 and t <= timeEnd) or not vectorField.IsInside(pos_3d):
            break
        if NumericalMethod == "RK4":
            v1 = vectorField.get_vector(pos_3d[0], pos_3d[1], t)
            v2 = vectorField.get_vector(pos_3d[0] + 0.5 * stepSize * v1[0], pos_3d[1] + 0.5 * stepSize * v1[1], t + 0.5 * stepSize)
            v3 = vectorField.get_vector(pos_3d[0] + 0.5 * stepSize * v2[0], pos_3d[1] + 0.5 * stepSize * v2[1], t + 0.5 * stepSize)
            v4 = vectorField.get_vector(pos_3d[0] + stepSize * v3[0], pos_3d[1] + stepSize * v3[1], t + stepSize)
            delta = (stepSize / 6.0) * (v1 + 2 * v2 + 2 * v3 + v4)
        elif NumericalMethod == "RK5":
            k1 = vectorField.get_vector(pos_3d[0], pos_3d[1], t)
            p2 = pos_3d[:2] + stepSize * (1/4 * k1)
            k2 = vectorField.get_vector(p2[0], p2[1], t + 1/4 * stepSize)
            p3 = pos_3d[:2] + stepSize * (3/32 * k1 + 9/32 * k2)
            k3 = vectorField.get_vector(p3[0], p3[1], t + 3/8 * stepSize)
            p4 = pos_3d[:2] + stepSize * (1932/2197 * k1 - 7200/2197 * k2 + 7296/2197 * k3)
            k4 = vectorField.get_vector(p4[0], p4[1], t + 12/13 * stepSize)
            p5 = pos_3d[:2] + stepSize * (439/216 * k1 - 8 * k2 + 3680/513 * k3 - 845/4104 * k4)
            k5 = vectorField.get_vector(p5[0], p5[1], t + 1 * stepSize)
            p6 = pos_3d[:2] + stepSize * (-8/27 * k1 + 2 * k2 - 3544/2565 * k3 + 1859/4104 * k4 - 11/40 * k5)
            k6 = vectorField.get_vector(p6[0], p6[1], t + 1/2 * stepSize)
            delta = stepSize * (16/135 * k1 + 6656/12825 * k3 + 28561/56430 * k4 - 9/50 * k5 + 2/55 * k6)
        elif NumericalMethod == "Euler":
            v = vectorField.get_vector(pos_3d[0], pos_3d[1], t)
            delta = stepSize * v
        else:
            raise ValueError(f"Unknown NumericalMethod: {NumericalMethod}")
    
        delta_pos_3d = np.array([delta[0], delta[1], 0.0], dtype=np.float32)
        pos_3d = pos_3d + delta_pos_3d
        t = t + stepSize
        path.append((pos_3d.copy(), t))
    return path
    
def pathline_integration_one_direction_3D(
    vectorField: UnsteadyVectorField3D,
    start_pos:np.ndarray[np.float32,3],
    timeStart:float,
    timeEnd:float,
    stepSize:float=0.01,
    maxIterations:int=5000,
    NumericalMethod:str="RK4"):
    """
    Integrate a pathline in one direction through an unsteady vector field.
    Args:
        vectorField: UnsteadyVectorField3D 
        start_pos: [x, y,z=0] initial position
        timeStart: float, start time
        timeEnd: float, end time
        stepSize: float, integration time step
        maxIterations: int, maximum number of steps
        NumericalMethod: str, "RK4" or "Euler"
    Returns:
        List of (position, time) tuples
    """
    pos_3d = np.array(start_pos, dtype=np.float32)
    t = timeStart
    path = [(pos_3d.copy(), t)]
    direction = 1 if timeEnd > timeStart else -1
    stepSize = abs(stepSize) * direction
    for i in range(maxIterations):
        if (direction > 0 and t >= timeEnd) or (direction < 0 and t <= timeEnd) or not vectorField.IsInside(pos_3d):
            break
        if NumericalMethod == "RK4":
            v1 = vectorField.get_vector(pos_3d[0], pos_3d[1],pos_3d[2], t)
            v2 = vectorField.get_vector(pos_3d[0] + 0.5 * stepSize * v1[0], pos_3d[1] + 0.5 * stepSize * v1[1],pos_3d[2] + 0.5 * stepSize * v1[2], t + 0.5 * stepSize)
            v3 = vectorField.get_vector(pos_3d[0] + 0.5 * stepSize * v2[0], pos_3d[1] + 0.5 * stepSize * v2[1],pos_3d[2] + 0.5 * stepSize * v2[2], t + 0.5 * stepSize)
            v4 = vectorField.get_vector(pos_3d[0] + stepSize * v3[0], pos_3d[1] + stepSize * v3[1],pos_3d[2] + stepSize * v3[2], t + stepSize)
            delta = (stepSize / 6.0) * (v1 + 2 * v2 + 2 * v3 + v4)
        elif NumericalMethod == "RK5":
            k1 = vectorField.get_vector(pos_3d[0], pos_3d[1], pos_3d[2], t)
            p2 = pos_3d + stepSize * (1/4 * k1)
            k2 = vectorField.get_vector(p2[0], p2[1], p2[2], t + 1/4 * stepSize)
            p3 = pos_3d + stepSize * (3/32 * k1 + 9/32 * k2)
            k3 = vectorField.get_vector(p3[0], p3[1], p3[2], t + 3/8 * stepSize)
            p4 = pos_3d + stepSize * (1932/2197 * k1 - 7200/2197 * k2 + 7296/2197 * k3)
            k4 = vectorField.get_vector(p4[0], p4[1], p4[2], t + 12/13 * stepSize)
            p5 = pos_3d + stepSize * (439/216 * k1 - 8 * k2 + 3680/513 * k3 - 845/4104 * k4)
            k5 = vectorField.get_vector(p5[0], p5[1], p5[2], t + 1 * stepSize)
            p6 = pos_3d + stepSize * (-8/27 * k1 + 2 * k2 - 3544/2565 * k3 + 1859/4104 * k4 - 11/40 * k5)
            k6 = vectorField.get_vector(p6[0], p6[1], p6[2], t + 1/2 * stepSize)
            delta = stepSize * (16/135 * k1 + 6656/12825 * k3 + 28561/56430 * k4 - 9/50 * k5 + 2/55 * k6)
        elif NumericalMethod == "Euler":
            v = vectorField.get_vector(pos_3d[0], pos_3d[1],pos_3d[2], t)
            delta = stepSize * v
        else:
            raise ValueError(f"Unknown NumericalMethod: {NumericalMethod}")
    
        delta_pos_3d = np.array([delta[0], delta[1], delta[2]], dtype=np.float32)
        pos_3d = pos_3d + delta_pos_3d
        t = t + stepSize
        path.append((pos_3d.copy(), t))
    return path


def streamline_integration_one_direction_2D(
    vectorField: UnsteadyVectorField2D,
    start_pos,
    time,
    stepSize=0.01,
    maxIterations=5000,
    NumericalMethod="RK4",
    direction="forward"
):
    """
    Integrate a streamline in one direction through an unsteady vector field at a fixed time.
    Args:
        vectorField: UnsteadyVectorField2D instance
        start_pos: [x, y] initial position
        time: float, fixed time for streamline
        stepSize: float, integration step size
        maxIterations: int, maximum number of steps
        NumericalMethod: str, "RK4" or "Euler"
        direction: str, "forward" or "backward"
    Returns:
        List of (position, time) tuples
    """
    pos = np.array(start_pos, dtype=np.float32)
    path = [(pos.copy(), time)]
    step_direction = 1 if direction == "forward" else -1
    abs_step_size = abs(stepSize) * step_direction
    
    for i in range(maxIterations):
        if NumericalMethod == "RK4":
            v1 = vectorField.get_vector(pos[0], pos[1], time)
            v2 = vectorField.get_vector(pos[0] + 0.5 * abs_step_size * v1[0], pos[1] + 0.5 * abs_step_size * v1[1], time)
            v3 = vectorField.get_vector(pos[0] + 0.5 * abs_step_size * v2[0], pos[1] + 0.5 * abs_step_size * v2[1], time)
            v4 = vectorField.get_vector(pos[0] + abs_step_size * v3[0], pos[1] + abs_step_size * v3[1], time)
            delta = (abs_step_size / 6.0) * (v1 + 2 * v2 + 2 * v3 + v4)
        elif NumericalMethod == "RK5":
            t = time
            k1 = vectorField.get_vector(pos[0], pos[1], t)
            p2 = pos[:2] + abs_step_size * (1/4 * k1)
            k2 = vectorField.get_vector(p2[0], p2[1], t)
            p3 = pos[:2] + abs_step_size * (3/32 * k1 + 9/32 * k2)
            k3 = vectorField.get_vector(p3[0], p3[1], t)
            p4 = pos[:2] + abs_step_size * (1932/2197 * k1 - 7200/2197 * k2 + 7296/2197 * k3)
            k4 = vectorField.get_vector(p4[0], p4[1], t)
            p5 = pos[:2] + abs_step_size * (439/216 * k1 - 8 * k2 + 3680/513 * k3 - 845/4104 * k4)
            k5 = vectorField.get_vector(p5[0], p5[1], t)
            p6 = pos[:2] + abs_step_size * (-8/27 * k1 + 2 * k2 - 3544/2565 * k3 + 1859/4104 * k4 - 11/40 * k5)
            k6 = vectorField.get_vector(p6[0], p6[1], t)
            delta = abs_step_size * (16/135 * k1 + 6656/12825 * k3 + 28561/56430 * k4 - 9/50 * k5 + 2/55 * k6)
        elif NumericalMethod == "Euler":
            v = vectorField.get_vector(pos[0], pos[1], time)
            delta = abs_step_size * v
        else:
            raise ValueError(f"Unknown NumericalMethod: {NumericalMethod}")
        
        pos[0] = pos[0] + delta[0]
        pos[1] = pos[1] + delta[1]
        path.append((pos.copy(), time))
        
        # Check if we've gone out of bounds (optional safety check)
        if (pos[0] < vectorField.domainMinBoundary[0] or pos[0] > vectorField.domainMaxBoundary[0] or
            pos[1] < vectorField.domainMinBoundary[1] or pos[1] > vectorField.domainMaxBoundary[1]):
            break
    
    return path

def streamline_integration_one_direction_3D(
    vectorField: UnsteadyVectorField3D,
    start_pos,
    time,
    stepSize=0.01,
    maxIterations=5000,
    NumericalMethod="RK4",
    direction="forward"
):
    """
    Integrate a streamline in one direction through an unsteady 3D vector field at a fixed time.
    """
    pos_3d = np.array(start_pos, dtype=np.float32)
    path = [(pos_3d.copy(), time)]
    step_direction = 1 if direction == "forward" else -1
    abs_step_size = abs(stepSize) * step_direction

    for i in range(maxIterations):
        if not vectorField.IsInside(pos_3d):
            break

        if NumericalMethod == "RK4":
            v1 = vectorField.get_vector(pos_3d[0], pos_3d[1], pos_3d[2], time)
            v2_pos = pos_3d + 0.5 * abs_step_size * v1
            v2 = vectorField.get_vector(v2_pos[0], v2_pos[1], v2_pos[2], time)
            v3_pos = pos_3d + 0.5 * abs_step_size * v2
            v3 = vectorField.get_vector(v3_pos[0], v3_pos[1], v3_pos[2], time)
            v4_pos = pos_3d + abs_step_size * v3
            v4 = vectorField.get_vector(v4_pos[0], v4_pos[1], v4_pos[2], time)
            delta = (abs_step_size / 6.0) * (v1 + 2 * v2 + 2 * v3 + v4)
        elif NumericalMethod == "RK5":
            t = time
            k1 = vectorField.get_vector(pos_3d[0], pos_3d[1], pos_3d[2], t)
            p2 = pos_3d + abs_step_size * (1/4 * k1)
            k2 = vectorField.get_vector(p2[0], p2[1], p2[2], t)
            p3 = pos_3d + abs_step_size * (3/32 * k1 + 9/32 * k2)
            k3 = vectorField.get_vector(p3[0], p3[1], p3[2], t)
            p4 = pos_3d + abs_step_size * (1932/2197 * k1 - 7200/2197 * k2 + 7296/2197 * k3)
            k4 = vectorField.get_vector(p4[0], p4[1], p4[2], t)
            p5 = pos_3d + abs_step_size * (439/216 * k1 - 8 * k2 + 3680/513 * k3 - 845/4104 * k4)
            k5 = vectorField.get_vector(p5[0], p5[1], p5[2], t)
            p6 = pos_3d + abs_step_size * (-8/27 * k1 + 2 * k2 - 3544/2565 * k3 + 1859/4104 * k4 - 11/40 * k5)
            k6 = vectorField.get_vector(p6[0], p6[1], p6[2], t)
            delta = abs_step_size * (16/135 * k1 + 6656/12825 * k3 + 28561/56430 * k4 - 9/50 * k5 + 2/55 * k6)
        elif NumericalMethod == "Euler":
            v = vectorField.get_vector(pos_3d[0], pos_3d[1], pos_3d[2], time)
            delta = abs_step_size * v
        else:
            raise ValueError(f"Unknown NumericalMethod: {NumericalMethod}")

        pos_3d += delta
        path.append((pos_3d.copy(), time))

    return path


def compute_pathline_2D(args):
        vector_field, pos3d, t0, min_time, max_time, step_size, max_iteration, method = args
        forward = pathline_integration_one_direction_2D(vector_field, pos3d, t0, max_time, step_size, max_iteration, method)
        backward = pathline_integration_one_direction_2D(vector_field, pos3d, t0, min_time, step_size, max_iteration, method)
        backward = backward[::-1]
        full_path = backward + forward[1:]
        return full_path

def compute_pathline_3D(args):
    vector_field, pos3d, t0, min_time, max_time, step_size, max_iteration, method = args
    forward = pathline_integration_one_direction_3D(vector_field, pos3d, t0, max_time, step_size, max_iteration, method)
    backward = pathline_integration_one_direction_3D(vector_field, pos3d, t0, min_time, step_size, max_iteration, method)
    backward = backward[::-1]
    full_path = backward + forward[1:]
    return full_path

def compute_streamline_3D(args):
    vector_field, pos3d, time, step_size, max_iteration, method = args
    forward = streamline_integration_one_direction_3D(vector_field, pos3d, time, step_size, max_iteration, method, "forward")
    backward = streamline_integration_one_direction_3D(vector_field, pos3d, time, step_size, max_iteration, method, "backward")
    backward = backward[::-1]
    full_path = backward + forward[1:]
    return full_path

def compute_streamline_2D(args):
    vector_field, pos3d, time, step_size, max_iteration, method = args
    forward = streamline_integration_one_direction_2D(vector_field, pos3d, time, step_size, max_iteration, method, "forward")
    backward = streamline_integration_one_direction_2D(vector_field, pos3d, time, step_size, max_iteration, method, "backward")
    backward = backward[::-1]
    full_path = backward + forward[1:]
    return full_path


def batch_pathline_integration_2D_cuda(points:np.ndarray,vectorfield:UnsteadyVectorField2D,t_target:float,dt:float,max_steps:int,method:str):
    """
    Batch integrate pathlines in 2D unsteady field using CUDA (RK4).

    Args:
        points: np.ndarray, shape [N,3], physical coordinates (x,y,t)
        vectorfield: UnsteadyVectorField2D
        t_target: target physical time (direction depends on t_target vs t_start and sign of dt), t_start is the seeding time
        dt: single-step time interval (positive; direction handled internally)
        max_steps: maximum number of steps (including step0; output valid length in [1..max_steps])
        method: integration method, "RK4" or "Euler"

    Returns:
        positions: np.ndarray [N, max_steps, 3], (x,y,t)
        valid_steps: np.ndarray [N], valid point count of each line (including step0)
    """
    methodId=None
    if method.lower() not in ["rk4","euler"]:
        raise ValueError(f"Unknown NumericalMethod: {method}")
    elif method.lower() == "euler":
        methodId=0
    elif method.lower() == "rk4":
        methodId=1

    assert points.ndim == 2 and points.shape[1] == 3, "points must be [N,3]"
    data = vectorfield.getDataAsNumpy()  # (T,H,W,2)
    assert data is not None and data.ndim == 4 and data.shape[-1] == 2

    T, H, W, _ = data.shape
    dx = float(vectorfield.gridInterval[0])
    dy = float(vectorfield.gridInterval[1])
    vdt = float(vectorfield.timeInterval if vectorfield.time_steps > 1 else 1.0)

    # Split components and ensure contiguous memory
    field_u = np.ascontiguousarray(data[..., 0].astype(np.float32))
    field_v = np.ascontiguousarray(data[..., 1].astype(np.float32))

    # Relative coordinates/time (kernel uses [0,(W-1)*dx]×[0,(H-1)*dy], t_rel=t-tmin)
    dom_min = vectorfield.domainMinBoundary
    x0 = np.ascontiguousarray((points[:, 0] - float(dom_min[0])).astype(np.float64))
    y0 = np.ascontiguousarray((points[:, 1] - float(dom_min[1])).astype(np.float64))
    t0_rel = np.ascontiguousarray(points[:, 2] - float(vectorfield.tmin), dtype=np.float64)
    t_target= np.clip(t_target, float(vectorfield.tmin), float(vectorfield.tmax))
    t_target_rel = float(t_target - vectorfield.tmin)

    # 输出缓冲
    out_pos = np.zeros((points.shape[0], max_steps , 3), dtype=np.float32)
    out_valid = np.zeros((points.shape[0],), dtype=np.int32)

    # Compile/load kernel
    module = _get_or_compile_pathline2d_cuda_kernel()
    kernel = module.get_function("integrate_pathlines_2d_kernel")
    cuda = importlib.import_module('pycuda.driver')

    # Device buffers
    u_gpu = cuda.mem_alloc(field_u.nbytes)
    v_gpu = cuda.mem_alloc(field_v.nbytes)
    x_gpu = cuda.mem_alloc(x0.nbytes)
    y_gpu = cuda.mem_alloc(y0.nbytes)
    t0_gpu = cuda.mem_alloc(t0_rel.nbytes)
    outP_gpu = cuda.mem_alloc(out_pos.nbytes)
    outV_gpu = cuda.mem_alloc(out_valid.nbytes)

    cuda.memcpy_htod(u_gpu, field_u)
    cuda.memcpy_htod(v_gpu, field_v)
    cuda.memcpy_htod(x_gpu, x0)
    cuda.memcpy_htod(y_gpu, y0)
    cuda.memcpy_htod(t0_gpu, t0_rel)
    cuda.memcpy_htod(outP_gpu, out_pos)
    cuda.memcpy_htod(outV_gpu, out_valid)

    # Launch configuration
    threads = 64
    blocks = (points.shape[0] + threads - 1) // threads

    kernel(
        u_gpu, v_gpu,
        np.int32(W), np.int32(H), np.int32(T), np.float64(dx), np.float64(dy), np.float64(vdt),
        x_gpu, y_gpu, t0_gpu,
        np.float64(t_target_rel),
        np.float64(float(dt)),
        np.int32(int(max_steps)),
        np.int32(int(points.shape[0])),
        np.float64(float(vectorfield.tmin)),
        np.float64(float(dom_min[0])), np.float64(float(dom_min[1])),
        outP_gpu,
        outV_gpu,
        np.int32(methodId),
        block=(threads, 1, 1), grid=(blocks, 1, 1)
    )

    # Copy back results
    cuda.memcpy_dtoh(out_pos, outP_gpu)
    cuda.memcpy_dtoh(out_valid, outV_gpu)

    # Free buffers
    u_gpu.free(); v_gpu.free(); x_gpu.free(); y_gpu.free(); t0_gpu.free(); outP_gpu.free(); outV_gpu.free()

    return out_pos, out_valid


def compute_pathline_2D_cuda(list_args):
    """
    Single pathline (bidirectional concatenation) using CUDA; interface matches compute_pathline_2D.
    args = (vector_field, pos3d, t0, min_time, max_time, step_size, max_iteration, method)
    Returns [(pos3d, t), ...]
    """
    vector_field, pos3d, t0, min_time, max_time, step_size, max_iteration, method = list_args[0]

    # Build (N,3) tensor of seeds
    batch_seeding_point_pos_time = []
    for arg in list_args:
        _, pos3d, t0, _, _, _, _, _ = arg
        batch_seeding_point_pos_time.append([float(pos3d[0]), float(pos3d[1]), float(t0)])
    batch_seeding_point_pos_time = np.array(batch_seeding_point_pos_time, dtype=np.float64)  # (N, 3)
    # Forward integration
    Pathline_f, pathlineLength_f = batch_pathline_integration_2D_cuda(
        points=batch_seeding_point_pos_time,
        vectorfield=vector_field,
        t_target=float(max_time),
        dt=float(step_size),
        max_steps=int(max_iteration),
        method=method
    )
    # Backward integration
    Pathline_b, pathlineLength_b = batch_pathline_integration_2D_cuda(
        points=batch_seeding_point_pos_time,
        vectorfield=vector_field,
        t_target=float(min_time),
        dt=float(step_size),
        max_steps=int(max_iteration),
        method=method
    )

    # Use vectorized slicing to handle variable-length lines
    B, S, _ = Pathline_f.shape

    def _np_array_to_path(arr: np.ndarray):
        if arr.size == 0:
            return []
        K = arr.shape[0]
        pos3d = np.zeros((K, 3), dtype=np.float32)
        pos3d[:, :2] = arr[:, :2].astype(np.float32)
        t_col = arr[:, 2]
        return list(zip(pos3d, t_col.tolist()))
    
    results = []
    for line_idx in range(B):
        kb = int(pathlineLength_b[line_idx])
        kf = int(pathlineLength_f[line_idx])
        bwd_arr = Pathline_b[line_idx, :kb, :][::-1]
        fwd_arr = Pathline_f[line_idx, 1:kf, :] if kf > 1 else Pathline_f[line_idx, 0:0, :]
        full_arr = np.concatenate([bwd_arr, fwd_arr], axis=0) if bwd_arr.size or fwd_arr.size else Pathline_f[line_idx, 0:0, :]
        results.append(_np_array_to_path(full_arr))

    return results

# ---------------- Internal: CUDA module cache ----------------
_PATHLINE2D_CUDA_MODULE = None
def _get_or_compile_pathline2d_cuda_kernel():
    global _PATHLINE2D_CUDA_MODULE
    if _PATHLINE2D_CUDA_MODULE is not None:
        return _PATHLINE2D_CUDA_MODULE
    with open("assets/cuda_kernal/PathlineIntegration2D.cu", "r") as f:
        src = f.read()
    try:
        # Lazy import and initialize CUDA context
        importlib.import_module('pycuda.autoinit')
        SourceModule = importlib.import_module('pycuda.compiler').SourceModule
        _PATHLINE2D_CUDA_MODULE = SourceModule(src, options=["-O3", "-use_fast_math"])
        print("✅ PathlineIntegration2D CUDA kernel compiled successfully.")
    except Exception as e:
        print(f"❌ Unexpected CUDA error: {e}")
        raise
    return _PATHLINE2D_CUDA_MODULE

# ---------------- Backend binder and auto-dispatch ----------------
USE_CUDA_PATHLINE = None  # None: unknown; True/False: decided

def Bind_Flowline_IntegrationBackend():
    """
    Bind the backend for pathline integration:
    - If pycuda is unavailable or compilation fails, use CPU implementation.
    - If compilation succeeds, use CUDA implementation (compute_pathline_2D_cuda and batch_pathline_integration_2D_cuda).
    Returns True/False indicating whether CUDA backend is enabled.
    """
    global USE_CUDA_PATHLINE
    if USE_CUDA_PATHLINE is not None:
        return USE_CUDA_PATHLINE
    # Check pycuda modules
    try:
        importlib.import_module('pycuda.driver')
        importlib.import_module('pycuda.compiler')
        importlib.import_module('pycuda.autoinit')  # 初始化上下文（若可用）
    except Exception:
        USE_CUDA_PATHLINE = False
        print("[BindFlowlineBackend] pycuda not available. Fallback to CPU integration.")
        return USE_CUDA_PATHLINE

    # Try compile kernel
    try:
        mod = _get_or_compile_pathline2d_cuda_kernel()
        USE_CUDA_PATHLINE = (mod is not None)
    except Exception as e:
        print(f"[BindFlowlineBackend] CUDA kernel compile failed: {e}. Fallback to CPU.")
        USE_CUDA_PATHLINE = False
    return USE_CUDA_PATHLINE


def compute_pathline_2D_auto(list_args):
    """Automatically choose CUDA or CPU 2D pathline computation."""
    args0 = list_args[0]
    method = args0[-1]
    use_cuda = Bind_Flowline_IntegrationBackend() and method.lower() in ["rk4","euler"]
    if use_cuda:
        try:
            return compute_pathline_2D_cuda(list_args)
        except Exception as e:
            print(f"[compute_pathline_2D_auto] CUDA failed: {e}. Fallback to CPU.")

    return list(map(compute_pathline_2D, list_args))


def batch_pathlineCross_integration_2D_auto(points:np.ndarray,vectorfield:UnsteadyVectorField2D,t_start:float,t_target:float,dt:float,max_steps:int,offsets_size:float,method:str="rk4"):
    """
    Automatically choose CUDA or CPU for batch pathline integration.
    - Expand cross offsets (center, x±, y±), total lines M=N*5.
    - CPU fallback computes per-line (slower).
    Return torch.Tensors (positions[M, S, 3], valid_steps[M]) consistent with CUDA version.
    """
    use_cuda = Bind_Flowline_IntegrationBackend()
    # Expand seeds to cross (center, x±, y±)
    if points.size == 0:
        return torch.empty(0, max_steps, 3), torch.empty(0, dtype=torch.int32)
    offs = np.array([[0.0, 0.0], [offsets_size, 0.0], [-offsets_size, 0.0], [0.0, offsets_size], [0.0, -offsets_size]], dtype=np.float64)
    seeds = (points[:, None, :] + offs[None, :, :]).reshape(-1, 2)
    seeds = np.concatenate([seeds, np.ones((seeds.shape[0], 1)) * t_start], axis=1)

    if use_cuda and method.lower() in ["rk4", "euler"]:
        try:
            pos_np, val_np = batch_pathline_integration_2D_cuda(seeds, vectorfield, t_target, dt, max_steps, method)
            return torch.from_numpy(pos_np).float(), torch.from_numpy(val_np).to(torch.int32)
        except Exception as e:
            print(f"[batch_pathline_integration_2D_auto] CUDA failed: {e}. Fallback to CPU.")

    # CPU fallback
    M = seeds.shape[0]
    out_pos = np.zeros((M, max_steps , 3), dtype=np.float32)
    out_valid = np.zeros((M,), dtype=np.int32)

    # Per-line CPU computation (forward only here)
    for i in range(M):
        pos3d = np.array([float(seeds[i,0]), float(seeds[i,1]), 0.0], dtype=np.float32)
        forward = pathline_integration_one_direction_2D(vectorfield, pos3d, float(t_start), float(t_target), float(dt), int(max_steps), method.upper())
        # backward = pathline_integration_one_direction_2D(vectorfield, pos3d, float(t_start), float(t_target), float(dt), int(max_steps), method.upper())
        # backward = backward[::-1]
        # full_path = backward + forward[1:]
        full_path = forward

        L = min(len(full_path), max_steps)
        for k in range(L):
            p3, t = full_path[k]
            out_pos[i, k, 0] = p3[0]
            out_pos[i, k, 1] = p3[1]
            out_pos[i, k, 2] = t
        out_valid[i] = L

    return torch.from_numpy(out_pos).float(), torch.from_numpy(out_valid).to(torch.int32)



