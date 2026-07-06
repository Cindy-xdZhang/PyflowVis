import numpy as np
import torch
from numba import njit, prange
from .VectorField2d import UnsteadyVectorField2D
from .VectorField3d import UnsteadyVectorField3D
import importlib
import logging
from contextlib import contextmanager


# =====================================================================================
# Numba parallel batch integrators for 3D flowlines (Phase 2).
#
# The serial `compute_streamline_3D` / `compute_pathline_3D` above integrate one seed at
# a time in pure Python (RK4 loop + list.append), which is the 3D interactivity killer.
# The kernels below integrate ALL seeds in parallel (prange over seeds) with the same
# trilinear/quadrilinear interpolation and the same RK4/Euler steps, writing straight
# into a padded (N, maxLen, 4) buffer that the renderer turns into a VBO with numpy
# (no per-vertex Python). Results match the serial path to float rounding (see test).
# Only RK4/Euler are implemented here; other methods fall back to the serial path.
# =====================================================================================

@njit(fastmath=True, cache=True)
def _interp3_trilinear(vol, dmin, ginv, xdim, ydim, zdim, x, y, z):
    """Trilinear sample of a (Z,Y,X,3) volume; clamping matches _quadrilinear_interpolate_njit."""
    gx = (x - dmin[0]) / ginv[0] if ginv[0] != 0.0 else 0.0
    gy = (y - dmin[1]) / ginv[1] if ginv[1] != 0.0 else 0.0
    gz = (z - dmin[2]) / ginv[2] if ginv[2] != 0.0 else 0.0
    x0 = int(np.floor(gx)); x1 = int(np.ceil(gx))
    y0 = int(np.floor(gy)); y1 = int(np.ceil(gy))
    z0 = int(np.floor(gz)); z1 = int(np.ceil(gz))
    if x0 < 0: x0 = 0
    elif x0 > xdim - 1: x0 = xdim - 1
    if x1 < 0: x1 = 0
    elif x1 > xdim - 1: x1 = xdim - 1
    if y0 < 0: y0 = 0
    elif y0 > ydim - 1: y0 = ydim - 1
    if y1 < 0: y1 = 0
    elif y1 > ydim - 1: y1 = ydim - 1
    if z0 < 0: z0 = 0
    elif z0 > zdim - 1: z0 = zdim - 1
    if z1 < 0: z1 = 0
    elif z1 > zdim - 1: z1 = zdim - 1
    wx = gx - x0; wy = gy - y0; wz = gz - z0
    c00 = vol[z0, y0, x0] * (1.0 - wx) + vol[z0, y0, x1] * wx
    c10 = vol[z0, y1, x0] * (1.0 - wx) + vol[z0, y1, x1] * wx
    c01 = vol[z1, y0, x0] * (1.0 - wx) + vol[z1, y0, x1] * wx
    c11 = vol[z1, y1, x0] * (1.0 - wx) + vol[z1, y1, x1] * wx
    c0 = c00 * (1.0 - wy) + c10 * wy
    c1 = c01 * (1.0 - wy) + c11 * wy
    r = c0 * (1.0 - wz) + c1 * wz
    return r[0], r[1], r[2]


@njit(fastmath=True, cache=True)
def _interp4_quadrilinear(field, dmin, ginv, xdim, ydim, zdim, tmin, tinv, tsteps, x, y, z, t):
    """Quadrilinear sample of a (T,Z,Y,X,3) field: trilinear in space, linear in time."""
    if tinv != 0.0:
        tg = (t - tmin) / tinv
    else:
        tg = 0.0
    t0 = int(np.floor(tg)); t1 = int(np.ceil(tg))
    if t0 < 0: t0 = 0
    elif t0 > tsteps - 1: t0 = tsteps - 1
    if t1 < 0: t1 = 0
    elif t1 > tsteps - 1: t1 = tsteps - 1
    wt = tg - t0
    ax, ay, az = _interp3_trilinear(field[t0], dmin, ginv, xdim, ydim, zdim, x, y, z)
    bx, by, bz = _interp3_trilinear(field[t1], dmin, ginv, xdim, ydim, zdim, x, y, z)
    return ax * (1.0 - wt) + bx * wt, ay * (1.0 - wt) + by * wt, az * (1.0 - wt) + bz * wt


@njit(fastmath=True, cache=True)
def _stream_one_dir_3d(vol, dmin, dmax, ginv, xdim, ydim, zdim, sx, sy, sz, signed_step, max_iter, method_id, buf):
    """Integrate one streamline direction at a fixed time. Fills buf[:count] with stepped
    positions (excluding the seed) and returns count. method_id: 1=RK4, else Euler."""
    px = sx; py = sy; pz = sz
    count = 0
    for _ in range(max_iter):
        if px < dmin[0] or px > dmax[0] or py < dmin[1] or py > dmax[1] or pz < dmin[2] or pz > dmax[2]:
            break
        if method_id == 1:
            v1x, v1y, v1z = _interp3_trilinear(vol, dmin, ginv, xdim, ydim, zdim, px, py, pz)
            v2x, v2y, v2z = _interp3_trilinear(vol, dmin, ginv, xdim, ydim, zdim, px + 0.5 * signed_step * v1x, py + 0.5 * signed_step * v1y, pz + 0.5 * signed_step * v1z)
            v3x, v3y, v3z = _interp3_trilinear(vol, dmin, ginv, xdim, ydim, zdim, px + 0.5 * signed_step * v2x, py + 0.5 * signed_step * v2y, pz + 0.5 * signed_step * v2z)
            v4x, v4y, v4z = _interp3_trilinear(vol, dmin, ginv, xdim, ydim, zdim, px + signed_step * v3x, py + signed_step * v3y, pz + signed_step * v3z)
            dx = (signed_step / 6.0) * (v1x + 2.0 * v2x + 2.0 * v3x + v4x)
            dy = (signed_step / 6.0) * (v1y + 2.0 * v2y + 2.0 * v3y + v4y)
            dz = (signed_step / 6.0) * (v1z + 2.0 * v2z + 2.0 * v3z + v4z)
        else:
            v1x, v1y, v1z = _interp3_trilinear(vol, dmin, ginv, xdim, ydim, zdim, px, py, pz)
            dx = signed_step * v1x; dy = signed_step * v1y; dz = signed_step * v1z
        px += dx; py += dy; pz += dz
        buf[count, 0] = px; buf[count, 1] = py; buf[count, 2] = pz
        count += 1
    return count


@njit(parallel=True, fastmath=True, cache=True)
def _batch_stream_3d(vol, dmin, dmax, ginv, xdim, ydim, zdim, seeds, abs_step, max_iter, method_id, time_val, out_pos, out_len):
    """out_pos: (N, 2*max_iter+1, 4)=(x,y,z,t); out_len: (N,). Mirrors compute_streamline_3D."""
    N = seeds.shape[0]
    for s in prange(N):
        fwd = np.empty((max_iter, 3), np.float32)
        bwd = np.empty((max_iter, 3), np.float32)
        sx = seeds[s, 0]; sy = seeds[s, 1]; sz = seeds[s, 2]
        kf = _stream_one_dir_3d(vol, dmin, dmax, ginv, xdim, ydim, zdim, sx, sy, sz, abs_step, max_iter, method_id, fwd)
        kb = _stream_one_dir_3d(vol, dmin, dmax, ginv, xdim, ydim, zdim, sx, sy, sz, -abs_step, max_iter, method_id, bwd)
        idx = 0
        for j in range(kb - 1, -1, -1):
            out_pos[s, idx, 0] = bwd[j, 0]; out_pos[s, idx, 1] = bwd[j, 1]; out_pos[s, idx, 2] = bwd[j, 2]; out_pos[s, idx, 3] = time_val
            idx += 1
        out_pos[s, idx, 0] = sx; out_pos[s, idx, 1] = sy; out_pos[s, idx, 2] = sz; out_pos[s, idx, 3] = time_val
        idx += 1
        for j in range(kf):
            out_pos[s, idx, 0] = fwd[j, 0]; out_pos[s, idx, 1] = fwd[j, 1]; out_pos[s, idx, 2] = fwd[j, 2]; out_pos[s, idx, 3] = time_val
            idx += 1
        out_len[s] = idx


@njit(fastmath=True, cache=True)
def _path_one_dir_3d(field, dmin, dmax, ginv, xdim, ydim, zdim, tmin, tinv, tsteps, sx, sy, sz, t_start, t_end, abs_step, max_iter, method_id, buf):
    """Integrate one pathline direction (time-dependent). Fills buf[:count] with stepped
    (x,y,z,t) (excluding the seed) and returns count. Mirrors pathline_integration_one_direction_3D."""
    # Time is accumulated in float64 (like the serial reference) so the t>=t_end stop
    # boundary is hit on the exact same step; positions accumulate in float32 (also like serial).
    direction = 1.0 if t_end > t_start else -1.0
    step = np.float64(abs_step) * direction
    px = sx; py = sy; pz = sz
    t = np.float64(t_start)
    te = np.float64(t_end)
    count = 0
    for _ in range(max_iter):
        if (direction > 0.0 and t >= te) or (direction < 0.0 and t <= te):
            break
        if px < dmin[0] or px > dmax[0] or py < dmin[1] or py > dmax[1] or pz < dmin[2] or pz > dmax[2]:
            break
        if method_id == 1:
            v1x, v1y, v1z = _interp4_quadrilinear(field, dmin, ginv, xdim, ydim, zdim, tmin, tinv, tsteps, px, py, pz, t)
            v2x, v2y, v2z = _interp4_quadrilinear(field, dmin, ginv, xdim, ydim, zdim, tmin, tinv, tsteps, px + 0.5 * step * v1x, py + 0.5 * step * v1y, pz + 0.5 * step * v1z, t + 0.5 * step)
            v3x, v3y, v3z = _interp4_quadrilinear(field, dmin, ginv, xdim, ydim, zdim, tmin, tinv, tsteps, px + 0.5 * step * v2x, py + 0.5 * step * v2y, pz + 0.5 * step * v2z, t + 0.5 * step)
            v4x, v4y, v4z = _interp4_quadrilinear(field, dmin, ginv, xdim, ydim, zdim, tmin, tinv, tsteps, px + step * v3x, py + step * v3y, pz + step * v3z, t + step)
            dx = (step / 6.0) * (v1x + 2.0 * v2x + 2.0 * v3x + v4x)
            dy = (step / 6.0) * (v1y + 2.0 * v2y + 2.0 * v3y + v4y)
            dz = (step / 6.0) * (v1z + 2.0 * v2z + 2.0 * v3z + v4z)
        else:
            v1x, v1y, v1z = _interp4_quadrilinear(field, dmin, ginv, xdim, ydim, zdim, tmin, tinv, tsteps, px, py, pz, t)
            dx = step * v1x; dy = step * v1y; dz = step * v1z
        px = px + np.float32(dx); py = py + np.float32(dy); pz = pz + np.float32(dz)
        t = t + step
        buf[count, 0] = px; buf[count, 1] = py; buf[count, 2] = pz; buf[count, 3] = t
        count += 1
    return count


@njit(parallel=True, fastmath=True, cache=True)
def _batch_path_3d(field, dmin, dmax, ginv, xdim, ydim, zdim, tmin, tinv, tsteps, seeds, min_time, max_time, abs_step, max_iter, method_id, out_pos, out_len):
    """out_pos: (N, 2*max_iter+1, 4)=(x,y,z,t); out_len:(N,). seeds:(N,4)=(x,y,z,t0). Mirrors compute_pathline_3D."""
    N = seeds.shape[0]
    for s in prange(N):
        fwd = np.empty((max_iter, 4), np.float32)
        bwd = np.empty((max_iter, 4), np.float32)
        # seeds is float64 (time parity with the serial reference); positions run in float32.
        sx = np.float32(seeds[s, 0]); sy = np.float32(seeds[s, 1]); sz = np.float32(seeds[s, 2]); t0 = seeds[s, 3]
        kf = _path_one_dir_3d(field, dmin, dmax, ginv, xdim, ydim, zdim, tmin, tinv, tsteps, sx, sy, sz, t0, max_time, abs_step, max_iter, method_id, fwd)
        kb = _path_one_dir_3d(field, dmin, dmax, ginv, xdim, ydim, zdim, tmin, tinv, tsteps, sx, sy, sz, t0, min_time, abs_step, max_iter, method_id, bwd)
        idx = 0
        for j in range(kb - 1, -1, -1):
            out_pos[s, idx, 0] = bwd[j, 0]; out_pos[s, idx, 1] = bwd[j, 1]; out_pos[s, idx, 2] = bwd[j, 2]; out_pos[s, idx, 3] = bwd[j, 3]
            idx += 1
        out_pos[s, idx, 0] = sx; out_pos[s, idx, 1] = sy; out_pos[s, idx, 2] = sz; out_pos[s, idx, 3] = t0
        idx += 1
        for j in range(kf):
            out_pos[s, idx, 0] = fwd[j, 0]; out_pos[s, idx, 1] = fwd[j, 1]; out_pos[s, idx, 2] = fwd[j, 2]; out_pos[s, idx, 3] = fwd[j, 3]
            idx += 1
        out_len[s] = idx


def _method_id_or_none(method: str):
    """Map an integrator name to the numba kernel id (1=RK4, 0=Euler); None if unsupported."""
    m = str(method).upper()
    if m == "RK4":
        return 1
    if m == "EULER":
        return 0
    return None


def _field3d_as_numpy(vector_field: UnsteadyVectorField3D):
    field = vector_field.field
    if field is None:
        return None
    if isinstance(field, torch.Tensor):
        field = field.detach().cpu().numpy()
    return np.ascontiguousarray(field, dtype=np.float32)


def compute_streamlines_3D_batch(vector_field: UnsteadyVectorField3D, seeds_xyz: np.ndarray, time: float,
                                 step_size: float, max_iteration: int, method: str):
    """Parallel batch streamline integration at a fixed time.

    Returns (out_pos (N, 2*max_iter+1, 4)=(x,y,z,t), out_len (N,)) or None to signal the
    caller to fall back to the serial per-seed path (unsupported method / empty input)."""
    method_id = _method_id_or_none(method)
    if method_id is None:
        return None
    field = _field3d_as_numpy(vector_field)
    if field is None:
        return None
    seeds = np.ascontiguousarray(np.asarray(seeds_xyz, dtype=np.float32).reshape(-1, 3))
    if seeds.shape[0] == 0:
        return None

    # Blend the two bracketing time slices into one steady snapshot (quadrilinear == snapshot+trilinear).
    T = int(vector_field.time_steps)
    tg = (time - vector_field.tmin) / vector_field.timeInterval if vector_field.timeInterval > 0 else 0.0
    t0 = max(0, min(int(np.floor(tg)), T - 1))
    t1 = max(0, min(int(np.ceil(tg)), T - 1))
    wt = np.float32(tg - t0)
    snap = ((1.0 - wt) * field[t0] + wt * field[t1]).astype(np.float32)

    dmin = np.asarray(vector_field.domainMinBoundary, dtype=np.float32)
    dmax = np.asarray(vector_field.domainMaxBoundary, dtype=np.float32)
    ginv = np.asarray(vector_field.gridInterval, dtype=np.float32)
    max_iter = int(max_iteration)
    out_pos = np.zeros((seeds.shape[0], 2 * max_iter + 1, 4), dtype=np.float32)
    out_len = np.zeros((seeds.shape[0],), dtype=np.int32)
    _batch_stream_3d(snap, dmin, dmax, ginv, int(vector_field.Xdim), int(vector_field.Ydim), int(vector_field.Zdim),
                     seeds, np.float32(abs(step_size)), max_iter, method_id, np.float32(time), out_pos, out_len)
    return out_pos, out_len


def compute_pathlines_3D_batch(vector_field: UnsteadyVectorField3D, seeds_xyzt: np.ndarray, min_time: float,
                              max_time: float, step_size: float, max_iteration: int, method: str):
    """Parallel batch pathline integration (bidirectional in time).

    seeds_xyzt: (N,4)=(x,y,z,t0). Returns (out_pos, out_len) or None for serial fallback."""
    method_id = _method_id_or_none(method)
    if method_id is None:
        return None
    field = _field3d_as_numpy(vector_field)
    if field is None:
        return None
    # float64 for all time-domain quantities (step, t0, bounds, tmin/interval) to match the
    # serial reference's float64 time accumulation exactly; positions stay float32.
    seeds = np.ascontiguousarray(np.asarray(seeds_xyzt, dtype=np.float64).reshape(-1, 4))
    if seeds.shape[0] == 0:
        return None

    dmin = np.asarray(vector_field.domainMinBoundary, dtype=np.float32)
    dmax = np.asarray(vector_field.domainMaxBoundary, dtype=np.float32)
    ginv = np.asarray(vector_field.gridInterval, dtype=np.float32)
    tinv = np.float64(vector_field.timeInterval if vector_field.time_steps > 1 else 0.0)
    max_iter = int(max_iteration)
    out_pos = np.zeros((seeds.shape[0], 2 * max_iter + 1, 4), dtype=np.float32)
    out_len = np.zeros((seeds.shape[0],), dtype=np.int32)
    _batch_path_3d(field, dmin, dmax, ginv, int(vector_field.Xdim), int(vector_field.Ydim), int(vector_field.Zdim),
                   np.float64(vector_field.tmin), tinv, int(vector_field.time_steps),
                   seeds, np.float64(min_time), np.float64(max_time), np.float64(abs(step_size)), max_iter,
                   method_id, out_pos, out_len)
    return out_pos, out_len


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
    device_index = int(torch.cuda.current_device()) if torch.cuda.is_available() else 0
    module = _get_or_compile_pathline2d_cuda_kernel(device_index)
    kernel = module.get_function("integrate_pathlines_2d_kernel")
    cuda = importlib.import_module('pycuda.driver')

    with _torch_compatible_cuda_context(device_index):
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
        cuda.Context.synchronize()
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
_PATHLINE2D_CUDA_MODULES = {}

@contextmanager
def _torch_compatible_cuda_context(device_index: int | None = None):
    cuda = importlib.import_module('pycuda.driver')
    cuda.init()
    if device_index is None:
        device_index = int(torch.cuda.current_device()) if torch.cuda.is_available() else 0
    dev = cuda.Device(int(device_index))
    ctx = dev.retain_primary_context()
    ctx.push()
    try:
        yield ctx
    finally:
        try:
            cuda.Context.synchronize()
        except Exception:
            pass
        ctx.pop()

def _get_or_compile_pathline2d_cuda_kernel(device_index: int | None = None):
    global _PATHLINE2D_CUDA_MODULES
    if device_index is None:
        device_index = int(torch.cuda.current_device()) if torch.cuda.is_available() else 0
    if device_index in _PATHLINE2D_CUDA_MODULES:
        return _PATHLINE2D_CUDA_MODULES[device_index]
    with open("assets/cuda_kernal/PathlineIntegration2D.cu", "r") as f:
        src = f.read()
    try:
        cuda = importlib.import_module('pycuda.driver')
        compiler = importlib.import_module('pycuda.compiler')
        with _torch_compatible_cuda_context(device_index):
            SourceModule = compiler.SourceModule
            module = SourceModule(src, options=["-O3", "-use_fast_math"])
        _PATHLINE2D_CUDA_MODULES[device_index] = module
        logging.info("[flowlineIntegral] CUDA kernel compiled successfully for device %d", device_index)
        return module
    except Exception as e:
        logging.error("[flowlineIntegral] CUDA kernel compile failed: %s", e)
        raise

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
    except Exception:
        USE_CUDA_PATHLINE = False
        logging.info("[BindFlowlineBackend] pycuda not available. Fallback to CPU integration.")
        return USE_CUDA_PATHLINE

    # Try compile kernel
    try:
        dev_idx = int(torch.cuda.current_device()) if torch.cuda.is_available() else 0
        mod = _get_or_compile_pathline2d_cuda_kernel(dev_idx)
        USE_CUDA_PATHLINE = (mod is not None)
    except Exception as e:
        logging.info(f"[BindFlowlineBackend] CUDA kernel compile failed: {e}. Fallback to CPU.")
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
            logging.info(f"[compute_pathline_2D_auto] CUDA failed: {e}. Fallback to CPU.")

    return list(map(compute_pathline_2D, list_args))


def batch_pathlineCross_integration_2D_auto(points:np.ndarray,vectorfield:UnsteadyVectorField2D,t_start:float,t_target:float,dt:float,max_steps:int,offsets_size:float,method:str="rk4"):
    """
    Automatically choose CUDA or CPU for batch pathline integration.
    - Expand cross offsets (center, x±, y±), total lines M=N*5.
    - CPU fallback computes per-line (slower).
    Return torch.Tensors (positions[M, S, 3], valid_steps[M]) consistent with CUDA version.
    """

    maxSeedPerGpuCall = 102400

    use_cuda = Bind_Flowline_IntegrationBackend()
    # Expand seeds to cross (center, x±, y±)
    if points.size == 0:
        return torch.empty(0, max_steps, 3), torch.empty(0, dtype=torch.int32)
    offs = np.array([[0.0, 0.0], [offsets_size, 0.0], [-offsets_size, 0.0], [0.0, offsets_size], [0.0, -offsets_size]], dtype=np.float64)
    seeds = (points[:, None, :] + offs[None, :, :]).reshape(-1, 2)
    seeds = np.concatenate([seeds, np.ones((seeds.shape[0], 1)) * t_start], axis=1)

    if use_cuda and method.lower() in ["rk4", "euler"]:
        try:
            total = seeds.shape[0]
            if total <= maxSeedPerGpuCall:
                pos_np, val_np = batch_pathline_integration_2D_cuda(seeds, vectorfield, t_target, dt, max_steps, method)
            else:
                # chunked CUDA calls
                pos_chunks = []
                val_chunks = []
                for s0 in range(0, total, maxSeedPerGpuCall):
                    s1 = min(total, s0 + maxSeedPerGpuCall)
                    pos_part, val_part = batch_pathline_integration_2D_cuda(
                        seeds[s0:s1], vectorfield, t_target, dt, max_steps, method
                    )
                    pos_chunks.append(pos_part)
                    val_chunks.append(val_part)
                pos_np = np.concatenate(pos_chunks, axis=0) if len(pos_chunks) > 1 else pos_chunks[0]
                val_np = np.concatenate(val_chunks, axis=0) if len(val_chunks) > 1 else val_chunks[0]
            return torch.from_numpy(pos_np).float(), torch.from_numpy(val_np).to(torch.int32)
            
        except Exception as e:
            logging.info(f"[batch_pathline_integration_2D_auto] CUDA failed: {e}. Fallback to CPU.")




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



