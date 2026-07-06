"""3D scalar fields derived from a 3D vector field, plus a marching-cubes helper.

Phase 3 of the 3D-flow-vis effort (see docs/flow3dvis.md). Provides:
  - single-time-slice scalar derivations from velocity: magnitude, vorticity magnitude,
    Q-criterion, lambda-2, and IVD (all operate on a (Z,Y,X,3) array with grid spacing);
  - a minimal `ScalarField3D` holder (also reused by the DVR phase);
  - `marching_cubes_world`, wrapping skimage into world-space (x,y,z) verts/normals/faces.
"""
import numpy as np
from numba import njit, prange


# --------------------------------------------------------------------------------------
# Scalar derivations. Input `data` is a single time slice (Z, Y, X, 3) of velocity;
# axes are Z=0, Y=1, X=2. Each returns a (Z, Y, X) float32 volume.
# --------------------------------------------------------------------------------------

def compute_velocity_magnitude_3D(data, dx, dy, dz):
    return np.linalg.norm(data, axis=-1).astype(np.float32)


def _vorticity_components(data, dx, dy, dz):
    u = data[..., 0]; v = data[..., 1]; w = data[..., 2]
    dw_dy = np.gradient(w, dy, axis=1); dv_dz = np.gradient(v, dz, axis=0)
    du_dz = np.gradient(u, dz, axis=0); dw_dx = np.gradient(w, dx, axis=2)
    dv_dx = np.gradient(v, dx, axis=2); du_dy = np.gradient(u, dy, axis=1)
    wx = dw_dy - dv_dz
    wy = du_dz - dw_dx
    wz = dv_dx - du_dy
    return wx, wy, wz


def compute_vorticity_magnitude_3D(data, dx, dy, dz):
    wx, wy, wz = _vorticity_components(data, dx, dy, dz)
    return np.sqrt(wx * wx + wy * wy + wz * wz).astype(np.float32)


def _velocity_gradients(data, dx, dy, dz):
    u = data[..., 0]; v = data[..., 1]; w = data[..., 2]
    dudx = np.gradient(u, dx, axis=2); dudy = np.gradient(u, dy, axis=1); dudz = np.gradient(u, dz, axis=0)
    dvdx = np.gradient(v, dx, axis=2); dvdy = np.gradient(v, dy, axis=1); dvdz = np.gradient(v, dz, axis=0)
    dwdx = np.gradient(w, dx, axis=2); dwdy = np.gradient(w, dy, axis=1); dwdz = np.gradient(w, dz, axis=0)
    return dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz


def compute_q_criterion_3D(data, dx, dy, dz):
    dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz = _velocity_gradients(data, dx, dy, dz)
    # ||S||_F^2 and ||Omega||_F^2 for the 3x3 velocity gradient tensor
    s_norm2 = (dudx * dudx + dvdy * dvdy + dwdz * dwdz
               + 0.5 * ((dudy + dvdx) ** 2 + (dudz + dwdx) ** 2 + (dvdz + dwdy) ** 2))
    omega_norm2 = 0.5 * ((dudy - dvdx) ** 2 + (dudz - dwdx) ** 2 + (dvdz - dwdy) ** 2)
    return (0.5 * (omega_norm2 - s_norm2)).astype(np.float32)


@njit(parallel=True, fastmath=True, cache=True)
def _lambda2_kernel_3d(dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz, out):
    Z, Y, X = dudx.shape
    for z in prange(Z):
        for y in range(Y):
            for x in range(X):
                J = np.array([[dudx[z, y, x], dudy[z, y, x], dudz[z, y, x]],
                              [dvdx[z, y, x], dvdy[z, y, x], dvdz[z, y, x]],
                              [dwdx[z, y, x], dwdy[z, y, x], dwdz[z, y, x]]], dtype=np.float64)
                S = 0.5 * (J + J.T)
                Om = 0.5 * (J - J.T)
                A = S @ S + Om @ Om
                ev = np.linalg.eigvals(A).real
                ev.sort()
                out[z, y, x] = ev[1]  # median eigenvalue = lambda2


def compute_lambda2_criterion_3D(data, dx, dy, dz):
    grads = _velocity_gradients(data, dx, dy, dz)
    out = np.zeros(data.shape[:3], dtype=np.float64)
    _lambda2_kernel_3d(*[np.ascontiguousarray(g, dtype=np.float64) for g in grads], out)
    return out.astype(np.float32)


def compute_ivd_3D(data, dx, dy, dz):
    """Instantaneous Vorticity Deviation: ||omega - <omega>|| (mean over the slice)."""
    wx, wy, wz = _vorticity_components(data, dx, dy, dz)
    wx = wx - wx.mean(); wy = wy - wy.mean(); wz = wz - wz.mean()
    return np.sqrt(wx * wx + wy * wy + wz * wz).astype(np.float32)


SCALAR_OPS_3D = {
    'MAGNITUDE': compute_velocity_magnitude_3D,
    'VORTICITY': compute_vorticity_magnitude_3D,
    'Q_CRITERION': compute_q_criterion_3D,
    'LAMBDA2': compute_lambda2_criterion_3D,
    'IVD': compute_ivd_3D,
}


def velocity_slice(field, time_index) -> np.ndarray:
    """Return the (Z, Y, X, 3) numpy velocity slice of an UnsteadyVectorField3D at a time
    index, indexing `field.field` directly. This deliberately avoids the field's own
    `getSlice()` / `SteadyVectorField3D` path, which has a pre-existing constructor bug
    (passes domain bounds into the `timesteps` positional slot)."""
    raw = getattr(field, "field", None)
    if raw is None:
        return None
    ti = int(np.clip(time_index, 0, field.time_steps - 1))
    frame = raw[ti]
    if hasattr(frame, "detach"):
        frame = frame.detach().cpu().numpy()
    return np.ascontiguousarray(frame, dtype=np.float32)


def compute_scalar_slice_3D(operation: str, data_zyx3: np.ndarray, grid_interval) -> np.ndarray:
    """Dispatch a named scalar operation on a single (Z,Y,X,3) velocity slice -> (Z,Y,X)."""
    op = SCALAR_OPS_3D.get(str(operation))
    if op is None:
        raise ValueError(f"Unknown 3D scalar operation: {operation}")
    dx = float(grid_interval[0]) or 1.0
    dy = float(grid_interval[1]) or 1.0
    dz = float(grid_interval[2]) or 1.0
    return np.ascontiguousarray(op(np.ascontiguousarray(data_zyx3, dtype=np.float32), dx, dy, dz))


# --------------------------------------------------------------------------------------
# Minimal 3D scalar-field holder (reused by the DVR phase for texture upload / IO).
# Data layout mirrors the 3D vector field: (T, Z, Y, X).
# --------------------------------------------------------------------------------------

class ScalarField3D:
    def __init__(self, Xdim, Ydim, Zdim, time_steps,
                 domainMinBoundary=[-2.0, -2.0, -2.0], domainMaxBoundary=[2.0, 2.0, 2.0],
                 tmin=0.0, tmax=0.0, dtype=np.float32):
        self.Xdim = int(Xdim); self.Ydim = int(Ydim); self.Zdim = int(Zdim)
        self.time_steps = int(time_steps)
        self.domainMinBoundary = np.asarray(domainMinBoundary, dtype=np.float32)
        self.domainMaxBoundary = np.asarray(domainMaxBoundary, dtype=np.float32)
        self.tmin = float(tmin); self.tmax = float(tmax)
        self.gridInterval = np.array([
            (self.domainMaxBoundary[0] - self.domainMinBoundary[0]) / (Xdim - 1) if Xdim > 1 else 0.0,
            (self.domainMaxBoundary[1] - self.domainMinBoundary[1]) / (Ydim - 1) if Ydim > 1 else 0.0,
            (self.domainMaxBoundary[2] - self.domainMinBoundary[2]) / (Zdim - 1) if Zdim > 1 else 0.0,
        ], dtype=np.float32)
        self.timeInterval = (self.tmax - self.tmin) / (self.time_steps - 1) if self.time_steps > 1 else 0.0
        self.dtype = dtype
        self.field = None  # (T, Z, Y, X)

    def getDim(self):
        return 3

    def set_discrete_data(self, data):
        if isinstance(data, list):
            data = np.stack(data, axis=0)
        if self.time_steps == 1 and data.shape == (self.Zdim, self.Ydim, self.Xdim):
            data = data[np.newaxis]
        assert data.shape == (self.time_steps, self.Zdim, self.Ydim, self.Xdim), \
            f"expected {(self.time_steps, self.Zdim, self.Ydim, self.Xdim)}, got {data.shape}"
        self.field = np.ascontiguousarray(data, dtype=self.dtype)

    def getSlice(self, timeSlice):
        return self.field[int(np.clip(timeSlice, 0, self.time_steps - 1))]

    def getPhysicalTime(self, idt):
        return self.timeInterval * idt + self.tmin

    def getNearestTimeIndex(self, time):
        if self.timeInterval == 0:
            return 0
        return int(np.clip(round((time - self.tmin) / self.timeInterval), 0, self.time_steps - 1))

    def compute_min_max(self):
        return float(np.min(self.field)), float(np.max(self.field))


# --------------------------------------------------------------------------------------
# Marching cubes -> world-space mesh.
# --------------------------------------------------------------------------------------

def marching_cubes_world(vol_zyx: np.ndarray, level: float, grid_interval, origin_xyz):
    """Extract an isosurface at `level` and return (verts_xyz, normals_xyz, faces).

    vol_zyx: (Z, Y, X) scalar volume. grid_interval=(dx,dy,dz), origin_xyz=(x0,y0,z0).
    Returns float32 (V,3) world verts, float32 (V,3) normals, uint32 (F,3) faces, or None
    if `level` is outside the value range (no surface).
    """
    from skimage import measure
    vmin = float(vol_zyx.min())
    vmax = float(vol_zyx.max())
    if not (vmin < level < vmax):
        return None
    dx = float(grid_interval[0]); dy = float(grid_interval[1]); dz = float(grid_interval[2])
    # skimage indexes the volume as [z, y, x]; spacing follows that axis order.
    verts, faces, normals, _ = measure.marching_cubes(vol_zyx, level=level, spacing=(dz, dy, dx))
    world = np.empty_like(verts, dtype=np.float32)
    world[:, 0] = origin_xyz[0] + verts[:, 2]  # x
    world[:, 1] = origin_xyz[1] + verts[:, 1]  # y
    world[:, 2] = origin_xyz[2] + verts[:, 0]  # z
    n = np.empty_like(normals, dtype=np.float32)
    n[:, 0] = normals[:, 2]; n[:, 1] = normals[:, 1]; n[:, 2] = normals[:, 0]
    return world, n, faces.astype(np.uint32)
