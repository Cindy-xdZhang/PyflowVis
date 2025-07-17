import numpy as np
from numba import njit, prange
from typeguard import typechecked
from FLowUtils.VectorField2d import *
from FLowUtils.decoration import singleton

@typechecked
def compute_velocity_magnitude_2D(vector_field, **kwargs):
    data = vector_field.getDataAsNumpy()  # shape: (T, Y, X, 2)
    # NumPy本身已SIMD
    return np.linalg.norm(data, axis=-1)  # (T, Y, X)

def compute_curl_magnitude_2D(vector_field, **kwargs):
    data = vector_field.getDataAsNumpy()  # (T, Y, X, 2)
    dx = vector_field.gridInterval[0]
    dy = vector_field.gridInterval[1]
    u = data[..., 0]
    v = data[..., 1]
    # np.gradient支持多维并行
    du_dy = np.gradient(u, dy, axis=1)  # (T, Y, X)
    dv_dx = np.gradient(v, dx, axis=2)
    curl = dv_dx - du_dy
    return np.abs(curl)

def compute_q_criterion_2D(vector_field, **kwargs):
    data = vector_field.getDataAsNumpy()  # (T, Y, X, 2)
    dx = vector_field.gridInterval[0]
    dy = vector_field.gridInterval[1]
    u = data[..., 0]
    v = data[..., 1]
    du_dx = np.gradient(u, dx, axis=2)
    du_dy = np.gradient(u, dy, axis=1)
    dv_dx = np.gradient(v, dx, axis=2)
    dv_dy = np.gradient(v, dy, axis=1)
    Sxx = du_dx
    Syy = dv_dy
    Sxy = 0.5 * (du_dy + dv_dx)
    Omega_xy = 0.5 * (du_dy - dv_dx)
    S_norm2 = Sxx**2 + Syy**2 + 2 * Sxy**2
    Omega_norm2 = 2 * Omega_xy**2
    Q = 0.5 * (Omega_norm2 - S_norm2)
    return Q

@njit(parallel=True, fastmath=True)
def _lambda2_kernel(du_dx, du_dy, dv_dx, dv_dy, out):
    T, Y, X = du_dx.shape
    for t in prange(T):
        for y in range(Y):
            for x in range(X):
                J = np.array([[du_dx[t, y, x], du_dy[t, y, x]],
                              [dv_dx[t, y, x], dv_dy[t, y, x]]], dtype=np.float32)
                S = 0.5 * (J + J.T)
                Omega = 0.5 * (J - J.T)
                A = S @ S + Omega @ Omega
                eigvals = np.linalg.eigvals(A)
                out[t, y, x] = np.min(eigvals.real)  # 只取实部
    return out

def compute_lambda2_criterion_2D(vector_field, **kwargs):
    data = vector_field.getDataAsNumpy()  # (T, Y, X, 2)
    dx = vector_field.gridInterval[0]
    dy = vector_field.gridInterval[1]
    u = data[..., 0]
    v = data[..., 1]
    du_dx = np.gradient(u, dx, axis=2)
    du_dy = np.gradient(u, dy, axis=1)
    dv_dx = np.gradient(v, dx, axis=2)
    dv_dy = np.gradient(v, dy, axis=1)
    out = np.zeros_like(u)
    return _lambda2_kernel(du_dx, du_dy, dv_dx, dv_dy, out)

def compute_ivd_2D(vector_field, **kwargs):
    curl = compute_curl_magnitude_2D(vector_field)
    mean_curl = np.mean(curl, axis=(1, 2), keepdims=True)  # (T, 1, 1)
    ivd = np.abs(curl - mean_curl)
    return ivd




class ScalarField2D:
    def __init__(self, Xdim, Ydim, time_steps, domainMinBoundary, domainMaxBoundary, tmin, tmax, dtype=np.float32):
        self.Xdim = Xdim
        self.Ydim = Ydim
        self.time_steps = time_steps
        self.domainMinBoundary = domainMinBoundary
        self.domainMaxBoundary = domainMaxBoundary
        self.tmin = tmin
        self.tmax = tmax
        self.dtype = dtype
        self.gridInterval = [
            (domainMaxBoundary[0] - domainMinBoundary[0]) / (Xdim - 1),
            (domainMaxBoundary[1] - domainMinBoundary[1]) / (Ydim - 1)
        ]
        self.timeInterval = (tmax - tmin) / (time_steps - 1) if time_steps > 1 else 0
        self.field = None  # 离散数据
        self.analytical_func = None  # 分析表达式
    def getMinTime(self):
        return self.tmin
    def getMaxTime(self):
        return self.tmax
  
    def set_discrete_data(self, data):
        assert data.shape == (self.time_steps, self.Ydim, self.Xdim)
        self.field = data.astype(self.dtype)

    def set_analytical_func(self, func):
        self.analytical_func = func

    def has_discrete_data(self):
        return self.field is not None

    def has_analytical_func(self):
        return self.analytical_func is not None

    def get_value_at_grid(self, x, y, t):
        """x, y, t为整数网格索引"""
        if self.has_discrete_data():
            return self.field[t, y, x]
        elif self.has_analytical_func():
            pos = self.convert_grid_index_to_physical(x, y)
            time = self.convert_time_index_to_physical(t)
            return self.analytical_func(pos[0], pos[1], time)
        else:
            raise RuntimeError("No data available")

    def get_value_at_physical_pos(self, xpos, ypos, time):
        """xpos, ypos, time为物理坐标，自动三线性插值"""
        if self.has_discrete_data():
            gx, gy = self.convert_physical_to_grid(xpos, ypos)
            gt = self.convert_physical_time_to_grid(time)
            return self.trilinear_interpolate(gx, gy, gt)
        elif self.has_analytical_func():
            return self.analytical_func(xpos, ypos, time)
        else:
            raise RuntimeError("No data available")

    def convert_grid_index_to_physical(self, x, y):
        px = self.domainMinBoundary[0] + x * self.gridInterval[0]
        py = self.domainMinBoundary[1] + y * self.gridInterval[1]
        return px, py

    def convert_time_index_to_physical(self, t):
        return self.tmin + t * self.timeInterval

    def convert_physical_to_grid(self, xpos, ypos):
        gx = (xpos - self.domainMinBoundary[0]) / self.gridInterval[0]
        gy = (ypos - self.domainMinBoundary[1]) / self.gridInterval[1]
        return gx, gy

    def convert_physical_time_to_grid(self, time):
        return (time - self.tmin) / self.timeInterval

    def trilinear_interpolate(self, gx, gy, gt):
        # 取最近的8个点做三线性插值
        x0, x1 = int(np.floor(gx)), int(np.ceil(gx))
        y0, y1 = int(np.floor(gy)), int(np.ceil(gy))
        t0, t1 = int(np.floor(gt)), int(np.ceil(gt))
        x0 = np.clip(x0, 0, self.Xdim - 1)
        x1 = np.clip(x1, 0, self.Xdim - 1)
        y0 = np.clip(y0, 0, self.Ydim - 1)
        y1 = np.clip(y1, 0, self.Ydim - 1)
        t0 = np.clip(t0, 0, self.time_steps - 1)
        t1 = np.clip(t1, 0, self.time_steps - 1)
        xd, yd, td = gx - x0, gy - y0, gt - t0

        c000 = self.field[t0, y0, x0]
        c100 = self.field[t0, y0, x1]
        c010 = self.field[t0, y1, x0]
        c110 = self.field[t0, y1, x1]
        c001 = self.field[t1, y0, x0]
        c101 = self.field[t1, y0, x1]
        c011 = self.field[t1, y1, x0]
        c111 = self.field[t1, y1, x1]

        c00 = c000 * (1 - xd) + c100 * xd
        c01 = c001 * (1 - xd) + c101 * xd
        c10 = c010 * (1 - xd) + c110 * xd
        c11 = c011 * (1 - xd) + c111 * xd

        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd

        c = c0 * (1 - td) + c1 * td
        return c

    def compute_min_max(self):
        if self.has_discrete_data():
            return float(np.min(self.field)), float(np.max(self.field))
        elif self.has_analytical_func():
            # 采样一遍
            samples = []
            for t in np.linspace(self.tmin, self.tmax, self.time_steps):
                for y in np.linspace(self.domainMinBoundary[1], self.domainMaxBoundary[1], self.Ydim):
                    for x in np.linspace(self.domainMinBoundary[0], self.domainMaxBoundary[0], self.Xdim):
                        samples.append(self.analytical_func(x, y, t))
            return float(np.min(samples)), float(np.max(samples))
        else:
            raise RuntimeError("No data available")


    
@singleton
class ScalarFieldManager:
    def __init__(self):
        self.scalar_fields = {}  # key: (field_name, operation, time_range) -> ScalarField2D
        self.builtin_ops = {
            'MAGNITUDE': compute_velocity_magnitude_2D,
            'CURL': compute_curl_magnitude_2D,
            'Q_CRITERION': compute_q_criterion_2D,
            'LAMBDA2': compute_lambda2_criterion_2D,
            'IVD': compute_ivd_2D,
        }

    def _make_key(self, field_name, operation):
        return str(operation)+"("+ str(field_name)+")"

    def request_scalar_field(self, field_name,targetField, operation, compute_func=None):
        key = self._make_key(field_name, operation)
        if key in self.scalar_fields:
            return self.scalar_fields[key]
        if compute_func is None:
            compute_func = self.builtin_ops[operation]
        if not callable(compute_func):
            raise ValueError(f"Invalid compute function for operation: {operation}")
        
        scalar_field_data = compute_func(targetField)
        scalar_field = ScalarField2D(targetField.Xdim, targetField.Ydim,targetField.time_steps, targetField.domainMinBoundary, 
                                     targetField.domainMaxBoundary,
                                         targetField.tmin, targetField.tmax, scalar_field_data.dtype)        
        scalar_field.set_discrete_data(scalar_field_data)
        self.scalar_fields[key] = scalar_field
        return scalar_field,key

    def has_scalar_field(self, field_name, operation):
        key = self._make_key(field_name, operation)
        return key in self.scalar_fields

    def get_scalar_field(self, field_name, operation):
        key = self._make_key(field_name, operation)
        return self.scalar_fields.get(key, None)


    
