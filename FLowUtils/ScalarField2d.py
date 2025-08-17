import numpy as np
from numba import njit, prange
from typeguard import typechecked
from FLowUtils.VectorField2d import *
from FLowUtils.decoration import singleton
from .VectorField2d import IDiscreteField2D

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




class ScalarField2D(IDiscreteField2D):
    def __init__(self, Xdim, Ydim, time_steps, domainMinBoundary=[-2.0,-2.0,0.0], domainMaxBoundary=[2.0,2.0,2.0], dtype=np.float32):
        """
        Initialize a 2D scalar field.
        
        :param Xdim: X dimension of the scalar field
        :param Ydim: Y dimension of the scalar field  
        :param time_steps: Number of time steps
        :param domainMinBoundary: Minimum boundaries for (x,y,t)
        :param domainMaxBoundary: Maximum boundaries for (x,y,t)
        :param dtype: Data type for the field
        """
        super(ScalarField2D, self).__init__(Xdim, Ydim, domainMinBoundary, domainMaxBoundary, time_steps)
        self.dtype = dtype
        self.field = None  # Discrete data
        self.analytical_func = None  # Analytical expression

    def set_discrete_data(self, data):
        """为标量场设置离散数据。支持 data 为 list[slice_data,...] 或 ndarray。"""
        # 如果 data 是 list（如 [slice0, slice1, ...]），将其转换为 ndarray
        if isinstance(data, list):
            data = np.stack(data, axis=0)
        # 处理 time_steps=1 且 data.shape=(Y,X) 的情况
        if self.time_steps == 1 and data.shape == (self.Ydim, self.Xdim):
            data = data[np.newaxis, :, :]
        assert data.shape == (self.time_steps, self.Ydim, self.Xdim), f"期望形状 {(self.time_steps, self.Ydim, self.Xdim)}, 实际为 {data.shape}"
        self.field = data.astype(self.dtype)


    def set_analytical_func(self, func):
        """Set analytical function for the scalar field."""
        self.analytical_func = func

    def has_discrete_data(self):
        """Check if discrete data is available."""
        return self.field is not None

    def has_analytical_func(self):
        """Check if analytical function is available."""
        return self.analytical_func is not None

    def getSlice(self, timeSlice):
        """Get a slice of the scalar field at a specific time step."""
        if self.has_discrete_data():
            return self.field[timeSlice]
        elif self.has_analytical_func():
            # Create discrete data for the time slice
            slice_data = np.zeros((self.Ydim, self.Xdim), dtype=self.dtype)
            time = self.getPhysicalTime(timeSlice)
            for y in range(self.Ydim):
                for x in range(self.Xdim):
                    pos_x, pos_y = self.convert_grid_pos_2_physical_pos(x, y)
                    slice_data[y, x] = self.analytical_func(pos_x, pos_y, time)
            return slice_data
        else:
            raise RuntimeError("No data available")

    def get_value_at_grid(self, x, y, t):
        """
        Get value at grid point.
        
        :param x: Grid X index
        :param y: Grid Y index  
        :param t: Time step index
        :return: Scalar value at grid point
        """
        if self.has_discrete_data():
            return self.field[t, y, x]
        elif self.has_analytical_func():
            pos_x, pos_y = self.convert_grid_pos_2_physical_pos(x, y)
            time = self.getPhysicalTime(t)
            return self.analytical_func(pos_x, pos_y, time)
        else:
            raise RuntimeError("No data available")

    def get_value_at_physical_pos(self, xpos, ypos, time):
        """
        Get interpolated value at arbitrary physical position.
        
        :param xpos: Physical X coordinate
        :param ypos: Physical Y coordinate
        :param time: Physical time
        :return: Interpolated scalar value
        """
        if self.has_discrete_data():
            gx, gy = self.convert_physical_pos_2_grid_pos(xpos, ypos)
            gt = self.getFloatGridTime(time)
            return self.trilinear_interpolate(gx, gy, gt)
        elif self.has_analytical_func():
            return self.analytical_func(xpos, ypos, time)
        else:
            raise RuntimeError("No data available")

    def trilinear_interpolate(self, gx, gy, gt):
        """
        Perform trilinear interpolation.
        
        :param gx: Grid X coordinate (float)
        :param gy: Grid Y coordinate (float) 
        :param gt: Grid time coordinate (float)
        :return: Interpolated value
        """
        # Get surrounding grid indices
        x0, x1 = int(np.floor(gx)), int(np.ceil(gx))
        y0, y1 = int(np.floor(gy)), int(np.ceil(gy))
        t0, t1 = int(np.floor(gt)), int(np.ceil(gt))
        
        # Clamp to grid boundaries
        x0 = np.clip(x0, 0, self.Xdim - 1)
        x1 = np.clip(x1, 0, self.Xdim - 1)
        y0 = np.clip(y0, 0, self.Ydim - 1)
        y1 = np.clip(y1, 0, self.Ydim - 1)
        t0 = np.clip(t0, 0, self.time_steps - 1)
        t1 = np.clip(t1, 0, self.time_steps - 1)
        
        # Get interpolation weights
        xd, yd, td = gx - x0, gy - y0, gt - t0

        # Get values at surrounding grid points
        c000 = self.field[t0, y0, x0]
        c100 = self.field[t0, y0, x1]
        c010 = self.field[t0, y1, x0]
        c110 = self.field[t0, y1, x1]
        c001 = self.field[t1, y0, x0]
        c101 = self.field[t1, y0, x1]
        c011 = self.field[t1, y1, x0]
        c111 = self.field[t1, y1, x1]

        # Trilinear interpolation
        c00 = c000 * (1 - xd) + c100 * xd
        c01 = c001 * (1 - xd) + c101 * xd
        c10 = c010 * (1 - xd) + c110 * xd
        c11 = c011 * (1 - xd) + c111 * xd

        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd

        c = c0 * (1 - td) + c1 * td
        return c

    def compute_min_max(self):
        """
        Compute minimum and maximum values of the scalar field.
        
        :return: Tuple of (min_value, max_value)
        """
        if self.has_discrete_data():
            return float(np.min(self.field)), float(np.max(self.field))
        elif self.has_analytical_func():
            # Sample the analytical function
            samples = []
            for t in range(self.time_steps):
                time = self.getPhysicalTime(t)
                for y in range(self.Ydim):
                    for x in range(self.Xdim):
                        pos_x, pos_y = self.convert_grid_pos_2_physical_pos(x, y)
                        samples.append(self.analytical_func(pos_x, pos_y, time))
            return float(np.min(samples)), float(np.max(samples))
        else:
            raise RuntimeError("No data available")

    def getDataAsNumpy(self):
        """Get field data as numpy array."""
        if self.has_discrete_data():
            return self.field
        else:
            raise RuntimeError("No discrete data available")

    def numpy2torch(self):
        """Convert field data from numpy array to torch tensor."""
        if self.has_discrete_data():
            import torch
            self.field = torch.tensor(self.field)
        else:
            raise RuntimeError("No discrete data available")

    def torch2numpy(self):
        """Convert field data from torch tensor to numpy array."""
        if self.has_discrete_data() and hasattr(self.field, 'detach'):
            self.field = self.field.detach().cpu().numpy()
        else:
            raise RuntimeError("No discrete data available or not a torch tensor")


    
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
            return self.scalar_fields[key],key
        if compute_func is None:
            compute_func = self.builtin_ops[operation]
        if not callable(compute_func):
            raise ValueError(f"Invalid compute function for operation: {operation}")
        
        scalar_field_data = compute_func(targetField)
        scalar_field = ScalarField2D(targetField.Xdim, targetField.Ydim,targetField.time_steps, targetField.domainMinBoundary, 
                                     targetField.domainMaxBoundary, scalar_field_data.dtype)        
        scalar_field.set_discrete_data(scalar_field_data)
        self.scalar_fields[key] = scalar_field
        return scalar_field,key

    def has_scalar_field(self, field_name, operation):
        key = self._make_key(field_name, operation)
        return key in self.scalar_fields

    def get_scalar_field(self, field_name, operation):
        key = self._make_key(field_name, operation)
        return self.scalar_fields.get(key, None)


    
