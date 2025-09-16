import numpy as np
import os
import json
import re
from numba import njit, prange
from typeguard import typechecked
from FLowUtils.VectorField2d import *
from FLowUtils.decoration import singleton
from .VectorField2d import IDiscreteField2D
from decimal import Decimal, ROUND_HALF_UP, localcontext
import logging
def _sanitize_name(name: str) -> str:
    """将任意名称转换为安全文件名（Windows 友好）。"""
    name = str(name).strip().replace(' ', '_')
    # 只保留字母数字、下划线、短横线和点
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)

def _format_float(value: float, digits: int = 4) -> str:
    """稳定格式化小数，避免 0.01 显示为 0.0099999... 等浮点数artifact。"""
    try:
        with localcontext() as ctx:
            ctx.prec = max(28, digits + 8)
            d = Decimal(str(value))
            q = Decimal('1') if digits <= 0 else (Decimal('1') / (Decimal('10') ** digits))
            d = d.quantize(q, rounding=ROUND_HALF_UP)
            s = format(d, 'f')
    except Exception:
        s = f"{float(value):.{digits}f}"
    if '.' in s:
        s = s.rstrip('0').rstrip('.')
    if s == '-0':
        s = '0'
    return s

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

def compute_ivd_minus_mean_2D(vector_field, **kwargs):
    ivd = compute_ivd_2D(vector_field)
    mean_ivd = np.mean(ivd, axis=(1, 2), keepdims=True)  # (T, 1, 1)
    ivd_minus_mean = np.abs(ivd - mean_ivd)
    return ivd_minus_mean




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
            'IVD_MINUS_MEAN': compute_ivd_minus_mean_2D,
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

    # ---------------------- Naming & IO Helpers ----------------------

    def build_scalar_filename(self, field_name: str, scalar_field: 'ScalarField2D',
                              extra_tags: dict | None = None, ext: str = ".npz") -> str:
        """构造标准化文件名: scalar__<field>__<X>x<Y>x<T>__t<tmin>-<tmax>__<dtype>[__k=v...].npz"""
        fname = [
            "scalar",
            _sanitize_name(field_name),
            f"{scalar_field.Xdim}x{scalar_field.Ydim}x{scalar_field.time_steps}",
            f"t{_format_float(scalar_field.tmin)}-{_format_float(scalar_field.tmax)}",
            str(np.dtype(scalar_field.dtype).name)
        ]
        if extra_tags:
            for k in extra_tags:
                fname.append(f"{self._sanitize_name(k)}")
        return "__".join(fname) + ext

    # ---------------------- NPZ Compressed IO ----------------------
    def save_scalar_field_to_file(self, scalar_field: 'ScalarField2D', out_dir: str ,   field_name: str | None = None, extra_tags: list[str] | None = None,
                                  quantization: str | None = None,
                                  overwrite: bool = True):
        """
        保存标量场为压缩 .npz，包含必要元数据。

        参数:
        - scalar_field: 要保存的 `ScalarField2D`
        - out_dir: 输出目录，必填
        - field_name: 文件名中的场名称部分
        - extra_tags: 附加到文件名中的键值信息（如数据源/备注）
        - quantization: None | 'float16' | 'uint16'；'uint16'将保存 min/max 以反量化
        - overwrite: 若 False 且文件存在，将在文件名后追加递增序号
        """
        assert scalar_field is not None and isinstance(scalar_field, ScalarField2D)

        data = scalar_field.getDataAsNumpy()
        orig_dtype_name = str(np.dtype(scalar_field.dtype).name)
        save_arr = data
        qinfo = None

        if quantization is not None:
            q = quantization.lower()
            if q == 'float16':
                save_arr = data.astype(np.float16, copy=False)
                qinfo = {"type": "float16"}
            elif q == 'uint16':
                vmin = float(np.min(data))
                vmax = float(np.max(data))
                if vmax == vmin:
                    scale = 1.0
                else:
                    scale = (vmax - vmin) / 65535.0
                # 映射到 [0, 65535]
                save_arr = np.round((data - vmin) / scale).clip(0, 65535).astype(np.uint16)
                qinfo = {"type": "uint16", "min": vmin, "max": vmax}
            else:
                raise ValueError("quantization 仅支持 None|'float16'|'uint16'")

        meta = {
            "version": 1,
            "Xdim": int(scalar_field.Xdim),
            "Ydim": int(scalar_field.Ydim),
            "time_steps": int(scalar_field.time_steps),
            "domainMinBoundary": list(map(float, scalar_field.domainMinBoundary)),
            "domainMaxBoundary": list(map(float, scalar_field.domainMaxBoundary)),
            "tmin": float(scalar_field.tmin),
            "tmax": float(scalar_field.tmax),
            "dtype": orig_dtype_name,
            "quantization": qinfo,
            "field_name": str(field_name) if field_name is not None else None,
        }

        # 自动命名（不包含 operation）
        fname = self.build_scalar_filename(
            field_name or "unnamed", scalar_field, extra_tags=extra_tags
        )
        os.makedirs(out_dir, exist_ok=True)
        file_path = os.path.join(out_dir, fname)

        if not overwrite and os.path.exists(file_path):
            base, ext = os.path.splitext(file_path)
            idx = 1
            new_path = f"{base}__{idx}{ext}"
            while os.path.exists(new_path):
                idx += 1
                new_path = f"{base}__{idx}{ext}"
            file_path = new_path

        np.savez_compressed(
            file_path,
            data=save_arr,
            meta=json.dumps(meta)
        )
        logging.getLogger().info(f"Saved scalar field {field_name} to {file_path}")
        return file_path

    def load_scalar_field_from_file(self, file_path: str) -> 'ScalarField2D':
        """
        加载标量场：
          1) 优先按照本工程保存的 .npz 规范（含 'data' 与 'meta' 键）读取；
          2) 若为 .npz 但缺少 meta/data，则取第一个数组作为数据并推断元信息；
          3) 若为 .npy 或被误写为 .npz 的原始数组，直接按数据形状推断；
        均保持对已有保存文件的兼容，不修改保存逻辑。
        返回 (ScalarField2D, field_name or None)
        """
        # 第一次尝试：严格不允许 pickle
        try:
            obj = np.load(file_path, allow_pickle=False)
        except Exception:
            # 安全兜底：允许 pickle 以兼容历史文件
            obj = np.load(file_path, allow_pickle=True)

        # 情况 A：标准 .npz 文件
        if isinstance(obj, np.lib.npyio.NpzFile):
            try:
                meta_raw = obj.get('meta', None)
                data_arr = obj.get('data', None)

                # 若无 data，则尝试取第一个数组
                if data_arr is None:
                    keys = list(obj.keys())
                    if len(keys) == 0:
                        raise ValueError("空 .npz 文件：未找到任何数组")
                    data_arr = obj[keys[0]]

                # 解析 meta（若存在）
                meta = None
                if meta_raw is not None:
                    meta = json.loads(str(meta_raw))

                # 量化反解
                if meta and isinstance(meta, dict) and meta.get('quantization'):
                    qinfo = meta['quantization']
                    if isinstance(qinfo, dict) and qinfo.get('type') == 'float16':
                        data = data_arr.astype(np.float32)
                    elif isinstance(qinfo, dict) and qinfo.get('type') == 'uint16':
                        vmin = float(qinfo['min']); vmax = float(qinfo['max'])
                        if vmax == vmin:
                            data = np.full(data_arr.shape, vmin, dtype=np.float32)
                        else:
                            scale = (vmax - vmin) / 65535.0
                            data = (data_arr.astype(np.float32) * scale + vmin)
                    else:
                        data = data_arr
                else:
                    data = data_arr

                # 构造 ScalarField2D
                if meta is not None:
                    dtype = np.dtype(meta.get('dtype', 'float32'))
                    sf = ScalarField2D(
                        int(meta['Xdim']), int(meta['Ydim']), int(meta['time_steps']),
                        meta['domainMinBoundary'], meta['domainMaxBoundary'], dtype=dtype
                    )
                    sf.set_discrete_data(data.astype(dtype, copy=False))
                    fieldName = meta.get('field_name')
                    return sf, fieldName
                else:
                    # 无 meta：按数据形状推断 (T, Y, X)
                    if data.ndim == 3:
                        T, Y, X = data.shape
                        sf = ScalarField2D(X, Y, T,
                                           [0.0, 0.0, 0.0],
                                           [float(X-1), float(Y-1), float(max(0, T-1))],
                                           dtype=data.dtype)
                        sf.set_discrete_data(data)
                        return sf, None
                    elif data.ndim == 2:
                        Y, X = data.shape
                        sf = ScalarField2D(X, Y, 1,
                                           [0.0, 0.0, 0.0],
                                           [float(X-1), float(Y-1), 0.0],
                                           dtype=data.dtype)
                        sf.set_discrete_data(data)
                        return sf, None
                    else:
                        raise ValueError(f"无法从数据形状 {data.shape} 推断标量场元信息")
            finally:
                try:
                    obj.close()
                except Exception:
                    pass

        # 情况 B：直接是 ndarray（.npy 或其它）
        if isinstance(obj, np.ndarray):
            data = obj
            if data.ndim == 3:
                T, Y, X = data.shape
                sf = ScalarField2D(X, Y, T,
                                   [0.0, 0.0, 0.0],
                                   [float(X-1), float(Y-1), float(max(0, T-1))],
                                   dtype=data.dtype)
                sf.set_discrete_data(data)
                return sf, None
            elif data.ndim == 2:
                Y, X = data.shape
                sf = ScalarField2D(X, Y, 1,
                                   [0.0, 0.0, 0.0],
                                   [float(X-1), float(Y-1), 0.0],
                                   dtype=data.dtype)
                sf.set_discrete_data(data)
                return sf, None
            else:
                raise ValueError(f"无法从数据形状 {data.shape} 推断标量场元信息")

        raise ValueError("无法识别的标量场文件格式：" + str(file_path))

    # ---------------------- Batch IO ----------------------
    def save_all_cached(self, out_dir: str, quantization: str | None = None, overwrite: bool = True):
        """将当前缓存的所有标量场保存到指定目录，文件名由命名规范自动生成。"""
        os.makedirs(out_dir, exist_ok=True)
        results = []
        for key, sf in self.scalar_fields.items():
            # 从 key 中解析 field_name（形如 OP(field)），若失败则使用 key
            field_name = str(key)
            if isinstance(key, str) and '(' in key and key.endswith(')'):
                try:
                    left = key.index('(')
                    field_name = key[left+1:-1]
                except Exception:
                    field_name = str(key)
            path = self.save_scalar_field_to_file(
                sf, out_dir=out_dir, field_name=field_name,
                quantization=quantization, overwrite=overwrite
            )
            results.append((key, path))
        return results

    def load_all_from_dir(self, dir_path: str):
        """加载目录下所有 .npz 标量场文件并返回列表。"""
        loaded = []
        for name in os.listdir(dir_path):
            if not name.lower().endswith('.npz'):
                continue
            path = os.path.join(dir_path, name)
            try:
                sf = self.load_scalar_field_from_file(path)
                loaded.append((name, sf))
            except Exception:
                # 忽略不兼容文件
                continue
        return loaded

    
