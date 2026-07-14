"""
FLowUtils/ReferenceFrameOptimization.py

参考系优化 (reference-frame / observer optimization) 功能包。

给定非定常流场 v(x,t)，寻找观察者速度场 u(x,t)，使观测场 v_hat = v - u 尽量客观/定常。

核心：所有方法都最小化 observed time derivative
    L_u(v-u) = ∂v/∂t - ∂u/∂t + ∇v·u - ∇u·v ,
把 u=Σ qⁱ·eᵢ 展开后它对 (q, q̇) 是线性的，全域积分成二次型，解正规方程 AᵀA·x = Aᵀb 即可。
（参考 Variational Computation of Optimal Reference Frames，Eq 3/10a；论文的变分/Hamilton 求解不在本包范围。）

复现 optimal-connection C++ 的两类方法（详见 docs/referenceframeOptimization.md）：

- Killing mode        : eᵢ 取精确 Killing 基（2D k=3, 3D k=6），u 为刚体运动，无需正则化，逐时间片全局解。
      * killing_optimization_2d  <-  ComputePerTimestepLSQ (Vis26variationalRFT)   # 含 ∂u/∂t (q̇)
      * killing_optimization_3d  <-  KillingLeastSquareOptimization3d (flow3d)      # 省略 ∂u/∂t 项
- Non-killing mode    : Günther 2017 SIGGRAPH，逐点局部最优参考系（邻域最小二乘 + SAT，隐式正则）。
      * gunther17_optimization_2d <- runGunther17Optimization (Objectivity)
      * gunther17_optimization_3d <- GenericLocalOptimization3d (Objective)

数据结构对接 VectorField2d.UnsteadyVectorField2D (field: (T,Y,X,2))
与 VectorField3d.UnsteadyVectorField3D (field: (T,Z,Y,X,3))。
J=∇v 与 ∂v/∂t 用中心差分 (numpy.gradient) 按物理间距计算。

注意：Günther 逐点法会构造 (Y,X,6,6) / (Z,Y,X,12,12) 的稠密张量，大网格内存开销较高，
必要时先降采样。
"""

import itertools
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .VectorField2d import UnsteadyVectorField2D
from .VectorField3d import UnsteadyVectorField3D


# ═══════════════════════════════════════════════════════════════════════════
# Result container
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class ReferenceFrameResult:
    """参考系优化结果。

    u_field:          观察者速度场 u。
    v_minus_u_field:  观测场 v - u（在最优参考系里观测到的流场）。
    params:           Killing mode 下逐时间片参数
                      (2D: (T,3)=(a,b,c); 3D: (T,6)=(tx,ty,tz,wx,wy,wz))；
                      Günther mode 下为 None（逐点场无单一参数）。
    mode:             方法标识字符串。
    """
    u_field: object
    v_minus_u_field: object
    params: Optional[np.ndarray] = None
    mode: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# Small numeric helpers
# ═══════════════════════════════════════════════════════════════════════════
def _as_numpy(field) -> np.ndarray:
    """取出场数据为 float64 numpy 数组 (支持 torch tensor)。"""
    data = field.field
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    return np.asarray(data, dtype=np.float64)


def _dvdt(data: np.ndarray, dt: float) -> np.ndarray:
    """时间偏导 ∂v/∂t，中心差分。data 首轴为时间。dt<=0 或单帧 -> 0。"""
    if data.shape[0] < 2 or dt <= 0.0:
        return np.zeros_like(data)
    return np.gradient(data, dt, axis=0)


def _jacobian_2d(slice_v: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """slice_v (Y,X,2) -> J (Y,X,2,2)，J[...,i,j] = ∂v_i/∂x_j (col0=∂/∂x, col1=∂/∂y)。"""
    dv_dx = np.gradient(slice_v, dx, axis=1)  # axis1 = x
    dv_dy = np.gradient(slice_v, dy, axis=0)  # axis0 = y
    return np.stack([dv_dx, dv_dy], axis=-1)


def _jacobian_3d(slice_v: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
    """slice_v (Z,Y,X,3) -> J (Z,Y,X,3,3)，J[...,i,j]=∂v_i/∂x_j (col 0/1/2 = ∂/∂x,∂/∂y,∂/∂z)。"""
    dv_dx = np.gradient(slice_v, dx, axis=2)  # axis2 = x
    dv_dy = np.gradient(slice_v, dy, axis=1)  # axis1 = y
    dv_dz = np.gradient(slice_v, dz, axis=0)  # axis0 = z
    return np.stack([dv_dx, dv_dy, dv_dz], axis=-1)


def _skew_field(P: np.ndarray) -> np.ndarray:
    """向量场 (...,3) -> 反对称阵场 (...,3,3)，skew(a)·b = a × b。"""
    z = np.zeros(P.shape[:-1], dtype=P.dtype)
    ax, ay, az = P[..., 0], P[..., 1], P[..., 2]
    row0 = np.stack([z, -az, ay], axis=-1)
    row1 = np.stack([az, z, -ax], axis=-1)
    row2 = np.stack([-ay, ax, z], axis=-1)
    return np.stack([row0, row1, row2], axis=-2)


def _windowed_sum_nd(arr: np.ndarray, radius: int, n_spatial: int) -> np.ndarray:
    """在前 n_spatial 个空间轴上做 (2*radius+1)^n_spatial 盒式邻域求和（边界钳制，闭区间，
    等价于 C++ 的 [i-U, i+U] clamp 邻域），用积分图 (summed-area-table) O(1) 取窗口和。
    后续轴（矩阵/向量分量）原样保留。"""
    if radius <= 0:
        return arr.astype(np.float64, copy=True)
    S = arr.shape[:n_spatial]
    integ = arr.astype(np.float64, copy=True)
    for ax in range(n_spatial):
        integ = np.cumsum(integ, axis=ax)
    pad_width = [(1, 0)] * n_spatial + [(0, 0)] * (arr.ndim - n_spatial)
    integ = np.pad(integ, pad_width)  # (S0+1, ..., *rest)；integ[i+1]=sum arr[:i+1]

    idx = [np.arange(s) for s in S]
    lo = [np.clip(idx[k] - radius, 0, S[k] - 1) for k in range(n_spatial)]        # 低角
    hi = [np.clip(idx[k] + radius, 0, S[k] - 1) + 1 for k in range(n_spatial)]    # 高角(=hi+1)

    out = np.zeros_like(arr, dtype=np.float64)
    for corner in itertools.product((0, 1), repeat=n_spatial):  # 0=低角, 1=高角
        sign = 1.0
        take = []
        for k in range(n_spatial):
            if corner[k]:
                take.append(hi[k])
            else:
                take.append(lo[k])
                sign = -sign
        out += sign * integ[np.ix_(*take)]
    return out


def _batched_solve(MTM: np.ndarray, MTb: np.ndarray,
                   ridge: float = 1e-9, floor: float = 1e-12) -> np.ndarray:
    """批量解 MTM · x = MTb（MTM: (...,k,k), MTb: (...,k)）。
    加极小 Tikhonov 正则（按每点迹缩放 + 绝对下限），等价 C++ 满秩 QR 最小二乘，避免奇异点崩溃。"""
    k = MTM.shape[-1]
    eye = np.eye(k)
    trace = np.einsum("...ii->...", MTM)[..., None, None]
    scale = np.maximum(trace / k, 0.0)
    reg = MTM + (ridge * scale + floor) * eye
    return np.linalg.solve(reg, MTb[..., None])[..., 0]


def _solve_sym_small(A: np.ndarray, b: np.ndarray, rcond: float = 1e-8) -> np.ndarray:
    """解单个小对称系统（Killing 全局）的最小二乘。

    对秩亏输入（纯刚体/解析场，A 列秩亏）必须用**相对** rcond 截断近零奇异方向，否则
    rcond=None（绝对阈值 eps·max(M,N)≈1e-15）会保留近零奇异值、给出最小残差但范数巨大的解
    （3D Killing 在纯刚体场上会爆到 ~1e5）。rcond=1e-8 截断相对最大奇异值 <1e-8 的方向。
    """
    A = 0.5 * (A + A.T)
    sol, *_ = np.linalg.lstsq(A, b, rcond=rcond)
    return sol


def _normalize_mask(mask, shape_txy) -> Optional[np.ndarray]:
    """把 mask 归一化为 (T, *spatial) 的 float 权重；接受 (T,*spatial) / (*spatial) / None。"""
    if mask is None:
        return None
    m = np.asarray(mask, dtype=np.float64)
    T = shape_txy[0]
    spatial = shape_txy[1:]
    if m.shape == tuple(spatial):            # 每个时间片共用同一空间 mask
        m = np.broadcast_to(m, (T,) + tuple(spatial))
    assert m.shape == tuple(shape_txy), f"mask shape {m.shape} != {tuple(shape_txy)}"
    return m


# ═══════════════════════════════════════════════════════════════════════════
# Output field builders
# ═══════════════════════════════════════════════════════════════════════════
def _make_like_2d(field: UnsteadyVectorField2D, data: np.ndarray) -> UnsteadyVectorField2D:
    out = UnsteadyVectorField2D(field.Xdim, field.Ydim, field.time_steps,
                                list(field.domainMinBoundary), list(field.domainMaxBoundary))
    out.field = np.ascontiguousarray(data, dtype=np.float32)
    return out


def _make_like_3d(field: UnsteadyVectorField3D, data: np.ndarray) -> UnsteadyVectorField3D:
    out = UnsteadyVectorField3D(field.Xdim, field.Ydim, field.Zdim, field.time_steps,
                                list(np.asarray(field.domainMinBoundary)[:3]),
                                list(np.asarray(field.domainMaxBoundary)[:3]),
                                float(field.tmin), float(field.tmax))
    out.field = np.ascontiguousarray(data, dtype=np.float32)
    return out


def _coords_2d(field: UnsteadyVectorField2D):
    xs = field.domainMinBoundary[0] + field.gridInterval[0] * np.arange(field.Xdim)
    ys = field.domainMinBoundary[1] + field.gridInterval[1] * np.arange(field.Ydim)
    Yg, Xg = np.meshgrid(ys, xs, indexing="ij")  # (Y,X)
    return Xg, Yg


def _coords_3d(field: UnsteadyVectorField3D):
    gi = np.asarray(field.gridInterval)
    bmin = np.asarray(field.domainMinBoundary)
    xs = bmin[0] + gi[0] * np.arange(field.Xdim)
    ys = bmin[1] + gi[1] * np.arange(field.Ydim)
    zs = bmin[2] + gi[2] * np.arange(field.Zdim)
    Zg, Yg, Xg = np.meshgrid(zs, ys, xs, indexing="ij")  # (Z,Y,X)
    return Xg, Yg, Zg


# ═══════════════════════════════════════════════════════════════════════════
# 1. Killing mode — 2D 逐时间片最小二乘 (ComputePerTimestepLSQ)
# ═══════════════════════════════════════════════════════════════════════════
def killing_optimization_2d(field: UnsteadyVectorField2D,
                            boundary_skip: int = 2,
                            mask=None) -> ReferenceFrameResult:
    """2D Killing 参考系优化（逐时间片全局最小二乘，求最优 Killing 系数 (a,b,c) 及其时间导数 q̇）。

    = 最小化完整 observed time derivative  L_u(v−u)=∂v/∂t−∂u/∂t+∇v·u−∇u·v  的平方，
    残差 Σ_x ‖ L·q − W·q̇ + v_t ‖²（含 −∂u/∂t=−W·q̇ 项），其中 u=W·q=(a−c·y, b+c·x)，
    W=[[1,0,−y],[0,1,x]]，L=∇v·W−∇WV，∇WV.col2=(−v_y, v_x)。

    Args:
        field:         非定常 2D 流场。
        boundary_skip: 累加时跳过的边界层数（默认 2，与 C++ 一致）。
        mask:          可选 (T,Y,X) 或 (Y,X) 权重，仅在被选点上累加。
    Returns:
        ReferenceFrameResult，params 形状 (T,3)=(a,b,c)。
    """
    data = _as_numpy(field)                       # (T,Y,X,2)
    T, Y, X, _ = data.shape
    dx, dy = float(field.gridInterval[0]), float(field.gridInterval[1])
    dt = float(field.timeInterval)
    Xg, Yg = _coords_2d(field)                    # (Y,X)
    vt_all = _dvdt(data, dt)                       # (T,Y,X,2)
    mask = _normalize_mask(mask, (T, Y, X))

    # W(x) (Y,X,2,3)，与时间无关
    W = np.zeros((Y, X, 2, 3), dtype=np.float64)
    W[:, :, 0, 0] = 1.0
    W[:, :, 1, 1] = 1.0
    W[:, :, 0, 2] = -Yg
    W[:, :, 1, 2] = Xg

    # 内部区域权重（跳过边界层）
    base_w = np.zeros((Y, X), dtype=np.float64)
    s = max(0, int(boundary_skip))
    if Y - 2 * s > 0 and X - 2 * s > 0:
        base_w[s:Y - s, s:X - s] = 1.0
    else:
        base_w[:] = 1.0

    u_out = np.zeros_like(data)
    params = np.zeros((T, 3), dtype=np.float64)

    for t in range(T):
        slice_v = data[t]                          # (Y,X,2)
        J = _jacobian_2d(slice_v, dx, dy)          # (Y,X,2,2)
        vt = vt_all[t]                             # (Y,X,2)

        # L = ∇v·W − ∇WV
        L = np.einsum("yxik,yxkj->yxij", J, W)     # (Y,X,2,3)
        L[:, :, 0, 2] -= -slice_v[:, :, 1]         # 减去 ∇WV.col2 = (−v_y, v_x)
        L[:, :, 1, 2] -= slice_v[:, :, 0]

        # C = [L | −W]  (Y,X,2,6)
        C = np.concatenate([L, -W], axis=-1)

        w = base_w if mask is None else base_w * mask[t]
        ATA = np.einsum("yxki,yxkj,yx->ij", C, C, w)          # (6,6)
        Atb = np.einsum("yxki,yxk,yx->i", C, -vt, w)          # (6,)  RHS = Cᵀ·(−v_t)

        sol = _solve_sym_small(ATA, Atb)           # [q; q̇]
        a, b, c = sol[0], sol[1], sol[2]
        params[t] = (a, b, c)

        # 观察者 u = (a − c·y, b + c·x)，全网格重建
        u_out[t, :, :, 0] = a - c * Yg
        u_out[t, :, :, 1] = b + c * Xg

    vmu = data - u_out
    return ReferenceFrameResult(
        u_field=_make_like_2d(field, u_out),
        v_minus_u_field=_make_like_2d(field, vmu),
        params=params, mode="killing_2d")


# ═══════════════════════════════════════════════════════════════════════════
# 2. Killing mode — 3D 全局 (KillingLeastSquareOptimization3d)
# ═══════════════════════════════════════════════════════════════════════════
def killing_optimization_3d(field: UnsteadyVectorField3D,
                            mask=None) -> ReferenceFrameResult:
    """3D Killing 参考系优化（逐时间片对整场拟合刚体运动 u = t + w×x）。

    = 最小化 observed time derivative（2024 写法省略 −∂u/∂t 项）的平方，
    残差 Σ_x ‖ ∂v/∂t + ∇v·u − ∇u·v ‖² = Σ_x ‖ A·[t;w] + v_t ‖²，
    其中 A=[ J | skew(v)−J·skew(x) ]，RHS=−v_t。

    Args:
        field: 非定常 3D 流场，field.field 形状 (T,Z,Y,X,3)。
        mask:  可选 (T,Z,Y,X) 或 (Z,Y,X)，仅在活动点上累加。
    Returns:
        ReferenceFrameResult，params 形状 (T,6)=(tx,ty,tz,wx,wy,wz)。
    """
    data = _as_numpy(field)                        # (T,Z,Y,X,3)
    T, Z, Y, X, _ = data.shape
    gi = np.asarray(field.gridInterval, dtype=np.float64)
    dx, dy, dz = float(gi[0]), float(gi[1]), float(gi[2])
    dt = float(field.timeInterval)
    Xg, Yg, Zg = _coords_3d(field)                 # (Z,Y,X)
    pos = np.stack([Xg, Yg, Zg], axis=-1)          # (Z,Y,X,3)
    skewX = _skew_field(pos)                        # (Z,Y,X,3,3)，与时间无关
    vt_all = _dvdt(data, dt)
    mask = _normalize_mask(mask, (T, Z, Y, X))

    u_out = np.zeros_like(data)
    params = np.zeros((T, 6), dtype=np.float64)

    for t in range(T):
        slice_v = data[t]                          # (Z,Y,X,3)
        J = _jacobian_3d(slice_v, dx, dy, dz)      # (Z,Y,X,3,3)
        skewV = _skew_field(slice_v)               # (Z,Y,X,3,3)
        vt = vt_all[t]                             # (Z,Y,X,3)

        # A = [ J | skew(V) − J·skew(X) ]  (Z,Y,X,3,6)
        blockW = skewV - np.einsum("zyxik,zyxkj->zyxij", J, skewX)
        A = np.concatenate([J, blockW], axis=-1)   # (Z,Y,X,3,6)
        b = -vt                                     # (Z,Y,X,3)

        if mask is None:
            MTM = np.einsum("zyxki,zyxkj->ij", A, A)
            MTb = np.einsum("zyxki,zyxk->i", A, b)
        else:
            w = mask[t]
            MTM = np.einsum("zyxki,zyxkj,zyx->ij", A, A, w)
            MTb = np.einsum("zyxki,zyxk,zyx->i", A, b, w)

        q = _solve_sym_small(MTM, MTb)             # [t(3); w(3)]
        trans, omega = q[:3], q[3:]
        params[t] = q

        # u(x) = t + w×x，全网格重建
        u_out[t] = trans + np.cross(omega, pos)

    vmu = data - u_out
    return ReferenceFrameResult(
        u_field=_make_like_3d(field, u_out),
        v_minus_u_field=_make_like_3d(field, vmu),
        params=params, mode="killing_3d")


# ═══════════════════════════════════════════════════════════════════════════
# 3. Non-killing mode — 2D (runGunther17Optimization, Objectivity)
# ═══════════════════════════════════════════════════════════════════════════
def gunther17_optimization_2d(field: UnsteadyVectorField2D,
                              neighborhood: int = 3,
                              use_sat: bool = True,
                              mask=None) -> ReferenceFrameResult:
    """2D Günther 2017 客观化（逐点邻域最小二乘，Objectivity 模式，systemSize=6）。

    每点 M(2×6)=[Jxpvp | J.col0 | J.col1 | e_x | e_y | Xp]，RHS b=∂v/∂t，
    邻域内累加 MᵀM、Mᵀb（积分图 O(1)），解 uu，重建 v_hat = v + (uu1,uu2) − uu0·Xp。

    Args:
        field:        非定常 2D 流场。
        neighborhood: 邻域半径 U（窗口 (2U+1)²）。
        use_sat:      保留参数以对齐 C++ API；numpy 实现始终用积分图求邻域和。
        mask:         可选 (T,Y,X) 或 (Y,X)，屏蔽点不进入邻域累加。
    Returns:
        ReferenceFrameResult（params=None）。
    """
    data = _as_numpy(field)                        # (T,Y,X,2)
    T, Y, X, _ = data.shape
    dx, dy = float(field.gridInterval[0]), float(field.gridInterval[1])
    dt = float(field.timeInterval)
    Xg, Yg = _coords_2d(field)                     # (Y,X)
    Xp = np.stack([-Yg, Xg], axis=-1)              # (Y,X,2)  位置 90° 旋转
    vt_all = _dvdt(data, dt)
    mask = _normalize_mask(mask, (T, Y, X))
    U = max(0, int(neighborhood))

    u_out = np.zeros_like(data)
    for t in range(T):
        slice_v = data[t]                          # (Y,X,2)
        J = _jacobian_2d(slice_v, dx, dy)          # (Y,X,2,2)
        vt = vt_all[t]                             # (Y,X,2)
        Vp = np.stack([-slice_v[:, :, 1], slice_v[:, :, 0]], axis=-1)     # (−v_y, v_x)
        Jxpvp = -np.einsum("yxik,yxk->yxi", J, Xp) + Vp                    # (Y,X,2)

        # M (Y,X,2,6)
        M = np.zeros((Y, X, 2, 6), dtype=np.float64)
        M[:, :, 0, 0] = Jxpvp[:, :, 0]
        M[:, :, 1, 0] = Jxpvp[:, :, 1]
        M[:, :, 0, 1] = J[:, :, 0, 0]
        M[:, :, 0, 2] = J[:, :, 0, 1]
        M[:, :, 1, 1] = J[:, :, 1, 0]
        M[:, :, 1, 2] = J[:, :, 1, 1]
        M[:, :, 0, 3] = 1.0
        M[:, :, 1, 4] = 1.0
        M[:, :, 0, 5] = Xp[:, :, 0]
        M[:, :, 1, 5] = Xp[:, :, 1]

        MTM = np.einsum("yxki,yxkj->yxij", M, M)   # (Y,X,6,6)
        MTb = np.einsum("yxki,yxk->yxi", M, vt)    # (Y,X,6)  RHS = +∂v/∂t
        if mask is not None:
            w = mask[t][:, :, None, None]
            MTM = MTM * w
            MTb = MTb * w[..., 0]

        MTM_win = _windowed_sum_nd(MTM, U, 2)
        MTb_win = _windowed_sum_nd(MTb, U, 2)
        uu = _batched_solve(MTM_win, MTb_win)      # (Y,X,6)

        # v_hat = v + (uu1, uu2) − uu0·Xp
        u0 = uu[:, :, 0]
        u_out[t, :, :, 0] = slice_v[:, :, 0] + uu[:, :, 1] - u0 * Xp[:, :, 0]
        u_out[t, :, :, 1] = slice_v[:, :, 1] + uu[:, :, 2] - u0 * Xp[:, :, 1]

    # u_out 此时存的是 v_hat；换算 observer u = v − v_hat
    vmu = u_out
    u_field_data = data - vmu
    return ReferenceFrameResult(
        u_field=_make_like_2d(field, u_field_data),
        v_minus_u_field=_make_like_2d(field, vmu),
        params=None, mode="gunther17_2d")


# ═══════════════════════════════════════════════════════════════════════════
# 4. Non-killing mode — 3D (GenericLocalOptimization3d, Objective)
# ═══════════════════════════════════════════════════════════════════════════
def gunther17_optimization_3d(field: UnsteadyVectorField3D,
                              neighborhood: int = 3,
                              use_sat: bool = True,
                              mask=None) -> ReferenceFrameResult:
    """3D Günther 2017 客观化（逐点邻域最小二乘，Objective 模式，systemSize=12）。

    每点 M(3×12)=[F | J | skew(X) | −I]，F=−J·skew(X)+skew(V)，RHS b=∂v/∂t，
    立方邻域累加（3D 积分图），解 uu，重建 v_hat = V + ω×X + c，其中 ω=uu[0:3]、c=uu[3:6]。

    Args:
        field:        非定常 3D 流场，field.field 形状 (T,Z,Y,X,3)。
        neighborhood: 邻域半径 U（窗口 (2U+1)³）。
        use_sat:      保留参数以对齐 C++ API；numpy 实现始终用积分图。
        mask:         可选 (T,Z,Y,X) 或 (Z,Y,X)，屏蔽点不进入邻域累加。
    Returns:
        ReferenceFrameResult（params=None）。
    """
    data = _as_numpy(field)                        # (T,Z,Y,X,3)
    T, Z, Y, X, _ = data.shape
    gi = np.asarray(field.gridInterval, dtype=np.float64)
    dx, dy, dz = float(gi[0]), float(gi[1]), float(gi[2])
    dt = float(field.timeInterval)
    Xg, Yg, Zg = _coords_3d(field)
    pos = np.stack([Xg, Yg, Zg], axis=-1)          # (Z,Y,X,3)
    skewX = _skew_field(pos)                        # (Z,Y,X,3,3)
    negI = -np.eye(3)
    vt_all = _dvdt(data, dt)
    mask = _normalize_mask(mask, (T, Z, Y, X))
    U = max(0, int(neighborhood))

    u_out = np.zeros_like(data)
    for t in range(T):
        slice_v = data[t]                          # (Z,Y,X,3)
        J = _jacobian_3d(slice_v, dx, dy, dz)      # (Z,Y,X,3,3)
        skewV = _skew_field(slice_v)               # (Z,Y,X,3,3)
        vt = vt_all[t]                             # (Z,Y,X,3)
        F = -np.einsum("zyxik,zyxkj->zyxij", J, skewX) + skewV            # (Z,Y,X,3,3)

        # M (Z,Y,X,3,12) = [F | J | skew(X) | −I]
        negI_b = np.broadcast_to(negI, (Z, Y, X, 3, 3))
        M = np.concatenate([F, J, skewX, negI_b], axis=-1)               # (Z,Y,X,3,12)

        MTM = np.einsum("zyxki,zyxkj->zyxij", M, M)   # (Z,Y,X,12,12)
        MTb = np.einsum("zyxki,zyxk->zyxi", M, vt)    # (Z,Y,X,12)  RHS = +∂v/∂t
        if mask is not None:
            w = mask[t][..., None, None]
            MTM = MTM * w
            MTb = MTb * w[..., 0]

        MTM_win = _windowed_sum_nd(MTM, U, 3)
        MTb_win = _windowed_sum_nd(MTb, U, 3)
        uu = _batched_solve(MTM_win, MTb_win)      # (Z,Y,X,12)

        omega = uu[..., 0:3]                        # (Z,Y,X,3)
        cvec = uu[..., 3:6]
        # v_hat = V + ω×X + c
        u_out[t] = slice_v + np.cross(omega, pos) + cvec

    # u_out 存的是 v_hat；observer u = v − v_hat
    vmu = u_out
    u_field_data = data - vmu
    return ReferenceFrameResult(
        u_field=_make_like_3d(field, u_field_data),
        v_minus_u_field=_make_like_3d(field, vmu),
        params=None, mode="gunther17_3d")


__all__ = [
    "ReferenceFrameResult",
    "killing_optimization_2d",
    "killing_optimization_3d",
    "gunther17_optimization_2d",
    "gunther17_optimization_3d",
]
