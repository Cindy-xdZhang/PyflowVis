import torch
from torch.nn.functional import upsample
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from FLowUtils.VectorField2d import UnsteadyVectorField2D
import os,logging,hashlib
import numpy as np
import matplotlib.pyplot as plt  # used by the visualize_* helpers below
from DeepUtils.utils.stable_hash import stable_hash
from FLowUtils.ScalarField2d import ScalarField2D,ScalarFieldManager
from FMT_Utils.FlowlinePostProcessing import AngleAwareSampling, LocLines, temporal_downsamplePathlineCrossPrimitiveRegular
from FLowUtils.flowlineIntegral import batch_pathlineCross_integration_2D_auto
from FLowUtils.flowDatasetUtils.NetCDF_AmiraLoader import load_UnsteadyVectorFields_general
from FMT_Utils import debug_checks as dbg


GLOBAL_UniformValueTemporalAndSpatial=15.0


def flowmap_to_relative(fm, scale: float):
    """Jacobian-aware representation for flow-map upsampling.

    A flow map cell is [..., 5, 2, 3] = (5 cross lines: center,x+,x-,y+,y-) x
    (2 endpoints: head,tail) x (3: x,y,t). FTLE depends on the *difference* between
    neighbour-line endpoints (~O(2*offset_dist)=O(scale)) hidden inside O(1) absolute
    positions, so plain position MSE cannot resolve it.

    This rewrites the xy of the 4 neighbour lines as their offset from the center line,
    divided by `scale`, lifting the FTLE-relevant signal to O(1) so MSE weights it fairly.
    The center line (index 0) keeps absolute xy; the t channel is left untouched. The map
    is exactly invertible via `flowmap_from_relative` with the same `scale`.
    """
    out = fm.clone() if isinstance(fm, torch.Tensor) else fm.copy()
    center_xy = fm[..., 0:1, :, 0:2]                       # [...,1,2,2] center head/tail xy
    out[..., 1:, :, 0:2] = (fm[..., 1:, :, 0:2] - center_xy) / float(scale)
    return out


def flowmap_from_relative(fm_rel, scale: float):
    """Inverse of `flowmap_to_relative`: rebuild absolute (x,y) for the 4 neighbour lines."""
    out = fm_rel.clone() if isinstance(fm_rel, torch.Tensor) else fm_rel.copy()
    center_xy = fm_rel[..., 0:1, :, 0:2]
    out[..., 1:, :, 0:2] = fm_rel[..., 1:, :, 0:2] * float(scale) + center_xy
    return out


torch.set_printoptions(precision=4, threshold=10000, linewidth=200, sci_mode=False)

#every scalar field is a 2d-time volume.
# we can generate thousands of training samples from this scalar field.
# each sample is a ((pos2d position+time), label from scalar field)
def load_FTLE_npz_as_scalar_fields(path: str) -> list[ScalarField2D]:
    mgr = ScalarFieldManager()
    result: list[ScalarField2D] = []
    if os.path.isdir(path):
        for file in os.listdir(path):
            if file.lower().endswith(".npz"):
                sf = mgr.load_scalar_field_from_file(os.path.join(path, file))
                result.append(sf)
    else:
        if path.lower().endswith(".npz") and os.path.exists(path):
            sf = mgr.load_scalar_field_from_file(path)
            result.append(sf)
    return result


def check_group_have_same_lengths(valid_steps: torch.Tensor, nerbors: int = 5):
    """
    valid_steps: [M], where M = N*nerbors, ordered as (seed0's 5 lines, seed1's 5 lines, ...)
    Returns:
      - same:        [N] whether each group has identical lengths
      - bad_groups:  [B] indices of groups with unequal lengths
      - g_min, g_max:[N] min and max steps in each group

    Note: we only check the equality of lengths inside each group here. After temporal sampling, you should check again whether the resulting length meets the target.
    """
    assert valid_steps.numel() % nerbors == 0, "lines count must be multiple of nerbors"
    g = valid_steps.view(-1, nerbors)            # [N, 5]
    same = (g == g[:, [0]]).all(dim=1)           # 行内是否全等
    bad_groups = (~same).nonzero(as_tuple=False).squeeze(1)  # [B]
    g_min = g.min(dim=1).values
    g_max = g.max(dim=1).values
    return same, bad_groups, g_min, g_max

def select_good_groups(valid_steps: torch.Tensor, nerbors: int, K: int, strict: bool = True):
    """
    Select groups that satisfy:
      - same == True (all 5 lines in a group have identical valid lengths)
      - g_min > K (strict=True) or g_min >= K (strict=False)

    Returns:
      - keep_groups: [N]  group-level boolean mask
      - keep_lines:  [M]  per-line boolean mask (group mask repeated nerbors times)
      - good_idx:    [G]  indices of valid groups
    """
    same, _, g_min, _ = check_group_have_same_lengths(valid_steps, nerbors)
    cond_len = (g_min > K) if strict else (g_min >= K)
    # keep_groups = same & cond_len                     # [N]
    keep_groups = cond_len                     # [N]
    good_idx = torch.nonzero(keep_groups, as_tuple=False).squeeze(1)  # [G]
    keep_lines = keep_groups.repeat_interleave(nerbors)               # [M]
    return keep_groups, keep_lines, good_idx

def computeFTLEFromPathlineCrossPrimitive(points_grouped: torch.Tensor,
                                         vectorfield_dt: float=0.05 ) -> torch.Tensor:
    """
    Classical FTLE estimation based on cross-primitive (5 pathlines).
    Ensure time values are in physical time range (not normalized time).
    Inputs:
      - points_grouped: [N, sample_nerbors, (line_steps=2), 3], sample_nerbors is ordered as (center, x+, x-, y+, y-)
      - sample_nerbors: 5 (cross)
      - vectorfield_dt: physical time step to validate effective time span
    Output:
      - ftle: [N] FTLE for the center sample in each group
    """
    assert points_grouped.dim() == 4 and points_grouped.shape[-1] >= 2 and points_grouped.shape[2] >= 2
    NumberofCross,NumberOfNeighbors,NumberOfLineSteps,Dim=points_grouped.shape

    # Initial and final positions (physical coordinates)
    # Neighbor order convention: 0=center, 1=x+, 2=x-, 3=y+, 4=y-
    pStart= points_grouped[:, :, 0, :2]  # [N,5,2]
    pEnd = points_grouped[:, :, -1, :2]  # [N,5,2]

    # Initial offsets (scalars, axis-aligned)
    dx0 = (pStart[:, 1, 0] - pStart[:, 2, 0]).clamp_min(1e-12)  # [N]
    dy0 = (pStart[:, 3, 1] - pStart[:, 4, 1]).clamp_min(1e-12)  # [N]

    # Flow map Jacobian J = [dPhi/dx0, dPhi/dy0] (as column vectors)
    dPhi_dx = (pEnd[:, 1, :2] - pEnd[:, 2, :2]) / dx0.unsqueeze(-1)
    dPhi_dy = (pEnd[:, 3, :2] - pEnd[:, 4, :2]) / dy0.unsqueeze(-1)
    J = torch.stack([dPhi_dx, dPhi_dy], dim=-1)  # [N,2,2]
    # Cauchy–Green tensor C = J^T J
    JT = J.transpose(1, 2)
    C = torch.bmm(JT, J)  # [N,2,2]

    # Largest eigenvalue (C is symmetric positive semidefinite)
    eigvals = torch.linalg.eigvalsh(C)  # [N,2], 升序
    lambda_max = eigvals[:, 1].clamp_min(1e-12)

    # Physical time span ΔT (use center line time)
    t_norm_start = points_grouped[:, 0, 0, -1]
    t_norm_end   = points_grouped[:, 0, -1, -1]
    T_phys = (t_norm_end - t_norm_start).abs() 
    valid = T_phys >= 2 * vectorfield_dt
    T_safe = T_phys.clamp_min(1e-12)
    ftle_raw = 0.5 * torch.log(lambda_max) / T_safe

    ftle = torch.where(valid, ftle_raw, torch.zeros_like(ftle_raw))
    return ftle



def sample_random_starts(vectorfield: UnsteadyVectorField2D, count: int) -> torch.Tensor:
    xrange = vectorfield.domainMaxBoundary[0] - vectorfield.domainMinBoundary[0]
    yrange = vectorfield.domainMaxBoundary[1] - vectorfield.domainMinBoundary[1]
    xmin, ymin, _ = vectorfield.domainMinBoundary+0.01*np.array([xrange, yrange, 0])
    xmax, ymax, _ = vectorfield.domainMaxBoundary-0.01*np.array([xrange, yrange, 0])

    rx = torch.rand(count)
    ry = torch.rand(count)
    xs = xmin + rx * (xmax - xmin)
    ys = ymin + ry * (ymax - ymin)
    return torch.stack([xs, ys], dim=1)  # [count,2]


def generate_training_samples(
    vectorfield: UnsteadyVectorField2D,
    count: int,
    max_steps: int,
    dt: float,
    t_start: float,
    t_target: float,
    offset_dist: float,
    nerbors: int = 5
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly sample seeds online, generate cross pathlines, and discard groups with any line shorter than max_steps.
    Re-sample until we collect exactly 'count' groups with full length (max_steps).
    If cacheSystem is True, build a unique tag by parameters and cache to outputs/temp/{tag}.npz so next time we can directly load.
    Returns:
      - P_all:     [count, nerbors, max_steps, 3]
      - valid_all: [count, nerbors] (equals max_steps everywhere)
    """

    collected_groups = 0
    kept_P = []
    kept_V = []

    while collected_groups < count:
        need = count - collected_groups
        runSize=max(need, 16)
        starts_xy = sample_random_starts(vectorfield, count=runSize)
        P_b, PathlineLength_b = batch_pathlineCross_integration_2D_auto(
            points=starts_xy.numpy(),
            vectorfield=vectorfield,
            t_start=float(t_start), t_target=float(t_target),
            dt=float(dt), max_steps=int(max_steps),
            offsets_size=float(offset_dist), method="rk4"
        )
        if P_b.numel() == 0:# number of elements in the tensor
            continue

        TotalPathlines, PathlineData, Dimensions = P_b.shape
        assert TotalPathlines% nerbors == 0, "lines count must be multiple of nerbors"
        assert PathlineData == max_steps, "PathlineData must be equal to max_steps"
        GroupCount = TotalPathlines // nerbors
        P_g = P_b.view(GroupCount, nerbors, max_steps, Dimensions)
        PathlineLength_g = PathlineLength_b.view(GroupCount, nerbors)
        good_mask = (PathlineLength_g == max_steps).all(dim=1)  # [G]
        good_idx = torch.nonzero(good_mask, as_tuple=False).squeeze(1)
        if good_idx.numel() == 0:
            continue

        take = int(min(need, good_idx.numel()))
        good_idx = good_idx[:take]

        P_sel = P_g[good_idx]
        V_sel = PathlineLength_g[good_idx]

        kept_P.append(P_sel)
        kept_V.append(V_sel)
        collected_groups += take

    P_all = torch.cat(kept_P, dim=0)
    V_all = torch.cat(kept_V, dim=0)
    # Keep only the first 'count' groups (defensive slicing)
    P_all = P_all[:count * nerbors]
    V_all = V_all[:count * nerbors]
   
    return P_all, V_all
        


def generate_seedingGrid_2D(vectorfield: UnsteadyVectorField2D,resolutionUPsampling:float,boundary_offset:float=0.1):
    xmin, ymin, _ = vectorfield.domainMinBoundary
    xmax, ymax, _ = vectorfield.domainMaxBoundary
    xmin_grid=xmin+boundary_offset*(xmax-xmin)
    ymin_grid=ymin+boundary_offset*(ymax-ymin)  
    xmax_grid=xmax-boundary_offset*(xmax-xmin)
    ymax_grid=ymax-boundary_offset*(ymax-ymin)
    assert resolutionUPsampling>0.01

    # Use linspace to specify number of samples instead of step size
    num_x = int(round(vectorfield.Xdim * resolutionUPsampling))
    num_y = int(round(vectorfield.Ydim * resolutionUPsampling))
    xs = np.linspace(xmin_grid, xmax_grid, num_x, dtype=np.float32)
    ys = np.linspace(ymin_grid, ymax_grid, num_y, dtype=np.float32)

    #np.meshgrid(xs, ys) uses indexing='xy' by default, 
    # so both XX and YY have shapes (ny, nx). 
    # After flattening by C-order, the rows and columns of linear index i are row=i//nx and col=i%nx,
    #  with X changing rapidly in the column direction and Y changing slowly in the row direction.
    XX, YY = np.meshgrid(xs, ys)
    starts_xy = np.stack([XX.reshape(-1), YY.reshape(-1)], axis=1)
    return starts_xy,xs,ys



def generate_Flowmap_SLICE(vectorfield: UnsteadyVectorField2D, physcial_time: float, dt: float,
                           maxIterations: int, offesetDist: float, resolutionUPsampling: float = 1.0):
    """
    生成一张时间切片的 flow map（直接输出 pathline 的首/尾点位置，不计算 FTLE，不做归一化）。

    与 `generate_FTLE_SLICE` 的关系: FTLE 本质就是这里 flow map 的派生量
    （对每个格点的 pathline 取首尾 + 线性变化求 Cauchy-Green 特征值开根号，见
    `computeFTLEFromPathlineCrossPrimitive` / `FTLEFromFlowMap`）。本函数把内部 pathline
    的首尾点直接输出, 去掉了 FTLE 计算部分, 供 flow map 上采样训练使用。

    与旧的 `generate_FLowMap_SLICE` 的区别:
        - 保留全部 5 个 neighbor（不再只取中心点）;
        - 输出首/尾两个点的 **原始物理坐标** (x,y,t)，形状 [..., 2, 3]，而非位移、也不归一化。

    返回:
        flowmap_field:    torch.FloatTensor [ny*nx, nerbors=5, 2, 3]  每个格点 5 条 pathline 的(首点,尾点)原始坐标
        Pathline_g:       torch.FloatTensor [ny*nx, nerbors, max_steps, 3]
        PathlineLength_g: torch.Tensor      [ny*nx, nerbors]
        nx, ny: x/y 方向的格点数
    """
    nerbors = 5
    offset_dist = float(offesetDist)
    max_steps = int(maxIterations)
    starts_xy, xs, ys = generate_seedingGrid_2D(vectorfield, resolutionUPsampling)
    ny, nx = len(ys), len(xs)
    Pathline_b, PathlineLength_b = batch_pathlineCross_integration_2D_auto(
        points=starts_xy,
        vectorfield=vectorfield,
        t_start=float(physcial_time), t_target=float(physcial_time + dt * maxIterations),
        dt=float(dt), max_steps=int(max_steps),
        offsets_size=float(offset_dist), method="rk4"
    )
    Pathline_g = Pathline_b.view(nx * ny, nerbors, max_steps, 3)
    PathlineLength_g = PathlineLength_b.view(nx * ny, nerbors)

    # 首点(积分起点)与尾点(积分终点)的原始坐标, 不做归一化
    head = Pathline_g[:, :, 0, :]              # [ny*nx, 5, 3]
    tail = Pathline_g[:, :, max_steps - 1, :]  # [ny*nx, 5, 3]
    flowmap_field = torch.stack([head, tail], dim=2).contiguous()  # [ny*nx, 5, 2, 3]

    # 未完整积分的 group(任一 cross line 长度 != max_steps)其尾点是 0 填充的垃圾值, 会污染
    # 由其派生的 FTLE。与 generate_FTLE_SLICE 一致: 把这些 group 整体置零, 经
    # computeFTLEFromPathlineCrossPrimitive 的 T_phys<2*dt 检查后 FTLE 会自然变成 0。
    keep_groups_full = (PathlineLength_g == max_steps).all(dim=1)  # [ny*nx]
    flowmap_field[~keep_groups_full] = 0.0

    return flowmap_field, Pathline_g, PathlineLength_g, nx, ny


def generate_FTLE_SLICE(cfg,vectorfield: UnsteadyVectorField2D,physcial_time:float,dt:float,maxIterations:int, resolutionUPsampling:float=1.0):
    nerbors=5
    offset_dist = float(cfg.pcds.offset_dist)
    starts_xy,xs,ys=generate_seedingGrid_2D(vectorfield,resolutionUPsampling)
    ny, nx = len(ys), len(xs)
    grid_low=np.zeros((ny,nx),dtype=np.float32)
    Pathline_b, PathlineLength_b = batch_pathlineCross_integration_2D_auto(
                points=starts_xy,
                vectorfield=vectorfield,
                t_start=float(physcial_time), t_target=float(physcial_time+dt*maxIterations),
                dt=float(dt), max_steps=int(maxIterations),
                offsets_size=float(offset_dist), method="rk4"
            )
    Pathline_g = Pathline_b.view(nx*ny, nerbors, maxIterations, 3)
    PathlineLength_g = PathlineLength_b.view(nx*ny, nerbors)
    y_all=computeFTLEFromPathlineCrossPrimitive(Pathline_g, vectorfield_dt=vectorfield.timeInterval)

    keep_groups_full = (PathlineLength_g == maxIterations).all(dim=1)
    true_grid = np.full((ny, nx), 0, dtype=np.float32)
    linear_index=np.arange(nx*ny)
    valid_index=linear_index[keep_groups_full]
    rows=valid_index//nx
    cols=valid_index%nx
    true_grid[rows, cols] = y_all[valid_index].detach().cpu().numpy()
    # Stage-1 debug: verify the FTLE field is finite, non-degenerate, and that a
    # reasonable fraction of pathlines stayed valid (catches integration overshoot).
    dbg.check_ftle_field(f"FTLE_slice(res~{resolutionUPsampling},t={physcial_time:.3f})",
                         true_grid, expected_shape=(ny, nx))
    return true_grid,Pathline_g,nx,ny





def generate_test_points(cfg,vectorfield: UnsteadyVectorField2D,physcial_time:float,target_time:float, cacheSystem: bool = True):
    nerbors=int(cfg.pcds.num_cross_points_per_seeding)
    max_steps = int(cfg.pcds.max_iterations)
    dt = float(cfg.pcds.dt)
    offset_dist = float(cfg.pcds.offset_dist)
    localized = bool(cfg.pcds.localized)
    normalized = bool(cfg.pcds.normalized)
    resolutionUPsampling=float(cfg.resolutionUPsampling)
    starts_xy,xs,ys=generate_seedingGrid_2D(vectorfield,resolutionUPsampling)
    ny, nx = len(ys), len(xs)
    if cacheSystem:
        dom_min = tuple(map(float, vectorfield.domainMinBoundary))
        dom_max = tuple(map(float, vectorfield.domainMaxBoundary))
        key_str = (
            f"neighbors={nerbors}|ups={resolutionUPsampling}|ms={max_steps}|dt={dt:.8g}|ts={physcial_time:.8g}|tt={target_time:.8g}|"
            f"off={offset_dist:.8g}|nb={offset_dist}|tmin={float(vectorfield.tmin):.8g}|tmax={float(vectorfield.tmax):.8g}|"
            f"domMin={dom_min}|domMax={dom_max}|localized={localized}|normalized={normalized}"
        )
        tag = "TestSlice_" + hashlib.md5(key_str.encode("utf-8")).hexdigest()[:16]
        cache_dir = os.path.join("./outputs", "temp")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{tag}.npz")
        if os.path.exists(cache_path):
            try:
                data = np.load(cache_path)
                P_np = data["P"]
                V_np = data["V"]
                Pathline_g = torch.from_numpy(P_np).float()
                PathlineLength_g = torch.from_numpy(V_np).to(torch.int32)
                return Pathline_g, PathlineLength_g,nx,ny
            except Exception as e:
                print(f"[generate_test_points] cache load failed: {e}. Regenerating...")
   
   
    Pathline_b, PathlineLength_b = batch_pathlineCross_integration_2D_auto(
                points=starts_xy,
                vectorfield=vectorfield,
                t_start=float(physcial_time), t_target=float(target_time),
                dt=float(dt), max_steps=int(max_steps),
                offsets_size=float(offset_dist), method="rk4"
            )
    Pathline_g = Pathline_b.view(nx*ny, nerbors, max_steps, 3)
    PathlineLength_g = PathlineLength_b.view(nx*ny, nerbors)

    # Save cache
    if cacheSystem:
        try:
            # 使用同样的 tag 构造路径（若前面加载失败，需重新构造）
            dom_min = tuple(map(float, vectorfield.domainMinBoundary))
            dom_max = tuple(map(float, vectorfield.domainMaxBoundary))
            key_str = (
               f"neighbors={nerbors}|ups={resolutionUPsampling}|ms={max_steps}|dt={dt:.8g}|ts={physcial_time:.8g}|tt={target_time:.8g}|"
            f"off={offset_dist:.8g}|nb={offset_dist}|tmin={float(vectorfield.tmin):.8g}|tmax={float(vectorfield.tmax):.8g}|"
            f"domMin={dom_min}|domMax={dom_max}|localized={localized}|normalized={normalized}"
            )
            tag = "TestSlice_" + hashlib.md5(key_str.encode("utf-8")).hexdigest()[:16]
            cache_dir = os.path.join("outputs", "temp")
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, f"{tag}.npz")
            np.savez(cache_path,
                     P=Pathline_g.detach().cpu().numpy().astype(np.float32),
                     V=PathlineLength_g.detach().cpu().numpy().astype(np.int32),
                     meta=key_str)
        except Exception as e:
            print(f"[generate_test_points] cache save failed: {e}")
    return Pathline_g, PathlineLength_g,nx,ny
    



# ---------------- IVD slice generation (low-res scalar + low-res pathlines) ----------------
def generate_IVD_SLICE(cfg, vectorfield: UnsteadyVectorField2D, physcial_time: float,
                       dt: float, maxIterations: int, resolutionUPsampling: float = 1.0):
    """
    生成 IVD 标量场切片（按给定采样分辨率）以及对应的低分辨率 pathline groups。

    返回:
        ivd_grid: np.ndarray [ny, nx]
        Pathline_g: torch.Tensor [nx*ny, nerbors, max_steps, 3]
        xs, ys: 物理坐标网格轴（用于外部对齐）
    """
    nerbors = int(cfg.pcds.num_cross_points_per_seeding)
    max_steps = int(cfg.pcds.max_iterations)
    assert max_steps == int(maxIterations)
    offset_dist = float(cfg.pcds.offset_dist)

    # 1) 生成低分辨率网格采样点（物理坐标）
    starts_xy, xs, ys = generate_seedingGrid_2D(vectorfield, resolutionUPsampling)
    ny, nx = len(ys), len(xs)

    # 2) 计算 IVD 三维场，并在给定时间切片上按物理坐标采样
    #    IVD: (T, Y, X) 对应 vectorfield 的网格与时间
    ivd_3d = compute_ivd_2D(vectorfield)
    ivd_field = ScalarField2D(vectorfield.Xdim, vectorfield.Ydim, vectorfield.time_steps,
                              vectorfield.domainMinBoundary, vectorfield.domainMaxBoundary,
                              dtype=ivd_3d.dtype)
    ivd_field.set_discrete_data(ivd_3d)

    ivd_grid = np.zeros((ny, nx), dtype=np.float32)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            ivd_grid[j, i] = float(ivd_field.get_value_at_physical_pos(float(x), float(y), float(physcial_time)))

    # 3) 生成与低分辨率网格对应的 pathlines（供模型特征输入使用，结构与 FTLE 保持一致）
    Pathline_b, PathlineLength_b = batch_pathlineCross_integration_2D_auto(
        points=starts_xy,
        vectorfield=vectorfield,
        t_start=float(physcial_time), t_target=float(physcial_time + dt * maxIterations),
        dt=float(dt), max_steps=int(max_steps),
        offsets_size=float(offset_dist), method="rk4"
    )
    Pathline_g = Pathline_b.view(nx * ny, nerbors, max_steps, 3)

    return ivd_grid, Pathline_g, xs, ys

def sample_center_points_from_groups(points_grouped: torch.Tensor, sample_nerbors: int, line_steps: int) -> torch.Tensor:
    B = points_grouped.shape[0]
    pts = points_grouped.reshape(B, sample_nerbors, line_steps + 1, -1)
    center = pts[:, 0, 0, :]
    return center

def preprocess_localization_normalization(points_grouped: torch.Tensor, sample_nerbors: int, line_steps: int,
                               localized: bool, normalized: bool) -> torch.Tensor:
    x = points_grouped
    if localized:
        x = LocLines(sample_nerbors=sample_nerbors, points=x)
    # if normalized:
    #     x = normalizeLines(sample_nerbors=sample_nerbors, points=x, vectorfield=vectorfield)
    return x

def visualize_twoftle_slices(true_grid: np.ndarray, pred_grid: np.ndarray, domainMinBoundary,domainMaxBoundary,  dpi: int = 300):
    def robust_minmax(a):
        if np.all(np.isnan(a)):
            return 0.0, 1.0
        vmin = np.nanpercentile(a, 2)
        vmax = np.nanpercentile(a, 98)
        if not np.isfinite(vmin): vmin = np.nanmin(a)
        if not np.isfinite(vmax): vmax = np.nanmax(a)
        if vmin == vmax:
            vmax = vmin + 1e-6
        return float(vmin), float(vmax)

    xmin, ymin, _ = domainMinBoundary
    xmax, ymax, _ = domainMaxBoundary
    extent = [xmin, xmax, ymin, ymax]
    vmin_t, vmax_t = robust_minmax(true_grid)
    vmin=vmin_t
    vmax=vmax_t

    fig, axes = plt.subplots(2, 1, figsize=(4, 16), constrained_layout=True, dpi=dpi)
    ims = []
    ims.append(axes[0].imshow(true_grid, origin='lower', extent=extent, cmap='coolwarm', vmin=vmin, vmax=vmax, interpolation='bilinear'))
    axes[0].set_title('FTLE A'); axes[0].set_xlabel('X'); axes[0].set_ylabel('Y'); axes[0].set_aspect('equal')
    ims.append(axes[1].imshow(pred_grid, origin='lower', extent=extent, cmap='coolwarm', vmin=vmin, vmax=vmax, interpolation='bilinear'))
    axes[1].set_title(f'FTLE B,'); axes[1].set_xlabel('X'); axes[1].set_ylabel('Y'); axes[1].set_aspect('equal')
   

    for ax, im in zip(axes, ims):
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.formatter.set_powerlimits((0, 0))
        cb.update_ticks()


    plt.show(block=True)
    plt.close(fig)


def visualize_FTLEUpampling(true_grid: np.ndarray, pred_grid: np.ndarray,low_res_grid: np.ndarray, domainMinBoundary,domainMaxBoundary,save_path: str | None = None,
                         upscale_factor: int = 1, dpi: int = 300, show: bool = True):
    def robust_minmax(a):
        if np.all(np.isnan(a)):
            return 0.0, 1.0
        vmin = np.nanpercentile(a, 2)
        vmax = np.nanpercentile(a, 98)
        if not np.isfinite(vmin): vmin = np.nanmin(a)
        if not np.isfinite(vmax): vmax = np.nanmax(a)
        if vmin == vmax:
            vmax = vmin + 1e-6
        return float(vmin), float(vmax)

    # extent from physical domain
    xmin, ymin, _ = domainMinBoundary
    xmax, ymax, _ = domainMaxBoundary
    extent = [xmin, xmax, ymin, ymax]

    # display range from true high-res
    vmin, vmax = robust_minmax(true_grid)

    # bilinear upsample low-res to high-res size
    Hh, Wh = int(true_grid.shape[0]), int(true_grid.shape[1])
    with torch.no_grad():
        lr = torch.from_numpy(low_res_grid)[None, None, ...].float()
        lr_up = torch.nn.functional.interpolate(lr, size=(Hh, Wh), mode='bilinear', align_corners=False)[0, 0]
        bilinear_grid = lr_up.cpu().numpy()

    # PSNRs
    _, _, _, psnr_pred = compute_metrics(true_grid, pred_grid)
    _, _, _, psnr_bilin = compute_metrics(true_grid, bilinear_grid)

    # plot 2x2 without colorbars
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True, dpi=dpi)
    ax00, ax01 = axes[0]
    ax10, ax11 = axes[1]

    im0 = ax00.imshow(true_grid, origin='lower', extent=extent, cmap='coolwarm', vmin=vmin, vmax=vmax, interpolation='bilinear')
    ax00.set_title('HighRes (GT)')
    ax00.set_xlabel('X'); ax00.set_ylabel('Y'); ax00.set_aspect('equal')

    im1 = ax01.imshow(pred_grid, origin='lower', extent=extent, cmap='coolwarm', vmin=vmin, vmax=vmax, interpolation='bilinear')
    ax01.set_title(f'Predict, PSNR={psnr_pred:.2f} dB')
    ax01.set_xlabel('X'); ax01.set_ylabel('Y'); ax01.set_aspect('equal')

    im2 = ax10.imshow(low_res_grid, origin='lower', extent=extent, cmap='coolwarm', interpolation='nearest')
    ax10.set_title('LowRes')
    ax10.set_xlabel('X'); ax10.set_ylabel('Y'); ax10.set_aspect('equal')

    im3 = ax11.imshow(bilinear_grid, origin='lower', extent=extent, cmap='coolwarm', vmin=vmin, vmax=vmax, interpolation='bilinear')
    ax11.set_title(f'Bilinear, PSNR={psnr_bilin:.2f} dB')
    ax11.set_xlabel('X'); ax11.set_ylabel('Y'); ax11.set_aspect('equal')

    if save_path is not None:
        plt.savefig(save_path, dpi=200)
    # if show:
    #     plt.show(block=True)
    plt.close(fig)




def visualize_OneScalarField(true_grid: np.ndarray, domainMinBoundary,domainMaxBoundary,
                         upscale_factor: int = 1, dpi: int = 300):
    def robust_minmax(a):
        if np.all(np.isnan(a)):
            return 0.0, 1.0
        vmin = np.nanpercentile(a, 2)
        vmax = np.nanpercentile(a, 98)
        if not np.isfinite(vmin): vmin = np.nanmin(a)
        if not np.isfinite(vmax): vmax = np.nanmax(a)
        if vmin == vmax:
            vmax = vmin + 1e-6
        return float(vmin), float(vmax)
    
    xmin, ymin, _ = domainMinBoundary
    xmax, ymax, _ = domainMaxBoundary
    extent = [xmin, xmax, ymin, ymax]
    vmin, vmax = robust_minmax(true_grid)

    # 可选上采样，仅用于显示
    grid_to_show = true_grid
    if upscale_factor and upscale_factor > 1:
        with torch.no_grad():
            g = torch.from_numpy(true_grid)[None, None, ...].float()
            g_hi = torch.nn.functional.interpolate(g, scale_factor=upscale_factor, mode='bilinear', align_corners=False)[0, 0]
            grid_to_show = g_hi.cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(10, 8), constrained_layout=True, dpi=dpi)
    im = ax.imshow(grid_to_show, origin='lower', extent=extent, cmap='coolwarm', vmin=vmin, vmax=vmax, interpolation='bilinear')
    ax.set_title('Scalar Field')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_aspect('equal')

    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.formatter.set_powerlimits((0, 0))
    cb.update_ticks()

    plt.show(block=True)
    plt.close(fig)


def visualize_ftle_sliceComparison(true_grid: np.ndarray, pred_grid: np.ndarray, domainMinBoundary,domainMaxBoundary,psnr: float=0.0,save_path: str | None = None,
                         upscale_factor: int = 1, dpi: int = 300):
    # 公共显示范围（稳健地忽略极端值）
    def robust_minmax(a):
        if np.all(np.isnan(a)):
            return 0.0, 1.0
        vmin = np.nanpercentile(a, 2)
        vmax = np.nanpercentile(a, 98)
        if not np.isfinite(vmin): vmin = np.nanmin(a)
        if not np.isfinite(vmax): vmax = np.nanmax(a)
        if vmin == vmax:
            vmax = vmin + 1e-6
        return float(vmin), float(vmax)

    xmin, ymin, _ = domainMinBoundary
    xmax, ymax, _ = domainMaxBoundary
    extent = [xmin, xmax, ymin, ymax]

    vmin_t, vmax_t = robust_minmax(true_grid)

    vmin=vmin_t
    vmax=vmax_t

    err = pred_grid - true_grid
    eabs = np.abs(err)
    evmin, evmax = robust_minmax(eabs)

    # 可选：提升栅格分辨率（先双线性上采样，再像素级显示）
    if upscale_factor and upscale_factor > 1:
        with torch.no_grad():
            tg = torch.from_numpy(true_grid)[None, None, ...].float()
            pg = torch.from_numpy(pred_grid)[None, None, ...].float()
            tg_hi = torch.nn.functional.interpolate(tg, scale_factor=upscale_factor, mode='bilinear', align_corners=False)[0, 0]
            pg_hi = torch.nn.functional.interpolate(pg, scale_factor=upscale_factor, mode='bilinear', align_corners=False)[0, 0]
            true_grid = tg_hi.cpu().numpy()
            pred_grid = pg_hi.cpu().numpy()

    fig, axes = plt.subplots(3, 1, figsize=(4, 12), constrained_layout=True, dpi=dpi)
    ims = []
    ims.append(axes[0].imshow(true_grid, origin='lower', extent=extent, cmap='coolwarm', vmin=vmin, vmax=vmax, interpolation='bilinear'))
    axes[0].set_title('FTLE True'); axes[0].set_xlabel('X'); axes[0].set_ylabel('Y'); axes[0].set_aspect('equal')
    ims.append(axes[1].imshow(pred_grid, origin='lower', extent=extent, cmap='coolwarm', vmin=vmin, vmax=vmax, interpolation='bilinear'))
    axes[1].set_title(f'Pred,PSNR={psnr:.2f}dB'); axes[1].set_xlabel('X'); axes[1].set_ylabel('Y'); axes[1].set_aspect('equal')
    ims.append(axes[2].imshow(eabs, origin='lower', extent=extent, cmap='coolwarm', vmin=evmin, vmax=evmax, interpolation='bilinear'))
    axes[2].set_title('|Pred-True|'); axes[2].set_xlabel('X'); axes[2].set_ylabel('Y'); axes[2].set_aspect('equal')

    for ax, im in zip(axes, ims):
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.formatter.set_powerlimits((0, 0))
        cb.update_ticks()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)
    plt.show(block=True)
    plt.close(fig)
    


def compute_metrics(in_true_grid:np.ndarray|torch.Tensor, in_pred_grid:np.ndarray|torch.Tensor):

    true_grid= in_true_grid.cpu().numpy()  if  isinstance(in_true_grid, torch.Tensor) else in_true_grid 
    pred_grid= in_pred_grid.cpu().numpy() if  isinstance(in_pred_grid, torch.Tensor) else in_pred_grid 

    y_global_max = true_grid.max()
    y_global_min = true_grid.min()
    mse = np.mean((true_grid - pred_grid) ** 2)
    mae = np.mean(np.abs(true_grid - pred_grid))
    maxe = np.max(np.abs(true_grid - pred_grid))
    dyn_range = max(abs(y_global_max - y_global_min), 1e-12)
    psnr = float('inf') if mse <= 1e-20 else 20.0 * np.log10(dyn_range) - 10.0 * np.log10(mse)
    return mse, mae, maxe,  psnr

def test_PointWiseFTLE_model(cfg,model: nn.Module, device: str = "cuda", visualize: bool = False):
    #first generate grid points for this time, then generate pathlines, 
    # then call computeFTLEFromPathlineCrossPrimitive get correct ftle
    # then compare the ftle from model and the correct ftle
    #report the error
    vectorfield = cfg['vectorfield']
    nerb = int(cfg.pcds.num_cross_points_per_seeding)
    LstepsPerline = int(cfg.pcds.sampled_points_per_line)
    max_steps = int(cfg.pcds.max_iterations)
    localized = bool(cfg.pcds.localized)
    normalized = bool(cfg.pcds.normalized)
    starts_chunk = int(cfg.bs)

    # 时间设置：以输入的 physical time 为起始，目标时间与训练保持相同的时间跨度
    tmin, tmax = float(vectorfield.tmin), float(vectorfield.tmax)
    base_t_target_ratio = float(0.9)
    physical_start = float(0.6 * (vectorfield.tmax - vectorfield.tmin) + vectorfield.tmin)
    physical_target = float(np.clip(tmin + base_t_target_ratio * (tmax - tmin), tmin, tmax))
    Pathline_all, PathlineLength_all,nx,ny=generate_test_points(cfg,vectorfield,physical_start,physical_target)
    true_grid = np.full((ny, nx), 0, dtype=np.float32)
    pred_grid = np.full((ny, nx), 0, dtype=np.float32)
    M_all = Pathline_all.shape[0]
    assert M_all==nx*ny

    total_groups = 0
    se_sum = 0.0
    ae_sum = 0.0
    max_abs_err = 0.0


    ftle_min=cfg['ftle_min']
    ftle_max=cfg['ftle_max']

    with torch.no_grad():
        model.eval()
        for s0 in range(0, M_all, max(1, int(starts_chunk))):
            s1 = min(M_all, s0 + int(starts_chunk))
            Pathline_batch_group = Pathline_all[s0:s1]
            PathlineLength_batch_group = PathlineLength_all[s0:s1]
            assert PathlineLength_batch_group.shape[1] == nerb,  "lines count must be multiple of nerbors"
            GroupSize_BatchSize = Pathline_batch_group.shape[0]
            keep_groups_full = (PathlineLength_batch_group == max_steps).all(dim=1)  # [GroupSize_BatchSize]

            #reduce number of points saved
            P_K = temporal_downsamplePathlineCrossPrimitiveRegular(Pathline_batch_group, LstepsPerline) #P_K.view(G_b, nerb, LstepsPerline, 3)

            # 计算真值 FTLE，并将非满长组置零
            if not (keep_groups_full).any():
                continue
            y_true_b = computeFTLEFromPathlineCrossPrimitive(Pathline_batch_group, vectorfield_dt=vectorfield.timeInterval)
       
            y_valid = y_true_b[keep_groups_full]
            y_valid = y_valid.to(device).float()


            # 预处理并分批预测（必要时 pad 到 64 的倍数）
            P_in = preprocess_localization_normalization(P_K, nerb, LstepsPerline, localized, normalized).to(device).float()
            P_in = P_in.to(device)
            B = P_in.shape[0]
            pad = (-B) % 64
            if pad > 0:
                P_in_pad = torch.cat([P_in, P_in[-1:].repeat(pad, 1,1, 1)], dim=0)
            else:
                P_in_pad = P_in

            pred_all = model(P_in_pad).to(device).float()
            pred_b = pred_all[:B]
            pred_b = pred_b*(ftle_max-ftle_min)+ftle_min
            valid_pred_b = pred_b[keep_groups_full]

            total_groups+=int(GroupSize_BatchSize)
            # fill back the grid
            idx_global = (np.arange(GroupSize_BatchSize) + s0)
            valid_idx_global = idx_global[keep_groups_full]
            rows = valid_idx_global // nx
            cols = valid_idx_global % nx
            true_grid[rows, cols] = y_valid.detach().cpu().numpy()
            pred_grid[rows, cols] = valid_pred_b.detach().cpu().numpy()

        if total_groups == 0:
            print("[test_ftle] No full-length groups found across all chunks.")
            return  {
            "mse": 0,
            "mae": 0,
            "maxe": 0,
            "psnr": 0,
            "true_grid": true_grid,
            "pred_grid": pred_grid,
        }



        mse, mae, maxe, psnr = compute_metrics(true_grid, pred_grid)
        # 可视化 2D 切片
        if visualize:
            visualize_ftle_sliceComparison(true_grid, pred_grid, vectorfield.domainMinBoundary, vectorfield.domainMaxBoundary,psnr)

        return {
            "mse": mse,
            "mae": mae,
            "maxe": maxe,
            "psnr": psnr,
            "true_grid": true_grid,
            "pred_grid": pred_grid
        }


# Torch Dataset for training samples generated on-the-fly via generate_training_samples
class PointWiseFTLETrainDataset(Dataset):
    def __init__( self,  config, cacheSystem: bool = True):
        nerbors = int(config.pcds.num_cross_points_per_seeding)
        LstepsPerline = int(config.pcds.sampled_points_per_line)
        total_points_count =getattr(config, 'train_points_count', 640*80*4)
        max_steps = int(config.pcds.max_iterations)
        flowline_dt = float(config.pcds.dt)
        time_slice = int(config.time_slice)
        offset_dist = float(config.pcds.offset_dist)
        unsteadyFieldNames=str([name for name in config.dataset.names])
        time_window_start_ratio=float(config.pcds.t_start)
        time_window_target_ratio=float(config.pcds.t_target)
        P_all=[]
        V_all=[]
        cacheSuccess=False
        # Cache logic: build a unique tag and try to load
        if cacheSystem:
            key_obj = {
                "name": "pointwise_ftle_train",
                "cnt": int(total_points_count),
                "ms": int(max_steps),
                "dt": float(flowline_dt),
                "ts": float(time_window_start_ratio),
                "tt": float(time_window_target_ratio),
                "off": float(offset_dist),
                "nb": int(nerbors),
                "unsteadyFieldNames": list(name for name in [name for name in config.dataset.names]),
                "time_slice": int(time_slice),
                "LstepsPerline": int(LstepsPerline),
            }
            tag = stable_hash(key_obj, prefix="TrainPL_")
            cache_dir = os.path.join("./outputs", "temp")
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, f"{tag}.npz")

            if os.path.exists(cache_path):
                try:
                    data = np.load(cache_path)
                    P_np = data["P"]
                    V_np = data["V"]
                    P_all = torch.from_numpy(P_np).float()
                    V_all = torch.from_numpy(V_np).float()
                    cacheSuccess=True
                except Exception as e:
                    logging.error(f"[generate_training_samples] cache load failed: {e}. Regenerating...")
        
        #generate training samples
        if not cacheSuccess:
            UnsteadyVectorFields=load_UnsteadyVectorFields_general(config.dataset.dat_dir,config.dataset.names)
            integration_interval=float(flowline_dt*max_steps)
            for i,vectorfield in enumerate(UnsteadyVectorFields):
                logging.info(f"[generate_training_samples] generate training samples for {i+1} vector field of {len(UnsteadyVectorFields)}...")
                dataset_timewindow_start = float(config.pcds.t_start * (vectorfield.tmax - vectorfield.tmin) + vectorfield.tmin)
                dataset_timewindow_target = float(config.pcds.t_target * (vectorfield.tmax - vectorfield.tmin) + vectorfield.tmin)
                #if start integration time is too close to the end of the vector field, we will not generate training samples with same length, 
                # which will make generate_training_samples failed
                dataset_timewindow_target=min(dataset_timewindow_target, vectorfield.tmax-integration_interval-flowline_dt)
                ftle_timeslices=np.linspace(dataset_timewindow_start, dataset_timewindow_target, time_slice)
                for physcial_start_time in ftle_timeslices:
                    physical_target_time =min(physcial_start_time + float(flowline_dt*max_steps), vectorfield.tmax)
                    P_OneTimewindow, length_OneTimewindow = generate_training_samples(
                        vectorfield=vectorfield,
                        count=int(total_points_count),
                        max_steps=int(max_steps),
                        dt=float(flowline_dt),
                        t_start=float(physcial_start_time),
                        t_target=float(physical_target_time),
                        offset_dist=float(offset_dist),
                        nerbors=int(nerbors)
                    )
                    NGroups,nerborsCrossSize,PathlineLength,Dim=P_OneTimewindow.shape
                    assert NGroups==total_points_count and nerborsCrossSize==nerbors and PathlineLength==max_steps and Dim==3,"P_OneTimewindow shape is not correct"
                    y_OneTimewindow = computeFTLEFromPathlineCrossPrimitive(P_OneTimewindow)
                    P_all.append(P_OneTimewindow)
                    V_all.append(y_OneTimewindow)
            P_all=torch.cat(P_all, dim=0)
            V_all=torch.cat(V_all, dim=0)
        
        if cacheSystem and not cacheSuccess:
            try:
                key_obj = {
                    "name": "pointwise_ftle_train",
                    "cnt": int(total_points_count),
                    "ms": int(max_steps),
                    "dt": float(flowline_dt),
                    "ts": float(time_window_start_ratio),
                    "tt": float(time_window_target_ratio),
                    "off": float(offset_dist),
                    "nb": int(nerbors),
                    "unsteadyFieldNames": list(name for name in [name for name in config.dataset.names]),
                    "time_slice": int(time_slice),
                    "LstepsPerline": int(LstepsPerline),
                }
                tag = stable_hash(key_obj, prefix="TrainPL_")
                cache_dir = os.path.join("outputs", "temp")
                os.makedirs(cache_dir, exist_ok=True)
                cache_path = os.path.join(cache_dir, f"{tag}.npz")
                np.savez(cache_path,
                        P=P_all.detach().cpu().numpy().astype(np.float32),
                        V=V_all.detach().cpu().numpy().astype(np.float32),
                        meta=str(key_obj))
            except Exception as e:
                print(f"[generate_training_samples] cache save failed: {e}")


        temporal_sampled_P_all=temporal_downsamplePathlineCrossPrimitive(P_all, int(LstepsPerline))
        localized=bool(config.pcds.localized)
        localization=preprocess_localization_normalization(temporal_sampled_P_all, int(nerbors), int(LstepsPerline), bool(localized), False ).cpu().float()

        self.ftle_min = float(V_all.min())
        self.ftle_max = float(V_all.max())
        normalized_y=(V_all-self.ftle_min)/(self.ftle_max-self.ftle_min)
        normalized_y=normalized_y.clamp(0,1)
        self.points = localization   # [N, nerb*K, 3]
        self.labels = normalized_y       # [N]
        print(f"[PointWiseFTLETrainDataset] generate {self.points.shape[0]} training samples")
    def __len__(self):
        return self.points.shape[0]

    def __getitem__(self, idx):
        return self.points[idx], self.labels[idx]






# Unified flow-map upsampling dataset.
#
# 合并自原 FTLEUpsamplingTrainDataset + FLowMapUpsamplingTrainDataset。核心思路:
# 直接 upsample flow map（pathline 首/尾点位置），而不是 upsample FTLE —— 因为 FTLE 只是
# flow map 的派生量（首尾 + 线性变化 + Cauchy-Green 特征值开根号, 见 FTLEFromFlowMap）。
# 每个 slice 通过 `generate_Flowmap_SLICE` 得到原始坐标的 flow map（[NxNy,5,2,3]），再做
# 滑窗 patch tiling。flow map 不做任何归一化（输出原始物理坐标）。
class FlowMapUpsamplingTrainDataset(Dataset):
    def __init__(self,   config, useCacheSystem: bool = True):
        UnsteadyVectorFields=[]
        # 支持 input_names（与 FTLE 数据集一致）；兼容旧的 names 字段。
        all_vectorfieldsname=[name for name in getattr(config.dataset, 'input_names', getattr(config.dataset, 'names', []))]
        timesliceCount=int(getattr(config.dataset, 'trainTimesliceCount', getattr(config.dataset, 'timesliceCount', 20)))
        UPsampling=int(config.dataset.UPsampling)
        low_res_grid_sampling=float(config.dataset.low_res_grid_sampling)
        max_steps: int=config.pcds.max_iterations
        flowline_dt: float=config.pcds.dt
        offset_dist: float =float(config.pcds.offset_dist)
        time_window_start_ratio=float(config.dataset.t_start)
        time_window_target_ratio=float(config.dataset.t_target)
        LstepsPerline=int(config.pcds.sampled_points_per_line)
        patch_size=int(getattr(config.dataset, 'patchSize', 32))
        patch_stride=int(getattr(config.dataset, 'patchStride', 4))
        LoadCacheSuccess=False
        key_obj = {
            "name": "flowmap_upsampling_train",
            "vectorfields": list(all_vectorfieldsname),
            "timesliceCount": int(timesliceCount),
            "UPsampling": int(UPsampling),
            "lowResGridIntervalScale": float(low_res_grid_sampling),
            "time_window_start_ratio": float(time_window_start_ratio),
            "time_window_target_ratio": float(time_window_target_ratio),
            "max_steps": int(max_steps),
            "dt": float(flowline_dt),
            "offset_dist": float(offset_dist),
            "LstepsPerline": int(LstepsPerline),
            "patchSize": int(patch_size),
            "patchStride": int(patch_stride),
            # bump these when the stored data semantics change so old caches are not reused.
            "normalize": f"div{GLOBAL_UniformValueTemporalAndSpatial}",
            "representation": "relCenterScaled_2offset",
            }
        tag = stable_hash(key_obj, prefix="FlowMapUpsamplingTrainingDataset_")
        cache_dir = os.path.join(config.cache_dir, "temp")
        cache_path = os.path.join(cache_dir, f"{tag}.npz")
        if useCacheSystem and os.path.exists(cache_path):
            try:
                data = np.load(cache_path)
                data_np = data["Data"]
                labels_np = data["Labels"]
                lowResPathlines_np = data["LowResPathlines"]
                self.lowResFlowMap = torch.from_numpy(data_np).float()
                self.lowResPathlines = torch.from_numpy(lowResPathlines_np).float()
                self.labels = torch.from_numpy(labels_np).float()
                LoadCacheSuccess=True
                logging.info(f"[FlowMapUpsamplingTrainDataset] loaded {self.lowResFlowMap.shape[0]} samples from cache {cache_path}")
            except Exception as e:
                print(f"[FlowMapUpsamplingTrainDataset] cache load failed: {e}. Regenerating...")

        if not LoadCacheSuccess:
            # helper: compute starts so that last window touches boundary (may overlap previous)
            def _tiling_starts(length: int, k: int, stride: int) -> list[int]:
                length = int(length)
                k = int(k)
                if length <= 0:
                    logging.warning("[FlowMapUpsamplingTrainDataset] empty tiling dimension; no starts generated.")
                    return []
                if k <= 0:
                    raise ValueError(f"patch size must be positive, got {k}")
                if k >= length:
                    if k > length:
                        logging.warning(f"[FlowMapUpsamplingTrainDataset] patch_size={k} > grid length={length}; "
                                        "using one edge-padded tile.")
                    return [0]
                s = max(1, int(stride))
                starts = list(range(0, length - k + 1, s))
                last = length - k
                if starts[-1] != last:
                    starts.append(last)
                return starts

            def _patch_indices(start: int, length: int, k: int) -> list[int]:
                start = max(0, min(int(start), max(0, int(length) - 1)))
                end = min(start + int(k), int(length))
                indices = list(range(start, end))
                if indices and int(k) > int(length):
                    indices.extend([indices[-1]] * (int(k) - len(indices)))
                return indices

            UnsteadyVectorFields=load_UnsteadyVectorFields_general(config.dataset.dat_dir, all_vectorfieldsname)
            FLowMap_fieldsLowRes=[]      # list[Tensor (patch_hw, 5, 2, 3)]
            FLowMap_fieldsHighRes=[]     # list[Tensor (patch_hw*UP*UP, 5, 2, 3)]
            lowResPathlinesData=[]       # list[Tensor (patch_hw, 5, L, 3)]
            high_res_sampling=float(UPsampling*low_res_grid_sampling)
            for i,vectorfield in enumerate(UnsteadyVectorFields):
                logging.info(f"[FlowMapUpsamplingTrainDataset] generate training samples for {i+1} vector field of {len(UnsteadyVectorFields)}...")
                field_patches_before = len(FLowMap_fieldsLowRes)
                time_window_start = float(time_window_start_ratio * (vectorfield.tmax - vectorfield.tmin) + vectorfield.tmin)
                time_window_target = float(time_window_target_ratio * (vectorfield.tmax - vectorfield.tmin) + vectorfield.tmin)
                # Forward integration needs seed_time + dt*max_steps <= tmax, otherwise every
                # pathline runs out of time. Cap the latest seed time so the horizon fits the domain.
                integ_horizon = float(flowline_dt) * int(max_steps)
                safe_target = float(vectorfield.tmax) - integ_horizon
                if time_window_target > safe_target:
                    logging.warning(f"[FlowMapUpsamplingTrainDataset] t_target={time_window_target:.3f} + integ horizon "
                                    f"{integ_horizon:.3f} exceeds tmax={float(vectorfield.tmax):.3f}; "
                                    f"clamping slice end to {max(time_window_start, safe_target):.3f}")
                    time_window_target = max(time_window_start, safe_target)
                timeslice=np.linspace(time_window_start, time_window_target, timesliceCount)
                for physcialTime in timeslice:
                    # low/high-res flow map slices, raw coords:
                    #   *_flow      : torch [ny*nx, 5, 2, 3]
                    #   lowResPathlines: torch [ny*nx, 5, max_steps, 3]
                    low_resFlowMap_field, lowResPathlines, _low_len, nx_low, ny_low = generate_Flowmap_SLICE(
                        vectorfield, physcialTime, flowline_dt, max_steps, offset_dist, low_res_grid_sampling)
                    high_resFlowMap_field, _h_pl, _h_len, nx_hi, ny_hi = generate_Flowmap_SLICE(
                        vectorfield, physcialTime, flowline_dt, max_steps, offset_dist, high_res_sampling)

                    # lowResPathlinesPreprocessed: [ny*nx, 5, LstepsPerline, 3]
                    lowResPathlinesPreprocessed=AngleAwareSampling(lowResPathlines, LstepsPerline)

                    # sliding window tiling over the (ny_low, nx_low) grid; flow map / pathlines are
                    # indexed flat (row*nx + col), so build flat index lists for each patch.
                    # Only edge-pad when the whole grid dimension is smaller than patch_size
                    # (e.g. 20 -> 32); normal partial/misaligned patches are dropped below.
                    row_starts = _tiling_starts(ny_low, patch_size, patch_stride)
                    col_starts = _tiling_starts(nx_low, patch_size, patch_stride)

                    for row_tile_idx, i0 in enumerate(row_starts):
                        row_idx = _patch_indices(i0, ny_low, patch_size)
                        for col_tile_idx, j0 in enumerate(col_starts):
                            col_idx = _patch_indices(j0, nx_low, patch_size)
                            # map to high-res indices (align last window to boundary)
                            hi_h = max(1, patch_size * UPsampling)
                            hi_w = max(1, patch_size * UPsampling)
                            hi_i0 = int(round(i0 * UPsampling))
                            hi_j0 = int(round(j0 * UPsampling))
                            if row_tile_idx == len(row_starts) - 1:
                                hi_i0 = ny_hi - hi_h
                            if col_tile_idx == len(col_starts) - 1:
                                hi_j0 = nx_hi - hi_w
                            hi_i0 = max(0, hi_i0)
                            hi_j0 = max(0, hi_j0)
                            hi_row_idx = _patch_indices(hi_i0, ny_hi, hi_h)
                            hi_col_idx = _patch_indices(hi_j0, nx_hi, hi_w)

                            #flatten the index to 1D index   ,GPT check Y,X 排序是正确的：rr 是 y/row，cc 是 x/col；同一个 row 内 x 连续增长，row 变化时 y 才增长。
                            lo_idx = [rr * nx_low + cc for rr in row_idx for cc in col_idx]
                            hi_idx = [rr * nx_hi + cc for rr in hi_row_idx for cc in hi_col_idx]
                            if len(lo_idx) == 0 or len(hi_idx) == 0:
                                continue
                            lo_t = torch.as_tensor(lo_idx, dtype=torch.long)
                            hi_t = torch.as_tensor(hi_idx, dtype=torch.long)

                            lr_patch = low_resFlowMap_field[lo_t].float()   # [P, 5, 2, 3]
                            hr_patch = high_resFlowMap_field[hi_t].float()  # [P*UP*UP, 5, 2, 3]
                            pl_patch = lowResPathlinesPreprocessed[lo_t].float()  # [P, 5, L, 3]

                            # (1) Jacobian-aware representation: rewrite neighbour-line xy as
                            # offset-from-center / (2*offset_dist), lifting the FTLE signal to
                            # O(1). (2) Then scale all coords by 1/GLOBAL for image-like inputs.
                            # Both input and label get the SAME transforms; test mirrors them in
                            # reverse (inverse_normalize -> from_relative) before computing FTLE.
                            rel_scale = 2.0 * offset_dist
                            lr_patch = flowmap_to_relative(lr_patch, rel_scale) / GLOBAL_UniformValueTemporalAndSpatial
                            hr_patch = flowmap_to_relative(hr_patch, rel_scale) / GLOBAL_UniformValueTemporalAndSpatial

                            # keep only patches whose hi/lo counts respect the UP^2 relation
                            if hr_patch.shape[0] == lr_patch.shape[0] * UPsampling * UPsampling \
                               and pl_patch.shape[0] == lr_patch.shape[0]:
                                FLowMap_fieldsLowRes.append(lr_patch)
                                FLowMap_fieldsHighRes.append(hr_patch)
                                lowResPathlinesData.append(pl_patch)
                            else:
                                logging.warning(f"[FlowMapUpsamplingTrainDataset] dropped patch: "
                                                f"pl={tuple(pl_patch.shape)}, lr={tuple(lr_patch.shape)}, hr={tuple(hr_patch.shape)}")

                n_field_patches = len(FLowMap_fieldsLowRes) - field_patches_before
                _vfname = all_vectorfieldsname[i] if i < len(all_vectorfieldsname) else f"field{i}"
                if n_field_patches == 0:
                    logging.warning(f"[FlowMapUpsamplingTrainDataset] '{_vfname}': 0 full {patch_size}x{patch_size} "
                                    f"patches (low-res grid smaller than patch_size); excluded from training.")
                else:
                    logging.info(f"[FlowMapUpsamplingTrainDataset] '{_vfname}': collected {n_field_patches} patches.")

            self.lowResFlowMap = torch.stack(FLowMap_fieldsLowRes)
            self.lowResPathlines = torch.stack(lowResPathlinesData)
            self.labels = torch.stack(FLowMap_fieldsHighRes)

        if useCacheSystem and not LoadCacheSuccess:
            try:
                os.makedirs(cache_dir, exist_ok=True)
                np.savez(cache_path,
                        Data=self.lowResFlowMap.detach().cpu().numpy().astype(np.float32),
                        Labels=self.labels.detach().cpu().numpy().astype(np.float32),
                        LowResPathlines=self.lowResPathlines.detach().cpu().numpy().astype(np.float32))
                logging.info(f"[FlowMapUpsamplingTrainDataset] saved {self.lowResFlowMap.shape[0]} samples to cache {cache_path}")
            except Exception as e:
                logging.error(f"[FlowMapUpsamplingTrainDataset] cache save failed: {e}")


        # flow map 输出原始物理坐标, 不做归一化 (归一化交给下游处理)。
    def __len__(self):
        return len(self.lowResFlowMap)

    def __getitem__(self, idx):
        return (self.lowResFlowMap[idx], self.lowResPathlines[idx]), self.labels[idx]







