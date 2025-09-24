from torch.nn.functional import upsample
import torch.nn.functional as F
from torch.utils.data import Dataset
from FLowUtils.VectorField2d import UnsteadyVectorField2D
import torch,os
import numpy as np
import hashlib
from DeepUtils.utils.stable_hash import stable_hash
from FLowUtils.ScalarField2d import ScalarField2D,ScalarFieldManager
import matplotlib.pyplot as plt
from FMT_Utils.FlowlinePostProcessing import AngleAwareSampling, LocLines, temporal_downsamplePathlineCrossPrimitiveRegular
from FLowUtils.flowlineIntegral import batch_pathlineCross_integration_2D_auto
from FLowUtils.netCDFLoader import *
from FLowUtils.AnalyticalFlowCreator import *
from FLowUtils.ScalarField2d import compute_ivd_2D

global_UniformValueSpatical=8.0
global_UniformValueTemporal=12.5663704#4 pi

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
      - points_grouped: [N, sample_nerbors, (line_steps), 3], ordered as (center, x+, x-, y+, y-)
      - sample_nerbors: 5 (cross)
      - line_steps:     number of integration steps (K), each line has K+1 points
      - vectorfield_dt: physical time step to validate effective time span

    Output:
      - ftle: [N] FTLE for the center sample in each group
    """
    assert points_grouped.dim() == 4 and points_grouped.shape[-1] >= 2
    N = int(points_grouped.shape[0])
    pts = points_grouped
    L=pts.shape[2]
    k = L - 1

    # Initial and final positions (physical coordinates)
    # Neighbor order convention: 0=center, 1=x+, 2=x-, 3=y+, 4=y-
    p0 = pts[:, :, 0, :2]  # [N,5,2]
    pk = pts[:, :, k, :2]  # [N,5,2]

    # Initial offsets (scalars, axis-aligned)
    dx0 = (p0[:, 1, 0] - p0[:, 2, 0]).clamp_min(1e-12)  # [N]
    dy0 = (p0[:, 3, 1] - p0[:, 4, 1]).clamp_min(1e-12)  # [N]

    # Flow map Jacobian J = [dPhi/dx0, dPhi/dy0] (as column vectors)
    dPhi_dx = (pk[:, 1, :2] - pk[:, 2, :2]) / dx0.unsqueeze(-1)  # [N,2]
    dPhi_dy = (pk[:, 3, :2] - pk[:, 4, :2]) / dy0.unsqueeze(-1)  # [N,2]
    J = torch.stack([dPhi_dx, dPhi_dy], dim=-1)  # [N,2,2]

    # Cauchy–Green tensor C = J^T J
    JT = J.transpose(1, 2)
    C = torch.bmm(JT, J)  # [N,2,2]

    # Largest eigenvalue (C is symmetric positive semidefinite)
    eigvals = torch.linalg.eigvalsh(C)  # [N,2], 升序
    lambda_max = eigvals[:, 1].clamp_min(1e-12)

    # Physical time span ΔT (use center line time)
    t_norm_start = pts[:, 0, 0, 2]
    t_norm_end   = pts[:, 0, k, 2]
    T_phys = (t_norm_end - t_norm_start).abs() 

    # If T_phys is too small (< 2*vectorfield_dt), the pathline is invalid → set FTLE to 0
    ftle = 0.5 * torch.log(lambda_max) / T_phys
    ftle = torch.where(T_phys <2* vectorfield_dt , torch.zeros_like(ftle), ftle)
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
        


def generate_seedingGrid_2D(vectorfield: UnsteadyVectorField2D,resolutionUPsampling:float,boundary_offset:float=0.05):
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




def generate_FLowMap_SLICE(vectorfield: UnsteadyVectorField2D,physcial_time:float,dt:float,maxIterations:int, offesetDist:float, resolutionUPsampling:float=1.0):
    nerbors=5
    offset_dist = float(offesetDist)
    max_steps = int(maxIterations)
    starts_xy,xs,ys=generate_seedingGrid_2D(vectorfield,resolutionUPsampling)
    ny, nx = len(ys), len(xs)
    Pathline_b, PathlineLength_b = batch_pathlineCross_integration_2D_auto(
                points=starts_xy,
                vectorfield=vectorfield,
                t_start=float(physcial_time), t_target=float(physcial_time+dt*maxIterations),
                dt=float(dt), max_steps=int(max_steps),
                offsets_size=float(offset_dist), method="rk4"
            )
    Pathline_g = Pathline_b.view(nx*ny, nerbors, max_steps, 3)
    PathlineLength_g = PathlineLength_b.view(nx*ny, nerbors)
    # y_all=computeFTLEFromPathlineCrossPrimitive(Pathline_g, vectorfield_dt=vectorfield.timeInterval)

    keep_groups_full = (PathlineLength_g == max_steps).all(dim=1)
    true_grid = np.full((ny, nx, 6), 0, dtype=np.float32)
    linear_index=np.arange(nx*ny)
    valid_index=linear_index[keep_groups_full]
    rows=valid_index//nx
    cols=valid_index%nx
    centralPointfirt_pos=Pathline_g[valid_index, 0, 0,:]
    centralPointlast_pos=Pathline_g[valid_index, 0, max_steps-1,:]
    #concat centralPointfirt_pos and centralPointlast_pos
    flowMap=torch.concat([centralPointfirt_pos,centralPointlast_pos-centralPointfirt_pos],axis=1)
    flowMap[...,0:2]=flowMap[...,0:2]/global_UniformValueSpatical
    flowMap[...,2]=flowMap[...,2]/global_UniformValueTemporal
    flowMap[...,3:5]=flowMap[...,3:5]/global_UniformValueSpatical
    flowMap[...,5]=flowMap[...,5]/global_UniformValueTemporal
    true_grid[rows, cols,:] =flowMap.detach().cpu().numpy()

    return true_grid,Pathline_g,nx,ny


def generate_FTLE_SLICE(cfg,vectorfield: UnsteadyVectorField2D,physcial_time:float,dt:float,maxIterations:int, resolutionUPsampling:float=1.0):
    nerbors=5
    offset_dist = float(cfg.pcds.offset_dist)
    max_steps = int(cfg.pcds.max_iterations)
    starts_xy,xs,ys=generate_seedingGrid_2D(vectorfield,resolutionUPsampling)
    ny, nx = len(ys), len(xs)
    grid_low=np.zeros((ny,nx),dtype=np.float32)
    Pathline_b, PathlineLength_b = batch_pathlineCross_integration_2D_auto(
                points=starts_xy,
                vectorfield=vectorfield,
                t_start=float(physcial_time), t_target=float(physcial_time+dt*maxIterations),
                dt=float(dt), max_steps=int(max_steps),
                offsets_size=float(offset_dist), method="rk4"
            )
    Pathline_g = Pathline_b.view(nx*ny, nerbors, max_steps, 3)
    PathlineLength_g = PathlineLength_b.view(nx*ny, nerbors)
    y_all=computeFTLEFromPathlineCrossPrimitive(Pathline_g, vectorfield_dt=vectorfield.timeInterval)

    keep_groups_full = (PathlineLength_g == max_steps).all(dim=1)
    true_grid = np.full((ny, nx), 0, dtype=np.float32)
    linear_index=np.arange(nx*ny)
    valid_index=linear_index[keep_groups_full]
    rows=valid_index//nx
    cols=valid_index%nx
    true_grid[rows, cols] = y_all[valid_index].detach().cpu().numpy()
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
    plt.show(block=True)
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
            UnsteadyVectorFields=load_UnsteadyVectorFields_netCDFOrAnalytical(config.dataset.dat_dir,config.dataset.names)
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




# Torch Dataset for training samples generated on-the-fly via generate_training_samples
class FTLEUpsamplingTrainDataset(Dataset):
    def __init__(self,   config, useCacheSystem: bool = True):
        UnsteadyVectorFields=[]
        ftle_resolutionUPsampling=float(config.dataset.UPsampling)
        all_vectorfieldsname=[name for name in config.dataset.names]
        all_vectorfieldsname_str=",".join(all_vectorfieldsname)
        timesliceCount=config.dataset.timesliceCount
        UPsampling=int(config.dataset.UPsampling)
        low_res_grid_sampling=float(config.dataset.low_res_grid_sampling)
        max_steps: int=config.pcds.max_iterations
        flowline_dt: float=config.pcds.dt
        offset_dist: float =float(config.pcds.offset_dist)
        time_window_start_ratio=float(config.dataset.t_start)
        time_window_target_ratio=float(config.dataset.t_target)
        LstepsPerline=int(config.pcds.sampled_points_per_line)
        patch_size=int(getattr(config.dataset, 'patchSize', 32))
        patch_stride=4*int(getattr(config.dataset, 'patchStride', 4))
        LoadCacheSuccess=False
       
   
        key_obj = {
            "name": "ftle_upsampling_train",
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
            "patchStride": int(patch_stride)
                }
        tag = stable_hash(key_obj, prefix="FTLEUpsamplingTrainingDataset_")
        cache_dir = os.path.join(config.cache_dir, "temp")
        cache_path = os.path.join(cache_dir, f"{tag}.npz")
        if useCacheSystem and os.path.exists(cache_path):
            try:
                data = np.load(cache_path)
                data_np = data["Data"]
                labels_np = data["Labels"]
                lowResPathlines_np = data["LowResPathlines"]
                self.lowResFTLE = torch.from_numpy(data_np).float()
                self.lowResPathlines = torch.from_numpy(lowResPathlines_np).float()
                self.labels = torch.from_numpy(labels_np).float()
                LoadCacheSuccess=True
                logging.info(f"[FTLEUpsamplingTrainDataset] loaded {self.lowResFTLE.shape[0]} samples from cache {cache_path}")
            except Exception as e:
                print(f"[generate_training_samples] cache load failed: {e}. Regenerating...")
                         
        UnsteadyVectorFields=load_UnsteadyVectorFields_netCDFOrAnalytical(config.dataset.dat_dir,config.dataset.names)
        FTLE_fieldsLowRes=[]      # list[Tensor patch_yx]
        FTLE_fieldsHighRes=[]     # list[Tensor patch_yx (hi)]
        lowResPathlinesData=[]    # list[Tensor (patch_hw groups, nerbors, L, 3)]
        # itemMap2VectorField=[]

        if not LoadCacheSuccess:
            # helper: compute starts so that last window touches boundary (may overlap previous)
            def _tiling_starts(length: int, k: int, stride: int) -> list[int]:
                if k >= length:
                    return [0]
                s = max(1, int(stride))
                starts = list(range(0, length - k + 1, s))
                last = length - k
                if starts[-1] != last:
                    starts.append(last)
                return starts

            high_res_sampling=float(UPsampling*low_res_grid_sampling)
            for i,vectorfield in enumerate(UnsteadyVectorFields):
                logging.info(f"[FTLEUpsamplingTrainDataset] generate training samples for {i+1} vector field of {len(UnsteadyVectorFields)}...")
                time_window_start = float(time_window_start_ratio * (vectorfield.tmax - vectorfield.tmin) + vectorfield.tmin)
                time_window_target = float(time_window_target_ratio * (vectorfield.tmax - vectorfield.tmin) + vectorfield.tmin)
                timeslice=np.linspace(time_window_start, time_window_target, timesliceCount)
                for time_slice in timeslice:
                    # Low-res grid seeding,lowResPathlines shape: (lowResX*lowResY, nerbors, max_steps, 3)
                    low_resFTLE_field,lowResPathlines,low_res_xs,low_res_ys=generate_FTLE_SLICE(config,vectorfield,time_slice,flowline_dt,max_steps,low_res_grid_sampling)  
                    high_resFTLE_field,_,high_res_xs,high_res_ys=generate_FTLE_SLICE(config,vectorfield,time_slice,flowline_dt,max_steps,high_res_sampling)
                    # visualize_twoftle_slices(low_resFTLE_field, high_resFTLE_field, vectorfield.domainMinBoundary, vectorfield.domainMaxBoundary)
                    # pathline_length_in_save_data=max(max_steps//2, LstepsPerline)

                    lowResPathlinesPreprocessed=AngleAwareSampling(lowResPathlines, LstepsPerline)
                    # lowResPathlinesPreprocessed=preprocess_localization_normalization(temporal_sampled_P_all, 5, int(LstepsPerline), bool(localized), False ).cpu().float()

                    # sliding window tiling into patches that fully cover the 2D plane
                    ny_low, nx_low = low_resFTLE_field.shape
                    ny_hi, nx_hi = high_resFTLE_field.shape
                    ry = UPsampling
                    rx = UPsampling

                    row_starts = _tiling_starts(ny_low, patch_size, patch_stride)
                    col_starts = _tiling_starts(nx_low, patch_size, patch_stride)

                    for i0 in row_starts:
                        i1 = min(i0 + patch_size, ny_low)
                        for j0 in col_starts:
                            j1 = min(j0 + patch_size, nx_low)

                            # map to high-res indices (align last window to boundary)
                            hi_h = max(1,patch_size*UPsampling)
                            hi_w = max(1,patch_size*UPsampling)
                            hi_i0 = int(round(i0 * ry))
                            hi_j0 = int(round(j0 * rx))
                            if i0 == row_starts[-1]:
                                hi_i0 = ny_hi - hi_h
                            if j0 == col_starts[-1]:
                                hi_j0 = nx_hi - hi_w
                            hi_i1 = hi_i0 + hi_h
                            hi_j1 = hi_j0 + hi_w

                            # low-res FTLE patch
                            lr_patch = torch.from_numpy(low_resFTLE_field[i0:i1, j0:j1]).float()

                            # high-res FTLE patch
                            hr_patch = torch.from_numpy(high_resFTLE_field[hi_i0:hi_i1, hi_j0:hi_j1]).float()

                            # pathlines patch: select groups that fall inside the low-res window
                            idx_list = []
                            for rr in range(i0, i1):
                                base = rr * nx_low
                                for cc in range(j0, j1):
                                    idx_list.append(base + cc)
                            if len(idx_list) == 0:
                                continue
                            idx_tensor = torch.as_tensor(idx_list, dtype=torch.long)
                            pl_patch = lowResPathlinesPreprocessed[idx_tensor]
                            if pl_patch.shape[0] == lr_patch.shape[0]*lr_patch.shape[1] and lr_patch.shape[1]*UPsampling  == hr_patch.shape[1]and\
                               lr_patch.shape[0]*UPsampling== hr_patch.shape[0]:
                                FTLE_fieldsLowRes.append(lr_patch)
                                FTLE_fieldsHighRes.append(hr_patch)
                                lowResPathlinesData.append(pl_patch)
                            else:
                                logging.warning(f"[FTLEUpsamplingTrainDataset] pl_patch.shape: {pl_patch.shape}, lr_patch.shape: {lr_patch.shape}, hr_patch.shape: {hr_patch.shape}")
                            # itemMap2VectorField.append(vectorfield)

            self.lowResFTLE = torch.stack(FTLE_fieldsLowRes)
            self.lowResPathlines = torch.stack(lowResPathlinesData)
            self.labels = torch.stack(FTLE_fieldsHighRes)

        if useCacheSystem and not LoadCacheSuccess:
            try:
                os.makedirs(cache_dir, exist_ok=True)
                np.savez(cache_path,
                        Data=self.lowResFTLE.detach().cpu().numpy().astype(np.float32),
                        Labels=self.labels.detach().cpu().numpy().astype(np.float32),
                        LowResPathlines=self.lowResPathlines.detach().cpu().numpy().astype(np.float32))
                logging.info(f"[FTLEUpsamplingTrainDataset] saved {self.lowResFTLE.shape[0]} samples to cache {cache_path}")
            except Exception as e:
                print(f"[generate_training_samples] cache save failed: {e}")


          #normalize the ftle
       
        #put preprocessing here if any

        #normalize the ftle
        self.ftle_min = float(self.labels.min())
        self.ftle_max = float(self.labels.max())
        self.lowResFTLE = self.lowResFTLE.clamp(self.ftle_min, self.ftle_max)
        self.lowResFTLE = (self.lowResFTLE-self.ftle_min)/(self.ftle_max-self.ftle_min)
        normalized_y=(self.labels-self.ftle_min)/(self.ftle_max-self.ftle_min)
        normalized_y=normalized_y.clamp(0,1)
        self.labels = normalized_y       # [N]



    def __len__(self):
        return len(self.lowResFTLE)

    def __getitem__(self, idx):
        return (self.lowResFTLE[idx], self.lowResPathlines[idx]), self.labels[idx]


# Torch Dataset for training samples generated on-the-fly via generate_training_samples
class FLowMapUpsamplingTrainDataset(Dataset):
    def __init__(self,   config, useCacheSystem: bool = True):
        UnsteadyVectorFields=[]
        ftle_resolutionUPsampling=float(config.dataset.UPsampling)
        all_vectorfieldsname=[name for name in config.dataset.names]
        all_vectorfieldsname_str=",".join(all_vectorfieldsname)
        timesliceCount=config.dataset.timesliceCount
        UPsampling=int(config.dataset.UPsampling)
        low_res_grid_sampling=float(config.dataset.low_res_grid_sampling)
        max_steps: int=config.pcds.max_iterations
        flowline_dt: float=config.pcds.dt
        offset_dist: float =float(config.pcds.offset_dist)
        time_window_start_ratio=float(config.dataset.t_start)
        time_window_target_ratio=float(config.dataset.t_target)
        LstepsPerline=int(config.pcds.sampled_points_per_line)
        patch_size=int(getattr(config.dataset, 'patchSize', 32))
        patch_stride=4*int(getattr(config.dataset, 'patchStride', 4))
        LoadCacheSuccess=False
        key_obj = {
            "name": "ftle_upsampling_train",
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
            "patchStride": int(patch_stride)
            }
        tag = stable_hash(key_obj, prefix="FLowMapUpsamplingTrainingDataset_")
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
                logging.info(f"[FLowMapUpsamplingTrainDataset] loaded {self.lowResFlowMap.shape[0]} samples from cache {cache_path}")
            except Exception as e:
                print(f"[FLowMapUpsamplingTrainDataset] cache load failed: {e}. Regenerating...")
                         
        if not LoadCacheSuccess:
            # helper: compute starts so that last window touches boundary (may overlap previous)
            def _tiling_starts(length: int, k: int, stride: int) -> list[int]:
                if k >= length:
                    return [0]
                s = max(1, int(stride))
                starts = list(range(0, length - k + 1, s))
                last = length - k
                if starts[-1] != last:
                    starts.append(last)
                return starts

            UnsteadyVectorFields=load_UnsteadyVectorFields_netCDFOrAnalytical(config.dataset.dat_dir,config.dataset.names)
            FLowMap_fieldsLowRes=[]      # list[Tensor patch_yx]
            FLowMap_fieldsHighRes=[]     # list[Tensor patch_yx (hi)]
            lowResPathlinesData=[]    # list[Tensor (patch_hw groups, nerbors, L, 3)]
            high_res_sampling=int(UPsampling*low_res_grid_sampling)
            for i,vectorfield in enumerate(UnsteadyVectorFields):
                logging.info(f"[FLowMapUpsamplingTrainDataset] generate training samples for {i+1} vector field of {len(UnsteadyVectorFields)}...")
                time_window_start = float(time_window_start_ratio * (vectorfield.tmax - vectorfield.tmin) + vectorfield.tmin)
                time_window_target = float(time_window_target_ratio * (vectorfield.tmax - vectorfield.tmin) + vectorfield.tmin)
                timeslice=np.linspace(time_window_start, time_window_target, timesliceCount)
                for time_slice in timeslice:
                    # Low-res grid seeding,lowResPathlines shape: (lowResX*lowResY, nerbors, max_steps, 3)
                    low_resFTLE_field,lowResPathlines,low_res_xs,low_res_ys=generate_FLowMap_SLICE(vectorfield,time_slice,flowline_dt,max_steps,offset_dist,low_res_grid_sampling)  
                    high_resFTLE_field,_,high_res_xs,high_res_ys=generate_FLowMap_SLICE(vectorfield,time_slice,flowline_dt,max_steps,offset_dist,high_res_sampling)
                    # visualize_twoftle_slices(low_resFTLE_field, high_resFTLE_field, vectorfield.domainMinBoundary, vectorfield.domainMaxBoundary)
                    # pathline_length_in_save_data=max(max_steps//2, LstepsPerline)

                    lowResPathlinesPreprocessed=AngleAwareSampling(lowResPathlines, LstepsPerline)
                    # lowResPathlinesPreprocessed=preprocess_localization_normalization(temporal_sampled_P_all, 5, int(LstepsPerline), bool(localized), False ).cpu().float()

                    # sliding window tiling into patches that fully cover the 2D plane
                    ny_low, nx_low = low_resFTLE_field.shape[:2]
                    ny_hi, nx_hi = high_resFTLE_field.shape[:2]
                    ry = UPsampling
                    rx = UPsampling

                    row_starts = _tiling_starts(ny_low, patch_size, patch_stride)
                    col_starts = _tiling_starts(nx_low, patch_size, patch_stride)

                    for i0 in row_starts:
                        i1 = i0 + patch_size
                        for j0 in col_starts:
                            j1 = j0 + patch_size

                            # map to high-res indices (align last window to boundary)
                            hi_h = max(1,patch_size*UPsampling)
                            hi_w = max(1,patch_size*UPsampling)
                            hi_i0 = int(round(i0 * ry))
                            hi_j0 = int(round(j0 * rx))
                            if i0 == row_starts[-1]:
                                hi_i0 = ny_hi - hi_h
                            if j0 == col_starts[-1]:
                                hi_j0 = nx_hi - hi_w
                            hi_i1 = hi_i0 + hi_h
                            hi_j1 = hi_j0 + hi_w

                            # low-res FTLE patch
                            lr_patch = torch.from_numpy(low_resFTLE_field[i0:i1, j0:j1,:]).float()

                            # high-res FTLE patch
                            hr_patch = torch.from_numpy(high_resFTLE_field[hi_i0:hi_i1, hi_j0:hi_j1,:]).float()

                            # pathlines patch: select groups that fall inside the low-res window
                            idx_list = []
                            for rr in range(i0, i1):
                                base = rr * nx_low
                                for cc in range(j0, j1):
                                    idx_list.append(base + cc)
                            if len(idx_list) == 0:
                                continue
                            idx_tensor = torch.as_tensor(idx_list, dtype=torch.long)
                            pl_patch = lowResPathlinesPreprocessed[idx_tensor]
                            if pl_patch.shape[0] == lr_patch.shape[0]*lr_patch.shape[1] and lr_patch.shape[1]*UPsampling  == hr_patch.shape[1]and\
                               lr_patch.shape[0]*UPsampling== hr_patch.shape[0]:
                                FLowMap_fieldsLowRes.append(lr_patch)
                                FLowMap_fieldsHighRes.append(hr_patch)
                                lowResPathlinesData.append(pl_patch)
                            else:
                                logging.warning(f"[FLowMapUpsamplingTrainDataset] pl_patch.shape: {pl_patch.shape}, lr_patch.shape: {lr_patch.shape}, hr_patch.shape: {hr_patch.shape}")
                            # itemMap2VectorField.append(vectorfield)

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
                logging.info(f"[FLowMapUpsamplingTrainDataset] saved {self.lowResFlowMap.shape[0]} samples to cache {cache_path}")
            except Exception as e:
                logging.error(f"[FLowMapUpsamplingTrainDataset] cache save failed: {e}")


          #normalize the ftle
       
        #put preprocessing here if any

        #normalize the ftle
        # self.ftle_min = float(self.labels.min())
        # self.ftle_max = float(self.labels.max())
        # self.lowResFTLE = self.lowResFTLE.clamp(self.ftle_min, self.ftle_max)
        # self.lowResFTLE = (self.lowResFTLE-self.ftle_min)/(self.ftle_max-self.ftle_min)
        # normalized_y=(self.labels-self.ftle_min)/(self.ftle_max-self.ftle_min)
        # normalized_y=normalized_y.clamp(0,1)
        # self.labels = normalized_y       # [N]



    def __len__(self):
        return len(self.lowResFlowMap)

    def __getitem__(self, idx):
        return (self.lowResFlowMap[idx], self.lowResPathlines[idx]), self.labels[idx]


def FTLEFromFlowMap(flowMapRegularGrid:torch.Tensor, deltaT: float = 1.0, dxdy: float = 1.0):
    """
    基于 flowmap 的 2D FTLE 估计（模仿 computeFTLEFromPathlineCrossPrimitive）。

    输入:
      - flowMapRegularGrid: [..., H, W, 6]，通道顺序为 [x0, y0, t0, x1, y1, t1]
        若输入为 numpy，将自动转为 torch.float32。
      - deltaT: 若缺失时间通道则使用的全局时间跨度；若提供 t0/t1 则按像素 |t1-t0|
      - dxdy: 兼容参数，当前未使用（保留以兼容旧接口）
    输出:
      - ftle_grid: [..., H, W] 的 torch.float32 张量。

    做法:
      - 以起点坐标 (x0,y0) 的中心差分作为分母，对终点坐标 (x1,y1) 做中心差分作为分子，得到 J 的两列。
      - 计算 Cauchy–Green 张量 C=J^T J 的最大特征值 λ_max。
      - 对每个像素用物理时间跨度 ΔT=|t1-t0| 归一化：FTLE=0.5*log(λ_max)/ΔT。
    """
    # 支持 numpy 输入
    if isinstance(flowMapRegularGrid, np.ndarray):
        flow = torch.from_numpy(flowMapRegularGrid)
    else:
        flow = flowMapRegularGrid

    assert flow.dtype in (torch.float16, torch.float32, torch.float64) or flow.is_floating_point(), "flow map must be float tensor"
    flow = flow.to(dtype=torch.float32)

    # 统一维度为 [..., H, W, C]
    if flow.dim() == 3:
        H, W, C = flow.shape
        flow = flow.unsqueeze(0)  # [1, H, W, C]
    elif flow.dim() >= 4:
        H, W, C = flow.shape[-3], flow.shape[-2], flow.shape[-1]
    else:
        raise ValueError("flowMapRegularGrid must be [..., H, W, C] with C>=5")
    if C < 5:
        raise ValueError("flow map must have at least 5 channels: [x0,y0,t0,x1,y1(,t1)]")

    # 提取起终点分量
    x0 = flow[..., 0]
    y0 = flow[..., 1]
    t0 = flow[..., 2] if C >= 3 else None
    X1 = flow[..., 3]
    Y1 = flow[..., 4]
    t1 = flow[..., 5] if C >= 6 else None

    # 复制填充，便于中心差分（边界采用复制）
    def _pad(t: torch.Tensor):
        return F.pad(t, (1, 1, 1, 1), mode='replicate')  # [..., H+2, W+2]

    x0p = _pad(x0)
    y0p = _pad(y0)
    X1p = _pad(X1)
    Y1p = _pad(Y1)

    # 初始坐标间距（分母），使用起点坐标的中心差分
    dx0_raw = (x0p[..., 1:-1, 2:] - x0p[..., 1:-1, :-2])
    dy0_raw = (y0p[..., 2:, 1:-1] - y0p[..., :-2, 1:-1])
    zero_mask = (dx0_raw.abs() <= 1e-5) | (dy0_raw.abs() <= 1e-5)


    # 终点坐标关于起点的偏导（分子），中心差分
    dX_dx0 = (X1p[..., 1:-1, 2:] - X1p[..., 1:-1, :-2]) / dx0_raw
    dY_dx0 = (Y1p[..., 1:-1, 2:] - Y1p[..., 1:-1, :-2]) / dx0_raw
    dX_dy0 = (X1p[..., 2:, 1:-1] - X1p[..., :-2, 1:-1]) / dx0_raw
    dY_dy0 = (Y1p[..., 2:, 1:-1] - Y1p[..., :-2, 1:-1]) / dx0_raw


    # 显式展开 Cauchy–Green 张量 C = J^T J 的分量
    C11 = dX_dx0 * dX_dx0 + dY_dx0 * dY_dx0
    C22 = dX_dy0 * dX_dy0 + dY_dy0 * dY_dy0
    C12 = dX_dx0 * dX_dy0 + dY_dx0 * dY_dy0

    # 最大特征值解析式
    trace = C11 + C22
    diff = C11 - C22
    rad = torch.sqrt((diff * diff + 4.0 * (C12 * C12)).clamp_min(1e-30))
    lambda_max = 0.5 * (trace + rad)
    lambda_max = lambda_max.clamp_min(1e-30)

    # 时间跨度 ΔT(i,j)
    if (t0 is not None) and (t1 is not None):
        T_phys = (t1 - t0).abs().clamp_min(1e-12)
    else:
        # 若缺失时间通道，使用传入的 deltaT（默认 1.0）
        T_val = float(abs(deltaT)) if abs(deltaT) > 0 else 1e-12
        T_phys = torch.full_like(lambda_max, T_val)

    # FTLE: 0.5*log(lambda_max) / ΔT
    ftle_core = 0.5 * torch.log(lambda_max) / T_phys
    # 对分母为零的位置（由 zero_mask 标记）置 NaN，便于后续过滤
    ftle_core = torch.where(zero_mask,  torch.zeros_like(dY_dx0), ftle_core)

    # 还原成原始批次维度
    if flowMapRegularGrid.dim() == 3:
        return ftle_core.squeeze(0)
    return ftle_core
