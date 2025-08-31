from torch.utils.data import Dataset
from FLowUtils.VectorField2d import UnsteadyVectorField2D
import torch,os
import numpy as np
import hashlib
from FLowUtils.ScalarField2d import ScalarField2D,ScalarFieldManager
import matplotlib.pyplot as plt
from pnn.libs.flows import temporal_downsamplePathlineCrossPrimitive, LocLines, normalizeLines
from FLowUtils.flowlineIntegral import batch_pathlineCross_integration_2D_auto
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
    valid_steps: [M]，M = N*nerbors，按 (seed0的5条, seed1的5条, ...) 排序
    返回:
      same:        [N] whether we have same length
      bad_groups:  [B] inequal length group index
      g_min, g_max:[N] min steps and max steps in each group
    
    DL: be careful that we only filter whether the grup have the same length, we need to check again whether we have correct length after sampling
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
    选出满足条件的组：
      - same == True（组内5条长度相等）
      - g_min >  K  (strict=True)  或  g_min >= K (strict=False)

    返回：
      keep_groups: [N]  组级布尔掩码
      keep_lines:  [M]  逐线布尔掩码（按组重复 nerbors 次）
      good_idx:    [G]  满足条件的组索引
    """
    same, _, g_min, _ = check_group_have_same_lengths(valid_steps, nerbors)
    cond_len = (g_min > K) if strict else (g_min >= K)
    # keep_groups = same & cond_len                     # [N]
    keep_groups = cond_len                     # [N]
    good_idx = torch.nonzero(keep_groups, as_tuple=False).squeeze(1)  # [G]
    keep_lines = keep_groups.repeat_interleave(nerbors)               # [M]
    return keep_groups, keep_lines, good_idx

def computeFTLEFromPathlineCrossPrimitive(points_grouped: torch.Tensor,
                                          sample_nerbors: int,
                                          line_steps: int,
                                          vectorfield: UnsteadyVectorField2D,
                                          step_idx: int | None = None) -> torch.Tensor:
    """
    传统FTLE算法（基于cross primitive 的5条pathlines）
    确保time value is in physcial time range not in normalized time range
    输入：
      - points_grouped: [N,sample_nerbors,(line_steps), 3]，按 (center, x+, x-, y+, y-) 顺序展开
      - sample_nerbors: 5（cross）
      - line_steps:     轨迹步数（K），即每条线点数为 K+1
      - vectorfield:    用于还原物理时间跨度 ΔT = (t_end - t_start)
      - step_idx:       取哪个时间步计算流映射，None 表示使用最后一步（K）

    输出：
      - ftle: [N] 每组中心采样点的 FTLE 值
    """
    assert points_grouped.dim() == 4 and points_grouped.shape[-1] >= 2
    N = int(points_grouped.shape[0])
    pts = points_grouped
    L=pts.shape[2]
    k = L - 1

    # 初始与最终点（物理坐标）
    # 邻居顺序约定：0=center, 1=x+, 2=x-, 3=y+, 4=y-
    p0 = pts[:, :, 0, :2]  # [N,5,2]
    pk = pts[:, :, k, :2]  # [N,5,2]

    # 初始偏移（标量，轴对齐）
    dx0 = (p0[:, 1, 0] - p0[:, 2, 0]).clamp_min(1e-12)  # [N]
    dy0 = (p0[:, 3, 1] - p0[:, 4, 1]).clamp_min(1e-12)  # [N]

    # 流映射雅可比 J = [dPhi/dx0, dPhi/dy0] ，列向量形式
    dPhi_dx = (pk[:, 1, :2] - pk[:, 2, :2]) / dx0.unsqueeze(-1)  # [N,2]
    dPhi_dy = (pk[:, 3, :2] - pk[:, 4, :2]) / dy0.unsqueeze(-1)  # [N,2]
    J = torch.stack([dPhi_dx, dPhi_dy], dim=-1)  # [N,2,2]

    # Cauchy–Green 张量 C = J^T J
    JT = J.transpose(1, 2)
    C = torch.bmm(JT, J)  # [N,2,2]

    # 最大特征值（C 对称半正定）
    eigvals = torch.linalg.eigvalsh(C)  # [N,2], 升序
    lambda_max = eigvals[:, 1].clamp_min(1e-12)

    # 物理时间跨度 ΔT（使用中心线时间）
    t_norm_start = pts[:, 0, 0, 2]
    t_norm_end   = pts[:, 0, k, 2]
    T_phys = (t_norm_end - t_norm_start).abs() 

    # 如果T_phys小于vectorfield.dt，说明没有得到正确的pathline，ftle值设置为0
    ftle = 0.5 * torch.log(lambda_max) / T_phys
    ftle = torch.where(T_phys <2* vectorfield.timeInterval , torch.zeros_like(ftle), ftle)
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
    nerbors: int = 5,
    cacheSystem: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    在线随机采样起点，生成 cross pathlines，并丢弃所有组内存在长度<max_steps 的样本，
    反复补采直到收集到 count 组“满长(max_steps)”的 pathlines。
    若启用 cacheSystem，则根据参数生成唯一 tag 并缓存到 outputs/temp/{tag}.npz，后续相同参数可直接加载。
    返回：
      - P_all:       [count,nerbors, max_steps, 3]
      - valid_all:   [count,nerbors] (恒等于 max_steps)
    """
    # 缓存逻辑：构造唯一 tag 并尝试加载
    if cacheSystem:
        dom_min = tuple(map(float, vectorfield.domainMinBoundary))
        dom_max = tuple(map(float, vectorfield.domainMaxBoundary))
        key_str = (
            f"cnt={count}|ms={max_steps}|dt={dt:.8g}|ts={t_start:.8g}|tt={t_target:.8g}|"
            f"off={offset_dist:.8g}|nb={nerbors}|tmin={float(vectorfield.tmin):.8g}|tmax={float(vectorfield.tmax):.8g}|"
            f"domMin={dom_min}|domMax={dom_max}"
        )
        tag = "TrainPL_" + hashlib.md5(key_str.encode("utf-8")).hexdigest()[:16]
        cache_dir = os.path.join("./outputs", "temp")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{tag}.npz")

        if os.path.exists(cache_path):
            try:
                data = np.load(cache_path)
                P_np = data["P"]
                V_np = data["V"]
                P_all = torch.from_numpy(P_np).float()
                V_all = torch.from_numpy(V_np).to(torch.int32)
                return P_all, V_all
            except Exception as e:
                print(f"[generate_training_samples] cache load failed: {e}. Regenerating...")
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
        if P_b.numel() == 0:#number of elements in the tensor
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
    # 只保留前 count 组（防御性切片）
    P_all = P_all[:count * nerbors]
    V_all = V_all[:count * nerbors]
    # 保存缓存
    if cacheSystem:
        try:
            # 使用同样的 tag 构造路径（若前面加载失败，需重新构造）
            dom_min = tuple(map(float, vectorfield.domainMinBoundary))
            dom_max = tuple(map(float, vectorfield.domainMaxBoundary))
            key_str = (
                f"cnt={count}|ms={max_steps}|dt={dt:.8g}|ts={t_start:.8g}|tt={t_target:.8g}|"
                f"off={offset_dist:.8g}|nb={nerbors}|tmin={float(vectorfield.tmin):.8g}|tmax={float(vectorfield.tmax):.8g}|"
                f"domMin={dom_min}|domMax={dom_max}"
            )
            tag = "TrainPL_" + hashlib.md5(key_str.encode("utf-8")).hexdigest()[:16]
            cache_dir = os.path.join("outputs", "temp")
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, f"{tag}.npz")
            np.savez(cache_path,
                     P=P_all.detach().cpu().numpy().astype(np.float32),
                     V=V_all.detach().cpu().numpy().astype(np.int32),
                     meta=key_str)
        except Exception as e:
            print(f"[generate_training_samples] cache save failed: {e}")
    return P_all, V_all
        


# Torch Dataset for training samples generated on-the-fly via generate_training_samples
class FTLETrainDataset(Dataset):
    def __init__(
        self,
        cfg,
        vectorfield: UnsteadyVectorField2D,
        count: int,
        max_steps: int,
        dt: float,
        t_start: float,
        t_target: float,
        offset_dist: float,
        nerbors: int,
        LstepsPerline: int,
        localized: bool,
        normalized: bool,
    ):
        # 1) 生成样本（跨路径线），只保留满长组
        P_all, V_all = generate_training_samples(
            vectorfield=vectorfield,
            count=int(count),
            max_steps=int(max_steps),
            dt=float(dt),
            t_start=float(t_start),
            t_target=float(t_target),
            offset_dist=float(offset_dist),
            nerbors=int(nerbors),
            cacheSystem=True,
        )
        NGroups,nerborsCrossSize,PathlineLength,Dim=P_all.shape
        assert NGroups==count
        assert nerborsCrossSize==nerbors
        assert PathlineLength==max_steps
        assert Dim==3


        # add test data
        # test_seeding,_,_ = generate_test_seeds(cfg,vectorfield)
        # P_test, V_test = batch_pathlineCross_integration_2D_auto(
        #     points=test_seeding,
        #     vectorfield=vectorfield,
        #     t_start=float(t_start), t_target=float(t_target),
        #     dt=float(dt), max_steps=int(max_steps),
        #     offsets_size=float(offset_dist), method="rk4"
        # )
        #concatenate P_all and P_test
        # P_all = torch.cat([P_all, P_test], dim=0)
        # V_all = torch.cat([V_all, V_test], dim=0)

        # 2) 重采样到固定 Kstep, the input tensor must have shape [Group,LInesPerGroup,timesteps,3]
        P_K = temporal_downsamplePathlineCrossPrimitive(P_all, int(LstepsPerline))  

        # 3) 计算每组的真值 FTLE（标签）
        y = computeFTLEFromPathlineCrossPrimitive(
            P_K, sample_nerbors=int(nerbors), line_steps=int(LstepsPerline), vectorfield=vectorfield
        ).cpu().float()
        self.ftle_min=y.min()
        self.ftle_max=y.max()
        normalized_y=(y-self.ftle_min)/(self.ftle_max-self.ftle_min)
        normalized_y=normalized_y.clamp(0,1)

        X = preprocess_localization_normalization(
            P_K, int(nerbors), int(LstepsPerline), bool(localized), bool(normalized), vectorfield
        ).cpu().float()

        # y_true_unsampled = computeFTLEFromPathlineCrossPrimitive(P_all, sample_nerbors=int(nerbors), line_steps=int(LstepsPerline), vectorfield=vectorfield)
        # diff=y_true_unsampled-y
        # #assert temporal downsample does not change the ftle value
        # print(f"diff: {diff.mean()}, {diff.std()}")
        # y_localization = computeFTLEFromPathlineCrossPrimitive(
        #     X , sample_nerbors=int(nerbors), line_steps=int(LstepsPerline), vectorfield=vectorfield
        # ).cpu().float()
        # diff=y_localization-y
        # # I already verified that the y_localization==y
        # print(f"diff: {diff.mean()}, {diff.std()}")

        X=X.reshape(NGroups,nerbors,LstepsPerline,3)
        self.points = X   # [N, nerb*K, 3]
        self.labels = normalized_y       # [N]

    def __len__(self):
        return self.points.shape[0]

    def __getitem__(self, idx):
        return self.points[idx], self.labels[idx]



def generate_test_seeds(cfg,vectorfield: UnsteadyVectorField2D):
   # 2) 生成均匀采样的起点网格（覆盖全域，步长=interval_scale*gridInterval）
    xmin, ymin, _ = vectorfield.domainMinBoundary
    xmax, ymax, _ = vectorfield.domainMaxBoundary
    gx, gy = vectorfield.gridInterval
    step_x = float(gx) * float(cfg.pcds.interval_scale)
    step_y = float(gy) * float(cfg.pcds.interval_scale)
    boundary_offset = 1e-1
    xs = np.arange(xmin+boundary_offset, xmax - boundary_offset, step_x, dtype=np.float32)
    ys = np.arange(ymin+boundary_offset, ymax - boundary_offset, step_y, dtype=np.float32)
    XX, YY = np.meshgrid(xs, ys)
    starts_xy = np.stack([XX.reshape(-1), YY.reshape(-1)], axis=1)
    return starts_xy,xs,ys



def sample_center_points_from_groups(points_grouped: torch.Tensor, sample_nerbors: int, line_steps: int) -> torch.Tensor:
    B = points_grouped.shape[0]
    pts = points_grouped.reshape(B, sample_nerbors, line_steps + 1, -1)
    center = pts[:, 0, 0, :]
    return center

def preprocess_localization_normalization(points_grouped: torch.Tensor, sample_nerbors: int, line_steps: int,
                               localized: bool, normalized: bool, vectorfield: UnsteadyVectorField2D) -> torch.Tensor:
    x = points_grouped
    if localized:
        x = LocLines(sample_nerbors=sample_nerbors, points=x)
    if normalized:
        x = normalizeLines(sample_nerbors=sample_nerbors, points=x, vectorfield=vectorfield)
    return x


def visualize_ftle_slice(true_grid: np.ndarray, pred_grid: np.ndarray,psnr: float, vectorfield: UnsteadyVectorField2D,
                         xs: np.ndarray, ys: np.ndarray, save_path: str | None = None,
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

    xmin, ymin, _ = vectorfield.domainMinBoundary
    xmax, ymax, _ = vectorfield.domainMaxBoundary
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

