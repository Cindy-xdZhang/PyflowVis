#step 1: load FTLE dataset
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

from DeepUtils.utils import EasyConfig
from DeepUtils.loss import build_criterion_from_cfg
from DeepUtils.optim import build_optimizer_from_cfg
import matplotlib.pyplot as plt
import hashlib

from FLowUtils.ScalarField2d import ScalarField2D,ScalarFieldManager
from FLowUtils.VectorField2d import *
from FLowUtils.netCDFLoader import *
from pnn.models.point_nn import EncNP
from FLowUtils.flowlineIntegral import *
from pnn.libs.flows import resample_to_fixed_count, LocLines, normalizeLines


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

    输入：
      - points_grouped: [N, sample_nerbors*(line_steps+1), 3]，按 (center, x+, x-, y+, y-) 顺序展开
      - sample_nerbors: 5（cross）
      - line_steps:     轨迹步数（K），即每条线点数为 K+1
      - vectorfield:    用于还原物理时间跨度 ΔT = (t_end - t_start)
      - step_idx:       取哪个时间步计算流映射，None 表示使用最后一步（K）

    输出：
      - ftle: [N] 每组中心采样点的 FTLE 值
    """
    assert points_grouped.dim() == 3 and points_grouped.shape[-1] >= 2
    N = points_grouped.shape[0]
    L = line_steps 

    pts = points_grouped.reshape(N, sample_nerbors, L, -1)

    # 选择时间步：默认最后一步
    k = line_steps if step_idx is None else int(step_idx)
    k = max(0, min(L - 1, k))

    # 初始与最终点（物理坐标）
    # 邻居顺序约定：0=center, 1=x+, 2=x-, 3=y+, 4=y-
    p0 = pts[:, :, 0, :2]  # [N,5,2]
    pk = pts[:, :, k, :2]  # [N,5,2]

    # 初始偏移（标量，轴对齐）
    dx0 = (p0[:, 1, 0] - p0[:, 2, 0]).abs().clamp_min(1e-12)  # [N]
    dy0 = (p0[:, 3, 1] - p0[:, 4, 1]).abs().clamp_min(1e-12)  # [N]

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
    T_phys = (t_norm_end - t_norm_start).abs() * (vectorfield.tmax - vectorfield.tmin)
    T_phys = T_phys.clamp_min(1e-12)

    # FTLE = (1/(2|T|)) * ln(lambda_max)
    ftle = 0.5 * torch.log(lambda_max) / T_phys
    return ftle

class Decoder(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.DecoderLayer = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.DecoderLayer(x.contiguous())
        return x.squeeze(-1)
    
class FTLERegressor(nn.Module):
    def __init__(self, KStpesPerline,num_stages=2, embed_dim=72, k_neighbors=6, cross_neighborsize=5,beta=100, alpha=1000):
        super().__init__()
        #FTLERegressor is a point wise regressor, has no cross primitive, no neighbor information aggregation
        self.cross_neighborsize=cross_neighborsize
        self.pointsPerPrimitive=KStpesPerline*cross_neighborsize
        self.encoder = EncNP(self.pointsPerPrimitive, num_stages, embed_dim, k_neighbors, alpha, beta)

        self.decoderInputDim=embed_dim * (2 ** (num_stages - 0))+ 3*2*cross_neighborsize
        self.decoder = Decoder(in_dim=self.decoderInputDim)

    def forward(self, pts: torch.Tensor):
        B,_,N,=pts.shape
        xyz = pts.permute(0, 2, 1)
        feat = self.encoder(xyz, pts)
        #feat: (B, embed_dim)
        start_pos=xyz.view(B,-1,self.cross_neighborsize, 3)[:,0,:,:].reshape(B, -1)
        end_pos=xyz.view(B,-1,self.cross_neighborsize,3)[:,-1,:,:].reshape(B, -1)

        every_cross_feature=torch.cat([feat,start_pos,end_pos],dim=1)

        #feature dimension is in_dim*K(steps per line)
        pred = self.decoder(every_cross_feature)
        return pred
    

# Torch Dataset for training samples generated on-the-fly via generate_training_samples
class FTLETrainDataset(Dataset):
    def __init__(
        self,
        vectorfield: UnsteadyVectorField2D,
        count: int,
        max_steps: int,
        dt: float,
        t_start: float,
        t_target: float,
        offset_dist: float,
        nerbors: int,
        K: int,
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
            device="cpu",
            cacheSystem=True,
        )

        # 2) 重采样到固定 K，并按组拼接为 [N, nerb*K, 3]
        P_K = resample_to_fixed_count(P_all.cpu(), V_all.cpu(), int(K))  # [count*nerb, K, 3]
        group_N = P_K.shape[0] // int(nerbors)
        P_grouped = P_K.view(group_N, int(nerbors), int(K), -1).reshape(group_N, int(nerbors)*int(K), -1)

        # 3) 计算每组的真值 FTLE（标签）
        #    先恢复为 [N, nerb, K, 3] 再喂入 primitive 计算
        P_grouped_for_label = P_K.view(group_N, int(nerbors), int(K), -1)
        y = computeFTLEFromPathlineCrossPrimitive(
            P_grouped_for_label, sample_nerbors=int(nerbors), line_steps=int(K), vectorfield=vectorfield
        ).cpu().float()

        # 4) 预处理（本地化/归一化），作为网络输入 [N, nerb*K, 3]
        X = preprocess_localization_normalization(
            P_grouped, int(nerbors), int(K), bool(localized), bool(normalized), vectorfield
        ).cpu().float()

        # 存储为 CPU Tensor，配合 DataLoader 的 pin_memory 提升拷贝效率
        self.points = X.contiguous()         # [N, nerb*K, 3]
        self.labels = y.contiguous()         # [N]

    def __len__(self):
        return self.points.shape[0]

    def __getitem__(self, idx):
        return self.points[idx], self.labels[idx]

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

def sample_center_points_from_groups(points_grouped: torch.Tensor, sample_nerbors: int, line_steps: int) -> torch.Tensor:
    B = points_grouped.shape[0]
    pts = points_grouped.reshape(B, sample_nerbors, line_steps + 1, -1)
    center = pts[:, 0, 0, :]
    return center

def gather_ftle_labels_at_points(center_pts_phys_t: torch.Tensor, ftle_volume: ScalarField2D) -> torch.Tensor:
    device = center_pts_phys_t.device
    labels = []
    for i in range(center_pts_phys_t.shape[0]):
        x, y, t_norm = center_pts_phys_t[i].tolist()
        t_phys = t_norm * (ftle_volume.tmax - ftle_volume.tmin) + ftle_volume.tmin
        val = ftle_volume.get_value_at_physical_pos(x, y, t_phys)
        labels.append(val)
    return torch.tensor(labels, device=device, dtype=torch.float32)

def preprocess_localization_normalization(points_grouped: torch.Tensor, sample_nerbors: int, line_steps: int,
                               localized: bool, normalized: bool, vectorfield: UnsteadyVectorField2D) -> torch.Tensor:
    x = points_grouped
    if localized:
        x = LocLines(sample_nerbors=sample_nerbors, points=x)
    if normalized:
        x = normalizeLines(sample_nerbors=sample_nerbors, points=x, vectorfield=vectorfield)
    return x

def make_optimizer(model: nn.Module, lr: float = 1e-3, wd: float = 1e-5):
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


def sample_random_starts(vectorfield: UnsteadyVectorField2D, count: int, device: str = "cuda") -> torch.Tensor:
    xrange = vectorfield.domainMaxBoundary[0] - vectorfield.domainMinBoundary[0]
    yrange = vectorfield.domainMaxBoundary[1] - vectorfield.domainMinBoundary[1]
    xmin, ymin, _ = vectorfield.domainMinBoundary+0.01*np.array([xrange, yrange, 0])
    xmax, ymax, _ = vectorfield.domainMaxBoundary-0.01*np.array([xrange, yrange, 0])

    rx = torch.rand(count, device=device)
    ry = torch.rand(count, device=device)
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
    device: str = "cuda",
    cacheSystem: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    在线随机采样起点，生成 cross pathlines，并丢弃所有组内存在长度<max_steps 的样本，
    反复补采直到收集到 count 组“满长(max_steps)”的 pathlines。
    若启用 cacheSystem，则根据参数生成唯一 tag 并缓存到 outputs/temp/{tag}.npz，后续相同参数可直接加载。
    返回：
      - P_all:       [count*nerbors, max_steps, 3]
      - valid_all:   [count*nerbors] (恒等于 max_steps)
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
        tag = "PL_" + hashlib.md5(key_str.encode("utf-8")).hexdigest()[:16]
        cache_dir = os.path.join("./outputs", "temp")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{tag}.npz")

        if os.path.exists(cache_path):
            try:
                data = np.load(cache_path)
                P_np = data["P"]
                V_np = data["V"]
                P_all = torch.from_numpy(P_np).to(device).float()
                V_all = torch.from_numpy(V_np).to(device).to(torch.int32)
                return P_all, V_all
            except Exception as e:
                print(f"[generate_training_samples] cache load failed: {e}. Regenerating...")
    collected_groups = 0
    kept_P = []
    kept_V = []

    while collected_groups < count:
        need = count - collected_groups
        batch_groups = max(need, 16)
        starts_xy = sample_random_starts(vectorfield, count=batch_groups, device=str(device))
        P_b, V_b = batch_pathlineCross_integration_2D_auto(
            points=starts_xy.detach().cpu().numpy(),
            vectorfield=vectorfield,
            t_start=float(t_start), t_target=float(t_target),
            dt=float(dt), max_steps=int(max_steps),
            offsets_size=float(offset_dist), method="rk4"
        )
        # 按组筛选“满长”
        if P_b.numel() == 0:
            continue
        M = P_b.shape[0]
        assert M % nerbors == 0, "lines count must be multiple of nerbors"
        G = M // nerbors
        P_g = P_b.view(G, nerbors, max_steps, -1)
        V_g = V_b.view(G, nerbors)
        good_mask = (V_g == max_steps).all(dim=1)  # [G]
        good_idx = torch.nonzero(good_mask, as_tuple=False).squeeze(1)
        if good_idx.numel() == 0:
            continue

        take = int(min(need, good_idx.numel()))
        good_idx = good_idx[:take]

        P_sel = P_g[good_idx].reshape(-1, max_steps, P_b.shape[-1])
        V_sel = V_g[good_idx].reshape(-1)

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
            tag = "PL_" + hashlib.md5(key_str.encode("utf-8")).hexdigest()[:16]
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
        

def test_ftle_from_model(cfg,model: nn.Module, vectorfield: UnsteadyVectorField2D, time:float, device: str = "cuda", save_path: str | None = None,
                        starts_chunk: int = 64):
    #first generate grid points for this time, then generate pathlines, 
    # then call computeFTLEFromPathlineCrossPrimitive get correct ftle
    # then compare the ftle from model and the correct ftle
    #report the error
    nerb = int(cfg.pcds.num_cross_points_per_seeding)
    K = int(cfg.pcds.sampled_points_per_line)
    max_steps = int(cfg.pcds.max_iterations)
    dt = float(cfg.pcds.dt)
    offset_dist = vectorfield.gridInterval[0] * float(cfg.pcds.offset_scale)
    localized = bool(cfg.pcds.localized)
    normalized = bool(cfg.pcds.normalized)

    # 时间设置：以输入的 physical time 为起始，目标时间与训练保持相同的时间跨度
    tmin, tmax = float(vectorfield.tmin), float(vectorfield.tmax)
    base_t_start_ratio = float(cfg.pcds.t_start)
    base_t_target_ratio = float(cfg.pcds.t_target)
    delta_T = (base_t_target_ratio - base_t_start_ratio) * (tmax - tmin)
    t_start = float(time)
    t_target = float(np.clip(t_start + delta_T, tmin, tmax))

    # 2) 生成均匀采样的起点网格（覆盖全域，步长=interval_scale*gridInterval）
    xmin, ymin, _ = vectorfield.domainMinBoundary
    xmax, ymax, _ = vectorfield.domainMaxBoundary
    gx, gy = vectorfield.gridInterval
    step_x = float(gx) * float(cfg.pcds.interval_scale)
    step_y = float(gy) * float(cfg.pcds.interval_scale)
    boundary_offset = 5e-2
    xs = np.arange(xmin+boundary_offset, xmax - boundary_offset, step_x, dtype=np.float32)
    ys = np.arange(ymin+boundary_offset, ymax - boundary_offset, step_y, dtype=np.float32)
    XX, YY = np.meshgrid(xs, ys)
    starts_xy = torch.from_numpy(np.stack([XX.reshape(-1), YY.reshape(-1)], axis=1)).to(device).float()

    # 3) 分批生成 pathlines 与分批预测，避免内存溢出
    ny, nx = len(ys), len(xs)
    true_grid = np.full((ny, nx), 0, dtype=np.float32)
    pred_grid = np.full((ny, nx), 0, dtype=np.float32)

    total_groups = 0
    se_sum = 0.0
    ae_sum = 0.0
    max_abs_err = 0.0
    y_global_min = float('inf')
    y_global_max = float('-inf')

    M_all = starts_xy.shape[0]
    with torch.no_grad():
        for s0 in range(0, M_all, max(1, int(starts_chunk))):
            s1 = min(M_all, s0 + int(starts_chunk))
            starts_xy_b = starts_xy[s0:s1]

            P_b, V_b = batch_pathlineCross_integration_2D_auto(
                points=starts_xy_b.detach().cpu().numpy(),
                vectorfield=vectorfield,
                t_start=float(t_start), t_target=float(t_target),
                dt=float(dt), max_steps=int(max_steps),
                offsets_size=float(offset_dist), method="rk4"
            )
            if P_b.numel() == 0:
                continue

            assert P_b.shape[0] % nerb == 0, "lines count must be multiple of nerbors"
            G_b = P_b.shape[0] // nerb
            P_g = P_b.view(G_b, nerb, max_steps, -1)
            V_g = V_b.view(G_b, nerb)
            keep_groups = (V_g == max_steps).all(dim=1)
            if not keep_groups.any():
                continue

            keep_local = torch.nonzero(keep_groups, as_tuple=False).squeeze(1)
            keep_global = (keep_local + s0).cpu().numpy()

            # 重采样到 K
            P_kept = P_g[keep_groups].reshape(-1, max_steps, P_b.shape[-1])
            V_kept = V_g[keep_groups].reshape(-1)
            P_K = resample_to_fixed_count(P_kept, V_kept, K)                # [Ng*nerb, K, 3]
            Ng_b = P_K.shape[0] // nerb
            P_K = P_K.view(Ng_b, nerb, K, -1).reshape(Ng_b, nerb*K, -1)    # [Ng, nerb*K, 3]

            # 计算真值 FTLE（按组）
            y_true_b = computeFTLEFromPathlineCrossPrimitive(P_K, sample_nerbors=nerb, line_steps=K, vectorfield=vectorfield)
            y_true_b = y_true_b.to(device).float()
            y_global_min = min(y_global_min, float(y_true_b.min().item()))
            y_global_max = max(y_global_max, float(y_true_b.max().item()))

            # 预处理并分批预测
            P_in = preprocess_localization_normalization(P_K, nerb, K, localized, normalized, vectorfield).to(device).float()
            P_in = P_in.reshape(Ng_b, nerb*(K), 3)
            points_all = P_in.to(device).permute(0, 2, 1)
            pred_b = model(points_all).to(device).float()

            # 误差累计
            diff = pred_b - y_true_b
            se_sum += float((diff ** 2).sum().item())
            ae_sum += float(diff.abs().sum().item())
            max_abs_err = max(max_abs_err, float(diff.abs().max().item()))
            total_groups += int(Ng_b)

            # 回填网格
            rows = keep_global // nx
            cols = keep_global % nx
            true_grid[rows, cols] = y_true_b.detach().cpu().numpy()
            pred_grid[rows, cols] = pred_b.detach().cpu().numpy()

        if total_groups == 0:
            print("[test_ftle] No full-length groups found across all chunks.")
            return

        mse = se_sum / total_groups
        mae = ae_sum / total_groups
        maxe = max_abs_err
        dyn_range = max(abs(y_global_max - y_global_min), 1e-12)
        psnr = float('inf') if mse <= 1e-20 else 20.0 * np.log10(dyn_range) - 10.0 * np.log10(mse)
        print(f"[test_ftle] Ngroups={total_groups}, MSE={mse:.6f}, MAE={mae:.6f}, MaxE={maxe:.6f}, PSNR={psnr:.2f} dB")

        # 可视化 2D 切片
        visualize_ftle_slice(true_grid, pred_grid,psnr, vectorfield, xs, ys, save_path=save_path)

        return {
            "groups": int(total_groups),
            "mse": mse,
            "mae": mae,
            "maxe": maxe,
            "psnr": psnr,
            "true_grid": true_grid,
            "pred_grid": pred_grid,
            "xs": xs,
            "ys": ys
        }

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
    vmin_p, vmax_p = robust_minmax(pred_grid)
    # vmin = min(vmin_t, vmin_p)
    # vmax = max(vmax_t, vmax_p)
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



if __name__=="__main__":
    config = EasyConfig()
    config.load("config/dev_FMT_FTLE.yaml", recursive=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) 加载向量场 and precomputed FTLE
    vectorfield_datapath=f"{config.dataset.dat_dir}\\{config.dataset.name}.{config.dataset.extension}"
    netCDF = NetCDFLoader()
    vectorfield = netCDF.load_vector_field2d(vectorfield_datapath)
  
    # 3 模式切换
    mode = getattr(config, 'mode', 'point_regression')

    if mode == 'point_regression':
        total_points_count = 20000
        nerb = int(config.pcds.num_cross_points_per_seeding)
        K = int(config.pcds.sampled_points_per_line)
        #the input point cloud tensor runing in the network is (B,  N=K*nerb, 3)
        model = FTLERegressor(KStpesPerline=K, num_stages=config.pnn.stages, embed_dim=config.pnn.dim,
                              k_neighbors=config.pnn.k, beta=config.pnn.beta, alpha=config.pnn.alpha).to(device)
        #training params
        optimizer = build_optimizer_from_cfg(model, lr=config.lr, **config.optimizer)
        loss_fn = build_criterion_from_cfg(config.loss)
        model.train()
        batch_size = int(config.bs)
        print_freq = int(config.print_freq)
        epochs = int(config.epochs)

        #point cloud params
        max_steps = int(config.pcds.max_iterations)
        dt = float(config.pcds.dt)
        offset_dist = vectorfield.gridInterval[0] * float(config.pcds.offset_scale)
        t_start = float(config.pcds.t_start * (vectorfield.tmax - vectorfield.tmin) + vectorfield.tmin)
        t_target = float(config.pcds.t_target * (vectorfield.tmax - vectorfield.tmin) + vectorfield.tmin)
        
        iters_per_epoch = int(total_points_count/batch_size)


        #generate training samples: 先只保留“满长(max_steps)”的组，再随机K点重采样
        total_P, total_valid = generate_training_samples(
            vectorfield=vectorfield,
            count=total_points_count,
            max_steps=max_steps,
            dt=dt,
            t_start=t_start,
            t_target=t_target,
            offset_dist=offset_dist,
            nerbors=nerb,
            device=str(device)
        )

        # 先对每条线重采样到 K 个 time-slices，然后按组(nerbors条/组)拼回 [N, nerbors*K, 3]
        total_Pathline_points = resample_to_fixed_count(total_P, total_valid, K)  # [count*nerb, K, 3]
        group_N = total_Pathline_points.shape[0] // nerb
        total_Pathline_points = total_Pathline_points.view(group_N, nerb, K, -1).reshape(group_N, nerb*K, -1)

        total_y = computeFTLEFromPathlineCrossPrimitive(total_Pathline_points, sample_nerbors=nerb, line_steps=K, vectorfield=vectorfield)

        total_points = preprocess_localization_normalization(total_Pathline_points, nerb, K, config.pcds.localized, config.pcds.normalized, vectorfield).to(device).float()


        for epoch in range(epochs):
            epoch_avg_loss = 0
            for iter in range(iters_per_epoch):
                Pk = total_points[iter*batch_size:(iter+1)*batch_size]
                Pk = Pk.reshape(batch_size, nerb*(K), 3)
                label_y = total_y[iter*batch_size:(iter+1)*batch_size]
                label_y = label_y.to(device).float()

                points = Pk.to(device).permute(0, 2, 1)
                pred = model(points).to(device).float()
                loss = 10.0*loss_fn(pred, label_y)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                epoch_avg_loss += loss.item()
                if iter % print_freq == 0:
                    print(f"epoch {epoch}, iter {iter}: loss={loss.item():.6f}")
           
            epoch_avg_loss /= iters_per_epoch
            print(f"epoch {epoch}: loss={epoch_avg_loss:.6f}")

        #step 2: test model
        test_ftle_from_model(config,model, vectorfield, time=t_start, device=str(device))
    elif mode == 'volume_prediction':
        # 直接学习一个体素的回归可视化原型（此处留空接口，后续可接高维解码器）
     pass


    elif mode == 'upsampling':
        # 未来模式：低分辨pathlines+低分辨FTLE -> 高分辨FTLE
        print("[upsampling] 暂留接口：准备双分支输入与监督。")
    else:
        raise ValueError(f"Unknown mode: {mode}")

