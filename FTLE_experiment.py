#step 1: load FTLE dataset
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

from DeepUtils.utils import EasyConfig
from DeepUtils.loss import build_criterion_from_cfg
from DeepUtils.optim import build_optimizer_from_cfg

from FLowUtils.ScalarField2d import ScalarField2D,ScalarFieldManager
from FLowUtils.VectorField2d import *
from FLowUtils.netCDFLoader import *
from pnn.models.point_nn import EncNP
from pnn.libs.parallel_flows import run_pathlines_batched, batched_pathlines
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
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.DecoderLayer(x)
        return x.squeeze(-1)
    
class FTLERegressor(nn.Module):
    def __init__(self, input_points=1024, num_stages=2, embed_dim=72, k_neighbors=6, beta=100, alpha=1000):
        super().__init__()
        self.encoder = EncNP(input_points, num_stages, embed_dim, k_neighbors, alpha, beta)
        self.decoder = Decoder(in_dim=embed_dim * (2 ** (num_stages - 0)))

    def forward(self, pts: torch.Tensor):
        xyz = pts.permute(0, 2, 1)
        feat = self.encoder(xyz, pts)
        pred = self.decoder(feat)
        return pred
    

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
    xmin, ymin, _ = vectorfield.domainMinBoundary+0.1*np.array([xrange, yrange, 0])
    xmax, ymax, _ = vectorfield.domainMaxBoundary-0.1*np.array([xrange, yrange, 0])

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
    device: str = "cuda"
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    在线随机采样起点，生成 cross pathlines，并丢弃所有组内存在长度<max_steps 的样本，
    反复补采直到收集到 count 组“满长(max_steps)”的 pathlines。
    返回：
      - P_all:       [count*nerbors, max_steps, 3]
      - valid_all:   [count*nerbors] (恒等于 max_steps)
    """
    collected_groups = 0
    kept_P = []
    kept_V = []

    while collected_groups < count:
        need = count - collected_groups
        batch_groups = max(need, 16)
        starts_xy = sample_random_starts(vectorfield, count=batch_groups, device=str(device))
        P_b, V_b = batched_pathlines(
            field_tensor=torch.from_numpy(vectorfield.field).to(device),
            domain_min=vectorfield.domainMinBoundary,
            domain_max=vectorfield.domainMaxBoundary,
            tmin=vectorfield.tmin, tmax=vectorfield.tmax,
            time_interval=vectorfield.timeInterval,
            starts_xy=starts_xy,
            t_start=t_start, t_target=t_target,
            dt=dt, max_steps=max_steps,
            offsets_size=offset_dist
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
    return P_all, V_all
        



if __name__=="__main__":
    config = EasyConfig()
    config.load("config/dev_FMT_FTLE.yaml", recursive=False)

    # 1) 加载向量场 and precomputed FTLE
    vectorfield_datapath=f"{config.dataset.dat_dir}\\{config.dataset.name}.{config.dataset.extension}"
    netCDF = NetCDFLoader()
    vectorfield = netCDF.load_vector_field2d(vectorfield_datapath)
  
    # 3 模式切换
    mode = getattr(config, 'mode', 'point_regression')

    if mode == 'point_regression':
        # 在线随机采样起点 -> 生成 cross pathlines -> 传统FTLE作为标签 -> 训练
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 构造一个临时feature shape来初始化模型（N, nerbors*(K+1), 3）
        tmp_points_size = config.pcds.num_cross_points_per_seeding*(config.pcds.max_iterations)
        model = FTLERegressor(input_points=tmp_points_size, num_stages=config.pnn.stages, embed_dim=config.pnn.dim,
                              k_neighbors=config.pnn.k, beta=config.pnn.beta, alpha=config.pnn.alpha).to(device)
        #training params
        optimizer = build_optimizer_from_cfg(model, lr=config.lr, **config.optimizer)
        loss_fn = build_criterion_from_cfg(config.loss)
        model.train()
        epochs = 3
        batch_size = int(config.bs)
        print_freq = int(config.print_freq)
        epochs = int(config.epochs)

        #point cloud params
        K = int(config.pcds.sampled_points_per_line)
        nerb = int(config.pcds.num_cross_points_per_seeding)
        max_steps = int(config.pcds.max_iterations)
        dt = float(config.pcds.dt)
        offset_dist = vectorfield.gridInterval[0] * float(config.pcds.offset_scale)
        t_start = float(config.pcds.t_start * (vectorfield.tmax - vectorfield.tmin) + vectorfield.tmin)
        t_target = float(config.pcds.t_target * (vectorfield.tmax - vectorfield.tmin) + vectorfield.tmin)
        
        total_points_count = 32*40
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

        total_points = preprocess_localization_normalization(total_Pathline_points, nerb, K, config.pcds.localized, config.pcds.normalized, vectorfield).to(device).float()
        total_y = computeFTLEFromPathlineCrossPrimitive(total_points, sample_nerbors=nerb, line_steps=K, vectorfield=vectorfield)


        for epoch in range(epochs):
            epoch_avg_loss = 0
            for iter in range(iters_per_epoch):
                Pk = total_points[iter*batch_size:(iter+1)*batch_size]
                Pk = Pk.reshape(batch_size, nerb*(K), 3)
                label_y = total_y[iter*batch_size:(iter+1)*batch_size]
                label_y = label_y.to(device).float()

                points = Pk.to(device).permute(0, 2, 1)
                pred = model(points).to(device).float()
                loss = loss_fn(pred, label_y)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                epoch_avg_loss += loss.item()
                if iter % print_freq == 0:
                    print(f"epoch {epoch}, iter {iter}: loss={loss.item():.6f}")
           
            epoch_avg_loss /= iters_per_epoch
            print(f"epoch {epoch}: loss={epoch_avg_loss:.6f}")


    elif mode == 'volume_prediction':
        # 直接学习一个体素的回归可视化原型（此处留空接口，后续可接高维解码器）
        print("[volume_prediction] 暂留接口，建议使用卷积/UNet式解码器对 EncNP 全局特征做体生成。")

    elif mode == 'upsampling':
        # 未来模式：低分辨pathlines+低分辨FTLE -> 高分辨FTLE
        print("[upsampling] 暂留接口：准备双分支输入与监督。")
    else:
        raise ValueError(f"Unknown mode: {mode}")






#step 1: train model 