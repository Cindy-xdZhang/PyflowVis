import torch
import numpy as np
from FLowUtils.VectorField2d import *
from FLowUtils.flowDatasetUtils.NetCDF_AmiraLoader import NetCDFLoader




def LocLines(sample_nerbors,points):
    """
    Function to localization the points to the first points
    
    input:
        sample_nerbors: 5
        points: N,3
    
    output:
        points_loc: N,3 # the points convert to local space
    """
    points_ori=points.reshape(points.shape[0],sample_nerbors,-1,points.shape[-1])
    
    # x: [N, sample, step, 3]
    first_pts = points_ori[:, 0, 0, :]            # 取出每条线的第一个点，形状 [N, sample, 3]
    first_pts = first_pts.unsqueeze(1).unsqueeze(2)   # 在 step 维度上扩展成 [N, sample, 1, 3]

    x_shifted = points_ori - first_pts            # 自动广播为 [N, sample, step, 3]
    return x_shifted



def LocLines4Time(sample_nerbors,points):
    """
    Function to localization the points along the lines
    
    input:
        sample_nerbors: 5
        points: N,3
    
    output:
        points_loc: N,3 # the points convert to local space
    """
    pts = points.reshape(points.shape[0],sample_nerbors,-1,points.shape[-1])

    center = pts[:, 0:1, :, :]       # [B, 1, step, 3]
    neighbors = pts[:, 1:, :, :]     # [B, 4, step, 3]
    neighbors_loc = neighbors - center  #  [B,4,step,3]
    return neighbors_loc,center
    
def normalizeLines(sample_nerbors,points,vectorfield):
    """
    Function to normalize the points to the first points
    
    input:
        sample_nerbors: 5
        points: N,3
        vectorfeild: provide the field scale
    output:
        points_loc: N,3 # the points convert to local space
    """
    
    x_scale=vectorfield.domainMaxBoundary[0]-vectorfield.domainMinBoundary[0]
    y_scale=vectorfield.domainMaxBoundary[1]-vectorfield.domainMinBoundary[1]
    t_scale=vectorfield.tmax-vectorfield.tmin
    
    points_ori=points.reshape(points.shape[0],sample_nerbors,-1,points.shape[-1])
    device=points_ori.device
    mins = torch.tensor(
        [vectorfield.domainMinBoundary[0],
         vectorfield.domainMinBoundary[1],
         vectorfield.tmin],
        device=device,
        dtype=points_ori.dtype
    )
    maxs = torch.tensor(
        [vectorfield.domainMaxBoundary[0],
         vectorfield.domainMaxBoundary[1],
         vectorfield.tmax],
        device=device,
        dtype=points_ori.dtype
    )
    scales = maxs - mins  # shape (3,)

    # 变形成可以广播到 [N, sample, step, 3]
    mins   = mins.view(1, 1, 1, 3)    # [1,1,1,3]
    scales = scales.view(1, 1, 1, 3)  # [1,1,1,3]
    lines_norm = (points_ori - mins) / scales
    return lines_norm
    

@torch.no_grad()
def AngleAwareSampling(points: torch.Tensor, K: int) -> torch.Tensor:
    """
    Perceptual/importance-driven temporal sampling for pathline-cross primitives.
    -  points: tensor, shape [groups, lines_per_group, L_timesteps, D]
    - outputs:       [G, LinesPerGroup, K, D]

    策略：
    按时间估计“转向角”显著性（基于切向向量夹角），对每组在 1..T-2 中选取 top-(K-2) 的时刻，
       并与首(0)尾(T-1)一起作为采样索引；组内所有 L 条线共享同一组选取的时间索引。
    """
    if K < 2:
        raise ValueError("K must be >= 2 to keep both head and tail")
    if points.dim() != 4:
        raise ValueError("points must have shape [G, L, T, D]")
    device = points.device
    G, L, T, D = points.shape[0], points.shape[1], points.shape[2], points.shape[3]
    assert K <= T, "K must be <= number of timesteps T"


    # 计算基于切向矢量的转角显著性：角度越大越重要
    # 仅使用前两维 (x,y) 来估计方向变化
    Pxy = points[..., :2]  # [G,L,T,2]
    if T <= 2:
        # 这里 K < T 已经保证 T>=3，否则会走上面的 K>=T 分支
        return points[:, :, :K, :]

    V = Pxy[:, :, 1:, :] - Pxy[:, :, :-1, :]          # [G,L,T-1,2]
    V_prev = V[:, :, :-1, :]                           # [G,L,T-2,2]
    V_next = V[:, :, 1:, :]                            # [G,L,T-2,2]
    eps = 1e-8
    dot = (V_prev * V_next).sum(dim=-1)                # [G,L,T-2]
    n0 = V_prev.norm(dim=-1).clamp_min(eps)            # [G,L,T-2]
    n1 = V_next.norm(dim=-1).clamp_min(eps)            # [G,L,T-2]
    cos_theta = (dot / (n0 * n1)).clamp(-1.0, 1.0)     # [G,L,T-2]
    angle = torch.acos(cos_theta)                      # [G,L,T-2]

    # 全局聚合（跨 G 和 L）得到每个时间步的显著性
    saliency_global = angle.mean(dim=(0, 1))           # [T-2]

    # 选出中间的 K-2 个时间索引（位于 [1..T-2]），再与 0 和 T-1 合并，生成 1-D sel_idx
    m = max(K - 2, 0)
    if m > 0:
        mid_idx = torch.topk(saliency_global, k=m, dim=0, largest=True, sorted=False).indices  # [m] in [0..T-3]
        mid_idx = mid_idx + 1  # shift 到真实时间索引 [1..T-2]
        mid_idx, _ = torch.sort(mid_idx)               # 升序
        sel_idx = torch.cat([
            torch.tensor([0], device=device, dtype=torch.long),
            mid_idx,
            torch.tensor([T - 1], device=device, dtype=torch.long)
        ], dim=0)                                      # [K]
    else:
        sel_idx = torch.tensor([0, T - 1], device=device, dtype=torch.long)

    out = points[:, :, sel_idx, :]
    return out



@torch.no_grad()
def temporal_downsamplePathlineCrossPrimitiveRandom(points: torch.Tensor,  K: int) -> torch.Tensor:
    """
    Resample pathlines along the time dimension to a fixed number of steps K: keep head and tail, randomly sample the middle.
    Args:
      - points: tensor, shape [groups, lines_per_group, timesteps, ...]
      - K: target number of sampled steps (>=2)

    Returns:
      - Tensor with the same number of dimensions as points, but the time dimension is K
    """
    if points.dim() != 4:
        raise ValueError("points must have shape [G, L, T, D]")
    device = points.device
    G, L, T, D = points.shape[0], points.shape[1], points.shape[2], points.shape[3]
    assert K <= T and K >= 2, "K must be <= number of timesteps T and >= 2"

    if K == 2:
        sel_idx = torch.tensor([0, T - 1], device=device, dtype=torch.long)
    else:
        num_mid = K - 2
        # Randomly sample the middle indices from [1, T-2] (globally consistent), and sort by time
        idx_mid = torch.randint(1, T - 1, (num_mid,), device=device)
        idx_mid, _ = torch.sort(idx_mid)
        sel_idx = torch.cat([
            torch.tensor([0], device=device, dtype=torch.long),
            idx_mid,
            torch.tensor([T - 1], device=device, dtype=torch.long)
        ], dim=0)
        
    out = points[:, :, sel_idx, :]
    return out



@torch.no_grad()
def temporal_downsamplePathlineCrossPrimitiveRegular(points: torch.Tensor, K: int) -> torch.Tensor:
    """
    均匀间隔下采样（时间维）：保留首尾，并在 (1..T-2) 区间内等间距选取 num_mid=K-2 个索引。
    Args:
      - points: [G, L, T, D]
      - K: 目标时间步数（>=2 且 <= T）
    Returns:
      - [G, L, K, D]
    """
    if points.dim() != 4:
        raise ValueError("points must have shape [G, L, T, D]")
    device = points.device
    G, L, T, D = points.shape[0], points.shape[1], points.shape[2], points.shape[3]
    assert 2 <= K <= T, "K must be <= number of timesteps T and >= 2"

    if K == 2:
        sel_idx = torch.tensor([0, T - 1], device=device, dtype=torch.long)
    else:
        num_mid = K - 2
        if T - 2 <= 0:
            # 没有中间可选步，退化为首尾
            sel_idx = torch.tensor([0, T - 1], device=device, dtype=torch.long)
        elif num_mid >= (T - 2):
            # 取尽所有中间索引
            mid_idx = torch.arange(1, T - 1, device=device, dtype=torch.long)
            sel_idx = torch.cat([
                torch.tensor([0], device=device, dtype=torch.long),
                mid_idx,
                torch.tensor([T - 1], device=device, dtype=torch.long)
            ], dim=0)
        else:
            # 使用 linspace 在 [1, T-2] 上等间距选取 num_mid 个位置，并四舍五入到最近整数
            mid_pos = torch.linspace(1, T - 2, steps=num_mid, device=device)
            mid_idx = torch.round(mid_pos).to(torch.long)
            # 保序与去重（极少数情况下 round 可能产生重复，保持稳定性）
            mid_idx = torch.unique(mid_idx, sorted=True)
            # 若因去重不足，则用 clamp+补齐策略
            while mid_idx.numel() < num_mid:
                # 尝试在剩余可用集合中插入缺少的索引（简单策略：扩展邻近位置）
                need = num_mid - mid_idx.numel()
                candidates = torch.arange(1, T - 1, device=device, dtype=torch.long)
                mask = torch.ones_like(candidates, dtype=torch.bool)
                mask[mid_idx - 1] = False if mid_idx.numel() > 0 else True
                add = candidates[mask][:need]
                mid_idx = torch.sort(torch.cat([mid_idx, add]))[0]

            sel_idx = torch.cat([
                torch.tensor([0], device=device, dtype=torch.long),
                mid_idx[:num_mid],
                torch.tensor([T - 1], device=device, dtype=torch.long)
            ], dim=0)

    out = points[:, :, sel_idx, :]
    return out




