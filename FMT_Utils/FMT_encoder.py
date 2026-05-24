import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from FMT_Utils.FlowlinePostProcessing import AngleAwareSampling
def _tiling_starts(length: int, k: int):
    if k >= length:
        return [0]
    s = int(k)
    starts = list(range(0, length - k + 1, s))
    last = length - k
    if starts[-1] != last:
        starts.append(last)
    return starts


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx



class kNN(nn.Module):
    def __init__(self,  k_neighbors):
        super().__init__()
        self.k_neighbors = k_neighbors
    def forward(self, xyz, x):
        # xyz: (B, N, 3)
        # x: (B, N, C)
        B, N, _ = xyz.shape
        lc_xyz = xyz
        lc_x = x
        # kNN
        knn_idx = knn_point(self.k_neighbors, xyz, lc_xyz)
        knn_xyz = index_points(xyz, knn_idx)
        knn_x = index_points(x, knn_idx)

        return lc_xyz, lc_x, knn_xyz, knn_x



"""
Our data is not unordered points, KNN dosn't make sense. and it's also time consuming.
how to describe shape of pathlineCluster in an objective way? 
pathlines shape [B, N=neighbors*Ltimestep,3])->reshape as  [B, neighbors,Ltimestep,3]) and every point only pick nearby points at the same timestep as it's neighbor..
"""
class GeoLinePicker(nn.Module):
    def __init__(self, LSteps:int):
        super().__init__()
        self.LSteps= int(LSteps)

    def forward(self, xyz: torch.Tensor, x: torch.Tensor):
        """
        xyz: [B, N, 3]，x: [B, N, C]
        N =  neighbors(K) * Ltimestep
        重排为 [B, K, L=Ltimestep]，每个点仅选取同一时间步 t 的 另外k-1条线的点作为邻居。
        返回：
          - lc_xyz: [B, N, 3]
          - lc_x:   [B, N, C]
          - knn_xyz:[B, N, K, 3]
          - knn_x:  [B, N, K, C]
        """

        B, N, _ = xyz.shape
        L = int(self.LSteps)
        assert N % ( L) == 0, f"N must be divisible by 5*Lstep (5*{L}), got N={N}"
        P = N // ( L)
        device = xyz.device
        # 1. 每个点的 timestep 编号 (0..L-1)
        falttern_one_idx = torch.arange(N, device=device) # [N]
        every_point_timestep_idx = falttern_one_idx % L # [N]
        # points share same timestep Li their flattern idx are:  (Pid=0,1,2....P-1)*L+Li
        every_point_neigborPoints_idx = every_point_timestep_idx.unsqueeze(1)  + (torch.arange(P, device=device) * L).unsqueeze(0)

        # 4. 扩展 batch
        idx_batched = every_point_neigborPoints_idx.unsqueeze(0).expand(B, -1, -1)  # [B, N, P]

        knn_xyz = index_points(xyz, idx_batched)  # [B, N, P, 3]
        knn_x = index_points(x, idx_batched)      # [B, N, P, C]



        lc_xyz = xyz
        lc_x = x

        return lc_xyz, lc_x, knn_xyz, knn_x



# Pooling
class Pooling(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.out_transform = nn.Sequential(
                nn.BatchNorm1d(out_dim),
                nn.GELU())

    def forward(self, knn_x_w):
        # knn_x_w: (B, out_dim, N, K)
        # Feature Aggregation (Pooling along the KNN dimension as merge nerighboring information)
        lc_x = knn_x_w.max(-1)[0] + knn_x_w.mean(-1)
        lc_x = self.out_transform(lc_x)
        return lc_x


# PosE for Raw-point Embedding 
class PosE_Initial(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta

    def forward(self, xyz):
        # xyz: (B,  3, N)
        B, _, N = xyz.shape    
        feat_dim = self.out_dim // (self.in_dim * 2)
        
        feat_range = torch.arange(feat_dim, device=xyz.device, dtype=torch.float32)
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim)
        div_embed = torch.div(self.beta * xyz.unsqueeze(-1), dim_embed)

        sin_embed = torch.sin(div_embed)
        cos_embed = torch.cos(div_embed)
        position_embed = torch.stack([sin_embed, cos_embed], dim=4).flatten(3)
        position_embed = position_embed.permute(0, 1, 3, 2).reshape(B, self.out_dim, N)
        
        return position_embed


# PosE for Local Geometry Extraction
class PosE_Geo(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta
        
    def forward(self, knn_xyz, knn_x):
        B, _, G, K = knn_xyz.shape
        feat_dim = self.out_dim // (self.in_dim * 2)

        feat_range = torch.arange(feat_dim, device=knn_xyz.device, dtype=torch.float32)
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim)
        div_embed = torch.div(self.beta * knn_xyz.unsqueeze(-1), dim_embed)

        sin_embed = torch.sin(div_embed)
        cos_embed = torch.cos(div_embed)
        position_embed = torch.stack([sin_embed, cos_embed], dim=5).flatten(4)
        position_embed = position_embed.permute(0, 1, 4, 2, 3).reshape(B, self.out_dim, G, K)

        # Weigh
        knn_x_w = knn_x + position_embed
        knn_x_w *= position_embed

        return knn_x_w

# Local Geometry Aggregation
class LGA(nn.Module):
    def __init__(self, out_dim, alpha, beta):
        super().__init__()
        self.geo_extract = PosE_Geo(3, out_dim, alpha, beta)

    def forward(self, lc_xyz, lc_x, knn_xyz, knn_x):

        # Normalize x (features) and xyz (coordinates)
        mean_x = lc_x.unsqueeze(dim=-2)
        diff_x = knn_x - mean_x
        std_x = torch.std(diff_x, dim=2, keepdim=True, unbiased=False)

        mean_xyz = lc_xyz.unsqueeze(dim=-2)
        diff_xyz = knn_xyz - mean_xyz
        std_xyz = torch.std(diff_xyz, dim=2, keepdim=True, unbiased=False)

        knn_x = diff_x / (std_x + 1e-5)
        knn_xyz = diff_xyz / (std_xyz + 1e-5)

        # Feature Expansion
        B, G, K, C = knn_x.shape
        knn_x = torch.cat([knn_x, lc_x.reshape(B, G, 1, -1).repeat(1, 1, K, 1)], dim=-1)

        # Geometry Extraction
        knn_xyz = knn_xyz.permute(0, 3, 1, 2)
        knn_x = knn_x.permute(0, 3, 1, 2)
        knn_x_w = self.geo_extract(knn_xyz, knn_x)

        return knn_x_w


class TemporalDFT(nn.Module):
    """
    沿时间维 L 做离散傅里叶变换 (DFT) 的可学习时序模块：
      x --rfft--> X(f) --(learnable complex filter)--> Y(f) --irfft--> y
    设计目标：把“轨迹是有序序列”的归纳偏置放到 encoder 的 temporal head 里，
    避免把 N=K*L 当成无序点云直接做全局池化。
    """
    def __init__(self, channels: int, L: int, dropout: float = 0.0, residual: bool = True):
        super().__init__()
        self.channels = int(channels)
        self.L = int(L)
        self.residual = bool(residual)
        # rfft 的频点数
        self.F = self.L // 2 + 1

        # 逐通道、逐频点的复数权重：Y = X * W
        self.weight_real = nn.Parameter(torch.ones(self.channels, self.F))
        self.weight_imag = nn.Parameter(torch.zeros(self.channels, self.F))

        self.dropout = nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity()
        # BN1d 支持 [B, C, L]
        self.norm = nn.BatchNorm1d(self.channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, L]
        return: [B, C, L]
        """
        assert x.dim() == 3, f"TemporalDFT expects [B,C,L], got {tuple(x.shape)}"
        B, C, L = x.shape
        assert C == self.channels, f"channels mismatch: expect {self.channels}, got {C}"
        assert L == self.L, f"L mismatch: expect {self.L}, got {L}"

        X = torch.fft.rfft(x, dim=-1, norm='ortho')  # [B, C, F], complex
        W = torch.complex(self.weight_real, self.weight_imag).unsqueeze(0)  # [1, C, F]
        Y = X * W
        y = torch.fft.irfft(Y, n=self.L, dim=-1, norm='ortho')  # [B, C, L], real

        y = self.dropout(self.act(self.norm(y)))
        return (x + y) if self.residual else y



# Non-Parametric Encoder
class FMT(nn.Module):  
    def __init__(self, PathlineLtimesteps:int, num_stages, embed_dim, alpha, beta, temporal_head: str = "dft"):
        super().__init__()
        self.PathlineLtimesteps = PathlineLtimesteps
        self.num_stages = num_stages
        self.embed_dim = embed_dim
        self.alpha, self.beta = alpha, beta
        self.temporal_head = str(temporal_head).lower()


        # Raw-point Embedding
        self.raw_point_embed = PosE_Initial(3, self.embed_dim, self.alpha, self.beta)

        self.FPS_kNN_list = nn.ModuleList() # FPS, kNN
        self.LGA_list = nn.ModuleList() # Local Geometry Aggregation
        self.Pooling_list = nn.ModuleList() # Pooling
        
        out_dim = self.embed_dim

        # Multi-stage Hierarchy
        for i in range(self.num_stages):
            out_dim = out_dim * 2
            #disable FPS
            # group_num = group_num // 2
            self.FPS_kNN_list.append(GeoLinePicker(self.PathlineLtimesteps))
            self.LGA_list.append(LGA(out_dim, self.alpha, self.beta))
            self.Pooling_list.append(Pooling(out_dim))

        # temporal head：在最后把 N=K*L 还原为时序并做频域学习
        self.out_dim = out_dim
        if self.temporal_head == "dft":
            self.temporal_dft = TemporalDFT(channels=self.out_dim, L=self.PathlineLtimesteps, dropout=0.0, residual=True)
        else:
            self.temporal_dft = None


    def forward(self, xyz, x):
        # xyz: (B,  N=Lstep*Cross_n_neighbors, 3)
        #x: (B, 3, N)
        # Raw-point Embedding
        x = self.raw_point_embed(x)
        B, embed_dim, N = x.shape
        # Multi-stage Hierarchy
        for i in range(self.num_stages):
            # FPS, kNN
            xyz, lc_x, knn_xyz, knn_x = self.FPS_kNN_list[i](xyz, x.permute(0, 2, 1))
            # Local Geometry Aggregation
            knn_x_w = self.LGA_list[i](xyz, lc_x, knn_xyz, knn_x)
            # Pooling
            x = self.Pooling_list[i](knn_x_w)

        # x: [B, out_dim, N]
        x = x.view(B, -1, N)

        # Temporal head (DFT along L): N must be K*L
        if self.temporal_dft is not None:
            L = int(self.PathlineLtimesteps)
            assert N % L == 0, f"N must be divisible by L={L}, got N={N}"
            K = N // L
            # 还原为 [B, C, K, L]
            x_ckl = x.reshape(B, self.out_dim, K, L)
            # 逐线做 DFT，再跨线聚合得到时序 token: [B, C, L]
            x_line = x_ckl.permute(0, 2, 1, 3).reshape(B * K, self.out_dim, L)  # [B*K, C, L]
            x_line = self.temporal_dft(x_line)  # [B*K, C, L]
            x_ckl = x_line.reshape(B, K, self.out_dim, L).permute(0, 2, 1, 3)  # [B, C, K, L]
            x_cl = x_ckl.max(dim=2)[0] + x_ckl.mean(dim=2)  # [B, C, L]
            # time pooling -> token
            every_cross_feature = x_cl.max(dim=-1)[0] + x_cl.mean(dim=-1)  # [B, C]
            return every_cross_feature

        # Fallback: old global pooling (treat as unordered points)
        every_cross_feature = x.max(-1)[0] + x.mean(-1)
        return every_cross_feature






class HierachyFMT_encoder(nn.Module):  
    def __init__(self, ReceptiveFieldList:list[int], base_num_stages:int, embed_dim:int, PathlineLtimesteps:int, alpha:float,  beta:float, temporal_head: str = "dft"):
        super().__init__()
        """
        ReceptiveFieldList: 感受野窗口边长列表（单位：低分辨率网格格点数），例如 [4, 8, 16]
        每个感受野对应一个 FMT 编码器：对该窗口内所有 cross-primitive 的点集进行编码，得到一个 token 向量，
        将这些 token 填充到 coarse 特征图 [B, D_i, Hc_i, Wc_i]，再双线性插值到低分辨率大小 [B, D_i, X, Y]，最后在通道维拼接。

        base_num_stages: FMT 的基础层数，实际层数 = base_num_stages + floor(log2(receptive_field))，并裁剪到 [0,4]
        embed_dim: FMT 的基础通道数
        k_neighbors, alpha, beta: FMT 内部使用的 KNN 和位置编码超参
        """
        assert isinstance(ReceptiveFieldList, (list, tuple)) and len(ReceptiveFieldList) > 0
        self.receptive_fields = [int(max(1, k)) for k in ReceptiveFieldList]
        self.FMT_num_stages = [min(4, max(0, int(base_num_stages +  math.floor(math.log2(k))))) for k in self.receptive_fields]

        # self.k_neighbors = int(k_neighbors)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.embed_dim = int(embed_dim)
        self.PathlineLtimesteps = int(PathlineLtimesteps)
        self.temporal_head = str(temporal_head).lower()

        # 为每个感受野构建一个 FMT 编码器
        self.fmts = nn.ModuleList()
        self.out_dims: list[int] = []
        for stages in self.FMT_num_stages:
            # FMT 的输出维度约为 embed_dim * (2**stages)
            out_dim = self.embed_dim * (2 ** max(0, stages))
            self.out_dims.append(out_dim)
            self.fmts.append(FMT(PathlineLtimesteps=self.PathlineLtimesteps, num_stages=stages, embed_dim=self.embed_dim, alpha=self.alpha, beta=self.beta, temporal_head=self.temporal_head))

    def _normalize_input(self, pathlines: torch.Tensor):
        """
        接受以下形状之一并归一化为 [B, Y,  X,   K, L, 3], WHEN X and Y not not given, X=Y=int(math.isqrt(X*Y)):
          - [X*Y, K, L, 3]
          - [B, X*Y, K, L, 3]
          - [B, Y, X, K, L, 3]
        """
        Dim = pathlines.shape[-1]
        assert Dim == 3, f"last dim must be 3, got {Dim}"
        if pathlines.dim() == 4:  # [X*Y, K, L, 3]
            N, K, L, _ = pathlines.shape
            X = int(math.isqrt(N))
            assert X * X == N, f"Cannot infer square grid from N={N}. Provide [B,X,Y,...] shape."
            Y = X
            B = 1
            pl = pathlines.unsqueeze(0).reshape(B, Y,X, K, L, 3)
            return pl
        if pathlines.dim() == 5:  # [B, N, K, L, 3]
            B, N, K, L, _ = pathlines.shape
            X = int(math.isqrt(N))
            assert X * X == N, f"Cannot infer square grid from N={N}. Provide [B,X,Y,...] shape."
            Y = X
            pl = pathlines.reshape(B, Y, X, K, L, 3)
            return pl
        if pathlines.dim() == 6 :  #[B, X, Y, K, L, 3]
            return pathlines
        raise ValueError(f"Unsupported pathlines shape: {tuple(pathlines.shape)}")

    def forward(self, pathlines: torch.Tensor):
        """
        输入 pathlines:
          - [B, X*Y, K, L, 3] 或 [X*Y, K, L, 3] 或 [B, Y, X, K, L, 3]
        输出:
          - multi-scale 特征图拼接: [B, sum(D_i), X, Y]
        """
        pl = self._normalize_input(pathlines)
        B, Y, X, K, Ls, Dim = pl.shape
        device = pl.device
        outputs: list[torch.Tensor] = []

        for fmt, receptive_field, out_dim in zip(self.fmts, self.receptive_fields, self.out_dims):
            row_starts = _tiling_starts(int(Y), receptive_field)
            col_starts = _tiling_starts(int(X), receptive_field)
            Hc, Wc = len(row_starts), len(col_starts)
            feat_coarse = torch.zeros((B, out_dim, Hc, Wc), device=device, dtype=pl.dtype)

            for ri, i0 in enumerate(row_starts):
                i1 = min(i0 + receptive_field, int(Y))
                for ci, j0 in enumerate(col_starts):
                    j1 = min(j0 + receptive_field, int(X))
                    # 收集窗口内的线性索引
                    idx_list = []
                    for rr in range(i0, i1):
                        base = rr * int(X)
                        idx_list.extend(range(base + j0, base + j1))
                    if len(idx_list) == 0:
                        continue
                    idx_tensor = torch.as_tensor(idx_list, dtype=torch.long, device=device)
                    # 选择窗口内的 pathlines: [B, M, K, L, 3]
                    pl_win = pl.reshape(B, Y * X, K, Ls, Dim)[:, idx_tensor, ...]
                    B2, M, _, _, _ = pl_win.shape
                    P = pl_win.reshape(B2, M * K * Ls, Dim).contiguous()  # [B, M*K*L, 3]
                    points_N3 = P
                    points_3N = P.permute(0, 2, 1).contiguous()
                    feat_win = fmt(points_N3, points_3N)  # [B, out_dim]
                    feat_coarse[:, :, ri, ci] = feat_win

            # 上采样到 [X, Y]
            feat_map = F.interpolate(feat_coarse, size=(int(Y), int(X)), mode='bilinear', align_corners=False)
            outputs.append(feat_map)

        # 通道拼接
        fused = torch.cat(outputs, dim=1) if len(outputs) > 0 else torch.zeros((B, 0, Y, X), device=device, dtype=pl.dtype)
        return fused


