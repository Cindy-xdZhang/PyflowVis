import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pnn.models.point_nn import EncNPNew
from FMT_Utils.DCT_FMT_encoder import DCT_FMT


def calculate_model_parm_size(model: nn.Module):
    """
    Summarize parameter/buffer counts and memory footprint (MB) of a PyTorch model.
    Returns a dict: param_count, trainable_param_count, buffer_count,
                    param_size_mb, buffer_size_mb, total_size_mb.
    """
    with torch.no_grad():
        param_count = sum(p.numel() for p in model.parameters())
        trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        buffer_count = sum(b.numel() for b in model.buffers())

        param_bytes = sum(p.numel() * (p.element_size() if p.is_floating_point() or p.dtype is not None else 4) for p in model.parameters())
        buffer_bytes = sum(b.numel() * (b.element_size() if b.is_floating_point() or b.dtype is not None else 4) for b in model.buffers())
        total_bytes = param_bytes + buffer_bytes

        mb = 1024.0 * 1024.0
        return {
            "param_count": int(param_count),
            "trainable_param_count": int(trainable_param_count),
            "buffer_count": int(buffer_count),
            "param_size_mb": float(param_bytes / mb),
            "buffer_size_mb": float(buffer_bytes / mb),
            "total_size_mb": float(total_bytes / mb),
        }

class Decoder(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.ln1=nn.Linear(in_dim, 256)
        self.bn1=nn.BatchNorm1d(256)
        self.ln2=nn.Linear(256, 256)
        self.bn2=nn.BatchNorm1d(256)
        self.ln3=nn.Linear(256, 64)
        self.bn3=nn.BatchNorm1d(64)
        self.ln4=nn.Linear(64, 1)

    def forward(self, x):
        x = x.contiguous()
        x = self.ln1(x.contiguous())
        x = self.bn1(x)
        x = nn.GELU()(x)


        res=x
        x = self.ln2(x)
        x = self.bn2(x)
        x = nn.GELU()(x)
        x = x + res


        x = self.ln3(x)
        x = self.bn3(x)
        x = nn.GELU()(x)

        x = self.ln4(x)
        x = nn.Sigmoid()(x)
        return x.squeeze(-1)
    
class PointWiseFMT_Regressor(nn.Module):
    def __init__(self, LStpesPerline,num_stages=2, embed_dim=72, k_neighbors=6, cross_neighborsize=5,beta=100, alpha=1000):
        super().__init__()
        #FTLERegressor is a point wise regressor, has no cross primitive, no neighbor information aggregation
        self.cross_neighborsize=cross_neighborsize
        self.pointsPerPrimitive=LStpesPerline*cross_neighborsize
        self.encoder = EncNPNew(self.pointsPerPrimitive, num_stages, embed_dim, k_neighbors, alpha, beta)
        self.decoderInputDim=embed_dim * (2 ** (num_stages - 0))
        self.decoder = Decoder(in_dim=self.decoderInputDim)

    def forward(self, pts: torch.Tensor):
        B,CrossSize,LstepsPerline,Dim=pts.shape
        PointsRaw=pts.reshape(B,CrossSize*LstepsPerline,Dim)
        points_3N=PointsRaw.permute(0, 2, 1)
        points_N3=PointsRaw
        FMT_feat = self.encoder(points_N3, points_3N)
        #feature dimension is in_dim*K(steps per line)
        pred = self.decoder(FMT_feat)
        return pred

class PointWiseMLP_Regressor(nn.Module):
    def __init__(self, LStpesPerline,num_stages=2, embed_dim=72, k_neighbors=6, cross_neighborsize=5,beta=100, alpha=1000):
        super().__init__()
        #FTLERegressor is a point wise regressor, has no cross primitive, no neighbor information aggregation
        self.cross_neighborsize=cross_neighborsize
        self.pointsPerPrimitive=LStpesPerline*cross_neighborsize
        self.decoderInputDim=3*2*cross_neighborsize
        self.decoder = Decoder(in_dim=self.decoderInputDim)

    def forward(self, pts: torch.Tensor):
        B,CrossSize,LstepsPerline,Dim=pts.shape
        # xyz = pts.permute(0, 2, 1)#B,N,3
        #feat: (B, embed_dim)
        start_pos=pts[:,:,0,:].reshape(B, -1)
        end_pos=pts[:,:,-1,:].reshape(B, -1)
        every_cross_feature=torch.cat([start_pos,end_pos],dim=1)
        pred = self.decoder(every_cross_feature)
        return pred
 



class FTLEUpsamplingFMT_Unet(nn.Module):
    """
    Encode each grid cell's cross-primitive (5 pathlines, L points, Dim=3) into a 32-D feature via EncNPNew,
    concatenate with low-resolution FTLE and coordinate channels (X_norm, Y_norm) to form (32 + 1) channels,
    then upsample to high-resolution FTLE with a UNet.

    Inputs:
      - lowResFTLE:      [B, X, Y]
      - lowResPathlines: [B, X*Y, 5, L, 3]
    Output:
      - pred:            [B, X*UP, Y*UP]
    """
    def __init__(self, cfg, lowResX: int, lowResY: int, upscale: int = 4,
                 embed_dim: int = 36):
        super().__init__()
        self.lowResX = int(lowResX)
        self.lowResY = int(lowResY)
        self.upscale = int(upscale)
        self.embed_dim = int(embed_dim)  # target feature dimension (prefer multiples of 6)

        # EncNPNew encodes cross-primitive: points_N3 [B, N, 3], points_3N [B, 3, N]
        # Output dim ≈ embed_dim * (2**stages). To keep 32-D, enforce stages=0.
        stages = int(getattr(cfg.pnn, 'stages', 0)) if hasattr(cfg, 'pnn') else 0
        k = int(getattr(cfg.pnn, 'k', 6)) if hasattr(cfg, 'pnn') else 6
        alpha = float(getattr(cfg.pnn, 'alpha', 1000)) if hasattr(cfg, 'pnn') else 1000.0
        beta = float(getattr(cfg.pnn, 'beta', 100)) if hasattr(cfg, 'pnn') else 100.0
        nerbors = int(getattr(cfg.pcds, 'num_cross_points_per_seeding', 5)) if hasattr(cfg, 'pcds') else 5
        LstepsPerline = int(getattr(cfg.pcds, 'sampled_points_per_line', 4)) if hasattr(cfg, 'pcds') else 4
        self.cross_neighborsize = nerbors
        self.pointsPerPrimitive = LstepsPerline * nerbors
        # Enforce stages=0 to keep feature dimension fixed
        self.encoder = EncNPNew(self.pointsPerPrimitive, 0, self.embed_dim, k, alpha, beta)

        in_channels = self.embed_dim + 3  # self.embed_dim + (FTLE 1ch + XY 2ch)
        base_ch = in_channels
        self.inc = DoubleConv(in_channels, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.up1 = Up(base_ch * 4, base_ch * 2)
        self.up2 = Up(base_ch * 2, base_ch)
        self.out_low = nn.Conv2d(base_ch, base_ch, kernel_size=1)

        n_up = max(0, int(round(math.log2(max(1, self.upscale)))))
        self.up_blocks = nn.ModuleList()
        in_ch = base_ch
        for i in range(n_up):
            out_ch = base_ch if i < n_up - 1 else base_ch // 2
            self.up_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            )
            in_ch = out_ch
        self.out_high = nn.Conv2d(in_ch, 1, kernel_size=1)


        
        self.cache_coordGrid=None

    def construct_cache_coordGrid(self,B: int,lowResX: int,lowResY: int):
        yy = torch.linspace(0, 1, steps=lowResX)
        xx = torch.linspace(0, 1, steps=lowResY)
        Y_grid, X_grid = torch.meshgrid(yy, xx, indexing='ij')  # [X,Y]
        coord = torch.stack([X_grid, Y_grid], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)  # [B,2,X,Y]
        self.cache_coordGrid=coord

    def forward(self, lowResFTLE: torch.Tensor, lowResPathlines: torch.Tensor) -> torch.Tensor:
        B, X, Y = lowResFTLE.shape
        _, N, nerbors, L, Dim = lowResPathlines.shape
        assert N == X * Y, "lowResPathlines second dim must be X*Y"
        assert nerbors == self.cross_neighborsize, "nerbors mismatch with model setting"
        if self.cache_coordGrid is None or self.cache_coordGrid.shape[0] != B or self.cache_coordGrid.shape[2] != X or self.cache_coordGrid.shape[3] != Y:
            self.construct_cache_coordGrid(B, X, Y)

        # 1) Encode cross-primitive to per-cell 32-D feature
        P = lowResPathlines.reshape(B * N, nerbors * L, Dim).contiguous()
        points_N3 = P  # [B*N, K, 3]
        points_3N = P.permute(0, 2, 1).contiguous()  # [B*N, 3, K]
        feat = self.encoder(points_N3, points_3N)  # [B*N, 32]
        feat = feat.reshape(B, X, Y, self.embed_dim).permute(0, 3, 1, 2).contiguous()  # [B,32,X,Y]

        # 2) Coordinate channels normalized to [0,1]
        coord = self.cache_coordGrid.to(lowResFTLE.device)

        # 3) Concatenate input channels (1 + 2 + 32)
        ftle_in = lowResFTLE.unsqueeze(1)
        x_in = torch.cat([ftle_in, coord, feat], dim=1)  # [B, 35, X, Y]

        # 4) UNet encode-decode + upsampling head
        x1 = self.inc(x_in)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.out_low(x)
        for blk in self.up_blocks:
            x = blk(x)
        pred = self.out_high(x).squeeze(1)  # [B, X*UP, Y*UP]

        # Align to target spatial size
        target_h = int(X * max(1, self.upscale))
        target_w = int(Y * max(1, self.upscale))
        if pred.shape[-2] != target_h or pred.shape[-1] != target_w:
            pred = F.interpolate(pred.unsqueeze(1), size=(target_h, target_w), mode='bilinear', align_corners=False).squeeze(1)
        return pred

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # pad if needed to handle odd sizes
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        if diffY != 0 or diffX != 0:
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C]
        h = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = h + attn_out

        h2 = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = h2 + x
        return x



class UpsamplingUnetModel(nn.Module):
    """
    UNet for upsampling FTLE from lowRes to highRes
    """
    def __init__(self, cfg, lowResX: int, lowResY: int, upscale:float,base_ch: int = 24):
        super().__init__()
        self.upscale = int(upscale)
        self.inc = DoubleConv(1, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.up1 = Up(base_ch * 4, base_ch * 2)
        self.up2 = Up(base_ch * 2, base_ch)

        # Low-resolution feature output (same spatial size as input)
        self.out_low = nn.Conv2d(base_ch, base_ch, kernel_size=1)

        # Progressive upsampling head (constructed by powers of 2)
        import math
        n_up = max(0, int(round(math.log2(max(1, self.upscale)))))
        self.up_blocks = nn.ModuleList()
        in_ch = base_ch
        for i in range(n_up):
            out_ch = base_ch if i < n_up - 1 else base_ch // 2
            self.up_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            )
            in_ch = out_ch
        self.out_high = nn.Conv2d(in_ch, 1, kernel_size=1)

    def forward(self, lowResFTLE: torch.Tensor, lowResPathlines: torch.Tensor | None = None) -> torch.Tensor:
        B, X, Y = lowResFTLE.shape
        x = lowResFTLE.unsqueeze(1).contiguous()  # [B,1,X,Y]

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)  # [B, base_ch, X, Y]
        x = self.out_low(x)   # [B, base_ch, X, Y]

        for blk in self.up_blocks:
            x = blk(x)

        pred = self.out_high(x).squeeze(1)  # [B, X*UP, Y*UP]

        # Align to expected size (avoid 1-pixel misalignment due to odd/even sizes)
        target_h = int(X * max(1, self.upscale))
        target_w = int(Y * max(1, self.upscale))
        if pred.shape[-2] != target_h or pred.shape[-1] != target_w:
            pred = F.interpolate(pred.unsqueeze(1), size=(target_h, target_w), mode='bilinear', align_corners=False).squeeze(1)
        return pred


class UpsamplingUnetModelV2(nn.Module):
    """
    Super-resolution-oriented UNet variant of UpsamplingUnetModel:
      - higher channel width (base_ch=64 vs 24)
      - ONE FEWER downsampling stage (a single 2x down instead of two)

    Rationale: pixel super-resolution needs high-frequency detail, which a deep
    encoder-decoder discards when it downsamples 32->16->8. Keeping a shallower
    (32->16) bottleneck with more channels preserves that detail. Tests whether a
    properly-sized UNet can beat the flat ESPCN CNN on this task.

    Inputs:  lowResFTLE [B, X, Y]   (pathlines ignored)
    Output:  pred       [B, X*UP, Y*UP]
    """
    def __init__(self, cfg, lowResX: int, lowResY: int, upscale: float, base_ch: int = 64):
        super().__init__()
        self.upscale = int(upscale)
        self.inc = DoubleConv(1, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)        # single downsample: X -> X/2
        self.up1 = Up(base_ch * 2, base_ch)            # single upsample back to X
        self.out_low = nn.Conv2d(base_ch, base_ch, kernel_size=1)

        n_up = max(0, int(round(math.log2(max(1, self.upscale)))))
        self.up_blocks = nn.ModuleList()
        in_ch = base_ch
        for i in range(n_up):
            out_ch = base_ch if i < n_up - 1 else base_ch // 2
            self.up_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            )
            in_ch = out_ch
        self.out_high = nn.Conv2d(in_ch, 1, kernel_size=1)

    def forward(self, lowResFTLE: torch.Tensor, lowResPathlines: torch.Tensor | None = None) -> torch.Tensor:
        B, X, Y = lowResFTLE.shape
        x = lowResFTLE.unsqueeze(1).contiguous()  # [B,1,X,Y]

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x = self.up1(x2, x1)   # [B, base_ch, X, Y]
        x = self.out_low(x)

        for blk in self.up_blocks:
            x = blk(x)
        pred = self.out_high(x).squeeze(1)  # [B, X*UP, Y*UP]

        target_h = int(X * max(1, self.upscale))
        target_w = int(Y * max(1, self.upscale))
        if pred.shape[-2] != target_h or pred.shape[-1] != target_w:
            pred = F.interpolate(pred.unsqueeze(1), size=(target_h, target_w), mode='bilinear', align_corners=False).squeeze(1)
        return pred


class AttentionFusion(nn.Module):
    """
    融合来自 pathline 的特征图 F ∈ [B, D, X, Y] 与低分辨率 FTLE 的单通道映射 L ∈ [B, 1, X, Y]。
    - 首先将 L 通过 1x1 卷积映射到 D 维，与 F 对齐；
    - 计算通道维度上的注意力权重：
        a = sigmoid(Conv1x1([F, L_proj])) ∈ [B, D, X, Y]
      最终输出：a · F + (1-a) · L_proj
    该设计能在逐空间位置和逐通道上自适应选择来自 pathline 的信息或来自低分辨率 FTLE 的先验。
    """
    def __init__(self, channels: int):
        super().__init__()
        self.to_d_from_lr = nn.Conv2d(1, channels, kernel_size=1, bias=True)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, feat_from_pathline: torch.Tensor, lowres_ftle_1ch: torch.Tensor) -> torch.Tensor:
        # feat_from_pathline: [B, D, X, Y]
        # lowres_ftle_1ch:    [B, 1, X, Y]
        Lp = self.to_d_from_lr(lowres_ftle_1ch)
        gate_in = torch.cat([feat_from_pathline, Lp], dim=1)
        a = self.gate(gate_in)
        return a * feat_from_pathline + (1.0 - a) * Lp


class FTLEupsamplingFMT_UnetV3(nn.Module):
    """
    基于 FTLEupsamplingFMT_UnetV2 的滑窗 FMT 特征提取流程，但将 "特征与低分辨率 FTLE 的简单拼接"
    替换为 "注意力融合"（AttentionFusion）。

    Pipeline:
      1) 滑窗聚合局部 pathline，EncNPNew → coarse FMT 特征图 [B, D, Hc, Wc]
      2) 双线性上采样到 [B, D, X, Y]
      3) 与低分辨率 FTLE 的 1 通道图 [B,1,X,Y] 通过 AttentionFusion 融合 → [B, D, X, Y]
      4) 以 D 通道作为 UNet 编码器输入，随后与 V2 相同的上采样头输出高分辨率预测
    """
    def __init__(self, cfg, lowResX: int, lowResY: int, upscale: float, base_ch: int = 32,
                 embed_dim: int | None = None):
        super().__init__()
        self.lowResX = int(lowResX)
        self.lowResY = int(lowResY)
        self.upscale = int(upscale)

        # FMT 编码器（滑窗）
        self.FMT_focus_area = int(getattr(cfg, 'FMT_focus_area', 8))
        num_stages = int(getattr(cfg.pnn, 'stages', 1)) if hasattr(cfg, 'pnn') else 1
        k = int(getattr(cfg.pnn, 'k', 6)) if hasattr(cfg, 'pnn') else 6
        alpha = float(getattr(cfg.pnn, 'alpha', 1000)) if hasattr(cfg, 'pnn') else 1000.0
        beta = float(getattr(cfg.pnn, 'beta', 100)) if hasattr(cfg, 'pnn') else 100.0
        nerbors = int(getattr(cfg.pcds, 'num_cross_points_per_seeding', 5)) if hasattr(cfg, 'pcds') else 5
        LstepsPerline = int(getattr(cfg.pcds, 'sampled_points_per_line', 4)) if hasattr(cfg, 'pcds') else 4
        self.cross_neighborsize = nerbors
        self.pointsPerPrimitive = LstepsPerline * nerbors
        self.embed_dim = int(embed_dim if embed_dim is not None else getattr(cfg.pnn, 'dim', 36))
        self.encoder = EncNPNew(self.pointsPerPrimitive, num_stages, self.embed_dim, k, alpha, beta)
        self.fmt_feature_dim = self.embed_dim * (2 ** (num_stages - 0))

        # 注意力融合：将 [B,D,X,Y] 与 [B,1,X,Y] 融合到 [B,D,X,Y]
        self.fuse = AttentionFusion(self.fmt_feature_dim)

        # Unet 主干：输入通道使用融合后的 D 通道（不再拼接 1 通道 FTLE）
        in_channels = self.fmt_feature_dim
        self.inc = DoubleConv(in_channels, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.up1 = Up(base_ch * 4, base_ch * 2)
        self.up2 = Up(base_ch * 2, base_ch)
        self.out_low = nn.Conv2d(base_ch, base_ch, kernel_size=1)

        n_up = max(0, int(round(math.log2(max(1, self.upscale)))))
        self.up_blocks = nn.ModuleList()
        in_ch = base_ch
        for i in range(n_up):
            out_ch = base_ch if i < n_up - 1 else base_ch // 2
            self.up_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            )
            in_ch = out_ch
        self.out_high = nn.Conv2d(in_ch, 1, kernel_size=1)

    def _tiling_starts(self, length: int, k: int):
        if k >= length:
            return [0]
        s = int(k)
        starts = list(range(0, length - k + 1, s))
        last = length - k
        if starts[-1] != last:
            starts.append(last)
        return starts

    def forward(self, lowResFTLE: torch.Tensor, lowResPathlines: torch.Tensor) -> torch.Tensor:
        B, X, Y = lowResFTLE.shape
        _, N, nerbors, L, Dim = lowResPathlines.shape
        assert N == X * Y, "lowResPathlines second dim must be X*Y"
        assert nerbors == self.cross_neighborsize, "nerbors mismatch with model setting"

        # 1) 滑窗局部 FMT → coarse feature map [B, D, Hc, Wc]
        k = int(self.FMT_focus_area)
        row_starts = self._tiling_starts(int(X), k)
        col_starts = self._tiling_starts(int(Y), k)
        Hc, Wc = len(row_starts), len(col_starts)

        feat_coarse = lowResFTLE.new_zeros((B, self.fmt_feature_dim, Hc, Wc))
        for ri, i0 in enumerate(row_starts):
            i1 = min(i0 + k, int(X))
            for ci, j0 in enumerate(col_starts):
                j1 = min(j0 + k, int(Y))
                idx_list = []
                for rr in range(i0, i1):
                    base = rr * int(Y)
                    idx_list.extend(range(base + j0, base + j1))
                if len(idx_list) == 0:
                    continue
                idx_tensor = torch.as_tensor(idx_list, dtype=torch.long, device=lowResFTLE.device)
                pl_win = lowResPathlines[:, idx_tensor, ...]  # [B, M, nerbors, L, Dim]
                B2, M, _, _, _ = pl_win.shape
                P = pl_win.reshape(B2, M * nerbors * L, Dim).contiguous()
                points_N3 = P
                points_3N = P.permute(0, 2, 1).contiguous()
                feat_win = self.encoder(points_N3, points_3N)  # [B, D]
                feat_coarse[:, :, ri, ci] = feat_win

        # 2) 上采样到低分辨率大小 [B, D, X, Y]
        feat_map = F.interpolate(feat_coarse, size=(int(X), int(Y)), mode='bilinear', align_corners=False)

        # 3) 注意力融合 FMT 特征与 1 通道低分辨率 FTLE
        ftle_in = lowResFTLE.unsqueeze(1)  # [B,1,X,Y]
        fused = self.fuse(feat_map, ftle_in)  # [B,D,X,Y]

        # 4) UNet 编码-解码 + 渐进上采样
        x1 = self.inc(fused)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.out_low(x)
        for blk in self.up_blocks:
            x = blk(x)
        pred = self.out_high(x).squeeze(1)  # [B, X*UP, Y*UP]

        # 尺寸对齐
        target_h = int(X * max(1, self.upscale))
        target_w = int(Y * max(1, self.upscale))
        if pred.shape[-2] != target_h or pred.shape[-1] != target_w:
            pred = F.interpolate(pred.unsqueeze(1), size=(target_h, target_w), mode='bilinear', align_corners=False).squeeze(1)
        return pred



class FTLEupsamplingFMT_UnetV2(nn.Module):
    """
    Local sliding-window FMT features + low-resolution FTLE → UNet upsampling

    - Use a 2D sliding window (window size = self.FMT_focus_area, e.g., 8x8; stride equals window)
      on the low-resolution grid to extract local patches
    - Within each patch, concatenate all cross-primitives into a point cloud and feed into EncNPNew
      to obtain a local feature vector f ∈ R^D
    - Arrange the local features into a coarse feature map [B, D, Hc, Wc] (Hc/Wc are window centers)
    - Bilinearly upsample this feature map to [B, D, X, Y], then concatenate with the 1-channel
      low-resolution FTLE → [B, D+1, X, Y]
    - Apply UNet encode-decode, followed by transposed-convolution upsampling to high resolution

    Inputs:
      lowResFTLE:      [B, X, Y]
      lowResPathlines: [B, X*Y, 5, L, 3]
    Output:
      pred:            [B, X*UP, Y*UP]
    """
    def __init__(self, cfg, lowResX: int, lowResY: int, upscale: float, base_ch: int = 32,
                 embed_dim: int | None = None):
        super().__init__()
        self.lowResX = int(lowResX)
        self.lowResY = int(lowResY)
        self.upscale = int(upscale)

        # FMT encoder (global): use small num_stages; output dimension fixed by embed_dim
        self.FMT_focus_area=8# if this is too large, too many points will make knn too slow, self.FMT_focus_area should divide 64.
        num_stages= int(getattr(cfg.pnn, 'stages', 1)) if hasattr(cfg, 'pnn') else 1
        k = int(getattr(cfg.pnn, 'k', 6)) if hasattr(cfg, 'pnn') else 6
        alpha = float(getattr(cfg.pnn, 'alpha', 1000)) if hasattr(cfg, 'pnn') else 1000.0
        beta = float(getattr(cfg.pnn, 'beta', 100)) if hasattr(cfg, 'pnn') else 100.0
        nerbors = int(getattr(cfg.pcds, 'num_cross_points_per_seeding', 5)) if hasattr(cfg, 'pcds') else 5
        LstepsPerline = int(getattr(cfg.pcds, 'sampled_points_per_line', 4)) if hasattr(cfg, 'pcds') else 4
        self.cross_neighborsize = nerbors
        self.pointsPerPrimitive = LstepsPerline * nerbors
        self.embed_dim = int(embed_dim if embed_dim is not None else getattr(cfg.pnn, 'dim', 36))

        self.encoder = EncNPNew(self.pointsPerPrimitive, num_stages, self.embed_dim, k, alpha, beta)

        self.fmt_feature_dim= self.embed_dim* (2 ** (num_stages - 0))

        in_channels = self.fmt_feature_dim+ 1  # Global D channels + 1 channel of low-res FTLE
        self.inc = DoubleConv(in_channels, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.up1 = Up(base_ch * 4, base_ch * 2)
        self.up2 = Up(base_ch * 2, base_ch)
        self.out_low = nn.Conv2d(base_ch, base_ch, kernel_size=1)

        n_up = max(0, int(round(math.log2(max(1, self.upscale)))))
        self.up_blocks = nn.ModuleList()
        in_ch = base_ch
        for i in range(n_up):
            out_ch = base_ch if i < n_up - 1 else base_ch // 2
            self.up_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            )
            in_ch = out_ch
        self.out_high = nn.Conv2d(in_ch, 1, kernel_size=1)

    def _tiling_starts(self, length: int, k: int):
        if k >= length:
            return [0]
        s = int(k)
        starts = list(range(0, length - k + 1, s))
        last = length - k
        if starts[-1] != last:
            starts.append(last)
        return starts

    def forward(self, lowResFTLE: torch.Tensor, lowResPathlines: torch.Tensor) -> torch.Tensor:
        B, X, Y = lowResFTLE.shape
        _, N, nerbors, L, Dim = lowResPathlines.shape
        assert N == X * Y, "lowResPathlines second dim must be X*Y"
        assert nerbors == self.cross_neighborsize, "nerbors mismatch with model setting"

        # 1) Extract local point-cloud features via sliding window to get coarse feature map [B, D, Hc, Wc]
        k = int(self.FMT_focus_area)
        row_starts = self._tiling_starts(int(X), k)
        col_starts = self._tiling_starts(int(Y), k)
        Hc, Wc = len(row_starts), len(col_starts)

        feat_coarse = lowResFTLE.new_zeros((B, self.fmt_feature_dim, Hc, Wc))
        for ri, i0 in enumerate(row_starts):
            i1 = min(i0 + k, int(X))
            for ci, j0 in enumerate(col_starts):
                j1 = min(j0 + k, int(Y))
                # Collect linear indices within this window
                idx_list = []
                for rr in range(i0, i1):
                    base = rr * int(Y)
                    idx_list.extend(range(base + j0, base + j1))
                if len(idx_list) == 0:
                    continue
                idx_tensor = torch.as_tensor(idx_list, dtype=torch.long, device=lowResFTLE.device)
                # Select pathlines within the window and build the point cloud
                pl_win = lowResPathlines[:, idx_tensor, ...]  # [B, M, nerbors, L, Dim]
                B2, M, _, _, _ = pl_win.shape
                P = pl_win.reshape(B2, M * nerbors * L, Dim).contiguous()  # [B, M*K, 3]
                points_N3 = P
                points_3N = P.permute(0, 2, 1).contiguous()
                feat_win = self.encoder(points_N3, points_3N)  # [B, D]
                feat_coarse[:, :, ri, ci] = feat_win

        # 2) Bilinear upsample the coarse feature map to [X, Y]
        feat_map = F.interpolate(feat_coarse, size=(int(X), int(Y)), mode='bilinear', align_corners=False)
        # 3) Concatenate with low-resolution FTLE
        ftle_in = lowResFTLE.unsqueeze(1)  # [B, 1, X, Y]
        x_in = torch.cat([ftle_in, feat_map], dim=1)  # [B, D+1, X, Y]

        # UNet encode-decode + upsampling head
        x1 = self.inc(x_in)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.out_low(x)
        for blk in self.up_blocks:
            x = blk(x)
        pred = self.out_high(x).squeeze(1)  # [B, X*UP, Y*UP]

        # Size alignment
        target_h = int(X * max(1, self.upscale))
        target_w = int(Y * max(1, self.upscale))
        if pred.shape[-2] != target_h or pred.shape[-1] != target_w:
            pred = F.interpolate(pred.unsqueeze(1), size=(target_h, target_w), mode='bilinear', align_corners=False).squeeze(1)
        return pred


class FTLEupsamplingDCT_FMT_UnetV2(nn.Module):
    """
    Twin of :class:`FTLEupsamplingFMT_UnetV2`, with the FMT point-cloud encoder
    (EncNPNew: KNN + PosE + pooling) replaced by the training-free, Fourier-based
    :class:`~FMT_Utils.DCT_FMT_encoder.DCT_FMT` tokenizer.

    Everything else is identical to V2 so the two models form a fair A/B test of
    the *tokenizer* only (same sliding-window mechanism, same UNet backend, same
    upsampling head):
      1) sliding window (k = self.FMT_focus_area, stride = k) over the low-res grid
      2) DCT_FMT encodes each window's *structured* pathlines -> token [B, D]
         (note: pathlines are kept as [B, M, K, L, 3]; NOT flattened to a cloud)
      3) coarse feature map [B, D, Hc, Wc] -> bilinear upsample to [B, D, X, Y]
      4) concat with the 1-channel low-res FTLE -> UNet -> transposed-conv upsample

    Inputs:
      lowResFTLE:      [B, X, Y]
      lowResPathlines: [B, X*Y, nerbors, L, 3]
    Output:
      pred:            [B, X*UP, Y*UP]
    """
    def __init__(self, cfg, lowResX: int, lowResY: int, upscale: float, base_ch: int = 32):
        super().__init__()
        self.lowResX = int(lowResX)
        self.lowResY = int(lowResY)
        self.upscale = int(upscale)

        # Sliding-window size (same default/intent as V2).
        self.FMT_focus_area = int(getattr(cfg, 'FMT_focus_area', 8))
        nerbors = int(getattr(cfg.pcds, 'num_cross_points_per_seeding', 5)) if hasattr(cfg, 'pcds') else 5
        LstepsPerline = int(getattr(cfg.pcds, 'sampled_points_per_line', 4)) if hasattr(cfg, 'pcds') else 4
        self.cross_neighborsize = nerbors

        # DCT_FMT hyper-parameters (optional `dct:` config block; sensible defaults otherwise).
        dct_cfg = getattr(cfg, 'dct', None)
        dct_k = int(getattr(dct_cfg, 'k', 6)) if dct_cfg is not None else 6
        dct_weight = float(getattr(dct_cfg, 'weight', 0.5)) if dct_cfg is not None else 0.5
        neighbor_diff_scale = float(getattr(dct_cfg, 'neighbor_diff_scale', 100.0)) if dct_cfg is not None else 100.0

        self.encoder = DCT_FMT(nerbors=nerbors, L=LstepsPerline, dct_k=dct_k,
                               dct_weight=dct_weight, neighbor_diff_scale=neighbor_diff_scale)
        self.fmt_feature_dim = self.encoder.out_dim

        in_channels = self.fmt_feature_dim + 1  # D feature channels + 1 channel low-res FTLE
        self.inc = DoubleConv(in_channels, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.up1 = Up(base_ch * 4, base_ch * 2)
        self.up2 = Up(base_ch * 2, base_ch)
        self.out_low = nn.Conv2d(base_ch, base_ch, kernel_size=1)

        n_up = max(0, int(round(math.log2(max(1, self.upscale)))))
        self.up_blocks = nn.ModuleList()
        in_ch = base_ch
        for i in range(n_up):
            out_ch = base_ch if i < n_up - 1 else base_ch // 2
            self.up_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            )
            in_ch = out_ch
        self.out_high = nn.Conv2d(in_ch, 1, kernel_size=1)

    def _tiling_starts(self, length: int, k: int):
        if k >= length:
            return [0]
        s = int(k)
        starts = list(range(0, length - k + 1, s))
        last = length - k
        if starts[-1] != last:
            starts.append(last)
        return starts

    def forward(self, lowResFTLE: torch.Tensor, lowResPathlines: torch.Tensor) -> torch.Tensor:
        B, X, Y = lowResFTLE.shape
        _, N, nerbors, L, Dim = lowResPathlines.shape
        assert N == X * Y, "lowResPathlines second dim must be X*Y"
        assert nerbors == self.cross_neighborsize, "nerbors mismatch with model setting"

        # 1) Sliding-window DCT_FMT -> coarse feature map [B, D, Hc, Wc]
        k = int(self.FMT_focus_area)
        row_starts = self._tiling_starts(int(X), k)
        col_starts = self._tiling_starts(int(Y), k)
        Hc, Wc = len(row_starts), len(col_starts)

        feat_coarse = lowResFTLE.new_zeros((B, self.fmt_feature_dim, Hc, Wc))
        for ri, i0 in enumerate(row_starts):
            i1 = min(i0 + k, int(X))
            for ci, j0 in enumerate(col_starts):
                j1 = min(j0 + k, int(Y))
                idx_list = []
                for rr in range(i0, i1):
                    base = rr * int(Y)
                    idx_list.extend(range(base + j0, base + j1))
                if len(idx_list) == 0:
                    continue
                idx_tensor = torch.as_tensor(idx_list, dtype=torch.long, device=lowResFTLE.device)
                # Keep structure: [B, M, nerbors, L, Dim] (do NOT flatten to a point cloud)
                pl_win = lowResPathlines[:, idx_tensor, ...]
                feat_win = self.encoder(pl_win)  # [B, D]
                feat_coarse[:, :, ri, ci] = feat_win

        # 2) Bilinear upsample the coarse feature map to [X, Y]
        feat_map = F.interpolate(feat_coarse, size=(int(X), int(Y)), mode='bilinear', align_corners=False)
        # 3) Concatenate with low-resolution FTLE
        ftle_in = lowResFTLE.unsqueeze(1)  # [B, 1, X, Y]
        x_in = torch.cat([ftle_in, feat_map], dim=1)  # [B, D+1, X, Y]

        # 4) UNet encode-decode + upsampling head
        x1 = self.inc(x_in)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        x = self.out_low(x)
        for blk in self.up_blocks:
            x = blk(x)
        pred = self.out_high(x).squeeze(1)  # [B, X*UP, Y*UP]

        # Size alignment
        target_h = int(X * max(1, self.upscale))
        target_w = int(Y * max(1, self.upscale))
        if pred.shape[-2] != target_h or pred.shape[-1] != target_w:
            pred = F.interpolate(pred.unsqueeze(1), size=(target_h, target_w), mode='bilinear', align_corners=False).squeeze(1)
        return pred



      
# ============================================================================
# Flow-map upsampling models
# ----------------------------------------------------------------------------
# Input/label are the cross-primitive flow map [B, P, 5, 2, 3] = (P grid cells)
# x (5 cross lines: center,x+,x-,y+,y-) x (2 endpoints: head,tail) x (3: x,y,t).
# FTLE is computed downstream from this via computeFTLEFromPathlineCrossPrimitive.
#
# Channel handling: the time channel (t) of every endpoint is CONSTANT across the
# grid (all seeds share the slice time and integration window), so BatchNorm would
# destroy it. We therefore route only the (x,y) channels (5*2*2=20) through the CNN
# and pass the t channels (5*2=10) through by nearest-neighbour resize (exact for a
# constant field). This keeps FTLE's physical time span ΔT correct.
# ============================================================================
class _FlowMapUpsamplerBase(nn.Module):
    """Shared field<->image plumbing for flow-map upsamplers. Subclasses implement
    `_cnn(xy_img)` mapping [B,20,H,W] -> [B,20,H*UP,W*UP]."""
    NERB = 5
    ENDPTS = 2
    FeactureChannel_XY = 5 * 2 * 2   # 20
    FeactureChannel_T = 5 * 2        # 10
    C_XY = FeactureChannel_XY        # alias used by subclasses (e.g. UNet_FlowMap)

    def __init__(self, upscale: int):
        super().__init__()
        self.upscale = int(upscale)

    def _cnn(self, xy_img: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, lowResFlowMap: torch.Tensor, lowResPathlines: torch.Tensor | None = None,
                hw: tuple[int, int] | None = None) -> torch.Tensor:
        # lowResFlowMap: [B, P, 5, 2, 3]; P=H*W (square patch in training, or full grid in eval via hw)
        B, P = lowResFlowMap.shape[0], lowResFlowMap.shape[1]
        if hw is None:
            side = int(round(P ** 0.5))
            assert side * side == P, f"non-square P={P}; pass hw=(H,W) explicitly"
            H, W = side, side
        else:
            H, W = int(hw[0]), int(hw[1])
        UP = max(1, self.upscale)
        Hh, Wh = H * UP, W * UP

        fm = lowResFlowMap.reshape(B, H, W, self.NERB, self.ENDPTS, 3)
        xy = fm[..., :2]   # [B,H,W,5,2,2]
        t  = fm[..., 2:]   # [B,H,W,5,2,1]
        xy_img = xy.reshape(B, H, W, self.FeactureChannel_XY).permute(0, 3, 1, 2).contiguous()  # [B,20,H,W]
        t_img  = t.reshape(B, H, W, self.FeactureChannel_T).permute(0, 3, 1, 2).contiguous()    # [B,10,H,W]
        xy_hi = self._cnn(xy_img)  # [B,20,Hh,Wh]
        if xy_hi.shape[-2] != Hh or xy_hi.shape[-1] != Wh:
            xy_hi = F.interpolate(xy_hi, size=(Hh, Wh), mode='bilinear', align_corners=False)
        # t is (piecewise) constant -> nearest resize reproduces it exactly
        t_hi = F.interpolate(t_img, size=(Hh, Wh), mode='nearest')  # [B,10,Hh,Wh]

        xy_hi_s = xy_hi.permute(0, 2, 3, 1).reshape(B, Hh, Wh, self.NERB, self.ENDPTS, 2)
        t_hi_s  = t_hi.permute(0, 2, 3, 1).reshape(B, Hh, Wh, self.NERB, self.ENDPTS, 1)
        out = torch.cat([xy_hi_s, t_hi_s], dim=-1)  # [B,Hh,Wh,5,2,3]
        return out.reshape(B, Hh * Wh, self.NERB, self.ENDPTS, 3)


class ESPCN_FlowMap(_FlowMapUpsamplerBase):
    """ESPCN-style flat CNN for flow-map upsampling (20-channel in/out)."""
    def __init__(self, cfg, lowResX: int, lowResY: int, upscale: int = 2):
        super().__init__(upscale)
        f1 = int(getattr(cfg.model, 'f1', 5))
        f2 = int(getattr(cfg.model, 'f2', 3))
        n1 = int(getattr(cfg.model, 'n1', 64))
        n2 = int(getattr(cfg.model, 'n2', 32))
        p1, p2 = f1 // 2, f2 // 2
        self.conv1 = nn.Conv2d(self.FeactureChannel_XY, n1, kernel_size=f1, padding=p1, bias=True)
        self.conv2 = nn.Conv2d(n1, n2, kernel_size=f2, padding=p2, bias=True)
        k_up = max(2, 2 * self.upscale)
        s_up = max(1, self.upscale)
        p_up = self.upscale // 2
        self.deconv = nn.ConvTranspose2d(n2, self.FeactureChannel_XY, kernel_size=k_up, stride=s_up, padding=p_up, bias=True)
        self.act = nn.ReLU(inplace=True)

    def _cnn(self, xy_img: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(xy_img))
        x = self.act(self.conv2(x))
        return self.deconv(x)


class UNet_FlowMap(_FlowMapUpsamplerBase):
    """SR-oriented shallow UNet (one downsample) for flow-map upsampling (20-ch in/out).
    Mirrors UpsamplingUnetModelV2 but multi-channel and with no t through BatchNorm."""
    def __init__(self, cfg, lowResX: int, lowResY: int, upscale: int = 2, base_ch: int = 64):
        super().__init__(upscale)
        self.inc = DoubleConv(self.C_XY, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.up1 = Up(base_ch * 2, base_ch)
        self.out_low = nn.Conv2d(base_ch, base_ch, kernel_size=1)
        n_up = max(0, int(round(math.log2(max(1, self.upscale)))))
        self.up_blocks = nn.ModuleList()
        in_ch = base_ch
        for i in range(n_up):
            out_ch = base_ch if i < n_up - 1 else base_ch // 2
            self.up_blocks.append(nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))
            in_ch = out_ch
        self.out_high = nn.Conv2d(in_ch, self.C_XY, kernel_size=1)

    def _cnn(self, xy_img: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(xy_img)
        x2 = self.down1(x1)
        x = self.up1(x2, x1)
        x = self.out_low(x)
        for blk in self.up_blocks:
            x = blk(x)
        return self.out_high(x)


class ESPCN_SR(nn.Module):
    """Canonical efficient sub-pixel CNN (Shi et al. 2016) for 2-channel flow-map SR,
    matching Jakob et al. 2020: two conv layers (f1,f2; n1,n2 features, ReLU) followed by
    a sub-pixel convolution that upsamples by `upscale`. Input/output are 2-channel images
    (the particle end-position flow map). Fully convolutional -> any H x W at inference."""
    def __init__(self, cfg, upscale: int = 2, in_ch: int = 2):
        super().__init__()
        f1 = int(getattr(cfg.model, 'f1', 3))
        f2 = int(getattr(cfg.model, 'f2', 3))
        n1 = int(getattr(cfg.model, 'n1', 128))
        n2 = int(getattr(cfg.model, 'n2', 128))
        self.upscale = int(upscale)
        self.in_ch = int(in_ch)
        self.conv1 = nn.Conv2d(in_ch, n1, kernel_size=f1, padding=f1 // 2)
        self.conv2 = nn.Conv2d(n1, n2, kernel_size=f2, padding=f2 // 2)
        self.conv3 = nn.Conv2d(n2, in_ch * self.upscale * self.upscale, kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(self.upscale)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 2, H, W] -> [B, 2, H*UP, W*UP]
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        return self.shuffle(x)


class UNet_SR(nn.Module):
    """UNet body (2 downsamples) + sub-pixel upsampling head for 2-channel flow-map SR.
    Same I/O contract as ESPCN_SR (2-ch image -> 2-ch image at k x). Fully convolutional."""
    def __init__(self, cfg, upscale: int = 2, in_ch: int = 2, base: int = 64):
        super().__init__()
        base = int(getattr(cfg.model, 'base', base))
        self.upscale = int(upscale)
        self.inc = DoubleConv(in_ch, base)
        self.down1 = Down(base, base * 2)
        self.down2 = Down(base * 2, base * 4)
        self.up1 = Up(base * 4, base * 2)
        self.up2 = Up(base * 2, base)
        self.head = nn.Sequential(
            nn.Conv2d(base, in_ch * self.upscale * self.upscale, kernel_size=3, padding=1),
            nn.PixelShuffle(self.upscale),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        return self.head(x)


def build_model(config, device):
    nerb = int(config.pcds.num_cross_points_per_seeding)
    L = int(config.pcds.sampled_points_per_line)
    if config.model.NAME == 'FMT_Regressor':
        model = PointWiseFMT_Regressor(LStpesPerline=L,num_stages=config.pnn.stages, embed_dim=config.pnn.dim,
                              k_neighbors=config.pnn.k, beta=config.pnn.beta, alpha=config.pnn.alpha).to(device)
    elif config.model.NAME == 'MLP_Regressor':
        model = PointWiseMLP_Regressor(LStpesPerline=L,num_stages=config.pnn.stages, embed_dim=config.pnn.dim,
                              k_neighbors=config.pnn.k, beta=config.pnn.beta, alpha=config.pnn.alpha).to(device)
        return model
    elif config.model.NAME == 'UpsamplingUnetModel':
        lowResX = int(config.lowResX)
        lowResY = int(config.lowResY)
        model = UpsamplingUnetModel(config, lowResX, lowResY,upscale=int(config.dataset.UPsampling)).to(device)
        return model
    elif config.model.NAME == 'UpsamplingUnetModelV2':
        lowResX = int(config.lowResX)
        lowResY = int(config.lowResY)
        model = UpsamplingUnetModelV2(config, lowResX, lowResY,upscale=int(config.dataset.UPsampling)).to(device)
        return model
    elif config.model.NAME == 'FTLEUpsamplingFMT_Unet':
        lowResX = int(config.lowResX)
        lowResY = int(config.lowResY)
        model = FTLEUpsamplingFMT_Unet(config, lowResX, lowResY,upscale=int(config.dataset.UPsampling)).to(device)
        return model
    elif config.model.NAME == 'FTLEUpsamplingFMT_UnetV2':
        lowResX = int(config.lowResX)
        lowResY = int(config.lowResY)
        model = FTLEupsamplingFMT_UnetV2(config, lowResX, lowResY,upscale=int(config.dataset.UPsampling)).to(device)
        return model
    elif config.model.NAME == 'FTLEUpsamplingDCT_FMT_UnetV2' or config.model.NAME == 'DCT_FMT_UnetV2':
        lowResX = int(config.lowResX)
        lowResY = int(config.lowResY)
        model = FTLEupsamplingDCT_FMT_UnetV2(config, lowResX, lowResY,upscale=int(config.dataset.UPsampling)).to(device)
        return model
    elif config.model.NAME == 'FTLEUpsamplingFMT_UnetV3' or config.model.NAME == 'FTLEUpsamplingFMT_Unet_V3':
        lowResX = int(config.lowResX)
        lowResY = int(config.lowResY)
        model = FTLEupsamplingFMT_UnetV3(config, lowResX, lowResY,upscale=int(config.dataset.UPsampling)).to(device)
        return model
    elif config.model.NAME == 'ESPCN_FlowMap':
        lowResX = int(config.lowResX)
        lowResY = int(config.lowResY)
        model = ESPCN_FlowMap(config, lowResX, lowResY, upscale=int(config.dataset.UPsampling)).to(device)
        return model
    elif config.model.NAME == 'UNet_FlowMap':
        lowResX = int(config.lowResX)
        lowResY = int(config.lowResY)
        model = UNet_FlowMap(config, lowResX, lowResY, upscale=int(config.dataset.UPsampling)).to(device)
        return model
    elif config.model.NAME == 'ESPCN_SR':
        model = ESPCN_SR(config, upscale=int(config.dataset.UPsampling), in_ch=2).to(device)
        return model
    elif config.model.NAME == 'UNet_SR':
        model = UNet_SR(config, upscale=int(config.dataset.UPsampling), in_ch=2).to(device)
        return model
    else:
        raise ValueError(f"Unknown model: {config.model.NAME}")

