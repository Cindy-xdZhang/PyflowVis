import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pnn.models.point_nn import EncNPNew
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
 
def sampling_lowResPathlines_based_on_lowResFTLE(lowResFTLE: torch.Tensor, lowResPathlines: torch.Tensor, sampling_ratio: float = 0.5):
    """
    Sample low-resolution pathlines based on low-resolution FTLE values.
    Args:
        lowResFTLE:        [B, X, Y]
        lowResPathlines:   [B, X*Y, nerbors, max_steps, Dim]
        sampling_ratio:    0~1 

    Returns:
        sampled_pathlines: [B, M, nerbors, max_steps, Dim], where M = floor(X*Y*ratio) and M>=1
        sampled_indices:   [B, M], linear indices (0..X*Y-1) per batch
    """
    assert lowResFTLE.dim() == 3, "lowResFTLE must be [B, X, Y]"
    assert lowResPathlines.dim() == 5, "lowResPathlines must be [B, X*Y, nerbors, max_steps, Dim]"
    B, lowResX, lowResY = lowResFTLE.shape
    B2, N, nerbors, max_steps, Dim = lowResPathlines.shape
    assert B == B2, "Batch size mismatch between FTLE and pathlines"
    assert N == lowResX * lowResY, "Pathlines second dim must equal X*Y"

    sampling_ratio = float(max(0.0, min(1.0, sampling_ratio)))
    M = int(max(100, int(N * sampling_ratio)))  
    device = lowResFTLE.device
    sampled_list = []
    sampled_idx_list = []

    for b in range(B):
        ftle_b = lowResFTLE[b].reshape(-1).float()
        ftle_min = torch.min(ftle_b)
        ftle_max = torch.max(ftle_b)
        if torch.isfinite(ftle_min) and torch.isfinite(ftle_max) and (ftle_max - ftle_min) > 1e-12:
            prob = (ftle_b - ftle_min) / (ftle_max - ftle_min)
            prob = prob.clamp(min=0.0)
        else:
            prob = torch.ones_like(ftle_b)

        s = prob.sum()
        if not torch.isfinite(s) or s <= 0:
            prob = torch.ones_like(prob) / float(N)
        else:
            prob = prob / s

        take = min(M, N)
        idx_b = torch.multinomial(prob, num_samples=take, replacement=False)
        sampled_idx_list.append(idx_b)
        sampled_list.append(lowResPathlines[b, idx_b, ...])

    sampled_pathlines = torch.stack(sampled_list, dim=0).to(device)
    sampled_indices = torch.stack(sampled_idx_list, dim=0).to(device)
    return sampled_pathlines, sampled_indices


# class FTLEUpsamplingModel(nn.Module):
#     def __init__(self, cfg, lowResX, lowResY, num_stages=2, embed_dim=72, k_neighbors=6, cross_neighborsize=5,beta=100, alpha=1000):
#         super().__init__()
#         #FTLERegressor is a point wise regressor, has no cross primitive, no neighbor information aggregation
#         LStpesPerline=cfg.pcds.sampled_points_per_line
#         self.cross_neighborsize=5
#         self.lowResX=lowResX
#         self.lowResY=lowResY
#         self.pointsPerUnit=int(LStpesPerline*cross_neighborsize*lowResX*lowResY*0.25)
#         self.encoder = EncNP(self.pointsPerUnit, num_stages, embed_dim, k_neighbors, alpha, beta)

#         self.decoderInputDim=embed_dim * (2 ** (num_stages - 0))
#         # self.decoderInputDim=embed_dim * (2 ** (num_stages - 0))+ 3*2*cross_neighborsize
#         # self.decoderInputDim=3*2*cross_neighborsize
        
#         self.MLP1=nn.Sequential(
#             nn.Linear(self.decoderInputDim, self.decoderInputDim*2),
#             nn.ReLU(),
#             nn.Linear(self.decoderInputDim*2, self.decoderInputDim),
#             nn.ReLU(),
#             nn.Linear(self.decoderInputDim,15),
#             )
#         #deconve upsampling ftle from lowREs to highRes            
#         self.DECONV1=nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2)
#         self.DECONV2=nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2)
#         self.DECONV3=nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2)
        
        
    
#     #todo:better option is to sample lowResPathlines based on lowResFTLE
#     def forward(self, lowResFTLE: torch.Tensor, lowResPathlines: torch.Tensor):
#         B,lowResX,lowResY=lowResFTLE.shape
#         _,lowResXtimeslowResY,nerbors,max_steps,Dim=lowResPathlines.shape
#         sampled_pathlines, sampled_indices=sampling_lowResPathlines_based_on_lowResFTLE(lowResFTLE, lowResPathlines,0.25)
#         PointsWholeFieldRaw=sampled_pathlines.reshape(B, self.pointsPerUnit,Dim)



#         points_3N=PointsWholeFieldRaw.permute(0, 2, 1)
#         points_N3=PointsWholeFieldRaw
#         globalFMT_feat = self.encoder(points_N3, points_3N)

#         #shape of wholeFieldFMT_feat=[B, 15]
#         wholeFieldFMT_feat=self.decoder(globalFMT_feat).repeat(1,1,lowResX,lowResY).permute(0, 2, 3, 1)
#         lowResFTLE = lowResFTLE.reshape(B, lowResX, lowResY, 1)
#         concat_feat = torch.cat([wholeFieldFMT_feat, lowResFTLE], dim=-1)
#         upsampled_ftle=self.DECONV1(concat_feat)
#         upsampled_ftle=self.DECONV2(upsampled_ftle)
#         upsampled_ftle=self.DECONV3(upsampled_ftle)
#         pred=upsampled_ftle.reshape(B, -1, 1)
#         return pred
    



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
        yy = torch.linspace(0, 1, steps=lowResX, device=device)
        xx = torch.linspace(0, 1, steps=lowResY, device=device)
        Y_grid, X_grid = torch.meshgrid(yy, xx, indexing='ij')  # [X,Y]
        coord = torch.stack([X_grid, Y_grid], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)  # [B,2,X,Y]
        self.cache_coordGrid=coord

    def forward(self, lowResFTLE: torch.Tensor, lowResPathlines: torch.Tensor) -> torch.Tensor:
        B, X, Y = lowResFTLE.shape
        _, N, nerbors, L, Dim = lowResPathlines.shape
        assert N == X * Y, "lowResPathlines second dim must be X*Y"
        assert nerbors == self.cross_neighborsize, "nerbors mismatch with model setting"
        if self.cache_coordGrid is None or self.cache_coordGrid.shape[0] != B:
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


class ESPCN(nn.Module):
    """
    简化版 ESPCN：两层卷积 + 一层反卷积（转置卷积）。
    输入:  lowResFTLE [B, X, Y]（低分辨率标量场），忽略 pathlines 参数
    输出:  高分辨率预测 [B, X*UP, Y*UP]
    可配置: f1,f2 为前两层卷积核尺寸；n1,n2 为相应通道数。
    """
    def __init__(self, cfg, lowResX: int, lowResY: int, upscale: int = 4):
        super().__init__()
        self.lowResX = int(lowResX)
        self.lowResY = int(lowResY)
        self.upscale = int(upscale)

        # 读取超参数（若配置未提供则使用默认）
        try:
            f1 = int(getattr(cfg.model, 'f1', 5))
            f2 = int(getattr(cfg.model, 'f2', 3))
            n1 = int(getattr(cfg.model, 'n1', 64))
            n2 = int(getattr(cfg.model, 'n2', 32))
        except Exception:
            f1, f2, n1, n2 = 5, 3, 64, 32

        p1 = max(0, f1 // 2)
        p2 = max(0, f2 // 2)

        self.conv1 = nn.Conv2d(1, n1, kernel_size=f1, padding=p1, bias=True)
        self.conv2 = nn.Conv2d(n1, n2, kernel_size=f2, padding=p2, bias=True)

        # 反卷积做上采样（常用设置：kernel=2*scale, stride=scale, padding=scale//2）
        k_up = max(2, 2 * self.upscale)
        s_up = max(1, self.upscale)
        p_up = self.upscale // 2
        self.deconv = nn.ConvTranspose2d(n2, 1, kernel_size=k_up, stride=s_up, padding=p_up, output_padding=0, bias=True)

        self.act = nn.ReLU(inplace=True)

    def forward(self, lowResFTLE: torch.Tensor, lowResPathlines: torch.Tensor | None = None) -> torch.Tensor:
        B, X, Y = lowResFTLE.shape
        x = lowResFTLE.unsqueeze(1).contiguous()  # [B,1,X,Y]
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.deconv(x).squeeze(1)  # [B, Hh, Wh]

        target_h = int(X * max(1, self.upscale))
        target_w = int(Y * max(1, self.upscale))
        if x.shape[-2] != target_h or x.shape[-1] != target_w:
            x = F.interpolate(x.unsqueeze(1), size=(target_h, target_w), mode='bilinear', align_corners=False).squeeze(1)
        return x


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


class ConditionalFMTNet(nn.Module):
    """
    隐式查询网络：对任意查询像素 (x, y) 直接预测 f(x,y)=FTLE。
    实现思路：
      1) 对低分辨率每个栅格单元的 cross-primitive 路径线做 FMT 编码，得到 token feature map [B, D, X, Y]
      2) 将 token feature map 和低分辨率 FTLE 分别双线性上采样到高分辨率 [B, D, X*UP, Y*UP] 与 [B, 1, X*UP, Y*UP]
      3) 构造归一化坐标通道 [B, 2, X*UP, Y*UP]（x,y∈[0,1]）
      4) 按像素拼接 [token, lowres_ftle_interp, x, y]，经点 MLP 输出预测值
    
    输入：
      - lowResFTLE:      [B, X, Y]
      - lowResPathlines: [B, X*Y, nerbors, L, 3]
    输出：
      - pred:            [B, X*UP, Y*UP]（与高分辨率标签对齐）
    """
    def __init__(self, cfg, lowResX: int, lowResY: int, upscale: int = 4,
                 embed_dim: int | None = None, mlp_hidden: int = 256, mlp_depth: int = 3, dropout: float = 0.0):
        super().__init__()
        self.lowResX = int(lowResX)
        self.lowResY = int(lowResY)
        self.upscale = int(upscale)

        # FMT tokenizer（每低分辨率单元一 token）
        k = int(getattr(cfg.pnn, 'k', 6)) if hasattr(cfg, 'pnn') else 6
        alpha = float(getattr(cfg.pnn, 'alpha', 1000)) if hasattr(cfg, 'pnn') else 1000.0
        beta = float(getattr(cfg.pnn, 'beta', 100)) if hasattr(cfg, 'pnn') else 100.0
        nerbors = int(getattr(cfg.pcds, 'num_cross_points_per_seeding', 5)) if hasattr(cfg, 'pcds') else 5
        LstepsPerline = int(getattr(cfg.pcds, 'sampled_points_per_line', 4)) if hasattr(cfg, 'pcds') else 4
        self.cross_neighborsize = nerbors
        self.pointsPerPrimitive = LstepsPerline * nerbors
        self.num_stages = int(getattr(cfg.pnn, 'stages', 0)) if hasattr(cfg, 'pnn') else 0
        embed_dim= int(embed_dim if embed_dim is not None else getattr(cfg.pnn, 'dim', 36))
        self.fmt_feature_dim = embed_dim * (2 ** (self.num_stages - 0))
        # 为稳定输出维度，采用 stages=0（与 Unet/Vit 实现一致）
        self.encoder = EncNPNew(self.pointsPerPrimitive, self.num_stages, embed_dim, k, alpha, beta)

        # 点 MLP：输入维度 = token_dim + 1(lowres ftle) + 2(xy)
        in_dim = self.fmt_feature_dim + 3
        layers: list[nn.Module] = []
        dim = in_dim
        for i in range(int(max(1, mlp_depth))):
            next_dim = mlp_hidden if i < mlp_depth - 1 else 1
            layers.append(nn.Linear(dim, next_dim))
            if i < mlp_depth - 1:
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            dim = next_dim
        self.mlp = nn.Sequential(*layers)

    def forward(self, lowResFTLE: torch.Tensor, lowResPathlines: torch.Tensor) -> torch.Tensor:
        B, X, Y = lowResFTLE.shape
        _, N, nerbors, L, Dim = lowResPathlines.shape
        assert N == X * Y, "lowResPathlines second dim must be X*Y"
        assert nerbors == self.cross_neighborsize, "nerbors mismatch with model setting"

        # 1) 每格 FMT 编码 → [B, D, X, Y]
        P = lowResPathlines.reshape(B * N, nerbors * L, Dim).contiguous()
        points_N3 = P
        points_3N = P.permute(0, 2, 1).contiguous()
        feat = self.encoder(points_N3, points_3N)  # [B*N, D]
        feat = feat.reshape(B, X, Y, self.fmt_feature_dim).permute(0, 3, 1, 2).contiguous()

        # 2) 上采样至目标高分辨率
        target_h = int(X * max(1, self.upscale))
        target_w = int(Y * max(1, self.upscale))
        feat_hi = F.interpolate(feat, size=(target_h, target_w), mode='bilinear', align_corners=False)  # [B,D,Hh,Wh]
        lr_up = F.interpolate(lowResFTLE.unsqueeze(1), size=(target_h, target_w), mode='bilinear', align_corners=False)  # [B,1,Hh,Wh]

        # 3) 坐标通道（归一化到[0,1]）
        yy = torch.linspace(0, 1, steps=target_h, device=lowResFTLE.device)
        xx = torch.linspace(0, 1, steps=target_w, device=lowResFTLE.device)
        Y_grid, X_grid = torch.meshgrid(yy, xx, indexing='ij')  # [Hh,Wh]
        coord = torch.stack([X_grid, Y_grid], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)  # [B,2,Hh,Wh]

        # 4) 点查询 MLP（逐像素）
        # 拼接顺序：[token_feat(D), lowres_ftle(1), x(1), y(1)]
        B_, D_, Hh, Wh = feat_hi.shape
        token_flat = feat_hi.permute(0, 2, 3, 1).reshape(B_ * Hh * Wh, D_)
        lr_flat = lr_up.permute(0, 2, 3, 1).reshape(B_ * Hh * Wh, 1)
        xy_flat = coord.permute(0, 2, 3, 1).reshape(B_ * Hh * Wh, 2)
        mlp_in = torch.cat([token_flat, lr_flat, xy_flat], dim=1)  # [B*Hh*Wh, D+3]
        out = self.mlp(mlp_in).reshape(B_, Hh, Wh)
        # 训练中标签已归一化，无需此处再做激活，保持线性输出由损失控制
        return out

class FTLEUpsamplingFMT_Vit(nn.Module):
    """
    vision transformer for upsampling FTLE from lowRes to highRes
    """
    def __init__(self, cfg, lowResX: int, lowResY: int, upscale: float,base_ch: int = 32,
                 embed_dim: int | None = None, vit_dim: int = 64, depth: int = 4, heads: int = 4,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.lowResX = int(lowResX)
        self.lowResY = int(lowResY)
        self.upscale = int(upscale)

        # FMT tokenizer: EncNPNew per-cell features + FTLE + XY → token
        k = int(getattr(cfg.pnn, 'k', 6)) if hasattr(cfg, 'pnn') else 6
        alpha = float(getattr(cfg.pnn, 'alpha', 1000)) if hasattr(cfg, 'pnn') else 1000.0
        beta = float(getattr(cfg.pnn, 'beta', 100)) if hasattr(cfg, 'pnn') else 100.0
        nerbors = int(getattr(cfg.pcds, 'num_cross_points_per_seeding', 5)) if hasattr(cfg, 'pcds') else 5
        LstepsPerline = int(getattr(cfg.pcds, 'sampled_points_per_line', 4)) if hasattr(cfg, 'pcds') else 4
        self.cross_neighborsize = nerbors
        self.pointsPerPrimitive = LstepsPerline * nerbors
        embed_dim = int(embed_dim if embed_dim is not None else getattr(cfg.pnn, 'dim', 36))
        self.num_stages = int(getattr(cfg.pnn, 'stages', 0)) if hasattr(cfg, 'pnn') else 0
        self.encoder = EncNPNew(self.pointsPerPrimitive, self.num_stages, embed_dim, k, alpha, beta)

        self.fmt_feature_dim = embed_dim * (2 ** (self.num_stages - 0))


        # token dimension
        self.token_in_dim = self.fmt_feature_dim + 3
        self.vit_dim = int(vit_dim)
        self.token_proj = nn.Linear(self.token_in_dim, self.vit_dim)

        # transformer stack
        self.blocks = nn.ModuleList([
            TransformerBlock(self.vit_dim, heads=heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(int(depth))
        ])
        self.norm = nn.LayerNorm(self.vit_dim)

        # map tokens back to feature map
        self.out_proj = nn.Linear(self.vit_dim, base_ch)

        # upsampling head (like Unet head)
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

    @staticmethod
    def _pos_embed_1d(n: int, dim: int, device):
        assert dim % 2 == 0
        pos = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(1)
        i = torch.arange(dim // 2, device=device, dtype=torch.float32)
        denom = torch.pow(10000.0, (2 * i) / dim)
        angles = pos / denom
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)

    @classmethod
    def _pos_embed_2d(cls, h: int, w: int, dim: int, device):
        assert dim % 2 == 0
        dim_h = dim // 2
        dim_w = dim - dim_h
        pe_h = cls._pos_embed_1d(h, dim_h, device)
        pe_w = cls._pos_embed_1d(w, dim_w, device)
        pe_h = pe_h[:, None, :].expand(h, w, dim_h)
        pe_w = pe_w[None, :, :].expand(h, w, dim_w)
        pe = torch.cat([pe_h, pe_w], dim=2).reshape(1, h * w, dim)
        return pe

    def forward(self, lowResFTLE: torch.Tensor, lowResPathlines: torch.Tensor) -> torch.Tensor:
        B, X, Y = lowResFTLE.shape
        _, N, nerbors, L, Dim = lowResPathlines.shape
        assert N == X * Y, "lowResPathlines second dim must be X*Y"
        assert nerbors == self.cross_neighborsize, "nerbors mismatch with model setting"

        # FMT tokenization per cell
        P = lowResPathlines.reshape(B * N, nerbors * L, Dim).contiguous()
        points_N3 = P
        points_3N = P.permute(0, 2, 1).contiguous()
        feat = self.encoder(points_N3, points_3N)
        feat = feat.reshape(B, X, Y, self.fmt_feature_dim).permute(0, 3, 1, 2).contiguous()

        # coordinate channels
        yy = torch.linspace(0, 1, steps=X, device=lowResFTLE.device)
        xx = torch.linspace(0, 1, steps=Y, device=lowResFTLE.device)
        Y_grid, X_grid = torch.meshgrid(yy, xx, indexing='ij')
        coord = torch.stack([X_grid, Y_grid], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)

        # tokens
        ftle_in = lowResFTLE.unsqueeze(1)
        x_in = torch.cat([ftle_in, coord, feat], dim=1)  # [B, embed_dim+3, X, Y]
        tokens = x_in.permute(0, 2, 3, 1).reshape(B, X * Y, self.token_in_dim)

        z = self.token_proj(tokens)
        pos = self._pos_embed_2d(X, Y, self.vit_dim, device=lowResFTLE.device)
        z = z + pos

        for blk in self.blocks:
            z = blk(z)
        z = self.norm(z)

        fm = self.out_proj(z).reshape(B, X, Y, -1).permute(0, 3, 1, 2).contiguous()
        x = fm
        for blk in self.up_blocks:
            x = blk(x)
        pred = self.out_high(x).squeeze(1)

        target_h = int(X * max(1, self.upscale))
        target_w = int(Y * max(1, self.upscale))
        if pred.shape[-2] != target_h or pred.shape[-1] != target_w:
            pred = F.interpolate(pred.unsqueeze(1), size=(target_h, target_w), mode='bilinear', align_corners=False).squeeze(1)
        return pred # [B, X*UP, Y*UP]

class FTLEUpsamplingFMT_Vit_Version2(nn.Module):

    """
    局部 tile → FMT 编码成 coarse token → ViT → 上采样到高分辨率。

    步骤：
      1) 在低分辨率网格上以窗口大小 k=self.FMT_focus_area、步长同 k 做滑窗，
         将该窗口内所有 cross-primitive 拼成点集，经 EncNPNew 得到局部特征 f ∈ R^D，
         填入 coarse 特征图 [B, D, Hc, Wc]。
      2) 将低分辨率 FTLE 下采样得到 [B,1,Hc,Wc] 做为额外 token 分量，并构造二维坐标编码 [B,2,Hc,Wc]。
      3) 将 [FMT, FTLEc, XYc] 拼接为 coarse token，输入 ViT。
      4) ViT 输出映射回 [B, Cb, Hc, Wc]，双线性上采样至 [B, Cb, X, Y]，再经过上采样头输出 [B, X*UP, Y*UP]。
    输入：
      - lowResFTLE:      [B, X, Y]
      - lowResPathlines: [B, X*Y, nerbors, L, 3]
    输出：
      - pred:            [B, X*UP, Y*UP]
    """
    def __init__(self, cfg, lowResX: int, lowResY: int, upscale: float,
                 base_ch: int = 32, vit_dim: int = 64, depth: int = 4, heads: int = 4,
                 mlp_ratio: float = 4.0, dropout: float = 0.0,
                 embed_dim: int | None = None):
        super().__init__()
        self.lowResX = int(lowResX)
        self.lowResY = int(lowResY)
        self.upscale = int(upscale)

        # FMT encoder（局部 tile）
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

        # ViT token维度：FMT D + FTLEc(1) + XY(2)
        self.token_in_dim = self.fmt_feature_dim + 3
        self.vit_dim = int(vit_dim)
        self.token_proj = nn.Linear(self.token_in_dim, self.vit_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(self.vit_dim, heads=heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(int(depth))
        ])
        self.norm = nn.LayerNorm(self.vit_dim)

        # 将 token 还原为特征图通道数
        self.out_proj = nn.Linear(self.vit_dim, base_ch)

        # 上采样头（powers of 2）
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

    @staticmethod
    def _pos_embed_1d(n: int, dim: int, device):
        assert dim % 2 == 0
        pos = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(1)
        i = torch.arange(dim // 2, device=device, dtype=torch.float32)
        denom = torch.pow(10000.0, (2 * i) / dim)
        angles = pos / denom
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)

    @classmethod
    def _pos_embed_2d(cls, h: int, w: int, dim: int, device):
        assert dim % 2 == 0
        dim_h = dim // 2
        dim_w = dim - dim_h
        pe_h = cls._pos_embed_1d(h, dim_h, device)
        pe_w = cls._pos_embed_1d(w, dim_w, device)
        pe_h = pe_h[:, None, :].expand(h, w, dim_h)
        pe_w = pe_w[None, :, :].expand(h, w, dim_w)
        pe = torch.cat([pe_h, pe_w], dim=2).reshape(1, h * w, dim)
        return pe

    def forward(self, lowResFTLE: torch.Tensor, lowResPathlines: torch.Tensor) -> torch.Tensor:
        B, X, Y = lowResFTLE.shape
        _, N, nerbors, L, Dim = lowResPathlines.shape
        assert N == X * Y, "lowResPathlines second dim must be X*Y"
        assert nerbors == self.cross_neighborsize, "nerbors mismatch with model setting"

        # 1) 局部 FMT → coarse feature map [B, D, Hc, Wc]
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
                pl_win = lowResPathlines[:, idx_tensor, ...]
                B2, M, _, _, _ = pl_win.shape
                P = pl_win.reshape(B2, M * nerbors * L, Dim).contiguous()
                points_N3 = P
                points_3N = P.permute(0, 2, 1).contiguous()
                feat_win = self.encoder(points_N3, points_3N)  # [B, D]
                feat_coarse[:, :, ri, ci] = feat_win

        # 2) 构造 coarse 级别的 FTLE 与坐标
        ftle_coarse = F.interpolate(lowResFTLE.unsqueeze(1), size=(Hc, Wc), mode='bilinear', align_corners=False)  # [B,1,Hc,Wc]
        yy = torch.linspace(0, 1, steps=Hc, device=lowResFTLE.device)
        xx = torch.linspace(0, 1, steps=Wc, device=lowResFTLE.device)
        Y_grid, X_grid = torch.meshgrid(yy, xx, indexing='ij')
        coord_coarse = torch.stack([X_grid, Y_grid], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)  # [B,2,Hc,Wc]

        # 3) 组成 tokens 并送入 ViT
        x_in = torch.cat([feat_coarse, ftle_coarse, coord_coarse], dim=1)  # [B, D+3, Hc, Wc]
        tokens = x_in.permute(0, 2, 3, 1).reshape(B, Hc * Wc, self.token_in_dim)
        z = self.token_proj(tokens)
        pos = self._pos_embed_2d(Hc, Wc, self.vit_dim, device=lowResFTLE.device)
        z = z + pos

        for blk in self.blocks:
            z = blk(z)
        z = self.norm(z)

        fm_coarse = self.out_proj(z).reshape(B, Hc, Wc, -1).permute(0, 3, 1, 2).contiguous()  # [B,Cb,Hc,Wc]

        # 4) 上采样到 [X,Y] 再到 [X*UP,Y*UP]
        fm = F.interpolate(fm_coarse, size=(int(X), int(Y)), mode='bilinear', align_corners=False)  # [B,Cb,X,Y]
        x = fm
        for blk in self.up_blocks:
            x = blk(x)
        pred = self.out_high(x).squeeze(1)

        target_h = int(X * max(1, self.upscale))
        target_w = int(Y * max(1, self.upscale))
        if pred.shape[-2] != target_h or pred.shape[-1] != target_w:
            pred = F.interpolate(pred.unsqueeze(1), size=(target_h, target_w), mode='bilinear', align_corners=False).squeeze(1)
        return pred

class FTLEUpsamplingViTBaseline(nn.Module):
    """
    标准 ViT 基线（无 pathline/FMT）：仅基于低分辨率 FTLE 与二维坐标编码构造 token，
    经过 Transformer 编码，再通过上采样头输出高分辨率 FTLE。

    输入:
      - lowResFTLE: [B, X, Y]
      - lowResPathlines: 忽略，仅为保持接口一致
    输出:
      - pred: [B, X*UP, Y*UP]
    """
    def __init__(self, cfg, lowResX: int, lowResY: int, upscale: float,
                 base_ch: int = 32, vit_dim: int = 64, depth: int = 4, heads: int = 4,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.lowResX = int(lowResX)
        self.lowResY = int(lowResY)
        self.upscale = int(upscale)

        # token: [FTLE(1), X(1), Y(1)] → Linear → vit_dim
        self.token_in_dim = 3
        self.vit_dim = int(vit_dim)
        self.token_proj = nn.Linear(self.token_in_dim, self.vit_dim)

        # transformer stack
        self.blocks = nn.ModuleList([
            TransformerBlock(self.vit_dim, heads=heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(int(depth))
        ])
        self.norm = nn.LayerNorm(self.vit_dim)

        # map tokens back to feature map
        self.out_proj = nn.Linear(self.vit_dim, base_ch)

        # upsampling head
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

    @staticmethod
    def _pos_embed_1d(n: int, dim: int, device):
        assert dim % 2 == 0
        pos = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(1)
        i = torch.arange(dim // 2, device=device, dtype=torch.float32)
        denom = torch.pow(10000.0, (2 * i) / dim)
        angles = pos / denom
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)

    @classmethod
    def _pos_embed_2d(cls, h: int, w: int, dim: int, device):
        assert dim % 2 == 0
        dim_h = dim // 2
        dim_w = dim - dim_h
        pe_h = cls._pos_embed_1d(h, dim_h, device)
        pe_w = cls._pos_embed_1d(w, dim_w, device)
        pe_h = pe_h[:, None, :].expand(h, w, dim_h)
        pe_w = pe_w[None, :, :].expand(h, w, dim_w)
        pe = torch.cat([pe_h, pe_w], dim=2).reshape(1, h * w, dim)
        return pe

    def forward(self, lowResFTLE: torch.Tensor, lowResPathlines: torch.Tensor | None = None) -> torch.Tensor:
        B, X, Y = lowResFTLE.shape

        # coordinate channels
        yy = torch.linspace(0, 1, steps=X, device=lowResFTLE.device)
        xx = torch.linspace(0, 1, steps=Y, device=lowResFTLE.device)
        Y_grid, X_grid = torch.meshgrid(yy, xx, indexing='ij')
        coord = torch.stack([X_grid, Y_grid], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)  # [B,2,X,Y]

        # tokens: [B, X*Y, 3]
        ftle_in = lowResFTLE.unsqueeze(1)
        x_in = torch.cat([ftle_in, coord], dim=1)
        tokens = x_in.permute(0, 2, 3, 1).reshape(B, X * Y, self.token_in_dim)

        z = self.token_proj(tokens)
        pos = self._pos_embed_2d(X, Y, self.vit_dim, device=lowResFTLE.device)
        z = z + pos

        for blk in self.blocks:
            z = blk(z)
        z = self.norm(z)

        fm = self.out_proj(z).reshape(B, X, Y, -1).permute(0, 3, 1, 2).contiguous()

        x = fm
        for blk in self.up_blocks:
            x = blk(x)
        pred = self.out_high(x).squeeze(1)

        target_h = int(X * max(1, self.upscale))
        target_w = int(Y * max(1, self.upscale))
        if pred.shape[-2] != target_h or pred.shape[-1] != target_w:
            pred = F.interpolate(pred.unsqueeze(1), size=(target_h, target_w), mode='bilinear', align_corners=False).squeeze(1)
        return pred




      
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
    elif config.model.NAME == 'FTLEUpsamplingFMT_Unet':
        lowResX = int(config.lowResX)
        lowResY = int(config.lowResY)
        model = FTLEUpsamplingFMT_Unet(config, lowResX, lowResY,upscale=int(config.dataset.UPsampling)).to(device)
        return model
    elif config.model.NAME == 'FTLEUpsamplingFMT_Vit':
        lowResX = int(config.lowResX)
        lowResY = int(config.lowResY)
        model = FTLEUpsamplingFMT_Vit(config, lowResX, lowResY,upscale=int(config.dataset.UPsampling)).to(device)
        return model
    elif config.model.NAME == 'FTLEUpsamplingFMT_Vit_Version2':
        lowResX = int(config.lowResX)
        lowResY = int(config.lowResY)
        model = FTLEUpsamplingFMT_Vit_Version2(config, lowResX, lowResY,upscale=int(config.dataset.UPsampling)).to(device)
        return model
    elif config.model.NAME == 'FTLEUpsamplingFMT_UnetV2':
        lowResX = int(config.lowResX)
        lowResY = int(config.lowResY)
        model = FTLEupsamplingFMT_UnetV2(config, lowResX, lowResY,upscale=int(config.dataset.UPsampling)).to(device)
        return model
    elif config.model.NAME == 'FTLEUpsamplingViTBaseline':
        lowResX = int(config.lowResX)
        lowResY = int(config.lowResY)
        model = FTLEUpsamplingViTBaseline(config, lowResX, lowResY,upscale=int(config.dataset.UPsampling)).to(device)
        return model
    elif config.model.NAME == 'ConditionalFMTNet':
        lowResX = int(config.lowResX)
        lowResY = int(config.lowResY)
        model = ConditionalFMTNet(config, lowResX, lowResY, upscale=int(config.dataset.UPsampling)).to(device)
        return model
    elif config.model.NAME == 'ESPCN':
        lowResX = int(config.lowResX)
        lowResY = int(config.lowResY)
        model = ESPCN(config, lowResX, lowResY,upscale=int(config.dataset.UPsampling)).to(device)
        return model
    else:   
        raise ValueError(f"Unknown model: {config.model.NAME}")

