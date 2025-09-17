#step 1: load FTLE dataset
import os,random
import hashlib
import copy
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from DeepUtils.utils import EasyConfig
from DeepUtils.loss import build_criterion_from_cfg
from DeepUtils.optim import build_optimizer_from_cfg

from FLowUtils.VectorField2d import *
from pnn.models.point_nn import EncNPNew,EncNP
from FTLE_fitting_utils import *
from DeepUtils.MiscFunctions import *

import wandb
GLOBAL_WANDB_PROJECT_NAME="FlowMapTokenizer"

torch.backends.cuda.matmul.allow_tf32 = False  

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
    def __init__(self, cfg, lowResX: int, lowResY: int, upscale:float,base_ch: int = 32):
        super().__init__()
        self.upscale = int(upscale)
        self.inc = DoubleConv(1, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.up1 = Up(base_ch * 4, base_ch * 2)
        self.up2 = Up(base_ch * 2, base_ch)

        # 低分辨率特征输出（与输入相同尺寸）
        self.out_low = nn.Conv2d(base_ch, base_ch, kernel_size=1)

        # 逐级上采样头（根据 2 的幂次构建）
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

        # 对齐到期望尺寸（防止奇偶尺寸导致 1 像素偏差）
        target_h = int(X * max(1, self.upscale))
        target_w = int(Y * max(1, self.upscale))
        if pred.shape[-2] != target_h or pred.shape[-1] != target_w:
            pred = F.interpolate(pred.unsqueeze(1), size=(target_h, target_w), mode='bilinear', align_corners=False).squeeze(1)
        return pred






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
        self.embed_dim = int(embed_dim if embed_dim is not None else getattr(cfg.pnn, 'dim', 36))
        self.encoder = EncNPNew(self.pointsPerPrimitive, 0, self.embed_dim, k, alpha, beta)

        # token dimension
        self.token_in_dim = self.embed_dim + 3
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
        feat = feat.reshape(B, X, Y, self.embed_dim).permute(0, 3, 1, 2).contiguous()

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

def build_test_dataset(config):
    """
    生成测试所需的低分辨率/高分辨率 FTLE 切片与预处理后的低分辨率流线数据，并加入缓存机制。

    返回:
        lowResFTLE_all: np.ndarray [T, X_low, Y_low]
        lowResPathlines_all: torch.FloatTensor [T, X_low*Y_low, nerbors, L, 3]
        highResFTLE_all: np.ndarray [T, X_high, Y_high]
        vectorfield: UnsteadyVectorField2D 对象（仅用于可视化/边界信息）
    """
    # 参数收集
    netCDF = NetCDFLoader()
    test_vectorfield_name = config['test_vectorfield'] if 'test_vectorfield' in config else config.dataset.names[0]
    vectorfield_datapath = f"{config.dataset.dat_dir}\\{test_vectorfield_name}.{config.dataset.extension}"
    vectorfield = netCDF.load_vector_field2d(vectorfield_datapath)

    tmin, tmax = float(vectorfield.tmin), float(vectorfield.tmax)
    time_window_start_ratio = float(config.dataset.t_start)
    time_window_target_ratio = float(config.dataset.t_target)
    time_window_start = float(time_window_start_ratio * (tmax - tmin) + tmin)
    time_window_target = float(time_window_target_ratio * (tmax - tmin) + tmin)
    timesliceCount = int(getattr(config.dataset, 'timesliceCount', 8))
    sample_times = np.linspace(time_window_start, time_window_target, num=timesliceCount)

    low_res_grid_sampling = float(config.dataset.low_res_grid_sampling)
    up = int(config.dataset.UPsampling)
    max_steps = int(config.pcds.max_iterations)
    flowline_dt = float(config.pcds.dt)
    offset_dist = float(config.pcds.offset_dist)
    LstepsPerline = int(config.pcds.sampled_points_per_line)
    localized = bool(config.pcds.localized)

    # 缓存键
    key_str = (
        f"ftle_upsampling_test|vectorfield={test_vectorfield_name}|"
        f"timesliceCount={timesliceCount}|UPsampling={up}|"
        f"lowResGridIntervalScale={low_res_grid_sampling}|"
        f"time_window_start_ratio={time_window_start_ratio}|time_window_target_ratio={time_window_target_ratio}|"
        f"max_steps={max_steps}|dt={flowline_dt:.8g}|offset_dist={offset_dist:.8g}|"
        f"LstepsPerline={LstepsPerline}|localized={localized}"
    )
    tag = "FTLEUpsamplingTestDataset_" + hashlib.md5(key_str.encode("utf-8")).hexdigest()[:16]
    cache_dir = os.path.join("./outputs", "temp")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{tag}.npz")

    lowResFTLE_list = []
    highResFTLE_list = []
    lowResPathlines_list = []

    # 先尝试加载缓存
    if os.path.exists(cache_path):
        try:
            data = np.load(cache_path)
            lowResFTLE_all = data["LowResFTLE"]
            highResFTLE_all = data["HighResFTLE"]
            lowResPathlines_np = data["LowResPathlines"]
            # 转回 torch，后续测试流程对 pathlines 以 torch 进行索引
            lowResPathlines_all = torch.from_numpy(lowResPathlines_np).float()
            return lowResFTLE_all, lowResPathlines_all, highResFTLE_all, vectorfield
        except Exception as e:
            print(f"[build_test_dataset] cache load failed: {e}. Regenerating...")

    # 生成数据
    for time_slice in sample_times:
        # 低分辨率切片与对应流线
        low_resFTLE_field, lowResPathlines, low_res_xs, low_res_ys = generate_FTLE_SLICE(
            config, vectorfield, float(time_slice), flowline_dt, max_steps, low_res_grid_sampling
        )
        # 高分辨率（作为 GT）
        high_res_sampling = up * low_res_grid_sampling
        high_resFTLE_field, _, high_res_xs, high_res_ys = generate_FTLE_SLICE(
            config, vectorfield, float(time_slice), flowline_dt, max_steps, high_res_sampling
        )

        # 与训练一致的预处理：时间下采样与归一化（不做 FTLE 归一化）
        temporal_sampled_P_all = temporal_downsamplePathlineCrossPrimitive(lowResPathlines, int(LstepsPerline))
        # 训练集中此处使用了常量 5 作为 cross neighborsize，这里保持一致
        lowResPathlinesPreprocessed = preprocess_localization_normalization(
            temporal_sampled_P_all, 5, int(LstepsPerline), bool(localized), False
        ).cpu().float()

        lowResFTLE_list.append(torch.from_numpy(low_resFTLE_field).float())
        highResFTLE_list.append(torch.from_numpy(high_resFTLE_field).float())
        lowResPathlines_list.append(lowResPathlinesPreprocessed)

    # 堆叠
    lowResFTLE_all_t = torch.stack(lowResFTLE_list, dim=0)
    highResFTLE_all_t = torch.stack(highResFTLE_list, dim=0)
    lowResPathlines_all = torch.stack(lowResPathlines_list, dim=0)

    # 保存缓存（使用 float32）
    try:
        np.savez(
            cache_path,
            LowResFTLE=lowResFTLE_all_t.detach().cpu().numpy().astype(np.float32),
            HighResFTLE=highResFTLE_all_t.detach().cpu().numpy().astype(np.float32),
            LowResPathlines=lowResPathlines_all.detach().cpu().numpy().astype(np.float32),
            meta=key_str,
        )
    except Exception as e:
        print(f"[build_test_dataset] cache save failed: {e}")

    # 返回 numpy（FTLE）+ torch（Pathlines），与测试流程使用方式一致
    return (
        lowResFTLE_all_t.detach().cpu().numpy().astype(np.float32),
        lowResPathlines_all,
        highResFTLE_all_t.detach().cpu().numpy().astype(np.float32),
        vectorfield,
    )
def test_UpsamplingModel(config, model, device,visualize=False):
    with torch.no_grad():
        model.to(device).eval()

        # helper: compute starts so that last window touches boundary (may overlap previous)
        def _tiling_starts(length: int, k: int, stride: int):
            if k >= length:
                return [0]
            s = max(1, int(stride))
            starts = list(range(0, length - k + 1, s))
            last = length - k
            if starts[-1] != last:
                starts.append(last)
            return starts

        # 使用带缓存的数据集
        lowResFTLE_all, lowResPathlines_all, highResFTLE_all, vectorfield = build_test_dataset(config)

        patch_size = int(getattr(config.dataset, 'patchSize', 32))
        patch_stride = int(getattr(config.dataset, 'patchStride', 2))
        LstepsPerline = int(config.pcds.sampled_points_per_line)

        mse_sum = 0.0
        mae_sum = 0.0
        maxe_sum = 0.0
        psnr_sum = 0.0
        sample_count = int(lowResFTLE_all.shape[0])

        for test_i in range(sample_count):
            low_resFTLE_field = lowResFTLE_all[test_i]
            high_resFTLE_field = highResFTLE_all[test_i]
            lowResPathlinesPreprocessed = lowResPathlines_all[test_i]

            # normalization like training（以训练集统计量归一化低分辨率输入）
            ftle_min = float(config.ftle_min)
            ftle_max = float(config.ftle_max)
            low_resFTLE_field_clip = np.clip(low_resFTLE_field, ftle_min, ftle_max)
            low_resFTLE_field_norm = (low_resFTLE_field_clip - ftle_min) / max(1e-12, (ftle_max - ftle_min))

            ny_low, nx_low = low_resFTLE_field.shape
            ny_hi, nx_hi = high_resFTLE_field.shape
            ry = float(ny_hi) / float(max(1, ny_low))
            rx = float(nx_hi) / float(max(1, nx_low))

            row_starts = _tiling_starts(ny_low, patch_size, patch_stride)
            col_starts = _tiling_starts(nx_low, patch_size, patch_stride)

            pred_grid = np.zeros((ny_hi, nx_hi), dtype=np.float32)
            weight_grid = np.zeros((ny_hi, nx_hi), dtype=np.float32)

            for i0 in row_starts:
                i1 = min(i0 + patch_size, ny_low)
                for j0 in col_starts:
                    j1 = min(j0 + patch_size, nx_low)

                    # map to high-res
                    hi_h = max(1, int(round((i1 - i0) * ry)))
                    hi_w = max(1, int(round((j1 - j0) * rx)))
                    hi_i0 = int(round(i0 * ry))
                    hi_j0 = int(round(j0 * rx))
                    if i0 == row_starts[-1]:
                        hi_i0 = ny_hi - hi_h
                    if j0 == col_starts[-1]:
                        hi_j0 = nx_hi - hi_w
                    hi_i1 = hi_i0 + hi_h
                    hi_j1 = hi_j0 + hi_w

                    # build inputs
                    lr_patch_norm = torch.from_numpy(low_resFTLE_field_norm[i0:i1, j0:j1]).unsqueeze(0).to(device).float()

                    # select corresponding pathlines groups
                    idx_list = []
                    for rr in range(i0, i1):
                        base = rr * nx_low
                        for cc in range(j0, j1):
                            idx_list.append(base + cc)
                    idx_tensor = torch.as_tensor(idx_list, dtype=torch.long)
                    pl_patch = lowResPathlinesPreprocessed[idx_tensor].unsqueeze(0).to(device).float()

                    # forward
                    pred_patch = model(lr_patch_norm, pl_patch).to(device).float()  # [1, hi_h, hi_w]
                    # inverse normalization
                    pred_patch = pred_patch * (ftle_max - ftle_min) + ftle_min
                    patch_np = pred_patch.squeeze(0).detach().cpu().numpy()
                    pred_grid[hi_i0:hi_i1, hi_j0:hi_j1] += patch_np
                    weight_grid[hi_i0:hi_i1, hi_j0:hi_j1] += 1.0

            # metrics (raw scale)
            weight_grid = np.clip(weight_grid, 1.0, None)
            pred_grid = pred_grid / weight_grid

            label_y_b = high_resFTLE_field.astype(np.float32)
            pred_b = pred_grid.astype(np.float32)
            mse, mae, maxe, psnr = compute_metrics(label_y_b, pred_b)

            if visualize and test_i == sample_count-1:
                visualize_FTLEUpampling(label_y_b, pred_b, low_resFTLE_field, vectorfield.domainMinBoundary, vectorfield.domainMaxBoundary)
                
            mse_sum += mse
            mae_sum += mae
            maxe_sum += maxe
            psnr_sum += psnr

        mse = mse_sum / sample_count
        mae = mae_sum / sample_count
        maxe = maxe_sum / sample_count
        psnr = psnr_sum / sample_count
        print(f"test average: mse={mse:.6f}, mae={mae:.6f}, maxe={maxe:.6f}, psnr={psnr:.6f}")
        return {"mse": mse, "mae": mae, "maxe": maxe, "psnr": psnr}

     


def train_model(config, model, dataset, device):
    optimizer = build_optimizer_from_cfg(model, lr=config.lr, **config.optimizer)
    loss_fn = build_criterion_from_cfg(config.loss)
    batch_size = int(config.batch_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(getattr(config, 'num_workers', 0)),
        pin_memory=False,
        drop_last=True
    )
    print_freq = int(config.print_freq)
    epochs = int(config.epochs)
    LOSS_NAME = config.loss.NAME
    best_psnr = float('-inf')
    best_state_dict = None
    model.to(device).train()

    # new: loss history for lr scheduling
    loss_history = []
    patience = 4  # how many epochs to wait for loss to decrease
    min_delta = 1e-6  # minimum change in loss to be considered as improvement
    last_lr = optimizer.param_groups[0]['lr']

    if hasattr(config, 'test_tasks'):
        test_task_func_name=config['test_tasks']
        task_init_fn=eval(test_task_func_name)
        assert task_init_fn is not None and callable(task_init_fn)
    else:
        task_init_fn=None
    
    total_iterations=0
    for epoch in range(epochs):
        epoch_avg_loss = 0.0
        for it, (Pk, label_y) in enumerate(loader):
            # Pk: [B, nerb*K, 3]; reshape to [B, 3, nerb*K]
            label_y = label_y.to(device).float()
            if isinstance(Pk, tuple) or isinstance(Pk, list):
                input1 = Pk[0].to(device)
                input2 = Pk[1].to(device)
                pred = model(input1, input2).to(device).float()
            else:
                input1 = Pk.to(device)
                pred = model(input1).to(device).float()

            if torch.isnan(pred).any() or torch.isinf(pred).any():
                print(f"Warning: nan or inf in pred at epoch {epoch}, iter {it}")
                pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)

            loss = loss_fn(pred, label_y)
            optimizer.zero_grad()
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_avg_loss += float(loss.item())
            #clean Gpu memory
            torch.cuda.empty_cache()
            total_iterations += 1
            if it % print_freq == 0 or it == 0:
                print(f"epoch {epoch}, iter {it}: {LOSS_NAME} ={loss.item():.6f}, total_iterations: {total_iterations}")
                if config['wandb']:
                    wandb.log({"epoch": epoch,  "train_loss": loss.item(), "total_iterations": total_iterations})

        ##########################################################
        ################ #operatons per epoch ####################
        ##########################################################
        steps = max(1, len(loader))
        epoch_avg_loss /= steps
        loss_history.append(epoch_avg_loss)

        # simple lr scheduler: if loss does not decrease for patience epochs, halve the lr
        if len(loss_history) > patience:
            recent_losses = loss_history[-patience-1:]
            # check if the recent patience+1 losses are monotonically non-decreasing (i.e., loss did not decrease)
            is_stable = all(recent_losses[i] >= recent_losses[i-1] - min_delta for i in range(1, len(recent_losses)))
            if is_stable:
                new_lr = last_lr * 0.5
                if new_lr <1e-7:
                    print(f"[lr scheduler] epoch {epoch}: lr is too small, stop training")
                    break
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                print(f"[lr scheduler] epoch {epoch}: loss does not decrease for {patience} epochs, learning rate adjusted to {new_lr:.6g}")
                last_lr = new_lr
                # to avoid multiple triggers, clear loss_history, only keep the latest one
                loss_history = [loss_history[-1]]
        if task_init_fn is not None and callable(task_init_fn):
            Res = task_init_fn(config, model, device=str(device)) 
            cur_psnr = float(Res['psnr'])
            if cur_psnr > best_psnr:
                best_psnr = cur_psnr
                best_state_dict = copy.deepcopy(model.state_dict())
                print(f"[best] epoch {epoch}: psnr={best_psnr:.2f} dB (checkpoint updated)")
            print(f"epoch {epoch}: {LOSS_NAME} ={epoch_avg_loss:.6f}. Test:mse={Res['mse']:.6f}, \
            mae={Res['mae']:.6f}, maxe={Res['maxe']:.6f}, psnr={cur_psnr:.2f} dB") if Res is not None else print(f"epoch {epoch}: {LOSS_NAME} ={epoch_avg_loss:.6f}")
            if config['wandb']:
                wandb.log({"epoch": epoch,  "epoch_Loss": epoch_avg_loss, "test_mse": Res['mse'], "test_mae": Res['mae'], "test_maxe": Res['maxe'], "test_psnr": cur_psnr})
        else:
            print(f"epoch {epoch}: {LOSS_NAME} ={epoch_avg_loss:.6f}")
            if config['wandb']:
                    wandb.log({"epoch": epoch,  "epoch_Loss": epoch_avg_loss, "total_iterations": total_iterations})


    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"[final] loaded best checkpoint with best psnr={best_psnr:.2f} dB")
        test_result= task_init_fn(config, model, device=str(device), visualize=True) if task_init_fn is not None and callable(task_init_fn) else None
        if config['wandb'] and test_result is not None and isinstance(test_result,dict):            
            wandb.summary.update({"best_test_mse": test_result['mse'],"best_test_psnr": test_result['psnr']})
            for key, value in test_result.items():
                wandb.summary.update({key: value})
            wandb.finish()

      




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
    else:   
        raise ValueError(f"Unknown model: {config.model.NAME}")


def runNameTagGenerator_fmt(config,mode)->Tuple[str, List[str]]:
    seed=config['seed']
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    runName = f"{config['model']['NAME']}_bs_{config['batch_size']}_ep_{config['epochs']}_lr_{config['lr']}_{current_time}_seed_{seed}"
    tagGen0=mode
    tagGen1=config['model']['NAME']
    runTags= [tagGen0,tagGen1]
    return runName,runTags

if __name__=="__main__":
    print("PyTorch version:", torch.__version__)
    print("torch.cuda.is_available():",torch.cuda.is_available())  # Should return True
    print("torch.version.cuda:",torch.version.cuda)         # Should print the CUDA version PyTorch was built with
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # mode = 'point_FTLE_regression'
    mode = 'upsampling'

    cfg=argParseAndPrepareConfig()
    cfg["gitInfo"]=get_git_commit_id()
    run_Name,runTags=runNameTagGenerator_fmt(cfg,mode)
    cfg['run_name']=run_Name
    logging.info(f"run name: {run_Name}, run tags: {runTags}")

 
       
    print_args(cfg)
    # Initialize wandb
    if cfg['wandb']:
        run=wandb.init(project=GLOBAL_WANDB_PROJECT_NAME,
                    name=run_Name,
                    tags=runTags,
                    config=cfg)
    if mode == 'point_FTLE_regression':
        config = EasyConfig()
        netCDF = NetCDFLoader()
        config.load("config/PointWiseFTLERegressor.yaml", recursive=False)
        vectorfield_datapath=f"{config.dataset.dat_dir}\\{config.dataset.test_name}.{config.dataset.extension}"
        vectorfield = netCDF.load_vector_field2d(vectorfield_datapath)
        config['vectorfield']=vectorfield
   
        model = build_model(config, device)
        # use Dataset + DataLoader (support shuffle / multi-threading etc.)
        dataset = PointWiseFTLETrainDataset( config=config )
        config['ftle_min']=dataset.ftle_min
        config['ftle_max']=dataset.ftle_max
        train_model(config,model,dataset,device)
    elif mode == 'upsampling':
        # future mode: low resolution pathlines + low resolution FTLE -> high resolution FTLE
        dataset = FTLEUpsamplingTrainDataset(cfg, useCacheSystem=True)
        lowResX,lowResY=dataset.lowResFTLE[0].shape[0],dataset.lowResFTLE[0].shape[1]
        cfg['lowResX']=lowResX
        cfg['lowResY']=lowResY
        cfg['ftle_min']=dataset.ftle_min
        cfg['ftle_max']=dataset.ftle_max
        model = build_model(cfg, device)
        train_model(cfg,model,dataset,device)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    

