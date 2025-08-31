#step 1: load FTLE dataset
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from DeepUtils.utils import EasyConfig
from DeepUtils.loss import build_criterion_from_cfg
from DeepUtils.optim import build_optimizer_from_cfg

from FLowUtils.ScalarField2d import ScalarField2D
from FLowUtils.VectorField2d import *
from FLowUtils.netCDFLoader import *
from pnn.models.point_nn import EncNP
from FTLE_fitting_utils import *

torch.backends.cuda.matmul.allow_tf32 = False  


class Decoder(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.ln1=nn.Linear(in_dim, 128)
        self.bn1=nn.BatchNorm1d(128)
        self.ln2=nn.Linear(128, 256)
        self.bn2=nn.BatchNorm1d(256)
        self.ln3=nn.Linear(256, 64)
        self.bn3=nn.BatchNorm1d(64)
        self.ln4=nn.Linear(64, 1)
    def forward(self, x):
        x = x.contiguous()
        x = self.ln1(x.contiguous())
        x = self.bn1(x)
        x = nn.GELU()(x)

        x = self.ln2(x)
        x = self.bn2(x)
        x = nn.GELU()(x)

        x = self.ln3(x)
        x = self.bn3(x)
        x = nn.GELU()(x)

        x = self.ln4(x)
        x = nn.Sigmoid()(x)
        return x.squeeze(-1)
    
class FTLERegressor(nn.Module):
    def __init__(self, LStpesPerline,num_stages=2, embed_dim=72, k_neighbors=6, cross_neighborsize=5,beta=100, alpha=1000):
        super().__init__()
        #FTLERegressor is a point wise regressor, has no cross primitive, no neighbor information aggregation
        self.cross_neighborsize=cross_neighborsize
        self.pointsPerPrimitive=LStpesPerline*cross_neighborsize
        self.encoder = EncNP(self.pointsPerPrimitive, num_stages, embed_dim, k_neighbors, alpha, beta)

        # self.decoderInputDim=embed_dim * (2 ** (num_stages - 0))+ 3*2*cross_neighborsize
        self.decoderInputDim=3*2*cross_neighborsize
        self.decoder = Decoder(in_dim=self.decoderInputDim)

    def forward(self, pts: torch.Tensor):
        B,CrossSize,LstepsPerline,Dim=pts.shape

        # xyz = pts.permute(0, 2, 1)#B,N,3
        # feat = self.encoder(xyz, pts)
        #feat: (B, embed_dim)
        start_pos=pts[:,:,0,:].reshape(B, -1)
        end_pos=pts[:,:,-1,:].reshape(B, -1)

        every_cross_feature=torch.cat([start_pos,end_pos],dim=1)

        #feature dimension is in_dim*K(steps per line)
        pred = self.decoder(every_cross_feature)
        return pred
    
def test_ftle_from_model(cfg,model: nn.Module, vectorfield: UnsteadyVectorField2D, time:float, device: str = "cuda", save_path: str | None = None,
                        starts_chunk: int = 64):
    #first generate grid points for this time, then generate pathlines, 
    # then call computeFTLEFromPathlineCrossPrimitive get correct ftle
    # then compare the ftle from model and the correct ftle
    #report the error
    nerb = int(cfg.pcds.num_cross_points_per_seeding)
    LstepsPerline = int(cfg.pcds.sampled_points_per_line)
    max_steps = int(cfg.pcds.max_iterations)
    dt = float(cfg.pcds.dt)
    offset_dist = vectorfield.gridInterval[0] * float(cfg.pcds.offset_scale)
    localized = bool(cfg.pcds.localized)
    normalized = bool(cfg.pcds.normalized)

    # 时间设置：以输入的 physical time 为起始，目标时间与训练保持相同的时间跨度
    tmin, tmax = float(vectorfield.tmin), float(vectorfield.tmax)
    base_t_target_ratio = float(cfg.pcds.t_target)

    physical_start = float(time)
    physical_target = float(np.clip(tmin + base_t_target_ratio * (tmax - tmin), tmin, tmax))

    starts_xy,xs,ys = generate_test_seeds(cfg,vectorfield)
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
    ftle_min=cfg['ftle_min']
    ftle_max=cfg['ftle_max']

    with torch.no_grad():
        model.eval()
        for s0 in range(0, M_all, max(1, int(starts_chunk))):
            s1 = min(M_all, s0 + int(starts_chunk))
            starts_xy_b = starts_xy[s0:s1]

            Pathline_b, PathlineLength_b = batch_pathlineCross_integration_2D_auto(
                points=starts_xy_b,
                vectorfield=vectorfield,
                t_start=float(physical_start), t_target=float(physical_target),
                dt=float(dt), max_steps=int(max_steps),
                offsets_size=float(offset_dist), method="rk4"
            )

            assert Pathline_b.shape[0] % nerb == 0, "lines count must be multiple of nerbors"
            GroupSize_BatchSize = Pathline_b.shape[0] // nerb
            Pathline_g = Pathline_b.view(GroupSize_BatchSize, nerb, max_steps, -1)
            PathlineLength_g = PathlineLength_b.view(GroupSize_BatchSize, nerb)
            keep_groups_full = (PathlineLength_g == max_steps).all(dim=1)  # [GroupSize_BatchSize]
            # 重采样到 LstepsPerline（对所有组）
            P_K = temporal_downsamplePathlineCrossPrimitive(Pathline_g, LstepsPerline) #P_K.view(G_b, nerb, LstepsPerline, 3)
            # 计算真值 FTLE，并将非满长组置零
            if not (keep_groups_full).any():
                continue
            y_true_b = computeFTLEFromPathlineCrossPrimitive(Pathline_g, sample_nerbors=nerb, line_steps=LstepsPerline, vectorfield=vectorfield)
       
            y_valid = y_true_b[keep_groups_full]
            y_global_min = min(y_global_min, float(y_valid.min().item()))
            y_global_max = max(y_global_max, float(y_valid.max().item()))
            y_valid = y_valid.to(device).float()


            # 预处理并分批预测（必要时 pad 到 64 的倍数）
            P_in = preprocess_localization_normalization(P_K, nerb, LstepsPerline, localized, normalized, vectorfield).to(device).float()
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

            # 误差累计（仅统计原始 B 条）
            diff = valid_pred_b- y_valid
            se_sum += float((diff ** 2).sum().item())
            ae_sum += float(diff.abs().sum().item())
            max_abs_err = max(max_abs_err, float(diff.abs().max().item()))
            total_groups += int(GroupSize_BatchSize)

            # 回填网格（对本批所有组），非满长组真值已置零
            idx_global = (np.arange(GroupSize_BatchSize) + s0)
            valid_idx_global = idx_global[keep_groups_full]
            rows = valid_idx_global // nx
            cols = valid_idx_global % nx
            true_grid[rows, cols] = y_valid.detach().cpu().numpy()
            pred_grid[rows, cols] = valid_pred_b.detach().cpu().numpy()

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


if __name__=="__main__":
    print("PyTorch version:", torch.__version__)
    print("torch.cuda.is_available():",torch.cuda.is_available())  # Should return True
    print("torch.version.cuda:",torch.version.cuda)         # Should print the CUDA version PyTorch was built with
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
        total_points_count =getattr(config, 'train_points_count', 640*80*4)
        nerb = int(config.pcds.num_cross_points_per_seeding)
        L = int(config.pcds.sampled_points_per_line)
        #the input point cloud tensor runing in the network is (B,  N=K*nerb, 3)
        model = FTLERegressor(LStpesPerline=L,num_stages=config.pnn.stages, embed_dim=config.pnn.dim,
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
        
        # 使用 Dataset + DataLoader（支持 shuffle / 多线程等）
        dataset = FTLETrainDataset(
            cfg=config,
            vectorfield=vectorfield,
            count=total_points_count,
            max_steps=max_steps,
            dt=dt,
            t_start=t_start,
            t_target=t_target,
            offset_dist=offset_dist,
            nerbors=nerb,
            LstepsPerline=L,
            localized=bool(config.pcds.localized),
            normalized=bool(config.pcds.normalized),
        )
        config['ftle_min']=dataset.ftle_min
        config['ftle_max']=dataset.ftle_max

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=int(getattr(config, 'num_workers', 0)),
            pin_memory=False,
            drop_last=True
        )

        for epoch in range(epochs):
            epoch_avg_loss = 0.0
            for it, (Pk, label_y) in enumerate(loader):
                # Pk: [B, nerb*K, 3]; 转为 [B, 3, nerb*K]
                points = Pk.to(device)
                label_y = label_y.to(device).float()

                pred = model(points).to(device).float()
                if torch.isnan(pred).any() or torch.isinf(pred).any():
                    print(f"Warning: nan or inf in pred at epoch {epoch}, iter {it}")
                    pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)

                loss = loss_fn(pred, label_y)
                optimizer.zero_grad(); loss.backward()
                #gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_avg_loss += float(loss.item())

                if it % print_freq == 0 and it > 0:
                    print(f"epoch {epoch}, iter {it}: loss={loss.item():.6f}")

            steps = max(1, len(loader))
            epoch_avg_loss /= steps
            print(f"epoch {epoch}: loss={epoch_avg_loss:.6f}")
        test_ftle_from_model(config,model, vectorfield, time=t_start, device=str(device),starts_chunk=batch_size)
    elif mode == 'volume_prediction':
        # 直接学习一个体素的回归可视化原型（此处留空接口，后续可接高维解码器）
     pass


    elif mode == 'upsampling':
        # 未来模式：低分辨pathlines+低分辨FTLE -> 高分辨FTLE
        print("[upsampling] 暂留接口：准备双分支输入与监督。")
    else:
        raise ValueError(f"Unknown mode: {mode}")

