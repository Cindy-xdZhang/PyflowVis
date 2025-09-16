#step 1: load FTLE dataset
import os,random
import copy
import numpy as np
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

torch.backends.cuda.matmul.allow_tf32 = False  


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
    sampling lowResPathlines based on lowResFTLE
    Args:
        lowResFTLE:        [B, X, Y]
        lowResPathlines:   [B, X*Y, nerbors, max_steps, Dim]
        sampling_ratio:    0~1 

    Returns:
        sampled_pathlines: [B, M, nerbors, max_steps, Dim], 其中 M = floor(X*Y*ratio) 且 M>=1
        sampled_indices:   [B, M]，在每个 batch 内对应的线性索引（0..X*Y-1）
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



class FTLEUpsamplingModel(nn.Module):
    def __init__(self, cfg, lowResX, lowResY, num_stages=2, embed_dim=72, k_neighbors=6, cross_neighborsize=5,beta=100, alpha=1000):
        super().__init__()
        #FTLERegressor is a point wise regressor, has no cross primitive, no neighbor information aggregation
        LStpesPerline=cfg.pcds.sampled_points_per_line
        self.cross_neighborsize=5
        self.lowResX=lowResX
        self.lowResY=lowResY
        self.pointsPerUnit=int(LStpesPerline*cross_neighborsize*lowResX*lowResY*0.25)
        self.encoder = EncNP(self.pointsPerUnit, num_stages, embed_dim, k_neighbors, alpha, beta)

        self.decoderInputDim=embed_dim * (2 ** (num_stages - 0))
        # self.decoderInputDim=embed_dim * (2 ** (num_stages - 0))+ 3*2*cross_neighborsize
        # self.decoderInputDim=3*2*cross_neighborsize
        
        self.MLP1=nn.Sequential(
            nn.Linear(self.decoderInputDim, self.decoderInputDim*2),
            nn.ReLU(),
            nn.Linear(self.decoderInputDim*2, self.decoderInputDim),
            nn.ReLU(),
            nn.Linear(self.decoderInputDim,15),
            )
        #deconve upsampling ftle from lowREs to highRes            
        self.DECONV1=nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2)
        self.DECONV2=nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2)
        self.DECONV3=nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2)
        
        
    
    #todo:better option is to sample lowResPathlines based on lowResFTLE
    def forward(self, lowResFTLE: torch.Tensor, lowResPathlines: torch.Tensor):
        B,lowResX,lowResY=lowResFTLE.shape
        _,lowResXtimeslowResY,nerbors,max_steps,Dim=lowResPathlines.shape
        sampled_pathlines, sampled_indices=sampling_lowResPathlines_based_on_lowResFTLE(lowResFTLE, lowResPathlines,0.25)
        PointsWholeFieldRaw=sampled_pathlines.reshape(B, self.pointsPerUnit,Dim)



        points_3N=PointsWholeFieldRaw.permute(0, 2, 1)
        points_N3=PointsWholeFieldRaw
        globalFMT_feat = self.encoder(points_N3, points_3N)

        #shape of wholeFieldFMT_feat=[B, 15]
        wholeFieldFMT_feat=self.decoder(globalFMT_feat).repeat(1,1,lowResX,lowResY).permute(0, 2, 3, 1)
        lowResFTLE = lowResFTLE.reshape(B, lowResX, lowResY, 1)
        concat_feat = torch.cat([wholeFieldFMT_feat, lowResFTLE], dim=-1)
        upsampled_ftle=self.DECONV1(concat_feat)
        upsampled_ftle=self.DECONV2(upsampled_ftle)
        upsampled_ftle=self.DECONV3(upsampled_ftle)
        pred=upsampled_ftle.reshape(B, -1, 1)
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


def test_UpsamplingModel(config, dataset,model, device):
    with torch.no_grad():
        model.to(device).eval()
        sample_count=5
        mse_sum=0.0
        mae_sum=0.0
        maxe_sum=0.0
        psnr_sum=0.0
        #random pick 5 samples for testing
        test_samples = random.sample(range(len(dataset)), sample_count)
        for test_i,i in enumerate(test_samples):
            Pk, label_y = dataset[i]
            if isinstance(Pk, tuple) or isinstance(Pk, list):
                #create batch dimension
                LowResFTLE= Pk[0].unsqueeze(0).to(device)
                LowResPathlines = Pk[1].unsqueeze(0).to(device)
                pred = model(LowResFTLE, LowResPathlines).to(device).float()
            else:
                input1 = Pk.unsqueeze(0).to(device)
                pred = model(input1).to(device).float()
            #invert back ftle to original scale, then compute metrics 
            pred_b=pred_b.cpu().squeeze(0)
            label_y_b = label_y*(dataset.ftle_max-dataset.ftle_min)+dataset.ftle_min
            pred_b = pred*(dataset.ftle_max-dataset.ftle_min)+dataset.ftle_min

            label_y_b=label_y_b.cpu().numpy()
            pred_b=pred_b.cpu().numpy()
            mse, mae, maxe, psnr = compute_metrics(label_y_b, pred_b)
            print(f"test {test_i}: mse={mse:.6f}, mae={mae:.6f}, maxe={maxe:.6f}, psnr={psnr:.6f}")
            if test_i == 0:
                visualize_ftle_sliceComparison(label_y_b, pred_b, config['test_vectorfield'].domainMinBoundary, config['test_vectorfield'].domainMaxBoundary,psnr)
            mse_sum += mse
            mae_sum += mae
            maxe_sum += maxe
            psnr_sum += psnr
        mse = mse_sum / sample_count
        mae = mae_sum / sample_count
        maxe = maxe_sum / sample_count
        psnr = psnr_sum / sample_count
        print(f"test average: mse={mse:.6f}, mae={mae:.6f}, maxe={maxe:.6f}, psnr={psnr:.6f}")
        return {
            "mse": mse,
            "mae": mae,
            "maxe": maxe,
            "psnr": psnr
        }

     


def train_model(config, model, dataset, device):
    optimizer = build_optimizer_from_cfg(model, lr=config.lr, **config.optimizer)
    loss_fn = build_criterion_from_cfg(config.loss)
    batch_size = int(config.bs)
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
            if it % print_freq == 0 and it > 0:
                print(f"epoch {epoch}, iter {it}: {LOSS_NAME} ={loss.item():.6f}")

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
        else:
            print(f"epoch {epoch}: {LOSS_NAME} ={epoch_avg_loss:.6f}")

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"[final] loaded best checkpoint with best psnr={best_psnr:.2f} dB")
        task_init_fn(config, model, device=str(device), visualize=True) if task_init_fn is not None and callable(task_init_fn) else None




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
    else:   
        raise ValueError(f"Unknown model: {config.model.NAME}")


if __name__=="__main__":
    print("PyTorch version:", torch.__version__)
    print("torch.cuda.is_available():",torch.cuda.is_available())  # Should return True
    print("torch.version.cuda:",torch.version.cuda)         # Should print the CUDA version PyTorch was built with
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # mode = 'point_FTLE_regression'
    mode = 'upsampling'

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
        config = EasyConfig()
        config.load("config/FTLEUpsampling.yaml", recursive=False)
        dataset = FTLEUpsamplingTrainDataset(config, useCacheSystem=True)
        lowResX,lowResY=dataset.lowResFTLE[0].shape[0],dataset.lowResFTLE[0].shape[1]
        model = UpsamplingUnetModel(config, lowResX, lowResY,upscale=int(config.dataset.UPsampling))
        train_model(config,model,dataset,device)
        test_UpsamplingModel(config,dataset,model,device)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    

