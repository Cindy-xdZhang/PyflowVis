#step 1: load FTLE dataset
import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from DeepUtils.utils import EasyConfig
from DeepUtils.loss import build_criterion_from_cfg
from DeepUtils.optim import build_optimizer_from_cfg

from FLowUtils.VectorField2d import *
from pnn.models.point_nn import EncNP
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
    
class PointWiseFTLERegressor(nn.Module):
    def __init__(self, LStpesPerline,num_stages=2, embed_dim=72, k_neighbors=6, cross_neighborsize=5,beta=100, alpha=1000):
        super().__init__()
        #FTLERegressor is a point wise regressor, has no cross primitive, no neighbor information aggregation
        self.cross_neighborsize=cross_neighborsize
        self.pointsPerPrimitive=LStpesPerline*cross_neighborsize
        self.encoder = EncNP(self.pointsPerPrimitive, num_stages, embed_dim, k_neighbors, alpha, beta)

        # self.decoderInputDim=embed_dim * (2 ** (num_stages - 0))+ 3*2*cross_neighborsize
        self.decoderInputDim=embed_dim * (2 ** (num_stages - 0))
        # self.decoderInputDim=3*2*cross_neighborsize
        self.decoder = Decoder(in_dim=self.decoderInputDim)

    def forward(self, pts: torch.Tensor):
        B,CrossSize,LstepsPerline,Dim=pts.shape
        PointsRaw=pts.reshape(B,CrossSize*LstepsPerline,Dim)
        points_3N=PointsRaw.permute(0, 2, 1)
        points_N3=PointsRaw
        FMT_feat = self.encoder(points_N3, points_3N)

        # xyz = pts.permute(0, 2, 1)#B,N,3
        #feat: (B, embed_dim)
        # start_pos=pts[:,:,0,:].reshape(B, -1)
        # end_pos=pts[:,:,-1,:].reshape(B, -1)
        # every_cross_feature=torch.cat([start_pos,end_pos],dim=1)

        #feature dimension is in_dim*K(steps per line)
        pred = self.decoder(FMT_feat)
        # pred = self.decoder(every_cross_feature)
        return pred
    





class FTLEUpsamplingModel(nn.Module):
    def __init__(self, LStpesPerline,num_stages=2, embed_dim=72, k_neighbors=6, cross_neighborsize=5,beta=100, alpha=1000):
        super().__init__()
        #FTLERegressor is a point wise regressor, has no cross primitive, no neighbor information aggregation
        self.cross_neighborsize=cross_neighborsize
        self.pointsPerPrimitive=LStpesPerline*cross_neighborsize
        self.encoder = EncNP(self.pointsPerPrimitive, num_stages, embed_dim, k_neighbors, alpha, beta)

        # self.decoderInputDim=embed_dim * (2 ** (num_stages - 0))+ 3*2*cross_neighborsize
        # self.decoderInputDim=embed_dim * (2 ** (num_stages - 0))
        self.decoderInputDim=3*2*cross_neighborsize
        self.decoder = Decoder(in_dim=self.decoderInputDim)

    def forward(self, pts: torch.Tensor):
        B,CrossSize,LstepsPerline,Dim=pts.shape
        PointsRaw=pts.reshape(B,CrossSize*LstepsPerline,Dim)
        points_3N=PointsRaw.permute(0, 2, 1)
        points_N3=PointsRaw
        # FMT_feat = self.encoder( points_N3, points_3N)


        # xyz = pts.permute(0, 2, 1)#B,N,3
        #feat: (B, embed_dim)
        start_pos=pts[:,:,0,:].reshape(B, -1)
        end_pos=pts[:,:,-1,:].reshape(B, -1)

        every_cross_feature=torch.cat([start_pos,end_pos],dim=1)

        #feature dimension is in_dim*K(steps per line)
        # pred = self.decoder(FMT_feat)
        pred = self.decoder(every_cross_feature)
        return pred
    

    

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

    test_task_func_name=config['test_tasks']
    task_init_fn=eval(test_task_func_name)
    assert task_init_fn is not None and callable(task_init_fn)
    

    for epoch in range(epochs):
        epoch_avg_loss = 0.0
        for it, (Pk, label_y) in enumerate(loader):
            # Pk: [B, nerb*K, 3]; reshape to [B, 3, nerb*K]
            points = Pk.to(device)
            label_y = label_y.to(device).float()

            pred = model(points).to(device).float()
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

        Res = task_init_fn(config, model, device=str(device))
        cur_psnr = float(Res['psnr'])
        if cur_psnr > best_psnr:
            best_psnr = cur_psnr
            best_state_dict = copy.deepcopy(model.state_dict())
            print(f"[best] epoch {epoch}: psnr={best_psnr:.2f} dB (checkpoint updated)")
        print(f"epoch {epoch}: {LOSS_NAME} ={epoch_avg_loss:.6f}. Test:mse={Res['mse']:.6f}, mae={Res['mae']:.6f}, maxe={Res['maxe']:.6f}, psnr={cur_psnr:.2f} dB")

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"[final] loaded best checkpoint with best psnr={best_psnr:.2f} dB")
        task_init_fn(config, model, device=str(device), visualize=True)


if __name__=="__main__":
    print("PyTorch version:", torch.__version__)
    print("torch.cuda.is_available():",torch.cuda.is_available())  # Should return True
    print("torch.version.cuda:",torch.version.cuda)         # Should print the CUDA version PyTorch was built with
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mode = 'point_regression'

    if mode == 'point_regression':
        config = EasyConfig()
        netCDF = NetCDFLoader()
        config.load("config/PointWiseFTLERegressor.yaml", recursive=False)
        vectorfield_datapath=f"{config.dataset.dat_dir}\\{config.dataset.name}.{config.dataset.extension}"
        vectorfield = netCDF.load_vector_field2d(vectorfield_datapath)
        config['vectorfield']=vectorfield
        nerb = int(config.pcds.num_cross_points_per_seeding)
        L = int(config.pcds.sampled_points_per_line)
        #the input point cloud tensor runing in the network is (B,  N=K*nerb, 3)
        model = PointWiseFTLERegressor(LStpesPerline=L,num_stages=config.pnn.stages, embed_dim=config.pnn.dim,
                              k_neighbors=config.pnn.k, beta=config.pnn.beta, alpha=config.pnn.alpha).to(device)

      
        # use Dataset + DataLoader (support shuffle / multi-threading etc.)
        dataset = PointWiseFTLETrainDataset( config=config, vectorfield=vectorfield   )
        config['ftle_min']=dataset.ftle_min
        config['ftle_max']=dataset.ftle_max
        train_model(config,model,dataset,device)

    elif mode == 'upsampling':
        # future mode: low resolution pathlines + low resolution FTLE -> high resolution FTLE
        config = EasyConfig()
        config.load("config/FTLEUpsampling.yaml", recursive=False)
        dataset = FTLEUpsamplingTrainDataset(config, useCacheSystem=True)
        model = FTLEUpsamplingModel(config)
        train_model(config,model,dataset,device)

    else:
        raise ValueError(f"Unknown mode: {mode}")
    

