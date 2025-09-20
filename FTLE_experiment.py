#step 1: load FTLE dataset
import os,random
import logging
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
from DeepUtils.utils.stable_hash import stable_hash
from FTLE_fitting_utils import *
from DeepUtils.MiscFunctions import *
from model_zoo import *

import wandb
GLOBAL_WANDB_PROJECT_NAME="FlowMapTokenizer"

torch.backends.cuda.matmul.allow_tf32 = False  

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def build_test_dataset(config):
    """
    Generate low/high-resolution FTLE slices and preprocessed low-resolution pathlines for testing,
    with a simple caching mechanism.

    Returns:
        lowResFTLE_all: np.ndarray [T, X_low, Y_low]
        lowResPathlines_all: torch.FloatTensor [T, X_low*Y_low, nerbors, L, 3]
        highResFTLE_all: np.ndarray [T, X_high, Y_high]
        vectorfield: UnsteadyVectorField2D object (for visualization/boundary info)
    """
    # Parameter collection
    netCDF = NetCDFLoader()
    test_vectorfield_name = config['test_vectorfield'] if 'test_vectorfield' in config else config.dataset.names[0]
    vectorfield_datapath = os.path.join(config.dataset.dat_dir, f"{test_vectorfield_name}.{config.dataset.extension}")
    vectorfield = netCDF.load_vector_field2d(vectorfield_datapath)
    if vectorfield is None:
        logging.info(f"[build_test_dataset] load {test_vectorfield_name} failed. Skip this field.")
        return None, None, None, None
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

    # Cache key
    key_obj = {
        "name": "ftle_upsampling_test",
        "vectorfield": str(test_vectorfield_name),
        "timesliceCount": int(timesliceCount),
        "UPsampling": int(up),
        "lowResGridIntervalScale": float(low_res_grid_sampling),
        "time_window_start_ratio": float(time_window_start_ratio),
        "time_window_target_ratio": float(time_window_target_ratio),
        "max_steps": int(max_steps),
        "dt": float(flowline_dt),
        "offset_dist": float(offset_dist),
        "LstepsPerline": int(LstepsPerline),
        "localized": bool(localized),
    }
    tag = stable_hash(key_obj, prefix="FTLEUpsamplingTestDataset_")
    cache_dir = os.path.join(config.cache_dir, "temp")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{tag}.npz")

    lowResFTLE_list = []
    highResFTLE_list = []
    lowResPathlines_list = []

    # Try loading cache first
    if os.path.exists(cache_path):
        try:
            data = np.load(cache_path)
            lowResFTLE_all = data["LowResFTLE"]
            highResFTLE_all = data["HighResFTLE"]
            lowResPathlines_np = data["LowResPathlines"]
            # Convert back to torch; subsequent tests index pathlines using torch tensors
            lowResPathlines_all = torch.from_numpy(lowResPathlines_np).float()
            logging.info(f"[build_test_dataset] loaded {lowResFTLE_all.shape[0]} samples from cache {cache_path}")
            return lowResFTLE_all, lowResPathlines_all, highResFTLE_all, vectorfield
        except Exception as e:
            logging.info(f"[build_test_dataset] cache load failed: {e}. Regenerating...")

    # Generate data
    for time_slice in sample_times:
        # Low-resolution slice and corresponding pathlines
        low_resFTLE_field, lowResPathlines, low_res_xs, low_res_ys = generate_FTLE_SLICE(
            config, vectorfield, float(time_slice), flowline_dt, max_steps, low_res_grid_sampling
        )
        # High-resolution (as ground truth)
        high_res_sampling = up * low_res_grid_sampling
        high_resFTLE_field, _, high_res_xs, high_res_ys = generate_FTLE_SLICE(
            config, vectorfield, float(time_slice), flowline_dt, max_steps, high_res_sampling
        )

        # Preprocessing consistent with training: temporal downsampling and normalization (no FTLE normalization)
        temporal_sampled_P_all = temporal_downsamplePathlineCrossPrimitive(lowResPathlines, int(LstepsPerline))
        # The training set used constant 5 as cross neighborsize; keep consistent here
        lowResPathlinesPreprocessed = preprocess_localization_normalization(
            temporal_sampled_P_all, 5, int(LstepsPerline), bool(localized), False
        ).cpu().float()

        lowResFTLE_list.append(torch.from_numpy(low_resFTLE_field).float())
        highResFTLE_list.append(torch.from_numpy(high_resFTLE_field).float())
        lowResPathlines_list.append(lowResPathlinesPreprocessed)

    # Stack
    lowResFTLE_all_t = torch.stack(lowResFTLE_list, dim=0)
    highResFTLE_all_t = torch.stack(highResFTLE_list, dim=0)
    lowResPathlines_all = torch.stack(lowResPathlines_list, dim=0)

    # Save cache (use float32)
    try:
        np.savez(
            cache_path,
            LowResFTLE=lowResFTLE_all_t.detach().cpu().numpy().astype(np.float32),
            HighResFTLE=highResFTLE_all_t.detach().cpu().numpy().astype(np.float32),
            LowResPathlines=lowResPathlines_all.detach().cpu().numpy().astype(np.float32),
        )
        logging.info(f"[build_test_dataset] saved {lowResFTLE_all_t.shape[0]} samples to cache {cache_path}")
    except Exception as e:
        logging.info(f"[build_test_dataset] cache save failed: {e}")

    # Return numpy (FTLE) + torch (Pathlines), consistent with test usage
    return (
        lowResFTLE_all_t.detach().cpu().numpy().astype(np.float32),
        lowResPathlines_all,
        highResFTLE_all_t.detach().cpu().numpy().astype(np.float32),
        vectorfield,
    )
def test_UpsamplingModel(config, model,test_dataset, device,visualize=False):
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

        # Use cached dataset
        if test_dataset is None:
            lowResFTLE_all, lowResPathlines_all, highResFTLE_all, vectorfield = build_test_dataset(config)
        else:
            lowResFTLE_all, lowResPathlines_all, highResFTLE_all, vectorfield = test_dataset

        if lowResFTLE_all is not None and lowResPathlines_all is not None and highResFTLE_all is not None and vectorfield is not None:
            patch_size = int(getattr(config.dataset, 'patchSize', 32))
            patch_stride = int(getattr(config.dataset, 'patchStride', 2))
            patch_stride=patch_stride*4 if visualize==False else patch_stride # faster test if not visualize

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

                # Normalization as in training (normalize low-res input using train-set statistics)
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
            logging.info(f"test average: mse={mse:.6f}, mae={mae:.6f}, maxe={maxe:.6f}, psnr={psnr:.6f}")
            return {"mse": mse, "mae": mae, "maxe": maxe, "psnr": psnr}

def train_model(config, model, dataset, device,test_dataset=None):
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
                logging.info(f"Warning: nan or inf in pred at epoch {epoch}, iter {it}")
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
                logging.info(f"epoch {epoch}, iter {it}: {LOSS_NAME} ={loss.item():.6f}, total_iterations: {total_iterations}")
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
                    logging.info(f"[lr scheduler] epoch {epoch}: lr is too small, stop training")
                    break
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                logging.info(f"[lr scheduler] epoch {epoch}: loss does not decrease for {patience} epochs, learning rate adjusted to {new_lr:.6g}")
                last_lr = new_lr
                # to avoid multiple triggers, clear loss_history, only keep the latest one
                loss_history = [loss_history[-1]]
        if task_init_fn is not None and callable(task_init_fn):
            Res = task_init_fn(config, model,test_dataset, device=str(device)) 
            cur_psnr = float(Res['psnr'])
            if cur_psnr > best_psnr:
                best_psnr = cur_psnr
                best_state_dict = copy.deepcopy(model.state_dict())
                logging.info(f"[best] epoch {epoch}: psnr={best_psnr:.2f} dB (checkpoint updated)")
            logging.info(f"epoch {epoch}: {LOSS_NAME} ={epoch_avg_loss:.6f}. Test:mse={Res['mse']:.6f}, mae={Res['mae']:.6f}, maxe={Res['maxe']:.6f}, psnr={cur_psnr:.2f} dB") if Res is not None else logging.info(f"epoch {epoch}: {LOSS_NAME} ={epoch_avg_loss:.6f}")
            if config['wandb']:
                wandb.log({"epoch": epoch,  "epoch_Loss": epoch_avg_loss, "test_mse": Res['mse'], "test_mae": Res['mae'], "test_maxe": Res['maxe'], "test_psnr": cur_psnr})
        else:
            logging.info(f"epoch {epoch}: {LOSS_NAME} ={epoch_avg_loss:.6f}")
            if config['wandb']:
                    wandb.log({"epoch": epoch,  "epoch_Loss": epoch_avg_loss, "total_iterations": total_iterations})


    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        logging.info(f"[final] loaded best checkpoint with best psnr={best_psnr:.2f} dB")
        test_result= task_init_fn(config, model,test_dataset, device=str(device), visualize=True) if task_init_fn is not None and callable(task_init_fn) else None
        if config['wandb'] and test_result is not None and isinstance(test_result,dict):            
            wandb.summary.update({"best_test_mse": test_result['mse'],"best_test_psnr": test_result['psnr']})
            for key, value in test_result.items():
                wandb.summary.update({key: value})
            wandb.finish()


def runNameTagGenerator_fmt(config,mode)->Tuple[str, List[str]]:
    seed=config['seed']
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    runName = f"{config['model']['NAME']}_bs_{config['batch_size']}_ep_{config['epochs']}_lr_{config['lr']}_{current_time}_seed_{seed}"
    tagGen0=mode
    tagGen1=config['model']['NAME']
    runTags= [tagGen0,tagGen1]
    return runName,runTags

def relocate_flow2d_dataset_folder(config):
    import platform
    if platform.system() == "Windows":
        config.dataset.dat_dir="C:\\Users\\xingdi\\OneDrive - KAUST\\WorkingInProcess\\FLowVisAssets\\flowData2d"
        config.cache_dir="./outputs/"
    elif platform.system() == "Linux":
        config.dataset.dat_dir="/ibex/user/zhanx0o/FLowDataFolder/"
        config.cache_dir="/ibex/user/zhanx0o/outputs/"
    else:
        raise ValueError(f"Unknown system: {platform.system()}")


    
if __name__=="__main__":
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"torch.cuda.is_available(): {torch.cuda.is_available()}")  # Should return True
    logging.info(f"torch.version.cuda: {torch.version.cuda}")         # Should print the CUDA version PyTorch was built with
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # mode = 'point_FTLE_regression'
    mode = 'upsampling'

    cfg=argParseAndPrepareConfig()
    cfg["gitInfo"]=get_git_commit_id()
    relocate_flow2d_dataset_folder(cfg)
    run_Name,runTags=runNameTagGenerator_fmt(cfg,mode)
    cfg['run_name']=run_Name
    logging.info(f"run name: {run_Name}, run tags: {runTags}")

 
       
    print_args(cfg,printer=logging.info)
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
        vectorfield_datapath=os.path.join(config.dataset.dat_dir, f"{config.dataset.test_name}.{config.dataset.extension}")
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
        test_dataset=build_test_dataset(cfg)
        logging.info(f"build_dataset done, train dataset lenth: {dataset.lowResFTLE.shape[0]}")
        lowResX,lowResY=dataset.lowResFTLE[0].shape[0],dataset.lowResFTLE[0].shape[1]
        cfg['lowResX']=lowResX
        cfg['lowResY']=lowResY
        cfg['ftle_min']=dataset.ftle_min
        cfg['ftle_max']=dataset.ftle_max
        model = build_model(cfg, device)
        train_model(cfg,model,dataset,device,test_dataset)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    

