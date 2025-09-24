#step 1: load FTLE dataset
from calendar import c
import os,random
import logging
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
from DeepUtils.utils.stable_hash import stable_hash
from FMT_Utils.FTLE_fitting_utils import *
from DeepUtils.MiscFunctions import *
from FMT_Utils.model_zoo import *
import pickle
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
    # 支持多个测试流场名称；兼容字符串输入
    names_cfg = config['test_vectorfield'] if 'test_vectorfield' in config else [config.dataset.names[0]]
    test_vectorfield_names = names_cfg if isinstance(names_cfg, (list, tuple)) else [names_cfg]
    

    time_window_start_ratio = float(config.dataset.t_start)
    time_window_target_ratio = float(config.dataset.t_target)
    timesliceCount =int(getattr(config.dataset, 'timesliceCount', 8))//2

    low_res_grid_sampling = float(config.dataset.low_res_grid_sampling)
    UPsampling = int(config.dataset.UPsampling)
    max_steps = int(config.pcds.max_iterations)
    flowline_dt = float(config.pcds.dt)
    offset_dist = float(config.pcds.offset_dist)
    LstepsPerline = int(config.pcds.sampled_points_per_line)
    localized = bool(config.pcds.localized)
    mode=config['mode']

    # Cache key
    key_obj = {
        "name": "ftle_upsampling_test",
        "vectorfields": list(map(str, test_vectorfield_names)),
        "timesliceCount": int(timesliceCount),
        "UPsampling": int(UPsampling),
        "lowResGridIntervalScale": float(low_res_grid_sampling),
        "time_window_start_ratio": float(time_window_start_ratio),
        "time_window_target_ratio": float(time_window_target_ratio),
        "max_steps": int(max_steps),
        "dt": float(flowline_dt),
        "offset_dist": float(offset_dist),
        "LstepsPerline": int(LstepsPerline)
    }
    tag = stable_hash(key_obj, prefix=f"{mode}TestDataset_")
    cache_dir = os.path.join(config.cache_dir, "temp")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{tag}.pkl")

    lowResFTLEorFLowMap_list = []
    highResFTLEorFLowMap_list = []
    lowResPathlines_list = []

    # Try loading cache first
    if os.path.exists(cache_path):
        try:
            data = pickle.load(open(cache_path, "rb"))
            lowResFTLEorFLowMap_all_list = data["lowResFTLE_list"]
            highResFTLEorFLowMap_all_list = data["highResFTLE_list"]
            lowResPathlines_all = data["lowResPathlines_list"]
            assert len(lowResFTLEorFLowMap_all_list) == len(highResFTLEorFLowMap_all_list) ==len(lowResPathlines_all)
            logging.info(f"[build_test_dataset] loaded {len(lowResPathlines_all)} samples from cache {cache_path}")
            return lowResFTLEorFLowMap_all_list,highResFTLEorFLowMap_all_list,  lowResPathlines_all
        except Exception as e:
            logging.info(f"[build_test_dataset] cache load failed: {e}. Regenerating...")


    for vf_name in test_vectorfield_names:
        vf_obj = load_UnsteadyVectorFields_netCDFOrAnalytical(config.dataset.dat_dir,vf_name)[0]
        if vf_obj is None:
            logging.info(f"[build_test_dataset] load {vf_name} failed. Skip this field.")
            continue
  
        # 针对每个流场单独确定时间窗口
        tmin, tmax = float(vf_obj.tmin), float(vf_obj.tmax)
        time_window_start = float(time_window_start_ratio * (tmax - tmin) + tmin)
        time_window_target = float(time_window_target_ratio * (tmax - tmin) + tmin)
        sample_times = np.linspace(time_window_start, time_window_target, num=timesliceCount)
        high_res_sampling=float(UPsampling*low_res_grid_sampling)
        if mode == 'upsamplingFTLE':
            for time_slice in sample_times:
                # Low-resolution slice and corresponding pathlines
                low_resFTLE_field, lowResPathlines, low_res_xs, low_res_ys = generate_FTLE_SLICE(
                    config, vf_obj, float(time_slice), flowline_dt, max_steps, low_res_grid_sampling
                )
                # High-resolution (as ground truth)
                high_resFTLE_field, _, high_res_xs, high_res_ys = generate_FTLE_SLICE(
                    config, vf_obj, float(time_slice), flowline_dt, max_steps, high_res_sampling
                )

                # Preprocessing consistent with training: temporal downsampling and normalization (no FTLE normalization)
                # pathline_length_in_save_data=max(max_steps//2, LstepsPerline)
                # temporal_sampled_P_all = temporal_downsamplePathlineCrossPrimitiveRegular(lowResPathlines, int(LstepsPerline))
                temporal_sampled_P_all=AngleAwareSampling(lowResPathlines, int(LstepsPerline))

                lowResFTLEorFLowMap_list.append(low_resFTLE_field)
                highResFTLEorFLowMap_list.append(high_resFTLE_field)
                lowResPathlines_list.append(temporal_sampled_P_all)
        # elif mode == 'upsamplingFLowMap':
        #         for time_slice in sample_times:
        #             # Low-resolution slice and corresponding pathlines
        #             low_resFTLE_field, lowResPathlines, low_res_xs, low_res_ys = generate_FLowMap_SLICE(
        #                 config, vf_obj, float(time_slice), flowline_dt, max_steps, low_res_grid_sampling
        #             )
        #             # High-resolution (as ground truth)
        #             high_res_sampling = up * low_res_grid_sampling
        #             high_resFTLE_field, _, high_res_xs, high_res_ys = generate_FLowMap_SLICE(
        #                 config, vf_obj, float(time_slice), flowline_dt, max_steps, high_res_sampling
        #             )
        #             # Preprocessing consistent with training: temporal downsampling and normalization (no FTLE normalization)
        #             # pathline_length_in_save_data=max(max_steps//2, LstepsPerline)
        #             # temporal_sampled_P_all = temporal_downsamplePathlineCrossPrimitiveRegular(lowResPathlines, int(LstepsPerline))
        #             temporal_sampled_P_all=AngleAwareSampling(lowResPathlines, int(LstepsPerline))

        #             lowResFTLEorFLowMap_list.append(low_resFTLE_field)
        #             highResFTLEorFLowMap_list.append(high_resFTLE_field)
        #             lowResPathlines_list.append(temporal_sampled_P_all)
    
    # Save cache (use float32)
    try:
        with open(cache_path, "wb") as f:
            pickle.dump({"lowResFTLE_list": lowResFTLEorFLowMap_list, "highResFTLE_list": highResFTLEorFLowMap_list, "lowResPathlines_list": lowResPathlines_list}, f)
        logging.info(f"[build_test_dataset] saved {len(lowResFTLEorFLowMap_list)} samples to cache {cache_path}")
    except Exception as e:
        logging.info(f"[build_test_dataset] cache save failed: {e}")

    # Return numpy (FTLE) + torch (Pathlines), consistent with test usage

    return (
        lowResFTLEorFLowMap_list,
        highResFTLEorFLowMap_list,
        lowResPathlines_list
    )


def _inverse_normalization(flowmap: np.ndarray|torch.Tensor):
    flowmap[...,0:2]=flowmap[...,0:2]*global_UniformValueSpatical
    flowmap[...,2]=flowmap[...,2]*global_UniformValueTemporal
    flowmap[...,3:5]=flowmap[...,3:5]*global_UniformValueSpatical
    flowmap[...,5]=flowmap[...,5]*global_UniformValueTemporal
    return flowmap


test_times=0

def test_UpsamplingModel(config, model,test_dataset, device,visualize=False):
    global test_times
    test_times += 1
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
            lowResFTLE_all,  highResFTLE_all ,lowResPathlines_all,= build_test_dataset(config)
        else:
            lowResFTLE_all, highResFTLE_all, lowResPathlines_all= test_dataset


        if lowResFTLE_all is not None and lowResPathlines_all is not None and highResFTLE_all is not None and config['mode'] == 'upsamplingFTLE':
            patch_size = int(getattr(config.dataset, 'patchSize', 32))
            patch_stride = int(getattr(config.dataset, 'patchStride', 2))
            patch_stride=patch_stride*2 if visualize==False else patch_stride # faster test if not visualize

            LstepsPerline = int(config.pcds.sampled_points_per_line)
            mse_sum = 0.0
            mae_sum = 0.0
            maxe_sum = 0.0
            psnr_sum = 0.0
            psnr_bilinear_sum = 0.0
            psnr_cubic_sum = 0.0
            sample_count = int(len(lowResPathlines_all))
            UPsampling = int(config.dataset.UPsampling)
            for test_i in range(sample_count):
                low_resFTLE_field = lowResFTLE_all[test_i]
                high_resFTLE_field =highResFTLE_all[test_i]
                lowResPathlinesPreprocessed =lowResPathlines_all[test_i]

                # Normalization as in training (normalize low-res input using train-set statistics)
                ftle_min = float(config.ftle_min)
                ftle_max = float(config.ftle_max)
                low_resFTLE_field_clip = np.clip(low_resFTLE_field, ftle_min, ftle_max)
                low_resFTLE_field_norm = (low_resFTLE_field_clip - ftle_min) / max(1e-12, (ftle_max - ftle_min))
                low_resFTLE_field_norm=torch.from_numpy(low_resFTLE_field_norm).float()

                ny_low, nx_low = low_resFTLE_field.shape[:2]
                ny_hi, nx_hi = high_resFTLE_field.shape[:2]
                ry = UPsampling
                rx = UPsampling

                row_starts = _tiling_starts(ny_low, patch_size, patch_stride)
                col_starts = _tiling_starts(nx_low, patch_size, patch_stride)

                pred_grid = np.zeros((ny_hi, nx_hi), dtype=np.float32)
                weight_grid = np.zeros((ny_hi, nx_hi), dtype=np.float32)

                for i0 in row_starts:
                    i1 = min(i0 + patch_size, ny_low)
                    for j0 in col_starts:
                        j1 = min(j0 + patch_size, nx_low)

                        # map to high-res
                        hi_h = max(1,patch_size*UPsampling)
                        hi_w = max(1,patch_size*UPsampling)
                        hi_i0 = int(round(i0 * ry))
                        hi_j0 = int(round(j0 * rx))
                        if i0 == row_starts[-1]:
                            hi_i0 = ny_hi - hi_h
                        if j0 == col_starts[-1]:
                            hi_j0 = nx_hi - hi_w
                        hi_i1 = hi_i0 + hi_h
                        hi_j1 = hi_j0 + hi_w

                        # build inputs
                        lr_patch_norm = low_resFTLE_field_norm[i0:i1, j0:j1].unsqueeze(0).to(device).float()

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
                        patch_np = pred_patch.squeeze(0).detach().cpu().numpy()
                        hi_i0=max(0,hi_i0)
                        hi_j0=max(0,hi_j0)
                        hi_i1=min(ny_hi,hi_i1)
                        hi_j1=min(nx_hi,hi_j1)
                        pred_grid[hi_i0:hi_i1, hi_j0:hi_j1] += patch_np
                        weight_grid[hi_i0:hi_i1, hi_j0:hi_j1] += 1.0

                # metrics (raw scale)
                weight_grid = np.clip(weight_grid, 1.0, None)
                pred_grid = pred_grid / weight_grid
                pred_grid = pred_grid * (ftle_max - ftle_min) + ftle_min

                label_y_b = high_resFTLE_field.astype(np.float32)
                pred_b = pred_grid.astype(np.float32)
                mse, mae, maxe, psnr = compute_metrics(label_y_b, pred_b)

                # 计算插值基线的 PSNR（双线性与立方）
                if test_times == 1:
                    with torch.no_grad():
                        lr = torch.from_numpy(low_resFTLE_field)[None, None, ...].float()
                        lr_up = torch.nn.functional.interpolate(lr, size=(label_y_b.shape[0], label_y_b.shape[1]), mode='bilinear', align_corners=False)[0, 0]
                        bilinear_grid = lr_up.detach().cpu().numpy().astype(np.float32)
                    _, _, _, psnr_bilinear = compute_metrics(label_y_b, bilinear_grid)
                    psnr_bilinear_sum += psnr_bilinear
                    with torch.no_grad():
                        lr_up_cubic = torch.nn.functional.interpolate(lr, size=(label_y_b.shape[0], label_y_b.shape[1]), mode='bicubic', align_corners=False)[0, 0]
                        cubic_grid = lr_up_cubic.detach().cpu().numpy().astype(np.float32)
                    _, _, _, psnr_cubic = compute_metrics(label_y_b, cubic_grid)
                    psnr_cubic_sum += psnr_cubic

                if visualize and test_i == sample_count-1:
                    vectorfield_4vis = config.test_vectorfield[-1] if isinstance(config.test_vectorfield, list) else [config.test_vectorfield]
                    vectorfield = load_UnsteadyVectorFields_netCDFOrAnalytical(config.dataset.dat_dir,vectorfield_4vis)[-1]
                    visualize_FTLEUpampling(label_y_b, pred_b, low_resFTLE_field, vectorfield.domainMinBoundary, vectorfield.domainMaxBoundary)

                mse_sum += mse
                mae_sum += mae
                maxe_sum += maxe
                psnr_sum += psnr

            mse = mse_sum / sample_count
            mae = mae_sum / sample_count
            maxe = maxe_sum / sample_count
            psnr = psnr_sum / sample_count
            psnr_bilinear = psnr_bilinear_sum / sample_count
            psnr_cubic = psnr_cubic_sum / sample_count
            if test_times == 1:
                logging.info(f"baseline psnr_bilinear={psnr_bilinear:.6f}, psnr_cubic={psnr_cubic:.6f}")

            # logging.info(f"test average: mse={mse:.6f}, mae={mae:.6f}, maxe={maxe:.6f}, psnr={psnr:.6f}")
            return {"mse": mse, "mae": mae, "maxe": maxe, "psnr": psnr}


        # elif lowResFTLE_all is not None and lowResPathlines_all is not None and highResFTLE_all is not None and config['mode'] == 'upsamplingFLowMap':
        #     patch_size = int(getattr(config.dataset, 'patchSize', 32))
        #     patch_stride = int(getattr(config.dataset, 'patchStride', 2))
        #     patch_stride=patch_stride*2 if visualize==False else patch_stride # faster test if not visualize
        #     LstepsPerline = int(config.pcds.sampled_points_per_line)
        #     mse_sum = 0.0
        #     mae_sum = 0.0
        #     maxe_sum = 0.0
        #     psnr_sum = 0.0
        #     psnr_bilinear_sum = 0.0
        #     psnr_cubic_sum = 0.0
        #     sample_count = int(len(lowResPathlines_all))
        #     UPsampling = int(config.dataset.UPsampling)
        #     dt = float(config.pcds.dt)
        #     max_steps = int(config.pcds.max_iterations)
        #     def _ftle_from_flowmap_np(flowmap_np: np.ndarray) -> np.ndarray:
        #         # flowmap_np: [H, W, 2 or 3]
        #         fm = torch.from_numpy(flowmap_np).float()
        #         ftle_core = FTLEFromFlowMap(fm)  # 0.5*log(lambda_max)
        #         return ftle_core.detach().cpu().numpy().astype(np.float32)
        #     for test_i in range(sample_count):
        #         low_resFlowMap_field = lowResFTLE_all[test_i]
        #         high_resFlowMap_field =highResFTLE_all[test_i]
        #         lowResPathlinesPreprocessed =lowResPathlines_all[test_i]
        #         ny_low, nx_low = low_resFlowMap_field.shape[:2]
        #         ny_hi, nx_hi = high_resFlowMap_field.shape[:2]
        #         ry = UPsampling
        #         rx = UPsampling
        #         row_starts = _tiling_starts(ny_low, patch_size, patch_stride)
        #         col_starts = _tiling_starts(nx_low, patch_size, patch_stride)
        #         # 预测的是流映射 (x,y[,t])，先拼接再转 FTLE
        #         pred_grid = np.zeros((ny_hi, nx_hi, low_resFlowMap_field.shape[-1]), dtype=np.float32)
        #         weight_grid = np.zeros((ny_hi, nx_hi), dtype=np.float32)
        #         for i0 in row_starts:
        #             i1 = min(i0 + patch_size, ny_low)
        #             for j0 in col_starts:
        #                 j1 = min(j0 + patch_size, nx_low)
        #                 # map to high-res
        #                 hi_h = max(1,patch_size*UPsampling)
        #                 hi_w = max(1,patch_size*UPsampling)
        #                 hi_i0 = int(round(i0 * ry))
        #                 hi_j0 = int(round(j0 * rx))
        #                 if i0 == row_starts[-1]:
        #                     hi_i0 = ny_hi - hi_h
        #                 if j0 == col_starts[-1]:
        #                     hi_j0 = nx_hi - hi_w
        #                 hi_i1 = hi_i0 + hi_h
        #                 hi_j1 = hi_j0 + hi_w
        #                 # build inputs
        #                 lr_patch_norm = torch.from_numpy(low_resFlowMap_field[i0:i1, j0:j1]).unsqueeze(0).to(device).float()
        #                 # select corresponding pathlines groups
        #                 idx_list = []
        #                 for rr in range(i0, i1):
        #                     base = rr * nx_low
        #                     for cc in range(j0, j1):
        #                         idx_list.append(base + cc)
        #                 idx_tensor = torch.as_tensor(idx_list, dtype=torch.long)
        #                 pl_patch = lowResPathlinesPreprocessed[idx_tensor].unsqueeze(0).to(device).float()
        #                 # forward
        #                 pred_patch = model(lr_patch_norm, pl_patch).to(device).float()  # [1, hi_h, hi_w, C] or [1, hi_h, hi_w] if model outputs FTLE directly
        #                 patch_np = pred_patch.permute(0,2,3,1).squeeze(0).detach().cpu().numpy()
        #                 hi_i0=max(0,hi_i0)
        #                 hi_j0=max(0,hi_j0)
        #                 hi_i1=min(ny_hi,hi_i1)
        #                 hi_j1=min(nx_hi,hi_j1)
        #                 pred_grid[hi_i0:hi_i1, hi_j0:hi_j1, :patch_np.shape[-1]] += patch_np
        #                 weight_grid[hi_i0:hi_i1, hi_j0:hi_j1] += 1.0
        #         # metrics (raw scale)
        #         weight_grid = np.clip(weight_grid, 1.0, None)
        #         pred_grid = pred_grid / weight_grid[..., None]
        #         #inverse normalization
        #         pred_grid=_inverse_normalization(pred_grid)
        #         pred_grid[...,3:6]= pred_grid[...,3:6]+ pred_grid[...,0:3]
        #         # 转换为 FTLE 再评估
        #         label_y_b = _ftle_from_flowmap_np(high_resFlowMap_field.astype(np.float32)) 
        #         pred_b = _ftle_from_flowmap_np(pred_grid.astype(np.float32))
        #         mse, mae, maxe, psnr = compute_metrics(label_y_b, pred_b)
        #         # 计算插值基线的 PSNR（双线性与立方）
        #         if test_times == 1:
        #             low_resFTLE_field= torch.from_numpy(_ftle_from_flowmap_np( low_resFlowMap_field.astype(np.float32))).unsqueeze(0).unsqueeze(0)
        #             with torch.no_grad():
        #                 # 对低分辨率流映射的前两通道分别做双线性/双三次插值，再转 FTLE
        #                 bilinear_lr_up_bi = torch.nn.functional.interpolate(low_resFTLE_field, size=(label_y_b.shape[0], label_y_b.shape[1]), mode='bilinear', align_corners=False)[0].squeeze()
        #             _, _, _, psnr_bilinear = compute_metrics(label_y_b, bilinear_lr_up_bi)
        #             psnr_bilinear_sum += psnr_bilinear
        #             with torch.no_grad():
        #                 lr_up_cubic = torch.nn.functional.interpolate(low_resFTLE_field, size=(label_y_b.shape[0], label_y_b.shape[1]), mode='bicubic', align_corners=False)[0].squeeze()
        #             _, _, _, psnr_cubic = compute_metrics(label_y_b, lr_up_cubic)
        #             psnr_cubic_sum += psnr_cubic
        #         if visualize and test_i == sample_count-1:
        #             vectorfield_4vis = config.test_vectorfield[-1] if isinstance(config.test_vectorfield, list) else [config.test_vectorfield]
        #             vectorfield = load_UnsteadyVectorFields_netCDFOrAnalytical(config.dataset.dat_dir,vectorfield_4vis)[-1]
        #             low_resFTLE_field=_ftle_from_flowmap_np(low_resFlowMap_field.astype(np.float32))
        #             visualize_FTLEUpampling(label_y_b, pred_b, low_resFTLE_field, vectorfield.domainMinBoundary, vectorfield.domainMaxBoundary)
        #             visualize_OneScalarField(pred_b, vectorfield.domainMinBoundary, vectorfield.domainMaxBoundary)
        #             # visualize_OneScalarField(low_resFTLE_field, vectorfield.domainMinBoundary, vectorfield.domainMaxBoundary)
        #             # visualize_OneScalarField(label_y_b, vectorfield.domainMinBoundary, vectorfield.domainMaxBoundary)
        #         mse_sum += mse
        #         mae_sum += mae
        #         maxe_sum += maxe
        #         psnr_sum += psnr
        #     mse = mse_sum / sample_count
        #     mae = mae_sum / sample_count
        #     maxe = maxe_sum / sample_count
        #     psnr = psnr_sum / sample_count
        #     psnr_bilinear = psnr_bilinear_sum / sample_count
        #     psnr_cubic = psnr_cubic_sum / sample_count
        #     if test_times == 1:
        #         logging.info(f"baseline psnr_bilinear={psnr_bilinear:.6f}, psnr_cubic={psnr_cubic:.6f}")
        #     logging.info(f"test average: mse={mse:.6f}, mae={mae:.6f}, maxe={maxe:.6f}, psnr={psnr:.6f}")
        #     return {"mse": mse, "mae": mae, "maxe": maxe, "psnr": psnr}





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
            if Res is not None:
                logging.info(f"epoch {epoch}: {LOSS_NAME} ={epoch_avg_loss:.6f}. Test:avg mse={Res['mse']:.6f}, avg mae={Res['mae']:.6f}, avg maxe={Res['maxe']:.6f}, avg psnr={cur_psnr:.2f} dB")
            else:
                logging.info(f"epoch {epoch}: {LOSS_NAME} ={epoch_avg_loss:.6f}")
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
            wandb.summary.update({"best_test_mse": test_result['mse'],"best_test_psnr": test_result['psnr'], "best_test_psnr_bilinear": test_result.get('psnr_bilinear', float('nan'))})
            for key, value in test_result.items():
                wandb.summary.update({key: value})
            wandb.finish()
        if  getattr(config, "save_model", False):
            save_checkpoint(best_state_dict, path=f"./outputModels/{config['run_name']}", checkpoint_name=f"best_checkpoint.pth.tar")


def runNameTagGenerator_fmt(config,mode)->Tuple[str, List[str]]:
    seed=config['seed']
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    runName = f"{mode}_{config['model']['NAME']}_bs_{config['batch_size']}_ep_{config['epochs']}_lr_{config['lr']}_{current_time}_seed_{seed}"
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


    cfg=argParseAndPrepareConfig()
    cfg["gitInfo"]=get_git_commit_id()
    mode=cfg['mode']
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
    elif mode == 'upsamplingFTLE':
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
    elif mode == 'upsamplingFLowMap':
        dataset = FLowMapUpsamplingTrainDataset(cfg, useCacheSystem=True)
        test_dataset=build_test_dataset(cfg)
        logging.info(f"upsamplingFLowMap task build_dataset done, train dataset lenth: {dataset.lowResFlowMap.shape[0]}")
        lowResX,lowResY=dataset.lowResFlowMap[0].shape[0],dataset.lowResFlowMap[0].shape[1]
        cfg['lowResX']=lowResX
        cfg['lowResY']=lowResY
        model = build_model(cfg, device)
        train_model(cfg,model,dataset,device,test_dataset)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    

