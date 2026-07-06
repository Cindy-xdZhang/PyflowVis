#step 1: load FTLE dataset
from calendar import c
import os
import logging
import copy
import numpy as np
import pickle
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from DeepUtils.utils import EasyConfig
from DeepUtils.loss import build_criterion_from_cfg
from DeepUtils.optim import build_optimizer_from_cfg
from DeepUtils.utils.stable_hash import stable_hash
from DeepUtils.MiscFunctions import *

from FLowUtils.VectorField2d import *
from FLowUtils.flowDatasetUtils.NetCDF_AmiraLoader import relocate_flow2d_dataset_folder
from FLowUtils.flowDatasetUtils.NetCDF_AmiraLoader import (
    load_UnsteadyVectorFields_general as load_UnsteadyVectorFields_netCDFOrAnalytical,)

from FMT_Utils.FTLE_fitting_utils import *
from FMT_Utils import debug_checks as dbg
from FMT_Utils.model_zoo import *
from FMT_Utils.flowmap_sr import (
    FlowMapSRTrainDataset, build_FlowMapSR_test_dataset,
    flowmap_unit_normalize, flowmap_unit_denormalize, ftle_from_endpos_grid)


GLOBAL_WANDB_PROJECT_NAME="FlowMapTokenizer"
torch.backends.cuda.matmul.allow_tf32 = False
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

def build_test_dataset(config):
    """
    Build test slices with the same low/high-resolution generation path used by training.

    Returns:
        lowResFTLEorFLowMap_list:
            - upsamplingFTLE: np.ndarray [ny_low, nx_low]
            - upsamplingFLowMap: np.ndarray [ny_low, nx_low, 5, 2, 3]
        highResFTLEorFLowMap_list:
            - upsamplingFTLE: np.ndarray [ny_hi, nx_hi]
            - upsamplingFLowMap: np.ndarray [ny_hi, nx_hi, 5, 2, 3]
        lowResPathlines_list: torch.FloatTensor [ny_low*nx_low, 5, L, 3]
    """
    # 支持多个测试流场名称；兼容字符串输入。
    names_cfg =  config.test.vectorfield          
    test_vectorfield_names = names_cfg if isinstance(names_cfg, (list, tuple)) else [names_cfg]
    time_window_start_ratio = float(config.dataset.t_start)
    time_window_target_ratio = float(config.dataset.t_target)
    timesliceCount = int(config.test.timesliceCount)

    low_res_grid_sampling = float(config.dataset.low_res_grid_sampling)
    UPsampling = int(config.dataset.UPsampling)
    max_steps = int(config.pcds.max_iterations)
    flowline_dt = float(config.pcds.dt)
    offset_dist = float(config.pcds.offset_dist)
    LstepsPerline = int(config.pcds.sampled_points_per_line)
    localized = bool(config.pcds.localized)
    mode=config['mode']

    # Keep cache parameters aligned with the train dataset keys where they affect generated data.
    key_obj = {
        "name": f"{mode}_test",
        "vectorfields": list(map(str, test_vectorfield_names)),
        "timesliceCount": int(timesliceCount),
        "UPsampling": int(UPsampling),
        "lowResGridIntervalScale": float(low_res_grid_sampling),
        "time_window_start_ratio": float(time_window_start_ratio),
        "time_window_target_ratio": float(time_window_target_ratio),
        "max_steps": int(max_steps),
        "dt": float(flowline_dt),
        "offset_dist": float(offset_dist),
        "LstepsPerline": int(LstepsPerline),
        "mode": str(mode),
    }
    tag = stable_hash(key_obj, prefix=f"{mode}TestDataset_")
    cache_dir = os.path.join(config.cache_dir, "temp")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{tag}.pkl")

    lowResFTLEorFLowMap_list = []
    highResFTLEorFLowMap_list = []
    lowResPathlines_list = []
    per_field_test_patches = {}

    # Try loading cache first
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            lowResFTLEorFLowMap_all_list = data["lowResFTLE_list"]
            highResFTLEorFLowMap_all_list = data["highResFTLE_list"]
            lowResPathlines_all = data["lowResPathlines_list"]
            assert len(lowResFTLEorFLowMap_all_list) == len(highResFTLEorFLowMap_all_list) ==len(lowResPathlines_all)
            logging.info(f"[build_test_dataset] loaded {len(lowResPathlines_all)} samples from cache {cache_path}")
            for vf_name, n_patches in data.get("per_field_test_patches", {}).items():
                logging.info(f"[build_test_dataset] '{vf_name}': loaded {int(n_patches)} test patches from cache.")
            return lowResFTLEorFLowMap_all_list,highResFTLEorFLowMap_all_list,  lowResPathlines_all
        except Exception as e:
            logging.info(f"[build_test_dataset] cache load failed: {e}. Regenerating...")


    vectorfields = load_UnsteadyVectorFields_netCDFOrAnalytical(config.dataset.dat_dir, test_vectorfield_names)
    for vf_idx, vf_obj in enumerate(vectorfields):
        vf_name = test_vectorfield_names[vf_idx] if vf_idx < len(test_vectorfield_names) else f"field{vf_idx}"
        if vf_obj is None:
            logging.info(f"[build_test_dataset] load {vf_name} failed. Skip this field.")
            continue
        field_patches_before = len(lowResFTLEorFLowMap_list)
  
        # 针对每个流场单独确定时间窗口
        tmin, tmax = float(vf_obj.tmin), float(vf_obj.tmax)
        time_window_start = float(time_window_start_ratio * (tmax - tmin) + tmin)
        time_window_target = float(time_window_target_ratio * (tmax - tmin) + tmin)
        # Forward FTLE integration needs seed_time + dt*max_steps <= tmax (else the field
        # collapses to all-zeros). Cap the latest seed time so the horizon fits the domain.
        integ_horizon = float(flowline_dt) * int(max_steps)
        safe_target = float(tmax) - integ_horizon
        if time_window_target > safe_target:
            logging.warning(f"[build_test_dataset] t_target={time_window_target:.3f} + integ horizon "
                            f"{integ_horizon:.3f} exceeds tmax={float(tmax):.3f}; "
                            f"clamping slice end to {max(time_window_start, safe_target):.3f}")
            time_window_target = max(time_window_start, safe_target)
        sample_times = np.linspace(time_window_start, time_window_target, timesliceCount)
        high_res_sampling=float(UPsampling*low_res_grid_sampling)
        
        for physcialTime in sample_times:
            # Same raw-coordinate flow map generation as FlowMapUpsamplingTrainDataset.
            # Test keeps whole-grid tensors for fully convolutional evaluation.
            low_flow, lowResPathlines, _llen, nx_low, ny_low = generate_Flowmap_SLICE(
                vf_obj, float(physcialTime), flowline_dt, max_steps, offset_dist, low_res_grid_sampling)
            high_flow, _hpl, _hlen, nx_hi, ny_hi = generate_Flowmap_SLICE(
                vf_obj, float(physcialTime), flowline_dt, max_steps, offset_dist, high_res_sampling)
            low_grid = low_flow.reshape(ny_low, nx_low, 5, 2, 3).detach().cpu().numpy().astype(np.float32)
            high_grid = high_flow.reshape(ny_hi, nx_hi, 5, 2, 3).detach().cpu().numpy().astype(np.float32)
            temporal_sampled_P_all = AngleAwareSampling(lowResPathlines, int(LstepsPerline))
            lowResFTLEorFLowMap_list.append(low_grid)
            highResFTLEorFLowMap_list.append(high_grid)
            lowResPathlines_list.append(temporal_sampled_P_all)

        n_field_patches = len(lowResFTLEorFLowMap_list) - field_patches_before
        per_field_test_patches[str(vf_name)] = int(n_field_patches)
        if n_field_patches == 0:
            logging.warning(f"[build_test_dataset] '{vf_name}': 0 test patches generated.")
        else:
            logging.info(f"[build_test_dataset] '{vf_name}': collected {n_field_patches} test patches.")
    
    
    
    # Save cache (use float32)
    try:
        with open(cache_path, "wb") as f:
            pickle.dump({
                "lowResFTLE_list": lowResFTLEorFLowMap_list,
                "highResFTLE_list": highResFTLEorFLowMap_list,
                "lowResPathlines_list": lowResPathlines_list,
                "per_field_test_patches": per_field_test_patches,
            }, f)
        logging.info(f"[build_test_dataset] saved {len(lowResFTLEorFLowMap_list)} samples to cache {cache_path}")
    except Exception as e:
        logging.info(f"[build_test_dataset] cache save failed: {e}")

    # Return numpy (FTLE) + torch (Pathlines), consistent with test usage

    return (
        lowResFTLEorFLowMap_list,
        highResFTLEorFLowMap_list,
        lowResPathlines_list
    )


def _normalize_flowmap(flowmap: np.ndarray|torch.Tensor):
    # Forward of _inverse_normalization. Must match the scaling applied in
    # FlowMapUpsamplingTrainDataset so train/test see the same input distribution.
    flowmap[...,0:3]=flowmap[...,0:3]/GLOBAL_UniformValueTemporalAndSpatial
    return flowmap


def _inverse_normalization(flowmap: np.ndarray|torch.Tensor):
    flowmap[...,0:3]=flowmap[...,0:3]*GLOBAL_UniformValueTemporalAndSpatial
    return flowmap


def _tiling_starts(length: int, k: int, stride: int) -> list[int]:
    """Sliding-window start indices identical to FlowMapUpsamplingTrainDataset, so test
    patches are tiled the same way the model was trained on."""
    length = int(length); k = int(k)
    if length <= 0:
        return []
    if k >= length:
        return [0]
    s = max(1, int(stride))
    starts = list(range(0, length - k + 1, s))
    if starts[-1] != length - k:
        starts.append(length - k)
    return starts


def _patch_blend_weight(h, w):
    """Separable raised-cosine (Hann) window, floored above 0, used to feather patch
    borders during overlap blending. Uniform weights would reproduce a plain mean but
    leave count-boundary seams; tapering the borders gives a seam-free weighted mean."""
    def hann(n):
        if n <= 1:
            return np.ones(n, dtype=np.float64)
        x = np.arange(n, dtype=np.float64)
        return 0.5 - 0.5 * np.cos(2.0 * np.pi * x / (n - 1))  # 0 at the two ends
    wy = 0.05 + 0.95 * hann(h)   # floor keeps every pixel's total weight > 0
    wx = 0.05 + 0.95 * hann(w)
    return (wy[:, None] * wx[None, :])  # [h, w]


def _sliding_window_predict_flowmap(model, low_grid, pathlines, UPsampling, patch_size,
                                    patch_stride, device, offset_dist):
    """Run the flow-map upsampler over the whole low-res grid in patch_size x patch_size
    windows (the same tiling used in training) and blend the high-res patch predictions
    by a feathered (Hann-windowed) weighted average over overlaps.

    Per patch: raw -> flowmap_to_relative -> normalize -> model -> inverse_normalize ->
    flowmap_from_relative -> raw, matching the training transforms exactly.

    Args:
        low_grid:  np [ny_low, nx_low, 5, 2, 3]   (raw physical coords)
        pathlines: torch [ny_low*nx_low, 5, L, 3] (geometry branch; ignored by the
                   fully-convolutional flow-map models but passed through for API parity)
        offset_dist: cross-line seed offset; rel_scale = 2*offset_dist.
    Returns:
        stitched high-res flow map: np [ny_low*UP, nx_low*UP, 5, 2, 3] (raw coords)
    """
    ny_low, nx_low = low_grid.shape[:2]
    UP = int(UPsampling)
    rel_scale = 2.0 * float(offset_dist)
    ny_hi, nx_hi = ny_low * UP, nx_low * UP

    low_flat = torch.from_numpy(low_grid).reshape(ny_low * nx_low, 5, 2, 3).float()
    acc = np.zeros((ny_hi, nx_hi, 5, 2, 3), dtype=np.float64)
    wsum = np.zeros((ny_hi, nx_hi, 1, 1, 1), dtype=np.float64)

    row_starts = _tiling_starts(ny_low, patch_size, patch_stride)
    col_starts = _tiling_starts(nx_low, patch_size, patch_stride)

    for ri, i0 in enumerate(row_starts):
        ph = min(int(patch_size), ny_low)
        row_idx = list(range(i0, i0 + ph))
        for ci, j0 in enumerate(col_starts):
            pw = min(int(patch_size), nx_low)
            col_idx = list(range(j0, j0 + pw))
            lo_flat = [r * nx_low + c for r in row_idx for c in col_idx]
            lo_t = torch.as_tensor(lo_flat, dtype=torch.long)

            patch = low_flat[lo_t].clone()                       # [P,5,2,3] raw
            patch = flowmap_to_relative(patch, rel_scale)        # Jacobian-aware rep
            fm = _normalize_flowmap(patch).reshape(1, ph * pw, 5, 2, 3).to(device).float()
            pl = pathlines[lo_t].unsqueeze(0).to(device).float() if pathlines is not None else None
            pred = model(fm, pl, hw=(ph, pw)).to(device).float()  # [1, ph*UP*pw*UP, 5,2,3] normalized-rel
            pred = _inverse_normalization(pred)                   # undo /GLOBAL
            pred = flowmap_from_relative(pred, rel_scale)         # back to absolute raw coords
            pred_patch = pred.reshape(ph * UP, pw * UP, 5, 2, 3).detach().cpu().numpy()

            # high-res placement; snap the last window to the boundary exactly like training
            hi_h, hi_w = ph * UP, pw * UP
            hi_i0 = int(round(i0 * UP))
            hi_j0 = int(round(j0 * UP))
            if ri == len(row_starts) - 1:
                hi_i0 = ny_hi - hi_h
            if ci == len(col_starts) - 1:
                hi_j0 = nx_hi - hi_w
            hi_i0 = max(0, hi_i0); hi_j0 = max(0, hi_j0)

            wgt = _patch_blend_weight(hi_h, hi_w)[..., None, None, None]  # [hi_h,hi_w,1,1,1]
            acc[hi_i0:hi_i0 + hi_h, hi_j0:hi_j0 + hi_w] += pred_patch * wgt
            wsum[hi_i0:hi_i0 + hi_h, hi_j0:hi_j0 + hi_w] += wgt

    wsum = np.maximum(wsum, 1e-12)
    return (acc / wsum).astype(np.float32)


def _fullgrid_predict_flowmap(model, low_grid, pathlines, UPsampling, device, offset_dist):
    """Single fully-convolutional forward over the WHOLE low-res grid (no tiling, no blend).

    These flow-map upsamplers are fully convolutional, so they accept any H x W via hw=.
    This is the standard SR inference path; it avoids the patch-border artifacts and
    overlap-averaging seams that sliding-window introduces. Same per-cell transforms as
    training: raw -> to_relative -> normalize -> model -> inverse_normalize -> from_relative.
    """
    ny_low, nx_low = low_grid.shape[:2]
    UP = int(UPsampling)
    rel_scale = 2.0 * float(offset_dist)
    low_flat = torch.from_numpy(low_grid).reshape(ny_low * nx_low, 5, 2, 3).float()
    patch = flowmap_to_relative(low_flat.clone(), rel_scale)
    fm = _normalize_flowmap(patch).reshape(1, ny_low * nx_low, 5, 2, 3).to(device).float()
    pl = pathlines.unsqueeze(0).to(device).float() if pathlines is not None else None
    pred = model(fm, pl, hw=(ny_low, nx_low)).to(device).float()
    pred = _inverse_normalization(pred)
    pred = flowmap_from_relative(pred, rel_scale)
    return pred.reshape(ny_low * UP, nx_low * UP, 5, 2, 3).detach().cpu().numpy().astype(np.float32)


def _flowmap_ftle_sensitivity_report(pred_grid, high_grid, offset_dist):
    """Quantify whether the model resolves the FTLE-relevant signal.

    FTLE's Jacobian is J = (pEnd[x+]-pEnd[x-]) / dx0 with dx0 = 2*offset_dist. What
    controls FTLE is therefore the *neighbour endpoint differences* d_x=pEnd[x+]-pEnd[x-]
    and d_y=pEnd[y+]-pEnd[y-], NOT the absolute positions (the center line's absolute
    position can be off without hurting FTLE). We report the model's relative error on
    those differences: ratio<1 means the Jacobian is resolved and FTLE PSNR can be good.
    """
    hg = torch.as_tensor(high_grid).float().reshape(-1, 5, 2, 3)
    pg = torch.as_tensor(pred_grid).float().reshape(-1, 5, 2, 3)
    valid = (hg.abs().reshape(hg.shape[0], -1).sum(1) > 0)
    if int(valid.sum()) == 0:
        logging.warning("[FlowMap][sensitivity] no valid cells; skip report.")
        return
    he, pe = hg[valid, :, 1, :2], pg[valid, :, 1, :2]   # tail endpoints (x,y) of the 5 lines
    # FTLE numerator differences (true vs predicted)
    true_dx = he[:, 1, :] - he[:, 2, :];  pred_dx = pe[:, 1, :] - pe[:, 2, :]
    true_dy = he[:, 3, :] - he[:, 4, :];  pred_dy = pe[:, 3, :] - pe[:, 4, :]
    sig = torch.cat([true_dx.norm(dim=-1), true_dy.norm(dim=-1)])
    err = torch.cat([(pred_dx - true_dx).norm(dim=-1), (pred_dy - true_dy).norm(dim=-1)])
    ratio = (err.median() / sig.median().clamp_min(1e-12)).item()
    dx0 = 2.0 * float(offset_dist)
    logging.info(
        f"[FlowMap][sensitivity] dx0(=2*offset_dist)={dx0:.4e}  "
        f"FTLE-signal median|pEnd[x+]-pEnd[x-]|={sig.median().item():.4e}  "
        f"model diff-error={err.median().item():.4e}  rel-error(on Jacobian)={ratio*100:.1f}%")
    if ratio > 0.5:
        logging.warning(
            f"[FlowMap][sensitivity] {ratio*100:.0f}% error on the FTLE Jacobian differences: "
            f"FTLE PSNR will be limited. Lower it with more training / better representation.")


test_times=0

def test_UpsamplingModel(config, model,test_dataset, device,visualize=True, show_plot=False):
    # visualize: produce+save the FTLE figure (every 10 tests, and whenever show_plot).
    # show_plot: pop an interactive (blocking) window; keep False for per-epoch eval so a
    #            200-epoch run never stalls, set True only for the final best-checkpoint test.
    global test_times
    test_times += 1
    with torch.no_grad():
        model.to(device).eval()
        lowResFTLE_all, highResFTLE_all, lowResPathlines_all= test_dataset
        if lowResFTLE_all is not None and lowResPathlines_all is not None and highResFTLE_all is not None and config['mode'] == 'upsamplingFLowMap':
            # Flow-map upsampling eval. The model predicts the high-res flow map; the metric
            # is computed in FTLE space (FTLE = computeFTLEFromPathlineCrossPrimitive on the
            # [N,5,2,3] flow map, treating the 2 endpoints as the line's head/tail).
            sample_count = int(len(lowResPathlines_all))
            UPsampling = int(config.dataset.UPsampling)
            dt = float(config.pcds.dt)
            # Mirror the training data construction: tile the low-res grid into
            # patchSize x patchSize windows, predict each, then blend. patchSize must match
            # training (config.dataset.patchSize); test.patchStride controls inference overlap
            # and defaults to half the patch (override via config test: block).
            patch_size = int(getattr(config.dataset, 'patchSize', 32))
            patch_stride = int(config.test.patchStride)
            off = float(config.pcds.offset_dist)
            # Inference mode for the official metric: 'fullgrid' (single conv pass, default)
            # or 'sliding' (tiled+blended). compareInference logs BOTH every test so the cost
            # of sliding can be measured against fullgrid on identical weights.
            inference_mode = str(getattr(config.test, 'inferenceMode', 'fullgrid'))
            compare_inference = bool(getattr(config.test, 'compareInference', False))

            def _predict(mode, low_grid, pl):
                if mode == 'sliding':
                    return _sliding_window_predict_flowmap(
                        model, low_grid, pl, UPsampling, patch_size, patch_stride, device, offset_dist=off)
                return _fullgrid_predict_flowmap(model, low_grid, pl, UPsampling, device, offset_dist=off)

            def _ftle_grid_from_flowmap(field_grid):
                # field_grid: np or torch [ny, nx, 5, 2, 3] -> FTLE np [ny, nx]
                fg = torch.from_numpy(field_grid) if isinstance(field_grid, np.ndarray) else field_grid
                fg = fg.float()
                ny_, nx_ = fg.shape[0], fg.shape[1]
                ftle = computeFTLEFromPathlineCrossPrimitive(fg.reshape(ny_ * nx_, 5, 2, 3), vectorfield_dt=dt)
                return ftle.reshape(ny_, nx_).detach().cpu().numpy().astype(np.float32)

            def _interp_flowmap(low_grid_np, ny_hi, nx_hi, mode):
                # low_grid_np: [ny,nx,5,2,3] -> interpolated [ny_hi,nx_hi,5,2,3]
                t = torch.from_numpy(low_grid_np).float()
                ny_, nx_ = t.shape[0], t.shape[1]
                img = t.reshape(ny_, nx_, 30).permute(2, 0, 1).unsqueeze(0)  # [1,30,ny,nx]
                up = F.interpolate(img, size=(ny_hi, nx_hi), mode=mode,
                                   align_corners=False if mode in ('bilinear', 'bicubic') else None)
                return up.squeeze(0).permute(1, 2, 0).reshape(ny_hi, nx_hi, 5, 2, 3)

            mse_sum = mae_sum = maxe_sum = psnr_sum = 0.0
            psnr_bilinear_sum = psnr_cubic_sum = 0.0
            psnr_alt_sum = 0.0  # the non-official inference mode (for compareInference)
            for test_i in range(sample_count):
                low_grid = lowResFTLE_all[test_i]          # np [ny_low,nx_low,5,2,3]
                high_grid = highResFTLE_all[test_i]        # np [ny_hi, nx_hi, 5,2,3]
                lowResPathlinesPreprocessed = lowResPathlines_all[test_i]  # torch [ny_low*nx_low,5,L,3]
                ny_low, nx_low = low_grid.shape[:2]
                ny_hi, nx_hi = high_grid.shape[:2]

                label_ftle = _ftle_grid_from_flowmap(high_grid)

                # official prediction via the configured inference mode
                pred_grid = _predict(inference_mode, low_grid, lowResPathlinesPreprocessed)
                pred_ftle = _ftle_grid_from_flowmap(pred_grid)
                mse, mae, maxe, psnr = compute_metrics(label_ftle, pred_ftle)

                # A/B: also score the other mode on the SAME weights to measure the gap
                if compare_inference:
                    alt_mode = 'sliding' if inference_mode == 'fullgrid' else 'fullgrid'
                    alt_grid = _predict(alt_mode, low_grid, lowResPathlinesPreprocessed)
                    _, _, _, psnr_alt = compute_metrics(label_ftle, _ftle_grid_from_flowmap(alt_grid))
                    psnr_alt_sum += psnr_alt

                if test_times == 1:
                    bil = _interp_flowmap(low_grid, ny_hi, nx_hi, 'bilinear')
                    cub = _interp_flowmap(low_grid, ny_hi, nx_hi, 'bicubic')
                    _, _, _, psnr_bilinear = compute_metrics(label_ftle, _ftle_grid_from_flowmap(bil))
                    _, _, _, psnr_cubic = compute_metrics(label_ftle, _ftle_grid_from_flowmap(cub))
                    psnr_bilinear_sum += psnr_bilinear
                    psnr_cubic_sum += psnr_cubic

                # One-time sensitivity check: FTLE = (neighbor endpoint difference)/dx0 with
                # dx0=2*offset_dist (tiny). If the model's absolute-position error exceeds that
                # endpoint-difference signal, FTLE PSNR is doomed regardless of flow-map MSE.
                # This is the diagnostic for "low train MSE but terrible FTLE PSNR".
                if test_times == 1 and test_i == 0:
                    _flowmap_ftle_sensitivity_report(pred_grid, high_grid,
                                                     offset_dist=float(config.pcds.offset_dist))

                # Render the FTLE image for the last test slice. Save it every 10 tests (and
                # on the final best-checkpoint test), but only POP a blocking window when
                # show_plot is set, so a 200-epoch run saves figures without ever stalling.
                if visualize and (test_times % 10 == 0 or show_plot) and test_i == sample_count - 1:
                    # The last test slice belongs to the last test field (fields are appended
                    # in order), so use that field's name + domain bounds for the render.
                    vf_cfg = config.test.vectorfield
                    vf_name = vf_cfg[-1] if isinstance(vf_cfg, (list, tuple)) else vf_cfg
                    vectorfield = load_UnsteadyVectorFields_netCDFOrAnalytical(config.dataset.dat_dir, [vf_name])[-1]
                    low_ftle = _ftle_grid_from_flowmap(low_grid)  # low-res FTLE (raw coords)
                    vis_dir = os.path.join(config.cache_dir, "vis")
                    os.makedirs(vis_dir, exist_ok=True)
                    save_path = os.path.join(vis_dir, f"{config['mode']}__{config['model']['NAME']}__{vf_name}_test{test_times}.png")
                    visualize_FTLEUpampling(label_ftle, pred_ftle, low_ftle,
                                            vectorfield.domainMinBoundary, vectorfield.domainMaxBoundary,
                                            save_path=save_path, show=bool(show_plot))
                    logging.info(f"[FlowMap] saved FTLE visualization to {save_path}")

                mse_sum += mse; mae_sum += mae; maxe_sum += maxe; psnr_sum += psnr

            mse = mse_sum / sample_count
            mae = mae_sum / sample_count
            maxe = maxe_sum / sample_count
            psnr = psnr_sum / sample_count
            if test_times == 1:
                psnr_bilinear = psnr_bilinear_sum / sample_count
                psnr_cubic = psnr_cubic_sum / sample_count
                logging.info(f"[FlowMap] baseline psnr_bilinear={psnr_bilinear:.6f}, psnr_cubic={psnr_cubic:.6f}")
            if compare_inference:
                alt_mode = 'sliding' if inference_mode == 'fullgrid' else 'fullgrid'
                logging.info(f"[FlowMap][A/B] official={inference_mode} psnr={psnr:.2f} dB  |  "
                             f"{alt_mode} psnr={psnr_alt_sum / sample_count:.2f} dB  "
                             f"(delta={psnr - psnr_alt_sum / sample_count:+.2f} dB)")
            return {"mse": mse, "mae": mae, "maxe": maxe, "psnr": psnr}
        else:
            raise ValueError(f"TEST Failed: Unknown mode: {config['mode']} or test_dataset is None")


def _sr_metrics(gt, pred):
    """MSE/MAE/MaxE/PSNR in raw flow-map space (PSNR uses the GT dynamic range)."""
    gt = np.asarray(gt, np.float32); pred = np.asarray(pred, np.float32)
    mse = float(np.mean((gt - pred) ** 2))
    mae = float(np.mean(np.abs(gt - pred)))
    maxe = float(np.max(np.abs(gt - pred)))
    rng = max(float(gt.max() - gt.min()), 1e-12)
    psnr = float('inf') if mse <= 1e-20 else float(20 * np.log10(rng) - 10 * np.log10(mse))
    return mse, mae, maxe, psnr


def test_FlowMapSR(config, model, test_dataset, device, visualize=True, show_plot=False):
    """Paper-style evaluation: predict the high-res flow map and report MSE/PSNR in
    FLOW-MAP space against cubic / bilinear interpolation (Jakob et al. 2020)."""
    global test_times
    test_times += 1
    k = int(config.dataset.UPsampling)
    model.to(device).eval()
    n = len(test_dataset)
    acc = {m: np.zeros(4) for m in ('model', 'cubic', 'bilinear')}
    with torch.no_grad():
        for ti, s in enumerate(test_dataset):
            lo, hi = s['lo'], s['hi']               # raw [ny,nx,2], [ny*k,nx*k,2]
            ny_hi, nx_hi = hi.shape[:2]
            # model: per-flow-map normalize (from low-res) -> net -> denormalize
            lo_n, mean, scale = flowmap_unit_normalize(lo)
            x = torch.from_numpy(lo_n).permute(2, 0, 1).unsqueeze(0).to(device).float()
            pred_n = model(x)[0].permute(1, 2, 0)   # [ny_hi,nx_hi,2]
            pred = flowmap_unit_denormalize(pred_n, mean, scale)
            if pred.shape[:2] != (ny_hi, nx_hi):
                pred = pred[:ny_hi, :nx_hi]
            # interpolation baselines on the raw low-res flow map
            lo_t = torch.from_numpy(lo).permute(2, 0, 1).unsqueeze(0).float()
            cub = F.interpolate(lo_t, size=(ny_hi, nx_hi), mode='bicubic', align_corners=False)[0].permute(1, 2, 0).numpy()
            bil = F.interpolate(lo_t, size=(ny_hi, nx_hi), mode='bilinear', align_corners=False)[0].permute(1, 2, 0).numpy()
            for name, p in (('model', pred), ('cubic', cub), ('bilinear', bil)):
                acc[name] += np.array(_sr_metrics(hi, p))

            if visualize and (test_times % 10 == 0 or show_plot) and ti == n - 1:
                xs, ys, tau = s['xs'], s['ys'], s['tau']
                ftle_gt = ftle_from_endpos_grid(hi, xs, ys, tau)
                ftle_pred = ftle_from_endpos_grid(pred, xs, ys, tau)
                ftle_lo = ftle_from_endpos_grid(lo, xs[::k], ys[::k], tau)
                vis_dir = os.path.join(config.cache_dir, "vis"); os.makedirs(vis_dir, exist_ok=True)
                save_path = os.path.join(vis_dir, f"flowmapSR__{config['model']['NAME']}__{s['name']}_test{test_times}.png")
                visualize_FTLEUpampling(ftle_gt, ftle_pred, ftle_lo, s['domMin'], s['domMax'],
                                        save_path=save_path, show=bool(show_plot))
                logging.info(f"[FlowMapSR] saved FTLE visualization to {save_path}")

    res = {m: acc[m] / max(n, 1) for m in acc}
    logging.info(f"[FlowMapSR] flow-map PSNR  model={res['model'][3]:.2f} dB  |  "
                 f"cubic={res['cubic'][3]:.2f} dB  |  bilinear={res['bilinear'][3]:.2f} dB  "
                 f"(model-cubic={res['model'][3]-res['cubic'][3]:+.2f} dB)")
    return {"mse": float(res['model'][0]), "mae": float(res['model'][1]),
            "maxe": float(res['model'][2]), "psnr": float(res['model'][3])}


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

    test_task_func_name = config.test.tasks
    if test_task_func_name is not None:
        task_init_fn=eval(test_task_func_name)
        assert task_init_fn is not None and callable(task_init_fn)
    else:
        task_init_fn=None
    
    total_iterations=0
    for epoch in range(epochs):
        # IMPORTANT: the per-epoch test below calls model.eval(); without re-asserting
        # train() here, every epoch after the first would train in eval mode (BatchNorm
        # frozen on epoch-0 running stats). This crippled all BatchNorm models (UNet/FMT/
        # DCT) while leaving the BN-free ESPCN unaffected.
        model.train()
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

            # Stage-3 debug (first iter each epoch): confirm the model is actually in
            # train() mode (guards the eval-mode bug) and pred/label shapes/finiteness match.
            if it == 0:
                dbg.check_train_step(model, pred, label_y)

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
        # Final eval on the best checkpoint: this is the ONLY call that pops a blocking window.
        test_result= task_init_fn(config, model,test_dataset, device=str(device), visualize=True, show_plot=True) if task_init_fn is not None and callable(task_init_fn) else None
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



    
if __name__=="__main__":
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"torch.cuda.is_available(): {torch.cuda.is_available()}")  # Should return True
    logging.info(f"torch.version.cuda: {torch.version.cuda}")         # Should print the CUDA version PyTorch was built with
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    cfg=argParseAndPrepareConfig()
    cfg["gitInfo"]=get_git_commit_id()
    dbg.set_debug(int(getattr(cfg, 'debug', 0)))  # pipeline validation checks (0=off)
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
    # if mode == 'point_FTLE_regression':
    #     config = EasyConfig()
    #     netCDF = NetCDFLoader()
    #     config.load("config/PointWiseFTLERegressor.yaml", recursive=False)
    #     vectorfield_datapath=os.path.join(config.dataset.dat_dir, f"{config.dataset.test_name}.{config.dataset.extension}")
    #     vectorfield = netCDF.load_vector_field2d(vectorfield_datapath)
    #     config['vectorfield']=vectorfield
   
    #     model = build_model(config, device)
    #     # use Dataset + DataLoader (support shuffle / multi-threading etc.)
    #     dataset = PointWiseFTLETrainDataset( config=config )
    #     config['ftle_min']=dataset.ftle_min
    #     config['ftle_max']=dataset.ftle_max
    #     train_model(config,model,dataset,device)
    # if mode == 'upsamplingFTLE':
    #     # future mode: low resolution pathlines + low resolution FTLE -> high resolution FTLE
    #     dataset = FTLEUpsamplingTrainDataset(cfg, useCacheSystem=True)
    #     test_dataset=build_test_dataset(cfg)
    #     logging.info(f"build_dataset done, train dataset lenth: {dataset.lowResFTLE.shape[0]}")
    #     lowResX,lowResY=dataset.lowResFTLE[0].shape[0],dataset.lowResFTLE[0].shape[1]
    #     cfg['lowResX']=lowResX
    #     cfg['lowResY']=lowResY
    #     cfg['ftle_min']=dataset.ftle_min
    #     cfg['ftle_max']=dataset.ftle_max
    #     model = build_model(cfg, device)
    #     train_model(cfg,model,dataset,device,test_dataset)
    if mode == 'upsamplingFLowMap':
        dataset = FlowMapUpsamplingTrainDataset(cfg, useCacheSystem=True)
        test_dataset=build_test_dataset(cfg)
        logging.info(f"upsamplingFLowMap task build_dataset done, train dataset lenth: {dataset.lowResFlowMap.shape[0]}")
        # lowResFlowMap patches are flat-indexed [P, 5, 2, 3]; the low-res spatial extent
        # is the (square) patch grid, so derive lowResX/Y from patchSize.
        patch_size=int(getattr(cfg.dataset, 'patchSize', 32))
        cfg['lowResX']=patch_size
        cfg['lowResY']=patch_size
        model = build_model(cfg, device)
        train_model(cfg,model,dataset,device,test_dataset)
    elif mode == 'flowmapSR':
        # Paper-faithful flow-map super-resolution (Jakob et al. 2020): 2-channel end-position
        # flow map, low-res = high-res subsampled by k, MSE/PSNR evaluated in flow-map space.
        dataset = FlowMapSRTrainDataset(cfg, useCacheSystem=True)
        test_dataset = build_FlowMapSR_test_dataset(cfg)
        logging.info(f"flowmapSR task build_dataset done, train patches: {len(dataset)}")
        model = build_model(cfg, device)
        train_model(cfg, model, dataset, device, test_dataset)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    

