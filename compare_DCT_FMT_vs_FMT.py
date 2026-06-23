"""
FTLE upsampling A/B benchmark: DCT_FMT (Fourier tokenizer) vs FMT (EncNPNew).

Both models share the *exact* same sliding-window mechanism, UNet backend and
upsampling head (FTLEupsamplingFMT_UnetV2 vs FTLEupsamplingDCT_FMT_UnetV2); only
the per-window pathline tokenizer differs:

    FMT      = EncNPNew    : KNN + positional encoding + max/mean pooling
    DCT_FMT  = DCT_FMT     : |FFT(z=x+iy)| temporal-frequency descriptor (no params)

For context the table also includes two classic super-resolution baselines that use
ONLY the low-resolution FTLE scalar field (no pathline features):
    ESPCN              : classic CNN super-resolution
    UpsamplingUnetModel: classic UNet
so the gain from the pathline tokenizers is visible against vanilla SR.

The training data, the cached test set, and the random seed are identical across
the two runs, so the difference in PSNR/MSE/MAE is attributable to the tokenizer.

Usage:
    python compare_DCT_FMT_vs_FMT.py --config config/FTLEUpsampling.yaml
    # optional: restrict / extend the list of models being compared
    python compare_DCT_FMT_vs_FMT.py --config config/FTLEUpsampling.yaml \
        --models FTLEUpsamplingFMT_UnetV2 FTLEUpsamplingDCT_FMT_UnetV2
"""

import os
import copy
import logging
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # benchmark runs headless; never pop up / block on plt.show()
import matplotlib.pyplot as plt

from DeepUtils.utils import EasyConfig
from DeepUtils.MiscFunctions import get_git_commit_id, print_args
from FLowUtils.flowDatasetUtils.NetCDF_AmiraLoader import relocate_flow2d_dataset_folder
from FMT_Utils.model_zoo import build_model, calculate_model_parm_size
from FMT_Utils.FTLE_fitting_utils import FTLEUpsamplingTrainDataset, compute_metrics
from FMT_Utils import debug_checks as dbg

# Reuse the dataset builders / train / test loop from the main experiment file.
import FTLE_experiment
from FTLE_experiment import (
    build_test_dataset,
    test_UpsamplingModel,
    train_model,
    runNameTagGenerator_fmt,
)

# The end-of-training visualize=True step calls a blocking, GUI-oriented helper
# (plt.show(block=True)); a benchmark doesn't need it, so disable it cleanly.
FTLE_experiment.visualize_FTLEUpampling = lambda *a, **k: None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# Classic scalar-field super-resolution baselines (no pathline features) followed by
# the two pathline tokenizers, so the table shows what FMT / DCT_FMT add on top.
DEFAULT_MODELS = [
    "ESPCN",                         # classic CNN super-resolution baseline
    "UpsamplingUnetModel",           # classic UNet baseline
    "FTLEUpsamplingFMT_UnetV2",      # FMT tokenizer (EncNPNew: KNN + PosE)
    "FTLEUpsamplingDCT_FMT_UnetV2",  # DCT_FMT tokenizer (training-free Fourier)
]


def set_reproducible(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/FTLEUpsampling.yaml")
    parser.add_argument("--models", type=str, nargs="*", default=DEFAULT_MODELS,
                        help="model.NAME values to compare")
    parser.add_argument("--epochs", type=int, default=None, help="override epochs for a quick run")
    parser.add_argument("--debug", type=int, default=None,
                        help="pipeline debug-check level: 0=off, 1=warn+log, 2=strict/raise")
    return parser.parse_args()


def _tiling_starts(length, k, stride):
    if k >= length:
        return [0]
    s = max(1, int(stride))
    starts = list(range(0, length - k + 1, s))
    last = length - k
    if starts[-1] != last:
        starts.append(last)
    return starts


def _reconstruct_pred_grid(cfg, model, low_field, pl_pre, ny_hi, nx_hi, device):
    """Reconstruct the full high-res prediction for one test sample via the same
    overlap-averaged patch tiling that test_UpsamplingModel uses."""
    patch_size = int(getattr(cfg.dataset, 'patchSize', 32))
    patch_stride = int(getattr(cfg.dataset, 'patchStride', 2)) * 2
    UP = int(cfg.dataset.UPsampling)
    ftle_min = float(cfg.ftle_min); ftle_max = float(cfg.ftle_max)
    ny_low, nx_low = low_field.shape

    low_norm = torch.from_numpy(
        (np.clip(low_field, ftle_min, ftle_max) - ftle_min) / max(1e-12, (ftle_max - ftle_min))
    ).float()

    row_starts = _tiling_starts(ny_low, patch_size, patch_stride)
    col_starts = _tiling_starts(nx_low, patch_size, patch_stride)
    pred_grid = np.zeros((ny_hi, nx_hi), np.float32)
    wgrid = np.zeros((ny_hi, nx_hi), np.float32)

    model.eval()
    with torch.no_grad():
        for i0 in row_starts:
            i1 = min(i0 + patch_size, ny_low)
            for j0 in col_starts:
                j1 = min(j0 + patch_size, nx_low)
                hi_h = max(1, patch_size * UP); hi_w = max(1, patch_size * UP)
                hi_i0 = int(round(i0 * UP)); hi_j0 = int(round(j0 * UP))
                if i0 == row_starts[-1]: hi_i0 = ny_hi - hi_h
                if j0 == col_starts[-1]: hi_j0 = nx_hi - hi_w
                hi_i1 = hi_i0 + hi_h; hi_j1 = hi_j0 + hi_w

                lr_patch = low_norm[i0:i1, j0:j1].unsqueeze(0).to(device).float()
                idx = []
                for rr in range(i0, i1):
                    base = rr * nx_low
                    for cc in range(j0, j1):
                        idx.append(base + cc)
                pl_patch = pl_pre[torch.as_tensor(idx, dtype=torch.long)].unsqueeze(0).to(device).float()
                patch = model(lr_patch, pl_patch).squeeze(0).detach().cpu().numpy()

                hi_i0 = max(0, hi_i0); hi_j0 = max(0, hi_j0)
                hi_i1 = min(ny_hi, hi_i1); hi_j1 = min(nx_hi, hi_j1)
                pred_grid[hi_i0:hi_i1, hi_j0:hi_j1] += patch[:hi_i1 - hi_i0, :hi_j1 - hi_j0]
                wgrid[hi_i0:hi_i1, hi_j0:hi_j1] += 1.0

    wgrid = np.clip(wgrid, 1.0, None)
    pred_grid = pred_grid / wgrid
    pred_grid = pred_grid * (ftle_max - ftle_min) + ftle_min
    return pred_grid


def save_upsampling_figures(cfg, model, test_dataset, device, out_dir, model_name, max_figs=4):
    """Save pred-vs-ground-truth FTLE comparison figures (low-res / pred / GT / |error|)."""
    os.makedirs(out_dir, exist_ok=True)
    low_all, high_all, pl_all = test_dataset
    n = min(int(max_figs), len(low_all))
    for i in range(n):
        low = np.asarray(low_all[i], dtype=np.float32)
        high = np.asarray(high_all[i], dtype=np.float32)
        pl_pre = pl_all[i]
        ny_hi, nx_hi = high.shape
        pred = _reconstruct_pred_grid(cfg, model, low, pl_pre, ny_hi, nx_hi, device)
        mse, mae, maxe, psnr = compute_metrics(high, pred)

        vmin = float(min(high.min(), pred.min()))
        vmax = float(max(high.max(), pred.max()))
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), constrained_layout=True)
        panels = [
            (low, "low-res input", dict(cmap='viridis')),
            (pred, f"prediction (PSNR {psnr:.2f} dB)", dict(cmap='viridis', vmin=vmin, vmax=vmax)),
            (high, "ground truth", dict(cmap='viridis', vmin=vmin, vmax=vmax)),
            (np.abs(pred - high), f"|error| (MAE {mae:.3f})", dict(cmap='magma')),
        ]
        for ax, (img, title, kw) in zip(axes, panels):
            im = ax.imshow(img, origin='lower', **kw)
            ax.set_title(f"{title}\n{img.shape[0]}x{img.shape[1]}", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"{model_name} — test sample {i}", fontsize=12)
        path = os.path.join(out_dir, f"{model_name}_sample{i}.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"[figure] saved {path}")


def run_one_model(base_cfg, model_name, train_dataset, test_dataset, device):
    """Train `model_name` on the shared dataset and return its best test metrics."""
    cfg = copy.deepcopy(base_cfg)
    cfg["model"]["NAME"] = model_name
    set_reproducible(int(cfg.get("seed", 42)))  # identical init/order per model

    run_name, run_tags = runNameTagGenerator_fmt(cfg, cfg["mode"])
    cfg["run_name"] = run_name
    logging.info(f"\n========== Training {model_name} ==========")

    model = build_model(cfg, device)
    try:
        size = calculate_model_parm_size(model)
        logging.info(f"[{model_name}] trainable params = {size['trainable_param_count']:,} "
                     f"({size['total_size_mb']:.2f} MB)")
    except Exception as e:  # non-fatal
        logging.info(f"[{model_name}] param-size probe failed: {e}")

    # train_model loads the best checkpoint back into `model` before returning.
    train_model(cfg, model, train_dataset, device, test_dataset)

    metrics = test_UpsamplingModel(cfg, model, test_dataset, device, visualize=False)

    # Save pred-vs-GT comparison figures to a temp folder in the repo.
    try:
        save_upsampling_figures(cfg, model, test_dataset, device,
                                os.path.join("outputs", "figures"), model_name)
    except Exception as e:
        logging.warning(f"[{model_name}] figure generation failed: {e}")

    return metrics


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"device={device}, torch={torch.__version__}, cuda={torch.version.cuda}")

    cfg = EasyConfig()
    cfg.load(args.config, recursive=True)
    cfg["config_yaml"] = args.config
    cfg["gitInfo"] = get_git_commit_id()
    if args.epochs is not None:
        cfg["epochs"] = int(args.epochs)
    cfg["wandb"] = False  # the A/B harness never logs to wandb

    # Pipeline debug/validation checks (FTLE / dataset / train / test). CLI overrides config.
    dbg.set_debug(int(args.debug) if args.debug is not None else int(getattr(cfg, 'debug', 0)))

    assert cfg["mode"] == "upsamplingFTLE", \
        f"This harness only supports mode=upsamplingFTLE, got {cfg['mode']}"

    relocate_flow2d_dataset_folder(cfg)
    print_args(cfg, printer=logging.info)

    # Build the (model-independent) datasets ONCE and share them across models.
    train_dataset = FTLEUpsamplingTrainDataset(cfg, useCacheSystem=True)
    test_dataset = build_test_dataset(cfg)
    cfg["lowResX"] = int(train_dataset.lowResFTLE[0].shape[0])
    cfg["lowResY"] = int(train_dataset.lowResFTLE[0].shape[1])
    cfg["ftle_min"] = train_dataset.ftle_min
    cfg["ftle_max"] = train_dataset.ftle_max
    logging.info(f"train samples={train_dataset.lowResFTLE.shape[0]}, "
                 f"lowRes=({cfg['lowResX']},{cfg['lowResY']})")

    results = {}
    for model_name in args.models:
        try:
            results[model_name] = run_one_model(cfg, model_name, train_dataset, test_dataset, device)
        except Exception as e:
            logging.exception(f"[{model_name}] failed: {e}")
            results[model_name] = None

    # Pretty comparison table.
    print("\n" + "=" * 78)
    print("FTLE Upsampling — classic SR (CNN/UNet) vs FMT vs DCT_FMT")
    print("=" * 78)
    header = f"{'model':<34}{'PSNR(dB)':>11}{'MSE':>11}{'MAE':>11}{'MAXE':>10}"
    print(header)
    print("-" * 78)
    for model_name, m in results.items():
        if m is None:
            print(f"{model_name:<34}{'FAILED':>11}")
            continue
        print(f"{model_name:<34}{m['psnr']:>11.3f}{m['mse']:>11.5f}{m['mae']:>11.5f}{m['maxe']:>10.4f}")
    print("=" * 78)
    print("(bicubic/bilinear baselines are logged once above as 'baseline psnr_*')")


if __name__ == "__main__":
    main()
