"""Diagnostic: why does a low-MSE flow-map model give terrible FTLE PSNR,
and why is bicubic so much worse than bilinear?

Loads ONE real test slice from the test cache and measures:
  - the magnitude of the cross-line offsets dx0/dy0 (the FTLE denominator)
  - the magnitude of the FTLE numerator |pEnd[x+]-pEnd[x-]| (the signal)
  - how a tiny absolute-position error (matching the train MSE) wrecks FTLE
  - bilinear vs bicubic FTLE PSNR (reproduce 39 vs 15 dB)
  - whether zeroed invalid cells create interpolation blow-ups
"""
import numpy as np
import torch
import torch.nn.functional as F

from DeepUtils.utils import EasyConfig
from FLowUtils.flowDatasetUtils.NetCDF_AmiraLoader import relocate_flow2d_dataset_folder
from FMT_Utils.FTLE_fitting_utils import (
    computeFTLEFromPathlineCrossPrimitive, compute_metrics,
    GLOBAL_UniformValueTemporalAndSpatial as G,
)
from FTLE_experiment import build_test_dataset

torch.manual_seed(0); np.random.seed(0)
DT = 0.005


def ftle_of(grid):
    fg = torch.from_numpy(grid) if isinstance(grid, np.ndarray) else grid
    fg = fg.float()
    ny, nx = fg.shape[0], fg.shape[1]
    f = computeFTLEFromPathlineCrossPrimitive(fg.reshape(ny * nx, 5, 2, 3), vectorfield_dt=DT)
    return f.reshape(ny, nx).numpy().astype(np.float32)


def interp(grid, ny_hi, nx_hi, mode):
    t = torch.from_numpy(grid).float()
    ny, nx = t.shape[0], t.shape[1]
    img = t.reshape(ny, nx, 30).permute(2, 0, 1).unsqueeze(0)
    up = F.interpolate(img, size=(ny_hi, nx_hi), mode=mode,
                       align_corners=False if mode in ('bilinear', 'bicubic') else None)
    return up.squeeze(0).permute(1, 2, 0).reshape(ny_hi, nx_hi, 5, 2, 3).numpy()


def main():
    cfg = EasyConfig(); cfg.load("config/FlowMapUpsampling.yaml", recursive=True)
    relocate_flow2d_dataset_folder(cfg)
    low_all, high_all, _pl = build_test_dataset(cfg)
    print(f"\n# test slices: {len(low_all)}")

    for si in [0, len(low_all) - 1]:
        low_grid, high_grid = low_all[si], high_all[si]
        ny_low, nx_low = low_grid.shape[:2]
        ny_hi, nx_hi = high_grid.shape[:2]
        print("\n" + "=" * 70)
        print(f"slice {si}: low {ny_low}x{nx_low}  ->  high {ny_hi}x{nx_hi}")

        hg = torch.from_numpy(high_grid).reshape(ny_hi * nx_hi, 5, 2, 3).float()
        pStart, pEnd = hg[:, :, 0, :2], hg[:, :, 1, :2]

        # valid (non-zeroed) groups
        valid = (hg.abs().reshape(hg.shape[0], -1).sum(1) > 0)
        print(f"valid (non-zero) cells: {valid.float().mean().item()*100:.1f}%   "
              f"({int(valid.sum())}/{hg.shape[0]})")

        # --- FTLE denominator: initial cross offsets ---
        dx0 = (pStart[valid, 1, 0] - pStart[valid, 2, 0])
        dy0 = (pStart[valid, 3, 1] - pStart[valid, 4, 1])
        print(f"dx0 (x+ minus x- seed): mean|.|={dx0.abs().mean():.4e}  "
              f"min|.|={dx0.abs().min():.4e}  max|.|={dx0.abs().max():.4e}")
        print(f"dy0 (y+ minus y- seed): mean|.|={dy0.abs().mean():.4e}")

        # --- FTLE numerator: end-position differences (the signal) ---
        numx = (pEnd[valid, 1, :] - pEnd[valid, 2, :]).norm(dim=-1)
        numy = (pEnd[valid, 3, :] - pEnd[valid, 4, :]).norm(dim=-1)
        print(f"|pEnd[x+]-pEnd[x-]| signal: mean={numx.mean():.4e}  median={numx.median():.4e}  max={numx.max():.4e}")
        print(f"|pEnd[y+]-pEnd[y-]| signal: mean={numy.mean():.4e}  median={numy.median():.4e}  max={numy.max():.4e}")

        # --- absolute position scale & the train-MSE-equivalent error ---
        xy_abs = pEnd[valid].abs()
        print(f"absolute end-position |x,y|: mean={xy_abs.mean():.3f}  max={xy_abs.max():.3f}")
        train_mse_norm = 5e-4
        eps_phys = (train_mse_norm ** 0.5) * G
        print(f"train MSE(norm)~{train_mse_norm}  ->  per-coord RMS error ~{eps_phys:.4f} physical units")
        print(f"  >> compare to FTLE signal median |pEnd diff| ~ {numx.median():.4e}")
        print(f"  >> error/signal ratio ~ {eps_phys/float(numx.median()):.1f}x")

        label = ftle_of(high_grid)
        print(f"label FTLE: min={label.min():.3f} max={label.max():.3f} "
              f"mean={label.mean():.3f} std={label.std():.3f}")

        # --- baselines ---
        for mode in ('bilinear', 'bicubic'):
            f = ftle_of(interp(low_grid, ny_hi, nx_hi, mode))
            _, _, mx, ps = compute_metrics(label, f)
            print(f"baseline {mode:8s}: PSNR={ps:7.3f} dB  maxerr={mx:.3f}  "
                  f"pred FTLE [min {f.min():.2f}, max {f.max():.2f}]")

        # --- noise experiment: add absolute-position error matching train MSE ---
        for eps in (0.0, 0.005, 0.02, 0.05, eps_phys):
            noisy = high_grid.copy()
            noisy[..., :2] += np.random.randn(*noisy[..., :2].shape).astype(np.float32) * eps
            noisy[high_grid == 0] = 0.0   # keep invalid cells zeroed
            f = ftle_of(noisy)
            _, _, _, ps = compute_metrics(label, f)
            print(f"  GT + N(0,{eps:.4f}) on xy -> FTLE PSNR={ps:7.3f} dB")


if __name__ == "__main__":
    main()
