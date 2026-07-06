"""Paper-faithful flow-map super-resolution (Jakob et al. 2020, "A Fluid Flow Data Set
for Machine Learning and its Application to Neural Flow Map Interpolation").

Key differences from the cross-primitive FTLE pipeline in FTLE_fitting_utils.py:
  * Flow map = 2-channel particle END position phi(x0)=(end_x,end_y) on a regular grid
    (NOT the [5,2,3] cross primitive).
  * Low-res input = the high-res ground truth subsampled by every k-th voxel.
  * Normalization = per-flow-map: shift mean to 0, uniformly scale into [-1,1]^2 (paper 4.4).
  * Loss / evaluation are done in FLOW-MAP space (MSE / PSNR) vs cubic interpolation.
  * FTLE is computed from the upsampled flow map via grid finite differences (eq. 10),
    used for qualitative comparison only.
"""
import os
import logging
import numpy as np
import torch
from torch.utils.data import Dataset

from DeepUtils.utils.stable_hash import stable_hash
from FLowUtils.flowlineIntegral import batch_pathline_integration_2D_cuda
from FLowUtils.flowDatasetUtils.NetCDF_AmiraLoader import load_UnsteadyVectorFields_general
from FMT_Utils.FTLE_fitting_utils import generate_seedingGrid_2D


# ----------------------------------------------------------------------------- #
# Flow-map (end-position) generation
# ----------------------------------------------------------------------------- #
def generate_endpos_flowmap_SLICE(vectorfield, t0: float, dt: float, max_steps: int,
                                  sampling: float, boundary_offset: float = 0.05):
    """Trace single particles on a regular grid and return their END positions.

    Returns:
        grid: np.float32 [ny, nx, 2]  absolute (x,y) position after integrating for
              tau = dt*max_steps starting at t0.
        xs, ys: 1D grids of the seed coordinates (for finite-difference spacing).
    """
    starts_xy, xs, ys = generate_seedingGrid_2D(vectorfield, sampling, boundary_offset)
    ny, nx = len(ys), len(xs)
    N = starts_xy.shape[0]
    pts = np.concatenate(
        [starts_xy.astype(np.float64), np.full((N, 1), float(t0), dtype=np.float64)], axis=1)
    positions, valid = batch_pathline_integration_2D_cuda(
        pts, vectorfield, t_target=float(t0 + dt * max_steps), dt=float(dt),
        max_steps=int(max_steps), method="rk4")
    # end = last VALID step (particles that exited early are clamped to their last position)
    idx = np.clip(np.asarray(valid).astype(np.int64) - 1, 0, int(max_steps) - 1)
    end = positions[np.arange(N), idx, :2]
    grid = end.reshape(ny, nx, 2).astype(np.float32)
    return grid, np.asarray(xs, np.float32), np.asarray(ys, np.float32)


# ----------------------------------------------------------------------------- #
# Per-flow-map normalization (paper section 4.4)
# ----------------------------------------------------------------------------- #
def flowmap_unit_normalize(grid, mean=None, scale=None):
    """Shift mean to (0,0) and uniformly scale into [-1,1]^2.

    If mean/scale are given (e.g. derived from the low-res input at inference), they are
    reused so that low-res and high-res share the same transform. Returns (norm, mean, scale).
    """
    g = grid.reshape(-1, 2)
    if mean is None:
        mean = g.mean(axis=0)
    centered = grid - mean
    if scale is None:
        scale = float(np.abs(centered).max()) + 1e-12
    return (centered / scale).astype(np.float32), np.asarray(mean, np.float32), float(scale)


def flowmap_unit_denormalize(norm_grid, mean, scale):
    arr = norm_grid.detach().cpu().numpy() if isinstance(norm_grid, torch.Tensor) else np.asarray(norm_grid)
    return (arr * float(scale) + np.asarray(mean, np.float32)).astype(np.float32)


# ----------------------------------------------------------------------------- #
# FTLE from an end-position grid via finite differences (eq. 10, qualitative only)
# ----------------------------------------------------------------------------- #
def ftle_from_endpos_grid(grid, xs, ys, tau: float):
    """FTLE = 1/|tau| * 0.5 * ln(lambda_max(J^T J)), J = grad of the flow map on the grid."""
    grid = grid.detach().cpu().numpy() if isinstance(grid, torch.Tensor) else np.asarray(grid, np.float32)
    dx = float(xs[1] - xs[0]) if len(xs) > 1 else 1.0
    dy = float(ys[1] - ys[0]) if len(ys) > 1 else 1.0
    dphidx = np.gradient(grid, dx, axis=1)   # [ny,nx,2]
    dphidy = np.gradient(grid, dy, axis=0)   # [ny,nx,2]
    # J columns are dphi/dx and dphi/dy ; C = J^T J (2x2 symmetric per cell)
    a = dphidx[..., 0] ** 2 + dphidx[..., 1] ** 2          # C00
    b = dphidx[..., 0] * dphidy[..., 0] + dphidx[..., 1] * dphidy[..., 1]  # C01
    d = dphidy[..., 0] ** 2 + dphidy[..., 1] ** 2          # C11
    tr = a + d
    disc = np.sqrt(np.maximum((tr * 0.5) ** 2 - (a * d - b * b), 0.0))
    lam_max = tr * 0.5 + disc
    ftle = 0.5 * np.log(np.maximum(lam_max, 1e-12)) / max(abs(float(tau)), 1e-12)
    return ftle.astype(np.float32)


def _crop_to_multiple(grid, k):
    ny, nx = grid.shape[:2]
    return grid[: (ny // k) * k, : (nx // k) * k]


# ----------------------------------------------------------------------------- #
# Training dataset: low-res patch -> high-res patch (flow-map super-resolution)
# ----------------------------------------------------------------------------- #
class FlowMapSRTrainDataset(Dataset):
    """Patches of (low-res flow map -> high-res flow map), normalized per source flow map.

    low-res is the high-res ground truth subsampled every k-th voxel, so the low-res grid
    points coincide exactly with a subset of the high-res grid (the paper's setup).
    """
    def __init__(self, config, useCacheSystem: bool = True):
        names = list(getattr(config.dataset, 'input_names', getattr(config.dataset, 'names', [])))
        k = int(config.dataset.UPsampling)
        sampling = float(config.dataset.low_res_grid_sampling) * k   # trace at HIGH res
        dt = float(config.pcds.dt)
        max_steps = int(config.pcds.max_iterations)
        t_start = float(config.dataset.t_start)
        t_target = float(config.dataset.t_target)
        timeslices = int(getattr(config.dataset, 'trainTimesliceCount', 20))
        p_lo = int(getattr(config.dataset, 'patchSize', 16))          # LOW-res patch size
        stride_lo = int(getattr(config.dataset, 'patchStride', 8))
        bnd = float(getattr(config.dataset, 'boundaryOffset', 0.05))

        key = {"name": "flowmap_sr_train", "fields": names, "k": k, "sampling": sampling,
               "dt": dt, "max_steps": max_steps, "t0": t_start, "t1": t_target,
               "slices": timeslices, "p": p_lo, "stride": stride_lo, "bnd": bnd, "rep": "endpos_v1"}
        tag = stable_hash(key, prefix="FlowMapSRTrain_")
        cache_dir = os.path.join(config.cache_dir, "temp")
        cache_path = os.path.join(cache_dir, f"{tag}.npz")
        if useCacheSystem and os.path.exists(cache_path):
            try:
                data = np.load(cache_path)
                self.lo = torch.from_numpy(data["lo"]).float()
                self.hi = torch.from_numpy(data["hi"]).float()
                logging.info(f"[FlowMapSRTrain] loaded {self.lo.shape[0]} patches from cache {cache_path}")
                return
            except Exception as e:
                logging.info(f"[FlowMapSRTrain] cache load failed: {e}. Regenerating...")

        vfs = load_UnsteadyVectorFields_general(config.dataset.dat_dir, names)
        lo_patches, hi_patches = [], []
        for vi, vf in enumerate(vfs):
            if vf is None:
                continue
            t0 = t_start * (vf.tmax - vf.tmin) + vf.tmin
            t1 = t_target * (vf.tmax - vf.tmin) + vf.tmin
            safe = float(vf.tmax) - dt * max_steps
            t1 = min(t1, max(t0, safe))
            n_before = len(lo_patches)
            for tt in np.linspace(t0, t1, timeslices):
                hi, xs, ys = generate_endpos_flowmap_SLICE(vf, float(tt), dt, max_steps, sampling, bnd)
                hi = _crop_to_multiple(hi, k)
                lo = hi[::k, ::k]                                   # subsample -> low-res
                # per-flow-map normalization derived from the LOW-res input (inference-consistent)
                lo_n, mean, scale = flowmap_unit_normalize(lo)
                hi_n, _, _ = flowmap_unit_normalize(hi, mean=mean, scale=scale)
                ny_lo, nx_lo = lo_n.shape[:2]
                for i0 in _starts(ny_lo, p_lo, stride_lo):
                    for j0 in _starts(nx_lo, p_lo, stride_lo):
                        lp = lo_n[i0:i0 + p_lo, j0:j0 + p_lo]
                        hp = hi_n[i0 * k:(i0 + p_lo) * k, j0 * k:(j0 + p_lo) * k]
                        if lp.shape[:2] == (p_lo, p_lo) and hp.shape[:2] == (p_lo * k, p_lo * k):
                            lo_patches.append(lp); hi_patches.append(hp)
            logging.info(f"[FlowMapSRTrain] '{names[vi] if vi < len(names) else vi}': "
                         f"{len(lo_patches) - n_before} patches.")

        self.lo = torch.from_numpy(np.stack(lo_patches)).float()   # [N, p, p, 2]
        self.hi = torch.from_numpy(np.stack(hi_patches)).float()   # [N, p*k, p*k, 2]
        if useCacheSystem:
            try:
                os.makedirs(cache_dir, exist_ok=True)
                np.savez(cache_path, lo=self.lo.numpy(), hi=self.hi.numpy())
                logging.info(f"[FlowMapSRTrain] saved {self.lo.shape[0]} patches to cache {cache_path}")
            except Exception as e:
                logging.error(f"[FlowMapSRTrain] cache save failed: {e}")

    def __len__(self):
        return self.lo.shape[0]

    def __getitem__(self, idx):
        # to channel-first images: [2, H, W]
        return self.lo[idx].permute(2, 0, 1), self.hi[idx].permute(2, 0, 1)


def _starts(length, k, stride):
    if length <= k:
        return [0]
    s = max(1, int(stride))
    out = list(range(0, length - k + 1, s))
    if out[-1] != length - k:
        out.append(length - k)
    return out


def build_FlowMapSR_test_dataset(config):
    """Whole-grid test flow maps (no patching). Returns list of dicts per slice:
       {hi, lo, xs_hi, ys_hi, tau, name}."""
    names_cfg = config.test.vectorfield
    names = names_cfg if isinstance(names_cfg, (list, tuple)) else [names_cfg]
    k = int(config.dataset.UPsampling)
    sampling = float(config.dataset.low_res_grid_sampling) * k
    dt = float(config.pcds.dt)
    max_steps = int(config.pcds.max_iterations)
    t_start = float(config.dataset.t_start)
    t_target = float(config.dataset.t_target)
    timeslices = int(getattr(config.test, 'timesliceCount', 5))
    bnd = float(getattr(config.dataset, 'boundaryOffset', 0.05))
    tau = dt * max_steps

    vfs = load_UnsteadyVectorFields_general(config.dataset.dat_dir, list(names))
    out = []
    for vi, vf in enumerate(vfs):
        if vf is None:
            continue
        t0 = t_start * (vf.tmax - vf.tmin) + vf.tmin
        t1 = t_target * (vf.tmax - vf.tmin) + vf.tmin
        safe = float(vf.tmax) - tau
        t1 = min(t1, max(t0, safe))
        for tt in np.linspace(t0, t1, timeslices):
            hi, xs, ys = generate_endpos_flowmap_SLICE(vf, float(tt), dt, max_steps, sampling, bnd)
            hi = _crop_to_multiple(hi, k)
            lo = hi[::k, ::k]
            out.append({"hi": hi, "lo": lo, "xs": xs[: hi.shape[1]], "ys": ys[: hi.shape[0]],
                        "tau": float(tau), "name": str(names[vi] if vi < len(names) else vi),
                        "domMin": vf.domainMinBoundary, "domMax": vf.domainMaxBoundary})
    logging.info(f"[FlowMapSR test] built {len(out)} whole-grid test flow maps.")
    return out
