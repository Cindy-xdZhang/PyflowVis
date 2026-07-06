"""Fit Vatistas vortex-profile parameters to real 2D flow patches, record their
statistical distribution, and re-sample it to synthesize new steady fields.
This reproduces the "Parameter Space Fitting" step of the VortexTransformer paper. See docs/vatistas_profile.md for the math.
Pipeline
--------
1. Load real unsteady flows via the same path used by FTLE_experiment.py
   (relocate_flow2d_dataset_folder + load_UnsteadyVectorFields_general) and slice
   them into spatial 32x32 velocity patches (sliding like a convolution kernel).
2. Fit each patch with a mixture of two steady Vatistas profiles (Eq. 10). Every
   profile has 7 continuous parameters (theta, sx, sy, rc, n, tx, ty) plus a
   discrete base shape S_i, so a patch has 14 continuous parameters. We minimize
   velocity MSE with 200 iterations of simulated annealing followed by 200
   iterations of gradient descent (following [KG19]).
3. Summarize the fitted parameters as per-parameter Gaussian distributions
   (mean/std) 

Usage
-----
    python FittingVatistasParam.py --config config/FittingVatistas.yaml
"""

from __future__ import annotations

import os
import json
import argparse
import logging

import numpy as np
import torch

from DeepUtils.utils import EasyConfig
from FLowUtils.flowDatasetUtils.NetCDF_AmiraLoader import (
    relocate_flow2d_dataset_folder,
    load_UnsteadyVectorFields_general,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# Vatistas model constants
# ---------------------------------------------------------------------------
# 7 continuous parameters per profile, in this fixed order.
PARAM_NAMES = ["theta", "sx", "sy", "rc", "n", "tx", "ty"]
TH, SX, SY, RC, N, TX, TY = range(7)

# Eq. 4: the three constant base-shape matrices.
#   0 -> S1 saddle,  1 -> S2 center clockwise,  2 -> S3 center counter-clockwise
S_MATRICES = torch.tensor(
    [
        [[1.0, 0.0], [0.0, -1.0]],   # S1 saddle
        [[0.0, 1.0], [-1.0, 0.0]],   # S2 center (cw)
        [[0.0, -1.0], [1.0, 0.0]],   # S3 center (ccw)
    ],
    dtype=torch.float32,
)
SHAPE_LABELS = {0: "S1_saddle", 1: "S2_center_cw", 2: "S3_center_ccw"}

TWO_PI = 2.0 * np.pi


# ---------------------------------------------------------------------------
# Forward model  (batched + differentiable)
# ---------------------------------------------------------------------------
def vatistas_mixture_velocity(coords, params, shape_idx, bounds, profile_mask=None):
    """Evaluate the mixture-of-profiles velocity field (Eq. 9 & 10), batched.

    Args:
        coords:      [H, W, 2] query grid (patch-local frame, normalized to [-1,1]^2).
        params:      [P, M, 7] continuous parameters (theta, sx, sy, rc, n, tx, ty).
        shape_idx:   [P, M] long, base-shape index in {0,1,2}.
        bounds:      dict with 'rc','n','s' = (min,max) used to keep the math valid.
        profile_mask:[P, M] optional {0,1} multiplier to switch profiles off.

    Returns:
        velocity: [P, H, W, 2] summed over the M profiles.
    """
    P, M, _ = params.shape
    H, W = coords.shape[0], coords.shape[1]
    eps = 1e-9

    theta = params[..., TH]                                   # [P, M]
    # Clamp the parameters that would otherwise break the math (positive rc, n, s).
    rc = params[..., RC].clamp(bounds["rc"][0], bounds["rc"][1])
    n = params[..., N].clamp(bounds["n"][0], bounds["n"][1])
    sx = params[..., SX].clamp(bounds["s"][0], bounds["s"][1])
    sy = params[..., SY].clamp(bounds["s"][0], bounds["s"][1])
    tx = params[..., TX]
    ty = params[..., TY]

    cos, sin = torch.cos(theta), torch.sin(theta)             # [P, M]

    # Inverse deformation D^{-1} = 1/(sx*sy) * [[sy c, sy s], [-sx s, sx c]]  (det D = sx*sy)
    inv_det = 1.0 / (sx * sy)
    Di00 = inv_det * sy * cos
    Di01 = inv_det * sy * sin
    Di10 = inv_det * (-sx * sin)
    Di11 = inv_det * sx * cos

    Xp = coords[..., 0].view(1, 1, H, W)                      # [1,1,H,W]
    Yp = coords[..., 1].view(1, 1, H, W)

    def e(a):  # [P,M] -> [P,M,1,1] for broadcasting against the grid
        return a.view(P, M, 1, 1)

    # Pull the query point back into the base (undeformed) frame: x = D^{-1}(x'-T).
    dx = Xp - e(tx)
    dy = Yp - e(ty)
    bx = e(Di00) * dx + e(Di01) * dy                          # [P,M,H,W]
    by = e(Di10) * dx + e(Di11) * dy

    r = torch.sqrt(bx * bx + by * by + eps)                   # [P,M,H,W]

    # Eq. 3 factor g(r) = v0(r)/r = 1/(2*pi*rc^2) * (1 + (r/rc)^{2n})^{-1/n},
    # evaluated in log-space for numerical stability (no divide-by-r, no overflow).
    log_ratio = torch.log(r) - torch.log(e(rc))
    p2n = (2.0 * e(n)) * log_ratio
    p2n = p2n.clamp(max=30.0)
    log_bracket = torch.log1p(torch.exp(p2n))                 # log(1 + (r/rc)^{2n})
    g = torch.exp(-(1.0 / e(n)) * log_bracket) / (TWO_PI * e(rc) * e(rc))

    # Base velocity vb = g * (S . x)   (Eq. 2)
    S = S_MATRICES.to(params.device)[shape_idx]               # [P,M,2,2]
    Sx = S[..., 0, 0].unsqueeze(-1).unsqueeze(-1) * bx + S[..., 0, 1].unsqueeze(-1).unsqueeze(-1) * by
    Sy = S[..., 1, 0].unsqueeze(-1).unsqueeze(-1) * bx + S[..., 1, 1].unsqueeze(-1).unsqueeze(-1) * by
    vbx = g * Sx
    vby = g * Sy

    # Push the velocity forward: v' = D . vb   (Eq. 9), D = [[sx c, -sy s],[sx s, sy c]]
    vpx = e(sx * cos) * vbx + e(-sy * sin) * vby
    vpy = e(sx * sin) * vbx + e(sy * cos) * vby

    if profile_mask is not None:
        vpx = vpx * e(profile_mask)
        vpy = vpy * e(profile_mask)

    vx = vpx.sum(dim=1)                                        # [P,H,W]
    vy = vpy.sum(dim=1)
    return torch.stack([vx, vy], dim=-1)                      # [P,H,W,2]


def per_patch_mse(pred, target):
    """MSE per patch: [P,H,W,2] vs [P,H,W,2] -> [P]."""
    return ((pred - target) ** 2).reshape(pred.shape[0], -1).mean(dim=1)


# ---------------------------------------------------------------------------
# Data: load real flows and slice into velocity patches
# ---------------------------------------------------------------------------
def _clamp_time_range(t_start, t_end, n_steps):
    a = 0 if t_start is None else int(t_start)
    b = n_steps if t_end is None else int(t_end)
    a = max(0, min(a, n_steps))
    b = max(0, min(b, n_steps))
    if b <= a:
        a, b = 0, n_steps
    return a, b


def extract_patches(cfg, device):
    """Load the configured real flows and tile every requested time slice into
    (patch_size x patch_size) velocity patches slid with `patch_stride`.

    Returns:
        targets: [P, H, W, 2] float32 (RMS-normalized if cfg.dataset.normalize_patch)
        scales:  [P] the per-patch RMS speed removed by normalization
        meta:    list of dicts describing each patch's origin
    """
    dcfg = cfg.dataset
    ps = int(dcfg.patch_size)
    stride = int(dcfg.patch_stride)
    min_rms = float(dcfg.min_patch_rms)
    normalize = bool(dcfg.normalize_patch)

    field_specs = list(dcfg.fields)
    names = [str(f["name"]) for f in field_specs]
    logging.info(f"[patches] loading real flows: {names}")
    fields = load_UnsteadyVectorFields_general(dcfg.dat_dir, names)

    patches, scales, meta = [], [], []
    for spec, vf in zip(field_specs, fields):
        if vf is None:
            logging.warning(f"[patches] field '{spec['name']}' failed to load; skip.")
            continue
        field = vf.field
        field = field.cpu().numpy() if isinstance(field, torch.Tensor) else np.asarray(field)
        field = field.astype(np.float32)                      # [T, Y, X, 2]
        T, Y, X = field.shape[0], field.shape[1], field.shape[2]
        a, b = _clamp_time_range(spec.get("t_start"), spec.get("t_end"), T)
        t_stride = max(1, int(spec.get("t_stride", 1)))
        t_indices = list(range(a, b, t_stride))
        if Y < ps or X < ps:
            logging.warning(f"[patches] field '{spec['name']}' ({Y}x{X}) smaller than patch {ps}; skip.")
            continue

        row_starts = list(range(0, Y - ps + 1, stride))
        col_starts = list(range(0, X - ps + 1, stride))
        n_before = len(patches)
        for t in t_indices:
            slice_t = field[t]                                # [Y, X, 2]
            for y0 in row_starts:
                for x0 in col_starts:
                    p = slice_t[y0:y0 + ps, x0:x0 + ps, :]    # [ps, ps, 2]
                    rms = float(np.sqrt(np.mean(p[..., 0] ** 2 + p[..., 1] ** 2)))
                    if rms < min_rms:
                        continue
                    scale = rms if normalize else 1.0
                    patches.append(p / (scale + 1e-12) if normalize else p.copy())
                    scales.append(scale)
                    meta.append({"field": str(spec["name"]), "t": int(t), "y0": int(y0), "x0": int(x0)})
        logging.info(f"[patches] '{spec['name']}': {len(t_indices)} slices -> {len(patches) - n_before} patches "
                     f"(time steps {a}:{b}:{t_stride}, grid {Y}x{X}).")

    if not patches:
        raise RuntimeError("No patches extracted. Check dataset paths / field names / min_patch_rms.")

    targets = np.stack(patches, axis=0).astype(np.float32)    # [P, ps, ps, 2]
    scales = np.asarray(scales, dtype=np.float32)

    # Cap the patch count (paper fits ~1500) with a reproducible random subsample.
    max_patches = int(dcfg.max_patches)
    if targets.shape[0] > max_patches:
        idx = np.random.default_rng(int(cfg.get("seed", 0))).permutation(targets.shape[0])[:max_patches]
        idx.sort()
        targets, scales, meta = targets[idx], scales[idx], [meta[i] for i in idx]
        logging.info(f"[patches] subsampled to max_patches={max_patches}.")

    logging.info(f"[patches] total {targets.shape[0]} patches of size {ps}x{ps}.")
    return torch.from_numpy(targets).to(device), torch.from_numpy(scales).to(device), meta


def make_local_grid(ps, device):
    """Patch-local query grid on [-1,1]^2: coords[i,j] = (x_j, y_i), shape [ps,ps,2]."""
    lin = torch.linspace(-1.0, 1.0, ps, device=device)
    yy, xx = torch.meshgrid(lin, lin, indexing="ij")
    return torch.stack([xx, yy], dim=-1)                      # [ps, ps, 2]


# ---------------------------------------------------------------------------
# Optimization: simulated annealing + gradient descent
# ---------------------------------------------------------------------------
def _project(params, shape_idx, bounds, rng=None):
    """Clamp continuous params into valid ranges (theta is left free; wrapped later)."""
    with torch.no_grad():
        params[..., RC].clamp_(bounds["rc"][0], bounds["rc"][1])
        params[..., N].clamp_(bounds["n"][0], bounds["n"][1])
        params[..., SX].clamp_(bounds["s"][0], bounds["s"][1])
        params[..., SY].clamp_(bounds["s"][0], bounds["s"][1])
        params[..., TX].clamp_(-bounds["t"], bounds["t"])
        params[..., TY].clamp_(-bounds["t"], bounds["t"])
    return params


def datadriven_seeds(target, coords, cfg, M):
    """Detect up to M vortex cores per patch from the vorticity field and return
    per-profile seed (tx, ty, base-shape). Peaks of |omega| localize the cores; the
    sign of omega picks ccw (S3) vs cw (S2). Successive peaks are suppressed in a disk
    so the two profiles seed different vortices. This gives SA/GD a physically-informed
    starting basin instead of a random one."""
    P, H, W, _ = target.shape
    u, v = target[..., 0], target[..., 1]                        # [P,H,W]
    dvdx = torch.gradient(v, spacing=2.0 / (W - 1), dim=2)[0]
    dudy = torch.gradient(u, spacing=2.0 / (H - 1), dim=1)[0]
    omega = dvdx - dudy                                          # vorticity [P,H,W]
    oabs = omega.abs().clone()

    cx, cy = coords[..., 0], coords[..., 1]                      # [H,W]
    cand = [int(c) for c in cfg.model.shape_candidates]
    has_cw, has_ccw = (1 in cand), (2 in cand)
    dev = target.device
    ii, jj = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
    rad = max(2, int(0.25 * min(H, W)))                          # suppression radius (grid cells)

    tx0 = torch.zeros(P, M, device=dev)
    ty0 = torch.zeros(P, M, device=dev)
    shape0 = torch.zeros(P, M, dtype=torch.long, device=dev)
    ar = torch.arange(P, device=dev)
    for m in range(M):
        idx = oabs.reshape(P, -1).argmax(dim=1)                  # [P] flat peak
        pi, pj = idx // W, idx % W
        tx0[:, m], ty0[:, m] = cx[pi, pj], cy[pi, pj]
        w_at = omega[ar, pi, pj]
        s = torch.where(w_at >= 0, torch.full_like(idx, 2), torch.full_like(idx, 1))  # ccw/cw
        if not has_ccw:
            s = torch.where(s == 2, torch.full_like(s, 1), s)
        if not has_cw:
            s = torch.where(s == 1, torch.full_like(s, 2), s)
        shape0[:, m] = s
        d2 = (ii.view(1, H, W) - pi.view(P, 1, 1)) ** 2 + (jj.view(1, H, W) - pj.view(P, 1, 1)) ** 2
        oabs = torch.where(d2 <= rad * rad, torch.zeros_like(oabs), oabs)
    return tx0, ty0, shape0


def init_params(P, M, cfg, bounds, device, restart=0, seeds=None):
    """Random-but-plausible initial parameters and base shapes for every profile.
    `restart` varies the RNG so multi-restart fitting explores different basins.
    If `seeds`=(tx0,ty0,shape0) is given (data-driven mode), core positions/shapes are
    initialized from vortex detection; restart 0 uses them exactly, later restarts add
    growing jitter + random shape flips to retain exploration."""
    g = torch.Generator(device="cpu").manual_seed(int(cfg.get("seed", 0)) + 9973 * int(restart))

    def u(lo, hi, shape):
        return (lo + (hi - lo) * torch.rand(shape, generator=g)).to(device)

    params = torch.empty(P, M, 7, device=device)
    params[..., TH] = u(-np.pi, np.pi, (P, M))
    params[..., SX] = u(0.5, 1.5, (P, M))
    params[..., SY] = u(0.5, 1.5, (P, M))
    params[..., RC] = u(0.2, 0.6, (P, M))
    params[..., N] = u(1.0, 3.0, (P, M))
    candidates = torch.tensor([int(c) for c in cfg.model.shape_candidates], dtype=torch.long)

    if seeds is not None:
        tx0, ty0, shape0 = seeds
        jit = 0.0 if restart == 0 else min(0.5, 0.12 * restart)
        params[..., TX] = tx0 + jit * torch.randn((P, M), generator=g).to(device)
        params[..., TY] = ty0 + jit * torch.randn((P, M), generator=g).to(device)
        if restart == 0:
            shape_idx = shape0.clone()
        else:
            flip = (torch.rand((P, M), generator=g).to(device) < 0.3)
            rnd = candidates[torch.randint(0, len(candidates), (P, M), generator=g)].to(device)
            shape_idx = torch.where(flip, rnd, shape0)
    else:
        params[..., TX] = u(-0.5, 0.5, (P, M))
        params[..., TY] = u(-0.5, 0.5, (P, M))
        pick = torch.randint(0, len(candidates), (P, M), generator=g)
        shape_idx = candidates[pick].to(device)
    return _project(params, shape_idx, bounds), shape_idx


def simulated_annealing(coords, target, params, shape_idx, cfg, bounds):
    """Batched Metropolis simulated annealing over all patches simultaneously."""
    ocfg = cfg.optim
    iters = int(ocfg.sa_iters)
    T0, Tend = float(ocfg.sa_T0), float(ocfg.sa_Tend)
    flip_p = float(ocfg.sa_shape_flip_p)
    cooling = (Tend / T0) ** (1.0 / max(1, iters))
    P, M, _ = params.shape
    device = params.device
    candidates = torch.tensor([int(c) for c in cfg.model.shape_candidates], dtype=torch.long, device=device)

    # per-parameter proposal scale (physical units), annealed with temperature
    sigma = torch.tensor([0.5, 0.3, 0.3, 0.15, 0.5, 0.2, 0.2], device=device)

    with torch.no_grad():
        cur = params.clone()
        cur_shape = shape_idx.clone()
        cur_loss = per_patch_mse(vatistas_mixture_velocity(coords, cur, cur_shape, bounds), target)
        best, best_shape, best_loss = cur.clone(), cur_shape.clone(), cur_loss.clone()

        T = T0
        for it in range(iters):
            frac = T / T0
            prop = cur + torch.randn_like(cur) * (sigma.view(1, 1, 7) * frac)
            prop = _project(prop, cur_shape, bounds)

            prop_shape = cur_shape
            if flip_p > 0.0:
                flip = torch.rand(P, M, device=device) < flip_p
                rnd_shape = candidates[torch.randint(0, len(candidates), (P, M), device=device)]
                prop_shape = torch.where(flip, rnd_shape, cur_shape)

            prop_loss = per_patch_mse(vatistas_mixture_velocity(coords, prop, prop_shape, bounds), target)

            delta = prop_loss - cur_loss
            accept = (delta < 0) | (torch.rand(P, device=device) < torch.exp(-delta / max(T, 1e-8)))
            a = accept.view(P, 1, 1)
            cur = torch.where(a, prop, cur)
            cur_shape = torch.where(accept.view(P, 1), prop_shape, cur_shape)
            cur_loss = torch.where(accept, prop_loss, cur_loss)

            improved = cur_loss < best_loss
            b = improved.view(P, 1, 1)
            best = torch.where(b, cur, best)
            best_shape = torch.where(improved.view(P, 1), cur_shape, best_shape)
            best_loss = torch.where(improved, cur_loss, best_loss)

            T *= cooling
            if (it + 1) % 50 == 0 or it == iters - 1:
                logging.info(f"[SA] iter {it+1}/{iters}  T={T:.4g}  mean best mse={best_loss.mean().item():.6f}")

    return best, best_shape, best_loss


def gradient_descent(coords, target, params, shape_idx, cfg, bounds):
    """Adam refinement of the continuous parameters (base shapes held fixed).

    The learning rate follows a cosine decay from gd_lr to gd_lr_final; with more GD
    iterations the decay lets the fit settle into a lower-MSE minimum than a fixed lr,
    which plateaus. Set gd_lr_final == gd_lr for a constant rate."""
    iters = int(cfg.optim.gd_iters)
    lr0 = float(cfg.optim.gd_lr)
    lr1 = float(cfg.optim.get("gd_lr_final", lr0))
    p = params.clone().requires_grad_(True)
    opt = torch.optim.Adam([p], lr=lr0)

    for it in range(iters):
        cur_lr = lr1 + 0.5 * (lr0 - lr1) * (1.0 + np.cos(np.pi * it / max(1, iters - 1)))
        for grp in opt.param_groups:
            grp["lr"] = cur_lr
        opt.zero_grad()
        pred = vatistas_mixture_velocity(coords, p, shape_idx, bounds)
        loss = per_patch_mse(pred, target).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p], max_norm=10.0)
        opt.step()
        _project(p, shape_idx, bounds)
        if (it + 1) % 50 == 0 or it == iters - 1:
            logging.info(f"[GD] iter {it+1}/{iters}  lr={cur_lr:.4g}  mean mse={loss.item():.6f}")

    with torch.no_grad():
        final = p.detach()
        final_loss = per_patch_mse(vatistas_mixture_velocity(coords, final, shape_idx, bounds), target)
    return final, final_loss


def lbfgs_polish(coords, target, params, shape_idx, cfg, bounds):
    """Second-order (L-BFGS + strong-Wolfe line search) polish after Adam. L-BFGS
    converges to a tighter minimum of this smooth MSE surface than first-order Adam,
    squeezing extra PSNR. rc/n/sx/sy stay valid via the forward's internal clamps;
    tx/ty are projected once at the end. No-op when optim.lbfgs_iters <= 0."""
    iters = int(cfg.optim.get("lbfgs_iters", 0))
    if iters <= 0:
        with torch.no_grad():
            return params, per_patch_mse(vatistas_mixture_velocity(coords, params, shape_idx, bounds), target)
    p = params.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([p], max_iter=iters, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-9, tolerance_change=1e-12)

    def closure():
        opt.zero_grad()
        loss = per_patch_mse(vatistas_mixture_velocity(coords, p, shape_idx, bounds), target).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        _project(p, shape_idx, bounds)
        final = p.detach()
        final_loss = per_patch_mse(vatistas_mixture_velocity(coords, final, shape_idx, bounds), target)
    # guard: never let the polish worsen a patch (line-search overshoot) or emit NaN
    orig_loss = per_patch_mse(vatistas_mixture_velocity(coords, params, shape_idx, bounds), target)
    bad = (final_loss > orig_loss) | ~torch.isfinite(final_loss)
    if bad.any():
        b = bad.view(-1, 1, 1)
        final = torch.where(b, params, final)
        final_loss = per_patch_mse(vatistas_mixture_velocity(coords, final, shape_idx, bounds), target)
    return final, final_loss


def fit_patches(coords, target, cfg, bounds, M, device):
    """Full fit for every patch: simulated annealing -> gradient descent, repeated
    from `cfg.optim.n_restarts` random inits with the best-per-patch result kept
    (multi-restart is the main lever for escaping SA/GD local minima)."""
    P = target.shape[0]
    n_restarts = int(cfg.optim.get("n_restarts", 1))
    seeds = None
    if str(cfg.optim.get("init_mode", "random")) == "datadriven":
        seeds = datadriven_seeds(target, coords, cfg, M)
    best_params = best_shape = best_loss = None
    for r in range(n_restarts):
        params, shape_idx = init_params(P, M, cfg, bounds, device, restart=r, seeds=seeds)
        params, shape_idx, _ = simulated_annealing(coords, target, params, shape_idx, cfg, bounds)
        params, loss = gradient_descent(coords, target, params, shape_idx, cfg, bounds)
        if int(cfg.optim.get("lbfgs_iters", 0)) > 0:
            params, loss = lbfgs_polish(coords, target, params, shape_idx, cfg, bounds)
        if best_loss is None:
            best_params, best_shape, best_loss = params, shape_idx, loss
        else:
            take = loss < best_loss                       # [P]
            b = take.view(P, 1, 1)
            best_params = torch.where(b, params, best_params)
            best_shape = torch.where(take.view(P, 1), shape_idx, best_shape)
            best_loss = torch.where(take, loss, best_loss)
        if n_restarts > 1:
            logging.info(f"[fit] restart {r+1}/{n_restarts}  this-run mean mse={loss.mean().item():.6f}  "
                         f"best-so-far mean mse={best_loss.mean().item():.6f}")
    return best_params, best_shape, best_loss


def fit_quality_report(coords, target, params, shape_idx, bounds, tag="fit"):
    """Compute and log MSE + PSNR of the fitted mixture vs the (normalized) patches.

    PSNR uses each patch's own peak-to-peak range as the reference (the convention in
    FTLE_experiment._sr_metrics), then averages over patches. Returns a metrics dict.
    """
    with torch.no_grad():
        pred = vatistas_mixture_velocity(coords, params, shape_idx, bounds)
        P = target.shape[0]
        se = (pred - target) ** 2
        mse_pp = se.reshape(P, -1).mean(dim=1)                          # [P]
        tgt = target.reshape(P, -1)
        rng = (tgt.max(dim=1).values - tgt.min(dim=1).values).clamp_min(1e-8)
        psnr_pp = 20.0 * torch.log10(rng) - 10.0 * torch.log10(mse_pp.clamp_min(1e-12))
        global_mse = se.mean()
        # global PSNR against the overall data range (single number for the whole set)
        grng = (target.max() - target.min()).clamp_min(1e-8)
        global_psnr = 20.0 * torch.log10(grng) - 10.0 * torch.log10(global_mse.clamp_min(1e-12))

    m = {
        "mse_mean": float(mse_pp.mean()), "mse_median": float(mse_pp.median()),
        "psnr_mean": float(psnr_pp.mean()), "psnr_median": float(psnr_pp.median()),
        "global_mse": float(global_mse), "global_psnr": float(global_psnr),
    }
    logging.info(f"[{tag}] velocity MSE  mean={m['mse_mean']:.6f}  median={m['mse_median']:.6f}  "
                 f"global={m['global_mse']:.6f}")
    logging.info(f"[{tag}] PSNR (dB)     mean={m['psnr_mean']:.2f}  median={m['psnr_median']:.2f}  "
                 f"global={m['global_psnr']:.2f}")
    return m


# ---------------------------------------------------------------------------
# Distribution fitting + generation
# ---------------------------------------------------------------------------
def profile_contributions(coords, params, shape_idx, bounds):
    """Per-profile RMS energy fraction within a patch -> [P, M] in [0,1].

    Used to drop degenerate (near-zero) profiles before building the distribution.
    """
    P, M, _ = params.shape
    with torch.no_grad():
        rms = torch.zeros(P, M, device=params.device)
        for m in range(M):
            mask = torch.zeros(P, M, device=params.device)
            mask[:, m] = 1.0
            v = vatistas_mixture_velocity(coords, params, shape_idx, bounds, profile_mask=mask)
            rms[:, m] = torch.sqrt((v ** 2).reshape(P, -1).mean(dim=1) + 1e-12)
        total = rms.sum(dim=1, keepdim=True) + 1e-12
        return rms / total


def fit_distribution(params, shape_idx, active_mask):
    """Pool active profiles and fit an independent Gaussian (mean/std) per parameter.

    Returns a JSON-serializable dict (the paper's Table 6 content).
    """
    prm = params.reshape(-1, 7).cpu().numpy()
    shp = shape_idx.reshape(-1).cpu().numpy()
    keep = active_mask.reshape(-1).cpu().numpy().astype(bool)
    prm, shp = prm[keep], shp[keep]

    # theta is an angle: wrap to (-pi, pi] and report circular mean.
    theta = np.arctan2(np.sin(prm[:, TH]), np.cos(prm[:, TH]))
    prm = prm.copy()
    prm[:, TH] = theta

    dist = {"num_profiles": int(prm.shape[0]), "params": {}}
    for i, name in enumerate(PARAM_NAMES):
        col = prm[:, i]
        if name == "theta":
            c_mean = float(np.arctan2(np.sin(col).mean(), np.cos(col).mean()))
            dist["params"][name] = {"mean": c_mean, "std": float(col.std()),
                                    "min": float(col.min()), "max": float(col.max())}
        else:
            dist["params"][name] = {"mean": float(col.mean()), "std": float(col.std()),
                                    "min": float(col.min()), "max": float(col.max())}

    shapes, counts = np.unique(shp, return_counts=True)
    probs = counts / counts.sum()
    dist["shape_distribution"] = {SHAPE_LABELS[int(s)]: float(pr) for s, pr in zip(shapes, probs)}
    dist["shape_index_probs"] = {int(s): float(pr) for s, pr in zip(shapes, probs)}
    return dist


def _sample_gaussian(spec, rng):
    """Draw one scalar from a fitted per-parameter Gaussian N(mean, std)."""
    return rng.normal(spec["mean"], max(spec["std"], 1e-6))


def generate_steady_fields(dist, cfg, bounds, device):
    """Sample `num_fields` parameter sets from the fitted distribution and render
    each as a steady velocity field on [-1,1]^2 (Eq. 10)."""
    gcfg = cfg.generate
    num_fields = int(gcfg.num_fields)   # NB: don't shadow the module-level N (param index)
    G = int(gcfg.grid)
    m_min, m_max = int(gcfg.min_vortices), int(gcfg.max_vortices)
    enforce_spacing = bool(gcfg.enforce_spacing)
    rng = np.random.default_rng(int(cfg.get("seed", 0)) + 1)

    shape_idx_list = sorted(dist["shape_index_probs"].keys())
    shape_probs = np.array([dist["shape_index_probs"][s] for s in shape_idx_list])
    pspec = dist["params"]

    def sample_profile():
        p = np.empty(7, np.float32)
        for i, name in enumerate(PARAM_NAMES):
            p[i] = _sample_gaussian(pspec[name], rng)
        p[RC] = np.clip(p[RC], bounds["rc"][0], bounds["rc"][1])
        p[N] = np.clip(p[N], bounds["n"][0], bounds["n"][1])
        p[SX] = np.clip(p[SX], bounds["s"][0], bounds["s"][1])
        p[SY] = np.clip(p[SY], bounds["s"][0], bounds["s"][1])
        p[TX] = np.clip(p[TX], -bounds["t"], bounds["t"])
        p[TY] = np.clip(p[TY], -bounds["t"], bounds["t"])
        p[TH] = np.arctan2(np.sin(p[TH]), np.cos(p[TH]))
        s = int(rng.choice(shape_idx_list, p=shape_probs))
        return p, s

    gen_params = np.zeros((num_fields, m_max, 7), np.float32)
    gen_shapes = np.zeros((num_fields, m_max), np.int64)
    gen_mask = np.zeros((num_fields, m_max), np.float32)
    gen_m = np.zeros(num_fields, np.int64)

    for k in range(num_fields):
        m = int(rng.integers(m_min, m_max + 1))
        profiles = []
        for _ in range(m):
            # For 2 vortices, keep cores spaced apart (dist > sum of radii), as in the paper.
            for _try in range(50):
                p, s = sample_profile()
                if not enforce_spacing or not profiles:
                    break
                ok = all(np.hypot(p[TX] - q[TX], p[TY] - q[TY]) > (p[RC] + q[RC]) for q, _ in profiles)
                if ok:
                    break
            profiles.append((p, s))
        for j, (p, s) in enumerate(profiles):
            gen_params[k, j] = p
            gen_shapes[k, j] = s
            gen_mask[k, j] = 1.0
        gen_m[k] = m

    # Render all fields with one batched forward pass (inactive profiles masked off).
    coords = make_local_grid(G, device)
    params_t = torch.from_numpy(gen_params).to(device)
    shapes_t = torch.from_numpy(gen_shapes).to(device)
    mask_t = torch.from_numpy(gen_mask).to(device)
    with torch.no_grad():
        fields = vatistas_mixture_velocity(coords, params_t, shapes_t, bounds, profile_mask=mask_t)
    logging.info(f"[generate] synthesized {num_fields} steady fields on [-1,1]^2 at {G}x{G} "
                 f"(vortices per field in [{m_min},{m_max}]).")
    return {
        "params": gen_params, "shapes": gen_shapes, "mask": gen_mask,
        "num_vortices": gen_m, "fields": fields.cpu().numpy().astype(np.float32),
        "coords": coords.cpu().numpy().astype(np.float32),
    }




def save_outputs(out_dir, params, shape_idx, fit_loss, contrib, active_mask, meta, dist, gen):
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(out_dir, "fitted_patch_params.npz"),
        params=params.cpu().numpy(), shapes=shape_idx.cpu().numpy(),
        fit_mse=fit_loss.cpu().numpy(), contribution=contrib.cpu().numpy(),
        active=active_mask.cpu().numpy(),
        meta_field=np.array([m["field"] for m in meta]),
        meta_t=np.array([m["t"] for m in meta]),
        meta_y0=np.array([m["y0"] for m in meta]),
        meta_x0=np.array([m["x0"] for m in meta]),
    )
    with open(os.path.join(out_dir, "vatistas_param_distribution.json"), "w") as f:
        json.dump(dist, f, indent=2)
    np.savez_compressed(
        os.path.join(out_dir, "generated_steady_flows.npz"),
        params=gen["params"], shapes=gen["shapes"], mask=gen["mask"],
        num_vortices=gen["num_vortices"], fields=gen["fields"], coords=gen["coords"],
    )
    logging.info(f"[save] wrote fitted params, distribution JSON and generated fields to {out_dir}")


def save_plots(out_dir, coords, target, params, shape_idx, bounds, dist, gen):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as ex:
        logging.warning(f"[plots] matplotlib unavailable ({ex}); skipping figures.")
        return

    # (1) parameter histograms
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    prm = params.reshape(-1, 7).cpu().numpy()
    for i, name in enumerate(PARAM_NAMES):
        ax = axes.flat[i]
        ax.hist(prm[:, i], bins=40, color="steelblue", alpha=0.85)
        s = dist["params"][name]
        ax.set_title(f"{name}: mu={s['mean']:.3f}, sig={s['std']:.3f}")
    axes.flat[7].axis("off")
    axes.flat[7].text(0.05, 0.5, "shapes:\n" + "\n".join(f"{k}: {v:.2f}" for k, v in dist["shape_distribution"].items()),
                      fontsize=11, va="center")
    fig.suptitle("Fitted Vatistas parameter distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "param_histograms.png"), dpi=120)
    plt.close(fig)

    def quiver(ax, cd, vel, title):
        step = max(1, cd.shape[0] // 24)
        X, Y = cd[..., 0], cd[..., 1]
        speed = np.sqrt(vel[..., 0] ** 2 + vel[..., 1] ** 2)
        ax.imshow(speed, extent=[-1, 1, -1, 1], origin="lower", cmap="viridis", alpha=0.7)
        ax.quiver(X[::step, ::step], Y[::step, ::step],
                  vel[::step, ::step, 0], vel[::step, ::step, 1], color="white", scale=None)
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])

    # (2) fit previews: target vs fitted for the first few patches
    with torch.no_grad():
        pred = vatistas_mixture_velocity(coords, params, shape_idx, bounds).cpu().numpy()
    cd = coords.cpu().numpy()
    tgt = target.cpu().numpy()
    npv = min(4, tgt.shape[0])
    fig, axes = plt.subplots(2, npv, figsize=(4 * npv, 8))
    axes = np.atleast_2d(axes)
    for j in range(npv):
        quiver(axes[0, j], cd, tgt[j], f"patch {j}: real")
        quiver(axes[1, j], cd, pred[j], f"patch {j}: fitted")
    fig.suptitle("Real patch (top) vs fitted Vatistas mixture (bottom)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fit_previews.png"), dpi=120)
    plt.close(fig)

    # (3) generated field previews
    gc, gf = gen["coords"], gen["fields"]
    ng = min(4, gf.shape[0])
    fig, axes = plt.subplots(1, ng, figsize=(4 * ng, 4))
    axes = np.atleast_1d(axes)
    for j in range(ng):
        quiver(axes[j], gc, gf[j], f"generated {j} (m={gen['num_vortices'][j]})")
    fig.suptitle("Synthesized steady fields sampled from the fitted distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "generated_previews.png"), dpi=120)
    plt.close(fig)
    logging.info(f"[plots] saved param_histograms.png, fit_previews.png, generated_previews.png to {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_bounds(cfg):
    b = cfg.model.bounds
    t = b["t"][0] if isinstance(b["t"], (list, tuple)) else float(b["t"])
    return {"rc": tuple(b["rc"]), "n": tuple(b["n"]), "s": tuple(b["s"]), "t": float(t)}


def main():
    parser = argparse.ArgumentParser(description="Fit Vatistas profile parameters to real flow patches.")
    parser.add_argument("--config", type=str, default="config/FittingVatistas.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max_patches", type=int, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args()

    cfg = EasyConfig()
    cfg.load(args.config, recursive=False)

    # Reuse the project's data-folder relocation (fills dataset.dat_dir + cache_dir).
    if cfg.dataset.get("dat_dir") is None:
        cfg.dataset["dat_dir"] = None
    relocate_flow2d_dataset_folder(cfg)

    if args.max_patches is not None:
        cfg.dataset["max_patches"] = args.max_patches
    if args.no_plots:
        cfg.output["save_plots"] = False

    seed = int(cfg.get("seed", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)

    dev_str = args.device or cfg.optim.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(dev_str)
    logging.info(f"[main] device={device}, seed={seed}")

    bounds = build_bounds(cfg)
    M = int(cfg.model.num_profiles)

    # 1) real-flow patches
    target, scales, meta = extract_patches(cfg, device)
    P = target.shape[0]
    coords = make_local_grid(int(cfg.dataset.patch_size), device)

    # 2) fit: simulated annealing -> gradient descent (optionally multi-restart)
    logging.info(f"[main] fitting {P} patches: simulated annealing -> gradient descent "
                 f"({int(cfg.optim.get('n_restarts', 1))} restart(s)) ...")
    params, shape_idx, fit_loss = fit_patches(coords, target, cfg, bounds, M, device)
    fit_quality_report(coords, target, params, shape_idx, bounds, tag="main")

    # 3) distribution over active profiles
    contrib = profile_contributions(coords, params, shape_idx, bounds)
    active_mask = contrib >= float(cfg.generate.min_contribution)
    dist = fit_distribution(params, shape_idx, active_mask)

    # 4) generate new steady fields from the fitted distribution
    gen = generate_steady_fields(dist, cfg, bounds, device)

    out_dir = args.out_dir or cfg.output.get("out_dir") or os.path.join(cfg.get("cache_dir", "./outputs/"), "vatistas_fit")
    save_outputs(out_dir, params, shape_idx, fit_loss, contrib, active_mask, meta, dist, gen)
    if bool(cfg.output.get("save_plots", True)):
        save_plots(out_dir, coords, target, params, shape_idx, bounds, dist, gen)

    logging.info("[main] all done.")


if __name__ == "__main__":
    main()
