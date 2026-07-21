"""3D experiment pipeline -- referenceframe_inr_3d.

Structure mirrors experiments/referenceframe_inr_v2/pipeline.py (frozen); the INR
trainer, normalizers, and architectures are IMPORTED from v2 unchanged (they are
dimension-generic: k = coords.shape[1], p = values.shape[1]).  3D specifics:
coords are (x, y, z, t_norm) -> k = 4, values (vx, vy, vz) -> p = 3; killing/
partition/frame come from the 3D modules.

Byte accounting (v2 rules carried over):
  baseline   side info = coord ranges (8 floats) + value lo/hi (6 floats)
  proposed   side info = 1 observer-variant tag byte
             + per window: cell label map (uint16/cell) IF N > 1 (N==1 stores none)
             + per region: observer params (variant-dependent, see
               observer_stored_bytes_3d) + xi bbox (6+6 floats) + value lo/hi
               (6 floats) + m_r (uint16)
Budget rule: total bytes = params*4 + side info <= budget_frac * raw field bytes
(float32), asserted on both plan and actuals.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, asdict, field as dc_field
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
_V2 = _ROOT / "experiments" / "referenceframe_inr_v2"
for pth in (str(_ROOT), str(_HERE), str(_V2)):
    if pth not in sys.path:
        sys.path.insert(0, pth)

import torch  # noqa: E402

from killing3d import compute_cell_stats_3d  # noqa: E402
from partition3d import (merge_partition_3d, single_region_partition,  # noqa: E402
                         split_windows, WindowPartition3D)
from frame3d import make_region_samples_3d, scatter_reconstruction_3d  # noqa: E402
from killing3d import solve_killing_3d, solve_killing_trans_3d  # noqa: E402
from inr import (TrainCfg, MinMax, train_inr_best_of_seeds, eval_inr, vpsnr,  # noqa: E402
                 coordnet_num_params, pick_m_for_budget, setup_determinism)

# 3D CFD dataset folder; override via env for non-desktop machines (e.g. Ibex)
import os  # noqa: E402
DATA_DIR_3D = Path(os.environ.get(
    "PYFLOWVIS_DATA3D",
    r"C:\Users\xingdi\OneDrive - KAUST\WorkingInProcess\FLowVisAssets\flowData3D"))


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------
@dataclass
class FieldData3D:
    name: str
    data: np.ndarray          # (T, Z, Y, X, 3) float64
    xs: np.ndarray
    ys: np.ndarray
    zs: np.ndarray
    ts: np.ndarray

    @property
    def shape(self):
        return self.data.shape

    @property
    def dt(self) -> float:
        return float(self.ts[1] - self.ts[0]) if len(self.ts) > 1 else 0.0

    def data_bytes(self) -> int:
        return int(np.prod(self.data.shape)) * 4   # stored as float32 on disk


# name -> filename of standard-layout (tdim,zdim,ydim,xdim) u/v/w NetCDF files
NC_FILES = {
    "deltawing": "deltaWing_mag0_3reesampled.nc",
    "halfcyl160": "halfcylinderRe160Resampled.nc",
    "halfcyl640": "halfcylinderRe640resampled.nc",
    "smoke": "SmokeBuoyancy80_239.nc",
}


def load_nc_3d(name: str, stride_t: int = 1, stride_xyz: int = 1,
               t_max: int = 0) -> FieldData3D:
    """Standard-layout loader: vars u/v/w over (tdim, zdim, ydim, xdim) plus
    coordinate vectors x/y/z/t.  Optional integer strides downsample time / all
    three space axes (protocol note: strided runs are their own dataset variant
    -- never mix numbers across strides)."""
    from netCDF4 import Dataset
    fn = DATA_DIR_3D / NC_FILES[name]
    if not fn.exists():
        raise FileNotFoundError(
            f"dataset file not found: {fn} -- set the PYFLOWVIS_DATA3D env var "
            f"to the folder holding {NC_FILES[name]}")
    ds = Dataset(str(fn))
    st, sxyz = int(stride_t), int(stride_xyz)
    tsl = slice(0, int(t_max) if t_max else None, st)
    ssl = slice(None, None, sxyz)
    u = ds.variables["u"][tsl, ssl, ssl, ssl]
    v = ds.variables["v"][tsl, ssl, ssl, ssl]
    w = ds.variables["w"][tsl, ssl, ssl, ssl]
    data = np.stack([np.asarray(u), np.asarray(v), np.asarray(w)],
                    axis=-1).astype(np.float64)
    xs = np.asarray(ds.variables["x"][ssl], dtype=np.float64)
    ys = np.asarray(ds.variables["y"][ssl], dtype=np.float64)
    zs = np.asarray(ds.variables["z"][ssl], dtype=np.float64)
    ts = np.asarray(ds.variables["t"][tsl], dtype=np.float64)
    ds.close()
    if st > 1 or sxyz > 1 or t_max:
        name += f"_s{st}x{sxyz}" + (f"_t{t_max}" if t_max else "")
    return FieldData3D(name=name, data=data, xs=xs, ys=ys, zs=zs, ts=ts)


def load_field_3d(name: str, stride_t: int = 1, stride_xyz: int = 1,
                  t_max: int = 0) -> FieldData3D:
    if name in NC_FILES:
        return load_nc_3d(name, stride_t, stride_xyz, t_max)
    if name == "rot3d":            # synthetic smoke/validation field
        import synth3d
        xs = ys = zs = np.linspace(-2, 2, 48 // max(stride_xyz, 1))
        ts = np.linspace(0, 2 * np.pi, 32 // max(stride_t, 1))
        data = synth3d.compose_rotating_frame_3d(
            synth3d.cells_steady_3d, 0.7, (0.3, 0.5, 0.8), (0.3, -0.2, 0.1),
            xs, ys, zs, ts)
        return FieldData3D("rot3d", data, xs, ys, zs, ts)
    raise ValueError(f"unknown 3D field '{name}'")


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
@dataclass
class ExpCfg3D:
    field: str = "deltawing"
    model: str = "coordnet"     # coordnet (SIREN) | mlp | finer   (v2 registry)
    finer_first_bias_scale: float | None = None
    stride_t: int = 1           # dataset time stride (downsample protocol tag)
    stride_xyz: int = 1         # dataset space stride
    t_max: int = 0              # 0 = all timesteps (applied before stride)
    m_base: int = 24
    d_base: int = 10
    budget_frac: float = 0.0
    max_inrs: int = 0
    k_cell: int = 4             # cell size (k^3 voxels); 4^3 keeps stats <1 GB
    tau: float = 0.05
    alloc: str = "uniform"      # uniform | pixels | capsmall  (v2 semantics)
    absorb_min_pixels: int = 0  # in voxels
    n_windows: int = 1
    allow_full_window: bool = True
    observer: str = "tvfull"    # tvfull | tvtrans | constfull | consttrans
    boundary_skip: int = 2
    epochs: int = 1000
    batch_size: int = 32000
    min_steps_per_epoch: int = 64
    max_steps_per_epoch: int = 0    # NEW vs v2: >0 caps steps/epoch (3D fields
                                    # have 50x the samples of 2D; the cap keeps
                                    # wall-clock sane and is an explicit,
                                    # logged protocol knob -- an epoch is then a
                                    # fixed-size random subsample, not a sweep)
    lr: float = 1e-5
    lr_final: float = 1e-6
    warmup_frac: float = 0.0
    grad_clip: float = 1.0
    weight_decay: float = 1e-6
    log_every: int = 100
    seed: int = 0
    n_seeds: int = 1
    device: str = ""
    out_dir: str = ""
    modes: tuple = ("baseline", "pro_budget")

    def train_cfg(self) -> TrainCfg:
        return TrainCfg(epochs=self.epochs, batch_size=self.batch_size,
                        min_steps_per_epoch=self.min_steps_per_epoch,
                        lr=self.lr, lr_final=self.lr_final,
                        warmup_frac=self.warmup_frac, grad_clip=self.grad_clip,
                        weight_decay=self.weight_decay, log_every=self.log_every)

    def model_kwargs(self) -> dict:
        if self.model == "finer" and self.finer_first_bias_scale is not None:
            return {"first_bias_scale": self.finer_first_bias_scale}
        return {}


def budget_B(cfg: ExpCfg3D) -> int:
    return coordnet_num_params(cfg.m_base, cfg.d_base, k=4, p=3)


def _subsample_train(coords_n: np.ndarray, vals_n: np.ndarray, cfg: ExpCfg3D,
                     seed: int, log=print):
    """Optional fixed-budget training subsample (max_steps_per_epoch cap).
    Evaluation ALWAYS runs on the full grid; only the training set shrinks."""
    if cfg.max_steps_per_epoch <= 0:
        return coords_n, vals_n
    n_keep = cfg.max_steps_per_epoch * cfg.batch_size
    n = coords_n.shape[0]
    if n <= n_keep:
        return coords_n, vals_n
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=n_keep, replace=False)
    log(f"    train subsample: {n:,} -> {n_keep:,} samples "
        f"({cfg.max_steps_per_epoch} steps/epoch cap; eval stays full-grid)")
    return coords_n[idx], vals_n[idx]


# ---------------------------------------------------------------------------
# baseline
# ---------------------------------------------------------------------------
def baseline_coords_values(fd: FieldData3D):
    gt, gz, gy, gx = np.meshgrid(fd.ts, fd.zs, fd.ys, fd.xs, indexing="ij")
    coords = np.stack([gx.ravel(), gy.ravel(), gz.ravel(), gt.ravel()], axis=-1)
    cmm = MinMax.fit(coords)
    vals = fd.data.reshape(-1, 3)
    vmm = MinMax.fit(vals)
    return cmm.encode(coords), vmm.encode(vals), vmm


def run_baseline(fd: FieldData3D, cfg: ExpCfg3D, device, log=print) -> dict:
    coords_n, vals_n, vmm = baseline_coords_values(fd)
    side_bytes = (8 + 6) * 4    # coord ranges (8 floats) + value lo/hi (6)
    m_b = cfg.m_base
    if cfg.budget_frac > 0:
        budget_p = int((cfg.budget_frac * fd.data_bytes() - side_bytes) // 4)
        m_b = pick_m_for_budget(budget_p, cfg.d_base, k=4, p=3)
        log(f"  budget_frac={cfg.budget_frac}: {budget_p:,} params allowed -> "
            f"m={m_b} d={cfg.d_base} "
            f"({coordnet_num_params(m_b, cfg.d_base, k=4, p=3):,} params)")
    tr_c, tr_v = _subsample_train(coords_n, vals_n, cfg, cfg.seed, log=log)
    model, st = train_inr_best_of_seeds(tr_c, tr_v, m_b, cfg.d_base,
                                        cfg.train_cfg(), device,
                                        seed_base=cfg.seed, n_seeds=cfg.n_seeds,
                                        tag=f"{fd.name}/baseline", log=log,
                                        model_name=cfg.model,
                                        model_kwargs=cfg.model_kwargs())
    pred_n = eval_inr(model, coords_n, device)
    recon = vmm.decode(pred_n).reshape(fd.shape)
    psnr = vpsnr(recon, fd.data)
    total_bytes = st["params"] * 4 + side_bytes
    if cfg.budget_frac > 0:
        assert total_bytes <= cfg.budget_frac * fd.data_bytes(), "over byte budget"
    return {"mode": "baseline", "psnr": psnr, "params_total": st["params"],
            "budget_frac": cfg.budget_frac,
            "side_info_bytes": side_bytes, "total_bytes": total_bytes,
            "compression_ratio": fd.data_bytes() / total_bytes,
            "n_inrs": 1, "regions_per_window": [1],
            "train_time_s": st["train_time_s"], "detail": [st]}


# ---------------------------------------------------------------------------
# proposed
# ---------------------------------------------------------------------------
def observer_stored_bytes_3d(observer: str, tw: int) -> int:
    """Stored bytes of ONE region's observer parameters (float32 each; the
    variant tag byte is global).  3D: 6 DOF full, 3 DOF translation-only."""
    return {"tvfull": tw * 6 * 4, "tvtrans": tw * 3 * 4,
            "constfull": 6 * 4, "consttrans": 3 * 4}[observer]


def resolve_observer_3d(reg, observer: str):
    """Re-solve one region's observer under the requested parameterization from
    its window-sliced sufficient statistics.  Returns (q (Tw,6), E_variant)."""
    if observer == "tvfull":
        return reg.q, reg.E
    if observer == "tvtrans":
        q, E = solve_killing_trans_3d(reg.AtA, reg.g, reg.e0)
        return q, float(E.sum())
    Tw = reg.AtA.shape[0]
    if observer == "constfull":
        qc, _ = solve_killing_3d(reg.AtA.sum(0), reg.g.sum(0), reg.e0.sum())
    elif observer == "consttrans":
        qc, _ = solve_killing_trans_3d(reg.AtA.sum(0), reg.g.sum(0), reg.e0.sum())
    else:
        raise ValueError(f"unknown observer '{observer}'")
    E = float(reg.e0.sum() + qc @ reg.g.sum(0))
    return np.tile(qc, (Tw, 1)), max(E, 0.0)


def compute_partitions_3d(fd: FieldData3D, cfg: ExpCfg3D, log=print
                          ) -> list[WindowPartition3D]:
    t0 = time.time()
    stats = compute_cell_stats_3d(fd.data, fd.xs, fd.ys, fd.zs, fd.dt,
                                  k=cfg.k_cell, boundary_skip=cfg.boundary_skip)
    windows = split_windows(fd.shape[0], cfg.n_windows,
                            allow_full=cfg.allow_full_window)
    parts = []
    for (it0, it1) in windows:
        if cfg.tau < 0:     # M=1 fast path (tau=-1 == whole domain, no merge)
            part = single_region_partition(stats, it0, it1)
        else:
            part = merge_partition_3d(stats, it0, it1, cfg.tau,
                                      absorb_min_pixels=cfg.absorb_min_pixels)
        parts.append(part)
        rr = [f"N={part.n_regions}" +
              (f" (absorbed {part.n_absorbed})" if part.n_absorbed else "")]
        for r in part.regions:
            rr.append(f"(vox={int(r.npix)}, E/E0={r.E / max(r.E0, 1e-300):.2e})")
        log(f"  window [{it0},{it1}) tau={cfg.tau}: " + " ".join(rr))
    log(f"  partition total {time.time() - t0:.1f}s")
    return parts


def run_proposed(fd: FieldData3D, cfg: ExpCfg3D, parts: list[WindowPartition3D],
                 device, budget_factor: float, use_observer: bool,
                 mode_name: str, log=print) -> dict:
    B = budget_B(cfg)
    n_inrs = sum(p.n_regions for p in parts)
    if cfg.budget_frac > 0:
        side_planned = 1 if use_observer else 0
        for part in parts:
            tw = part.it1 - part.it0
            if part.n_regions > 1:
                side_planned += part.labels_cells.size * 2
            side_planned += part.n_regions * (
                (observer_stored_bytes_3d(cfg.observer, tw) if use_observer else 0)
                + (6 + 6) * 4 + 2)
        params_pool = int((cfg.budget_frac * fd.data_bytes() - side_planned) // 4)
        assert params_pool > n_inrs, "budget_frac too small for side info"
        share = params_pool // n_inrs
        log(f"  budget_frac={cfg.budget_frac}: pool {params_pool:,} params after "
            f"{side_planned:,} B side info -> share {share:,}/INR (x{n_inrs})")
    else:
        params_pool = None
        share = int(budget_factor * B / n_inrs)
    d_r = cfg.d_base

    total_wpix = sum(float((p.labels_pixels == r_i).sum()) * (p.it1 - p.it0)
                     for p in parts for r_i in range(p.n_regions))
    share_map = {}
    if cfg.alloc == "capsmall":
        pool_total = params_pool if params_pool is not None else int(budget_factor * B)
        reg_ns = [(w_i, r_i, int((p.labels_pixels == r_i).sum()) * (p.it1 - p.it0))
                  for w_i, p in enumerate(parts) for r_i in range(p.n_regions)]
        big = max(reg_ns, key=lambda t: t[2])[:2]
        spent = 0
        for w_i, r_i, ns in reg_ns:
            if (w_i, r_i) == big:
                continue
            cap = coordnet_num_params(
                pick_m_for_budget(min(ns, pool_total // n_inrs), d_r, k=4, p=3),
                d_r, k=4, p=3)
            share_map[(w_i, r_i)] = cap
            spent += cap
        share_map[big] = pool_total - spent
        log(f"  alloc=capsmall: largest region w{big[0]}r{big[1]} gets "
            f"{share_map[big]:,} params; {n_inrs - 1} small region(s) capped "
            f"({spent:,} total)")

    recon = np.full(fd.shape, np.nan)
    details = []
    t_start = time.time()
    side_bytes = 1 if use_observer else 0
    for w_i, part in enumerate(parts):
        Tw = part.it1 - part.it0
        if part.n_regions > 1:
            side_bytes += part.labels_cells.size * 2
        for r_i, reg in enumerate(part.regions):
            if use_observer:
                q, E_obs = resolve_observer_3d(reg, cfg.observer)
                side_bytes += observer_stored_bytes_3d(cfg.observer, Tw)
            else:
                q, E_obs = np.zeros_like(reg.q), reg.E0
            pix_mask = part.labels_pixels == r_i
            samples = make_region_samples_3d(fd.data, fd.xs, fd.ys, fd.zs,
                                             fd.dt, pix_mask, part.it0,
                                             part.it1, q)
            ximm = MinMax.fit(samples.xi)
            vmm = MinMax.fit(samples.vtil)
            side_bytes += (6 + 6) * 4 + 2       # xi bbox + value lo/hi + m_r
            coords_n = np.concatenate(
                [ximm.encode(samples.xi),
                 samples.tn[:, None].astype(np.float32)], axis=1)
            vals_n = vmm.encode(samples.vtil)
            if cfg.alloc == "pixels":
                pool = params_pool if params_pool is not None else budget_factor * B
                w_share = int(pool * samples.xi.shape[0] / total_wpix)
            elif cfg.alloc == "capsmall":
                w_share = share_map[(w_i, r_i)]
            else:
                w_share = share
            m_r = pick_m_for_budget(w_share, d_r, k=4, p=3)
            tag = f"{fd.name}/{mode_name} w{w_i}r{r_i} m={m_r}"
            tr_c, tr_v = _subsample_train(coords_n, vals_n, cfg,
                                          cfg.seed + 7 * w_i + r_i, log=log)
            model, st = train_inr_best_of_seeds(tr_c, tr_v, m_r, d_r,
                                                cfg.train_cfg(), device,
                                                seed_base=cfg.seed + 1000 * w_i + r_i,
                                                n_seeds=cfg.n_seeds, tag=tag,
                                                log=log, model_name=cfg.model,
                                                model_kwargs=cfg.model_kwargs())
            pred_n = eval_inr(model, coords_n, device)
            vtil_pred = vmm.decode(pred_n)
            scatter_reconstruction_3d(recon, samples, vtil_pred)
            st.update({"window": w_i, "region": r_i,
                       "voxels": int(samples.n_pix),
                       "E_over_E0": reg.E / max(reg.E0, 1e-300),
                       "E_obs_over_E0": E_obs / max(reg.E0, 1e-300),
                       "observer": cfg.observer if use_observer else "none",
                       "use_observer": use_observer})
            details.append(st)

    assert not np.isnan(recon).any(), "reconstruction coverage hole"
    psnr = vpsnr(recon, fd.data)
    params_total = sum(st["params"] for st in details)
    total_bytes = params_total * 4 + side_bytes
    if cfg.budget_frac > 0:
        assert side_bytes == side_planned, "side-info plan drifted from actual"
        assert total_bytes <= cfg.budget_frac * fd.data_bytes(), "over byte budget"
    return {"mode": mode_name, "psnr": psnr, "params_total": params_total,
            "budget_B": B, "budget_factor": budget_factor,
            "per_inr_share": share, "budget_frac": cfg.budget_frac,
            "observer": cfg.observer if use_observer else "none",
            "side_info_bytes": side_bytes, "total_bytes": total_bytes,
            "compression_ratio": fd.data_bytes() / total_bytes,
            "n_inrs": n_inrs, "regions_per_window": [p.n_regions for p in parts],
            "tau": cfg.tau, "train_time_s": time.time() - t_start,
            "detail": details}


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
def run_experiment_3d(cfg: ExpCfg3D, log=print) -> dict:
    assert cfg.epochs <= 2000, "hard epoch cap (user rule) violated"
    setup_determinism(cfg.seed)
    device = torch.device(cfg.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    fd = load_field_3d(cfg.field, cfg.stride_t, cfg.stride_xyz, cfg.t_max)
    log(f"[{fd.name}] shape (T,Z,Y,X,3)={fd.shape}, "
        f"data={fd.data_bytes() / 2**20:.2f} MiB, model={cfg.model}, "
        f"device={device}"
        + (f", budget_frac={cfg.budget_frac} "
           f"({cfg.budget_frac * fd.data_bytes():,.0f} B)"
           if cfg.budget_frac > 0 else ""))

    results = {}
    parts = None
    if any(m != "baseline" for m in cfg.modes):
        log(f"[{fd.name}] tau-merge partition:")
        parts = compute_partitions_3d(fd, cfg, log=log)
        if cfg.max_inrs > 0:
            m_tot = sum(p.n_regions for p in parts)
            assert m_tot <= cfg.max_inrs, (
                f"partition yields {m_tot} INRs > max_inrs={cfg.max_inrs} "
                f"(tau={cfg.tau}, absorb={cfg.absorb_min_pixels}, "
                f"n_windows={cfg.n_windows}) -- refusing to split the budget")

    for mode in cfg.modes:
        log(f"[{fd.name}] === mode {mode} ===")
        if mode == "baseline":
            res = run_baseline(fd, cfg, device, log=log)
        elif mode == "pro_budget":
            res = run_proposed(fd, cfg, parts, device, 1.0, True, mode, log=log)
        elif mode == "no_observer":
            res = run_proposed(fd, cfg, parts, device, 1.0, False, mode, log=log)
        else:
            raise ValueError(f"unknown mode {mode}")
        results[mode] = res
        log(f"[{fd.name}] {mode}: PSNR={res['psnr']:.2f} dB  "
            f"params={res['params_total']:,}  bytes={res['total_bytes']:,}  "
            f"CR={res['compression_ratio']:.1f}x  #INR={res['n_inrs']}  "
            f"({res['train_time_s']:.0f}s)")

    out = {"cfg": asdict(cfg), "field_shape": list(fd.shape),
           "data_bytes": fd.data_bytes(), "results": results}
    if cfg.out_dir:
        od = Path(cfg.out_dir); od.mkdir(parents=True, exist_ok=True)
        with open(od / f"{fd.name}_metrics.json", "w") as f:
            json.dump(out, f, indent=2, default=float)
        if parts is not None:
            np.savez_compressed(
                od / f"{fd.name}_labels.npz",
                **{f"w{i}_labels": p.labels_pixels for i, p in enumerate(parts)},
                **{f"w{i}_q_r{j}": r.q for i, p in enumerate(parts)
                   for j, r in enumerate(p.regions)})
        log(f"[{fd.name}] wrote {od}")
    return out
