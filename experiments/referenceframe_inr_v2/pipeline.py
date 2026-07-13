"""End-to-end experiment pipeline -- v2 (docs/referenceframe_inr_v2.md).

Modes (all share the frozen training recipe and the same CoordNet class):
  baseline     one CoordNet(m_B, d_B) fits v(x,y,t) directly            budget B
  pro_budget   tau-merge regions, per-region observed-field INRs        total <= B
  pro_quality  same, per-INR share = 3B / #INRs                         total <= 3B
  no_observer  same partition & budgets as pro_budget but q == 0        total <= B
               (fits raw v per region -> isolates the observer's contribution)
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field as dc_field
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for pth in (str(_ROOT), str(_HERE)):
    if pth not in sys.path:
        sys.path.insert(0, pth)

from killing2d import compute_cell_stats  # noqa: E402
from partition import merge_partition, split_windows, WindowPartition  # noqa: E402
from frame import make_region_samples, scatter_reconstruction  # noqa: E402
from inr import (TrainCfg, MinMax, train_inr_best_of_seeds, eval_inr, vpsnr,  # noqa: E402
                 coordnet_num_params, pick_m_for_budget, setup_determinism)
import synth  # noqa: E402

# 2D CFD dataset folder; override via env for non-desktop machines (e.g. Ibex)
import os  # noqa: E402
DATA_DIR = Path(os.environ.get(
    "PYFLOWVIS_DATA2D",
    r"C:\Users\xingdi\OneDrive - KAUST\WorkingInProcess\FLowVisAssets\flowData2D"))


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------
@dataclass
class FieldData:
    name: str
    data: np.ndarray          # (T, Y, X, 2) float64
    xs: np.ndarray
    ys: np.ndarray
    ts: np.ndarray

    @property
    def shape(self):
        return self.data.shape

    @property
    def dt(self) -> float:
        return float(self.ts[1] - self.ts[0]) if len(self.ts) > 1 else 0.0

    def data_bytes(self) -> int:
        return int(np.prod(self.data.shape)) * 4   # stored as float32 on disk


def _from_unsteady(name: str, f) -> FieldData:
    d = np.asarray(f.getDataAsNumpy() if hasattr(f, "getDataAsNumpy") else f.field,
                   dtype=np.float64)
    T, Y, X, _ = d.shape
    xs = np.linspace(f.domainMinBoundary[0], f.domainMaxBoundary[0], X)
    ys = np.linspace(f.domainMinBoundary[1], f.domainMaxBoundary[1], Y)
    ts = np.linspace(f.tmin, f.tmax, T)
    return FieldData(name=name, data=d, xs=xs, ys=ys, ts=ts)


def load_field(name: str) -> FieldData:
    if name in ("rfc", "rfc64"):
        from FLowUtils.AnalyticalFlowCreator import rotation_four_center
        return _from_unsteady(name, rotation_four_center((64, 64), 64))
    if name == "rfc128":
        from FLowUtils.AnalyticalFlowCreator import rotation_four_center
        return _from_unsteady(name, rotation_four_center((128, 128), 64))
    if name == "beads2d":
        from FLowUtils.AnalyticalFlowCreator import beadsFLow
        return _from_unsteady(name, beadsFLow((128, 128), 32))
    if name == "tworotor":
        xs = np.linspace(-2, 2, 96); ys = np.linspace(-2, 2, 96)
        ts = np.linspace(0, 2 * np.pi, 64)
        return FieldData("tworotor", synth.two_rotor_field(xs, ys, ts), xs, ys, ts)
    if name in ("cylinder2d", "boussinesq"):
        from FLowUtils.flowDatasetUtils.NetCDF_AmiraLoader import NetCDFLoader
        if name == "cylinder2d":
            f = NetCDFLoader.load_vector_field2d(str(DATA_DIR / "cylinder2d.nc"), 800, 960)
            f.resample2UnsteadyField((128, 320, 80))     # in place; (T, X, Y) tuple!
        else:
            f = NetCDFLoader.load_vector_field2d(str(DATA_DIR / "boussinesq.nc"), 800, 960)
            f.resample2UnsteadyField((128, 75, 225))
        return _from_unsteady(name, f)
    raise ValueError(f"unknown field '{name}'")


# ---------------------------------------------------------------------------
# experiment config
# ---------------------------------------------------------------------------
@dataclass
class ExpCfg:
    field: str = "rfc"
    m_base: int = 24            # baseline CoordNet width  -> budget B
    d_base: int = 4             # baseline depth (also used for region INRs)
    k_cell: int = 2             # minimal cell size (k x k pixels)
    tau: float = 0.05           # merge tolerance
    absorb_min_pixels: int = 0  # >0: absorb smaller regions post-hoc (spec deviation)
    n_windows: int = 2          # time windows (window length <= T/2)
    allow_full_window: bool = False   # diagnostic only: permit n_windows=1
    boundary_skip: int = 2
    epochs: int = 1000
    batch_size: int = 32000
    min_steps_per_epoch: int = 64   # v2.3 adaptive batch
    lr: float = 1e-5
    lr_final: float = 1e-6
    grad_clip: float = 1.0
    weight_decay: float = 1e-6
    seed: int = 0
    n_seeds: int = 3            # v2.2: best-of-k seeds per INR (encode-time search)
    device: str = ""            # "" -> cuda if available
    out_dir: str = ""
    modes: tuple = ("baseline", "pro_budget", "pro_quality", "no_observer")

    def train_cfg(self) -> TrainCfg:
        return TrainCfg(epochs=self.epochs, batch_size=self.batch_size,
                        min_steps_per_epoch=self.min_steps_per_epoch,
                        lr=self.lr, lr_final=self.lr_final, grad_clip=self.grad_clip,
                        weight_decay=self.weight_decay)


def budget_B(cfg: ExpCfg) -> int:
    return coordnet_num_params(cfg.m_base, cfg.d_base, k=3, p=2)


# ---------------------------------------------------------------------------
# baseline
# ---------------------------------------------------------------------------
def baseline_coords_values(fd: FieldData) -> tuple[np.ndarray, np.ndarray, MinMax]:
    T, Y, X, _ = fd.shape
    gt, gy, gx = np.meshgrid(fd.ts, fd.ys, fd.xs, indexing="ij")
    coords = np.stack([gx.ravel(), gy.ravel(), gt.ravel()], axis=-1)
    cmm = MinMax.fit(coords)
    vals = fd.data.reshape(-1, 2)
    vmm = MinMax.fit(vals)
    return cmm.encode(coords), vmm.encode(vals), vmm


def run_baseline(fd: FieldData, cfg: ExpCfg, device, log=print) -> dict:
    coords_n, vals_n, vmm = baseline_coords_values(fd)
    model, st = train_inr_best_of_seeds(coords_n, vals_n, cfg.m_base, cfg.d_base,
                                        cfg.train_cfg(), device, seed_base=cfg.seed,
                                        n_seeds=cfg.n_seeds,
                                        tag=f"{fd.name}/baseline", log=log)
    pred_n = eval_inr(model, coords_n, device)
    recon = vmm.decode(pred_n).reshape(fd.shape)
    psnr = vpsnr(recon, fd.data)
    side_bytes = 10 * 4     # coord ranges (6) + value lo/hi (4) floats
    total_bytes = st["params"] * 4 + side_bytes
    return {"mode": "baseline", "psnr": psnr, "params_total": st["params"],
            "side_info_bytes": side_bytes, "total_bytes": total_bytes,
            "compression_ratio": fd.data_bytes() / total_bytes,
            "n_inrs": 1, "regions_per_window": [1],
            "train_time_s": st["train_time_s"], "detail": [st]}


# ---------------------------------------------------------------------------
# proposed
# ---------------------------------------------------------------------------
def compute_partitions(fd: FieldData, cfg: ExpCfg, log=print) -> list[WindowPartition]:
    t0 = time.time()
    stats = compute_cell_stats(fd.data, fd.xs, fd.ys, fd.dt,
                               k=cfg.k_cell, boundary_skip=cfg.boundary_skip)
    windows = split_windows(fd.shape[0], cfg.n_windows, allow_full=cfg.allow_full_window)
    parts = []
    for (it0, it1) in windows:
        part = merge_partition(stats, it0, it1, cfg.tau,
                               absorb_min_pixels=cfg.absorb_min_pixels)
        parts.append(part)
        rr = [f"N={part.n_regions}" +
              (f" (absorbed {part.n_absorbed})" if part.n_absorbed else "")]
        for r in part.regions:
            rr.append(f"(pix={int(r.npix)}, E/E0={r.E / max(r.E0, 1e-300):.2e})")
        log(f"  window [{it0},{it1}) tau={cfg.tau}: " + " ".join(rr))
    log(f"  partition total {time.time() - t0:.1f}s")
    return parts


def run_proposed(fd: FieldData, cfg: ExpCfg, parts: list[WindowPartition],
                 device, budget_factor: float, use_observer: bool,
                 mode_name: str, log=print) -> dict:
    T, Y, X, _ = fd.shape
    B = budget_B(cfg)
    n_inrs = sum(p.n_regions for p in parts)
    share = int(budget_factor * B / n_inrs)
    d_r = cfg.d_base

    recon = np.full(fd.shape, np.nan)
    details = []
    t_start = time.time()
    inr_idx = 0
    side_bytes = 0
    for w_i, part in enumerate(parts):
        Tw = part.it1 - part.it0
        side_bytes += part.labels_cells.size * 2            # uint16 cell label map
        for r_i, reg in enumerate(part.regions):
            q = reg.q if use_observer else np.zeros_like(reg.q)
            if use_observer:
                side_bytes += Tw * 3 * 4                    # (a,b,c)(t) float32
            pix_mask = part.labels_pixels == r_i
            samples = make_region_samples(fd.data, fd.xs, fd.ys, fd.dt,
                                          pix_mask, part.it0, part.it1, q)
            ximm = MinMax.fit(samples.xi)
            vmm = MinMax.fit(samples.vtil)
            side_bytes += (4 + 4) * 4 + 2                   # xi bbox + value lo/hi + m_r
            coords_n = np.concatenate([ximm.encode(samples.xi),
                                       samples.tn[:, None].astype(np.float32)], axis=1)
            vals_n = vmm.encode(samples.vtil)
            m_r = pick_m_for_budget(share, d_r)
            tag = f"{fd.name}/{mode_name} w{w_i}r{r_i} m={m_r}"
            model, st = train_inr_best_of_seeds(coords_n, vals_n, m_r, d_r,
                                                cfg.train_cfg(), device,
                                                seed_base=cfg.seed + 1000 * w_i + r_i,
                                                n_seeds=cfg.n_seeds, tag=tag, log=log)
            pred_n = eval_inr(model, coords_n, device)
            vtil_pred = vmm.decode(pred_n)
            scatter_reconstruction(recon, samples, vtil_pred)
            st.update({"window": w_i, "region": r_i, "pixels": int(samples.n_pix),
                       "E_over_E0": reg.E / max(reg.E0, 1e-300),
                       "use_observer": use_observer})
            details.append(st)
            inr_idx += 1

    assert not np.isnan(recon).any(), "reconstruction coverage hole"
    psnr = vpsnr(recon, fd.data)
    params_total = sum(st["params"] for st in details)
    total_bytes = params_total * 4 + side_bytes
    return {"mode": mode_name, "psnr": psnr, "params_total": params_total,
            "budget_B": B, "budget_factor": budget_factor, "per_inr_share": share,
            "side_info_bytes": side_bytes, "total_bytes": total_bytes,
            "compression_ratio": fd.data_bytes() / total_bytes,
            "n_inrs": n_inrs, "regions_per_window": [p.n_regions for p in parts],
            "tau": cfg.tau, "train_time_s": time.time() - t_start, "detail": details}


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
def run_experiment(cfg: ExpCfg, log=print) -> dict:
    setup_determinism(cfg.seed)
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    fd = load_field(cfg.field)
    log(f"[{fd.name}] shape (T,Y,X,2)={fd.shape}, data={fd.data_bytes()/2**20:.2f} MiB, "
        f"B={budget_B(cfg)} params (m={cfg.m_base}, d={cfg.d_base}), device={device}")

    results = {}
    parts = None
    if any(m != "baseline" for m in cfg.modes):
        log(f"[{fd.name}] tau-merge partition:")
        parts = compute_partitions(fd, cfg, log=log)

    for mode in cfg.modes:
        log(f"[{fd.name}] === mode {mode} ===")
        if mode == "baseline":
            res = run_baseline(fd, cfg, device, log=log)
        elif mode == "pro_budget":
            res = run_proposed(fd, cfg, parts, device, 1.0, True, mode, log=log)
        elif mode == "pro_quality":
            res = run_proposed(fd, cfg, parts, device, 3.0, True, mode, log=log)
        elif mode == "no_observer":
            res = run_proposed(fd, cfg, parts, device, 1.0, False, mode, log=log)
        else:
            raise ValueError(f"unknown mode {mode}")
        results[mode] = res
        log(f"[{fd.name}] {mode}: PSNR={res['psnr']:.2f} dB  params={res['params_total']:,}"
            f"  bytes={res['total_bytes']:,}  CR={res['compression_ratio']:.1f}x"
            f"  #INR={res['n_inrs']}  ({res['train_time_s']:.0f}s)")

    out = {"cfg": asdict(cfg), "field_shape": list(fd.shape),
           "data_bytes": fd.data_bytes(), "results": results}
    if cfg.out_dir:
        od = Path(cfg.out_dir); od.mkdir(parents=True, exist_ok=True)
        with open(od / f"{fd.name}_metrics.json", "w") as f:
            json.dump(out, f, indent=2, default=float)
        if parts is not None:
            np.savez_compressed(od / f"{fd.name}_labels.npz",
                                **{f"w{i}_labels": p.labels_pixels for i, p in enumerate(parts)},
                                **{f"w{i}_q_r{j}": r.q for i, p in enumerate(parts)
                                   for j, r in enumerate(p.regions)})
        log(f"[{fd.name}] wrote {od}")
    return out
