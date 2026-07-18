"""End-to-end experiment pipeline -- v2 (docs/referenceframe_inr_v2.md).

Modes (all share the frozen training recipe and the same CoordNet class):
  baseline     one CoordNet(m_B, d_B) fits v(x,y,t) directly            budget B
  pro_budget   tau-merge regions, per-region observed-field INRs        total <= B
  pro_quality  same, per-INR share = 4B / #INRs                         total <= 4B
               (3B before 2026-07-15; historical numbers keep the 3B label)
  no_observer  same partition & budgets as pro_budget but q == 0        total <= B
               (fits raw v per region -> isolates the observer's contribution)

Strict-compression budgets (mainExp_compress_1.1): cfg.budget_frac > 0 replaces
the B-derived budget with frac * raw-field-bytes for baseline / pro_budget /
no_observer -- baseline width is solved from the budget, proposed deducts the
exact side info first, and both assert total_bytes <= frac * field bytes.
Sizing preview: budget_calc.py.
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

from killing2d import compute_cell_stats, solve_killing, solve_killing_trans  # noqa: E402
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
        fn = DATA_DIR / f"{name}.nc"
        if not fn.exists():
            raise FileNotFoundError(
                f"dataset file not found: {fn} -- set the PYFLOWVIS_DATA2D env var "
                f"to the folder holding {name}.nc on this machine")
        f = NetCDFLoader.load_vector_field2d(str(fn), 800, 960)
        if f is None:
            raise RuntimeError(f"NetCDFLoader returned None for {fn} (corrupt file?)")
        if name == "cylinder2d":
            f.resample2UnsteadyField((128, 320, 80))     # in place; (T, X, Y) tuple!
        else:
            f.resample2UnsteadyField((128, 75, 225))
        return _from_unsteady(name, f)
    if name.startswith("gerris") and name[len("gerris"):].isdigit():
        # GerrisTinySet: 8 independent unsteady 2D Gerris flows, one per .am file.
        # gerris0..gerris7 select the i-th .am (sorted). Load ONE file only (not the
        # whole ~2GB set) and downsample to (T,X,Y)=(128,128,256), matching the
        # cylinder/boussinesq experiment scale so PSNR numbers are comparable.
        idx = int(name[len("gerris"):])
        from FLowUtils.flowDatasetUtils.NetCDF_AmiraLoader import AmiraLoader
        folder = DATA_DIR / "Gunther_GerrisSolver_ML_FlowMap_TinyTest"
        am_files = sorted(folder.glob("*.am"))
        if not am_files:
            raise FileNotFoundError(
                f"no .am files in {folder} -- set PYFLOWVIS_DATA2D to the folder "
                f"holding Gunther_GerrisSolver_ML_FlowMap_TinyTest")
        if idx >= len(am_files):
            raise IndexError(f"gerris{idx}: only {len(am_files)} .am files in {folder}")
        f = AmiraLoader.load_vector_field2d(str(am_files[idx]))
        f.resample2UnsteadyField((128, 128, 256))   # (T, X, Y)
        return _from_unsteady(name, f)
    raise ValueError(f"unknown field '{name}'")


# ---------------------------------------------------------------------------
# experiment config
# ---------------------------------------------------------------------------
@dataclass
class ExpCfg:
    field: str = "rfc"
    model: str = "coordnet"     # INR architecture: coordnet (SIREN, default) |
                                # mlp (v_MLP0.0 residual ReLU) | finer (v_FINER0.0)
    finer_first_bias_scale: float | None = None  # FINER first-layer bias U(+-k);
                                # None = official repo default (standard bias init)
    m_base: int = 24            # baseline CoordNet width  -> budget B
    d_base: int = 4             # baseline depth (also used for region INRs)
    budget_frac: float = 0.0    # >0: strict-compression budget = frac * raw field
                                # bytes; overrides B (m_base then ignored for the
                                # baseline width). pro_quality must not use it.
    max_inrs: int = 0           # >0: hard cap on total INR count -- fail loudly if
                                # the partition yields more (user 2026-07-16: tiny
                                # budgets must not be split across many regions)
    k_cell: int = 2             # minimal cell size (k x k pixels)
    tau: float = 0.05           # merge tolerance
    alloc: str = "uniform"      # per-INR budget split: uniform (spec) | pixels
    absorb_min_pixels: int = 0  # >0: absorb smaller regions post-hoc (spec deviation)
    n_windows: int = 2          # time windows (window length <= T/2)
    allow_full_window: bool = False   # diagnostic only: permit n_windows=1
    observer: str = "tvfull"    # observer parameterization (Verify_compresswin_1.3):
                                # tvfull    per-timestep (a,b,c)(t)   [historical default]
                                # tvtrans   per-timestep (a,b)(t), c=0
                                # constfull one (a,b,c) per (window, region), joint LS
                                # consttrans one (a,b), c=0 (uniformly translating frame)
                                # Partitioning always uses the tv-full criterion; the
                                # variant only re-solves the q used for the transform
                                # (and shrinks the stored side info accordingly).
    boundary_skip: int = 2
    epochs: int = 1000
    batch_size: int = 32000
    min_steps_per_epoch: int = 64   # v2.3 adaptive batch
    lr: float = 1e-5
    lr_final: float = 1e-6
    warmup_frac: float = 0.0        # linear lr warmup fraction (see TrainCfg)
    grad_clip: float = 1.0
    weight_decay: float = 1e-6
    log_every: int = 100        # epoch-MSE print interval (dense for smoke runs)
    seed: int = 0
    n_seeds: int = 3            # v2.2: best-of-k seeds per INR (encode-time search)
    device: str = ""            # "" -> cuda if available
    out_dir: str = ""
    modes: tuple = ("baseline", "pro_budget", "pro_quality", "no_observer")

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
    side_bytes = 10 * 4     # coord ranges (6) + value lo/hi (4) floats
    m_b = cfg.m_base
    if cfg.budget_frac > 0:
        budget_p = int((cfg.budget_frac * fd.data_bytes() - side_bytes) // 4)
        m_b = pick_m_for_budget(budget_p, cfg.d_base)
        log(f"  budget_frac={cfg.budget_frac}: {budget_p:,} params allowed -> "
            f"m={m_b} d={cfg.d_base} ({coordnet_num_params(m_b, cfg.d_base):,} params;"
            f" m_base ignored)")
    model, st = train_inr_best_of_seeds(coords_n, vals_n, m_b, cfg.d_base,
                                        cfg.train_cfg(), device, seed_base=cfg.seed,
                                        n_seeds=cfg.n_seeds,
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
def observer_stored_bytes(observer: str, tw: int) -> int:
    """Stored bytes of ONE region's observer parameters. The variant itself is a
    single global tag byte (added once per run); c == 0 / time-constancy are implied
    by the tag, so tvtrans stores 2 floats per timestep and the const variants 2-3
    floats total."""
    return {"tvfull": tw * 3 * 4, "tvtrans": tw * 2 * 4,
            "constfull": 3 * 4, "consttrans": 2 * 4}[observer]


def resolve_observer(reg, observer: str) -> tuple[np.ndarray, float]:
    """Re-solve one region's observer under the requested parameterization from the
    region's own LSQ sufficient statistics (already window-sliced sums). Returns
    (q (Tw,3), E_variant). E_variant >= E_tvfull by construction (fewer DOF); the
    tau-merge partition itself always uses the tv-full criterion."""
    if observer == "tvfull":
        return reg.q, reg.E
    if observer == "tvtrans":
        q, E = solve_killing_trans(reg.AtA, reg.g, reg.e0)
        return q, float(E.sum())
    Tw = reg.AtA.shape[0]
    if observer == "constfull":
        qc, _ = solve_killing(reg.AtA.sum(0), reg.g.sum(0), reg.e0.sum())
    elif observer == "consttrans":
        qc, _ = solve_killing_trans(reg.AtA.sum(0), reg.g.sum(0), reg.e0.sum())
    else:
        raise ValueError(f"unknown observer '{observer}'")
    E = float(reg.e0.sum() + qc @ reg.g.sum(0))
    return np.tile(qc, (Tw, 1)), max(E, 0.0)


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
    if cfg.budget_frac > 0:
        # strict-compression budget: frac * raw field bytes covers params AND
        # side info; deduct the exact side info (known from the partition,
        # mirrors the accumulation in the training loop below) before splitting.
        side_planned = 1 if use_observer else 0     # observer-variant tag byte
        for part in parts:
            tw = part.it1 - part.it0
            if part.n_regions > 1:      # N==1: label map is constant, store nothing
                side_planned += part.labels_cells.size * 2
            side_planned += part.n_regions * (
                (observer_stored_bytes(cfg.observer, tw) if use_observer else 0)
                + (4 + 4) * 4 + 2)
        params_pool = int((cfg.budget_frac * fd.data_bytes() - side_planned) // 4)
        assert params_pool > n_inrs, "budget_frac too small for side info"
        share = params_pool // n_inrs
        log(f"  budget_frac={cfg.budget_frac}: pool {params_pool:,} params after "
            f"{side_planned:,} B side info -> share {share:,}/INR (x{n_inrs})")
    else:
        params_pool = None
        share = int(budget_factor * B / n_inrs)
    d_r = cfg.d_base
    # pixel-proportional allocation (docs par.5 open question 1): weight each
    # (window, region) INR's share by its sample count instead of uniformly.
    # Motivated by cylinder v2.3: uniform gave the 92%-of-pixels region B/5 while
    # 4-pixel stragglers overfit 256 samples with the same share (MSE 1e-7).
    total_wpix = sum(float((p.labels_pixels == r_i).sum()) * (p.it1 - p.it0)
                     for p in parts for r_i in range(p.n_regions))
    # capsmall allocation = the par.4.4h lesson "proportional + capacity floor"
    # made concrete: every region except the largest-by-samples is capped at
    # n_samples params (an INR with >=1 param/sample only interpolates); the
    # largest region takes the entire remainder. Motivation (compresswin_1.4
    # wave 2, cylinder M=2 = [25384, 216] px): uniform starves the wake
    # (m 29->21) to overfeed a 216-px region; "pixels" starves the obstacle
    # to m=2 (the par.4.4h failure).
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
                pick_m_for_budget(min(ns, pool_total // n_inrs), d_r), d_r)
            share_map[(w_i, r_i)] = cap
            spent += cap
        share_map[big] = pool_total - spent
        log(f"  alloc=capsmall: largest region w{big[0]}r{big[1]} gets "
            f"{share_map[big]:,} params; {n_inrs - 1} small region(s) capped "
            f"at <=1 param/sample ({spent:,} total)")

    recon = np.full(fd.shape, np.nan)
    details = []
    t_start = time.time()
    inr_idx = 0
    side_bytes = 1 if use_observer else 0                   # observer-variant tag byte
    for w_i, part in enumerate(parts):
        Tw = part.it1 - part.it0
        if part.n_regions > 1:          # N==1: label map is constant, store nothing
            side_bytes += part.labels_cells.size * 2        # uint16 cell label map
        for r_i, reg in enumerate(part.regions):
            if use_observer:
                q, E_obs = resolve_observer(reg, cfg.observer)
                side_bytes += observer_stored_bytes(cfg.observer, Tw)
            else:
                q, E_obs = np.zeros_like(reg.q), reg.E0
            pix_mask = part.labels_pixels == r_i
            samples = make_region_samples(fd.data, fd.xs, fd.ys, fd.dt,
                                          pix_mask, part.it0, part.it1, q)
            ximm = MinMax.fit(samples.xi)
            vmm = MinMax.fit(samples.vtil)
            side_bytes += (4 + 4) * 4 + 2                   # xi bbox + value lo/hi + m_r
            coords_n = np.concatenate([ximm.encode(samples.xi),
                                       samples.tn[:, None].astype(np.float32)], axis=1)
            vals_n = vmm.encode(samples.vtil)
            if cfg.alloc == "pixels":
                pool = params_pool if params_pool is not None else budget_factor * B
                w_share = int(pool * samples.xi.shape[0] / total_wpix)
            elif cfg.alloc == "capsmall":
                w_share = share_map[(w_i, r_i)]
            else:
                w_share = share
            m_r = pick_m_for_budget(w_share, d_r)
            tag = f"{fd.name}/{mode_name} w{w_i}r{r_i} m={m_r}"
            model, st = train_inr_best_of_seeds(coords_n, vals_n, m_r, d_r,
                                                cfg.train_cfg(), device,
                                                seed_base=cfg.seed + 1000 * w_i + r_i,
                                                n_seeds=cfg.n_seeds, tag=tag, log=log,
                                                model_name=cfg.model,
                                                model_kwargs=cfg.model_kwargs())
            pred_n = eval_inr(model, coords_n, device)
            vtil_pred = vmm.decode(pred_n)
            scatter_reconstruction(recon, samples, vtil_pred)
            st.update({"window": w_i, "region": r_i, "pixels": int(samples.n_pix),
                       "E_over_E0": reg.E / max(reg.E0, 1e-300),
                       "E_obs_over_E0": E_obs / max(reg.E0, 1e-300),
                       "observer": cfg.observer if use_observer else "none",
                       "use_observer": use_observer})
            details.append(st)
            inr_idx += 1

    assert not np.isnan(recon).any(), "reconstruction coverage hole"
    psnr = vpsnr(recon, fd.data)
    params_total = sum(st["params"] for st in details)
    total_bytes = params_total * 4 + side_bytes
    if cfg.budget_frac > 0:
        assert side_bytes == side_planned, "side-info plan drifted from actual"
        assert total_bytes <= cfg.budget_frac * fd.data_bytes(), "over byte budget"
    return {"mode": mode_name, "psnr": psnr, "params_total": params_total,
            "budget_B": B, "budget_factor": budget_factor, "per_inr_share": share,
            "budget_frac": cfg.budget_frac,
            "observer": cfg.observer if use_observer else "none",
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
        f"B={budget_B(cfg)} params (m={cfg.m_base}, d={cfg.d_base}), "
        f"model={cfg.model}, device={device}"
        + (f", budget_frac={cfg.budget_frac} ({cfg.budget_frac * fd.data_bytes():,.0f} B)"
           if cfg.budget_frac > 0 else ""))

    results = {}
    parts = None
    if any(m != "baseline" for m in cfg.modes):
        log(f"[{fd.name}] tau-merge partition:")
        parts = compute_partitions(fd, cfg, log=log)
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
        elif mode == "pro_quality":
            # 4B since 2026-07-15 (user spec change); all runs up to and including
            # Verify_tau_1.1 / gerris pilot used 3B -- do not mix the two labels.
            if cfg.budget_frac > 0:
                raise ValueError("pro_quality is the 4B quality mode; "
                                 "budget_frac only applies to baseline/pro_budget/"
                                 "no_observer (strict-compression protocol)")
            res = run_proposed(fd, cfg, parts, device, 4.0, True, mode, log=log)
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
