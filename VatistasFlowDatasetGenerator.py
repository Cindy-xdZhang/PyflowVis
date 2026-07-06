"""Generate the VortexTransformer training data at the paper's exact scale: batch-synthesize STEADY Vatistas vortex fields, then observe each with many moving/rotating
(Killing) observers to produce UNSTEADY fields that share one objective vortex label.

Synthesized 3000 steady 2D fields:
1500 from real data, 1400 random samples, and 100 manually added Killing fields. Each
steady field was transformed via 20 randomly sampled rigid-body observers over T=[0,1],
yielding 60,000 unsteady fields"):

    STEADY   = 3000  = 1500 fitted (real-data patch fits)
                     + 1400 random (sampled from the fitted distribution)
                     + 100  manually-added Killing fields (rigid-motion hard negatives).
    UNSTEADY = 60000 = 3000 steady  x  20 rigid-body observers.

The steady->unsteady transform is a faithful NumPy port of `CppProjects/src/transformation.cpp::killingABCtransformation` (observer world-line RK4,
Q(t)=R^T, c(t)=Os-Q*p, w* = Q*v + Qdot*F^-1 + (-a,-b)); See docs/vatistas_profile.md.

NB: materializing all 60000 unsteady velocity grids is ~39 GB. By default we render+save
the full 3000 steady set (cheap, ~150 MB) and WRITE only a verification subset of
unsteady variants; pass --full to stream all 60000 to disk in shards. 

Run:
    python VatistasFlowDatasetGenerator.py --config config/VatistasDataset.yaml
    python VatistasFlowDatasetGenerator.py --full     # write all 60000 (sharded, ~39GB)
    python VatistasFlowDatasetGenerator.py --refit     # redo the vastistas parameters fitting first
"""

from __future__ import annotations

import os
import json
import math
import argparse
import logging

import numpy as np
import torch

from DeepUtils.utils import EasyConfig

# Reuse the paper-faithful, best-tuned forward model + fitter (do NOT re-derive it).
import FittingVatistasParam as fvp
from FittingVatistasParam import (
    vatistas_mixture_velocity, make_local_grid, build_bounds,
    PARAM_NAMES, TH, SX, SY, RC, N, TX, TY,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

CENTER_SHAPES = (1, 2)  # S2 (cw) / S3 (ccw); saddles carry no vortex label
SRC_RANDOM, SRC_FITTED, SRC_KILLING = 0, 1, 2


# ===========================================================================
# 0. Fitting + distribution  (load the cached best fit, or refit on request)
# ===========================================================================
def load_fit_outputs(fit_dir):
    dist_path = os.path.join(fit_dir, "vatistas_param_distribution.json")
    npz_path = os.path.join(fit_dir, "fitted_patch_params.npz")
    if not (os.path.isfile(dist_path) and os.path.isfile(npz_path)):
        raise FileNotFoundError(
            f"No cached fit in '{fit_dir}'. Run `python FittingVatistasParam.py` first, "
            f"or pass --refit to fit here.")
    with open(dist_path) as f:
        dist = json.load(f)
    d = np.load(npz_path, allow_pickle=True)
    fitted = {
        "params": d["params"].astype(np.float32),
        "shapes": d["shapes"].astype(np.int64),
        "mask": d["active"].astype(np.float32),
        "fit_mse": d["fit_mse"].astype(np.float32),
    }
    logging.info(f"[fit] loaded cached best fit: {fitted['params'].shape[0]} patches "
                 f"(mean MSE {float(fitted['fit_mse'].mean()):.4f}); distribution over "
                 f"{dist['num_profiles']} profiles.")
    return dist, fitted


def refit(cfg_fit_path, device):
    cfg = EasyConfig()
    cfg.load(cfg_fit_path, recursive=False)
    if cfg.dataset.get("dat_dir") is None:
        cfg.dataset["dat_dir"] = None
    fvp.relocate_flow2d_dataset_folder(cfg)
    bounds = build_bounds(cfg)
    M = int(cfg.model.num_profiles)
    target, _scales, meta = fvp.extract_patches(cfg, device)
    coords = make_local_grid(int(cfg.dataset.patch_size), device)
    params, shape_idx, fit_loss = fvp.fit_patches(coords, target, cfg, bounds, M, device)
    fvp.fit_quality_report(coords, target, params, shape_idx, bounds, tag="refit")
    contrib = fvp.profile_contributions(coords, params, shape_idx, bounds)
    active = contrib >= float(cfg.generate.min_contribution)
    dist = fvp.fit_distribution(params, shape_idx, active)
    fit_dir = cfg.output.get("out_dir") or os.path.join(cfg.get("cache_dir", "./outputs/"), "vatistas_fit")
    gen = fvp.generate_steady_fields(dist, cfg, bounds, device)
    fvp.save_outputs(fit_dir, params, shape_idx, fit_loss, contrib, active, meta, dist, gen)
    fitted = {"params": params.cpu().numpy().astype(np.float32),
              "shapes": shape_idx.cpu().numpy().astype(np.int64),
              "mask": active.cpu().numpy().astype(np.float32),
              "fit_mse": fit_loss.cpu().numpy().astype(np.float32)}
    return dist, fitted


# ===========================================================================
# 1. Steady field pool:  random (distribution) + fitted (real patches) + Killing
# ===========================================================================
def _subsample_fitted(fitted, cfg):
    f_params, f_shapes, f_mask = fitted["params"], fitted["shapes"], fitted["mask"]
    fcfg = cfg.get("fitted", {}) or {}
    want = fcfg.get("num_fields", None)
    want = None if want in (None, "null") else int(want)
    if want is not None and 0 < want < f_params.shape[0]:
        if bool(fcfg.get("prefer_low_mse", True)):
            order = np.argsort(fitted["fit_mse"])[:want]
        else:
            order = np.random.default_rng(int(cfg.get("seed", 0)) + 5).permutation(f_params.shape[0])[:want]
        order = np.sort(order)
        f_params, f_shapes, f_mask = f_params[order], f_shapes[order], f_mask[order]
    return f_params, f_shapes, f_mask


def make_killing_steady(cfg):
    """Sample the 100 manually-added Killing (rigid-motion) steady fields.
    v(x) = (a, b) + c*(-y, x). These are observer-producible -> NOT objective vortices,
    so they act as hard negatives with an EMPTY segmentation label."""
    kcfg = cfg.get("killing_fields", {}) or {}
    n = int(kcfg.get("num", 100))
    ar = kcfg.get("abc_range", [0.8, 0.8, 1.0])
    rng = np.random.default_rng(int(cfg.get("seed", 0)) + 11)
    abc = np.stack([rng.uniform(-ar[0], ar[0], n),
                    rng.uniform(-ar[1], ar[1], n),
                    rng.uniform(-ar[2], ar[2], n)], axis=1).astype(np.float32)  # [n,3]
    return abc


def build_steady_specs(dist, fitted, cfg, bounds, device):
    """Assemble the 3000-field steady pool as analytic specs. Returns a list of dicts;
    each is either a Vatistas mixture ({type:'vatistas', params[M,7], shapes[M], mask[M]})
    or a Killing field ({type:'killing', abc:[a,b,c]}), all tagged with 'source'."""
    specs = []

    # (a) random-from-distribution
    gen = fvp.generate_steady_fields(dist, cfg, bounds, device)
    for i in range(gen["params"].shape[0]):
        specs.append({"type": "vatistas", "params": gen["params"][i], "shapes": gen["shapes"][i],
                      "mask": gen["mask"][i], "source": SRC_RANDOM})

    # (b) fitted real-flow patches (rendered from their own fitted parameters)
    f_params, f_shapes, f_mask = _subsample_fitted(fitted, cfg)
    for i in range(f_params.shape[0]):
        specs.append({"type": "vatistas", "params": f_params[i], "shapes": f_shapes[i],
                      "mask": f_mask[i], "source": SRC_FITTED})

    # (c) manually-added Killing fields (hard negatives, empty label)
    kabc = make_killing_steady(cfg)
    for i in range(kabc.shape[0]):
        specs.append({"type": "killing", "abc": kabc[i], "source": SRC_KILLING})

    counts = {SRC_RANDOM: gen["params"].shape[0], SRC_FITTED: f_params.shape[0], SRC_KILLING: kabc.shape[0]}
    logging.info(f"[steady] pool = {len(specs)} fields  "
                 f"(random {counts[SRC_RANDOM]} + fitted {counts[SRC_FITTED]} + killing {counts[SRC_KILLING]}).")
    return specs, counts


# ---- per-field steady evaluator + label (analytic, works at arbitrary points) ----
def make_steady_eval(spec, bounds, device):
    """Return (vel_eval, label_eval): coords[H,W,2] -> velocity[H,W,2] / signed-distance[H,W]."""
    if spec["type"] == "killing":
        a, b, c = [float(x) for x in spec["abc"]]

        def vel(coords_np):
            x, y = coords_np[..., 0], coords_np[..., 1]
            return np.stack([a - c * y, b + c * x], axis=-1).astype(np.float32)

        def lab(coords_np):                      # rigid motion -> no objective vortex core
            return np.full(coords_np.shape[:2], -1.0, np.float32)
        return vel, lab

    params, shapes, mask = spec["params"], spec["shapes"], spec["mask"]
    pt = torch.from_numpy(params[None]).to(device)
    st = torch.from_numpy(shapes[None]).to(device)
    mt = torch.from_numpy(mask[None]).to(device)

    def vel(coords_np):
        c = torch.from_numpy(np.ascontiguousarray(coords_np, np.float32)).to(device)
        with torch.no_grad():
            v = vatistas_mixture_velocity(c, pt, st, bounds, profile_mask=mt)[0]
        return v.cpu().numpy().astype(np.float32)

    def lab(coords_np):
        return vatistas_signed_distance(coords_np.astype(np.float32), params, shapes, mask, bounds)
    return vel, lab


def vatistas_signed_distance(coords_np, params, shapes, mask, bounds):
    """[H,W,2] -> [H,W]: max over active CENTER profiles of (rc - ||D^-1 (x - T)||)
    (Eq. 5/8). Mirrors the forward model's pull-back exactly."""
    H, W, _ = coords_np.shape
    out = np.full((H, W), -1e9, np.float32)
    rc_lo, rc_hi = bounds["rc"]; s_lo, s_hi = bounds["s"]
    for m in range(params.shape[0]):
        if mask[m] <= 0.0 or int(shapes[m]) not in CENTER_SHAPES:
            continue
        th, sx, sy = params[m, TH], params[m, SX], params[m, SY]
        rc = float(np.clip(params[m, RC], rc_lo, rc_hi))
        sx = float(np.clip(sx, s_lo, s_hi)); sy = float(np.clip(sy, s_lo, s_hi))
        tx, ty = params[m, TX], params[m, TY]
        cos, sin = np.cos(th), np.sin(th); inv_det = 1.0 / (sx * sy)
        dx = coords_np[..., 0] - tx; dy = coords_np[..., 1] - ty
        bx = inv_det * (sy * cos) * dx + inv_det * (sy * sin) * dy
        by = inv_det * (-sx * sin) * dx + inv_det * (sx * cos) * dy
        out = np.maximum(out, (rc - np.sqrt(bx * bx + by * by)).astype(np.float32))
    return out


# ===========================================================================
# 2. Killing-ABC transformation (steady -> unsteady) -- port of transformation.cpp
# ===========================================================================
def sample_observer(rng, ocfg):
    """One rigid-body observer as the paper's 6 scalars (a,b,c,adot,bdot,cdot):
    a(t)=a0+adot*t, b(t)=b0+bdot*t, c(t)=c0+cdot*t. Returns (abc_func, six_scalars)."""
    a0 = rng.uniform(-ocfg["a0_range"], ocfg["a0_range"]); ad = rng.uniform(-ocfg["adot_range"], ocfg["adot_range"])
    b0 = rng.uniform(-ocfg["a0_range"], ocfg["a0_range"]); bd = rng.uniform(-ocfg["adot_range"], ocfg["adot_range"])
    c0 = rng.uniform(-ocfg["c0_range"], ocfg["c0_range"]); cd = rng.uniform(-ocfg["cdot_range"], ocfg["cdot_range"])
    six = np.array([a0, b0, c0, ad, bd, cd], np.float32)

    def abc_func(t):
        return np.array([a0 + ad * t, b0 + bd * t, c0 + cd * t], np.float64)
    return abc_func, six


def _killing_velocity(abc_func, center, pos, t):
    a, b, c = abc_func(t)
    rx, ry = pos[0] - center[0], pos[1] - center[1]
    return np.array([a - c * ry, b + c * rx], np.float64)


def integrate_observer_pathline(abc_func, start_pos, center, tmin, tmax, timesteps):
    """RK4-integrate the observer seed through its Killing field -> positions[T,2]."""
    dt = (tmax - tmin) / (timesteps - 1)
    pos = np.array(start_pos, np.float64)
    out = np.zeros((timesteps, 2), np.float64); out[0] = pos
    for i in range(1, timesteps):
        t = tmin + (i - 1) * dt
        k1 = _killing_velocity(abc_func, center, pos, t)
        k2 = _killing_velocity(abc_func, center, pos + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = _killing_velocity(abc_func, center, pos + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = _killing_velocity(abc_func, center, pos + dt * k3, t + dt)
        pos = pos + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        out[i] = pos
    return out


def killing_abc_transform(vel_eval, abc_func, ocfg, coords_np):
    """Observe a steady field (vel_eval: coords->vel) with one Killing observer -> [T,G,G,2].
    Faithful port of transformation.cpp::killingABCtransformation (analytic branch)."""
    T = int(ocfg["timesteps"]); G = coords_np.shape[0]
    tmin, tmax = float(ocfg["tmin"]), float(ocfg["tmax"])
    center = np.asarray(ocfg["center"], np.float64)
    dt = (tmax - tmin) / (T - 1)

    path = integrate_observer_pathline(abc_func, ocfg["start_pos"], center, tmin, tmax, T)
    Os = path[0]

    finv = np.zeros((T, G, G, 2), np.float64)
    Q_all = np.zeros((T, 2, 2)); Qdot_all = np.zeros((T, 2, 2)); Tdot_all = np.zeros((T, 2))
    theta = 0.0
    for i in range(T):
        ti = tmin + i * dt
        if i > 0:
            theta += dt * float(abc_func(ti)[2])            # cumulative rotation angle
        ct, stt = math.cos(theta), math.sin(theta)
        R_i = np.array([[ct, -stt], [stt, ct]])              # observer rotation Rot(theta)
        Q_i = R_i.T                                          # frame rotation Q = R^T
        c_t = Os - Q_i @ path[i]                             # frame translation
        shifted = coords_np - c_t.reshape(1, 1, 2)
        finv[i, ..., 0] = R_i[0, 0] * shifted[..., 0] + R_i[0, 1] * shifted[..., 1]
        finv[i, ..., 1] = R_i[1, 0] * shifted[..., 0] + R_i[1, 1] * shifted[..., 1]
        a_now, b_now, c_now = abc_func(ti)
        Q_all[i] = Q_i
        Qdot_all[i] = Q_i @ np.array([[0.0, c_now], [-c_now, 0.0]])
        Tdot_all[i] = np.array([-a_now, -b_now])

    v = vel_eval(finv.reshape(T * G, G, 2)).astype(np.float64).reshape(T, G, G, 2)
    field = np.zeros((T, G, G, 2), np.float32)
    for i in range(T):
        Q, Qd, Td, vi, fi = Q_all[i], Qdot_all[i], Tdot_all[i], v[i], finv[i]
        field[i, ..., 0] = Q[0, 0] * vi[..., 0] + Q[0, 1] * vi[..., 1] + Qd[0, 0] * fi[..., 0] + Qd[0, 1] * fi[..., 1] + Td[0]
        field[i, ..., 1] = Q[1, 0] * vi[..., 0] + Q[1, 1] * vi[..., 1] + Qd[1, 0] * fi[..., 0] + Qd[1, 1] * fi[..., 1] + Td[1]
    return field


# ===========================================================================
# Orchestration
# ===========================================================================
def read_observer_cfg(cfg):
    o = cfg.observer
    return {"per_field": int(o.per_field), "timesteps": int(o.timesteps),
            "grid": int(cfg.generate.grid), "tmin": float(o.tmin), "tmax": float(o.tmax),
            "center": list(o.center), "start_pos": list(o.start_pos),
            "a0_range": float(o.a0_range), "adot_range": float(o.adot_range),
            "c0_range": float(o.c0_range), "cdot_range": float(o.cdot_range),
            "make_labels": bool(o.make_labels)}


def render_steady_set(specs, ocfg, bounds, device):
    """Render all steady fields + their shared (t0) segmentation label, for saving."""
    G = ocfg["grid"]
    grid = make_local_grid(G, device).cpu().numpy().astype(np.float32)
    N = len(specs)
    fields = np.zeros((N, G, G, 2), np.float32)
    labels = np.zeros((N, G, G), np.float32)
    source = np.zeros(N, np.int64)
    evals = []
    for i, spec in enumerate(specs):
        vel, lab = make_steady_eval(spec, bounds, device)
        fields[i] = vel(grid)
        labels[i] = lab(grid)
        source[i] = spec["source"]
        evals.append(vel)
    return grid, fields, labels, source, evals


def generate_unsteady(specs, evals, grid, ocfg, cfg, out_dir, write_full):
    """For every steady field, sample `per_field` observers and transform. Streams the
    written subset/full set to sharded .npz files. Returns (target_total, written, abc_all)."""
    N = len(specs); per = ocfg["per_field"]
    target_total = N * per
    rng = np.random.default_rng(int(cfg.get("seed", 0)) + 17)

    ucfg = cfg.output
    subset = int(ucfg.get("unsteady_subset", 400))
    shard_size = int(ucfg.get("shard_size", 2000))
    to_write = target_total if write_full else min(subset, target_total)

    # visit steady fields in an interleaved order so a subset still spans all 3 sources.
    order = rng.permutation(N)
    T, G = ocfg["timesteps"], ocfg["grid"]

    shard_fields, shard_sidx, shard_obs = [], [], []
    written = 0
    shard_id = 0
    abc_all = np.zeros((to_write, 6), np.float32)

    def flush():
        nonlocal shard_id, shard_fields, shard_sidx, shard_obs
        if not shard_fields:
            return
        np.savez_compressed(
            os.path.join(out_dir, f"unsteady_shard_{shard_id:04d}.npz"),
            fields=np.stack(shard_fields).astype(np.float32),
            steady_index=np.asarray(shard_sidx, np.int64),
            observer=np.stack(shard_obs).astype(np.float32))
        logging.info(f"[unsteady] wrote shard {shard_id} ({len(shard_fields)} fields).")
        shard_id += 1
        shard_fields, shard_sidx, shard_obs = [], [], []

    done = False
    for si in order:
        vel = evals[si]
        for _ in range(per):
            abc_func, six = sample_observer(rng, ocfg)
            field = killing_abc_transform(vel, abc_func, ocfg, grid)
            shard_fields.append(field); shard_sidx.append(int(si)); shard_obs.append(six)
            abc_all[written] = six
            written += 1
            if len(shard_fields) >= shard_size:
                flush()
            if written >= to_write:
                done = True
                break
        if (written % max(1, to_write // 10 or 1) == 0) or done:
            logging.info(f"[unsteady] transformed {written}/{to_write} "
                         f"(target total = {target_total})")
        if done:
            break
    flush()
    return target_total, written, abc_all[:written]


def save_steady(out_dir, specs, grid, fields, labels, source, counts):
    os.makedirs(out_dir, exist_ok=True)
    # pack vatistas params (killing rows zeroed) + killing abc for reproducibility.
    N = len(specs)
    M = max((s["params"].shape[0] for s in specs if s["type"] == "vatistas"), default=2)
    params = np.zeros((N, M, 7), np.float32)
    shapes = np.zeros((N, M), np.int64)
    mask = np.zeros((N, M), np.float32)
    killing_abc = np.zeros((N, 3), np.float32)
    for i, s in enumerate(specs):
        if s["type"] == "vatistas":
            m = s["params"].shape[0]
            params[i, :m] = s["params"]; shapes[i, :m] = s["shapes"]; mask[i, :m] = s["mask"]
        else:
            killing_abc[i] = s["abc"]
    np.savez_compressed(
        os.path.join(out_dir, "steady_fields.npz"),
        params=params, shapes=shapes, mask=mask, killing_abc=killing_abc,
        source=source, fields=fields, labels=labels, coords=grid,
        counts=np.array([counts[SRC_RANDOM], counts[SRC_FITTED], counts[SRC_KILLING]], np.int64))
    logging.info(f"[save] wrote {N} steady fields (+shared labels) to {out_dir}/steady_fields.npz")


def save_plots(out_dir, specs, grid, steady_fields, steady_labels, source, ocfg, evals, cfg):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as ex:
        logging.warning(f"[plots] matplotlib unavailable ({ex}); skipping."); return
    G = steady_fields.shape[1]
    lin = np.linspace(-1, 1, G); X, Y = np.meshgrid(lin, lin); step = max(1, G // 20)
    rng = np.random.default_rng(int(cfg.get("seed", 0)) + 123)

    def quiver(ax, vel, title, label=None):
        sp = np.sqrt(vel[..., 0] ** 2 + vel[..., 1] ** 2)
        ax.imshow(sp, extent=[-1, 1, -1, 1], origin="lower", cmap="viridis", alpha=0.75)
        if label is not None and label.max() > 0:
            ax.contour(X, Y, label, levels=[0.0], colors="red", linewidths=1.5)
        ax.quiver(X[::step, ::step], Y[::step, ::step], vel[::step, ::step, 0], vel[::step, ::step, 1],
                  color="white", scale=None)
        ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])

    picks = []
    for src, name in [(SRC_RANDOM, "random"), (SRC_FITTED, "fitted"), (SRC_KILLING, "killing")]:
        ids = np.where(source == src)[0]
        if len(ids):
            picks.append((int(ids[0]), name))
    T = ocfg["timesteps"]; frames = [0, T // 2, T - 1]
    for which, name in picks:
        obs_func, _ = sample_observer(rng, ocfg)
        uf = killing_abc_transform(evals[which], obs_func, ocfg, grid)
        fig, axes = plt.subplots(1, 1 + len(frames), figsize=(4 * (1 + len(frames)), 4))
        quiver(axes[0], steady_fields[which], f"steady ({name}) #{which}", steady_labels[which])
        for a, t in zip(axes[1:], frames):
            quiver(a, uf[t], f"observed t={t}/{T-1}")
        fig.suptitle("Steady Vatistas field -> Killing-observed unsteady frames "
                     "(red = shared objective vortex label at t0)")
        fig.tight_layout()
        p = os.path.join(out_dir, f"preview_{name}_{which}.png")
        fig.savefig(p, dpi=120); plt.close(fig); logging.info(f"[plots] saved {p}")


def main():
    ap = argparse.ArgumentParser(description="Generate the paper-scale Vatistas steady + Killing-observed unsteady dataset.")
    ap.add_argument("--config", type=str, default="config/VatistasDataset.yaml")
    ap.add_argument("--refit", action="store_true", help="re-run the full parameter fit instead of loading the cache")
    ap.add_argument("--full", action="store_true", help="materialize ALL unsteady variants (paper 60000, sharded, ~39GB)")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--no_plots", action="store_true")
    args = ap.parse_args()

    cfg = EasyConfig(); cfg.load(args.config, recursive=False)
    seed = int(cfg.get("seed", 0)); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    bounds = build_bounds(cfg)
    logging.info(f"[main] device={device}, seed={seed}")

    # 0. fit + distribution
    if args.refit:
        dist, fitted = refit(cfg.get("fit_config", "config/FittingVatistas.yaml"), device)
    else:
        dist, fitted = load_fit_outputs(cfg.get("fit_dir", "outputs/vatistas_fit"))

    # 1. steady pool (random + fitted + killing = 3000)
    specs, counts = build_steady_specs(dist, fitted, cfg, bounds, device)
    ocfg = read_observer_cfg(cfg)
    grid, steady_fields, steady_labels, source, evals = render_steady_set(specs, ocfg, bounds, device)

    out_dir = args.out_dir or cfg.output.get("out_dir", "outputs/vatistas_unsteady")
    os.makedirs(out_dir, exist_ok=True)
    if bool(cfg.output.get("save_steady", True)):
        save_steady(out_dir, specs, grid, steady_fields, steady_labels, source, counts)

    # 2. observe each steady field with `per_field` observers -> unsteady (60000 target)
    target_total, written, abc_all = generate_unsteady(specs, evals, grid, ocfg, cfg, out_dir, args.full)

    np.savez_compressed(os.path.join(out_dir, "dataset_manifest.npz"),
                        target_steady=np.int64(len(specs)), target_unsteady=np.int64(target_total),
                        written_unsteady=np.int64(written), observers_per_field=np.int64(ocfg["per_field"]),
                        source_counts=np.array([counts[SRC_RANDOM], counts[SRC_FITTED], counts[SRC_KILLING]], np.int64),
                        written_observers=abc_all)

    logging.info("=" * 78)
    logging.info(f"[totals] PAPER TARGET  steady = {len(specs)}  "
                 f"(random {counts[SRC_RANDOM]} + fitted {counts[SRC_FITTED]} + killing {counts[SRC_KILLING]})")
    logging.info(f"[totals] PAPER TARGET  unsteady = {target_total}  "
                 f"(= {len(specs)} steady x {ocfg['per_field']} observers)")
    logging.info(f"[totals] MATERIALIZED  steady = {len(specs)}  unsteady = {written}"
                 f"{'  (FULL)' if args.full else '  (subset; pass --full for all 60000, ~39GB)'}")
    logging.info("=" * 78)

    if bool(cfg.output.get("save_plots", True)) and not args.no_plots:
        save_plots(out_dir, specs, grid, steady_fields, steady_labels, source, ocfg, evals, cfg)
    logging.info("[main] done.")


if __name__ == "__main__":
    main()
