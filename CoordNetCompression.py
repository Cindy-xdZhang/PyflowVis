"""CoordNet INR compression of a time-varying volume (Han & Wang, IEEE TVCG 2023).

Overfits an implicit neural representation  f(x,y,z,t) -> value  to ALL voxels of a
time-varying volume; the trained network weights ARE the compressed data. We report
PSNR + compression ratio + #params. See docs/coordnet.md for the method.

This is written as a *baseline-comparison harness*, factored into three pluggable
registries so the paper's method is one interchangeable choice and future variants slot in
without touching the training loop:

  * MODEL_REGISTRY      -- 'coordnet' (this paper) and 'plain_siren' baseline;
  * TRANSFORM_REGISTRY  -- 'identity' (paper) and a Fourier positional-encoding stub
                           (this is the seam for a DIFFERENT COORDINATE SYSTEM);
  * PARTITION_REGISTRY  -- 'none' (one global INR, paper) and 'blocks' (per-block INRs)
                           (this is the seam for a DIFFERENT SPATIAL SUBDIVISION).

Data defaults to a self-contained synthetic time-varying volume, so it runs with no
external data; real volumes load from .npy/.npz or raw float32.

Run:
    python CoordNetCompression.py --config config/CoordNetCompression.yaml
"""

from __future__ import annotations

import os
import time
import json
import math
import argparse
import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from DeepUtils.utils import EasyConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

COORD_DIM = 4  # (x, y, z, t) for a time-varying scalar volume


# ===========================================================================
# SIREN building blocks  (Sitzmann et al. 2020; residual variant = CoordNet)
# ===========================================================================
class SineLayer(nn.Module):
    """Linear layer + sin(omega_0 * (Wx+b)), with the SIREN weight initialization."""
    def __init__(self, in_features, out_features, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = float(omega_0)
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1.0 / in_features, 1.0 / in_features)
            else:
                b = math.sqrt(6.0 / in_features) / self.omega_0
                self.linear.weight.uniform_(-b, b)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class SirenResBlock(nn.Module):
    """CoordNet residual block: two SIREN layers on the main path, an identity (or SIREN
    projection when in!=out) skip, and output = (main + skip) / 2 to stay in [-1,1]."""
    def __init__(self, in_features, out_features, omega_0=30.0, is_first=False):
        super().__init__()
        self.l1 = SineLayer(in_features, out_features, is_first=is_first, omega_0=omega_0)
        self.l2 = SineLayer(out_features, out_features, is_first=False, omega_0=omega_0)
        self.proj = None
        if in_features != out_features:
            # project the skip to out_features (also a SIREN so its range is [-1,1]).
            self.proj = SineLayer(in_features, out_features, is_first=is_first, omega_0=omega_0)

    def forward(self, x):
        skip = x if self.proj is None else self.proj(x)
        out = self.l2(self.l1(x))
        return 0.5 * (out + skip)


# ===========================================================================
# Models (registry)
# ===========================================================================
MODEL_REGISTRY = {}


def register_model(name):
    def deco(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return deco


@register_model("coordnet")
class CoordNet(nn.Module):
    """Paper architecture: encoder [k->m, m->2m, 2m->4m, d*(4m->4m)] + decoder [4m->p].

    `final_activation`: 'sine' (paper, decoder is a residual block ending in sin -> [-1,1]);
    'linear' or 'tanh' replace the decoder with a plain head (handy for baselines/variants).
    """
    def __init__(self, in_dim, out_dim, m=64, d=10, omega_0=30.0, final_activation="sine", **_):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.encoder.append(SirenResBlock(in_dim, m, omega_0, is_first=True))
        self.encoder.append(SirenResBlock(m, 2 * m, omega_0))
        self.encoder.append(SirenResBlock(2 * m, 4 * m, omega_0))
        for _i in range(int(d)):
            self.encoder.append(SirenResBlock(4 * m, 4 * m, omega_0))
        self.final_activation = final_activation
        if final_activation == "sine":
            self.decoder = SirenResBlock(4 * m, out_dim, omega_0)
        else:
            self.decoder = nn.Linear(4 * m, out_dim)

    def forward(self, x):
        for blk in self.encoder:
            x = blk(x)
        x = self.decoder(x)
        if self.final_activation == "tanh":
            x = torch.tanh(x)
        return x


@register_model("plain_siren")
class PlainSIREN(nn.Module):
    """A vanilla (non-residual) SIREN MLP, as a simple INR baseline to compare against."""
    def __init__(self, in_dim, out_dim, hidden_features=256, hidden_layers=5,
                 omega_0=30.0, final_activation="linear", **_):
        super().__init__()
        layers = [SineLayer(in_dim, hidden_features, is_first=True, omega_0=omega_0)]
        for _i in range(hidden_layers):
            layers.append(SineLayer(hidden_features, hidden_features, omega_0=omega_0))
        self.net = nn.ModuleList(layers)
        self.final_activation = final_activation
        if final_activation == "sine":
            self.head = SineLayer(hidden_features, out_dim, omega_0=omega_0)
        else:
            self.head = nn.Linear(hidden_features, out_dim)

    def forward(self, x):
        for l in self.net:
            x = l(x)
        x = self.head(x)
        if self.final_activation == "tanh":
            x = torch.tanh(x)
        return x


def build_model(model_cfg, in_dim, out_dim):
    name = str(model_cfg.get("name", "coordnet"))
    if name not in MODEL_REGISTRY:
        raise ValueError(f"unknown model '{name}'. Registered: {list(MODEL_REGISTRY)}")
    kwargs = {k: v for k, v in dict(model_cfg).items() if k != "name"}
    return MODEL_REGISTRY[name](in_dim, out_dim, **kwargs)


# ===========================================================================
# Coordinate transforms (registry)  -- seam for a DIFFERENT COORDINATE SYSTEM
# ===========================================================================
TRANSFORM_REGISTRY = {}


def register_transform(name):
    def deco(cls):
        TRANSFORM_REGISTRY[name] = cls
        return cls
    return deco


class CoordinateTransform:
    """Maps normalized coords [N, COORD_DIM] in [-1,1] -> features [N, out_dim]."""
    def out_dim(self, in_dim: int) -> int:
        raise NotImplementedError

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


@register_transform("identity")
class IdentityTransform(CoordinateTransform):
    def __init__(self, **_):
        pass

    def out_dim(self, in_dim):
        return in_dim

    def __call__(self, coords):
        return coords


@register_transform("fourier")
class FourierFeatures(CoordinateTransform):
    """Positional encoding: [p, sin(f_i p), cos(f_i p)] with f_i = pi * scale^(i/L).
    A stub to show how a different coordinate parameterization plugs in."""
    def __init__(self, fourier_num_freqs=6, fourier_scale=6.0, **_):
        self.L = int(fourier_num_freqs)
        self.scale = float(fourier_scale)

    def out_dim(self, in_dim):
        return in_dim * (1 + 2 * self.L)

    def __call__(self, coords):
        feats = [coords]
        for i in range(self.L):
            f = math.pi * (self.scale ** (i / max(1, self.L)))
            feats.append(torch.sin(f * coords))
            feats.append(torch.cos(f * coords))
        return torch.cat(feats, dim=-1)


def build_transform(cfg):
    name = str(cfg.get("name", "identity"))
    if name not in TRANSFORM_REGISTRY:
        raise ValueError(f"unknown coord_transform '{name}'. Registered: {list(TRANSFORM_REGISTRY)}")
    return TRANSFORM_REGISTRY[name](**{k: v for k, v in dict(cfg).items() if k != "name"})


# ===========================================================================
# Spatial partitioning (registry)  -- seam for a DIFFERENT SPATIAL SUBDIVISION
# ===========================================================================
PARTITION_REGISTRY = {}


def register_partition(name):
    def deco(cls):
        PARTITION_REGISTRY[name] = cls
        return cls
    return deco


class VolumePartition:
    """Splits a (T,Z,Y,X) volume into spatial regions; each region is trained by its own
    INR. Time is never split. A region is a voxel-index bbox (z0,z1,y0,y1,x0,x1)."""
    def regions(self, T, Z, Y, X) -> List[Tuple[int, int, int, int, int, int]]:
        raise NotImplementedError


@register_partition("none")
class NoPartition(VolumePartition):
    def __init__(self, **_):
        pass

    def regions(self, T, Z, Y, X):
        return [(0, Z, 0, Y, 0, X)]


@register_partition("blocks")
class BlockPartition(VolumePartition):
    """Tile x/y/z into `blocks=[bx,by,bz]` non-overlapping bricks -> bx*by*bz per-block INRs.
    Each block-INR sees coordinates renormalized to [-1,1] within its own brick."""
    def __init__(self, blocks=(1, 1, 1), **_):
        self.bx, self.by, self.bz = [max(1, int(b)) for b in blocks]

    @staticmethod
    def _splits(n, k):
        edges = [round(i * n / k) for i in range(k + 1)]
        return [(edges[i], edges[i + 1]) for i in range(k)]

    def regions(self, T, Z, Y, X):
        out = []
        for z0, z1 in self._splits(Z, self.bz):
            for y0, y1 in self._splits(Y, self.by):
                for x0, x1 in self._splits(X, self.bx):
                    out.append((z0, z1, y0, y1, x0, x1))
        return out


def build_partition(cfg):
    name = str(cfg.get("name", "none"))
    if name not in PARTITION_REGISTRY:
        raise ValueError(f"unknown partition '{name}'. Registered: {list(PARTITION_REGISTRY)}")
    return PARTITION_REGISTRY[name](**{k: v for k, v in dict(cfg).items() if k != "name"})


# ===========================================================================
# Data: time-varying volume
# ===========================================================================
class TimeVaryingVolume:
    """Holds a (T, Z, Y, X) float32 array + value normalization to [-1,1]."""
    def __init__(self, data: np.ndarray, value_norm="minmax"):
        assert data.ndim == 4, f"expected (T,Z,Y,X), got {data.shape}"
        self.data = np.ascontiguousarray(data, dtype=np.float32)
        self.T, self.Z, self.Y, self.X = self.data.shape
        self.vmin = float(self.data.min())
        self.vmax = float(self.data.max())
        self.value_norm = value_norm
        self._scale = max(self.vmax - self.vmin, 1e-8)

    def normalize(self, v):   # data value -> [-1,1]
        return (2.0 * (v - self.vmin) / self._scale - 1.0).astype(np.float32)

    def denormalize(self, vn):  # [-1,1] -> data value
        return (0.5 * (vn + 1.0) * self._scale + self.vmin).astype(np.float32)

    def num_voxels(self):
        return self.T * self.Z * self.Y * self.X


def _synthetic_volume(kind, dim, T):
    """A smooth self-contained time-varying volume so the pipeline runs with no data."""
    X, Y, Z = [int(d) for d in dim]
    xs = np.linspace(-1, 1, X, dtype=np.float32)
    ys = np.linspace(-1, 1, Y, dtype=np.float32)
    zs = np.linspace(-1, 1, Z, dtype=np.float32)
    gz, gy, gx = np.meshgrid(zs, ys, xs, indexing="ij")  # (Z,Y,X)
    vol = np.zeros((T, Z, Y, X), np.float32)
    rng = np.random.default_rng(0)
    if kind == "vortex":
        for ti in range(T):
            t = ti / max(1, T - 1)
            cx, cy = 0.5 * math.cos(2 * math.pi * t), 0.5 * math.sin(2 * math.pi * t)
            r2 = (gx - cx) ** 2 + (gy - cy) ** 2
            vol[ti] = np.exp(-r2 / 0.1) * np.cos(6 * gz + 4 * math.pi * t)
    else:  # blobs
        K = 4
        phase = rng.uniform(0, 2 * math.pi, K)
        rad = rng.uniform(0.3, 0.7, K)
        amp = rng.uniform(0.6, 1.0, K)
        sig = rng.uniform(0.12, 0.22, K)
        for ti in range(T):
            t = ti / max(1, T - 1)
            for k in range(K):
                cx = rad[k] * math.cos(2 * math.pi * t + phase[k])
                cy = rad[k] * math.sin(2 * math.pi * t + phase[k])
                cz = 0.5 * math.sin(2 * math.pi * t + 0.5 * phase[k])
                r2 = (gx - cx) ** 2 + (gy - cy) ** 2 + (gz - cz) ** 2
                vol[ti] += amp[k] * np.exp(-r2 / (2 * sig[k] ** 2))
    return vol


def build_volume(cfg) -> TimeVaryingVolume:
    dcfg = cfg.data
    src = str(dcfg.get("source", "synthetic"))
    if src == "synthetic":
        data = _synthetic_volume(str(dcfg.get("synthetic_kind", "blobs")),
                                 list(dcfg.dim), int(dcfg.time_steps))
    elif src == "npy":
        arr = np.load(dcfg.path)
        data = arr["data"] if isinstance(arr, np.lib.npyio.NpzFile) else arr
        data = np.asarray(data, np.float32)
    elif src == "raw":
        X, Y, Z = [int(d) for d in dcfg.dim]
        T = int(dcfg.time_steps)
        data = np.fromfile(dcfg.path, dtype=np.dtype(str(dcfg.get("raw_dtype", "float32")))
                           ).astype(np.float32).reshape(T, Z, Y, X)
    else:
        raise ValueError(f"unknown data.source '{src}'")
    vol = TimeVaryingVolume(data, value_norm=str(dcfg.get("value_norm", "minmax")))
    logging.info(f"[data] volume (T,Z,Y,X)={vol.data.shape}  value range [{vol.vmin:.4g},{vol.vmax:.4g}]  "
                 f"{vol.num_voxels()/1e6:.2f}M voxels ({src}).")
    return vol


# ---- coordinate/value tensors for a spatial region (all time steps) ----
def region_coords_values(vol: TimeVaryingVolume, region, device):
    """Return normalized coords [Nvox, 4]=(x,y,z,t) in [-1,1] and target values [Nvox,1]
    in [-1,1], for one spatial bbox across all time. Space is normalized WITHIN the region
    bbox (so per-block INRs each see [-1,1]); time is normalized over the whole sequence."""
    z0, z1, y0, y1, x0, x1 = region
    sub = vol.data[:, z0:z1, y0:y1, x0:x1]        # (T,dz,dy,dx)
    T, dz, dy, dx = sub.shape

    def axis(n):
        return np.linspace(-1.0, 1.0, n, dtype=np.float32) if n > 1 else np.zeros(1, np.float32)
    xs, ys, zs, ts = axis(dx), axis(dy), axis(dz), axis(T)
    gt, gz, gy, gx = np.meshgrid(ts, zs, ys, xs, indexing="ij")  # (T,dz,dy,dx)
    coords = np.stack([gx.ravel(), gy.ravel(), gz.ravel(), gt.ravel()], axis=-1)  # (N,4)
    values = vol.normalize(sub).reshape(-1, 1)
    return (torch.from_numpy(coords).to(device),
            torch.from_numpy(values).to(device),
            (T, dz, dy, dx))


# ===========================================================================
# Metrics / sizes
# ===========================================================================
def num_params(model):
    return int(sum(p.numel() for p in model.parameters()))


def psnr(pred, gt, data_range):
    mse = float(np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2))
    if mse <= 0:
        return float("inf")
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


# ===========================================================================
# Compressor: fit an INR (per region) and reconstruct
# ===========================================================================
class INRVolumeCompressor:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.transform = build_transform(cfg.coord_transform)
        self.partition = build_partition(cfg.partition)
        self.models: List[nn.Module] = []
        self.regions: List = []

    def _train_region(self, model, coords, values, tag):
        import copy
        tcfg = self.cfg.train
        lr0 = float(tcfg.lr)
        opt = torch.optim.Adam(model.parameters(), lr=lr0,
                               betas=tuple(tcfg.get("betas", [0.9, 0.999])),
                               weight_decay=float(tcfg.get("weight_decay", 0.0)))
        n = coords.shape[0]
        bs = min(int(tcfg.batch_size), n)
        epochs = int(tcfg.epochs)
        clip = float(tcfg.get("grad_clip", 0.0))
        lr_final = float(tcfg.get("lr_final", lr0))   # cosine-decay target (== lr0 disables)
        steps = max(1, n // bs)
        # Keep the BEST weights: SIREN at higher lr can diverge late in training, and for
        # compression we want the lowest-error snapshot, not the last one.
        best_loss, best_state = float("inf"), copy.deepcopy(model.state_dict())
        for ep in range(epochs):
            cur_lr = lr_final + 0.5 * (lr0 - lr_final) * (1.0 + math.cos(math.pi * ep / max(1, epochs - 1)))
            for g in opt.param_groups:
                g["lr"] = cur_lr
            perm = torch.randperm(n, device=coords.device)
            ep_loss = 0.0
            for s in range(steps):
                idx = perm[s * bs:(s + 1) * bs]
                x = self.transform(coords[idx])
                pred = model(x)
                loss = ((pred - values[idx]) ** 2).mean()
                opt.zero_grad(); loss.backward()
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step()
                ep_loss += loss.item()
            ep_loss /= steps
            if ep_loss < best_loss:
                best_loss = ep_loss
                best_state = copy.deepcopy(model.state_dict())
            if (ep + 1) % int(tcfg.get("log_every", 10)) == 0 or ep == epochs - 1:
                logging.info(f"[train {tag}] epoch {ep+1}/{epochs}  lr={cur_lr:.2e}  "
                             f"mse={ep_loss:.6e}  best={best_loss:.6e}")
        model.load_state_dict(best_state)   # restore the lowest-error snapshot

    @torch.no_grad()
    def _reconstruct_region(self, model, coords, shape):
        preds = np.empty((coords.shape[0], 1), np.float32)
        bs = 200000
        for s in range(0, coords.shape[0], bs):
            x = self.transform(coords[s:s + bs])
            preds[s:s + bs] = model(x).clamp(-1, 1).cpu().numpy()
        return preds.reshape(shape)  # (T,dz,dy,dx) normalized

    def fit(self, vol: TimeVaryingVolume):
        self.regions = self.partition.regions(vol.T, vol.Z, vol.Y, vol.X)
        in_dim = self.transform.out_dim(COORD_DIM)
        recon_norm = np.zeros((vol.T, vol.Z, vol.Y, vol.X), np.float32)
        t0 = time.time()
        for ri, region in enumerate(self.regions):
            coords, values, shape = region_coords_values(vol, region, self.device)
            model = build_model(self.cfg.model, in_dim, 1).to(self.device)
            self.models.append(model)
            tag = f"r{ri}" if len(self.regions) > 1 else "full"
            self._train_region(model, coords, values, tag)
            rec = self._reconstruct_region(model, coords, shape)
            z0, z1, y0, y1, x0, x1 = region
            recon_norm[:, z0:z1, y0:y1, x0:x1] = rec
        train_time = time.time() - t0
        recon = vol.denormalize(recon_norm)
        return recon, train_time

    def total_params(self):
        return int(sum(num_params(m) for m in self.models))


def run_experiment(cfg, vol, device, tag="model"):
    """Train + evaluate one model variant; returns a metrics dict."""
    comp = INRVolumeCompressor(cfg, device)
    recon, train_time = comp.fit(vol)

    data_range = max(vol.vmax - vol.vmin, 1e-8)
    p = psnr(recon, vol.data, data_range)

    wbytes = 2 if str(cfg.compress.get("weight_dtype", "float32")) == "float16" else 4
    model_bytes = comp.total_params() * wbytes
    data_bytes = vol.num_voxels() * int(cfg.compress.get("data_bytes_per_voxel", 4))
    cr = data_bytes / max(1, model_bytes)
    m = {"tag": tag, "model": str(cfg.model.get("name")),
         "transform": str(cfg.coord_transform.get("name")),
         "partition": str(cfg.partition.get("name")),
         "num_regions": len(comp.regions), "num_params": comp.total_params(),
         "model_bytes": model_bytes, "data_bytes": data_bytes,
         "compression_ratio": cr, "psnr": p, "train_time_s": train_time}
    logging.info(f"[result {tag}] PSNR={p:.2f} dB  CR={cr:.1f}x  params={comp.total_params()/1e3:.1f}K  "
                 f"({train_time:.1f}s)")
    return m, recon, comp


# ===========================================================================
# Orchestration
# ===========================================================================
def _merge(base: dict, override: dict) -> EasyConfig:
    """Deep-merge `override` onto a copy of `base` (both dict-like) -> EasyConfig."""
    out = EasyConfig()
    out.update(json.loads(json.dumps(dict(base))))  # deep copy
    if override:
        out.update(json.loads(json.dumps(dict(override))))
    return out


def main():
    ap = argparse.ArgumentParser(description="CoordNet INR compression of a time-varying volume.")
    ap.add_argument("--config", type=str, default="config/CoordNetCompression.yaml")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    cfg = EasyConfig(); cfg.load(args.config, recursive=False)
    if args.epochs is not None:
        cfg.train["epochs"] = args.epochs
    seed = int(cfg.get("seed", 0)); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device(args.device or cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    logging.info(f"[main] device={device}, seed={seed}")

    vol = build_volume(cfg)
    out_dir = args.out_dir or cfg.output.get("out_dir", "outputs/coordnet_compression")
    os.makedirs(out_dir, exist_ok=True)

    # Build the list of variants to run: the base `model`, or every entry in `compare`.
    compare = list(cfg.get("compare", []) or [])
    if not compare:
        variants = [("model", cfg)]
    else:
        variants = []
        for entry in compare:
            e = dict(entry)
            tag = str(e.pop("tag", e.get("model", {}).get("name", "variant")))
            vcfg = EasyConfig(); vcfg.update(json.loads(json.dumps(dict(cfg))))
            for key in ("model", "coord_transform", "partition", "train"):
                if key in e:
                    vcfg[key].update(e[key])
            variants.append((tag, vcfg))

    results = []
    for tag, vcfg in variants:
        m, recon, comp = run_experiment(vcfg, vol, device, tag=tag)
        results.append(m)
        if bool(cfg.output.get("save_checkpoint", True)):
            ckpt = os.path.join(out_dir, f"coordnet_{tag}.pt")
            torch.save({"state_dicts": [mm.state_dict() for mm in comp.models],
                        "regions": comp.regions, "metrics": m,
                        "value_range": [vol.vmin, vol.vmax]}, ckpt)
        if bool(cfg.output.get("save_reconstruction", False)):
            np.save(os.path.join(out_dir, f"recon_{tag}.npy"), recon)

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # comparison table
    logging.info("=" * 92)
    logging.info(f"{'tag':>14} | {'model':>12} | {'transform':>9} | {'part':>6} | "
                 f"{'PSNR(dB)':>8} | {'CR':>8} | {'params':>9}")
    logging.info("-" * 92)
    for m in results:
        logging.info(f"{m['tag']:>14} | {m['model']:>12} | {m['transform']:>9} | {m['partition']:>6} | "
                     f"{m['psnr']:>8.2f} | {m['compression_ratio']:>7.1f}x | {m['num_params']:>9}")
    logging.info("=" * 92)
    logging.info(f"[main] wrote metrics + checkpoints to {out_dir}")


if __name__ == "__main__":
    main()
