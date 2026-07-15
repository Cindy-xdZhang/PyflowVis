"""INR training utilities -- v2. The network is the *verified baseline architecture*
(CoordNetCompression.CoordNet, checked line-by-line against Han & Wang TVCG 2023,
see docs/referenceframe_inr_v2.md par.1) so baseline and proposed use the same class.

Frozen training recipe v2.3 (docs par.1, applied IDENTICALLY to every mode):
Adam(0.9, 0.999), weight decay 1e-6, MSE, coords/values pre-normalized to [-1, 1],
1000 epochs default (hard cap 2000 since 2026-07-15; per-architecture lr allowed
for the non-SIREN variants, see docs par.4.4j), lr 1e-5 cosine-decayed to 1e-6,
grad-clip 1.0, and an
ADAPTIVE batch: batch = min(32000, max(1, n_samples // min_steps_per_epoch)) with
min_steps_per_epoch = 64, so every INR gets >= 64 optimizer steps per epoch
regardless of how many samples it owns.

Recipe history (evidence in docs par.1):
  v2.0  literal paper recipe (lr 1e-5 const, 300 ep, batch 32000): paper-scale data
        has ~16k steps/epoch, our small fields ~16 -> everything undertrained.
  v2.1/2.2  lr 3e-4 cosine: fixed small fields but COLLAPSED the paper-size net
        (m=64, d=10) on cylinder2d to the mean-flow attractor (both seeds ended at
        normalized MSE 3.277e-2 = fluctuation energy; PSNR ~23 dB, vs ~50 dB known
        for the same net+data at lr 1e-5). Also made the per-window INR quality
        depend on steps-per-epoch (window-split cost = halved steps, agent E1a).
  v2.3  steps-normalized batch + SIREN-safe paper lr fixes both failure modes with
        one mechanism; epochs stay 1000 (hard rule is on epochs, not steps).
"""
from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from CoordNetCompression import CoordNet  # noqa: E402,F401  (verified baseline model class)
from models_alt import build_inr_model  # noqa: E402  (v_MLP0.0 / v_FINER0.0 variants)


def setup_determinism(seed: int = 0) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def coordnet_num_params(m: int, d: int, k: int = 3, p: int = 2) -> int:
    """Closed-form parameter count of CoordNet(in_dim=k, out_dim=p, m, d).
    Verified against torch: m=32, d=8, k=4, p=1 -> 307364 (smoke run 2026-07-12)."""
    return (41 + 32 * d) * m * m + (2 * k + 21 + 8 * d + 8 * p) * m + (p * p + 3 * p)


def pick_m_for_budget(budget: int, d: int, k: int = 3, p: int = 2, m_min: int = 4) -> int:
    """Largest m with coordnet_num_params(m, d, k, p) <= budget (floored at m_min)."""
    A = 41 + 32 * d
    B = 2 * k + 21 + 8 * d + 8 * p
    C = (p * p + 3 * p) - budget
    m = int((-B + math.sqrt(max(B * B - 4 * A * C, 0.0))) / (2 * A))
    while coordnet_num_params(m + 1, d, k, p) <= budget:
        m += 1
    while m > m_min and coordnet_num_params(m, d, k, p) > budget:
        m -= 1
    return max(m, m_min)


@dataclass
class TrainCfg:
    epochs: int = 1000
    batch_size: int = 32000        # upper cap; effective batch adapts (see below)
    min_steps_per_epoch: int = 64  # v2.3: batch shrinks so steps/epoch >= this
    lr: float = 1e-5
    lr_final: float = 1e-6         # cosine-decay target; == lr disables the schedule
    grad_clip: float = 1.0         # 0 disables
    weight_decay: float = 1e-6
    betas: tuple = (0.9, 0.999)
    log_every: int = 100

    def effective_batch(self, n: int) -> int:
        return max(1, min(self.batch_size, n, n // max(1, self.min_steps_per_epoch) or 1))


@dataclass
class MinMax:
    """Per-component min-max normalizer to [-1, 1] (used for values; coords use an
    analogous per-axis normalization at the call site)."""
    lo: np.ndarray
    hi: np.ndarray

    @classmethod
    def fit(cls, arr: np.ndarray) -> "MinMax":
        return cls(lo=arr.min(axis=0), hi=arr.max(axis=0))

    def encode(self, arr: np.ndarray) -> np.ndarray:
        scale = np.maximum(self.hi - self.lo, 1e-12)
        return (2.0 * (arr - self.lo) / scale - 1.0).astype(np.float32)

    def decode(self, arr: np.ndarray) -> np.ndarray:
        scale = np.maximum(self.hi - self.lo, 1e-12)
        return (0.5 * (arr.astype(np.float64) + 1.0)) * scale + self.lo


def train_inr(coords_n: np.ndarray, values_n: np.ndarray, m: int, d: int,
              cfg: TrainCfg, device: torch.device, seed: int, tag: str = "",
              log=print, model_name: str = "coordnet",
              model_kwargs: dict | None = None) -> tuple[torch.nn.Module, dict]:
    """Fit an INR to pre-normalized (coords_n in [-1,1]^k, values_n in [-1,1]^p).

    model_name: 'coordnet' (default, SIREN CoordNet) | 'mlp' (v_MLP0.0) |
    'finer' (v_FINER0.0). All variants share the CoordNet skeleton, so the
    closed-form parameter count holds for every one (asserted below)."""
    k, p = coords_n.shape[1], values_n.shape[1]
    torch.manual_seed(seed)
    model = build_inr_model(model_name, k, p, m=m, d=d,
                            **(model_kwargs or {})).to(device)
    n_params = sum(pp.numel() for pp in model.parameters())
    assert n_params == coordnet_num_params(m, d, k, p), "param formula drift"

    coords_t = torch.from_numpy(coords_n.astype(np.float32)).to(device)
    values_t = torch.from_numpy(values_n.astype(np.float32)).to(device)
    n = coords_t.shape[0]
    bs = cfg.effective_batch(n)
    steps = max(1, n // bs)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=cfg.betas,
                           weight_decay=cfg.weight_decay)
    gen = torch.Generator(device="cpu").manual_seed(seed + 1)

    t0 = time.time()
    last_mse, best_mse = float("inf"), float("inf")
    for ep in range(cfg.epochs):
        cur_lr = cfg.lr_final + 0.5 * (cfg.lr - cfg.lr_final) * (
            1.0 + math.cos(math.pi * ep / max(1, cfg.epochs - 1)))
        for gparam in opt.param_groups:
            gparam["lr"] = cur_lr
        perm = torch.randperm(n, generator=gen).to(device)
        ep_loss = 0.0
        for s in range(steps):
            idx = perm[s * bs:(s + 1) * bs]
            pred = model(coords_t[idx])
            loss = ((pred - values_t[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            ep_loss += loss.item()
        last_mse = ep_loss / steps
        best_mse = min(best_mse, last_mse)
        if (ep + 1) % cfg.log_every == 0 or ep == cfg.epochs - 1:
            log(f"    [{tag}] epoch {ep+1}/{cfg.epochs} lr={cur_lr:.1e} mse={last_mse:.3e}")
    stats = {"params": int(n_params), "m": m, "d": d, "n_samples": int(n),
             "model": model_name, "train_time_s": time.time() - t0,
             "last_mse": last_mse, "best_mse": best_mse}
    return model, stats


def train_inr_best_of_seeds(coords_n: np.ndarray, values_n: np.ndarray, m: int, d: int,
                            cfg: TrainCfg, device: torch.device, seed_base: int,
                            n_seeds: int, tag: str = "", log=print,
                            model_name: str = "coordnet",
                            model_kwargs: dict | None = None
                            ) -> tuple[torch.nn.Module, dict]:
    """v2.2 protocol: train the same INR with n_seeds derived seeds and keep the
    weights with the lowest final MSE. Rationale: SIREN convergence is chaotically
    seed-sensitive (observed 2026-07-12: same data statistics, same shape, final MSE
    2e-4 vs 8e-6 across seeds). Best-of-k is a legitimate *encode-time* search --
    decode cost and stored bytes are unchanged -- and it is applied symmetrically to
    baseline and proposed."""
    best = None
    all_mse = []
    for si in range(n_seeds):
        seed = seed_base + 7777 * si
        model, st = train_inr(coords_n, values_n, m, d, cfg, device, seed=seed,
                              tag=f"{tag} s{si}", log=log, model_name=model_name,
                              model_kwargs=model_kwargs)
        all_mse.append(st["last_mse"])
        if best is None or st["last_mse"] < best[1]["last_mse"]:
            best = (model, st)
    model, st = best
    st["seed_mses"] = all_mse
    st["seed_spread"] = float(max(all_mse) / max(min(all_mse), 1e-300))
    if n_seeds > 1:
        log(f"    [{tag}] best-of-{n_seeds} mse={min(all_mse):.3e} "
            f"(spread x{st['seed_spread']:.1f})")
    return model, st


@torch.no_grad()
def eval_inr(model: torch.nn.Module, coords_n: np.ndarray, device: torch.device,
             batch: int = 262144) -> np.ndarray:
    model.eval()
    out = np.empty((coords_n.shape[0], model.decoder.l2.linear.out_features
                    if hasattr(model.decoder, "l2") else model.decoder.out_features),
                   np.float32)
    for s in range(0, coords_n.shape[0], batch):
        x = torch.from_numpy(coords_n[s:s + batch].astype(np.float32)).to(device)
        out[s:s + batch] = model(x).clamp(-1, 1).cpu().numpy()
    model.train()
    return out


def vpsnr(pred: np.ndarray, gt: np.ndarray) -> float:
    """Data-level PSNR over all components: 20 log10(range) - 10 log10(MSE),
    range = global max - min of gt. One formula for every method in v2."""
    gt64 = gt.astype(np.float64); pr64 = pred.astype(np.float64)
    mse = float(np.mean((pr64 - gt64) ** 2))
    rng = float(gt64.max() - gt64.min())
    if mse <= 0:
        return float("inf")
    return 20.0 * math.log10(max(rng, 1e-300)) - 10.0 * math.log10(mse)
