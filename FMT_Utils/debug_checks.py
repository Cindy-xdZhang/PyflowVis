"""
Opt-in validation / debug checks for the FTLE-upsampling pipeline.

These helpers verify, at the four key stages (FTLE computation, dataset construction,
training, testing), that tensors have the expected shape/range/finiteness and that the
data is not silently degenerate (e.g. all-zero FTLE from integration overshoot, or a
model accidentally training in eval mode).

Cost when disabled: a single int comparison per call (effectively free). Enable with:
  * environment variable  FMT_DEBUG=1   -> warn on problems, log a one-line summary
  *                       FMT_DEBUG=2   -> additionally RAISE on a failed check (strict)
  * or programmatically    set_debug(1) / set_debug(2)
"""

import os
import logging
import numpy as np
import torch


def _env_level() -> int:
    try:
        return int(os.environ.get("FMT_DEBUG", "0"))
    except Exception:
        return 0


# _LEVEL = _env_level()
_LEVEL = 1




def set_debug(level: int) -> None:
    """Set the debug level globally (0=off, 1=warn+log, 2=strict/raise)."""
    global _LEVEL
    _LEVEL = int(level)
    os.environ["FMT_DEBUG"] = str(int(level))
    if _LEVEL:
        logging.info(f"[FMT_DEBUG] enabled at level {_LEVEL}")


def debug_level() -> int:
    return _LEVEL


def enabled() -> bool:
    return _LEVEL >= 1


def _to_np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _fail(msg: str, strict: bool) -> None:
    logging.error("[FMT_DEBUG] CHECK FAILED: " + msg)
    if strict or _LEVEL >= 2:
        raise AssertionError(msg)


def check_finite(name: str, arr, strict: bool = False) -> None:
    if not enabled():
        return
    a = _to_np(arr)
    n_nan = int(np.isnan(a).sum())
    n_inf = int(np.isinf(a).sum())
    if n_nan or n_inf:
        _fail(f"{name}: {n_nan} NaN and {n_inf} Inf in array of shape {a.shape}", strict)


def check_ftle_field(name: str, field, expected_shape=None, strict: bool = False) -> None:
    """Stage 1 — FTLE computation: shape, finiteness, valid fraction, non-degeneracy."""
    if not enabled():
        return
    a = _to_np(field).astype(np.float64)
    check_finite(name, a, strict)
    if expected_shape is not None and tuple(a.shape) != tuple(expected_shape):
        _fail(f"{name}: shape {tuple(a.shape)} != expected {tuple(expected_shape)}", strict)
    tot = max(1, a.size)
    nz = int(np.count_nonzero(a))
    dr = float(a.max() - a.min())
    logging.info(f"[FMT_DEBUG] {name}: shape={tuple(a.shape)} none zero(valid) cells={nz}/{tot} "
                 f"({100.0*nz/tot:.1f}%) range=[{a.min():.4g},{a.max():.4g}] "
                 f"dynRange={dr:.4g} mean={a.mean():.4g}")
    if dr < 1e-6:
        _fail(f"{name}: DEGENERATE FTLE (dynRange={dr:.2e} ~ constant) — likely "
              f"integration overshoot/invalid pathlines (all-zero field)", strict)
    elif nz < 0.2 * tot:
        logging.warning(f"[FMT_DEBUG] {name}: only {100.0*nz/tot:.1f}% valid cells "
                        f"(many pathlines left the domain / ran out of time)")


def check_ftle_lowhigh(low, high, upsampling: int, strict: bool = False) -> None:
    """Stage 1 — low- vs high-res FTLE slice consistency (shape ratio sanity)."""
    if not enabled():
        return
    lh, lw = _to_np(low).shape
    hh, hw = _to_np(high).shape
    for d, (l, h) in enumerate([(lh, hh), (lw, hw)]):
        if abs(h - l * upsampling) > upsampling:  # tolerate linspace off-by-one
            logging.warning(f"[FMT_DEBUG] FTLE low/high: dim{d} high={h} is not ~"
                            f"{upsampling}x low={l} (grids may be misaligned)")


def check_dataset(name: str, lowResFTLE, labels, lowResPathlines,
                  nerbors: int, L: int, upsampling: int, strict: bool = False) -> None:
    """Stage 2 — dataset construction: shapes, finiteness, sample-count consistency,
    pathline structure, and normalized label range."""
    if not enabled():
        return
    lr = _to_np(lowResFTLE)
    lb = _to_np(labels)
    pl = _to_np(lowResPathlines)
    logging.info(f"[FMT_DEBUG] dataset[{name}]: lowResFTLE{tuple(lr.shape)} "
                 f"labels{tuple(lb.shape)} pathlines{tuple(pl.shape)}")
    check_finite(f"dataset[{name}].lowResFTLE", lr, strict)
    check_finite(f"dataset[{name}].labels", lb, strict)
    check_finite(f"dataset[{name}].pathlines", pl, strict)

    n = lr.shape[0]
    if not (n == lb.shape[0] == pl.shape[0]):
        _fail(f"dataset[{name}]: sample-count mismatch lowRes={lr.shape[0]} "
              f"labels={lb.shape[0]} pathlines={pl.shape[0]}", strict)

    # per-patch upsampling relation: label patch should be UP x lowRes patch
    if lr.ndim == 3 and lb.ndim == 3:
        if lb.shape[1] != lr.shape[1] * upsampling or lb.shape[2] != lr.shape[2] * upsampling:
            _fail(f"dataset[{name}]: label patch {lb.shape[1:]} != {upsampling}x "
                  f"lowRes patch {lr.shape[1:]}", strict)

    # pathline structure [N, nerbors, L, 3]
    if pl.ndim == 4 and (pl.shape[1] != nerbors or pl.shape[2] != L or pl.shape[3] < 2):
        logging.warning(f"[FMT_DEBUG] dataset[{name}]: pathline dims {tuple(pl.shape)} "
                        f"not [N,{nerbors},{L},>=2]")

    lo, hi = float(lb.min()), float(lb.max())
    logging.info(f"[FMT_DEBUG] dataset[{name}]: normalized label range=[{lo:.4g},{hi:.4g}] (expect ~[0,1])")
    if lo < -1e-3 or hi > 1.0 + 1e-3:
        logging.warning(f"[FMT_DEBUG] dataset[{name}]: labels outside [0,1]: [{lo:.4g},{hi:.4g}]")
    if hi - lo < 1e-6:
        _fail(f"dataset[{name}]: labels are ~constant (range {hi-lo:.2e}) — degenerate FTLE", strict)


def check_train_step(model, pred, label, strict: bool = False) -> None:
    """Stage 3 — training: model really in train() mode, shapes match, pred finite."""
    if not enabled():
        return
    if hasattr(model, "training") and not model.training:
        _fail("train step: model is NOT in train() mode — BatchNorm/Dropout are frozen "
              "(this was the eval-mode bug). Call model.train() each epoch.", strict)
    if tuple(pred.shape) != tuple(label.shape):
        _fail(f"train step: pred shape {tuple(pred.shape)} != label shape {tuple(label.shape)}", strict)
    check_finite("train.pred", pred, strict)


def check_pred_grid(name: str, pred_grid, gt_grid, strict: bool = False) -> None:
    """Stage 4 — testing: reconstructed prediction shape/finiteness and GT non-degeneracy."""
    if not enabled():
        return
    p = _to_np(pred_grid)
    g = _to_np(gt_grid)
    check_finite(f"test[{name}].pred", p, strict)
    if tuple(p.shape) != tuple(g.shape):
        _fail(f"test[{name}]: pred shape {tuple(p.shape)} != gt shape {tuple(g.shape)}", strict)
    gdr = float(g.max() - g.min())
    if gdr < 1e-6:
        logging.warning(f"[FMT_DEBUG] test[{name}]: GT dynRange={gdr:.2e} ~ constant "
                        f"-> PSNR (uses GT dynamic range) will be meaningless / hugely negative")
    logging.info(f"[FMT_DEBUG] test[{name}]: pred range=[{p.min():.4g},{p.max():.4g}] "
                 f"gt range=[{g.min():.4g},{g.max():.4g}]")
