"""Save partition label maps (and per-region angular velocity c) as PNGs -- v2.

Usage:
    python viz_partition.py --field cylinder2d --tau 0.1 --n_windows 2
Writes outputs/<field>/partition_tau<т>_w<i>.png (labels colored, region ids + c drawn).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from pipeline import ExpCfg, load_field, compute_partitions  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="rfc")
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--n_windows", type=int, default=2)
    ap.add_argument("--k_cell", type=int, default=2)
    args = ap.parse_args()

    fd = load_field(args.field)
    cfg = ExpCfg(field=args.field, tau=args.tau, n_windows=args.n_windows,
                 k_cell=args.k_cell)
    parts = compute_partitions(fd, cfg)
    od = _HERE / "outputs" / args.field
    od.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(parts):
        fig, ax = plt.subplots(figsize=(10, 10 * fd.shape[1] / fd.shape[2] + 1))
        rng = np.random.default_rng(0)
        order = rng.permutation(p.n_regions)          # shuffle colors for contrast
        ax.imshow(order[p.labels_pixels], origin="lower", cmap="tab20",
                  extent=[fd.xs[0], fd.xs[-1], fd.ys[0], fd.ys[-1]], aspect="equal",
                  interpolation="nearest")
        for r_i, r in enumerate(p.regions):
            if r.npix < 16:
                continue
            m = p.labels_pixels == r_i
            yy, xx = np.nonzero(m)
            cx, cy = fd.xs[int(xx.mean())], fd.ys[int(yy.mean())]
            ax.text(cx, cy, f"{r_i}\nc={r.q[:, 2].mean():.2f}", ha="center",
                    va="center", fontsize=7,
                    bbox=dict(fc="white", alpha=0.75, ec="none"))
        ax.set_title(f"{args.field} window[{p.it0},{p.it1}) tau={args.tau} "
                     f"N={p.n_regions}")
        fn = od / f"partition_tau{args.tau}_w{i}.png"
        fig.savefig(fn, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print("wrote", fn)


if __name__ == "__main__":
    main()
