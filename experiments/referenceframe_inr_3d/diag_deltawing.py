"""Dry-run diagnostics on deltaWing -- no training.

Answers, before any GPU experiment (the 2D project's E/E0 prior-applicability
criterion, docs/referenceframe_inr_v2.md par.4.9-1):
  1. How much of deltaWing's temporal energy does a single global rigid-motion
     observer explain?  (E/E0 per observer variant: tvfull / tvtrans /
     constfull / consttrans)
  2. What does the solved observer look like physically (mean translation /
     rotation rates)?
  3. How does the tau-merge partition behave (N vs tau) at a given cell size?

Run (downsampled first -- full-res cell stats need ~a minute and ~1 GB):
    python diag_deltawing.py --stride_t 2 --stride_xyz 2
    python diag_deltawing.py                     # full resolution
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for pth in (str(_HERE),):
    if pth not in sys.path:
        sys.path.insert(0, pth)

from pipeline3d import load_field_3d  # noqa: E402
from killing3d import (compute_cell_stats_3d, solve_killing_3d,  # noqa: E402
                       solve_killing_trans_3d)
from partition3d import merge_partition_3d  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="deltawing")
    ap.add_argument("--stride_t", type=int, default=1)
    ap.add_argument("--stride_xyz", type=int, default=1)
    ap.add_argument("--t_max", type=int, default=0)
    ap.add_argument("--k_cell", type=int, default=4)
    ap.add_argument("--boundary_skip", type=int, default=2)
    ap.add_argument("--taus", default="0.02,0.05,0.1,0.2")
    ap.add_argument("--absorb", type=int, default=0,
                    help="absorb_min_pixels (voxels) for the partition probe")
    ap.add_argument("--skip_partition", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    fd = load_field_3d(args.field, args.stride_t, args.stride_xyz, args.t_max)
    print(f"[{fd.name}] shape (T,Z,Y,X,3)={fd.shape}, "
          f"{fd.data_bytes() / 2**20:.1f} MiB, dt={fd.dt:.5f}, "
          f"load {time.time() - t0:.1f}s")
    print(f"  value range: [{fd.data.min():.4f}, {fd.data.max():.4f}], "
          f"mean|v|={np.linalg.norm(fd.data, axis=-1).mean():.4f}")

    t0 = time.time()
    stats = compute_cell_stats_3d(fd.data, fd.xs, fd.ys, fd.zs, fd.dt,
                                  k=args.k_cell,
                                  boundary_skip=args.boundary_skip)
    nC = stats.n_cells
    print(f"  cell stats: k={args.k_cell} -> nC={nC} "
          f"({nC[0] * nC[1] * nC[2]} cells), {time.time() - t0:.1f}s")

    # global (single-region) observer solves, all variants
    AtA = stats.AtA.sum(axis=(1, 2, 3))          # (T, 6, 6)
    g = stats.g.sum(axis=(1, 2, 3))
    e0 = stats.e0.sum(axis=(1, 2, 3))
    E0 = float(e0.sum())
    print(f"  global E0 (sum ||dv/dt||^2) = {E0:.6e}")

    q_tv, E_tv = solve_killing_3d(AtA, g, e0)
    q_tvt, E_tvt = solve_killing_trans_3d(AtA, g, e0)
    q_cf, _ = solve_killing_3d(AtA.sum(0), g.sum(0), e0.sum())
    E_cf = max(float(e0.sum() + q_cf @ g.sum(0)), 0.0)
    q_ct, _ = solve_killing_trans_3d(AtA.sum(0), g.sum(0), e0.sum())
    E_ct = max(float(e0.sum() + q_ct[:3] @ g.sum(0)[:3]), 0.0)

    print("  global observer variants (E/E0, lower = more explainable):")
    print(f"    tvfull     E/E0 = {float(E_tv.sum()) / E0:.4f}   "
          f"mean t=({q_tv[:, 0].mean():+.4f},{q_tv[:, 1].mean():+.4f},"
          f"{q_tv[:, 2].mean():+.4f}) "
          f"mean w=({q_tv[:, 3].mean():+.5f},{q_tv[:, 4].mean():+.5f},"
          f"{q_tv[:, 5].mean():+.5f})")
    print(f"    tvtrans    E/E0 = {float(E_tvt.sum()) / E0:.4f}")
    print(f"    constfull  E/E0 = {E_cf / E0:.4f}   "
          f"t=({q_cf[0]:+.4f},{q_cf[1]:+.4f},{q_cf[2]:+.4f}) "
          f"w=({q_cf[3]:+.5f},{q_cf[4]:+.5f},{q_cf[5]:+.5f})")
    print(f"    consttrans E/E0 = {E_ct / E0:.4f}   "
          f"t=({q_ct[0]:+.4f},{q_ct[1]:+.4f},{q_ct[2]:+.4f})")

    if not args.skip_partition:
        Tn = fd.shape[0]
        for tau in [float(s) for s in args.taus.split(",")]:
            t0 = time.time()
            part = merge_partition_3d(stats, 0, Tn, tau,
                                      absorb_min_pixels=args.absorb)
            sizes = sorted((int(r.npix) for r in part.regions), reverse=True)
            rhos = [r.E / max(r.E0, 1e-300) for r in part.regions]
            print(f"  tau={tau}: N={part.n_regions}"
                  + (f" (absorbed {part.n_absorbed})" if part.n_absorbed else "")
                  + f"  sizes(top5)={sizes[:5]}  "
                  f"E/E0(top5)={[f'{r:.3f}' for r in rhos[:5]]}  "
                  f"({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
