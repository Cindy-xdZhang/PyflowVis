"""CLI runner -- 3D experiments (docs/referenceframe_inr_3d.md).

Examples:
    python validate_rft3d.py                                  # must pass first
    python run_experiment3d.py --field rot3d --epochs 50 --modes baseline
    python run_experiment3d.py --field deltawing --stride_t 2 --stride_xyz 2 \
        --budget_frac 0.05 --tau 0.05 --n_windows 1 --allow_full_window \
        --max_inrs 3 --observer consttrans --model mlp --lr 1e-3
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from pipeline3d import ExpCfg3D, run_experiment_3d  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Reference-frame partition INR 3D")
    ap.add_argument("--field", default="deltawing", help="deltawing | rot3d")
    ap.add_argument("--modes", default="baseline,pro_budget")
    ap.add_argument("--model", default="coordnet",
                    choices=["coordnet", "mlp", "finer"])
    ap.add_argument("--finer_first_bias_scale", type=float, default=None)
    ap.add_argument("--stride_t", type=int, default=1,
                    help="dataset time stride (downsample; tags the field name)")
    ap.add_argument("--stride_xyz", type=int, default=1,
                    help="dataset space stride on all three axes")
    ap.add_argument("--t_max", type=int, default=0,
                    help=">0: truncate to the first t_max timesteps (pre-stride)")
    ap.add_argument("--m_base", type=int, default=24)
    ap.add_argument("--d_base", type=int, default=10)
    ap.add_argument("--budget_frac", type=float, default=0.0)
    ap.add_argument("--k_cell", type=int, default=4,
                    help="cell edge (k^3 voxels/cell); stats memory ~ "
                         "T*nCells*288 B")
    ap.add_argument("--tau", type=float, default=0.05,
                    help="merge tolerance; NEGATIVE (use --tau=-1) = M=1 fast "
                         "path: whole domain as one region, no merge walk")
    ap.add_argument("--absorb_min_pixels", type=int, default=0,
                    help=">0: post-merge absorb regions smaller than this many "
                         "VOXELS (spec deviation, logged)")
    ap.add_argument("--alloc", default="uniform",
                    choices=["uniform", "pixels", "capsmall"])
    ap.add_argument("--n_windows", type=int, default=1)
    ap.add_argument("--max_inrs", type=int, default=0)
    ap.add_argument("--allow_full_window", action="store_true", default=True,
                    help="single window is the v2-winning structure; pass "
                         "--no_full_window to forbid it")
    ap.add_argument("--no_full_window", dest="allow_full_window",
                    action="store_false")
    ap.add_argument("--observer", default="tvfull",
                    choices=["tvfull", "tvtrans", "constfull", "consttrans"])
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=32000)
    ap.add_argument("--min_steps_per_epoch", type=int, default=64)
    ap.add_argument("--max_steps_per_epoch", type=int, default=0,
                    help=">0: cap steps/epoch by training on a fixed random "
                         "subsample of the grid (3D wall-clock knob; eval is "
                         "always full-grid). 0 = full sweep like v2.")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lr_final", type=float, default=1e-6)
    ap.add_argument("--warmup_frac", type=float, default=0.0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_seeds", type=int, default=1)
    ap.add_argument("--device", default="")
    ap.add_argument("--out_dir", default=str(_HERE / "outputs"))
    args = ap.parse_args()

    assert args.epochs <= 2000, "hard rule: <= 2000 epochs"
    assert 0 <= args.budget_frac < 1, "budget_frac must be in [0, 1)"
    cfg = ExpCfg3D(field=args.field, model=args.model,
                   finer_first_bias_scale=args.finer_first_bias_scale,
                   stride_t=args.stride_t, stride_xyz=args.stride_xyz,
                   t_max=args.t_max,
                   m_base=args.m_base, d_base=args.d_base,
                   budget_frac=args.budget_frac,
                   k_cell=args.k_cell, tau=args.tau, alloc=args.alloc,
                   absorb_min_pixels=args.absorb_min_pixels,
                   n_windows=args.n_windows,
                   allow_full_window=args.allow_full_window,
                   max_inrs=args.max_inrs, observer=args.observer,
                   epochs=args.epochs, batch_size=args.batch_size,
                   min_steps_per_epoch=args.min_steps_per_epoch,
                   max_steps_per_epoch=args.max_steps_per_epoch,
                   lr=args.lr, lr_final=args.lr_final,
                   warmup_frac=args.warmup_frac, grad_clip=args.grad_clip,
                   log_every=args.log_every,
                   seed=args.seed, n_seeds=args.n_seeds, device=args.device,
                   out_dir=str(Path(args.out_dir) / args.field),
                   modes=tuple(s.strip() for s in args.modes.split(",")
                               if s.strip()))

    t0 = time.time()
    out = run_experiment_3d(cfg)
    print(f"\n=== summary ({args.field}, {time.time() - t0:.0f}s total) ===")
    print(f"{'mode':>12} | {'PSNR(dB)':>8} | {'params':>10} | {'bytes':>10} | "
          f"{'CR':>7} | {'#INR':>4} | N/window")
    for mode, r in out["results"].items():
        print(f"{mode:>12} | {r['psnr']:>8.2f} | {r['params_total']:>10,} | "
              f"{r['total_bytes']:>10,} | {r['compression_ratio']:>6.1f}x | "
              f"{r['n_inrs']:>4} | {r['regions_per_window']}")


if __name__ == "__main__":
    main()
