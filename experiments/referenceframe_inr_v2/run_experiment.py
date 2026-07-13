"""CLI runner -- v2 experiments (docs/referenceframe_inr_v2.md).

Examples:
    python validate_rfc.py                                    # must pass first
    python run_experiment.py --field rfc --epochs 300
    python run_experiment.py --field cylinder2d --m_base 64 --d_base 10 --epochs 300
    python run_experiment.py --field rfc --modes baseline,pro_budget
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from pipeline import ExpCfg, run_experiment  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Reference-frame partition INR v2")
    ap.add_argument("--field", default="rfc",
                    help="rfc | rfc128 | tworotor | beads2d | cylinder2d | boussinesq")
    ap.add_argument("--modes", default="baseline,pro_budget,pro_quality,no_observer")
    ap.add_argument("--m_base", type=int, default=24)
    ap.add_argument("--d_base", type=int, default=4)
    ap.add_argument("--k_cell", type=int, default=2)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--absorb_min_pixels", type=int, default=0,
                    help=">0: post-merge absorb smaller regions (spec deviation)")
    ap.add_argument("--alloc", default="uniform", choices=["uniform", "pixels"],
                    help="per-INR budget split (uniform = spec; pixels = proportional)")
    ap.add_argument("--n_windows", type=int, default=2)
    ap.add_argument("--allow_full_window", action="store_true",
                    help="DIAGNOSTIC: permit n_windows=1 (violates the <=T/2 spec rule)")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=32000)
    ap.add_argument("--min_steps_per_epoch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--lr_final", type=float, default=1e-6)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_seeds", type=int, default=3,
                    help="v2.2: best-of-k seeds per INR (encode-time search)")
    ap.add_argument("--device", default="")
    ap.add_argument("--out_dir", default=str(_HERE / "outputs"))
    args = ap.parse_args()

    assert args.epochs <= 1000, "hard rule: <= 1000 epochs"
    cfg = ExpCfg(field=args.field, m_base=args.m_base, d_base=args.d_base,
                 k_cell=args.k_cell, tau=args.tau, alloc=args.alloc,
                 absorb_min_pixels=args.absorb_min_pixels, n_windows=args.n_windows,
                 allow_full_window=args.allow_full_window,
                 min_steps_per_epoch=args.min_steps_per_epoch,
                 epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                 lr_final=args.lr_final, grad_clip=args.grad_clip,
                 seed=args.seed, n_seeds=args.n_seeds, device=args.device,
                 out_dir=str(Path(args.out_dir) / args.field),
                 modes=tuple(s.strip() for s in args.modes.split(",") if s.strip()))

    t0 = time.time()
    out = run_experiment(cfg)
    print(f"\n=== summary ({args.field}, {time.time() - t0:.0f}s total) ===")
    print(f"{'mode':>12} | {'PSNR(dB)':>8} | {'params':>10} | {'bytes':>10} | "
          f"{'CR':>7} | {'#INR':>4} | N/window")
    for mode, r in out["results"].items():
        print(f"{mode:>12} | {r['psnr']:>8.2f} | {r['params_total']:>10,} | "
              f"{r['total_bytes']:>10,} | {r['compression_ratio']:>6.1f}x | "
              f"{r['n_inrs']:>4} | {r['regions_per_window']}")


if __name__ == "__main__":
    main()
