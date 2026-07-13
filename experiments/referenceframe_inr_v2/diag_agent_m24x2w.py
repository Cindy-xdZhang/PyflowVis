"""E1b window-cost diagnostic: 2 windows, per-INR capacity RESTORED to m=24.

Uses pipeline.run_proposed with budget_factor=2.0: with M=2 region-INRs the
per-INR share = 2.0 * B(24) / 2 = B(24) = 99,154 params -> pick_m_for_budget
gives m_r = 24 for each window INR (total params 2B -- intentionally over
budget; this is an attribution diagnostic, not a compression result).

Prediction under H1 (capacity, not time span, is what pro_budget's halving
costs): PSNR ~ 62.7 dB, i.e. the window split itself costs ~nothing once the
per-INR capacity is restored.

Everything else mirrors pipeline.run_experiment's setup (setup_determinism,
device selection, v2.2 protocol: 1000 epochs, lr 3e-4 cosine->1e-5, clip 1.0,
best-of-3 seeds). pipeline.py is NOT modified.

Run:  python -u diag_agent_m24x2w.py > outputs/diag_agent_m24x2w.log
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for pth in (str(_ROOT), str(_HERE)):
    if pth not in sys.path:
        sys.path.insert(0, pth)

from pipeline import ExpCfg, budget_B, compute_partitions, load_field, run_proposed  # noqa: E402
from inr import setup_determinism  # noqa: E402


def main():
    cfg = ExpCfg(field="rfc", n_windows=2, tau=0.05, n_seeds=3,
                 out_dir=str(_HERE / "outputs" / "diag_agent_m24x2w"),
                 m_base=24, d_base=4)
    assert cfg.epochs <= 1000, "hard rule: <= 1000 epochs"

    # mirror pipeline.run_experiment setup
    setup_determinism(cfg.seed)
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    fd = load_field(cfg.field)
    print(f"[{fd.name}] shape (T,Y,X,2)={fd.shape}, B={budget_B(cfg)} params "
          f"(m={cfg.m_base}, d={cfg.d_base}), device={device}")
    print(f"[{fd.name}] tau-merge partition:")
    parts = compute_partitions(fd, cfg)

    print(f"[{fd.name}] === diagnostic mode m24x2w "
          f"(budget_factor=2.0 -> per-INR share = B, m_r=24 per window) ===")
    res = run_proposed(fd, cfg, parts, device, budget_factor=2.0,
                       use_observer=True, mode_name="m24x2w")

    print(f"\n[{fd.name}] m24x2w: PSNR={res['psnr']:.2f} dB  "
          f"params={res['params_total']:,} (per-INR share={res['per_inr_share']:,})  "
          f"#INR={res['n_inrs']}  regions/window={res['regions_per_window']}  "
          f"({res['train_time_s']:.0f}s)")
    for st in res["detail"]:
        print(f"  w{st['window']}r{st['region']}: m={st['m']} params={st['params']:,} "
              f"n={st['n_samples']} last_mse={st['last_mse']:.3e} "
              f"best_mse={st['best_mse']:.3e} seed_mses="
              + "[" + ", ".join(f"{x:.3e}" for x in st["seed_mses"]) + "]"
              + f" spread=x{st['seed_spread']:.1f}")

    od = Path(cfg.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    with open(od / "rfc_m24x2w_metrics.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(f"[{fd.name}] wrote {od / 'rfc_m24x2w_metrics.json'}")


if __name__ == "__main__":
    main()
