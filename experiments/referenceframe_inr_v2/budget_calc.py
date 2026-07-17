"""Model-size calculator for the strict-compression experiment
(mainExp_compress_1.1, docs/referenceframe_inr_v2.md).

Task: given the flow field an INR must fit (raw float32 size X bytes), compute
the byte budget at a fraction of X (default 5% / 10% / 20%) and solve the
CoordNet-skeleton network size (width m, depth d) that fits each budget:

    baseline:  params(m, d) * 4 + 40 B                  <= frac * X
    proposed:  sum_r params(m_r, d) * 4 + side_info     <= frac * X
      side_info = cell-label maps + per-region killing params (a,b,c)(t)
                  + per-region xi-bbox / value-range / width m_r   (docs par.2.5)

The closed-form parameter count holds bit-exactly for all three INR variants
(coordnet / mlp / finer share the CoordNet skeleton, docs par.1b), so one
calculator covers the "Coordinate INR" (SIREN CoordNet) and "MLP" baselines
alike.

Depth policy: d stays at the field's frozen baseline depth (rfc d=4,
cylinder2d/boussinesq d=10) so budget tiers differ ONLY in width m -- varying
d per tier would confound depth with budget and break comparability with every
recorded experiment.  --d_sweep prints an informational width/depth trade-off
table (which (m, d) pairs the same budget buys) without changing the policy.

The numbers here are planning values; the pipeline recomputes the exact side
info at run time from the actual partition (pipeline.run_proposed) and asserts
total_bytes <= frac * X.

Examples:
    python budget_calc.py                           # all experiment fields
    python budget_calc.py --field cylinder2d --d_sweep
    python budget_calc.py --field boussinesq --load  # verify table vs real data
    python budget_calc.py --shape 128 80 320 --d 10 --n_inrs 5
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from inr import coordnet_num_params, pick_m_for_budget  # noqa: E402

# (T, Y, X, C) after the resampling done in pipeline.load_field (source of
# truth for shapes: pipeline.py resample2UnsteadyField calls, verified against
# docs par.4.2). C = 2 velocity components, stored float32 -> 4 B each.
FIELD_SHAPES = {
    "rfc":        (64, 64, 64, 2),
    "rfc128":     (64, 128, 128, 2),
    "cylinder2d": (128, 80, 320, 2),
    "boussinesq": (128, 225, 75, 2),
    "gerris0":    (128, 256, 128, 2),
    "gerris4":    (128, 256, 128, 2),
}

# Frozen per-field baseline depth (mainExp_2.3 configs; see docs par.4.4j table).
FIELD_D = {"rfc": 4, "rfc128": 4, "cylinder2d": 10, "boussinesq": 10,
           "gerris0": 10, "gerris4": 10}

# Total #INRs M across the default 2 windows at each field's recorded tau /
# absorb operating point (provenance in docs):
#   rfc        tau=0.05 absorb=0   -> N=[1,1]  M=2   (mainExp_2.3)
#   cylinder2d tau=0.1  absorb=0   -> N=[3,2]  M=5   (mainExp_2.3)
#   boussinesq tau=0.5  absorb=256 -> N=[1,1]  M=2   (Verify_tau_1.1 win point)
#   gerris0    tau=0.6  absorb=0   -> N=[4,5]  M=9   (Verify_gerristiny_1.1)
FIELD_M = {"rfc": 2, "cylinder2d": 5, "boussinesq": 2, "gerris0": 9}

DEFAULT_FRACS = (0.05, 0.10, 0.20)


def raw_bytes(shape) -> int:
    t, y, x, c = shape
    return t * y * x * c * 4


def side_info_bytes(shape, n_inrs: int, n_windows: int = 2, k_cell: int = 2,
                    use_observer: bool = True, observer: str = "tvfull") -> int:
    """Mirror of pipeline.run_proposed side-info accounting (planning estimate;
    exact when windows are equal length and regions spread evenly).

    Byte-accounting v2 (Verify_compresswin_1.3): windows whose partition has N == 1
    store NO label map (it is constant); the observer parameterization determines
    the killing-parameter bytes (tvfull Tw*3*4, tvtrans Tw*2*4, constfull 12,
    consttrans 8 per region) plus one global variant tag byte. The planning
    estimate assumes regions spread evenly over windows, so N == 1 per window
    exactly when n_inrs == n_windows."""
    t, y, x, _ = shape
    n_cells = math.ceil(y / k_cell) * math.ceil(x / k_cell)
    n_per_window = max(1, n_inrs // n_windows)
    labels = 0 if n_per_window == 1 else n_cells * 2 * n_windows
    tw = math.ceil(t / n_windows)
    obs_b = {"tvfull": tw * 3 * 4, "tvtrans": tw * 2 * 4,
             "constfull": 3 * 4, "consttrans": 2 * 4}[observer]
    killing = n_inrs * obs_b + 1 if use_observer else 0    # + variant tag byte
    per_region = n_inrs * ((4 + 4) * 4 + 2)                # xi bbox + v range + m_r
    return labels + killing + per_region


def plan_baseline(shape, frac: float, d: int) -> dict:
    xb = raw_bytes(shape)
    side = 10 * 4                                          # coord ranges + value lo/hi
    budget_p = int((frac * xb - side) // 4)
    m = pick_m_for_budget(budget_p, d)
    params = coordnet_num_params(m, d)
    total = params * 4 + side
    return {"frac": frac, "budget_bytes": int(frac * xb), "m": m, "d": d,
            "params": params, "total_bytes": total, "util": total / (frac * xb),
            "cr": xb / total, "over": total > frac * xb}


def plan_proposed(shape, frac: float, d: int, n_inrs: int, n_windows: int = 2,
                  k_cell: int = 2, use_observer: bool = True,
                  observer: str = "tvfull") -> dict:
    xb = raw_bytes(shape)
    side = side_info_bytes(shape, n_inrs, n_windows, k_cell, use_observer, observer)
    budget_p_total = int((frac * xb - side) // 4)
    share = budget_p_total // n_inrs
    m_r = pick_m_for_budget(share, d)
    params_r = coordnet_num_params(m_r, d)
    total = params_r * n_inrs * 4 + side
    return {"frac": frac, "budget_bytes": int(frac * xb), "side_bytes": side,
            "n_inrs": n_inrs, "share": share, "m_r": m_r, "d": d,
            "params_r": params_r, "params_total": params_r * n_inrs,
            "total_bytes": total, "util": total / (frac * xb), "cr": xb / total,
            "over": total > frac * xb}


def print_field(name: str, shape, fracs, d: int, n_inrs: int | None,
                n_windows: int, k_cell: int, d_sweep: bool) -> None:
    xb = raw_bytes(shape)
    print(f"\n=== {name}: shape (T,Y,X,C)={tuple(shape)}  raw float32 = {xb:,} B "
          f"({xb / 2**20:.2f} MiB)  [d={d}" +
          (f", proposed M={n_inrs} INRs]" if n_inrs else ", proposed M unknown]"))
    hdr = (f"{'frac':>5} | {'budget(B)':>11} | {'baseline m':>10} {'params':>9} "
           f"{'bytes':>9} {'util':>5} {'CR':>6}")
    if n_inrs:
        hdr += f" | {'pro m_r':>7} {'params_tot':>10} {'bytes':>9} {'util':>5} {'CR':>6}"
    print(hdr)
    for frac in fracs:
        b = plan_baseline(shape, frac, d)
        line = (f"{frac * 100:>4.1f}% | {b['budget_bytes']:>11,} | {b['m']:>10} "
                f"{b['params']:>9,} {b['total_bytes']:>9,} {b['util']:>5.1%} "
                f"{b['cr']:>5.1f}x")
        if n_inrs:
            p = plan_proposed(shape, frac, d, n_inrs, n_windows, k_cell)
            line += (f" | {p['m_r']:>7} {p['params_total']:>10,} "
                     f"{p['total_bytes']:>9,} {p['util']:>5.1%} {p['cr']:>5.1f}x")
            if p["over"]:
                line += "  !! OVER BUDGET (m_min floor)"
        if b["over"]:
            line += "  !! OVER BUDGET (m_min floor)"
        print(line)
    if d_sweep:
        ds = [2, 4, 6, 8, 10]
        print(f"  d-sweep (baseline width m the same budget buys; policy stays d={d}):")
        print("  " + f"{'frac':>5} | " + " | ".join(f"d={dd:<2}" + " " * 4 for dd in ds))
        for frac in fracs:
            budget_p = int((frac * xb - 40) // 4)
            cells = []
            for dd in ds:
                mm = pick_m_for_budget(budget_p, dd)
                cells.append(f"m={mm:<3} {coordnet_num_params(mm, dd) / budget_p:>4.0%}")
            print("  " + f"{frac * 100:>4.0f}% | " + " | ".join(cells))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--field", default="all",
                    help="field name, 'all' = rfc,cylinder2d,boussinesq (experiment "
                         "fields), or any key of FIELD_SHAPES")
    ap.add_argument("--shape", type=int, nargs=3, metavar=("T", "Y", "X"),
                    help="explicit grid shape instead of --field (C=2 assumed)")
    ap.add_argument("--fracs", default="0.05,0.10,0.20",
                    help="comma-separated budget fractions of the raw field bytes")
    ap.add_argument("--d", type=int, default=None,
                    help="depth override (default: field's frozen baseline depth)")
    ap.add_argument("--n_inrs", type=int, default=None,
                    help="total #INRs M for the proposed plan (default: recorded "
                         "partition size at the field's tau operating point)")
    ap.add_argument("--n_windows", type=int, default=2)
    ap.add_argument("--k_cell", type=int, default=2)
    ap.add_argument("--d_sweep", action="store_true",
                    help="print informational width/depth trade-off table")
    ap.add_argument("--load", action="store_true",
                    help="load the real field via pipeline.load_field and verify "
                         "the FIELD_SHAPES table entry")
    args = ap.parse_args()
    fracs = tuple(float(s) for s in args.fracs.split(",") if s.strip())
    assert all(0 < f < 1 for f in fracs), "fractions must be in (0, 1)"

    if args.shape:
        shape = (*args.shape, 2)
        d = args.d if args.d is not None else 4
        print_field("custom", shape, fracs, d, args.n_inrs, args.n_windows,
                    args.k_cell, args.d_sweep)
        return

    names = (["rfc", "cylinder2d", "boussinesq"] if args.field == "all"
             else [args.field])
    for name in names:
        if name not in FIELD_SHAPES:
            raise SystemExit(f"unknown field '{name}' -- add it to FIELD_SHAPES "
                             f"or use --shape T Y X")
        shape = FIELD_SHAPES[name]
        if args.load:
            from pipeline import load_field
            fd = load_field(name)
            assert tuple(fd.shape) == shape, \
                f"FIELD_SHAPES stale for {name}: table {shape} vs loaded {fd.shape}"
            print(f"[{name}] loaded data confirms shape {tuple(fd.shape)}, "
                  f"{fd.data_bytes():,} B")
        d = args.d if args.d is not None else FIELD_D[name]
        n_inrs = args.n_inrs if args.n_inrs is not None else FIELD_M.get(name)
        print_field(name, shape, fracs, d, n_inrs, args.n_windows, args.k_cell,
                    args.d_sweep)


if __name__ == "__main__":
    main()
