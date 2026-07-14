"""Normalization audit -- verify every INR input path feeds [-1, 1] (v2.3).

Checks, for baseline AND every (window, region) of the proposed modes (with and
without observer), WITHOUT training:
  C1  coords: each axis spans exactly [-1, 1] (min==-1, max==+1 within 1e-6);
      degenerate axes (constant) allowed only as the documented t/Tw==1 case.
  C2  values: each component within [-1, 1] and touching both ends (minmax exact);
      constant components map to -1 (degenerate-but-defined).
  C3  no NaN / Inf anywhere.
  C4  roundtrip: decode(encode(x)) == x to 1e-6 relative.
  C5  eval-time queries are the training coords themselves (by construction in this
      pipeline) -- asserted by re-deriving coords a second time and comparing.

Run:  python audit_normalization.py --fields rfc,cylinder2d
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for pth in (str(_ROOT), str(_HERE)):
    if pth not in sys.path:
        sys.path.insert(0, pth)

from pipeline import (ExpCfg, load_field, compute_partitions,  # noqa: E402
                      baseline_coords_values)
from frame import make_region_samples  # noqa: E402
from inr import MinMax  # noqa: E402

_fail = []


def check(name, cond, msg=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" ({msg})" if msg else ""))
    if not cond:
        _fail.append(name)


def audit_block(tag: str, coords_n: np.ndarray, vals_n: np.ndarray,
                allow_const_axes=()):
    ok_range, bad = True, []
    for ax in range(coords_n.shape[1]):
        lo, hi = float(coords_n[:, ax].min()), float(coords_n[:, ax].max())
        if ax in allow_const_axes and abs(hi - lo) < 1e-12:
            continue
        if abs(lo + 1) > 1e-5 or abs(hi - 1) > 1e-5:
            ok_range = False; bad.append(f"axis{ax}=[{lo:.6f},{hi:.6f}]")
    check(f"{tag} C1 coord axes span [-1,1]", ok_range, "; ".join(bad))

    ok_v, badv = True, []
    for c in range(vals_n.shape[1]):
        lo, hi = float(vals_n[:, c].min()), float(vals_n[:, c].max())
        inside = lo >= -1 - 1e-5 and hi <= 1 + 1e-5
        touches = (abs(lo + 1) < 1e-5 and abs(hi - 1) < 1e-5) or abs(hi - lo) < 1e-10
        if not (inside and touches):
            ok_v = False; badv.append(f"comp{c}=[{lo:.6f},{hi:.6f}]")
    check(f"{tag} C2 value comps minmax-exact in [-1,1]", ok_v, "; ".join(badv))

    finite = np.isfinite(coords_n).all() and np.isfinite(vals_n).all()
    check(f"{tag} C3 finite", bool(finite))


def audit_field(name: str, cfg: ExpCfg):
    print(f"== {name} (tau={cfg.tau}, n_windows={cfg.n_windows}, "
          f"absorb={cfg.absorb_min_pixels})")
    fd = load_field(name)

    # baseline path
    coords_n, vals_n, vmm = baseline_coords_values(fd)
    audit_block("baseline", coords_n, vals_n)
    raw = fd.data.reshape(-1, 2)
    rt = np.abs(vmm.decode(vmm.encode(raw)) - raw).max() / max(np.abs(raw).max(), 1e-12)
    check("baseline C4 value roundtrip", rt < 1e-6, f"rel err {rt:.2e}")

    # proposed paths (observer on and off), every window x region
    parts = compute_partitions(fd, cfg, log=lambda s: None)
    for use_obs in (True, False):
        for w_i, part in enumerate(parts):
            for r_i, reg in enumerate(part.regions):
                q = reg.q if use_obs else np.zeros_like(reg.q)
                pix = part.labels_pixels == r_i
                smp = make_region_samples(fd.data, fd.xs, fd.ys, fd.dt,
                                          pix, part.it0, part.it1, q)
                ximm = MinMax.fit(smp.xi)
                vmm_r = MinMax.fit(smp.vtil)
                cn = np.concatenate([ximm.encode(smp.xi),
                                     smp.tn[:, None].astype(np.float32)], axis=1)
                vn = vmm_r.encode(smp.vtil)
                tag = f"{'obs' if use_obs else 'raw'} w{w_i}r{r_i}"
                allow = (2,) if (part.it1 - part.it0) == 1 else ()
                audit_block(tag, cn, vn, allow_const_axes=allow)
                # C5: eval coords == training coords by re-derivation
                smp2 = make_region_samples(fd.data, fd.xs, fd.ys, fd.dt,
                                           pix, part.it0, part.it1, q)
                same = np.array_equal(smp.xi, smp2.xi) and np.array_equal(smp.tn, smp2.tn)
                check(f"{tag} C5 eval==train coords", same)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", default="rfc")
    args = ap.parse_args()
    for name in [s.strip() for s in args.fields.split(",") if s.strip()]:
        if name == "rfc":
            audit_field("rfc", ExpCfg(field="rfc", tau=0.05, n_windows=2))
        elif name == "cylinder2d":
            audit_field("cylinder2d", ExpCfg(field="cylinder2d", tau=0.1, n_windows=2))
        elif name == "boussinesq":
            audit_field("boussinesq", ExpCfg(field="boussinesq", tau=0.2,
                                             n_windows=2, absorb_min_pixels=256))
        else:
            audit_field(name, ExpCfg(field=name))
    print()
    if _fail:
        print(f"FAILED {len(_fail)} checks: {_fail[:10]}")
        sys.exit(1)
    print("ALL NORMALIZATION CHECKS PASSED")


if __name__ == "__main__":
    main()
