"""Diagnostic (no training): observer-variant dry run for the boussinesq 2.5% cell.

Question (Verify_compresswin_1.3 design): the time-varying full killing observer
(w1M1 arm) loses to baseline at 2.5%/5% by 1-3 dB. Candidate fixes keep M=1 /
single window but change the observer's degrees of freedom:

  tv-full     per-timestep (a,b,c)(t)    -- current w1M1
  tv-trans    per-timestep (a,b)(t), c=0 -- no rotation sweep
  const-full  one (a,b,c) for the whole window (joint LS over all timesteps)
  const-trans one (a,b), c=0             -- uniformly translating frame
              (Taylor frozen-turbulence form)

For each variant this prints E/E0 (fraction of raw temporal energy NOT explained)
and the xi-bbox sweep inflation (union bbox over t of the transformed domain
rectangle / original domain area) -- the two sides of the win condition:
explained energy (want E/E0 low) vs coordinate sweep cost (want inflation ~1).

Also prints the tv-full q(t) trajectory stats to see what the observer actually
does on the plume, and the same table for rfc as a sanity anchor (expect c ~ -1,
E/E0 ~ 3e-4, and const-full ~ tv-full since its observer is time-invariant).

Run locally (CPU ok, ~1 min):  python diag_agent_observer_variants.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for pth in (str(_ROOT), str(_HERE)):
    if pth not in sys.path:
        sys.path.insert(0, pth)

from killing2d import compute_cell_stats, solve_killing  # noqa: E402
from frame import integrate_frame, rot  # noqa: E402
from pipeline import load_field  # noqa: E402


def solve_trans(AtA, g, e0):
    """2-DOF (a,b) LS with c fixed to 0: use the leading 2x2 block."""
    A2 = AtA[..., :2, :2]
    g2 = g[..., :2]
    tr = np.trace(A2, axis1=-2, axis2=-1)
    lam = 1e-12 * (tr / 2.0) + 1e-300
    M = A2 + lam[..., None, None] * np.eye(2)
    q2 = np.linalg.solve(M, -g2[..., None])[..., 0]
    E = e0 + np.einsum("...i,...i->...", q2, g2)
    q = np.zeros(g.shape[:-1] + (3,))
    q[..., :2] = q2
    return q, np.maximum(E, 0.0)


def variant_qE(AtA_t, g_t, e0_t, kind):
    """Return (q (Tw,3), E_total) for one observer variant."""
    Tw = AtA_t.shape[0]
    if kind == "tv-full":
        q, E = solve_killing(AtA_t, g_t, e0_t)
        return q, float(E.sum())
    if kind == "tv-trans":
        q, E = solve_trans(AtA_t, g_t, e0_t)
        return q, float(E.sum())
    if kind == "const-full":
        qc, _ = solve_killing(AtA_t.sum(0), g_t.sum(0), e0_t.sum())
        q = np.tile(qc, (Tw, 1))
        E = e0_t.sum() + float(qc @ g_t.sum(0))
        return q, max(E, 0.0)
    if kind == "const-trans":
        qc, _ = solve_trans(AtA_t.sum(0), g_t.sum(0), e0_t.sum())
        q = np.tile(qc[0] if qc.ndim > 1 else qc, (Tw, 1))
        E = e0_t.sum() + float(q[0] @ g_t.sum(0))
        return q, max(E, 0.0)
    raise ValueError(kind)


def bbox_inflation(q, dt, xs, ys):
    """Union-over-t xi-bbox area of the transformed domain rectangle / lab area."""
    theta, D = integrate_frame(q, dt)
    corners = np.array([[xs[0], ys[0]], [xs[0], ys[-1]],
                        [xs[-1], ys[0]], [xs[-1], ys[-1]]])
    lo = np.full(2, np.inf)
    hi = np.full(2, -np.inf)
    for i in range(q.shape[0]):
        xi = corners @ rot(-theta[i]).T - D[i]
        lo = np.minimum(lo, xi.min(0))
        hi = np.maximum(hi, xi.max(0))
    area_lab = (xs[-1] - xs[0]) * (ys[-1] - ys[0])
    return float(np.prod(hi - lo) / area_lab)


def report(field_name):
    fd = load_field(field_name)
    T = fd.shape[0]
    print(f"\n==== {field_name}  shape={fd.shape}  dt={fd.dt:.4f} ====")
    stats = compute_cell_stats(fd.data, fd.xs, fd.ys, fd.dt, k=2, boundary_skip=2)
    # global region = all cells summed, over window layouts 1w and 2w
    AtA_all = stats.AtA.sum(axis=(1, 2))    # (T, 3, 3)
    g_all = stats.g.sum(axis=(1, 2))        # (T, 3)
    e0_all = stats.e0.sum(axis=(1, 2))      # (T,)

    for (t0, t1), tag in [((0, T), "1 window"),
                          ((0, T // 2), "win A"), ((T // 2, T), "win B")]:
        A, g, e0 = AtA_all[t0:t1], g_all[t0:t1], e0_all[t0:t1]
        E0 = float(e0.sum())
        print(f"-- {tag} [{t0},{t1})  E0={E0:.4e}")
        for kind in ("tv-full", "tv-trans", "const-full", "const-trans"):
            q, E = variant_qE(A, g, e0, kind)
            infl = bbox_inflation(q, fd.dt, fd.xs, fd.ys)
            extra = ""
            if kind == "tv-full":
                th, D = integrate_frame(q, fd.dt)
                extra = (f"  c(t): mean={q[:, 2].mean():+.3f} std={q[:, 2].std():.3f}"
                         f"  |theta|max={np.abs(th).max():.3f} rad"
                         f"  |D|max={np.linalg.norm(D, axis=1).max():.3f}"
                         f"  a: {q[:, 0].mean():+.3f}+-{q[:, 0].std():.3f}"
                         f"  b: {q[:, 1].mean():+.3f}+-{q[:, 1].std():.3f}")
            if kind == "const-full":
                extra = f"  q=({q[0, 0]:+.4f}, {q[0, 1]:+.4f}, {q[0, 2]:+.4f})"
            if kind == "const-trans":
                extra = f"  q=({q[0, 0]:+.4f}, {q[0, 1]:+.4f})"
            print(f"   {kind:11s} E/E0={E / max(E0, 1e-300):.4e}  bbox_infl={infl:6.3f}{extra}")


if __name__ == "__main__":
    report("boussinesq")
    report("rfc")
