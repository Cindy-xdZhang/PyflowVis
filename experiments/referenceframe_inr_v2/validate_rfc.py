"""Correctness validation suite -- v2 (docs/referenceframe_inr_v2.md par.3).

All four tests must pass before any training experiment is trusted. No INR training
here -- pure least-squares / transform checks, CPU, runs in seconds.

  T1  manufactured rotating-frame field: recover (a,b,c) in closed form; pushforward
      of the solved observer reproduces the analytic steady field.
  T2  repository RFC (rotation_four_center, al_t=1 -> camera term = -x_perp,
      prediction c_true = -1): tau-merge yields N=1 for every window layout.
  T3  transform roundtrip is exact with a perfect-oracle INR.
  T4  two-rotor counter-control: does NOT collapse to N=1; per-half observers correct.

Run:  python validate_rfc.py
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

import synth  # noqa: E402
from killing2d import compute_cell_stats, region_solve, solve_killing, \
    solve_killing_trans  # noqa: E402
from partition import merge_partition, split_windows  # noqa: E402
from frame import make_region_samples, scatter_reconstruction, rot  # noqa: E402

PASS, FAIL = "PASS", "FAIL"
_failures = []


def check(name: str, cond: bool, msg: str = ""):
    status = PASS if cond else FAIL
    print(f"  [{status}] {name}" + (f"  ({msg})" if msg else ""))
    if not cond:
        _failures.append(name)


# ---------------------------------------------------------------------------
def _t1_solve(res_xy: int, res_t: int, omega0: float, c0):
    xs = np.linspace(-2, 2, res_xy)
    ys = np.linspace(-2, 2, res_xy)
    ts = np.linspace(0, 2 * np.pi, res_t)
    data = synth.compose_rotating_frame(synth.four_cell_steady, omega0, c0, xs, ys, ts)
    dt = float(ts[1] - ts[0])
    stats = compute_cell_stats(data, xs, ys, dt, k=2, boundary_skip=2)
    nCy, nCx = stats.n_cells
    q, E, E0 = region_solve(stats, np.arange(nCy * nCx), 0, len(ts))
    return q, E / E0, (data, xs, ys, ts, dt)


def t1_manufactured():
    print("T1: manufactured steady-pattern + rotating camera (closed-form answer)")
    omega0, c0 = 0.7, (0.3, -0.2)
    truth = np.array(synth.true_killing_params(omega0, c0))

    # discretization-convergence check: finite-difference error must shrink with
    # resolution and the fine solve must hit the closed-form params
    q_c, ratio_c, coarse = _t1_solve(96, 64, omega0, c0)
    q_f, ratio_f, _ = _t1_solve(192, 128, omega0, c0)
    err_c = np.abs(q_c.mean(axis=0) - truth).max()
    err_f = np.abs(q_f.mean(axis=0) - truth).max()
    check("killing solve converges to closed form", err_f < 6e-3 and err_f < 0.6 * err_c,
          f"max|q-q*|: coarse={err_c:.2e} -> fine={err_f:.2e}; "
          f"fine mean={q_f.mean(axis=0).round(4)} vs true={truth.round(4)}")
    check("residual ratio E/E0 -> 0 with resolution",
          ratio_f < 6e-4 and ratio_f < 0.6 * ratio_c,
          f"E/E0: coarse={ratio_c:.2e} -> fine={ratio_f:.2e}")

    data, xs, ys, ts, dt = coarse
    a_t, b_t, c_t = truth
    q = q_c

    # Pushforward steadiness. With the window anchored at t0 = 0 (frame == identity
    # there), the observed field of THIS construction is exactly the time-independent
    # function vtil(xi, t) = s(xi)  (derivation: v - u = R(-w t) s(F_t(xi)) and the
    # frame rotation cancels R(-w t)). So every sample, at every t, must match s(xi).
    it0, it1 = 0, len(ts)
    pix = np.zeros((len(ys), len(xs)), dtype=bool)
    pix[8:-8, 8:-8] = True
    q_exact = np.tile([a_t, b_t, c_t], (it1 - it0, 1)).astype(np.float64)
    smp = make_region_samples(data, xs, ys, dt, pix, it0, it1, q_exact)
    expected = synth.four_cell_steady(smp.xi[:, 0], smp.xi[:, 1])
    err_pf = np.abs(smp.vtil - expected).max()
    check("pushforward(exact q) == steady s(xi) for all t", err_pf < 5e-2,
          f"max|vtil - s|={err_pf:.2e} (trapezoid frame-integration error)")

    # same through the actual pipeline path: pushforward with the SOLVED q
    smp_s = make_region_samples(data, xs, ys, dt, pix, it0, it1, q)
    expected_s = synth.four_cell_steady(smp_s.xi[:, 0], smp_s.xi[:, 1])
    err_pf_s = np.abs(smp_s.vtil - expected_s).max()
    check("pushforward(solved q) == steady s(xi) for all t", err_pf_s < 0.15,
          f"max|vtil - s|={err_pf_s:.2e} (adds finite-difference LSQ error)")


# ---------------------------------------------------------------------------
def t2_rfc():
    print("T2: repository RFC (rotation_four_center) -> N must be 1, c ~ -1.0")
    from FLowUtils.AnalyticalFlowCreator import rotation_four_center
    f = rotation_four_center((64, 64), 64)
    data = np.asarray(f.field, dtype=np.float64)
    xs = np.linspace(f.domainMinBoundary[0], f.domainMaxBoundary[0], 64)
    ys = np.linspace(f.domainMinBoundary[1], f.domainMaxBoundary[1], 64)
    dt = (f.tmax - f.tmin) / (64 - 1)

    stats = compute_cell_stats(data, xs, ys, dt, k=2, boundary_skip=2)
    nCy, nCx = stats.n_cells
    q, E, E0 = region_solve(stats, np.arange(nCy * nCx), 0, 64)
    cbar = q[:, 2].mean()
    ab_mean = np.abs(q[:, :2]).mean()
    check("global observer c(t) ~ -1.0 (= -al_t)", abs(cbar + 1.0) < 5e-2,
          f"mean c={cbar:.4f}")
    check("global observer translation ~ 0", ab_mean < 5e-2, f"|a|,|b| mean={ab_mean:.2e}")
    check("RFC globally steadied: E/E0 << 1", E / E0 < 1e-2, f"E/E0={E / E0:.2e}")

    for n_windows in (2, 4):
        for tau in (0.01, 0.05, 0.2):
            Ns = []
            for (it0, it1) in split_windows(64, n_windows):
                part = merge_partition(stats, it0, it1, tau)
                Ns.append(part.n_regions)
            check(f"N==1 for n_windows={n_windows}, tau={tau}", all(n == 1 for n in Ns),
                  f"N per window={Ns}")


# ---------------------------------------------------------------------------
def t3_roundtrip():
    print("T3: pushforward/inverse roundtrip exact with a perfect-oracle INR")
    rng = np.random.default_rng(7)
    T, Y, X = 12, 40, 48
    xs = np.linspace(-1.5, 2.0, X)
    ys = np.linspace(-2.0, 1.0, Y)
    dt = 0.13
    data = rng.normal(size=(T, Y, X, 2))
    q = rng.normal(scale=0.7, size=(T, 3))     # arbitrary observer params
    pix = rng.random((Y, X)) < 0.6             # arbitrary scattered region
    smp = make_region_samples(data, xs, ys, dt, pix, 0, T, q)
    recon = np.full(data.shape, np.nan)
    scatter_reconstruction(recon, smp, smp.vtil)   # oracle: predictions == targets
    m = np.isfinite(recon[:, pix])
    err = np.abs(recon[:, pix] - data[:, pix]).max()
    check("v_hat == v to 1e-10", bool(m.all()) and err < 1e-10, f"max err={err:.2e}")


# ---------------------------------------------------------------------------
def t4_two_rotor():
    print("T4: two-rotor counter-control -> N >= 2, per-half observers correct")
    xs = np.linspace(-2, 2, 96)
    ys = np.linspace(-2, 2, 96)
    ts = np.linspace(0, 2 * np.pi, 64)
    omega1, omega2 = 0.8, -1.4
    data = synth.two_rotor_field(xs, ys, ts, omega1, omega2)
    dt = float(ts[1] - ts[0])
    stats = compute_cell_stats(data, xs, ys, dt, k=2, boundary_skip=2)
    Xg, Yg = np.meshgrid(xs, ys)

    def annulus(cx, cy, r0=0.15, r1=0.75):
        rr = np.hypot(Xg - cx, Yg - cy)
        return (rr > r0) & (rr < r1)

    # the unsteady signal lives in the annuli around each rotor core (the cores
    # themselves are locally ~rigid, hence ~steady: any observer explains them)
    mL, mR = annulus(-1.0, 0.0), annulus(1.0, 0.0)
    for tau in (0.01, 0.05):
        part = merge_partition(stats, 0, 32, tau)
        N = part.n_regions
        lbl = part.labels_pixels
        rL = int(np.bincount(lbl[mL].ravel()).argmax())   # majority label per annulus
        rR = int(np.bincount(lbl[mR].ravel()).argmax())
        cL = part.regions[rL].q[:, 2].mean()
        cR = part.regions[rR].q[:, 2].mean()
        check(f"tau={tau}: halves not merged (N={N})", N >= 2 and rL != rR,
              f"majority labels L={rL}, R={rR}")
        ok_obs = abs(cL + omega1) < 0.1 * max(1, abs(omega1)) and \
                 abs(cR + omega2) < 0.1 * max(1, abs(omega2))
        check(f"tau={tau}: per-half c ~ (-w1, -w2)", ok_obs,
              f"cL={cL:.3f} (want {-omega1}), cR={cR:.3f} (want {-omega2})")


# ---------------------------------------------------------------------------
def t5_observer_variants():
    """T5 (Verify_compresswin_1.3): observer parameterization variants.

    Manufactured uniformly-translating field v = s(x - ab0 t) + ab0: the
    translation-only observer (a, b) = ab0, c = 0 steadies it exactly, so
      - consttrans / tvtrans must recover ab0 with E/E0 << 1,
      - constfull must recover (ab0, c ~ 0),
    and on the T1 rotating-frame field the const-full observer must match the
    (time-invariant) tv-full solution while translation-only must FAIL (E/E0 ~ 1)
    -- rotation cannot be explained by translation."""
    print("T5: observer variants (translation-only / time-constant)")
    xs = np.linspace(-2, 2, 96)
    ys = np.linspace(-2, 2, 96)
    ts = np.linspace(0, 2.0, 48)
    ab0 = (0.4, -0.25)
    s_fn = synth.gauss_vortex_steady(-0.3, 0.2, sigma=0.45, strength=1.5)
    data = synth.compose_translating_frame(s_fn, ab0, xs, ys, ts)
    dt = float(ts[1] - ts[0])
    stats = compute_cell_stats(data, xs, ys, dt, k=2, boundary_skip=2)
    AtA = stats.AtA.sum(axis=(1, 2))
    g = stats.g.sum(axis=(1, 2))
    e0 = stats.e0.sum(axis=(1, 2))
    E0 = float(e0.sum())

    q_ct, _ = solve_killing_trans(AtA.sum(0), g.sum(0), e0.sum())
    E_ct = float(e0.sum() + q_ct @ g.sum(0))
    err_ct = np.abs(q_ct[:2] - np.asarray(ab0)).max()
    check("consttrans recovers (a,b) = ab0", err_ct < 5e-3,
          f"q=({q_ct[0]:+.4f}, {q_ct[1]:+.4f}) vs {ab0}, err={err_ct:.2e}")
    check("consttrans steadies the field: E/E0 << 1", E_ct / E0 < 1e-2,
          f"E/E0={E_ct / E0:.2e}")

    q_tv, E_tv = solve_killing_trans(AtA, g, e0)
    err_tv = np.abs(q_tv[:, :2] - np.asarray(ab0)).max()
    check("tvtrans per-timestep (a,b) ~ ab0", err_tv < 1e-2,
          f"max err={err_tv:.2e}; E/E0={float(E_tv.sum()) / E0:.2e}")

    q_cf, _ = solve_killing(AtA.sum(0), g.sum(0), e0.sum())
    check("constfull recovers (ab0, c ~ 0)",
          np.abs(q_cf[:2] - np.asarray(ab0)).max() < 5e-3 and abs(q_cf[2]) < 5e-3,
          f"q=({q_cf[0]:+.4f}, {q_cf[1]:+.4f}, {q_cf[2]:+.2e})")

    # counter-control on the T1 rotating field: translation-only cannot explain it
    omega0, c0 = 0.7, (0.3, -0.2)
    xs2 = np.linspace(-2, 2, 96)
    ts2 = np.linspace(0, 2 * np.pi, 64)
    data2 = synth.compose_rotating_frame(synth.four_cell_steady, omega0, c0, xs2, xs2, ts2)
    st2 = compute_cell_stats(data2, xs2, xs2, float(ts2[1] - ts2[0]), k=2, boundary_skip=2)
    A2, g2, e2 = st2.AtA.sum(axis=(1, 2)), st2.g.sum(axis=(1, 2)), st2.e0.sum(axis=(1, 2))
    E0_2 = float(e2.sum())
    truth = np.array(synth.true_killing_params(omega0, c0))
    q_cf2, _ = solve_killing(A2.sum(0), g2.sum(0), e2.sum())
    E_cf2 = float(e2.sum() + q_cf2 @ g2.sum(0))
    check("rotating field: constfull == closed-form observer",
          np.abs(q_cf2 - truth).max() < 2e-2 and E_cf2 / E0_2 < 1e-2,
          f"q={q_cf2.round(4)} vs true={truth.round(4)}, E/E0={E_cf2 / E0_2:.2e}")
    q_ct2, _ = solve_killing_trans(A2.sum(0), g2.sum(0), e2.sum())
    E_ct2 = float(e2.sum() + q_ct2 @ g2.sum(0))
    check("rotating field: translation-only must fail (E/E0 ~ 1)",
          E_ct2 / E0_2 > 0.5, f"E/E0={E_ct2 / E0_2:.2f}")


if __name__ == "__main__":
    t1_manufactured()
    t2_rfc()
    t3_roundtrip()
    t4_two_rotor()
    t5_observer_variants()
    print()
    if _failures:
        print(f"FAILED: {len(_failures)} check(s): {_failures}")
        sys.exit(1)
    print("ALL VALIDATION CHECKS PASSED")
