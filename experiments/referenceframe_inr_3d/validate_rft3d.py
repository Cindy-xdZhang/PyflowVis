"""Correctness validation suite -- referenceframe_inr_3d.

All checks must pass before any 3D training experiment is trusted.  No INR
training here -- pure least-squares / transform checks, CPU, runs in minutes.

  T1  manufactured 3D rotating-frame field (oblique axis): recover (t_vec, w) in
      closed form; pushforward of the exact/solved observer is steady.
  T2  manufactured translating field: consttrans recovers c_vec, constfull
      recovers (c_vec, w ~ 0); on the rotating field translation-only must FAIL.
  T3  pushforward/inverse roundtrip exact with a perfect-oracle INR.
  T4  two-rotor counter-control (different axes AND rates): tau-merge must NOT
      collapse to N=1; per-half w correct.
  T5  z-trivial embedding pins the 3D solvers/frame to the frozen, validated 2D
      module: killing3d == killing2d, integrate_frame_3d == integrate_frame,
      pullback samples match component-wise.

Run:  python validate_rft3d.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
_V2 = _ROOT / "experiments" / "referenceframe_inr_v2"
for pth in (str(_ROOT), str(_HERE), str(_V2)):
    if pth not in sys.path:
        sys.path.insert(0, pth)

import synth3d  # noqa: E402
from killing3d import compute_cell_stats_3d, region_solve_3d, solve_killing_3d, \
    solve_killing_trans_3d  # noqa: E402
from partition3d import merge_partition_3d  # noqa: E402
from frame3d import make_region_samples_3d, scatter_reconstruction_3d, \
    integrate_frame_3d  # noqa: E402

PASS, FAIL = "PASS", "FAIL"
_failures = []


def check(name: str, cond: bool, msg: str = ""):
    status = PASS if cond else FAIL
    print(f"  [{status}] {name}" + (f"  ({msg})" if msg else ""))
    if not cond:
        _failures.append(name)


# ---------------------------------------------------------------------------
def _t1_solve(res_xyz: int, res_t: int, omega0: float, axis, c0):
    xs = np.linspace(-2, 2, res_xyz)
    ys = np.linspace(-2, 2, res_xyz)
    zs = np.linspace(-2, 2, res_xyz)
    ts = np.linspace(0, 2 * np.pi, res_t)
    data = synth3d.compose_rotating_frame_3d(synth3d.cells_steady_3d, omega0,
                                             axis, c0, xs, ys, zs, ts)
    dt = float(ts[1] - ts[0])
    stats = compute_cell_stats_3d(data, xs, ys, zs, dt, k=2, boundary_skip=2)
    nC = stats.n_cells
    q, E, E0 = region_solve_3d(stats, np.arange(nC[0] * nC[1] * nC[2]), 0, len(ts))
    return q, E / E0, (data, xs, ys, zs, ts, dt)


def t1_manufactured():
    print("T1: manufactured steady 3D pattern + oblique-axis rotating camera")
    omega0 = 0.7
    axis = (0.3, 0.5, 0.8)
    c0 = (0.3, -0.2, 0.1)
    truth = synth3d.true_killing_params_3d(omega0, axis, c0)

    q_c, ratio_c, coarse = _t1_solve(40, 32, omega0, axis, c0)
    q_f, ratio_f, fine = _t1_solve(80, 64, omega0, axis, c0)
    err_c = np.abs(q_c.mean(axis=0) - truth).max()
    err_f = np.abs(q_f.mean(axis=0) - truth).max()
    check("killing solve converges to closed form",
          err_f < 2e-2 and err_f < 0.6 * err_c,
          f"max|q-q*|: coarse={err_c:.2e} -> fine={err_f:.2e}; "
          f"fine mean={q_f.mean(axis=0).round(4)} vs true={truth.round(4)}")
    check("residual ratio E/E0 -> 0 with resolution",
          ratio_f < 5e-3 and ratio_f < 0.6 * ratio_c,
          f"E/E0: coarse={ratio_c:.2e} -> fine={ratio_f:.2e}")

    data, xs, ys, zs, ts, dt = coarse
    it0, it1 = 0, len(ts)
    pix = np.zeros((len(zs), len(ys), len(xs)), dtype=bool)
    pix[6:-6, 6:-6, 6:-6] = True
    q_exact = np.tile(truth, (it1 - it0, 1)).astype(np.float64)
    smp = make_region_samples_3d(data, xs, ys, zs, dt, pix, it0, it1, q_exact)
    expected = synth3d.cells_steady_3d(smp.xi[:, 0], smp.xi[:, 1], smp.xi[:, 2])
    err_pf = np.abs(smp.vtil - expected).max()
    check("pushforward(exact q) == steady s(xi) for all t", err_pf < 5e-2,
          f"max|vtil - s|={err_pf:.2e} (frame-integration error)")

    # solved-q pushforward drifts linearly with the LSQ parameter error times
    # the window length (coarse: 2.5e-2 * 2pi * r ~ 0.3), so this check runs on
    # the FINE solve/data (q err 6e-3 -> expected drift < 0.1).  The mask is
    # strided: drift is per-point, so subsampling only saves memory.
    dataf, xsf, ysf, zsf, tsf, dtf = fine
    pixf = np.zeros((len(zsf), len(ysf), len(xsf)), dtype=bool)
    pixf[8:-8:2, 8:-8:2, 8:-8:2] = True
    smp_s = make_region_samples_3d(dataf, xsf, ysf, zsf, dtf, pixf,
                                   0, len(tsf), q_f)
    expected_s = synth3d.cells_steady_3d(smp_s.xi[:, 0], smp_s.xi[:, 1],
                                         smp_s.xi[:, 2])
    err_pf_s = np.abs(smp_s.vtil - expected_s).max()
    check("pushforward(solved q, fine) == steady s(xi) for all t",
          err_pf_s < 0.15,
          f"max|vtil - s|={err_pf_s:.2e} (adds finite-difference LSQ error)")


# ---------------------------------------------------------------------------
def t2_translation():
    print("T2: translating field -> consttrans/constfull recover c_vec")
    xs = ys = zs = np.linspace(-2, 2, 64)
    ts = np.linspace(0, 2.0, 32)
    c_vec = (0.4, -0.25, 0.15)
    s_fn = synth3d.gauss_swirl_steady_3d((-0.3, 0.2, 0.1), (0.2, 0.9, 0.4),
                                         sigma=0.45, strength=1.5)
    data = synth3d.compose_translating_frame_3d(s_fn, c_vec, xs, ys, zs, ts)
    dt = float(ts[1] - ts[0])
    stats = compute_cell_stats_3d(data, xs, ys, zs, dt, k=2, boundary_skip=2)
    AtA = stats.AtA.sum(axis=(1, 2, 3))
    g = stats.g.sum(axis=(1, 2, 3))
    e0 = stats.e0.sum(axis=(1, 2, 3))
    E0 = float(e0.sum())

    q_ct, _ = solve_killing_trans_3d(AtA.sum(0), g.sum(0), e0.sum())
    E_ct = float(e0.sum() + q_ct[:3] @ g.sum(0)[:3])
    err_ct = np.abs(q_ct[:3] - np.asarray(c_vec)).max()
    check("consttrans recovers t_vec = c_vec", err_ct < 1e-2,
          f"q={q_ct[:3].round(4)} vs {c_vec}, err={err_ct:.2e}")
    check("consttrans steadies the field: E/E0 << 1", E_ct / E0 < 1e-2,
          f"E/E0={E_ct / E0:.2e}")

    q_cf, _ = solve_killing_3d(AtA.sum(0), g.sum(0), e0.sum())
    check("constfull recovers (c_vec, w ~ 0)",
          np.abs(q_cf[:3] - np.asarray(c_vec)).max() < 1e-2
          and np.abs(q_cf[3:]).max() < 1e-2,
          f"t_vec={q_cf[:3].round(4)}, |w|max={np.abs(q_cf[3:]).max():.2e}")

    # counter-control: translation-only cannot explain the T1 rotating field
    omega0, axis, c0 = 0.7, (0.3, 0.5, 0.8), (0.3, -0.2, 0.1)
    xs2 = np.linspace(-2, 2, 40)
    ts2 = np.linspace(0, 2 * np.pi, 32)
    data2 = synth3d.compose_rotating_frame_3d(synth3d.cells_steady_3d, omega0,
                                              axis, c0, xs2, xs2, xs2, ts2)
    st2 = compute_cell_stats_3d(data2, xs2, xs2, xs2, float(ts2[1] - ts2[0]),
                                k=2, boundary_skip=2)
    A2 = st2.AtA.sum(axis=(1, 2, 3)); g2 = st2.g.sum(axis=(1, 2, 3))
    e2 = st2.e0.sum(axis=(1, 2, 3))
    E0_2 = float(e2.sum())
    q_ct2, _ = solve_killing_trans_3d(A2.sum(0), g2.sum(0), e2.sum())
    E_ct2 = float(e2.sum() + q_ct2[:3] @ g2.sum(0)[:3])
    check("rotating field: translation-only must fail (E/E0 ~ 1)",
          E_ct2 / E0_2 > 0.5, f"E/E0={E_ct2 / E0_2:.2f}")


# ---------------------------------------------------------------------------
def t3_roundtrip():
    print("T3: pushforward/inverse roundtrip exact with a perfect-oracle INR")
    rng = np.random.default_rng(7)
    T, Z, Y, X = 10, 18, 24, 20
    xs = np.linspace(-1.5, 2.0, X)
    ys = np.linspace(-2.0, 1.0, Y)
    zs = np.linspace(-0.8, 1.2, Z)
    dt = 0.13
    data = rng.normal(size=(T, Z, Y, X, 3))
    q = rng.normal(scale=0.7, size=(T, 6))     # arbitrary observer params
    pix = rng.random((Z, Y, X)) < 0.6          # arbitrary scattered region
    smp = make_region_samples_3d(data, xs, ys, zs, dt, pix, 0, T, q)
    recon = np.full(data.shape, np.nan)
    scatter_reconstruction_3d(recon, smp, smp.vtil)   # oracle: pred == target
    m = np.isfinite(recon[:, pix])
    err = np.abs(recon[:, pix] - data[:, pix]).max()
    check("v_hat == v to 1e-10", bool(m.all()) and err < 1e-10,
          f"max err={err:.2e}")


# ---------------------------------------------------------------------------
def t4_two_rotor():
    print("T4: 3D two-rotor (different axes) -> N >= 2, per-half w correct")
    xs = ys = zs = np.linspace(-2, 2, 48)
    ts = np.linspace(0, 2 * np.pi, 32)
    omega1, omega2 = 0.8, -1.4
    data = synth3d.two_rotor_field_3d(xs, ys, zs, ts, omega1, omega2)
    dt = float(ts[1] - ts[0])
    # k=3 (4096 cells): the counter-control verdict is granularity-independent
    # and k=2 (13.8k cells) made the merge the suite's only multi-hour step
    stats = compute_cell_stats_3d(data, xs, ys, zs, dt, k=3, boundary_skip=2)
    Zg, Yg, Xg = np.meshgrid(zs, ys, xs, indexing="ij")

    def shell(cx, r0=0.15, r1=0.75):
        rr = np.sqrt((Xg - cx) ** 2 + Yg ** 2 + Zg ** 2)
        return (rr > r0) & (rr < r1)

    mL, mR = shell(-1.0), shell(1.0)
    w1_true = np.array([0.0, 0.0, -omega1])    # w = -omega0 * axis
    w2_true = np.array([-omega2, 0.0, 0.0])
    for tau in (0.01, 0.05):
        part = merge_partition_3d(stats, 0, 16, tau)
        N = part.n_regions
        lbl = part.labels_pixels
        rL = int(np.bincount(lbl[mL].ravel()).argmax())
        rR = int(np.bincount(lbl[mR].ravel()).argmax())
        wL = part.regions[rL].q[:, 3:].mean(axis=0)
        wR = part.regions[rR].q[:, 3:].mean(axis=0)
        check(f"tau={tau}: halves not merged (N={N})", N >= 2 and rL != rR,
              f"majority labels L={rL}, R={rR}")
        ok = np.abs(wL - w1_true).max() < 0.15 and \
            np.abs(wR - w2_true).max() < 0.15
        check(f"tau={tau}: per-half w ~ truth", ok,
              f"wL={wL.round(3)} (want {w1_true}), wR={wR.round(3)} "
              f"(want {w2_true})")


# ---------------------------------------------------------------------------
def t5_2d_anchor():
    print("T5: z-trivial embedding == frozen 2D module (killing / frame / samples)")
    import synth as synth2d
    from killing2d import compute_cell_stats as compute_cell_stats_2d, \
        region_solve as region_solve_2d, solve_killing_trans as trans_2d
    from frame import make_region_samples as make_region_samples_2d, \
        integrate_frame as integrate_frame_2d

    omega0, c0 = 0.7, (0.3, -0.2)
    xs = np.linspace(-2, 2, 64)
    ys = np.linspace(-2, 2, 64)
    ts = np.linspace(0, 2 * np.pi, 48)
    data2d = synth2d.compose_rotating_frame(synth2d.four_cell_steady, omega0,
                                            c0, xs, ys, ts)
    dt = float(ts[1] - ts[0])
    n_z = 8
    zs = np.linspace(-0.7, 0.7, n_z)
    data3d = synth3d.embed_2d_field(data2d, n_z)

    st2 = compute_cell_stats_2d(data2d, xs, ys, dt, k=2, boundary_skip=2)
    nCy, nCx = st2.n_cells
    q2, E2, E02 = region_solve_2d(st2, np.arange(nCy * nCx), 0, len(ts))

    # boundary_skip=0 in z (the embedding is z-uniform; skipping z-shells only
    # rescales the sums) -- emulate by skipping in x/y via the 2D-matched mask:
    # easiest exact match: skip=2 in all axes but compare RATIOS and params.
    st3 = compute_cell_stats_3d(data3d, xs, ys, zs, dt, k=2, boundary_skip=2)
    nC = st3.n_cells
    q3, E3, E03 = region_solve_3d(st3, np.arange(nC[0] * nC[1] * nC[2]),
                                  0, len(ts))
    err_q = max(np.abs(q3[:, 0] - q2[:, 0]).max(),
                np.abs(q3[:, 1] - q2[:, 1]).max(),
                np.abs(q3[:, 5] - q2[:, 2]).max())
    err_zero = max(np.abs(q3[:, 2]).max(), np.abs(q3[:, 3]).max(),
                   np.abs(q3[:, 4]).max())
    check("killing3d(embedded) == killing2d: (tx,ty,wz) == (a,b,c)",
          err_q < 1e-6, f"max component err={err_q:.2e}")
    check("killing3d(embedded): out-of-plane DOF (tz,wx,wy) ~ 0",
          err_zero < 1e-6, f"max={err_zero:.2e}")
    check("killing3d(embedded): E/E0 matches 2D",
          abs(E3 / E03 - E2 / E02) < 1e-9,
          f"3D={E3 / E03:.3e} vs 2D={E2 / E02:.3e}")

    q2t, _ = trans_2d(st2.AtA.sum(axis=(1, 2)).sum(0),
                      st2.g.sum(axis=(1, 2)).sum(0),
                      st2.e0.sum())
    q3t, _ = solve_killing_trans_3d(st3.AtA.sum(axis=(1, 2, 3)).sum(0),
                                    st3.g.sum(axis=(1, 2, 3)).sum(0),
                                    st3.e0.sum())
    check("trans-only const solve matches 2D",
          np.abs(q3t[:2] - q2t[:2]).max() < 1e-6 and abs(q3t[2]) < 1e-6,
          f"3D={q3t[:3].round(5)} vs 2D={q2t[:2].round(5)}")

    # frame integration: w = (0, 0, c(t)) must reproduce theta/D of the 2D module
    rng = np.random.default_rng(3)
    qa = rng.normal(scale=0.6, size=(20, 3))          # 2D (a, b, c)
    q6 = np.zeros((20, 6))
    q6[:, :2] = qa[:, :2]
    q6[:, 5] = qa[:, 2]
    th2, D2 = integrate_frame_2d(qa, 0.17)
    R3, D3 = integrate_frame_3d(q6, 0.17)
    R_from_2d = np.stack([np.stack([np.cos(th2), -np.sin(th2)], -1),
                          np.stack([np.sin(th2), np.cos(th2)], -1)], -2)
    err_R = np.abs(R3[:, :2, :2] - R_from_2d).max()
    err_D = np.abs(D3[:, :2] - D2).max()
    err_z = max(np.abs(R3[:, 2, :2]).max(), np.abs(R3[:, :2, 2]).max(),
                np.abs(R3[:, 2, 2] - 1).max(), np.abs(D3[:, 2]).max())
    check("integrate_frame_3d(wz-only) == 2D theta/D",
          err_R < 1e-9 and err_D < 1e-9 and err_z < 1e-12,
          f"R err={err_R:.2e}, D err={err_D:.2e}, z-block err={err_z:.2e}")

    # pullback samples: single z-slice mask must reproduce 2D samples
    pix2 = np.zeros((len(ys), len(xs)), dtype=bool)
    pix2[10:-10, 12:-12] = True
    iz = 3
    pix3 = np.zeros((n_z, len(ys), len(xs)), dtype=bool)
    pix3[iz] = pix2
    q_emb = np.zeros((len(ts), 6))
    q_emb[:, :2] = q2[:, :2]
    q_emb[:, 5] = q2[:, 2]
    s2 = make_region_samples_2d(data2d, xs, ys, dt, pix2, 0, len(ts), q2)
    s3 = make_region_samples_3d(data3d, xs, ys, zs, dt, pix3, 0, len(ts), q_emb)
    err_xi = np.abs(s3.xi[:, :2] - s2.xi).max()
    err_vt = np.abs(s3.vtil[:, :2] - s2.vtil).max()
    err_w = np.abs(s3.vtil[:, 2]).max()
    check("pullback samples (xi, vtil) match 2D on a z-slice",
          err_xi < 1e-9 and err_vt < 1e-9 and err_w < 1e-12,
          f"xi err={err_xi:.2e}, vtil err={err_vt:.2e}, vz err={err_w:.2e}")


if __name__ == "__main__":
    import time
    for t_fn in (t1_manufactured, t2_translation, t3_roundtrip, t4_two_rotor,
                 t5_2d_anchor):
        t0 = time.time()
        t_fn()
        print(f"  -- {t_fn.__name__} done in {time.time() - t0:.0f}s", flush=True)
    print()
    if _failures:
        print(f"FAILED: {len(_failures)} check(s): {_failures}")
        sys.exit(1)
    print("ALL 3D VALIDATION CHECKS PASSED")
