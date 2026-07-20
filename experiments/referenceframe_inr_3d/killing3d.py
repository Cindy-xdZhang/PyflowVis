"""3D Killing (rigid-body) observer least squares -- referenceframe_inr_3d.

Direct dimensional lift of experiments/referenceframe_inr_v2/killing2d.py (kept
frozen there; validate_rft3d.py T5 pins this file to it on z-trivial fields).

Math. For an unsteady 3D field v(x, t) the observed-time-derivative residual with
the Killing ansatz
    u(x) = t_vec + w x x            q = (tx, ty, tz, wx, wy, wz)
is
    r(x, t) = dv/dt + A(x, t) q
with one column of A per Killing basis field u_k (translations e_i, rotations
e_i x x):
    A[:, i]     = J e_i                       i = 0, 1, 2
    A[:, 3 + i] = J (e_i x x) - e_i x v      i = 0, 1, 2
(J[r, c] = d v_r / d x_c).  The 2D module is the wz-slice of this: e_z x x =
(-y, x, 0) is its x_perp and e_z x v its v_perp.  Per-region per-timestep solve:
    (sum AtA) q = -(sum At dvdt),  E = E0 + q . (sum At dvdt),  E0 = sum ||dvdt||^2.

Everything is accumulated at the level of kxkxk voxel *cells* so that region
statistics (region = union of cells) merge in O(1).  Memory note: AtA is
(T, nCz, nCy, nCx, 6, 6) float64 = T * nCells * 288 bytes; deltaWing full res
(T=171, cells 4^3) is ~0.76 GB -- pick k_cell accordingly.

Axis conventions match FLowUtils.VectorField3d: data is (T, Z, Y, X, 3) with
components (vx, vy, vz); d/dx is np.gradient axis=3, d/dy axis=2, d/dz axis=1,
d/dt axis=0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CellStats3D:
    """Per-cell, per-timestep Killing LSQ sufficient statistics."""
    AtA: np.ndarray      # (T, nCz, nCy, nCx, 6, 6)  sum over cell voxels of A^T A
    g: np.ndarray        # (T, nCz, nCy, nCx, 6)     sum of A^T dvdt   (RHS is -g)
    e0: np.ndarray       # (T, nCz, nCy, nCx)        sum of ||dvdt||^2
    npix: np.ndarray     # (nCz, nCy, nCx)           voxels with weight 1
    cell_z0: np.ndarray  # (nCz+1,) voxel plane edges of cells
    cell_y0: np.ndarray  # (nCy+1,)
    cell_x0: np.ndarray  # (nCx+1,)
    k: int
    T: int
    Z: int
    Y: int
    X: int
    dt: float

    @property
    def n_cells(self) -> tuple[int, int, int]:
        return self.AtA.shape[1], self.AtA.shape[2], self.AtA.shape[3]


def _cell_edges(n: int, k: int) -> np.ndarray:
    edges = list(range(0, n, k)) + [n]
    return np.asarray(edges, dtype=np.int64)


def compute_cell_stats_3d(data: np.ndarray, xs: np.ndarray, ys: np.ndarray,
                          zs: np.ndarray, dt: float, k: int = 4,
                          boundary_skip: int = 2) -> CellStats3D:
    """Accumulate Killing LSQ terms into kxkxk cells.

    data: (T, Z, Y, X, 3) float array; xs/ys/zs physical coordinates; dt spacing.
    boundary_skip: outermost voxel shells excluded from the LSQ sums (one-sided
    finite differences there are unreliable); those voxels still belong to
    cells/regions and are reconstructed -- they just do not vote for the observer.
    """
    data = np.asarray(data, dtype=np.float64)
    T, Z, Y, X, C = data.shape
    assert C == 3, f"expected 3 components, got {C}"
    dx = float(xs[1] - xs[0]) if X > 1 else 1.0
    dy = float(ys[1] - ys[0]) if Y > 1 else 1.0
    dz = float(zs[1] - zs[0]) if Z > 1 else 1.0

    Zg, Yg, Xg = np.meshgrid(zs, ys, xs, indexing="ij")     # (Z, Y, X) physical
    w = np.zeros((Z, Y, X), dtype=np.float64)
    s = int(boundary_skip)
    if s > 0:
        w[s:Z - s, s:Y - s, s:X - s] = 1.0
    else:
        w[:] = 1.0

    ez = _cell_edges(Z, k)
    ey = _cell_edges(Y, k)
    ex = _cell_edges(X, k)
    nCz, nCy, nCx = len(ez) - 1, len(ey) - 1, len(ex) - 1

    dvdt = np.gradient(data, dt, axis=0) if (T > 1 and dt > 0) else np.zeros_like(data)

    AtA = np.zeros((T, nCz, nCy, nCx, 6, 6), dtype=np.float64)
    g = np.zeros((T, nCz, nCy, nCx, 6), dtype=np.float64)
    e0 = np.zeros((T, nCz, nCy, nCx), dtype=np.float64)

    def cell_reduce(arr):  # (Z, Y, X, ...) -> (nCz, nCy, nCx, ...)
        a = np.add.reduceat(arr, ez[:-1], axis=0)
        a = np.add.reduceat(a, ey[:-1], axis=1)
        return np.add.reduceat(a, ex[:-1], axis=2)

    for it in range(T):
        v = data[it]                                        # (Z, Y, X, 3)
        # J[..., r, c] = d v_r / d x_c ; gradient axes: x=2, y=1, z=0 of (Z,Y,X)
        J = np.empty((Z, Y, X, 3, 3), dtype=np.float64)
        for r in range(3):
            J[..., r, 0] = np.gradient(v[..., r], dx, axis=2)
            J[..., r, 1] = np.gradient(v[..., r], dy, axis=1)
            J[..., r, 2] = np.gradient(v[..., r], dz, axis=0)

        A = np.empty((Z, Y, X, 3, 6), dtype=np.float64)
        A[..., :, 0:3] = J
        # rotation basis fields:  e_x x X = (0, -z, y);  e_y x X = (z, 0, -x);
        #                         e_z x X = (-y, x, 0);  columns J*(e_i x X) - e_i x v
        bx = np.stack([np.zeros_like(Xg), -Zg, Yg], axis=-1)
        by = np.stack([Zg, np.zeros_like(Xg), -Xg], axis=-1)
        bz = np.stack([-Yg, Xg, np.zeros_like(Xg)], axis=-1)
        cx = np.stack([np.zeros_like(Xg), -v[..., 2], v[..., 1]], axis=-1)
        cy = np.stack([v[..., 2], np.zeros_like(Xg), -v[..., 0]], axis=-1)
        cz = np.stack([-v[..., 1], v[..., 0], np.zeros_like(Xg)], axis=-1)
        A[..., :, 3] = np.einsum("zyxrc,zyxc->zyxr", J, bx) - cx
        A[..., :, 4] = np.einsum("zyxrc,zyxc->zyxr", J, by) - cy
        A[..., :, 5] = np.einsum("zyxrc,zyxc->zyxr", J, bz) - cz

        b = dvdt[it]                                        # (Z, Y, X, 3)
        Aw = A * w[..., None, None]
        AtA_vox = np.einsum("zyxri,zyxrj->zyxij", Aw, A)    # (Z, Y, X, 6, 6)
        g_vox = np.einsum("zyxri,zyxr->zyxi", Aw, b)        # (Z, Y, X, 6)
        e0_vox = w * np.einsum("zyxr,zyxr->zyx", b, b)

        AtA[it] = cell_reduce(AtA_vox)
        g[it] = cell_reduce(g_vox)
        e0[it] = cell_reduce(e0_vox)

    npix = cell_reduce(w)
    return CellStats3D(AtA=AtA, g=g, e0=e0, npix=npix, cell_z0=ez, cell_y0=ey,
                       cell_x0=ex, k=k, T=T, Z=Z, Y=Y, X=X, dt=dt)


def solve_killing_3d(AtA: np.ndarray, g: np.ndarray, e0: np.ndarray,
                     ridge_rel: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Solve (AtA) q = -g for a batch of timesteps and return (q, E).

    AtA: (..., 6, 6), g: (..., 6), e0: (...,). E = e0 + q.g (>= 0, clipped).
    Trace-scaled ridge keeps degenerate systems (uniform/empty regions) solvable;
    there g = 0 so q = 0 and E = e0, the correct degenerate answer."""
    AtA = np.asarray(AtA, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    tr = np.trace(AtA, axis1=-2, axis2=-1)
    lam = ridge_rel * (tr / 6.0) + 1e-300
    M = AtA + lam[..., None, None] * np.eye(6)
    try:
        q = np.linalg.solve(M, -g[..., None])[..., 0]
    except np.linalg.LinAlgError:
        q = np.stack([np.linalg.lstsq(m, -gv, rcond=None)[0]
                      for m, gv in zip(M.reshape(-1, 6, 6), g.reshape(-1, 6))])
        q = q.reshape(g.shape)
    E = e0 + np.einsum("...i,...i->...", q, g)
    return q, np.maximum(E, 0.0)


def solve_killing_trans_3d(AtA: np.ndarray, g: np.ndarray, e0: np.ndarray,
                           ridge_rel: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Translation-only (3-DOF) observer LS: leading 3x3 block with w fixed to 0.
    Returns q as (..., 6) with q[..., 3:] = 0 so frame integration is unchanged."""
    AtA = np.asarray(AtA, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    A3 = AtA[..., :3, :3]
    g3 = g[..., :3]
    tr = np.trace(A3, axis1=-2, axis2=-1)
    lam = ridge_rel * (tr / 3.0) + 1e-300
    M = A3 + lam[..., None, None] * np.eye(3)
    try:
        q3 = np.linalg.solve(M, -g3[..., None])[..., 0]
    except np.linalg.LinAlgError:
        q3 = np.stack([np.linalg.lstsq(m, -gv, rcond=None)[0]
                       for m, gv in zip(M.reshape(-1, 3, 3), g3.reshape(-1, 3))])
        q3 = q3.reshape(g3.shape)
    E = e0 + np.einsum("...i,...i->...", q3, g3)
    q = np.zeros(g.shape[:-1] + (6,))
    q[..., :3] = q3
    return q, np.maximum(E, 0.0)


def region_solve_3d(stats: CellStats3D, cell_ids: np.ndarray, it0: int, it1: int
                    ) -> tuple[np.ndarray, float, float]:
    """Killing solve for a region = set of flat cell ids, over window [it0, it1).
    Returns (q (Tw, 6), E_total, E0_total).  Test convenience wrapper.

    Memory note: fancy indexing stats.AtA[t, cz, cy, cx] with an n-cell index
    materializes a (Tw, n, 6, 6) temporary -- 12 GB for a full 40^3-cell grid --
    so the whole-grid case reduces over the cell axes directly and partial sets
    accumulate in chunks."""
    nCz, nCy, nCx = stats.n_cells
    ids = np.asarray(cell_ids, dtype=np.int64)
    if ids.size == nCz * nCy * nCx:
        AtA = stats.AtA[it0:it1].sum(axis=(1, 2, 3))   # (Tw, 6, 6), no temp
        g = stats.g[it0:it1].sum(axis=(1, 2, 3))
        e0 = stats.e0[it0:it1].sum(axis=(1, 2, 3))
    else:
        Tw = it1 - it0
        AtA = np.zeros((Tw, 6, 6)); g = np.zeros((Tw, 6)); e0 = np.zeros(Tw)
        for chunk in np.array_split(ids, max(1, ids.size // 1024)):
            cz, cy, cx = np.unravel_index(chunk, (nCz, nCy, nCx))
            AtA += stats.AtA[it0:it1, cz, cy, cx].sum(axis=1)
            g += stats.g[it0:it1, cz, cy, cx].sum(axis=1)
            e0 += stats.e0[it0:it1, cz, cy, cx].sum(axis=1)
    q, E = solve_killing_3d(AtA, g, e0)
    return q, float(E.sum()), float(e0.sum())
