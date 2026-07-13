"""2D Killing (rigid-body) observer least squares -- v2 fresh implementation.

Math (docs/referenceframe_inr_v2.md par.2.1). For an unsteady 2D field v(x,y,t) the
observed-time-derivative residual with the low-order Killing ansatz
    u(x) = (a - c*y, b + c*x)          (translation (a,b), angular velocity c)
is
    r(x,t) = dv/dt + A(x,t) q,   q = (a,b,c)
    A = [ J | J*x_perp - v_perp ],   x_perp = (-y, x),  v_perp = (-v_y, v_x)
and the per-region per-timestep solve is
    (sum AtA) q = -(sum At*dvdt),   E = E0 + q . (sum At*dvdt),  E0 = sum ||dvdt||^2.

Everything is accumulated at the level of kxk pixel *cells* so that region statistics
(region = union of cells) merge in O(1): AtA/g/e0 of a union are just sums.

Axis conventions match FLowUtils.VectorField2d: field.field is (T, Y, X, 2) with
components (v_x, v_y); d/dx is np.gradient axis=2, d/dy axis=1, d/dt axis=0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CellStats:
    """Per-cell, per-timestep Killing LSQ sufficient statistics."""
    AtA: np.ndarray      # (T, nCy, nCx, 3, 3)  sum over cell pixels of A^T A
    g: np.ndarray        # (T, nCy, nCx, 3)     sum of A^T dvdt   (RHS is -g)
    e0: np.ndarray       # (T, nCy, nCx)        sum of ||dvdt||^2
    npix: np.ndarray     # (nCy, nCx)           pixels with weight 1 (after boundary skip)
    cell_y0: np.ndarray  # (nCy+1,) pixel row edges of cells
    cell_x0: np.ndarray  # (nCx+1,) pixel col edges of cells
    k: int
    T: int
    Y: int
    X: int
    dt: float

    @property
    def n_cells(self) -> tuple[int, int]:
        return self.AtA.shape[1], self.AtA.shape[2]


def _cell_edges(n: int, k: int) -> np.ndarray:
    """Pixel edges of k-sized cells along one axis (last cell may be smaller)."""
    edges = list(range(0, n, k)) + [n]
    return np.asarray(edges, dtype=np.int64)


def compute_cell_stats(data: np.ndarray, xs: np.ndarray, ys: np.ndarray, dt: float,
                       k: int = 2, boundary_skip: int = 2) -> CellStats:
    """Accumulate Killing LSQ terms into kxk cells.

    data: (T, Y, X, 2) float array; xs (X,), ys (Y,) physical coordinates; dt time spacing.
    boundary_skip: outermost pixel rings excluded from the LSQ sums (one-sided finite
    differences there are unreliable); those pixels still belong to cells/regions and
    are reconstructed -- they just do not vote for the observer.
    """
    data = np.asarray(data, dtype=np.float64)
    T, Y, X, C = data.shape
    assert C == 2, f"expected 2 components, got {C}"
    dx = float(xs[1] - xs[0]) if X > 1 else 1.0
    dy = float(ys[1] - ys[0]) if Y > 1 else 1.0

    Xg, Yg = np.meshgrid(xs, ys)                      # (Y, X) physical coords
    w = np.zeros((Y, X), dtype=np.float64)
    s = int(boundary_skip)
    if s > 0:
        w[s:Y - s, s:X - s] = 1.0
    else:
        w[:] = 1.0

    ey = _cell_edges(Y, k)
    ex = _cell_edges(X, k)
    nCy, nCx = len(ey) - 1, len(ex) - 1

    dvdt = np.gradient(data, dt, axis=0) if (T > 1 and dt > 0) else np.zeros_like(data)

    AtA = np.zeros((T, nCy, nCx, 3, 3), dtype=np.float64)
    g = np.zeros((T, nCy, nCx, 3), dtype=np.float64)
    e0 = np.zeros((T, nCy, nCx), dtype=np.float64)

    def cell_reduce(arr):  # (Y, X, ...) -> (nCy, nCx, ...)
        a = np.add.reduceat(arr, ey[:-1], axis=0)
        return np.add.reduceat(a, ex[:-1], axis=1)

    for it in range(T):
        v = data[it]                                   # (Y, X, 2)
        # Jacobian J[..., i, j] = d v_i / d x_j
        dvx_dx = np.gradient(v[..., 0], dx, axis=1)
        dvx_dy = np.gradient(v[..., 0], dy, axis=0)
        dvy_dx = np.gradient(v[..., 1], dx, axis=1)
        dvy_dy = np.gradient(v[..., 1], dy, axis=0)
        # A columns: A[:, 0] = J e_x, A[:, 1] = J e_y, A[:, 2] = J x_perp - v_perp
        a00, a01 = dvx_dx, dvx_dy
        a10, a11 = dvy_dx, dvy_dy
        a02 = -Yg * dvx_dx + Xg * dvx_dy - (-v[..., 1])   # row x of J*x_perp - v_perp
        a12 = -Yg * dvy_dx + Xg * dvy_dy - (+v[..., 0])   # row y
        b0 = dvdt[it, ..., 0]
        b1 = dvdt[it, ..., 1]

        A = np.stack([np.stack([a00, a01, a02], axis=-1),
                      np.stack([a10, a11, a12], axis=-1)], axis=-2)   # (Y, X, 2, 3)
        Aw = A * w[..., None, None]
        AtA_pix = np.einsum("yxri,yxrj->yxij", Aw, A)                 # (Y, X, 3, 3)
        b = np.stack([b0, b1], axis=-1)                               # (Y, X, 2)
        g_pix = np.einsum("yxri,yxr->yxi", Aw, b)                     # (Y, X, 3)
        e0_pix = w * (b0 * b0 + b1 * b1)

        AtA[it] = cell_reduce(AtA_pix)
        g[it] = cell_reduce(g_pix)
        e0[it] = cell_reduce(e0_pix)

    npix = cell_reduce(w)
    return CellStats(AtA=AtA, g=g, e0=e0, npix=npix, cell_y0=ey, cell_x0=ex,
                     k=k, T=T, Y=Y, X=X, dt=dt)


def solve_killing(AtA: np.ndarray, g: np.ndarray, e0: np.ndarray,
                  ridge_rel: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Solve (AtA) q = -g for a batch of timesteps and return (q, E).

    AtA: (..., 3, 3), g: (..., 3), e0: (...,). E = e0 + q.g (>= 0, clipped).
    A trace-scaled ridge keeps degenerate systems (uniform/empty regions) solvable;
    there g = 0 so q = 0 and E = e0, which is the correct degenerate answer.
    """
    AtA = np.asarray(AtA, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    tr = np.trace(AtA, axis1=-2, axis2=-1)
    lam = ridge_rel * (tr / 3.0) + 1e-300
    M = AtA + lam[..., None, None] * np.eye(3)
    try:
        q = np.linalg.solve(M, -g[..., None])[..., 0]
    except np.linalg.LinAlgError:
        q = np.stack([np.linalg.lstsq(m, -gv, rcond=None)[0]
                      for m, gv in zip(M.reshape(-1, 3, 3), g.reshape(-1, 3))])
        q = q.reshape(g.shape)
    E = e0 + np.einsum("...i,...i->...", q, g)
    return q, np.maximum(E, 0.0)


def region_solve(stats: CellStats, cell_ids: np.ndarray, it0: int, it1: int
                 ) -> tuple[np.ndarray, float, float]:
    """Killing solve for a region = set of flat cell ids, over window [it0, it1).

    Returns (q (Tw,3), E_total, E0_total). Convenience wrapper used by tests; the
    partition code keeps running sums instead of re-summing cells every time.
    """
    nCy, nCx = stats.n_cells
    cy, cx = np.unravel_index(np.asarray(cell_ids, dtype=np.int64), (nCy, nCx))
    AtA = stats.AtA[it0:it1, cy, cx].sum(axis=1)   # (Tw, 3, 3)
    g = stats.g[it0:it1, cy, cx].sum(axis=1)       # (Tw, 3)
    e0 = stats.e0[it0:it1, cy, cx].sum(axis=1)     # (Tw,)
    q, E = solve_killing(AtA, g, e0)
    return q, float(E.sum()), float(e0.sum())
