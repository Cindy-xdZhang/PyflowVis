"""Synthetic ground-truth 3D fields for validation -- referenceframe_inr_3d.

compose_rotating_frame_3d builds the manufactured case with a closed-form answer:
take a steady pattern s(y) and observe it from a camera rotating with angular
velocity omega0 about the axis n through center c0:

    y = c0 + R(omega0 t, n) (x - c0)                (observed -> lab point)
    v(x, t) = R(-omega0 t, n) s(y) - omega0 n x (x - c0)

The Killing observer that steadies v is u(x) = -omega0 n x (x - c0), i.e. in the
u = t_vec + w x x parameterization:
    w = -omega0 n,     t_vec = -w x c0 = c0 x w.
(The 2D module's (a, b, c) closed form is the n = e_z slice of this.)
"""
from __future__ import annotations

import numpy as np


def _axis_rot(n: np.ndarray, th: float) -> np.ndarray:
    """Rodrigues rotation about unit axis n by angle th."""
    n = np.asarray(n, dtype=np.float64)
    n = n / np.linalg.norm(n)
    K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def cells_steady_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Steady 3D vortex-cell pattern, gaussian-enveloped, with NO rotational
    symmetry about any axis (so composing with a rotating camera is genuinely
    unsteady).  Returns (..., 3)."""
    env = np.exp(-(x * x + y * y + z * z) / 1.2)
    vx = np.sin(np.pi * x) * np.cos(np.pi * y) * env
    vy = -np.cos(np.pi * x) * np.sin(np.pi * y) * env
    vz = 0.6 * np.sin(np.pi * z) * np.cos(np.pi * x) * env
    return np.stack([vx, vy, vz], axis=-1)


def gauss_swirl_steady_3d(center, axis, sigma: float = 0.45, strength: float = 1.5):
    """A gaussian swirl about `axis` through `center`, as a closure s(x,y,z)."""
    c = np.asarray(center, dtype=np.float64)
    n = np.asarray(axis, dtype=np.float64)
    n = n / np.linalg.norm(n)

    def s(x, y, z):
        rx, ry, rz = x - c[0], y - c[1], z - c[2]
        env = strength * np.exp(-(rx * rx + ry * ry + rz * rz)
                                / (2 * sigma * sigma))
        # n x r, enveloped
        vx = (n[1] * rz - n[2] * ry) * env
        vy = (n[2] * rx - n[0] * rz) * env
        vz = (n[0] * ry - n[1] * rx) * env
        return np.stack([vx, vy, vz], axis=-1)
    return s


def offset_pair_steady_3d(c, axis, d: float = 0.32, sigma: float = 0.24,
                          strength: float = 2.0):
    """Steady pattern with no rotational symmetry about (c, axis): +/- swirls
    offset by +-d along a direction perpendicular to the axis (they orbit when
    the camera rotates)."""
    c = np.asarray(c, dtype=np.float64)
    n = np.asarray(axis, dtype=np.float64); n = n / np.linalg.norm(n)
    perp = np.cross(n, [0.0, 0.0, 1.0])
    if np.linalg.norm(perp) < 1e-8:
        perp = np.cross(n, [0.0, 1.0, 0.0])
    perp = perp / np.linalg.norm(perp)
    s_a = gauss_swirl_steady_3d(c + d * perp, n, sigma, strength)
    s_b = gauss_swirl_steady_3d(c - d * perp, n, sigma, -strength)

    def s(x, y, z):
        return s_a(x, y, z) + s_b(x, y, z)
    return s


def compose_rotating_frame_3d(s_fn, omega0: float, axis, c0, xs, ys, zs, ts
                              ) -> np.ndarray:
    """v(x,t) = R(-w0 t, n) s(c0 + R(w0 t, n)(x - c0)) - w0 n x (x - c0) on the
    grid.  Returns (T, Z, Y, X, 3)."""
    c0 = np.asarray(c0, dtype=np.float64)
    n = np.asarray(axis, dtype=np.float64); n = n / np.linalg.norm(n)
    Zg, Yg, Xg = np.meshgrid(zs, ys, xs, indexing="ij")
    r = np.stack([Xg - c0[0], Yg - c0[1], Zg - c0[2]], axis=-1)   # (Z,Y,X,3)
    out = np.empty((len(ts), len(zs), len(ys), len(xs), 3))
    for i, t in enumerate(ts):
        R = _axis_rot(n, omega0 * float(t))
        lab = r @ R.T + c0                                       # y = c0 + R r
        s = s_fn(lab[..., 0], lab[..., 1], lab[..., 2])          # (Z,Y,X,3)
        v = s @ R                                                # R(-th) s
        frame = -omega0 * np.cross(np.broadcast_to(n, r.shape), r)
        out[i] = v + frame
    return out


def true_killing_params_3d(omega0: float, axis, c0) -> np.ndarray:
    """q = (t_vec, w) of the observer that steadies compose_rotating_frame_3d."""
    n = np.asarray(axis, dtype=np.float64); n = n / np.linalg.norm(n)
    w = -omega0 * n
    t_vec = np.cross(np.asarray(c0, dtype=np.float64), w)
    return np.concatenate([t_vec, w])


def compose_translating_frame_3d(s_fn, c_vec, xs, ys, zs, ts) -> np.ndarray:
    """Taylor-frozen form v(x, t) = s(x - c t) + c; the translation-only observer
    t_vec = c, w = 0 steadies it exactly.  Returns (T, Z, Y, X, 3)."""
    c = np.asarray(c_vec, dtype=np.float64)
    Zg, Yg, Xg = np.meshgrid(zs, ys, xs, indexing="ij")
    out = np.empty((len(ts), len(zs), len(ys), len(xs), 3))
    for i, t in enumerate(ts):
        tf = float(t)
        s = s_fn(Xg - c[0] * tf, Yg - c[1] * tf, Zg - c[2] * tf)
        out[i] = s + c
    return out


def two_rotor_field_3d(xs, ys, zs, ts, omega1: float = 0.8, omega2: float = -1.4
                       ) -> np.ndarray:
    """Counter-control: left half (x < 0) = orbiting swirl pair about axis e_z
    through c1 with omega1; right half = same about axis e_x through c2 with
    omega2 (different axis AND rate).  Each half is exactly steadied by its own
    Killing observer; no single observer explains both -> tau-merge must NOT
    collapse to N=1."""
    c1, c2 = (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)
    a1, a2 = (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)
    v1 = compose_rotating_frame_3d(offset_pair_steady_3d(c1, a1), omega1, a1, c1,
                                   xs, ys, zs, ts)
    v2 = compose_rotating_frame_3d(offset_pair_steady_3d(c2, a2), omega2, a2, c2,
                                   xs, ys, zs, ts)
    mask_right = (np.asarray(xs) >= 0.0)[None, None, None, :, None]
    return np.where(mask_right, v2, v1)


def embed_2d_field(data2d: np.ndarray, n_z: int) -> np.ndarray:
    """Replicate a 2D field (T, Y, X, 2) into (T, n_z, Y, X, 3) with vz = 0.
    Used by validate_rft3d T5 to pin the 3D solvers to the frozen 2D module."""
    T, Y, X, _ = data2d.shape
    out = np.zeros((T, n_z, Y, X, 3))
    out[..., :2] = data2d[:, None, :, :, :]
    return out
