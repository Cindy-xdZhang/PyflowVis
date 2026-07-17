"""Synthetic ground-truth fields for validation -- v2.

compose_rotating_frame builds the *manufactured* case where the answer is known in
closed form: take a steady field s(y) and observe it from a camera rotating with
angular velocity omega0 about center c0. Derivation (docs par.3 T1):

    y = c0 + R(omega0 t)(x - c0)            (observed point -> lab point)
    v(x, t) = R(-omega0 t) s(c0 + R(omega0 t)(x - c0)) - omega0 (x - c0)_perp

The Killing observer that steadies v is u(x) = -omega0 (x - c0)_perp, i.e. in the
u = (a - c y, b + c x) parameterization:
    c = -omega0,  a = -omega0 * c0_y,  b = +omega0 * c0_x.
"""
from __future__ import annotations

import numpy as np


def four_cell_steady(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Steady 2x2 vortex-cell pattern from stream function psi = sin(pi x) sin(pi y) / pi,
    gaussian-enveloped so it decays before the domain boundary.
    Returns (..., 2) stacked (v_x, v_y)."""
    env = np.exp(-(x * x + y * y) / 1.2)
    u = np.sin(np.pi * x) * np.cos(np.pi * y) * env
    v = -np.cos(np.pi * x) * np.sin(np.pi * y) * env
    return np.stack([u, v], axis=-1)


def gauss_vortex_steady(cx: float, cy: float, sigma: float = 0.45, strength: float = 1.5):
    """A single gaussian swirl centered at (cx, cy), as a closure s(x, y) -> (..., 2)."""
    def s(x, y):
        rx, ry = x - cx, y - cy
        env = strength * np.exp(-(rx * rx + ry * ry) / (2 * sigma * sigma))
        return np.stack([-ry * env, rx * env], axis=-1)
    return s


def compose_rotating_frame(s_fn, omega0: float, c0, xs: np.ndarray, ys: np.ndarray,
                           ts: np.ndarray) -> np.ndarray:
    """Evaluate v(x,t) = R(-w t) s(c0 + R(w t)(x - c0)) - w (x - c0)_perp analytically
    on the grid. Returns (T, Y, X, 2)."""
    c0 = np.asarray(c0, dtype=np.float64)
    Xg, Yg = np.meshgrid(xs, ys)
    rx, ry = Xg - c0[0], Yg - c0[1]
    out = np.empty((len(ts), len(ys), len(xs), 2))
    for i, t in enumerate(ts):
        th = omega0 * float(t)
        ct, st = np.cos(th), np.sin(th)
        # lab position y = c0 + R(th) (x - c0)
        lx = c0[0] + ct * rx - st * ry
        ly = c0[1] + st * rx + ct * ry
        s = s_fn(lx, ly)                              # (Y, X, 2)
        # rotate back by R(-th) and add frame velocity -w (x-c0)_perp
        sx, sy = s[..., 0], s[..., 1]
        vx = ct * sx + st * sy
        vy = -st * sx + ct * sy
        out[i, ..., 0] = vx + omega0 * ry
        out[i, ..., 1] = vy - omega0 * rx
    return out


def true_killing_params(omega0: float, c0) -> tuple[float, float, float]:
    """(a, b, c) of the observer that steadies compose_rotating_frame output."""
    return (-omega0 * float(c0[1]), omega0 * float(c0[0]), -omega0)


def compose_translating_frame(s_fn, ab0, xs: np.ndarray, ys: np.ndarray,
                              ts: np.ndarray) -> np.ndarray:
    """Steady pattern advected by a uniformly translating frame (Taylor frozen form):
        v(x, t) = s(x - ab0 * t) + ab0
    The translation-only observer u = ab0 (a = ab0_x, b = ab0_y, c = 0) steadies it
    exactly. Returns (T, Y, X, 2)."""
    a0, b0 = float(ab0[0]), float(ab0[1])
    Xg, Yg = np.meshgrid(xs, ys)
    out = np.empty((len(ts), len(ys), len(xs), 2))
    for i, t in enumerate(ts):
        s = s_fn(Xg - a0 * float(t), Yg - b0 * float(t))
        out[i, ..., 0] = s[..., 0] + a0
        out[i, ..., 1] = s[..., 1] + b0
    return out


def offset_pair_steady(c, d: float = 0.32, sigma: float = 0.24, strength: float = 2.0):
    """A steady pattern with NO rotational symmetry about c: a +swirl at c+(d,0) and a
    -swirl at c-(d,0). (A single gaussian swirl centered at the camera center is
    rotationally symmetric, so composing it with a rotation about that same center is
    STEADY -- a degenerate, useless counter-control. This pattern orbits instead.)"""
    s_a = gauss_vortex_steady(c[0] + d, c[1], sigma, strength)
    s_b = gauss_vortex_steady(c[0] - d, c[1], sigma, -strength)
    def s(x, y):
        return s_a(x, y) + s_b(x, y)
    return s


def two_rotor_field(xs: np.ndarray, ys: np.ndarray, ts: np.ndarray,
                    omega1: float = 0.8, omega2: float = -1.4) -> np.ndarray:
    """Counter-control field: left half = orbiting swirl-pair observed from a camera
    rotating about c1 with omega1; right half = same construction about c2 with omega2.
    Each half is exactly steadied by its own Killing observer, but no single observer
    explains both, so the tau-merge must NOT collapse to N=1. The seam at x=0 is
    genuinely discontinuous -- that is the point."""
    c1, c2 = (-1.0, 0.0), (1.0, 0.0)
    v1 = compose_rotating_frame(offset_pair_steady(c1), omega1, c1, xs, ys, ts)
    v2 = compose_rotating_frame(offset_pair_steady(c2), omega2, c2, xs, ys, ts)
    mask_right = (xs >= 0.0)[None, None, :, None]
    return np.where(mask_right, v2, v1)
