"""Observer frame integration, pushforward sampling, inverse reconstruction -- 3D.

Dimensional lift of experiments/referenceframe_inr_v2/frame.py.  Given per-timestep
Killing params q(t) = (t_vec, w) of a region over window [t0, t1):

  frame motion   R(t) in SO(3), observed->lab:  dR/dt = what(t) R,  R(t0) = I
                 (2D reduces to R = rot(theta), theta = int c: w = (0, 0, c))
                 D(t) = int_{t0}^{t} R(s)^T t_vec(s) ds          (trapezoid)
  lab -> observed  xi = R^T x - D
  observed value   vtil(xi, t) = R^T [ v(x, t) - u(x, t) ],  u(x) = t_vec + w x x
  reconstruction   v(x, t) = R vtil(xi(x, t), t) + u(x, t)

Consistency (why D solves Ddot = R^T t_vec): with the observer map
x = R(t) xi + b(t) one gets u(x) = bdot + w x (x - b), i.e. t_vec = bdot - w x b,
and D := R^T b satisfies Ddot = R^T (bdot - w x b) = R^T t_vec.

Unlike 2D, rotation increments do not commute; R is accumulated per step as
R_{i+1} = exp(what_avg dt) R_i with the trapezoid-averaged w (matrix exponential
via Rodrigues).  The INR is trained on samples taken exactly at the pulled-back
grid voxels of the region: no resampling, no interpolation, exact roundtrip by
construction (validate_rft3d.py T3 asserts this with a perfect-oracle INR).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def hat(w: np.ndarray) -> np.ndarray:
    """Skew matrix: hat(w) a = w x a."""
    wx, wy, wz = w
    return np.array([[0.0, -wz, wy],
                     [wz, 0.0, -wx],
                     [-wy, wx, 0.0]], dtype=np.float64)


def rot_exp(w: np.ndarray, dt: float) -> np.ndarray:
    """exp(hat(w) dt) by Rodrigues; series fallback for tiny angles."""
    phi = float(np.linalg.norm(w)) * dt
    K = hat(w) * dt
    if phi < 1e-12:
        return np.eye(3) + K + 0.5 * (K @ K)
    K = K / phi
    return np.eye(3) + np.sin(phi) * K + (1.0 - np.cos(phi)) * (K @ K)


def integrate_frame_3d(q: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """R (Tw, 3, 3) observed->lab and D (Tw, 3); R[0] = I, D[0] = 0.

    q: (Tw, 6) = (t_vec, w) per timestep.  R by per-step Rodrigues with
    trapezoid-averaged w; D by trapezoid on R^T t_vec (matches the 2D module
    bit-for-bit when w = (0, 0, c))."""
    q = np.asarray(q, dtype=np.float64)
    Tw = q.shape[0]
    R = np.empty((Tw, 3, 3))
    R[0] = np.eye(3)
    for i in range(1, Tw):
        w_avg = 0.5 * (q[i - 1, 3:] + q[i, 3:])
        R[i] = rot_exp(w_avg, dt) @ R[i - 1]
    integ = np.empty((Tw, 3))
    for i in range(Tw):
        integ[i] = R[i].T @ q[i, :3]
    D = np.zeros((Tw, 3))
    if Tw > 1:
        D[1:] = np.cumsum(0.5 * (integ[1:] + integ[:-1]) * dt, axis=0)
    return R, D


@dataclass
class RegionSamples3D:
    """Flattened training samples of one (window, region) INR + everything needed
    to map INR predictions back to lab-frame grid values."""
    xi: np.ndarray        # (N, 3) observed-frame coordinates (physical units)
    tn: np.ndarray        # (N,) time coordinate, normalized to [-1, 1] in window
    vtil: np.ndarray      # (N, 3) observed values (physical units)
    pix_z: np.ndarray     # (Npix,) region voxel indices (same for every timestep)
    pix_y: np.ndarray
    pix_x: np.ndarray
    R: np.ndarray         # (Tw, 3, 3) observed->lab rotations
    u: np.ndarray         # (Tw, Npix, 3) observer velocity at region voxels
    it0: int
    it1: int

    @property
    def n_pix(self) -> int:
        return self.pix_z.size

    @property
    def n_t(self) -> int:
        return self.it1 - self.it0


def t_norm_of_window(it0: int, it1: int) -> np.ndarray:
    Tw = it1 - it0
    return np.linspace(-1.0, 1.0, Tw) if Tw > 1 else np.zeros(1)


def make_region_samples_3d(data: np.ndarray, xs: np.ndarray, ys: np.ndarray,
                           zs: np.ndarray, dt: float, pix_mask: np.ndarray,
                           it0: int, it1: int, q: np.ndarray) -> RegionSamples3D:
    """Pushforward samples for one region. data (T,Z,Y,X,3); pix_mask (Z,Y,X)
    bool; q (Tw,6).  Pass q = zeros for the no-observer ablation (identity)."""
    pix_z, pix_y, pix_x = np.nonzero(pix_mask)
    px = xs[pix_x].astype(np.float64)                       # (Npix,) physical
    py = ys[pix_y].astype(np.float64)
    pz = zs[pix_z].astype(np.float64)
    P = np.stack([px, py, pz], axis=-1)                     # (Npix, 3)
    R, D = integrate_frame_3d(q, dt)
    tns = t_norm_of_window(it0, it1)

    Tw, Npix = it1 - it0, pix_z.size
    xi = np.empty((Tw, Npix, 3))
    vtil = np.empty((Tw, Npix, 3))
    u_all = np.empty((Tw, Npix, 3))
    for i in range(Tw):
        t_vec, w = q[i, :3], q[i, 3:]
        u = t_vec[None, :] + np.cross(np.broadcast_to(w, (Npix, 3)), P)
        v = data[it0 + i, pix_z, pix_y, pix_x, :].astype(np.float64)
        rel = v - u
        vtil[i] = rel @ R[i]                                # = (R^T rel^T)^T
        xi[i] = P @ R[i] - D[i]
        u_all[i] = u

    tn = np.repeat(tns, Npix)
    return RegionSamples3D(xi=xi.reshape(-1, 3), tn=tn, vtil=vtil.reshape(-1, 3),
                           pix_z=pix_z, pix_y=pix_y, pix_x=pix_x, R=R, u=u_all,
                           it0=it0, it1=it1)


def scatter_reconstruction_3d(recon: np.ndarray, samples: RegionSamples3D,
                              vtil_pred: np.ndarray) -> None:
    """Inverse transform INR predictions and write them into recon (T,Z,Y,X,3)
    in place.  vtil_pred: (N, 3) physical units, ordered like samples.xi."""
    Tw, Npix = samples.n_t, samples.n_pix
    vp = vtil_pred.reshape(Tw, Npix, 3)
    for i in range(Tw):
        v_lab = vp[i] @ samples.R[i].T + samples.u[i]
        recon[samples.it0 + i, samples.pix_z, samples.pix_y,
              samples.pix_x, :] = v_lab
