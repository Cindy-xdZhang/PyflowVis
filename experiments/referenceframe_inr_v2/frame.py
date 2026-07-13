"""Observer frame integration, pushforward sampling, inverse reconstruction -- v2.

Given per-timestep Killing params q(t) = (a, b, c) of a region over window [t0, t1):

  frame motion   theta(t) = int_{t0}^{t} c ds            (trapezoid)
                 D(t)     = int_{t0}^{t} R(-theta)(a,b) ds  (trapezoid)
  lab -> observed  xi = R(-theta) x - D
  observed value   vtil(xi, t) = R(-theta) [ v(x,t) - u(x,t) ],  u(x) = (a - c y, b + c x)
  reconstruction   v(x,t) = R(theta) vtil(xi(x,t), t) + u(x,t)

The INR is trained on samples taken exactly at the pulled-back grid pixels of the
region, so reconstruction queries the *same* coordinates it was trained on: no
resampling of v, no interpolation, no blending, exact roundtrip by construction
(validate_rfc.py T3 asserts this with a perfect-oracle INR).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def rot(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def integrate_frame(q: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """theta (Tw,), D (Tw, 2) by trapezoid; theta[0] = 0, D[0] = 0."""
    q = np.asarray(q, dtype=np.float64)
    Tw = q.shape[0]
    c = q[:, 2]
    theta = np.zeros(Tw)
    if Tw > 1:
        theta[1:] = np.cumsum(0.5 * (c[1:] + c[:-1]) * dt)
    integ = np.empty((Tw, 2))
    for i in range(Tw):
        integ[i] = rot(-theta[i]) @ q[i, :2]
    D = np.zeros((Tw, 2))
    if Tw > 1:
        D[1:] = np.cumsum(0.5 * (integ[1:] + integ[:-1]) * dt, axis=0)
    return theta, D


@dataclass
class RegionSamples:
    """Flattened training samples of one (window, region) INR + everything needed to
    map INR predictions back to lab-frame grid values."""
    xi: np.ndarray        # (N, 2) observed-frame coordinates (physical units)
    tn: np.ndarray        # (N,) time coordinate, normalized to [-1, 1] within window
    vtil: np.ndarray      # (N, 2) observed values (physical units)
    pix_y: np.ndarray     # (Npix,) region pixel rows (same for every timestep)
    pix_x: np.ndarray     # (Npix,)
    theta: np.ndarray     # (Tw,)
    u: np.ndarray         # (Tw, Npix, 2) observer velocity at region pixels
    it0: int
    it1: int

    @property
    def n_pix(self) -> int:
        return self.pix_y.size

    @property
    def n_t(self) -> int:
        return self.it1 - self.it0


def t_norm_of_window(it0: int, it1: int) -> np.ndarray:
    Tw = it1 - it0
    return np.linspace(-1.0, 1.0, Tw) if Tw > 1 else np.zeros(1)


def make_region_samples(data: np.ndarray, xs: np.ndarray, ys: np.ndarray, dt: float,
                        pix_mask: np.ndarray, it0: int, it1: int,
                        q: np.ndarray) -> RegionSamples:
    """Pushforward samples for one region. data (T,Y,X,2); pix_mask (Y,X) bool;
    q (Tw,3). Pass q = zeros for the no-observer ablation (identity transform)."""
    pix_y, pix_x = np.nonzero(pix_mask)
    Xg, Yg = np.meshgrid(xs, ys)
    px = Xg[pix_y, pix_x]                      # (Npix,) physical x
    py = Yg[pix_y, pix_x]
    theta, D = integrate_frame(q, dt)
    tns = t_norm_of_window(it0, it1)

    Tw, Npix = it1 - it0, pix_y.size
    xi = np.empty((Tw, Npix, 2))
    vtil = np.empty((Tw, Npix, 2))
    u_all = np.empty((Tw, Npix, 2))
    for i in range(Tw):
        a, b, c = q[i]
        R_m = rot(-theta[i])                   # lab -> observed rotation
        u = np.stack([a - c * py, b + c * px], axis=-1)          # (Npix, 2)
        v = data[it0 + i, pix_y, pix_x, :].astype(np.float64)
        rel = v - u
        vtil[i] = rel @ R_m.T
        xi[i] = np.stack([px, py], axis=-1) @ R_m.T - D[i]
        u_all[i] = u

    tn = np.repeat(tns, Npix)
    return RegionSamples(xi=xi.reshape(-1, 2), tn=tn, vtil=vtil.reshape(-1, 2),
                         pix_y=pix_y, pix_x=pix_x, theta=theta, u=u_all,
                         it0=it0, it1=it1)


def scatter_reconstruction(recon: np.ndarray, samples: RegionSamples,
                           vtil_pred: np.ndarray) -> None:
    """Inverse transform INR predictions and write them into recon (T,Y,X,2) in place.

    vtil_pred: (N, 2) predictions in physical units, ordered like samples.xi."""
    Tw, Npix = samples.n_t, samples.n_pix
    vp = vtil_pred.reshape(Tw, Npix, 2)
    for i in range(Tw):
        R_p = rot(samples.theta[i])            # observed -> lab rotation
        v_lab = vp[i] @ R_p.T + samples.u[i]
        recon[samples.it0 + i, samples.pix_y, samples.pix_x, :] = v_lab
