"""
Fourier / DCT based temporal feature extraction utilities.

These helpers are the building blocks for the training-free ``DCT_FMT`` encoder
(see :mod:`FMT_Utils.DCT_FMT_encoder`). The key idea (suggested by a colleague,
originally prototyped in ``DCT_utils_xx.py`` / ``FMT_Clustering_xx.py``) is to
treat a 2D pathline as a *complex* signal ``z[n] = x[n] + i*y[n]`` and use the
magnitude of its DFT as a rotation-invariant temporal descriptor:

    rotating the pathline by theta maps  z -> e^{i*theta} * z,
    hence                                Z[k] -> e^{i*theta} * Z[k],
    so |Z[k]| is unchanged  ==> rotation invariant.

This addresses FMT's weak temporal fusion: instead of pooling points as an
unordered cloud, we explicitly encode the per-frequency content along time.
"""

import math
import torch


def create_dct_matrix(N: int, device=None, dtype=None):
    """
    Create an orthonormal DCT-II transform matrix [N x N].
    Equivalent to scipy.fft.dct(type=2, norm='ortho').
    """
    n = torch.arange(N, device=device, dtype=dtype).reshape(1, N)
    k = torch.arange(N, device=device, dtype=dtype).reshape(N, 1)
    dct_mat = torch.cos(math.pi / N * (n + 0.5) * k) * math.sqrt(2.0 / N)
    dct_mat[0, :] /= math.sqrt(2.0)
    return dct_mat  # [N, N]


def dct_1d(x: torch.Tensor):
    """
    Orthonormal DCT-II for batched input.
    x: [B, N, C]
    """
    N = x.shape[1]
    dct_mat = create_dct_matrix(N, device=x.device, dtype=x.dtype)
    X = torch.einsum('nk,bkc->bnc', dct_mat, x)
    return X


def idct_1d(X: torch.Tensor):
    """
    Orthonormal inverse DCT (type III), inverse of :func:`dct_1d`.
    X: [B, N, C]
    """
    N = X.shape[1]
    dct_mat = create_dct_matrix(N, device=X.device, dtype=X.dtype)
    x = torch.einsum('kn,bkc->bnc', dct_mat, X)
    return x


def dft_complex_1d(x: torch.Tensor, return_mag: bool = True):
    """
    DFT on a complex pathline signal z[n] = x[n] + i*y[n].

    Args:
        x:          [..., N, 2] where x[...,0]=x-coord, x[...,1]=y-coord.
                    Any number of leading (batch) dimensions is supported.
        return_mag: if True, return the rotation-invariant magnitudes |Z[k]|
                    of shape [..., N]; otherwise return the complex Z[k].

    Rotation by theta maps z -> e^{i*theta} z, so Z[k] -> e^{i*theta} Z[k];
    the magnitudes are therefore rotation invariant.
    """
    z = x[..., 0].to(torch.float32) + 1j * x[..., 1].to(torch.float32)  # [..., N] complex
    Z = torch.fft.fft(z, dim=-1)  # [..., N] complex
    if return_mag:
        return Z.abs()  # [..., N] real, rotation-invariant
    return Z
