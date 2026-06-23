"""
DCT_FMT: a training-free, Fourier-based flowmap tokenizer.

Motivation
----------
The original FMT encoder (``pnn.models.point_nn.EncNPNew`` / ``FMT_encoder.FMT``)
treats a window of pathlines as an *unordered point cloud* (KNN + positional
encoding + max/mean pooling). This makes the temporal axis of each pathline a
weak signal: ordering along time is essentially discarded by the global pooling,
and KNN over ``M*K*L`` points is slow.

DCT_FMT keeps the same *role* (window of pathlines -> single feature token of a
fixed dimension) but extracts the token in the **frequency domain along time**:

  * Each 2D pathline is read as a complex signal  z[n] = x[n] + i*y[n].
  * We take |FFT(z)| of
        - the center pathline's per-step velocity   (center_dt),
        - each neighbor's relative displacement rate (d(neighbor-center)/dt).
  * Keeping the magnitude makes the descriptor rotation invariant
    (rotating a pathline multiplies every DFT coefficient by e^{i*theta}).
  * The first ``dct_k`` coefficients per channel are concatenated and the
    per-seeding vectors are max/mean pooled across the window.

The encoder has **no learnable parameters** (it is an ``nn.Module`` only so it
moves with ``.to(device)`` and plugs into the trainable UNet/ViT backends the
same way ``EncNPNew`` does).

Interface
---------
``forward(pathlines)`` expects *structured* pathlines (NOT a flat point cloud),
shape ``[B, M, K, L, D]`` with ``D >= 2``:
    B = batch, M = #seedings in the window, K = #cross pathlines per seeding
    (index 0 is the seeding's own / "center" line, 1.. are neighbors),
    L = #time samples per line, D = point dim (x, y, [t]).
Returns a token of shape ``[B, out_dim]``.

This mirrors how ``EncNPNew`` returns ``[B, out_dim]`` for a window, so the two
encoders are interchangeable inside the FTLE-upsampling models.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from FMT_Utils.DCT_utils import dft_complex_1d


class DCT_FMT(nn.Module):
    def __init__(self,
                 nerbors: int,
                 L: int,
                 dct_k: int = 6,
                 dct_weight: float = 0.5,
                 neighbor_diff_scale: float = 100.0,
                 use_center: bool = True):
        """
        Args:
            nerbors:             #cross pathlines per seeding (K), incl. the center line.
            L:                   #time samples per pathline.
            dct_k:               #low-frequency DFT magnitudes to keep per channel.
                                 Automatically clamped to the available sequence length.
            dct_weight:          scale applied to the neighbor block when concatenating
                                 (matches the colleague's clustering recipe).
            neighbor_diff_scale: multiplier on the neighbor relative-velocity signal
                                 (the relative displacement rate is small, so it is
                                 amplified before the DFT, as in the prototype).
            use_center:          include the center pathline's velocity-spectrum block.
        """
        super().__init__()
        self.nerbors = int(nerbors)
        self.L = int(L)
        self.dct_weight = float(dct_weight)
        self.neighbor_diff_scale = float(neighbor_diff_scale)
        self.use_center = bool(use_center)

        # DFT is applied to per-step differences, so the usable length is L-1.
        self.seq_len = max(1, self.L - 1)
        self.dct_k = int(max(1, min(int(dct_k), self.seq_len)))

        self.num_neighbors = max(0, self.nerbors - 1)

        # Output token dimension (deterministic; no learnable params).
        center_dim = self.dct_k if self.use_center else 0
        neighbor_dim = self.num_neighbors * self.dct_k
        self.out_dim = int(center_dim + neighbor_dim)
        assert self.out_dim > 0, "DCT_FMT produced an empty feature; check nerbors/L/dct_k."

    def extra_repr(self) -> str:
        return (f"nerbors={self.nerbors}, L={self.L}, dct_k={self.dct_k}, "
                f"dct_weight={self.dct_weight}, out_dim={self.out_dim}")

    def forward(self, pathlines: torch.Tensor) -> torch.Tensor:
        """
        pathlines: [B, M, K, L, D]  (structured window of pathlines, D >= 2)
        returns:   [B, out_dim]
        """
        assert pathlines.dim() == 5, \
            f"DCT_FMT expects [B, M, K, L, D], got {tuple(pathlines.shape)}"
        B, M, K, L, D = pathlines.shape
        assert K == self.nerbors, f"K mismatch: expected {self.nerbors}, got {K}"
        assert D >= 2, f"point dim must be >= 2 (x,y), got {D}"

        # keep only (x, y) -> complex signal channels
        xy = pathlines[..., :2].to(torch.float32)          # [B, M, K, L, 2]
        center = xy[:, :, 0:1, :, :]                        # [B, M, 1, L, 2]

        feats = []

        if self.use_center:
            # center per-step velocity, then |FFT| of z=x+iy
            center_dt = center[:, :, :, 1:, :] - center[:, :, :, :-1, :]  # [B,M,1,L-1,2]
            ce = center_dt.reshape(B * M * 1, self.seq_len, 2)
            ce_mag = dft_complex_1d(ce)                     # [B*M, L-1]
            ce_feat = ce_mag[:, :self.dct_k].reshape(B, M, 1 * self.dct_k)
            feats.append(ce_feat)

        if self.num_neighbors > 0:
            neighbor = xy[:, :, 1:, :, :]                   # [B, M, K-1, L, 2]
            nd = neighbor - center                          # relative pos  [B,M,K-1,L,2]
            nd_dt = (nd[:, :, :, 1:, :] - nd[:, :, :, :-1, :]) * self.neighbor_diff_scale
            ne = nd_dt.reshape(B * M * self.num_neighbors, self.seq_len, 2)
            ne_mag = dft_complex_1d(ne)                     # [B*M*(K-1), L-1]
            ne_feat = ne_mag[:, :self.dct_k].reshape(B, M, self.num_neighbors * self.dct_k)
            feats.append(self.dct_weight * ne_feat)

        per_seed = torch.cat(feats, dim=-1)                 # [B, M, out_dim]

        # Aggregate the window's seedings into one token (same max+mean as FMT).
        token = per_seed.max(dim=1)[0] + per_seed.mean(dim=1)  # [B, out_dim]
        return token
