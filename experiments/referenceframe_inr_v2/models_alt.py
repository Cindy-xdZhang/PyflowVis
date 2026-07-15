"""Alternative INR architectures -- v_MLP0.0 / v_FINER0.0 (docs/referenceframe_inr_v2.md par.1b).

Motivation: the large SIREN CoordNet is bimodally seed-unstable (docs par.4.9
point-4), and the handover next-steps list asks for non-SIREN architecture
counterparts. Both variants here keep the EXACT CoordNet skeleton -- encoder
[k->m, m->2m, 2m->4m, d x (4m->4m)], residual blocks with a two-layer main
path + a projection skip when in!=out + output=(main+skip)/2, decoder = one
residual block 4m->p -- and change ONLY the activation / initialization.
Every nn.Linear has the same shape as in CoordNetCompression.CoordNet, so
inr.coordnet_num_params() / inr.pick_m_for_budget() and the whole byte
accounting stay EXACT for every variant (parity asserted in inr.train_inr).

Variants:
  v_MLP0.0   name='mlp'   fully-connected residual ReLU MLP (common non-SIREN
             INR baseline). Per block: main = l2(relu(l1(x))), i.e. one ReLU on
             the hidden layer; block outputs stay LINEAR -- a ReLU there would
             clamp (main+skip)/2 to >= 0 and could not represent values in
             [-1,1]. Init: Kaiming-uniform (ReLU gain), zero bias. Raw (x,y,t)
             coordinates, no positional encoding (that would change in_dim and
             break parameter parity; a Fourier-feature variant would be a
             separate version).
  v_FINER0.0 name='finer' FINER (Liu et al., CVPR 2024,
             github.com/liuzhen0212/FINER): variable-periodic activation
             sin(omega * (|z|+1) * z) with z = Wx+b; SIREN weight init;
             optional first-layer bias ~ U(-first_bias_scale, first_bias_scale)
             (official repo default: first_bias_scale=None -> standard bias
             init, scale_req_grad=False -> the (|z|+1) factor is treated as a
             constant in backward; both defaults mirrored here).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

INR_MODEL_CHOICES = ("coordnet", "mlp", "finer")


# ---------------------------------------------------------------------------
# v_MLP0.0: residual ReLU MLP on the CoordNet skeleton
# ---------------------------------------------------------------------------
class _LinearAct(nn.Module):
    """nn.Linear + optional ReLU. Exposes `.linear` like SineLayer/FinerLayer so
    helpers that introspect `block.l2.linear.out_features` work unchanged."""

    def __init__(self, in_features: int, out_features: int, act: bool):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.act = bool(act)
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.linear.weight, nonlinearity="relu")
            self.linear.bias.zero_()

    def forward(self, x):
        y = self.linear(x)
        return torch.relu(y) if self.act else y


class ReLUResBlock(nn.Module):
    """Shape-identical to SirenResBlock: main = l2(relu(l1(x))), skip = x or a
    linear projection (in!=out), output = (main + skip) / 2."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.l1 = _LinearAct(in_features, out_features, act=True)
        self.l2 = _LinearAct(out_features, out_features, act=False)
        self.proj = None
        if in_features != out_features:
            self.proj = _LinearAct(in_features, out_features, act=False)

    def forward(self, x):
        skip = x if self.proj is None else self.proj(x)
        return 0.5 * (self.l2(self.l1(x)) + skip)


class MLPResNet(nn.Module):
    """v_MLP0.0: CoordNet skeleton with ReLU residual blocks; linear output."""

    def __init__(self, in_dim: int, out_dim: int, m: int = 64, d: int = 10, **_):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.encoder.append(ReLUResBlock(in_dim, m))
        self.encoder.append(ReLUResBlock(m, 2 * m))
        self.encoder.append(ReLUResBlock(2 * m, 4 * m))
        for _i in range(int(d)):
            self.encoder.append(ReLUResBlock(4 * m, 4 * m))
        self.decoder = ReLUResBlock(4 * m, out_dim)

    def forward(self, x):
        for blk in self.encoder:
            x = blk(x)
        return self.decoder(x)


# ---------------------------------------------------------------------------
# v_FINER0.0: FINER variable-periodic sine on the CoordNet skeleton
# ---------------------------------------------------------------------------
class FinerLayer(nn.Module):
    """Linear + sin(omega_0 * (|z|+1) * z), z = Wx+b (FINER, CVPR 2024).

    Weight init identical to SIREN (first layer U(+-1/in), hidden
    U(+-sqrt(6/in)/omega)). If `is_first` and `first_bias_scale` is set, the
    bias is drawn from U(+-first_bias_scale) (the FINER frequency-tuning knob);
    otherwise the nn.Linear default bias init is kept (repo default)."""

    def __init__(self, in_features: int, out_features: int, is_first: bool = False,
                 omega_0: float = 30.0, first_bias_scale: float | None = None,
                 scale_req_grad: bool = False):
        super().__init__()
        self.omega_0 = float(omega_0)
        self.scale_req_grad = bool(scale_req_grad)
        self.linear = nn.Linear(in_features, out_features)
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1.0 / in_features, 1.0 / in_features)
                if first_bias_scale is not None:
                    self.linear.bias.uniform_(-float(first_bias_scale),
                                              float(first_bias_scale))
            else:
                b = math.sqrt(6.0 / in_features) / self.omega_0
                self.linear.weight.uniform_(-b, b)

    def forward(self, x):
        z = self.linear(x)
        if self.scale_req_grad:
            scale = z.abs() + 1.0
        else:
            with torch.no_grad():
                scale = z.abs() + 1.0
        return torch.sin(self.omega_0 * scale * z)


class FinerResBlock(nn.Module):
    """SirenResBlock with FinerLayer everywhere (same shapes, same skip rule)."""

    def __init__(self, in_features: int, out_features: int, omega_0: float = 30.0,
                 is_first: bool = False, first_bias_scale: float | None = None,
                 scale_req_grad: bool = False):
        super().__init__()
        kw = dict(omega_0=omega_0, first_bias_scale=first_bias_scale,
                  scale_req_grad=scale_req_grad)
        self.l1 = FinerLayer(in_features, out_features, is_first=is_first, **kw)
        self.l2 = FinerLayer(out_features, out_features, is_first=False, **kw)
        self.proj = None
        if in_features != out_features:
            self.proj = FinerLayer(in_features, out_features, is_first=is_first, **kw)

    def forward(self, x):
        skip = x if self.proj is None else self.proj(x)
        return 0.5 * (self.l2(self.l1(x)) + skip)


class FinerNet(nn.Module):
    """v_FINER0.0: CoordNet skeleton with FINER blocks; decoder ends in the
    variable-periodic sine, so the output stays in [-1,1] like CoordNet."""

    def __init__(self, in_dim: int, out_dim: int, m: int = 64, d: int = 10,
                 omega_0: float = 30.0, first_bias_scale: float | None = None,
                 scale_req_grad: bool = False, **_):
        super().__init__()
        kw = dict(omega_0=omega_0, first_bias_scale=first_bias_scale,
                  scale_req_grad=scale_req_grad)
        self.encoder = nn.ModuleList()
        self.encoder.append(FinerResBlock(in_dim, m, is_first=True, **kw))
        self.encoder.append(FinerResBlock(m, 2 * m, **kw))
        self.encoder.append(FinerResBlock(2 * m, 4 * m, **kw))
        for _i in range(int(d)):
            self.encoder.append(FinerResBlock(4 * m, 4 * m, **kw))
        self.decoder = FinerResBlock(4 * m, out_dim, **kw)

    def forward(self, x):
        for blk in self.encoder:
            x = blk(x)
        return self.decoder(x)


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------
def build_inr_model(name: str, in_dim: int, out_dim: int, m: int, d: int,
                    **kwargs) -> nn.Module:
    """Build an INR with the CoordNet skeleton. kwargs are variant-specific
    (ignored where not applicable): omega_0, first_bias_scale, scale_req_grad."""
    name = str(name).lower()
    if name == "coordnet":
        from CoordNetCompression import CoordNet  # frozen verified baseline class
        return CoordNet(in_dim, out_dim, m=m, d=d,
                        omega_0=kwargs.get("omega_0", 30.0), final_activation="sine")
    if name == "mlp":
        return MLPResNet(in_dim, out_dim, m=m, d=d)
    if name == "finer":
        return FinerNet(in_dim, out_dim, m=m, d=d,
                        omega_0=kwargs.get("omega_0", 30.0),
                        first_bias_scale=kwargs.get("first_bias_scale", None),
                        scale_req_grad=kwargs.get("scale_req_grad", False))
    raise ValueError(f"unknown INR model '{name}'. Choices: {INR_MODEL_CHOICES}")
