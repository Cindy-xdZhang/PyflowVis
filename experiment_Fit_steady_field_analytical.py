"""
experiment_Fit_steady_field_analytical.py
=========================================
Experiment: Fit 10 steady 2D analytical vector fields with
  (A) Gaussian-vortex + affine background   (analytical, scipy optimisation)
  (B) Vorticity-blob / Biot–Savart          (analytical, scipy optimisation)
  (C) Implicit Neural Representation (SIREN) (neural network fitting)
and compare the reconstruction accuracy (MSE).

Usage:
    python experiment_Fit_steady_field_analytical.py
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')           # non-interactive backend for saving figures
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from copy import deepcopy

# ── project imports ──────────────────────────────────────────────────────────
from FLowUtils.AnalyticalFlowCreator import (
    AnalyticalFlowCreator,
    constant_rotation,
    rotation_four_center,
    beadsFLow,
    double_gyre_2D,
)

# ════════════════════════════════════════════════════════════════════════════
#  §1  Generate 10 steady 2D vector fields
# ════════════════════════════════════════════════════════════════════════════

GRID_SIZE = (64, 64)      # (Xdim, Ydim)
DOMAIN_MIN = [-2.0, -2.0, 0.0]
DOMAIN_MAX = [ 2.0,  2.0, 0.0]   # tmin=tmax=0 → steady


def _field_to_numpy(vf):
    """Return a single steady slice as (Ydim, Xdim, 2) numpy array."""
    d = vf.field
    if hasattr(d, 'numpy'):
        d = d.detach().cpu().numpy()
    if d.ndim == 4:          # (T, Y, X, 2)
        d = d[0]
    return d.astype(np.float64)


def _make_field(expr_x, expr_y, params=None):
    """Helper: create a steady field from string / callable expressions."""
    fc = AnalyticalFlowCreator(expr_x, expr_y, parameters=params)
    vf = fc.create_flow_field(GRID_SIZE, 1, DOMAIN_MIN, DOMAIN_MAX)
    return _field_to_numpy(vf)


# ── 10 test fields ──────────────────────────────────────────────────────────

def gen_constant_rotation():
    """Simple rigid-body rotation: v = (-y, x)."""
    vf = constant_rotation(GRID_SIZE, 1, DOMAIN_MIN, DOMAIN_MAX)
    return _field_to_numpy(vf)

def gen_lamb_oseen_single():
    """Single Lamb–Oseen vortex centred at origin, Γ=2π, r_c=0.5."""
    Gamma, rc = 2*np.pi, 0.5
    def ux(x, y, t):
        r2 = x**2 + y**2 + 1e-12
        return -Gamma/(2*np.pi) * y/r2 * (1 - np.exp(-r2/rc**2))
    def uy(x, y, t):
        r2 = x**2 + y**2 + 1e-12
        return  Gamma/(2*np.pi) * x/r2 * (1 - np.exp(-r2/rc**2))
    return _make_field(ux, uy)

def gen_two_co_rotating_vortices():
    """Two co-rotating Lamb–Oseen vortices at (±0.7, 0)."""
    centres = [(-0.7, 0.0), (0.7, 0.0)]
    Gammas  = [3.0, 3.0]
    rcs     = [0.4, 0.4]
    def ux(x, y, t):
        u = np.zeros_like(x)
        for (cx,cy), G, rc in zip(centres, Gammas, rcs):
            dx, dy = x-cx, y-cy
            r2 = dx**2 + dy**2 + 1e-12
            u += -G/(2*np.pi) * dy/r2 * (1 - np.exp(-r2/rc**2))
        return u
    def uy(x, y, t):
        v = np.zeros_like(x)
        for (cx,cy), G, rc in zip(centres, Gammas, rcs):
            dx, dy = x-cx, y-cy
            r2 = dx**2 + dy**2 + 1e-12
            v +=  G/(2*np.pi) * dx/r2 * (1 - np.exp(-r2/rc**2))
        return v
    return _make_field(ux, uy)

def gen_counter_rotating_vortex_pair():
    """Two counter-rotating vortices at (±0.7, 0)."""
    centres = [(-0.7, 0.0), (0.7, 0.0)]
    Gammas  = [4.0, -4.0]
    rcs     = [0.35, 0.35]
    def ux(x, y, t):
        u = np.zeros_like(x)
        for (cx,cy), G, rc in zip(centres, Gammas, rcs):
            dx, dy = x-cx, y-cy
            r2 = dx**2 + dy**2 + 1e-12
            u += -G/(2*np.pi) * dy/r2 * (1 - np.exp(-r2/rc**2))
        return u
    def uy(x, y, t):
        v = np.zeros_like(x)
        for (cx,cy), G, rc in zip(centres, Gammas, rcs):
            dx, dy = x-cx, y-cy
            r2 = dx**2 + dy**2 + 1e-12
            v +=  G/(2*np.pi) * dx/r2 * (1 - np.exp(-r2/rc**2))
        return v
    return _make_field(ux, uy)

def gen_uniform_shear():
    """Pure linear shear: v = (y, 0)."""
    return _make_field('y', '0.0*x')

def gen_saddle_point():
    """Saddle / strain flow: v = (x, -y)."""
    return _make_field('x', '-y')

def gen_source_sink():
    """Radial source: v = (x, y) / (r + ε)."""
    def ux(x, y, t):
        r = np.sqrt(x**2 + y**2) + 0.3
        return x / r
    def uy(x, y, t):
        r = np.sqrt(x**2 + y**2) + 0.3
        return y / r
    return _make_field(ux, uy)

def gen_rankine_vortex():
    """Rankine vortex: solid-body inside R, irrotational outside."""
    R = 0.8
    def ux(x, y, t):
        r = np.sqrt(x**2 + y**2) + 1e-12
        inside = r < R
        u = np.where(inside, -y, -y * R**2 / r**2)
        return u
    def uy(x, y, t):
        r = np.sqrt(x**2 + y**2) + 1e-12
        inside = r < R
        v = np.where(inside, x, x * R**2 / r**2)
        return v
    return _make_field(ux, uy)

def gen_beads_flow_steady():
    """Beads flow frozen at t=0."""
    vf = beadsFLow(GRID_SIZE, 1, DOMAIN_MIN, DOMAIN_MAX)
    return _field_to_numpy(vf)

def gen_double_gyre_steady():
    """Double gyre frozen at t=0 (steady snapshot)."""
    # Use domain [0,2]×[0,1], 1 timestep
    fc = AnalyticalFlowCreator(
        lambda x, y, t, eps, A, omega: -np.sin(np.pi*x)*np.cos(np.pi*y)*np.pi*A,
        lambda x, y, t, eps, A, omega:  np.cos(np.pi*x)*np.sin(np.pi*y)*np.pi*A,
        parameters={'eps': 0.0, 'A': 0.1, 'omega': 0.0}
    )
    vf = fc.create_flow_field(GRID_SIZE, 1, [0.0, 0.0, 0.0], [2.0, 1.0, 0.0])
    return _field_to_numpy(vf), [0.0, 0.0, 0.0], [2.0, 1.0, 0.0]


# Assemble all fields ────────────────────────────────────────────────────────
FIELD_NAMES = [
    "Constant Rotation",
    "Lamb-Oseen Single Vortex",
    "Two Co-rotating Vortices",
    "Counter-rotating Pair",
    "Uniform Shear",
    "Saddle / Strain",
    "Source / Sink",
    "Rankine Vortex",
    "Beads Flow (t=0)",
    "Double Gyre (t=0)",
]


def generate_all_fields():
    """Return list of (name, field_numpy, x_coords, y_coords)."""
    generators = [
        gen_constant_rotation,
        gen_lamb_oseen_single,
        gen_two_co_rotating_vortices,
        gen_counter_rotating_vortex_pair,
        gen_uniform_shear,
        gen_saddle_point,
        gen_source_sink,
        gen_rankine_vortex,
        gen_beads_flow_steady,
        gen_double_gyre_steady,
    ]
    fields = []
    for name, gen in zip(FIELD_NAMES, generators):
        result = gen()
        if isinstance(result, tuple):
            data, dmin, dmax = result
        else:
            data = result
            dmin, dmax = DOMAIN_MIN, DOMAIN_MAX
        xc = np.linspace(dmin[0], dmax[0], GRID_SIZE[0])
        yc = np.linspace(dmin[1], dmax[1], GRID_SIZE[1])
        fields.append((name, data, xc, yc))
    return fields


# ════════════════════════════════════════════════════════════════════════════
#  §2-A  Analytical fit: Gaussian vortex + affine background
# ════════════════════════════════════════════════════════════════════════════
"""
Model:
  v(x,y) = v_affine(x,y) + Σ_{k=1}^{K} v_vortex_k(x,y)
where
  v_affine = (a0 + a1*x + a2*y,  b0 + b1*x + b2*y)           (6 params)
  v_vortex_k:  Lamb-Oseen  centred at (cx_k, cy_k)
    with circulation Γ_k and core radius r_{c,k}               (4 params each)

Total params = 6 + 4*K.
We use K = 3 vortex elements as default.
"""

N_VORTEX_GAUSSAFF = 3   # number of vortex elements


def _gaussaff_velocity(params, X, Y, K=N_VORTEX_GAUSSAFF):
    """Evaluate the Gaussian-vortex + affine model on meshgrid X, Y."""
    a0, a1, a2, b0, b1, b2 = params[:6]
    Vx = a0 + a1*X + a2*Y
    Vy = b0 + b1*X + b2*Y
    idx = 6
    for _ in range(K):
        cx, cy, Gamma, rc = params[idx:idx+4]
        idx += 4
        rc = max(abs(rc), 0.05)   # clamp core radius away from 0
        dx = X - cx
        dy = Y - cy
        r2 = dx**2 + dy**2 + 1e-14
        factor = Gamma / (2*np.pi) * (1 - np.exp(-r2 / rc**2)) / r2
        Vx += -dy * factor
        Vy +=  dx * factor
    return Vx, Vy


def _gaussaff_loss(params, X, Y, Ux_gt, Uy_gt, K=N_VORTEX_GAUSSAFF):
    Vx, Vy = _gaussaff_velocity(params, X, Y, K)
    return np.mean((Vx - Ux_gt)**2 + (Vy - Uy_gt)**2)


def fit_gaussaff(field, xc, yc, K=N_VORTEX_GAUSSAFF, n_restarts=3):
    """Fit Gaussian-vortex + affine model.  Returns best_params, mse."""
    X, Y = np.meshgrid(xc, yc)
    Ux_gt = field[:, :, 0]
    Uy_gt = field[:, :, 1]
    n_params = 6 + 4*K
    best_res = None
    for _ in range(n_restarts):
        p0 = np.random.randn(n_params) * 0.5
        # initialise vortex centres inside domain
        for k in range(K):
            p0[6+4*k]   = np.random.uniform(xc[0], xc[-1])
            p0[6+4*k+1] = np.random.uniform(yc[0], yc[-1])
            p0[6+4*k+2] = np.random.randn() * 2.0     # Gamma
            p0[6+4*k+3] = np.random.uniform(0.2, 1.0) # rc
        res = minimize(_gaussaff_loss, p0, args=(X, Y, Ux_gt, Uy_gt, K),
                       method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-15})
        if best_res is None or res.fun < best_res.fun:
            best_res = res
    mse = best_res.fun
    return best_res.x, mse


def predict_gaussaff(params, xc, yc, K=N_VORTEX_GAUSSAFF):
    X, Y = np.meshgrid(xc, yc)
    Vx, Vy = _gaussaff_velocity(params, X, Y, K)
    pred = np.stack([Vx, Vy], axis=-1)
    return pred


# ════════════════════════════════════════════════════════════════════════════
#  §2-B  Analytical fit: Vorticity-blob / Biot–Savart
# ════════════════════════════════════════════════════════════════════════════
"""
Model:
  ω(x,y) = Σ_{k=1}^{M} (α_k / (π σ_k²)) * exp(-|x-x_k|²/σ_k²)
  v = K * ω   via regularised Biot–Savart:
    v_x(x,y) = Σ_k  -α_k/(2π) * (y-y_k)/r² * (1 - exp(-r²/σ_k²))
    v_y(x,y) = Σ_k   α_k/(2π) * (x-x_k)/r² * (1 - exp(-r²/σ_k²))
  + uniform translation (u0, v0)

Total params = 2 + 4*M.
We use M = 5 vortex particles as default.
"""

N_BLOBS = 5


def _blobbs_velocity(params, X, Y, M=N_BLOBS):
    """Evaluate vorticity-blob model on meshgrid X, Y."""
    u0, v0 = params[:2]
    Vx = np.full_like(X, u0, dtype=np.float64)
    Vy = np.full_like(X, v0, dtype=np.float64)
    idx = 2
    for _ in range(M):
        xk, yk, alpha, sigma = params[idx:idx+4]
        idx += 4
        sigma = max(abs(sigma), 0.05)
        dx = X - xk
        dy = Y - yk
        r2 = dx**2 + dy**2 + 1e-14
        factor = alpha / (2*np.pi) * (1 - np.exp(-r2 / sigma**2)) / r2
        Vx += -dy * factor
        Vy +=  dx * factor
    return Vx, Vy


def _blobbs_loss(params, X, Y, Ux_gt, Uy_gt, M=N_BLOBS):
    Vx, Vy = _blobbs_velocity(params, X, Y, M)
    return np.mean((Vx - Ux_gt)**2 + (Vy - Uy_gt)**2)


def fit_blobbs(field, xc, yc, M=N_BLOBS, n_restarts=3):
    """Fit vorticity-blob / Biot-Savart model. Returns best_params, mse."""
    X, Y = np.meshgrid(xc, yc)
    Ux_gt = field[:, :, 0]
    Uy_gt = field[:, :, 1]
    n_params = 2 + 4*M
    best_res = None
    for _ in range(n_restarts):
        p0 = np.random.randn(n_params) * 0.3
        for k in range(M):
            p0[2+4*k]   = np.random.uniform(xc[0], xc[-1])
            p0[2+4*k+1] = np.random.uniform(yc[0], yc[-1])
            p0[2+4*k+2] = np.random.randn() * 2.0       # alpha
            p0[2+4*k+3] = np.random.uniform(0.2, 1.2)   # sigma
        res = minimize(_blobbs_loss, p0, args=(X, Y, Ux_gt, Uy_gt, M),
                       method='L-BFGS-B', options={'maxiter': 3000, 'ftol': 1e-15})
        if best_res is None or res.fun < best_res.fun:
            best_res = res
    mse = best_res.fun
    return best_res.x, mse


def predict_blobbs(params, xc, yc, M=N_BLOBS):
    X, Y = np.meshgrid(xc, yc)
    Vx, Vy = _blobbs_velocity(params, X, Y, M)
    pred = np.stack([Vx, Vy], axis=-1)
    return pred


# ════════════════════════════════════════════════════════════════════════════
#  §3  INR fitting: SIREN (Sinusoidal Representation Network)
# ════════════════════════════════════════════════════════════════════════════

class SineLayer(nn.Module):
    """Sine activation with learnable ω₀."""
    def __init__(self, in_features, out_features, omega0=30.0, is_first=False):
        super().__init__()
        self.omega0 = omega0
        self.linear = nn.Linear(in_features, out_features)
        # SIREN initialisation
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1.0/in_features, 1.0/in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6.0/in_features)/omega0,
                     np.sqrt(6.0/in_features)/omega0)

    def forward(self, x):
        return torch.sin(self.omega0 * self.linear(x))


class SIREN_INR(nn.Module):
    """
    SIREN: implicit neural representation  (x,y) → (vx, vy).
    3 hidden layers of 128 neurons, sine activations.
    """
    def __init__(self, hidden_dim=128, n_hidden=3, omega0=30.0):
        super().__init__()
        layers = [SineLayer(2, hidden_dim, omega0=omega0, is_first=True)]
        for _ in range(n_hidden - 1):
            layers.append(SineLayer(hidden_dim, hidden_dim, omega0=omega0))
        self.net = nn.Sequential(*layers)
        self.out_linear = nn.Linear(hidden_dim, 2)
        with torch.no_grad():
            self.out_linear.weight.uniform_(
                -np.sqrt(6.0/hidden_dim)/omega0,
                 np.sqrt(6.0/hidden_dim)/omega0)

    def forward(self, coords):
        return self.out_linear(self.net(coords))


def fit_inr(field, xc, yc, epochs=2000, lr=1e-4, hidden_dim=128,
            n_hidden=3, device='cpu'):
    """Train SIREN INR on the given field.  Returns model, mse."""
    X, Y = np.meshgrid(xc, yc)
    # normalise inputs to [-1, 1]
    cx = (xc[-1] + xc[0]) / 2.0
    cy = (yc[-1] + yc[0]) / 2.0
    sx = (xc[-1] - xc[0]) / 2.0
    sy = (yc[-1] - yc[0]) / 2.0
    Xn = (X - cx) / sx
    Yn = (Y - cy) / sy

    coords = np.stack([Xn.ravel(), Yn.ravel()], axis=-1).astype(np.float32)
    targets = field.reshape(-1, 2).astype(np.float32)

    coords_t  = torch.from_numpy(coords).to(device)
    targets_t = torch.from_numpy(targets).to(device)

    model = SIREN_INR(hidden_dim=hidden_dim, n_hidden=n_hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for epoch in range(epochs):
        pred = model(coords_t)
        loss = torch.mean((pred - targets_t)**2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if (epoch+1) % 500 == 0:
            print(f"      INR epoch {epoch+1}/{epochs}  loss={loss.item():.6e}")

    model.eval()
    with torch.no_grad():
        pred = model(coords_t).cpu().numpy()
    mse = float(np.mean((pred - targets)**2))
    return model, mse, (cx, cy, sx, sy)


def predict_inr(model, xc, yc, norm_params, device='cpu'):
    cx, cy, sx, sy = norm_params
    X, Y = np.meshgrid(xc, yc)
    Xn = (X - cx) / sx
    Yn = (Y - cy) / sy
    coords = np.stack([Xn.ravel(), Yn.ravel()], axis=-1).astype(np.float32)
    coords_t = torch.from_numpy(coords).to(device)
    model.eval()
    with torch.no_grad():
        pred = model(coords_t).cpu().numpy()
    return pred.reshape(len(yc), len(xc), 2)


# ════════════════════════════════════════════════════════════════════════════
#  §4  Visualisation helpers
# ════════════════════════════════════════════════════════════════════════════

def _quiver(ax, xc, yc, field, title, skip=4):
    """Draw a quiver plot on the given axes."""
    X, Y = np.meshgrid(xc, yc)
    U = field[:, :, 0]
    V = field[:, :, 1]
    mag = np.sqrt(U**2 + V**2)
    ax.quiver(X[::skip, ::skip], Y[::skip, ::skip],
              U[::skip, ::skip], V[::skip, ::skip],
              mag[::skip, ::skip], cmap='coolwarm', scale=None)
    ax.set_title(title, fontsize=9)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=6)


def plot_comparison_for_field(name, gt, pred_ga, pred_bb, pred_inr,
                              xc, yc, out_dir):
    """Create a 2×2 panel: GT / GaussAff / BlobBS / INR."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    _quiver(axes[0, 0], xc, yc, gt, f"{name}\n(Ground Truth)")
    _quiver(axes[0, 1], xc, yc, pred_ga, "Gauss-Vortex + Affine fit")
    _quiver(axes[1, 0], xc, yc, pred_bb, "Vorticity Blob / Biot-Savart fit")
    _quiver(axes[1, 1], xc, yc, pred_inr, "SIREN INR fit")
    plt.tight_layout()
    safe = name.replace(' ', '_').replace('/', '_')
    fpath = os.path.join(out_dir, f"comparison_{safe}.png")
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"  ✓ saved {fpath}")


def plot_error_fields(name, gt, pred_ga, pred_bb, pred_inr, xc, yc, out_dir):
    """Plot per-point error magnitude for all three methods."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    X, Y = np.meshgrid(xc, yc)
    for ax, pred, label in zip(
            axes,
            [pred_ga, pred_bb, pred_inr],
            ["GaussVortex+Affine", "VorticityBlob/BS", "SIREN INR"]):
        err = np.sqrt(np.sum((pred - gt)**2, axis=-1))
        c = ax.pcolormesh(X, Y, err, cmap='hot', shading='auto')
        fig.colorbar(c, ax=ax, fraction=0.046)
        ax.set_title(f"{label}\nmax err={err.max():.4f}", fontsize=9)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=6)
    fig.suptitle(f"Error fields — {name}", fontsize=11)
    plt.tight_layout()
    safe = name.replace(' ', '_').replace('/', '_')
    fpath = os.path.join(out_dir, f"error_{safe}.png")
    plt.savefig(fpath, dpi=150)
    plt.close()


def plot_mse_bar(results, out_dir):
    """Grouped bar chart: MSE comparison across fields and methods."""
    names   = [r['name'] for r in results]
    mse_ga  = [r['mse_gaussaff'] for r in results]
    mse_bb  = [r['mse_blobbs'] for r in results]
    mse_inr = [r['mse_inr'] for r in results]

    x = np.arange(len(names))
    w = 0.25
    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - w, mse_ga,  w, label='GaussVortex+Affine', color='#4C72B0')
    bars2 = ax.bar(x,     mse_bb,  w, label='VorticityBlob/BS',   color='#DD8452')
    bars3 = ax.bar(x + w, mse_inr, w, label='SIREN INR',          color='#55A868')
    ax.set_yscale('log')
    ax.set_ylabel('MSE (log scale)')
    ax.set_title('Fitting Accuracy Comparison: Analytical vs. INR')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha='right', fontsize=8)
    ax.legend()
    plt.tight_layout()
    fpath = os.path.join(out_dir, "mse_comparison_bar.png")
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"  ✓ saved {fpath}")


def print_summary_table(results):
    """Pretty-print MSE table."""
    hdr = f"{'Field':<30s}  {'GaussAff':>12s}  {'BlobBS':>12s}  {'SIREN INR':>12s}"
    print("\n" + "="*70)
    print(hdr)
    print("-"*70)
    for r in results:
        print(f"{r['name']:<30s}  {r['mse_gaussaff']:12.6e}  {r['mse_blobbs']:12.6e}  {r['mse_inr']:12.6e}")
    print("="*70 + "\n")


# ════════════════════════════════════════════════════════════════════════════
#  §5  Main experiment loop
# ════════════════════════════════════════════════════════════════════════════

def main():
    out_dir = os.path.join(os.path.dirname(__file__), 'outputs', 'fit_steady_experiment')
    os.makedirs(out_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[experiment] device = {device}")

    fields = generate_all_fields()
    results = []

    for i, (name, gt_field, xc, yc) in enumerate(fields):
        print(f"\n[{i+1}/{len(fields)}] ── {name} ──")

        # ── (A) Gaussian-vortex + affine ────────────────────────────────
        print("  Fitting Gaussian-vortex + affine ...")
        params_ga, mse_ga = fit_gaussaff(gt_field, xc, yc)
        pred_ga = predict_gaussaff(params_ga, xc, yc)
        print(f"    MSE = {mse_ga:.6e}   (#params = {len(params_ga)})")

        # ── (B) Vorticity blob / Biot-Savart ────────────────────────────
        print("  Fitting Vorticity blob / Biot-Savart ...")
        params_bb, mse_bb = fit_blobbs(gt_field, xc, yc)
        pred_bb = predict_blobbs(params_bb, xc, yc)
        print(f"    MSE = {mse_bb:.6e}   (#params = {len(params_bb)})")

        # ── (C) SIREN INR ───────────────────────────────────────────────
        print("  Fitting SIREN INR ...")
        model_inr, mse_inr, norm_p = fit_inr(gt_field, xc, yc,
                                              epochs=2000, lr=1e-4,
                                              device=device)
        pred_inr = predict_inr(model_inr, xc, yc, norm_p, device=device)
        n_inr_params = sum(p.numel() for p in model_inr.parameters())
        print(f"    MSE = {mse_inr:.6e}   (#params = {n_inr_params})")

        # ── plots ───────────────────────────────────────────────────────
        plot_comparison_for_field(name, gt_field, pred_ga, pred_bb,
                                 pred_inr, xc, yc, out_dir)
        plot_error_fields(name, gt_field, pred_ga, pred_bb,
                          pred_inr, xc, yc, out_dir)

        results.append({
            'name': name,
            'mse_gaussaff': mse_ga,
            'mse_blobbs': mse_bb,
            'mse_inr': mse_inr,
            'n_params_ga': len(params_ga),
            'n_params_bb': len(params_bb),
            'n_params_inr': n_inr_params,
        })

    # ── aggregate plots ─────────────────────────────────────────────────
    plot_mse_bar(results, out_dir)
    print_summary_table(results)

    # ── save numerical results ──────────────────────────────────────────
    import json
    res_path = os.path.join(out_dir, 'results.json')
    with open(res_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ numerical results → {res_path}")
    print("\n[experiment] done.")


if __name__ == '__main__':
    main()
