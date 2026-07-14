"""Fourier-features + ReLU MLP INR (stable; replaces SIREN). Deterministic-friendly."""
import numpy as np, math, copy, torch, torch.nn as nn
from rfo_decompose_inr import DEV

class FFReLU(nn.Module):
    def __init__(self, in_dim=3, out_dim=2, hidden=512, layers=4, n_freq=256, scale=16.0, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("B", torch.randn(in_dim, n_freq, generator=g) * scale)
        L = [nn.Linear(2 * n_freq, hidden), nn.ReLU()]
        for _ in range(layers): L += [nn.Linear(hidden, hidden), nn.ReLU()]
        L += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*L)
    def forward(self, x):
        p = 2 * math.pi * (x @ self.B)
        return self.net(torch.cat([torch.sin(p), torch.cos(p)], -1))

def fit_ffrelu(coords, values, epochs, lr, batch, hidden=512, layers=4, n_freq=256, scale=16.0, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    N, C = values.shape
    ct = torch.from_numpy(coords).to(DEV); vt = torch.from_numpy(values).to(DEV)
    model = FFReLU(3, C, hidden, layers, n_freq, scale, seed).to(DEV)
    npar = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr); bs = min(batch, N); steps = max(1, N // bs); lr_f = lr * 0.05
    best = float("inf"); bstate = copy.deepcopy(model.state_dict()); ev = max(1, epochs // 60)
    for ep in range(epochs):
        cur = lr_f + 0.5 * (lr - lr_f) * (1 + math.cos(math.pi * ep / max(1, epochs - 1)))
        for g in opt.param_groups: g["lr"] = cur
        perm = torch.randperm(N, device=DEV)
        for s in range(steps):
            idx = perm[s * bs:(s + 1) * bs]; loss = ((model(ct[idx]) - vt[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if (ep + 1) % ev == 0 or ep == epochs - 1:
            with torch.no_grad(): fl = sum(((model(ct[s:s+400000]) - vt[s:s+400000]) ** 2).sum().item() for s in range(0, N, 400000)) / N
            if fl < best: best = fl; bstate = copy.deepcopy(model.state_dict())
    model.load_state_dict(bstate); model.eval()
    rec = np.empty((N, C), np.float32)
    with torch.no_grad():
        for s in range(0, N, 200000): rec[s:s+200000] = model(ct[s:s+200000]).clamp(-1, 1).cpu().numpy()
    return rec, 0.0, npar
