"""AUDIT: does monkeypatching DEC.fit_inr actually change what fit_region calls?
fit_region body calls bare name `fit_inr` -> resolves in rfo_decompose_inr module globals.
Patching DEC.fit_inr (== rfo_decompose_inr.fit_inr) rebinds that global, so fit_region SHOULD pick it up.
Verify by patching with a sentinel and calling fit_region's code path (mock the heavy INR)."""
import sys
sys.path.insert(0, r"C:\Users\xingdi\sources\PyflowVis")
sys.path.insert(0, r"C:\Users\xingdi\AppData\Local\Temp\claude\C--Users-xingdi-sources-PyflowVis\5b7dbb49-2a88-462c-b42d-687217cb2b30\scratchpad")
import numpy as np
import rfo_decompose_inr as DEC

called = {"which": None}
def sentinel_fit(coords, values, epochs, m, d, lr, batch, seed=0, omega_0=None):
    called["which"] = "PATCHED"
    N, C = values.shape
    return np.zeros((N, C), np.float32), 0.0, 12345
DEC.fit_inr = sentinel_fit

# Minimal field-like inputs for fit_region
T, Y, X = 4, 8, 8
data = np.random.default_rng(0).standard_normal((T, Y, X, 2))
mask = np.ones((Y, X), bool)
coeff = np.zeros((T, 3)); coeff[:,2] = 0.1
times = np.linspace(0, 1, T)
xph = np.arange(X, dtype=float); yph = np.arange(Y, dtype=float)

ys, xs, vx, vy, tt, npar = DEC.fit_region(data, mask, coeff, times, xph, yph, 10, 0, 0, 1e-3, 1000)
print("fit_region used:", called["which"], "| npar returned:", npar,
      "->", "PATCH REACHED fit_region" if called["which"]=="PATCHED" else "*** PATCH NOT APPLIED ***")

# Also confirm rfo_final2 baseline path (DEC.fit_inr directly) is the same object
print("DEC.fit_inr is sentinel:", DEC.fit_inr is sentinel_fit)
