"""AUDIT: numerically verify pushforward forward/inverse round-trip + param formula.
Pure numpy, no torch/CUDA needed. Does NOT import the experiment modules (avoids side effects)."""
import numpy as np, math

# ---------- 1. Round-trip: replicate fit_region forward/inverse EXACTLY ----------
# Forward (from fit_region):
#   cth=cos(-th), sth=sin(-th)
#   xi_x = cth*xr + (-sth)*yr - Dx ;  xi_y = sth*xr + cth*yr - Dy
#   ux = a - c*yr ; uy = b + c*xr
#   mx = v - u
#   vco_x = cth*mx + (-sth)*my ; vco_y = sth*mx + cth*my
# Inverse (reconstruction):
#   cB=cos(th), sB=sin(th)
#   vx = cB*vco_x + (-sB)*vco_y + ux ; vy = sB*vco_x + cB*vco_y + uy
# Test: if the INR is PERFECT (vco_r == vco), does (vx,vy) == original v?

rng = np.random.default_rng(0)
T, npix = 5, 7
theta = rng.uniform(-2, 2, T)
a = rng.uniform(-1, 1, T); b = rng.uniform(-1, 1, T); c = rng.uniform(-1, 1, T)
xr = rng.uniform(-3, 3, npix); yr = rng.uniform(-3, 3, npix)
vx0 = rng.uniform(-1, 1, (T, npix)); vy0 = rng.uniform(-1, 1, (T, npix))

cth = np.cos(-theta)[:, None]; sth = np.sin(-theta)[:, None]
ux = a[:, None] - c[:, None] * yr[None]
uy = b[:, None] + c[:, None] * xr[None]
mx = vx0 - ux; my = vy0 - uy
vco_x = cth * mx + (-sth) * my
vco_y = sth * mx + cth * my

# perfect INR: recon co-moving == forward co-moving
cB = np.cos(theta)[:, None]; sB = np.sin(theta)[:, None]
vx_rec = cB * vco_x + (-sB) * vco_y + ux
vy_rec = sB * vco_x + cB * vco_y + uy

err = max(np.abs(vx_rec - vx0).max(), np.abs(vy_rec - vy0).max())
print(f"[1] pushforward round-trip max|v_rec - v| = {err:.3e}  ->", "OK" if err < 1e-12 else "*** MISMATCH ***")

# Also confirm R(-th) then R(+th) is identity (sign pairing) directly
R_m = np.array([[math.cos(-0.7), -math.sin(-0.7)], [math.sin(-0.7), math.cos(-0.7)]])
R_p = np.array([[math.cos(0.7), -math.sin(0.7)], [math.sin(0.7), math.cos(0.7)]])
print(f"[1b] R(+th)R(-th)=I residual = {np.abs(R_p@R_m - np.eye(2)).max():.3e}")

# ---------- 2. FFReLU param count formula ----------
# FFReLU: B is a buffer (in_dim x n_freq), NOT a parameter (not counted).
# net = Linear(2*n_freq, h) + [Linear(h,h)]*layers + Linear(h, out)
# params = (2*n_freq*h + h)                      # first
#        + layers*(h*h + h)                      # hidden blocks
#        + (h*out + out)                         # final
def ffrelu_params(h, out=2, n_freq=256, layers=4):
    p = (2*n_freq*h + h) + layers*(h*h + h) + (h*out + out)
    return p
# closed form the prompt claims: 4h^2 + 514h  (for n_freq=256, layers=4, ignoring +out terms & bias const)
def claimed(h):
    return 4*h*h + 514*h
for h in [256, 384, 512, 930, 1343]:
    exact = ffrelu_params(h)
    print(f"[2] h={h:4d}: exact_params={exact:>10,} ({exact/1e6:.3f}M)   claimed 4h^2+514h={claimed(h):>10,} ({claimed(h)/1e6:.3f}M)  diff={exact-claimed(h)}")

# Budget targets
print("\n[2b] budget check:")
for tag, h in [("3.94M?",930), ("7.89M?",1343)]:
    print(f"    hidden={h}: {ffrelu_params(h)/1e6:.3f}M  ({tag})")
# proposed side: N INRs of hidden H
for N, h in [(3,512),(5,384),(10,256)]:
    tot = N*ffrelu_params(h)
    print(f"    proposed cut={N:2d} h={h}: {N} x {ffrelu_params(h)/1e6:.3f}M = {tot/1e6:.3f}M")

# ---------- 3. integrate_frame_motion: trapezoid sanity for constant c ----------
# If c is constant = c0, theta(t) should be c0 * t  (t measured from t0=times[0]).
times = np.linspace(0, 3, 7)
coeff = np.zeros((7,3)); coeff[:,2] = 0.5  # constant rotation rate
# replicate integrate_frame_motion
def integrate_frame_motion(coeff, times):
    T = len(times); a,b,c = coeff[:,0], coeff[:,1], coeff[:,2]
    theta = np.zeros(T); D = np.zeros((T,2))
    def Rm(th,vx,vy):
        ct,st = math.cos(-th), math.sin(-th); return ct*vx-st*vy, st*vx+ct*vy
    for k in range(1,T):
        dt = times[k]-times[k-1]
        theta[k] = theta[k-1] + 0.5*(c[k]+c[k-1])*dt
        f0 = Rm(theta[k-1], a[k-1], b[k-1]); f1 = Rm(theta[k], a[k], b[k])
        D[k,0] = D[k-1,0] + 0.5*(f0[0]+f1[0])*dt
        D[k,1] = D[k-1,1] + 0.5*(f0[1]+f1[1])*dt
    return theta, D
th, D = integrate_frame_motion(coeff, times)
expected = 0.5 * (times - times[0])
print(f"\n[3] theta(const c=0.5) max err vs c*t = {np.abs(th-expected).max():.3e}  D (a=b=0) max = {np.abs(D).max():.3e}")
