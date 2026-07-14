import sys, numpy as np
sys.path.insert(0, r"C:\Users\xingdi\sources\PyflowVis")
from FLowUtils.VectorField2d import UnsteadyVectorField2D
from FLowUtils.ReferenceFrameOptimization import (
    killing_optimization_2d, gunther17_optimization_2d, _dvdt, _jacobian_2d, _coords_2d)

# ── Build a field whose OBJECTIVE (observed) flow is a fixed steady field,
#    seen from a uniformly ROTATING + TRANSLATING observer (a true killing frame).
# Construct v(x,t) = u(x,t) + s(x)   where
#   u = killing observer with constant params (a,b,c):  u = (a - c y, b + c x)
#   s = an arbitrary STEADY field (so v-u = s is steady => observed td small).
# For a killing u with CONSTANT (a,b,c), du/dt=0, so low-order == high-order and
# the observed td of v w.r.t. u is exactly the (steady) advective residual of s,
# which need NOT be zero pointwise; but the LSQ observer that best kills dv/dt
# should recover THIS (a,b,c) if s is chosen steady & the rotation dominates.
# Cleanest test: make s == 0 so v == u exactly. Then the perfect observer is u,
# v-u=0, and residual must be ~0. Both methods must recover u (=> v_hat ~ 0).
Nx=Ny=41; T=7
xmin,xmax=-2.0,2.0; ymin,ymax=-2.0,2.0; tmin,tmax=0.0,3.0
a_true,b_true,c_true = 0.3, -0.2, 0.7   # translation + angular velocity (constant)
f = UnsteadyVectorField2D(Nx,Ny,T,[xmin,ymin,tmin],[xmax,ymax,tmax])
xs = xmin + f.gridInterval[0]*np.arange(Nx)
ys = ymin + f.gridInterval[1]*np.arange(Ny)
Yg,Xg = np.meshgrid(ys,xs,indexing="ij")
data = np.zeros((T,Ny,Nx,2),np.float64)
for t in range(T):
    data[t,:,:,0] = a_true - c_true*Yg
    data[t,:,:,1] = b_true + c_true*Xg
f.field = data.astype(np.float32)

def observed_td_energy(vfield, ufield):
    """Sum |dv/dt - du/dt + Jv u - Ju v|^2 over interior, per the doc D (HIGH order)."""
    v = np.asarray(vfield.field,np.float64); u = np.asarray(ufield.field,np.float64)
    dt=float(f.timeInterval); dx,dy=f.gridInterval
    vt=_dvdt(v,dt); ut=_dvdt(u,dt)
    tot=0.0
    for t in range(v.shape[0]):
        Jv=_jacobian_2d(v[t],dx,dy); Ju=_jacobian_2d(u[t],dx,dy)
        Jvu=np.einsum("yxik,yxk->yxi",Jv,u[t]); Juv=np.einsum("yxik,yxk->yxi",Ju,v[t])
        r = vt[t]-ut[t]+Jvu-Juv
        tot += float((r[2:-2,2:-2]**2).sum())
    return tot

raw = float((_dvdt(data,f.timeInterval)[:, 2:-2,2:-2]**2).sum())
print(f"raw ||dv/dt||^2 (interior) = {raw:.6e}")

rk = killing_optimization_2d(f, boundary_skip=2)
print("\n[killing_2d] recovered params (a,b,c) per frame:")
print(np.round(rk.params,4))
print("true (a,b,c) =", (a_true,b_true,c_true))
vhat_k = np.asarray(rk.v_minus_u_field.field,np.float64)
print("max|v-u| (killing) =", np.abs(vhat_k).max(), "(should be ~0)")
print("observed-td energy (killing) =", observed_td_energy(f, rk.u_field))

rg = gunther17_optimization_2d(f, neighborhood=3)
vhat_g = np.asarray(rg.v_minus_u_field.field,np.float64)
u_g = np.asarray(rg.u_field.field,np.float64)
print("\n[gunther17_2d] max|v-u| =", np.abs(vhat_g[:,4:-4,4:-4]).max(), "(should be ~0 interior)")
# check recovered observer matches true u
u_true = data.copy()
print("max|u_gunther - u_true| interior =", np.abs((u_g-u_true)[:,4:-4,4:-4]).max())
print("observed-td energy (gunther) =", observed_td_energy(f, rg.u_field))
