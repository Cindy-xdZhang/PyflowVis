import sys, numpy as np
sys.path.insert(0, r"C:\Users\xingdi\sources\PyflowVis")
from FLowUtils.VectorField2d import UnsteadyVectorField2D
from FLowUtils.ReferenceFrameOptimization import (
    killing_optimization_2d, gunther17_optimization_2d, _dvdt, _jacobian_2d)
from FLowUtils.referenceFrameDecompose import decompose_reference_frame_2d

# Purely LINEAR steady base (so v is exactly a low-order-representable unsteady
# field; rotation of a linear field stays linear => no higher-order discretization
# error, isolating the SIGN convention only).
Nx=Ny=61; T=9; L=2.0; Omega=0.5
f=UnsteadyVectorField2D(Nx,Ny,T,[-L,-L,0.0],[L,L,2.5])
xs=-L+f.gridInterval[0]*np.arange(Nx); ys=-L+f.gridInterval[1]*np.arange(Ny)
Yg,Xg=np.meshgrid(ys,xs,indexing="ij")
# steady LINEAR base s(p) = A p with a generic 2x2 A (non-symmetric)
A=np.array([[0.4,-0.9],[1.1,0.15]])
data=np.zeros((T,Ny,Nx,2),np.float64)
for t in range(T):
    th=Omega*f.timeInterval*t; c,s=np.cos(th),np.sin(th)
    px= c*Xg+s*Yg; py=-s*Xg+c*Yg
    sx=A[0,0]*px+A[0,1]*py; sy=A[1,0]*px+A[1,1]*py
    vx= c*sx - s*sy + Omega*(-Yg)
    vy= s*sx + c*sy + Omega*( Xg)
    data[t,:,:,0]=vx; data[t,:,:,1]=vy
f.field=data.astype(np.float32)

def otd(v,u,label,pad=4):
    dt=float(f.timeInterval); dx,dy=f.gridInterval
    vt=_dvdt(v,dt); ut=_dvdt(u,dt); tot=0.0
    for t in range(v.shape[0]):
        Jv=_jacobian_2d(v[t],dx,dy); Ju=_jacobian_2d(u[t],dx,dy)
        r=vt[t]-ut[t]+np.einsum("yxik,yxk->yxi",Jv,u[t])-np.einsum("yxik,yxk->yxi",Ju,v[t])
        tot+=float((r[pad:-pad,pad:-pad]**2).sum())
    print(f"  observed-td [{label}] = {tot:.6e}"); return tot

raw=float((_dvdt(data,f.timeInterval)[:,4:-4,4:-4]**2).sum())
print(f"raw ||dv/dt||^2 = {raw:.6e}")
rk=killing_optimization_2d(f,boundary_skip=4)
print("killing c per frame (true=%.3f):"%Omega, np.round(rk.params[:,2],4))
otd(data,np.asarray(rk.u_field.field,np.float64),"killing")
rg=gunther17_optimization_2d(f,neighborhood=5)
ug=np.asarray(rg.u_field.field,np.float64)
cy,cx=Ny//2,Nx//2
# true observer at every point: u=Omega*(-y,x)
u_true=np.zeros_like(data); u_true[...,0]=Omega*(-Yg); u_true[...,1]=Omega*Xg
print("gunther max|u-u_true| interior =", np.abs((ug-u_true)[:,8:-8,8:-8]).max())
otd(data,ug,"gunther")
otd(data,u_true,"exact-true-u")

# ── Decompose: single global observer should already explain it (1 flow) ──
print("\n[decompose] single-flow field: expect global observer ~ finest (benefit ~0):")
dec=decompose_reference_frame_2d(f,k=4,verbose=False)
for kk in ("raw_td_energy","global_observer_residual","finest_observer_residual",
           "global_residual_ratio","decomposition_benefit"):
    print(f"   {kk} = {dec.diag[kk]:.6e}")
# region observer for the trivial 1-region cut must recover c=Omega:
lab=dec.cut(n_regions=1)
obs=dec.region_observers(lab)[0]
print("   1-region observer c per frame:", np.round(obs[:,2],4), "(true=%.3f)"%Omega)
