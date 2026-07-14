import sys, numpy as np
sys.path.insert(0, r"C:\Users\xingdi\sources\PyflowVis")
from FLowUtils.VectorField2d import UnsteadyVectorField2D
from FLowUtils.ReferenceFrameOptimization import gunther17_optimization_2d, _dvdt, _jacobian_2d, _coords_2d, _windowed_sum_nd, _batched_solve

# Reproduce ONE Gunther per-point solve by hand on the same field and inspect uu.
Nx=Ny=61; T=9; L=2.0; Omega=0.5
f=UnsteadyVectorField2D(Nx,Ny,T,[-L,-L,0.0],[L,L,2.5])
xs=-L+f.gridInterval[0]*np.arange(Nx); ys=-L+f.gridInterval[1]*np.arange(Ny)
Yg,Xg=np.meshgrid(ys,xs,indexing="ij")
A=np.array([[0.4,-0.9],[1.1,0.15]])
data=np.zeros((T,Ny,Nx,2),np.float64)
for t in range(T):
    th=Omega*f.timeInterval*t; c,s=np.cos(th),np.sin(th)
    px= c*Xg+s*Yg; py=-s*Xg+c*Yg
    sx=A[0,0]*px+A[0,1]*py; sy=A[1,0]*px+A[1,1]*py
    data[t,:,:,0]= c*sx-s*sy+Omega*(-Yg)
    data[t,:,:,1]= s*sx+c*sy+Omega*( Xg)
f.field=data.astype(np.float32)

# True observer u = Omega*(-y,x). Gunther reconstructs v_hat = v + (uu1,uu2) - uu0*Xp,
# and observer u_recon = v - v_hat = uu0*Xp - (uu1,uu2). For u_recon to equal
# Omega*(-y,x) = Omega*Xp we need uu0=Omega, (uu1,uu2)=0.  Check the SOLVED uu0 sign.
dx,dy=f.gridInterval; dt=float(f.timeInterval)
vt_all=_dvdt(data,dt); Xp=np.stack([-Yg,Xg],-1)
t=T//2; slice_v=data[t]; J=_jacobian_2d(slice_v,dx,dy); vt=vt_all[t]
Vp=np.stack([-slice_v[:,:,1],slice_v[:,:,0]],-1)
Jxpvp=-np.einsum("yxik,yxk->yxi",J,Xp)+Vp
M=np.zeros((Ny,Nx,2,6))
M[:,:,0,0]=Jxpvp[:,:,0]; M[:,:,1,0]=Jxpvp[:,:,1]
M[:,:,0,1]=J[:,:,0,0]; M[:,:,0,2]=J[:,:,0,1]; M[:,:,1,1]=J[:,:,1,0]; M[:,:,1,2]=J[:,:,1,1]
M[:,:,0,3]=1.0; M[:,:,1,4]=1.0; M[:,:,0,5]=Xp[:,:,0]; M[:,:,1,5]=Xp[:,:,1]
MTM=np.einsum("yxki,yxkj->yxij",M,M); MTb=np.einsum("yxki,yxk->yxi",M,vt)
for U in (3,5,8,12):
    Mw=_windowed_sum_nd(MTM,U,2); bw=_windowed_sum_nd(MTb,U,2)
    uu=_batched_solve(Mw,bw)
    cy,cx=Ny//2,Nx//2
    print(f"U={U:2d}  uu0(center)={uu[cy,cx,0]:+.4f} (expect +{Omega})  "
          f"uu1={uu[cy,cx,1]:+.4f} uu2={uu[cy,cx,2]:+.4f} (expect 0)  "
          f"u_recon={uu[cy,cx,0]*Xp[cy,cx,0]-uu[cy,cx,1]:+.3f},{uu[cy,cx,0]*Xp[cy,cx,1]-uu[cy,cx,2]:+.3f} "
          f"(true {Omega*Xp[cy,cx,0]:+.3f},{Omega*Xp[cy,cx,1]:+.3f})")
# center is (0,0) so Xp=0 there -> pick an off-center interior point instead
cy,cx=Ny//2+10,Nx//2+7
print(f"\noff-center point (x={xs[cx]:.2f},y={ys[cy]:.2f}), Xp={Xp[cy,cx]}")
for U in (5,8,12):
    Mw=_windowed_sum_nd(MTM,U,2); bw=_windowed_sum_nd(MTb,U,2)
    uu=_batched_solve(Mw,bw)
    ur=uu[cy,cx,0]*Xp[cy,cx]-uu[cy,cx,1:3]
    print(f"U={U:2d}  uu0={uu[cy,cx,0]:+.4f}(exp+{Omega}) uu1={uu[cy,cx,1]:+.4f} uu2={uu[cy,cx,2]:+.4f}"
          f"  u_recon=({ur[0]:+.3f},{ur[1]:+.3f}) true=({Omega*Xp[cy,cx,0]:+.3f},{Omega*Xp[cy,cx,1]:+.3f})")
