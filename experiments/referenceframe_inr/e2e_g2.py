import sys, numpy as np
sys.path.insert(0, r"C:\Users\xingdi\sources\PyflowVis")
from FLowUtils.VectorField2d import UnsteadyVectorField2D
from FLowUtils.ReferenceFrameOptimization import gunther17_optimization_2d, _dvdt, _jacobian_2d
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
    data[t,:,:,0]= c*sx-s*sy+Omega*(-Yg); data[t,:,:,1]= s*sx+c*sy+Omega*( Xg)
f.field=data.astype(np.float32)
rg=gunther17_optimization_2d(f,neighborhood=5)
ug=np.asarray(rg.u_field.field,np.float64)
u_true=np.zeros_like(data); u_true[...,0]=Omega*(-Yg); u_true[...,1]=Omega*Xg
err=np.linalg.norm(ug-u_true,axis=-1)[T//2]   # (Y,X) error map at mid time
print("u-error percentiles over full grid:", np.round(np.percentile(err,[50,90,99,100]),4))
for pad in (0,4,8,12,16):
    e=err[pad:Ny-pad,pad:Nx-pad] if pad>0 else err
    print(f"  crop pad={pad:2d}: max u-err={e.max():.4f}  mean={e.mean():.4f}")
# where is the max?
iy,ix=np.unravel_index(np.argmax(err),err.shape)
print(f"max error at (iy={iy},ix={ix}) => {'BOUNDARY' if (iy<6 or ix<6 or iy>Ny-7 or ix>Nx-7) else 'INTERIOR'}")

# observed-td restricted to deep interior:
def otd(v,u,pad):
    dt=float(f.timeInterval); dx,dy=f.gridInterval; vt=_dvdt(v,dt); ut=_dvdt(u,dt); tot=0.0
    for t in range(v.shape[0]):
        Jv=_jacobian_2d(v[t],dx,dy); Ju=_jacobian_2d(u[t],dx,dy)
        r=vt[t]-ut[t]+np.einsum("yxik,yxk->yxi",Jv,u[t])-np.einsum("yxik,yxk->yxi",Ju,v[t])
        tot+=float((r[pad:-pad,pad:-pad]**2).sum())
    return tot
for pad in (4,8,12,16):
    print(f"  observed-td gunther deep-interior pad={pad}: {otd(data,ug,pad):.4e}  (true-u: {otd(data,u_true,pad):.4e})")
