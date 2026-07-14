import sys, numpy as np
sys.path.insert(0, r"C:\Users\xingdi\sources\PyflowVis")
from FLowUtils.VectorField2d import UnsteadyVectorField2D
from FLowUtils.ReferenceFrameOptimization import (
    killing_optimization_2d, gunther17_optimization_2d, _dvdt, _jacobian_2d)

# ── Genuinely unsteady lab field that is STEADY in a rotating frame. ──
# Steady base field in the rotating (observed) coords:  s(p) = a saddle-ish steady flow.
# Frame rotates at constant angular velocity Omega about origin:
#   x_lab(t) = R(Omega t) p        (observer worldline: pure rotation)
# A vector field steady in the rotating frame appears in the lab as:
#   v(x,t) = R(Omega t) s( R(-Omega t) x ) + Omega_perp(x)
#   where the Omega_perp = Omega*(-y,x) is the frame drag (killing observer u).
# Standard construction (Gunther GOV benchmark): pick steady s, then
#   v(x,t) = Rot . s(Rot^T x) + u(x),  u = Omega*(-y,x).
# Then the TRUE observer is u = Omega*(-y,x) (a=b=0, c=Omega), and v-u pushed
# back is steady. dv/dt is nonzero (rotation of s), so the LSQ is non-trivial.
Nx=Ny=61; T=9
L=2.0
Omega=0.6
f=UnsteadyVectorField2D(Nx,Ny,T,[-L,-L,0.0],[L,L,2*np.pi/ Omega * 0.4])  # partial period
xs=-L+f.gridInterval[0]*np.arange(Nx); ys=-L+f.gridInterval[1]*np.arange(Ny)
Yg,Xg=np.meshgrid(ys,xs,indexing="ij")

def steady_s(px,py):
    # a non-radially-symmetric steady field so rotation actually changes it in lab frame
    sx = -py + 0.3*px         # rotation + slight source
    sy =  px + 0.2*py*py      # nonlinear term to break symmetry
    return sx,sy

data=np.zeros((T,Ny,Nx,2),np.float64)
for t in range(T):
    th=Omega*f.timeInterval*t
    c,s=np.cos(th),np.sin(th)
    # pull back x to observed coords: p = R(-th) x
    px= c*Xg + s*Yg
    py=-s*Xg + c*Yg
    sx,sy=steady_s(px,py)
    # rotate the steady vector back to lab: R(th) s
    vx_obs= c*sx - s*sy
    vy_obs= s*sx + c*sy
    # add frame drag u = Omega*(-y, x)
    data[t,:,:,0]=vx_obs + Omega*(-Yg)
    data[t,:,:,1]=vy_obs + Omega*( Xg)
f.field=data.astype(np.float32)

def observed_td_energy(v,u,label):
    dt=float(f.timeInterval); dx,dy=f.gridInterval
    vt=_dvdt(v,dt); ut=_dvdt(u,dt); tot=0.0
    for t in range(v.shape[0]):
        Jv=_jacobian_2d(v[t],dx,dy); Ju=_jacobian_2d(u[t],dx,dy)
        r=vt[t]-ut[t]+np.einsum("yxik,yxk->yxi",Jv,u[t])-np.einsum("yxik,yxk->yxi",Ju,v[t])
        tot+=float((r[3:-3,3:-3]**2).sum())
    print(f"  observed-td energy [{label}] = {tot:.6e}")
    return tot

raw=float((_dvdt(data,f.timeInterval)[:,3:-3,3:-3]**2).sum())
print(f"raw ||dv/dt||^2 interior = {raw:.6e}   (nonzero => genuinely unsteady)")

# baseline: u=0 observed td == raw
zero_u=np.zeros_like(data)
print("\nbaseline u=0:"); observed_td_energy(data,zero_u,"u=0")

rk=killing_optimization_2d(f,boundary_skip=3)
print("\n[killing_2d] recovered (a,b,c) per frame (true ~ (0,0,%.3f)):"%Omega)
print(np.round(rk.params,4))
uk=np.asarray(rk.u_field.field,np.float64)
observed_td_energy(data,uk,"killing")

# WRONG-SIGN control: build observer with c of FLIPPED sign to prove sign matters
u_wrong=np.zeros_like(data)
u_wrong[...,0]= -(-Omega*Yg)   # c=-Omega
u_wrong[...,1]= -( Omega*Xg)
print("\n[control] observer with WRONG-sign c=-Omega:")
observed_td_energy(data,u_wrong,"wrong-sign c")

rg=gunther17_optimization_2d(f,neighborhood=4)
ug=np.asarray(rg.u_field.field,np.float64)
print("\n[gunther17_2d] recovered observer sampled at (interior center):")
cy,cx=Ny//2,Nx//2
print("  u_gunther(center,t=T//2) =",np.round(ug[T//2,cy,cx],4),
      " ; true u(center)=",np.round([Omega*(-ys[cy]),Omega*xs[cx]],4))
observed_td_energy(data,ug,"gunther")
