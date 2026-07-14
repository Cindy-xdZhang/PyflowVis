import sympy as sp
# ─────────────────────────────────────────────────────────────────────────
# Gunther17 2D self-consistency (Finding: is the M/RHS/reconstruction a
# coherent LSQ, regardless of the killing sign convention?)
#
# Code (gunther17_optimization_2d):
#   Vp    = (-vy, vx)
#   Jxpvp = -J*Xp + Vp                          (omega column, = Vp - J*Xp)
#   M = [ Jxpvp | J[:,0] | J[:,1] | e_x | e_y | Xp ]   (2x6)
#   RHS b = +dv/dt
#   solve MTM uu = MTb ; unknown uu=(uu0..uu5)
#   reconstruct v_hat = v + (uu1,uu2) - uu0*Xp
#   => observer u = v - v_hat = -( (uu1,uu2) - uu0*Xp ) = uu0*Xp - (uu1,uu2)
#
# The Gunther objective (doc): minimize D = |dv/dt + grad v . u - grad u . v|^2
#   with u parameterized rigidly u = w*Xp + b_trans, plus fake udot terms.
# Let's confirm the M columns + RHS encode  grad v . u - grad u . v = -dv/dt
# i.e. residual r = dv/dt + Jv.u - gradu.v, and that M*uu (translation+omega
# part) reproduces  -(Jv.u - gradu.v) so that M uu = dv/dt is the same equation.
# ─────────────────────────────────────────────────────────────────────────
x,y = sp.symbols('x y', real=True)
vx = sp.Function('vx')(x,y); vy = sp.Function('vy')(x,y)
vxt,vyt = sp.symbols('vxt vyt', real=True)
J = sp.Matrix([[sp.diff(vx,x),sp.diff(vx,y)],[sp.diff(vy,x),sp.diff(vy,y)]])
Xp = sp.Matrix([-y,x]); Vp = sp.Matrix([-vy,vx]); vt = sp.Matrix([vxt,vyt])

# Gunther omega column as in code:
Jxpvp = -J*Xp + Vp
M_omega = sp.simplify(Jxpvp)

# The observer in Gunther reconstruction: u = uu0*Xp - (uu1,uu2)
# Interpret uu0 = angular vel w, (uu1,uu2)=translation tt. So u = w*Xp - tt.
w, t1, t2 = sp.symbols('w t1 t2', real=True)
u = w*Xp - sp.Matrix([t1,t2])
gradu = u.jacobian([x,y])
# residual (low order, the D being minimized)
r = vt + J*u - gradu*v_ if False else vt + J*u - gradu*sp.Matrix([vx,vy])
r = sp.simplify(r)
# express r = Mcoef*(w,t1,t2) + vt  and read columns
q = sp.Matrix([w,t1,t2])
Mc = sp.Matrix([[sp.diff(r[i],q[j]) for j in range(3)] for i in range(2)])
Mc = sp.simplify(Mc)
const = sp.simplify(r.subs({w:0,t1:0,t2:0}))
print("residual columns for u = w*Xp - (t1,t2):")
print(" coeff of w :", list(sp.simplify(Mc[:,0])))
print(" coeff of t1:", list(Mc[:,1]))
print(" coeff of t2:", list(Mc[:,2]))
print(" const      :", list(const))

# Code's M omega col vs coeff-of-w here:
print("\ncode omega col Jxpvp == coeff_of_w ?", sp.simplify(Mc[:,0]-M_omega)==sp.zeros(2,1))
# Code translation cols are +J[:,0],+J[:,1]; here coeff of t1,t2:
print("code trans col0 (+J[:,0]) == coeff_of_t1 ?", sp.simplify(Mc[:,1]-J[:,0])==sp.zeros(2,1))
print("code trans col1 (+J[:,1]) == coeff_of_t2 ?", sp.simplify(Mc[:,2]-J[:,1])==sp.zeros(2,1))
print("const == +dv/dt (code RHS b=+dv/dt, so M uu = b means residual coeff form matches)?",
      sp.simplify(const - vt)==sp.zeros(2,1))
# Conclusion: r = [Jxpvp | J0 | J1] (w,t1,t2) + dv/dt.  Setting r=0 => M_partial uu = -dv/dt
# but CODE uses RHS = +dv/dt. Check whether that means uu solves -r=0 i.e. same thing.
print("\n--- RHS sign resolution for Gunther ---")
print("r = M_part*(w,t1,t2) + dv/dt.  r=0 => M_part*(w,t1,t2) = -dv/dt.")
print("Code solves M uu = +dv/dt.  If reconstruction maps solved uu to u=w*Xp-(t1,t2)")
print("with (w,t1,t2)=(uu0,uu1,uu2), then code is solving M_part*uu = +dv/dt = -(-dv/dt).")
print("=> code's uu = -(true w,t1,t2). Then reconstructed u_code = uu0*Xp-(uu1,uu2) = -(true u).")
print("   v_hat = v - u_code = v + true_u ??  Let's just check the fixed point directly below.")

# Direct end-to-end numeric check on an analytic rotating field where the TRUE
# observer is known, to see if gunther recon gives v_hat with ~zero observed-td.
