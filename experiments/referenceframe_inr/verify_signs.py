import sympy as sp

# ─────────────────────────────────────────────────────────────────────────
# 2D killing LSQ sign audit (Finding #2)
#
# observed time derivative, LOW order (drop du/dt), from doc section 0:
#   r = dv/dt + (grad v) u - (grad u) v
# killing observer (doc 1.3 / 1.4):  u = (a,b) + c*(-y, x)
# We want to write r = A q + (dv/dt)  with q=(a,b,c), then the LSQ RHS
# for   min ||A q - b||^2  is  b = -dv/dt  (since r=0 => A q = -dv/dt).
# QUESTION: is the omega column  (J*Xp - Vp)  or  (Vp - J*Xp)?
# ─────────────────────────────────────────────────────────────────────────

x, y = sp.symbols('x y', real=True)
a, b, c = sp.symbols('a b c', real=True)
# velocity field components as generic functions of x,y (and t implicitly)
vx = sp.Function('vx')(x, y)
vy = sp.Function('vy')(x, y)
vxt, vyt = sp.symbols('vxt vyt', real=True)  # dv/dt (time deriv, independent of the above)

v = sp.Matrix([vx, vy])
vt = sp.Matrix([vxt, vyt])

# Jacobian J[i,j] = d v_i / d x_j   (col0 = d/dx, col1 = d/dy)  -- matches code _jacobian_2d
J = sp.Matrix([[sp.diff(vx, x), sp.diff(vx, y)],
               [sp.diff(vy, x), sp.diff(vy, y)]])

# killing observer u = (a - c y, b + c x)
u = sp.Matrix([a - c*y, b + c*x])

# grad u : (grad u)[i,j] = d u_i / d x_j
gradu = sp.Matrix([[sp.diff(u[0], x), sp.diff(u[0], y)],
                   [sp.diff(u[1], x), sp.diff(u[1], y)]])

# observed time derivative (low order)
r = vt + J*u - gradu*v
r = sp.simplify(r)

# Now express r as  M*(a,b,c) + vt   and read the columns.
q = sp.Matrix([a, b, c])
# r is affine in (a,b,c). Extract coefficient matrix M (2x3) = d r / d q
M = sp.Matrix([[sp.diff(r[i], q[j]) for j in range(3)] for i in range(2)])
M = sp.simplify(M)

# constant part (a=b=c=0)
const = sp.simplify(r.subs({a:0,b:0,c:0}))

print("=== r = M*(a,b,c) + const ===")
print("M (coeff of a,b,c):")
sp.pprint(M)
print("const part:")
sp.pprint(const)

# Build the doc's candidate columns
Xp = sp.Matrix([-y, x])           # position perp
Vp = sp.Matrix([-vy, vx])         # velocity perp
JXp = J*Xp
col_JXp_minus_Vp = sp.simplify(JXp - Vp)
col_Vp_minus_JXp = sp.simplify(Vp - JXp)

print("\n=== omega column comparison ===")
print("M[:,2] (actual coeff of c in r):")
sp.pprint(sp.simplify(M[:,2]))
print("J*Xp - Vp :")
sp.pprint(col_JXp_minus_Vp)
print("Vp - J*Xp :")
sp.pprint(col_Vp_minus_JXp)

print("\nM[:,2] == (J*Xp - Vp) ? ",
      sp.simplify(M[:,2] - col_JXp_minus_Vp) == sp.zeros(2,1))
print("M[:,2] == (Vp - J*Xp) ? ",
      sp.simplify(M[:,2] - col_Vp_minus_JXp) == sp.zeros(2,1))

print("\nconst == vt ? ", sp.simplify(const - vt) == sp.zeros(2,1))

# So r = [J | M[:,2]] q + vt.  Setting r=0 => [J | M[:,2]] q = -vt.
# i.e. LSQ  A q = b  with A=[J | M[:,2]], b = -vt.
# decompose claims A = [dv/dx | dv/dy | J*Xp - Vp], b = -vt.
# NB dv/dx = J[:,0], dv/dy = J[:,1].
print("\n=== decompose A columns check ===")
print("A[:,0] should be dv/dx = J[:,0]:", sp.simplify(M[:,0] - J[:,0]) == sp.zeros(2,1))
print("A[:,1] should be dv/dy = J[:,1]:", sp.simplify(M[:,1] - J[:,1]) == sp.zeros(2,1))
