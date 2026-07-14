import sympy as sp

# ─────────────────────────────────────────────────────────────────────────
# Part B: consistency of the two sign conventions + 3D killing + Gunther RHS
# ─────────────────────────────────────────────────────────────────────────
# From Part A:  r = [J | J*Xp - Vp] * q + dv/dt     (exact, low order)
# The normal-equation solution q* is invariant to an overall sign flip of the
# residual r (min ||r||^2 == min ||-r||^2). Flipping r:
#   -r = [-J | -(J*Xp - Vp)] q - dv/dt = [-J | Vp - J*Xp] q - dv/dt
# So writing "A q = b":
#   Convention 1 (killing, code decompose):  A=[J | J*Xp-Vp], b=-dv/dt
#   Convention 2 (flip):                     A=[-J| Vp-J*Xp], b=+dv/dt  (both cols flip!)
# The doc Rosetta lists ω col = (Vp - J*Xp) WITH RHS b=-dv/dt (line 97+99).
# Is THAT internally a correct pair for the SAME LSQ? Only if the translation
# cols ALSO use -J. Let's check what q you'd get from the doc's stated pair
# (Vp-J*Xp for omega, but +J for translation, b=-dv/dt) vs the true q.
# ─────────────────────────────────────────────────────────────────────────

x, y = sp.symbols('x y', real=True)
vx = sp.Function('vx')(x, y); vy = sp.Function('vy')(x, y)
vxt, vyt = sp.symbols('vxt vyt', real=True)
J = sp.Matrix([[sp.diff(vx,x), sp.diff(vx,y)],[sp.diff(vy,x), sp.diff(vy,y)]])
Xp = sp.Matrix([-y, x]); Vp = sp.Matrix([-vy, vx])
vt = sp.Matrix([vxt, vyt])

# True system (from Part A):
A_true = J.row_join(J*Xp - Vp)          # [J | J*Xp - Vp]
b_true = -vt

# doc-Rosetta-literal system: translation cols = J (line 98), omega col = Vp - J*Xp (line 97), RHS -dv/dt (line 99)
A_doc = J.row_join(Vp - J*Xp)
b_doc = -vt

# Are these the SAME linear system A q = b (row-equivalent => same solution)?
# They share the same first two columns (J) and same b, but differ in col3 sign.
# => They are DIFFERENT systems (col3 flipped) => give q with c of opposite sign.
print("A_true col3 == -(A_doc col3)?",
      sp.simplify(A_true[:,2] + A_doc[:,2]) == sp.zeros(2,1),
      "(so doc-literal omega col alone would flip sign of c)")

# The point: the doc's *Rosetta table* pairs (Vp - J*Xp) with the OTHER RHS
# convention historically (Gunther uses +dv/dt). Let's confirm the fully-flipped
# Gunther-style pair reproduces the SAME residual magnitude as A_true.
#   Gunther residual uses M with omega col (Vp - J*Xp)? Actually code uses Jxpvp = -J*Xp + Vp = Vp - J*Xp, RHS +dv/dt.
# Build Gunther single-rigid analog (ignore fake udot cols; just omega+translation):
A_g = J.row_join(Vp - J*Xp)   # note: Gunther translation cols are +J too (code M[:, :,0,1..2]=J)
b_g = vt                       # RHS +dv/dt

# For a least squares A q = b, the minimizer and residual depend on (A,b).
# Claim to test: (A_g, b_g) and (A_true, b_true) yield the SAME residual r (up to sign)
#   only if A_g q = b_g  <=>  A_true q = b_true for the same q.
# A_true q = -vt ;  negate both sides: -A_true q = vt.
#   -A_true = [-J | Vp - J*Xp].  But A_g = [ +J | Vp - J*Xp ].  translation block differs in sign!
print("\n-A_true == A_g ?", sp.simplify((-A_true) - A_g) == sp.zeros(2,3),
      "  <-- if False, Gunther(+dv/dt,+J,Vp-JXp) is NOT the sign-flip of killing system")
print("Note translation block: -A_true[:, :2] =", "-J", " vs A_g[:, :2] = +J  => they differ")

# Resolve: the correct statement is the WHOLE matrix flips sign together with b.
# killing:   A=[J | JXp-Vp], b=-vt
# flip all:  A=[-J| Vp-JXp], b=+vt   (this is the exact same LSQ)
A_flip = (-J).row_join(Vp - J*Xp)
b_flip = vt
print("\nExact sign-flip of killing system: A=[-J | Vp-JXp], b=+vt")
print("  A_flip == -A_true ?", sp.simplify(A_flip + A_true) == sp.zeros(2,3))
print("  b_flip == -b_true ?", sp.simplify(b_flip - vt) == sp.zeros(2,1))

# ── 3D killing analog (doc 3.3): A = [J | skew(v) - J*skew(x)], b=-dv/dt ──
print("\n=== 3D killing: derive omega block sign, RHS -dv/dt ===")
X3 = sp.Matrix(sp.symbols('X Y Z', real=True))
v3 = sp.Matrix([sp.Function('v0')(*X3), sp.Function('v1')(*X3), sp.Function('v2')(*X3)])
vt3 = sp.Matrix(sp.symbols('vt0 vt1 vt2', real=True))
def skew(a):
    return sp.Matrix([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
J3 = v3.jacobian(X3)                       # J3[i,j] = d v_i/d X_j
t3 = sp.Matrix(sp.symbols('t0 t1 t2', real=True))
w3 = sp.Matrix(sp.symbols('w0 w1 w2', real=True))
u3 = t3 + w3.cross(X3)                      # killing field u = t + w x X
gradu3 = u3.jacobian(X3)
r3 = vt3 + J3*u3 - gradu3*v3                # low-order observed td
# coeff of w (omega block, 3x3)
Mw = sp.Matrix([[sp.diff(r3[i], w3[j]) for j in range(3)] for i in range(3)])
Mw = sp.simplify(Mw)
cand = sp.simplify(skew(v3) - J3*skew(X3))
print("omega block == skew(v) - J*skew(x) ?", sp.simplify(Mw - cand) == sp.zeros(3,3))
# coeff of t (translation block) should be J3
Mt = sp.Matrix([[sp.diff(r3[i], t3[j]) for j in range(3)] for i in range(3)])
print("translation block == J ?", sp.simplify(Mt - J3) == sp.zeros(3,3))
const3 = sp.simplify(r3.subs({t3[0]:0,t3[1]:0,t3[2]:0,w3[0]:0,w3[1]:0,w3[2]:0}))
print("const == dv/dt ?", sp.simplify(const3 - vt3) == sp.zeros(3,1),
      " => r3 = [J | skew(v)-J skew(x)] q + dv/dt => A q = b with b=-dv/dt  ✓ matches code")
