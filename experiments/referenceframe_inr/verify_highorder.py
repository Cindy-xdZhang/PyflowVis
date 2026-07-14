import sympy as sp
# killing_optimization_2d HIGH-order residual check (Finding #1a):
# code: L = J*W - gradW.V (with gradW.col2 = (-vy,vx)=Vp), W = [[1,0,-y],[0,1,x]]
#       C = [L | -W]  (2x6),  unknown [q; qdot] = [a,b,c, adot,bdot,cdot]
#       RHS = -vt  =>  min || C [q;qdot] - (-vt) ||^2  i.e. C[q;qdot] = -vt
# Claim: this encodes  D = |dv/dt - du/dt + Jv.u - Ju.v|^2  with
#        u = W q  (so du/dt = W qdot, since W is time-indep),
#        Jv.u - Ju.v = L q   (the low-order part), and -du/dt = -W qdot.
x,y=sp.symbols('x y',real=True)
vx=sp.Function('vx')(x,y); vy=sp.Function('vy')(x,y)
vxt,vyt=sp.symbols('vxt vyt',real=True)
a,b,c,ad,bd,cd=sp.symbols('a b c ad bd cd',real=True)
J=sp.Matrix([[sp.diff(vx,x),sp.diff(vx,y)],[sp.diff(vy,x),sp.diff(vy,y)]])
v=sp.Matrix([vx,vy]); vt=sp.Matrix([vxt,vyt])
W=sp.Matrix([[1,0,-y],[0,1,x]])
q=sp.Matrix([a,b,c]); qdot=sp.Matrix([ad,bd,cd])
u=W*q
du_dt=W*qdot                 # since dW/dt=0
gradu=u.jacobian([x,y])
# full observed td (HIGH order):
r_true = vt - du_dt + J*u - gradu*v
r_true=sp.simplify(r_true)
# code's residual: C[q;qdot] + vt  where C=[L|-W]
gradW_V_col2 = sp.Matrix([-vy,vx])   # code subtracts this into L col2
L = J*W
L = L - sp.Matrix.hstack(sp.zeros(2,2), gradW_V_col2)   # L[:,2]-=Vp
C = sp.Matrix.hstack(L, -W)
qq=sp.Matrix([a,b,c,ad,bd,cd])
r_code = C*qq + vt        # since RHS=-vt => residual C qq -(-vt) = C qq + vt
r_code=sp.simplify(r_code)
print("HIGH-order: r_code == r_true (full D incl -du/dt) ?",
      sp.simplify(r_code-r_true)==sp.zeros(2,1))
# and confirm L (low-order block) matches Jv.u-Ju.v as coeff of q:
low = sp.simplify(J*u - gradu*v)
print("L*q == Jv.u - Ju.v ?", sp.simplify(L*q - low)==sp.zeros(2,1))
print("-W block == coeff of qdot in (-du/dt) ?  -W*qdot == -du_dt ?",
      sp.simplify(-W*qdot + du_dt)==sp.zeros(2,1))
