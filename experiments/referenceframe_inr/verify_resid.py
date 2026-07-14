import numpy as np
# Finding #5: E = e - r^T S^-1 r  is the closed-form min ||A q - b||^2 ?
# with S=A^T A, r=A^T b, e=b^T b.  min over q of ||Aq-b||^2 = b^Tb - b^TA(A^TA)^-1A^Tb
#   = e - r^T S^-1 r.  Verify numerically incl. the merge additivity (sum of stats).
rng=np.random.default_rng(0)
for trial in range(4):
    m=rng.integers(20,60); A=rng.standard_normal((m,3)); b=rng.standard_normal(m)
    S=A.T@A; r=A.T@b; e=b@b
    q,*_=np.linalg.lstsq(A,b,rcond=None)
    res_direct=np.sum((A@q-b)**2)
    res_closed=e - r@np.linalg.solve(S,r)
    print(f"trial{trial}: direct={res_direct:.6e} closed={res_closed:.6e} match={np.isclose(res_direct,res_closed)}")
# additivity: stacking two blocks == summing their (S,r,e)
A1=rng.standard_normal((30,3)); b1=rng.standard_normal(30)
A2=rng.standard_normal((25,3)); b2=rng.standard_normal(25)
Acat=np.vstack([A1,A2]); bcat=np.concatenate([b1,b2])
Scat=Acat.T@Acat; rcat=Acat.T@bcat; ecat=bcat@bcat
S12=A1.T@A1+A2.T@A2; r12=A1.T@b1+A2.T@b2; e12=b1@b1+b2@b2
print("merge stats additive:", np.allclose(Scat,S12), np.allclose(rcat,r12), np.isclose(ecat,e12))
# so E_AB uses summed stats -> correct closed-form region residual. Delta=E_AB-E_A-E_B>=0.
