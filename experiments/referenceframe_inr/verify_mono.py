import sys, numpy as np
sys.path.insert(0, r"C:\Users\xingdi\sources\PyflowVis")
from FLowUtils.VectorField2d import UnsteadyVectorField2D
from FLowUtils.referenceFrameDecompose import decompose_reference_frame_2d
# Build a TWO-FLOW field (left half rotates one way, right half another) so the
# dendrogram has non-trivial structure and we can see if costs are monotone.
Nx=Ny=48; T=6; L=2.0
f=UnsteadyVectorField2D(Nx,Ny,T,[-L,-L,0.0],[L,L,2.0])
xs=-L+f.gridInterval[0]*np.arange(Nx); ys=-L+f.gridInterval[1]*np.arange(Ny)
Yg,Xg=np.meshgrid(ys,xs,indexing="ij")
data=np.zeros((T,Ny,Nx,2),np.float64)
for t in range(T):
    th1=0.6*f.timeInterval*t; th2=-0.5*f.timeInterval*t
    for (mask,th,cx) in [ (Xg<0, th1,-1.0),(Xg>=0,th2,1.0) ]:
        c,s=np.cos(th),np.sin(th)
        px=c*(Xg-cx)+s*Yg; py=-s*(Xg-cx)+c*Yg
        sx=-py; sy=px
        vx=c*sx-s*sy + (th/f.timeInterval if f.timeInterval else 0)*(-(Yg))
        vy=s*sx+c*sy + (th/f.timeInterval if f.timeInterval else 0)*((Xg-cx))
        data[t][mask]=np.stack([vx,vy],-1)[mask]
f.field=data.astype(np.float32)
dec=decompose_reference_frame_2d(f,k=2,verbose=False)
costs=dec.linkage[:,2]
print("n_merges=",len(costs))
diffs=np.diff(costs)
print("cost monotone non-decreasing over full linkage?", np.all(diffs>=-1e-12))
print("num descents (cost[i+1]<cost[i]):", int(np.sum(diffs< -1e-9)), "of", len(diffs))
# demonstrate the count-vs-prefix gap for a threshold
thr=np.median(costs)
cnt=int(np.count_nonzero(costs<=thr))
# length of leading prefix that is entirely <= thr:
prefix=0
for cst in costs:
    if cst<=thr: prefix+=1
    else: break
print(f"threshold=median: count(<=thr)={cnt}  leading-prefix(<=thr)={prefix}  "
      f"{'*** DIFFER => cut applies out-of-order merges ***' if cnt!=prefix else 'equal'}")
# show first place where a >thr merge precedes a <=thr merge
over=np.where(costs>thr)[0]
if len(over):
    firstover=over[0]
    later_ok=np.where(costs[firstover:]<=thr)[0]
    print(f"first merge > thr at index {firstover}; "
          f"{'a later merge <= thr EXISTS at +'+str(later_ok[0]) if len(later_ok) else 'no later <=thr merge'} "
          f"=> count() would wrongly re-include it" if len(later_ok) else "")
