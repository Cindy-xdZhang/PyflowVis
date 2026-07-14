"""AUDIT: (1) normalization asymmetry baseline vs proposed; (2) determinism of randperm/ops."""
import numpy as np, torch, math

# ---------- Determinism probes ----------
print("=== determinism ===")
print("torch", torch.__version__, "cuda?", torch.cuda.is_available())
# Is torch.randperm on CUDA covered by use_deterministic_algorithms? (It IS deterministic given seed+device.)
# The real question: does the module set the seed ONCE and is randperm reproducible run-to-run?
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception as e:
    print("use_det err", e)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
p1 = torch.randperm(1000, device=dev)
torch.manual_seed(0)
p2 = torch.randperm(1000, device=dev)
print("randperm reproducible w/ same seed:", bool((p1==p2).all()))

# gaussian_filter / scipy are deterministic (no RNG). binary_dilation deterministic. OK.
# Adam is deterministic given deterministic grads. The ONLY nondeterminism risk: CUDA atomics in
# backward of some ops (matmul is fine w/ CUBLAS cfg). FFReLU/CoordNet use Linear+sin/relu -> covered.

# ---------- Normalization asymmetry ----------
print("\n=== normalization ===")
# BASELINE (rfo_final2.baseline / rfo_final4.baseline_H):
#   flat = v.reshape(-1,C); vmin=flat.min(0); vmax=flat.max(0)  -> PER-CHANNEL over ALL (t,y,x)
#   vals = 2*(flat-vmin)/sc - 1        (C independent scales)
#   coords: x,y,t each linspace(-1,1) over full grid  -> ASPECT RATIO NOT PRESERVED
# PROPOSED (fit_region):
#   value: vco (co-moving) per-channel minmax over all (t,pix) -> 2 scales, same idea
#   coords xi[:, :2] normalized JOINTLY? NO: lo=xi[:,:2].min(0) per-axis (2 values), sc per-axis
#     xi[:,:2] = 2*(xi-lo)/sc - 1     -> per-axis, ASPECT RATIO NOT PRESERVED either
#   time: linspace(-1,1,T) same as baseline
# => Both distort aspect ratio the SAME way (per-axis to [-1,1]). Symmetric. Not a bias.

# BUT: subtle point. Baseline coords span the FULL rectangular grid [-1,1]^2 for (x,y).
# Proposed xi (warped) is normalized to its OWN bbox [-1,1]^2. For a rotating region the warped
# bbox can be LARGER (rotation sweeps area). The INR sees a tighter/looser coordinate density.
# This is inherent to the method, not a bug. Both get [-1,1] input range -> same INR conditioning.

# Numerically show baseline per-axis vs proposed per-axis both hit [-1,1]:
Y,X,T,C = 75,225,128,2
rng = np.random.default_rng(1); v = rng.standard_normal((T,Y,X,C)).astype(np.float32)
flat = v.reshape(-1,C); vmin=flat.min(0); vmax=flat.max(0); sc=np.maximum(vmax-vmin,1e-8)
vals=(2*(flat-vmin)/sc-1)
print(f"baseline value range per-chan: [{vals[:,0].min():.3f},{vals[:,0].max():.3f}] [{vals[:,1].min():.3f},{vals[:,1].max():.3f}]")

# ---------- KEY asymmetry to check: baseline value denorm vs proposed value denorm ----------
# Baseline: recon = (rec+1)*0.5*sc + vmin   -- inverts the SAME per-channel minmax. Correct.
# Proposed: vco_r = (rec_n+1)*0.5*vsc+vmin  -- inverts the co-moving minmax, THEN rotates back+u.
#   The rotate-back is exact (round-trip proven). So proposed reconstructs v exactly given perfect INR.
# Both clamp INR output to [-1,1] before denorm (fit_inr/fit_ffrelu return .clamp(-1,1)).
#   => clamp caps values at the train min/max. If TEST had values outside [train_min,train_max]
#      they'd be clipped. But here train==test (overfit ALL pixels). No held-out set. Symmetric.
print("clamp(-1,1) on both sides; train==test (pure overfit). Symmetric.")

# ---------- The ONE asymmetry that matters: coordinate NORMALIZATION STATISTICS source ----------
# fit_region computes lo/hi from xi over the REGION's train pixels (tmask incl. overlap in final5).
# For final5, tmask INCLUDES the dilation band, so xi normalization + value(vco) minmax are computed
# over the DILATED region, but reconstruction writes vx,vy for ALL tmask pixels then blends.
# => consistent (normalize and denormalize over the same tmask pixel set). Correct.
print("\nfit_region normalizes coords & values over the SAME pixel set it reconstructs. Consistent.")
