"""AUDIT: blend coverage + weight normalization for proposed_overlap.
Checks: (a) is every pixel covered by at least one region's gaussian weight (else 0/0 -> the
clip(1e-8) makes accum_v=0 there, i.e. reconstruction=0, silently wrong)? (b) does the core-only
gaussian leave interior-of-region weight but ZERO on a whole small region if gaussian_filter of a
tiny core underflows?"""
import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter

def simulate(labels, margin=4, sigma=4.0):
    Y, X = labels.shape
    accum_w = np.zeros((Y, X))
    per_region_wsum = {}
    zero_core_weight = []
    for rid in np.unique(labels):
        core = labels == rid
        tmask = binary_dilation(core, iterations=margin)
        w = gaussian_filter(core.astype(np.float64), sigma) * tmask
        wp = w[tmask]
        accum_w[tmask] += wp
        per_region_wsum[int(rid)] = float(w.sum())
        # does this region contribute ANY positive weight on its OWN core?
        if w[core].max() <= 0:
            zero_core_weight.append(int(rid))
    uncovered = int((accum_w <= 0).sum())
    tiny = int((accum_w < 1e-8).sum())
    return uncovered, tiny, per_region_wsum, zero_core_weight, accum_w

# Case A: normal contiguous regions (vertical stripes) -> should fully cover
lab = np.zeros((75, 225), int)
for i,(a,b) in enumerate([(0,75),(75,150),(150,225)]):
    lab[:, a:b] = i
u, t, wsum, zc, aw = simulate(lab)
print(f"[A] 3 vertical stripes 75x225: uncovered={u}, <1e-8={t}, zero-core-regions={zc}")
print(f"    accum_w min={aw.min():.4f} max={aw.max():.4f}")

# Case B: a thin 1-px region (pathological) - can gaussian of a 1px-wide core vanish where dilation reaches?
lab2 = np.zeros((75,225), int); lab2[:] = 0; lab2[:, 100] = 1  # region 1 is a single column
u2,t2,wsum2,zc2,aw2 = simulate(lab2)
print(f"[B] region1 = single column: uncovered={u2}, <1e-8={t2}, zero-core-regions={zc2}")

# Case C: a tiny isolated blob far from others (10 regions, one is a 2x2)
lab3 = np.zeros((64,64), int)
# 10 regions via coarse tiling
k=0
for iy in range(0,64,32):
    for ix in range(0,64,16):
        lab3[iy:iy+32, ix:ix+16] = k; k+=1
u3,t3,wsum3,zc3,aw3 = simulate(lab3)
print(f"[C] {k} tiled regions 64x64: uncovered={u3}, <1e-8={t3}, zero-core-regions={zc3}")

# KEY diagnostic: With gaussian_filter*tmask, the pixels at the OUTER edge of the dilation band
# (distance ~margin from core) get gaussian weight ~exp(-(margin/sigma)^2/2). With margin=4,sigma=4
# that's exp(-0.5)=0.61 * (spread) -- but gaussian_filter NORMALIZES, so peak weight in a large
# region interior ~1, edge lower. Question: can two adjacent regions BOTH have ~0 weight in the
# thin seam between their cores? Overlap band guarantees each covers the seam.
print("\n[note] gaussian_filter(core) peaks ~ (# core px within ~sigma) normalized; interior of a")
print("       region gets ~1.0, dilation edge lower but >0. Overlap (margin) ensures neighbors' bands meet.")
