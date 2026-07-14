"""AUDIT: (1) does cut(N) label EVERY pixel (disjoint-region recon in rfo_decompose_inr.run_field
leaves 0 in uncovered pixels, biasing PSNR)? (2) does region_observers key set == labels set,
so every region gets fit? (3) eval_every asymmetry fit_inr(//50) vs fit_ffrelu(//60)."""
import sys, numpy as np
sys.path.insert(0, r"C:\Users\xingdi\sources\PyflowVis")
from FLowUtils.referenceFrameDecompose import decompose_reference_frame_2d
from FLowUtils.AnalyticalFlowCreator import rotation_four_center

vf = rotation_four_center((64,64), 64)
dec = decompose_reference_frame_2d(vf, k=2)
for N in [1,3,5,10]:
    labels = dec.cut(n_regions=N)
    obs = dec.region_observers(labels)
    # labels is (Y,X). Every pixel labeled? (labels has no -1/NaN)
    uniq = np.unique(labels)
    # region_observers keys come from _leaf_labels (leaf-level). Do they cover all pixel labels?
    # In run_field: for rid in obs: mask = labels==rid ; fit. Union of masks must == all pixels.
    covered = np.zeros(labels.shape, bool)
    for rid in obs.keys():
        covered |= (labels == rid)
    print(f"cut={N:2d}: n_pixel_labels={len(uniq)} n_obs_keys={len(obs)} "
          f"all_pixels_covered={covered.all()} uncovered_px={int((~covered).sum())}")
    # Are obs keys exactly the pixel-label set?
    if set(int(x) for x in uniq) != set(int(k) for k in obs.keys()):
        print(f"   *** MISMATCH: pixel labels {set(int(x) for x in uniq)} vs obs keys {set(obs.keys())}")

# eval_every asymmetry
for ep in [600, 800, 2500, 4000]:
    print(f"epochs={ep}: fit_inr eval_every={max(1,ep//50)}  fit_ffrelu ev={max(1,ep//60)}")
