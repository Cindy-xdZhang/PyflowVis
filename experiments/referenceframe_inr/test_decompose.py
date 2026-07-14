"""Correctness tests for FLowUtils/referenceFrameDecompose.py"""
import sys, numpy as np
sys.path.insert(0, r"C:\Users\xingdi\sources\PyflowVis")
from FLowUtils.VectorField2d import UnsteadyVectorField2D
from FLowUtils.AnalyticalFlowCreator import rotation_four_center
from FLowUtils.referenceFrameDecompose import decompose_reference_frame_2d


def make_two_flow(X=64, Y=64, T=24, u1=0.4, u2=0.4):
    """Left half: vortex translating +x (observer (u1,0)); right half: translating +y (observer (0,u2)).
    A single global rigid observer cannot explain both => should split into 2 regions."""
    f = UnsteadyVectorField2D(X, Y, T, [-2, -2, 0.0], [2, 2, 2.0])
    xs = np.linspace(-2, 2, X); ys = np.linspace(-2, 2, Y)
    Yg, Xg = np.meshgrid(ys, xs, indexing="ij")
    data = np.zeros((T, Y, X, 2), np.float32)
    for it in range(T):
        t = 2.0 * it / (T - 1)
        # left translating-vortex (+x): swirl(x-u1 t, y) + (u1,0)
        xl = Xg - u1 * t
        envL = np.exp(-(xl**2 + Yg**2))
        vLx = -Yg * envL + u1; vLy = xl * envL
        # right translating-vortex (+y): swirl(x, y-u2 t) + (0,u2)
        yr = Yg - u2 * t
        envR = np.exp(-(Xg**2 + yr**2))
        vRx = -yr * envR; vRy = Xg * envR + u2
        left = (Xg < 0)
        data[it, :, :, 0] = np.where(left, vLx, vRx)
        data[it, :, :, 1] = np.where(left, vLy, vRy)
    f.field = data
    return f


def test_rfc():
    print("=== rfc (global rotation) ===")
    f = rotation_four_center((64, 64), 48)
    dec = decompose_reference_frame_2d(f, k=2, verbose=False)
    d = dec.diag
    print(f"  global_res_ratio={d['global_residual_ratio']:.4f}  finest_res_ratio={d['finest_residual_ratio']:.4f}  "
          f"benefit={d['decomposition_benefit']:.3f}")
    print("  ->", d["interpretation"])
    # rfc has a single global rotation observer -> global residual small, decomposition gain ~1
    assert d["global_residual_ratio"] < 0.1, "rfc should be explained by one global observer"
    assert d["decomposition_benefit"] < 0.05, "rfc should NOT benefit from multiple regions"
    print("  rfc OK\n")


def test_two_flow():
    print("=== synthetic two-flow (left +x, right +y) ===")
    f = make_two_flow()
    dec = decompose_reference_frame_2d(f, k=2, verbose=False)
    d = dec.diag
    print(f"  global_res_ratio={d['global_residual_ratio']:.4f}  finest_res_ratio={d['finest_residual_ratio']:.4f}  "
          f"benefit={d['decomposition_benefit']:.3f}")
    print("  ->", d["interpretation"])
    assert d["decomposition_benefit"] > 0.2, "two-flow should benefit from multiple regions"
    assert d["finest_residual_ratio"] < 0.3, "finest partition should explain it (not intrinsic unsteadiness)"

    labels = dec.cut(n_regions=2)
    Y, X = labels.shape
    # check the 2-region cut correlates with the true left/right split
    xs = np.linspace(-2, 2, X); left_true = (xs < 0)[None, :].repeat(Y, 0)
    lab_left = labels[:, xs < 0]; lab_right = labels[:, xs >= 0]
    # dominant label on each side
    from collections import Counter
    dl = Counter(lab_left.ravel()).most_common(1)[0][0]
    dr = Counter(lab_right.ravel()).most_common(1)[0][0]
    agree = (np.mean(lab_left == dl) + np.mean(lab_right == dr)) / 2
    print(f"  cut(2): dominant left-label={dl} right-label={dr}, side purity={agree:.2%}, split_ok={dl!=dr}")
    assert dl != dr and agree > 0.8, "cut(2) should recover left/right split"

    # recovered observers
    obs = dec.region_observers(labels)
    for rid, coeff in obs.items():
        m = coeff[2:-2].mean(0)  # (a,b,c)
        print(f"    region {rid}: observer mean (a,b,c)={np.round(m,3)}")
    print("  two-flow OK\n")


def test_curve_and_interfaces():
    print("=== interfaces (residual_curve, observer_field) ===")
    f = make_two_flow(X=48, Y=48, T=16)
    dec = decompose_reference_frame_2d(f, k=2)
    ns, res = dec.residual_curve(max_regions=8)
    print("  residual_curve n:", list(ns[:6]), " res:", np.round(res[:6], 3))
    assert np.all(np.diff(res) <= 1e-6), "residual must be non-increasing as #regions grows"
    labels = dec.cut(n_regions=2)
    uf = dec.observer_field(labels)
    assert uf.field.shape == f.field.shape and np.all(np.isfinite(uf.field))
    print("  observer_field shape", uf.field.shape, "OK")
    # cut by threshold works
    lab_t = dec.cut(cost_threshold=float(res.max()))
    print("  cut(threshold) #regions:", len(np.unique(lab_t)))
    print("  interfaces OK\n")


if __name__ == "__main__":
    test_rfc()
    test_two_flow()
    test_curve_and_interfaces()
    print("ALL DECOMPOSE TESTS PASSED")
