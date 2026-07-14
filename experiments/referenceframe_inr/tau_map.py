"""tau -> N mapping for cut_adaptive, per field. Pure CPU, no INR training.

cut_adaptive(tau) == cut(cost_threshold = tau * raw_td_energy), and cut(cost_threshold)
stops at the FIRST merge whose cost exceeds the threshold. So with L leaves and merge
costs c[0..L-2] (in merge order), N = L - first_index_with_cost>thr.

=> N is reachable by SOME tau iff c[L-N] > max(c[:L-N]) (i.e. that merge is a running max).
   The admissible tau interval for a reachable N is [max(c[:L-N]), c[L-N]) / raw_energy.
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir)))
import numpy as np
from FLowUtils.referenceFrameDecompose import decompose_reference_frame_2d
from FLowUtils.AnalyticalFlowCreator import rotation_four_center
from FLowUtils.flowDatasetUtils.NetCDF_AmiraLoader import NetCDFLoader

D = r"C:\Users\xingdi\OneDrive - KAUST\WorkingInProcess\FLowVisAssets\flowData2D"


def load(name):
    if name == "rfc":
        return rotation_four_center((64, 64), 64)
    if name == "cylinder2d":
        f = NetCDFLoader.load_vector_field2d(rf"{D}\cylinder2d.nc", 800, 960)
        f.resample2UnsteadyField((128, 320, 80)); return f
    f = NetCDFLoader.load_vector_field2d(rf"{D}\boussinesq.nc", 800, 960)
    f.resample2UnsteadyField((128, 75, 225)); return f


def analyze(name, max_n=12):
    dec = decompose_reference_frame_2d(load(name), k=2)
    costs = dec.linkage[:, 2]                       # merge order
    L = dec.n_leaves
    raw = float(dec.diag["raw_td_energy"])
    print(f"\n===== {name}  leaves={L}  raw_td_energy={raw:.4e}  "
          f"benefit={dec.diag['decomposition_benefit']:.3f} =====", flush=True)
    runmax = np.maximum.accumulate(np.concatenate([[0.0], costs]))[:-1]   # max(c[:i]) for each i
    rows = {}
    for N in range(1, max_n + 1):
        i = L - N                                   # index of the merge that would take N -> N-1
        if i >= len(costs):
            continue
        lo, hi = float(runmax[i] / raw), float(costs[i] / raw)   # tau in [lo,hi) yields exactly N regions
        reachable = bool(costs[i] > runmax[i])
        rows[N] = dict(tau_lo=lo, tau_hi=hi, reachable=reachable)
        mark = "" if reachable else "  <-- UNREACHABLE by any tau (cost not a running max)"
        print(f"  N={N:2d}: tau in [{lo:.5f}, {hi:.5f})   width={max(hi-lo,0):.5f}{mark}", flush=True)
    # sanity: verify by actually calling cut_adaptive on a few taus
    print("  [verify] tau -> N via cut_adaptive():", flush=True)
    for tau in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]:
        _, n = dec.cut_adaptive(tau)
        print(f"    tau={tau:<6} -> N={n}", flush=True)
    return rows


if __name__ == "__main__":
    out = {}
    for nm in ["rfc", "cylinder2d", "boussinesq"]:
        out[nm] = {str(k): v for k, v in analyze(nm).items()}
    with open("tau_map.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote tau_map.json", flush=True)
