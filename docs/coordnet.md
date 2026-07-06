# CoordNet — coordinate-based INR for time-varying volumes (and its use as a compressor)

Summary of **CoordNet: Data Generation and Visualization Generation for Time-Varying
Volumes via a Coordinate-Based Neural Network** (Jun Han & Chaoli Wang, *IEEE TVCG*
29(12), 2023; code: <https://github.com/stevenhan1991/CoordNet>).

Our reimplementation lives in
[`CoordNetCompression.py`](../CoordNetCompression.py) and targets **one** of CoordNet's
uses — **compressing a time-varying volume with an implicit neural representation (INR)**
— but is structured as a *baseline-comparison harness* (pluggable coordinate transforms,
spatial partitioning, and models) so it is easy to modify later.

---

## 1. Idea

CoordNet is a **single coordinate→value function** `f(x,y,z,t) = v` realized by an
**implicit neural representation (INR)** — a plain MLP that maps a coordinate to the
signal value there. The same architecture solves many tasks by only changing what the
coordinate/value mean (Table 1 in the paper): temporal/spatial super-resolution
`(x,y,z,t)→voxel`, view synthesis `(x,y,θ,φ)→RGB`, ambient-occlusion prediction, etc.

**Compression** (our target, following Lu et al. / *neurcomp*) is the degenerate case:
overfit `f` to **all** voxels of one volume. The trained network weights *are* the
compressed data — you throw the volume away and regenerate any voxel on demand from `f`.
The compression ratio is `original_bytes / weight_bytes`; quality is PSNR of the
reconstruction.

Why an INR (vs a CNN): resolution-independent (query any continuous coordinate), treats
the field as a continuous function, and one architecture works unchanged across tasks.

---

## 2. Building block: SIREN

A **SIREN** layer is a linear layer with a **sinusoidal activation**:

$$
\phi(\mathbf{x}) = \sin\!\big(\omega\,(W\mathbf{x}+\mathbf{b})\big),\qquad \omega = 30 .
$$

Sine activations train stably, have gradients almost everywhere, and fit high-frequency
signals well. Weights use the SIREN init (Sitzmann et al.): first layer
`W ~ U(-1/n, 1/n)`, hidden layers `W ~ U(-√(6/n)/ω, √(6/n)/ω)`. Because `sin ∈ [-1,1]`,
**coordinates and values are normalized to `[-1,1]`**.

## 3. SIREN residual block

To build a deep, well-conditioned network the paper wraps SIREN in a **residual block**:

- main path: two SIREN layers, `SIREN(in→out) → SIREN(out→out)`;
- skip path: identity if `in==out`, else a projection `SIREN(in→out)`;
- **output = (main + skip) / 2**.

The `/2` average is essential: two `sin` outputs sum into `[-2,2]`, outside the sine
range; halving returns to `[-1,1]` and keeps training stable.

## 4. CoordNet architecture (encoder–decoder)

`k`-dim coordinate in, `p`-dim value out. With `m` initial neurons and depth `d`
(paper: `m=64`, `d=10`, `ω=30`):

| stage | block | in → out |
|---|---|---|
| encoder | residual block | `k → m` |
| encoder | residual block | `m → 2m` |
| encoder | residual block | `2m → 4m` |
| encoder | `d ×` residual block | `4m → 4m` |
| decoder | residual block | `4m → p` |

So `3 + d + 1` residual blocks. For a time-varying scalar volume, `k=4` (x,y,z,t) and
`p=1`. The final block's `sin` keeps the output in `[-1,1]` = the normalized value range.

## 5. Objective & training

Pure **MSE** between predicted and ground-truth values:

$$
\mathcal{L} = \frac{1}{N}\sum_i \frac{1}{|C_i|}\sum_{(x,y,\dots)\in C_i}
\big\lVert f(x,y,\dots)-v\big\rVert_2^2 .
$$

(Adversarial/feature losses do not apply — the output is a single voxel, not a patch.)

Paper training settings: Adam (`β1=0.9, β2=0.999`), **batch = 32 000 coordinates**,
lr `1e-5`, L2 weight decay `1e-6`, **300 epochs**, coordinates & values scaled to
`[-1,1]`. Metric: **PSNR** (data level; also LPIPS/CD for the rendering tasks — not needed
for compression).

---

## 6. What our reimplementation does (and how to extend it)

[`CoordNetCompression.py`](../CoordNetCompression.py) fits an INR to a (synthetic or
loaded) time-varying volume and reports **PSNR + compression ratio + #params**. It is
deliberately factored into three pluggable registries so the paper's method is one
interchangeable choice and future variants slot in without touching the training loop:

- **coordinate transforms** (`TRANSFORM_REGISTRY`) — `identity` (paper) plus a Fourier
  positional-encoding stub. *This is where a different coordinate system goes.*
- **spatial partitioning** (`PARTITION_REGISTRY`) — `none` (one global INR, paper) plus a
  block-partition stub (per-block INRs / block-conditioned INR). *This is where a
  different spatial subdivision goes.*
- **models** (`MODEL_REGISTRY`) — `coordnet` (this paper) and a `plain_siren` baseline;
  add a class + register it to compare a new INR against CoordNet.

The runner reports a uniform metrics dict `{psnr, compression_ratio, num_params,
model_bytes, data_bytes, train_time}` per model, so several baselines can be compared in
one run. Data defaults to a self-contained **synthetic time-varying volume** (moving
Gaussian blobs / analytic vortex) so it runs with no external data; real volumes load from
`.npy`/`.npz`/raw `float32`.
