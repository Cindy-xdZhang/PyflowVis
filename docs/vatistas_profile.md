# The Vatistas Vortex Profile

This note summarizes the **Vatistas velocity profile** used by *VortexTransformer:
End-to-End Objective Vortex Detection in 2D Unsteady Flow* (Zhang, Rautek &
Hadwiger, 2025) to synthesize steady flow fields for training-label generation.
It corresponds to **Eq. (2)–(10)** of the paper (Section 3.1.1) and to the
Vatistas experimental vortex model [VKM91], as also used by [KG19] and [BCG20].

The implementation lives in [`FittingVatistasParam.py`](../FittingVatistasParam.py),
which fits these parameters to real flow patches and re-samples them to generate
new synthetic steady fields.

---

## 1. Radial velocity profile (Eq. 3)

The heart of the model is the scalar **Vatistas radial speed** at distance
$r = \lVert x \rVert$ from the vortex core:

$$
v_0(r) \;=\; \frac{r}{2\pi r_c^{2}}
\left[\left(\frac{r}{r_c}\right)^{2n} + 1\right]^{-\tfrac{1}{n}}
\tag{3}
$$

- $r_c$ — **core radius**: the distance at which the tangential speed is maximal.
- $n$ — **shape exponent**: controls how sharply the profile transitions from the
  inner solid-body region to the outer potential-vortex decay. Larger $n$ gives a
  flatter core and a sharper knee (the effect of $n$ is illustrated in [KG19]).

Limiting behaviour (unit circulation $\Gamma = 1$ is baked into the $\tfrac{1}{2\pi r_c^2}$ factor):

- **Inner region** $r \ll r_c$: $\;(r/r_c)^{2n}\to 0$, so $v_0(r)\approx \dfrac{r}{2\pi r_c^2}$ — linear, i.e. **solid-body rotation**.
- **Outer region** $r \gg r_c$: the bracket $\approx (r/r_c)^{2n}$ and $[\cdot]^{-1/n}\approx (r_c/r)^2$, so $v_0(r)\approx \dfrac{1}{2\pi r}$ — the decaying **potential vortex**.

This is exactly the canonical Vatistas $n$-model
$V_\theta(r) = \frac{\Gamma}{2\pi}\,\frac{r}{(r_c^{2n}+r^{2n})^{1/n}}$ after
factoring $r_c^{2}$ out of the denominator, hence the $-\tfrac{1}{n}$ exponent.

> **Numerical note.** In code we never divide by $r$. We evaluate the finite
> factor $g(r) = v_0(r)/r = \frac{1}{2\pi r_c^2}\big[(r/r_c)^{2n}+1\big]^{-1/n}$
> directly, which stays well-defined (and equals $\frac{1}{2\pi r_c^2}$) at $r=0$.

---

## 2. Steady base velocity field (Eq. 2 & 4)

The 2D steady velocity at a point $x=(x,y)$ is the radial profile spun into a
planar field by one of three constant **base-shape matrices** $S_i$:

$$
v(x) \;=\; S_i \cdot x \cdot \frac{v_0(\lVert x\rVert)}{\lVert x\rVert}
\;=\; g(\lVert x\rVert)\;\big(S_i\, x\big)
\tag{2}
$$

Because each $S_i$ is orthogonal, $\lVert S_i x\rVert = \lVert x\rVert = r$, so the
field's speed at radius $r$ equals $v_0(r)$ as intended.

$$
\underbrace{S_1=\begin{pmatrix}1&0\\0&-1\end{pmatrix}}_{\text{saddle}}
\qquad
\underbrace{S_2=\begin{pmatrix}0&1\\-1&0\end{pmatrix}}_{\text{center (cw)}}
\qquad
\underbrace{S_3=\begin{pmatrix}0&-1\\1&0\end{pmatrix}}_{\text{center (ccw)}}
\tag{4}
$$

- $S_1$ — **saddle** (real eigenvalues, strain/hyperbolic).
- $S_2$ — **clockwise center** (vortex).
- $S_3$ — **counter-clockwise center** ($S_3 = -S_2$).

The saddle and the two centers are *genuinely distinct*: the later deformation is
a similarity transform $D\,S_i\,D^{-1}$, which preserves eigenvalue type, so no
continuous deformation turns a saddle into a vortex. $S_i$ is therefore a
**discrete choice** per profile, not a continuous parameter.

### Vortex boundary (Eq. 5)

For the two center shapes ($S_2, S_3$), the vortex region is the disk of maximal
tangential velocity, described by the **signed distance**

$$
d(x) = r_c - \lVert x\rVert
\tag{5}
$$

so $d(x) > 0$ inside the vortex core disk. This signed distance is what becomes
the ground-truth vortex **segmentation label**.

---

## 3. Deformation: rotation, anisotropic scale, translation (Eq. 6–9)

An isolated circular Vatistas vortex is made into a general elliptical, rotated,
translated vortex by a **deformation matrix** $D$ (composition of a rotation
$\theta$ and a non-uniform scaling $s_x, s_y$) plus a **translation** $T=(t_x,t_y)$:

$$
D(\theta, s_x, s_y) =
\begin{pmatrix}
 s_x\cos\theta & -s_y\sin\theta\\
 s_x\sin\theta & \;\;s_y\cos\theta
\end{pmatrix}
\tag{6}
$$

$$
x' = D\,x + T
\tag{7}
$$

The signed distance and the velocity field are pushed forward accordingly
(evaluate the base field at the pulled-back point $x = D^{-1}(x'-T)$):

$$
d'(x') = d\!\big(D^{-1}(x'-T)\big)
\tag{8}
$$

$$
v'(x') = D\cdot v(x) = D\cdot v\!\big(D^{-1}(x'-T)\big)
\tag{9}
$$

Note $\det D = s_x s_y$, so $D^{-1} = \frac{1}{s_x s_y}\begin{pmatrix} s_y\cos\theta & s_y\sin\theta\\ -s_x\sin\theta & s_x\cos\theta\end{pmatrix}$.
The same $s_x, s_y$ that stretch the vortex spatially also scale the output
velocity magnitude (they appear in both the pull-back $D^{-1}$ and the push-forward
$D$). This recipe follows [BCG20].

> **Objectivity insight (from the paper).** The Vatistas profile is *nonlinear*
> in the radius, so a rotating (or otherwise moving) observer **cannot** cancel
> out the vortex — which is precisely why it is a good generator of ground-truth
> objective vortices.

### The 7 parameters per profile

Each single steady Vatistas profile is fully described by

$$
\boxed{\;\theta,\; s_x,\; s_y,\; r_c,\; n,\; t_x,\; t_y\;}
$$

(plus the discrete base shape $S_i$). Here $\theta, s_x, s_y$ come from $D$;
$r_c, n$ come from the radial profile; and $t_x, t_y$ come from $T$.

---

## 4. Mixture of up to two vortices (Eq. 10)

The paper extends [BCG20] to a **linear superposition of up to two** deformed
Vatistas profiles ($m \in \{1,2\}$):

$$
v(x,y) = \sum_{p=1}^{m} v_p(x,y),
\qquad m \in \{1,2\}
\tag{10}
$$

Each $v_p$ is a fully deformed profile (Eq. 9). When $m=2$ the two cores are kept
**sufficiently spaced apart** — the distance between the two vortex cores exceeds
the sum of their radii — so the vortices do not significantly perturb each other
and each keeps a clean segmentation label. A two-profile patch therefore has

$$
2 \times 7 = \mathbf{14}\ \text{continuous parameters}\quad(+\ 2\ \text{discrete shape choices}).
$$

---

## 5. How the parameters are obtained (Eq. 10, Section 3.1.2)

To keep the synthetic parameter space *physically plausible*, the paper fits the
14 parameters to **real** flow data (following [KG19]):

1. Split real unsteady flows into spatial **32×32 patches**, slid over the field
   like a convolution kernel. Sources: 2D Unsteady Cylinder Flow / von Kármán
   vortex street [Pop04] (time steps 600–1000), Boussinesq flow (time steps
   500–800), and RFC64 [GST16] (all time slices).
2. For every patch, fit the mixture model (Eq. 10) by **200 iterations of
   simulated annealing followed by 200 iterations of gradient descent**, using
   an MSE distance to the patch velocities.
3. Record the fitted $\{\theta, s_x, s_y, r_c, n, t_x, t_y\}$ statistics as
   **Gaussian distributions** (mean/variance per parameter). See Appendix B /
   Table 6 of the paper for the resulting histograms.

The fitted distribution is then re-sampled to synthesize brand-new steady fields
on the unit domain $\tilde X\times\tilde Y = [-1,1]^2$ (the paper draws 1400
random fields this way, on top of 1500 fitted patches and 100 hand-added Killing
fields). Each steady field is later observed by many moving observers to produce
the unsteady training set with consistent, objective vortex labels.

---

## 6. From steady to unsteady: the Killing / observer transformation

The fitted profiles only give **steady** fields. The paper turns each steady field
into an **unsteady** one by *observing it from a moving, rotating reference frame*
— a time-dependent **Killing** (rigid-motion) observer. Because the Vatistas
profile is nonlinear in the radius (§3), no such observer can cancel the vortex,
so the transported label stays an **objective** ground truth.

The generator [`VatistasFlowDatasetGenerator.py`](../VatistasFlowDatasetGenerator.py)
implements this as a faithful port of the C++ reference
[`CppProjects/src/transformation.cpp::killingABCtransformation`](../CppProjects/src/transformation.cpp).

### Killing observer field

A Killing observer is described by three time functions $a(t), b(t), c(t)$
(translation velocity $(a,b)$ and spin rate $c$ about a center $o$):

$$
u(x,t) \;=\; \begin{pmatrix}a(t)\\ b(t)\end{pmatrix}
\;+\; c(t)\begin{pmatrix}-(y-o_y)\\ \;\;x-o_x\end{pmatrix}
$$

### Frame transformation $x^{*}=Q(t)\,x+c(t)$

1. **Observer world-line.** RK4-integrate the observer seed $p_0$ through $u$ to get
   $p_i$ at each time $t_i = t_{\min}+i\,\Delta t$.
2. **Rotation.** Accumulate the observer angle $\theta_i=\sum_{k\le i}\Delta t\,c(t_k)$
   and set the observer rotation $R_i=\mathrm{Rot}(\theta_i)$. The **frame** rotation is
   $Q(t_i)=R_i^{\top}$.
3. **Translation.** $c(t_i)=O_s-Q(t_i)\,p_i$ with $O_s=p_0$, so the observer's own
   position maps back to its start.

### Pushed-forward velocity (and label)

For every output point $x^{*}$, pull it back into the steady frame,
$F^{-1}(x^{*})=Q(t)^{\top}(x^{*}-c(t))=R_i\,(x^{*}-c(t))$, sample the steady field
$v$ there, and push forward:

$$
w^{*}(x^{*},t) \;=\; Q(t)\,v\!\big(F^{-1}(x^{*})\big)
\;+\; \dot Q(t)\,F^{-1}(x^{*}) \;+\; \begin{pmatrix}-a(t)\\ -b(t)\end{pmatrix},
\qquad
\dot Q(t)=Q(t)\begin{pmatrix}0 & c\\ -c & 0\end{pmatrix}.
$$

The vortex **label** is the same material scalar carried along the transformation
(Eq. 8): $d^{*}(x^{*},t)=d\big(F^{-1}(x^{*})\big)$, i.e. the signed distance
$\max_p\,(r_{c,p}-\lVert D_p^{-1}(F^{-1}(x^{*})-T_p)\rVert)$ over the active center
profiles. A stationary observer $(a=b=c=0)$ reproduces the steady field on every
frame (verified numerically: max error $0$).

### What the generator produces (Section 3.1 pipeline, paper-exact totals)

[`VatistasFlowDatasetGenerator.py`](../VatistasFlowDatasetGenerator.py) reuses the
best-tuned fit from [`FittingVatistasParam.py`](../FittingVatistasParam.py) (loads the
cached distribution + fitted patch params, mean velocity MSE $0.0094 \approx 34.7$ dB)
and reproduces the paper's dataset at its **exact reported size** (dissertation Ch. 9,
p. 117):

1. a pool of **3000 steady fields** =
   **1500** fitted real-flow patch fits + **1400** random draws from the fitted
   distribution + **100** manually-added **Killing** fields (rigid-motion *hard
   negatives*: observer-producible, hence given an **empty** label);
2. each steady field is observed by **20** randomly-sampled rigid-body observers, each a
   set of **6 scalars** $(a,b,c,\dot a,\dot b,\dot c)$ giving $a(t)=a_0+\dot a\,t$ etc.,
   yielding **$3000\times20 = 60{,}000$ unsteady fields**. The frame transform is
   spatial-identity at $t_0$, so all 20 variants of a steady field **share one**
   objective segmentation label (the $t_0$ label).

Materializing all 60 000 velocity grids is $\sim$39 GB, so by default the script writes
the full 3000-field steady set plus a verification *subset* of unsteady variants (sharded
`.npz`); `--full` streams all 60 000. The reported target totals always equal the paper.

---

## References

- **[VKM91]** Vatistas, Kozel & Mih. *A simpler model for concentrated vortices.* Experiments in Fluids, 1991.
- **[KG19]** Kim & Günther. *Robust reference frame extraction from unsteady 2D vector fields with convolutional neural networks.* 2019.
- **[BCG20]** Baeza Rojo & Günther. *Vector field topology of time-dependent flows / vortex boundaries.* 2020.
- **[Pop04]** Popinet. *Free computational fluid dynamics (Gerris).* 2004.
- **[GST16]** Günther, Schulze & Theisel. *Rotation-invariant vortex data (RFC64).* 2016.
