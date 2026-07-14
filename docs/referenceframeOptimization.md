# Reference-Frame Optimization (Killing / Non-Killing)

本文总结 `optimal-connection` C++ 工程里的**参考系 (reference-frame / observer) 优化**算法，
并对应到 PyflowVis 的 Python 复现 `FLowUtils/ReferenceFrameOptimization.py`。

目标：给定非定常流场 `v(x,t)`，寻找**观察者速度场 `u(x,t)`**，使在该运动参考系里观测到的场
`v − u` 尽量小。⚠️ **注意**：`v − u` 只是**每个时刻的瞬时相对速度场**，**不是**真正的 observed field——
后者需积分 observer 的刚体运动、对 `v − u` 做 **pushforward**（见 [第 8 节](#8-相对速度-vu-vs-真正的-observed-fieldpushforward)）。
相对速度与 observed field 都可用于抽取客观涡旋 (IVD、objective vorticity 等)。

---

## 0. 统一视角：所有优化都是最小化 observed time derivative 的最小二乘

**核心被最小化量 = observed time derivative**（在 `u` 描述的参考系里看到的场的时间导数）：

```
L_u(v − u) = ∂v/∂t − ∂u/∂t + ∇v·u − ∇u·v            （observed time derivative）
```

把观察者用一组基 `eᵢ` 展开 `u(x,t) = Σ qⁱ(t)·eᵢ(x)`，则 `∂u/∂t = Σ (q̇ⁱ·eᵢ + qⁱ·∂eᵢ/∂t)`，
于是 observed time derivative 对未知量 `(q, q̇)` 是**线性**的。逐点取平方模、在域上积分，得到一个
**二次型 → 标准最小二乘**：

```
L(q,q̇) = ½ ‖ A(t)·(q; q̇) − b(t) ‖² ,   b = −∂v/∂t
        →  正规方程   AᵀA·(q; q̇) = Aᵀb            （对 (q,q̇) 求导置零）
```

据此区分两类方法：

| | **Killing mode**（本组工作，`killing_2d`/`killing_3d`） | **Non-killing mode**（Günther 2017 奠基工作，`gunther17_*`） |
|---|---|---|
| 观察者 `u` | 约束为**精确 Killing 场**（刚体运动），`eᵢ` 取 Killing 基（2D `k=3`、3D `k=6`） | **不**约束为单一 Killing 场：逐点局部最优参考系（整体非刚体） |
| 正则化 | **不需要**（`λ_K=λ_S=0`，Killing energy 自动为 0） | 需额外正则项：**Killing energy** `‖∇u+∇uᵀ‖²` 与 / 或 **proximal** `‖u−s‖²`（Günther 通过逐点邻域隐式正则） |
| 求解 | 每时间片全局解一个小最小二乘 | 每点在邻域内解一个最小二乘（SAT 加速） |

> **出处**：*Variational Computation of Optimal Reference Frames for Flow Visualization*，Eq (3)(9)(10)(11)。
> 该论文把 2D 的 `ComputePerTimestepLSQ` 当作对比 **baseline**；论文后续的变分 / Euler-Lagrange / Hamilton
> 求解**不在本包范围内**——本包只实现"逐时间片 / 逐点最小二乘"本身（即 `AᵀA·x = Aᵀb`）。

论文里的三个 Lagrangian 项（`L = ½∫_U (L^D + λ_K·L^K + λ_S·L^S) dx`，Eq 9-10）：

```
L^D = ‖ qⁱ(∇v·eᵢ − ∇eᵢ·v − ∂eᵢ/∂t) − q̇ⁱ·eᵢ − b ‖²   （observed time derivative，b=−∂v/∂t）  (10a)
L^K = ‖ qⁱ(∇eᵢ + (∇eᵢ)ᵀ) ‖²_F                        （Killing energy，仅 non-killing 需要）    (10b)
L^S = ‖ s − qⁱ·eᵢ ‖²                                  （proximal / 种子场 s）                    (10c)
```

Killing mode 只用 `L^D`（`λ_K=λ_S=0`），且 Killing 基 `∂eᵢ/∂t=0`。本包实现的就是这个 `L^D` 最小二乘。

---

## 0.1 公共记号

- **Jacobian** `J = ∇v`，`J[i,j] = ∂v_i/∂x_j`（`J.col(0)=∂v/∂x`, `J.col(1)=∂v/∂y`, …）。
- **时间偏导** `v_t = ∂v/∂t`。
- **Killing 基**（`∇eᵢ + (∇eᵢ)ᵀ = 0`，故 `L^K≡0`）：
  - 2D，`k=3`：`e₁=(1,0)`, `e₂=(0,1)`, `e₃=(−y,x)`  →  `u = (a, b) + c·(−y, x)`，`q=(a,b,c)`。
  - 3D，`k=6`：3 个平移基 + 3 个旋转基  →  `u = t + w×x`，`q=(t, w)`。
- 3D 反对称矩阵 `skew(a)·b = a×b`：`skew(a)=[[0,−a_z,a_y],[a_z,0,−a_x],[−a_y,a_x,0]]`。
- 各方法只是把上面的 observed time derivative 写成不同的 `A`、`b`；求解一律是正规方程 `AᵀA·x = Aᵀb`。

C++ 四个入口 → Python：

| Python | C++ | 文件 |
|---|---|---|
| `killing_optimization_2d`   | `ComputePerTimestepLSQ`             | `Vis26variationalRFT/ObjectAdapterVariationalRFT.cpp` |
| `killing_optimization_3d`   | `KillingLeastSquareOptimization3d`  | `flow3d/ReferenceFrame3d.cpp` |
| `gunther17_optimization_2d` | `runGunther17Optimization`          | `flow2d/ObjectReferenceFrame2d.cpp` |
| `gunther17_optimization_3d` | `GenericLocalOptimization3d`        | `flow3d/ReferenceFrame3d.cpp` |

---

## 1. Killing mode — 2D (`ComputePerTimestepLSQ`, 本组 2026)

最小化**完整** observed time derivative（Eq 3 全部四项，含 `−∂u/∂t`），未知量为 Killing 系数
`q=(a,b,c)` 与其时间导数 `q̇`（`q̇` 作为最小二乘的联合未知量，即 `L^D` 里的 `−q̇ⁱeᵢ` 项）。

### 逐点算子（把 Eq 10a 展开为矩阵）
Killing 基矩阵 `W(x)`（`u = W·q`）与线性算子 `L(x)`：
```
W(x) = [[1, 0, -y],
        [0, 1,  x]]                (2×3)   →  u = W·q = (a − c·y,  b + c·x)

∇WV = [[0, 0, -v_y],
       [0, 0,  v_x]]              (2×3)   （= ∇u·v 中只与 c 相关的部分，即 Σ qⁱ∇eᵢ·v）

L(x) = ∇v·W − ∇WV                 (2×3)   →  L·q = ∇v·u − ∇u·v      （即 Σ qⁱ(∇v·eᵢ − ∇eᵢ·v)）
```
拼装 `C = [ L | −W ]`（2×6，未知量 `[q; q̇]`），逐点残差正是 observed time derivative：
```
r(x) = C·[q; q̇] − (−v_t) = (∇v·u − ∇u·v) − u̇ + v_t = L_u(v − u)
```

### 逐时间片正规方程
对内部格点累加（`∂eᵢ/∂t=0`，`dA=dx·dy` 权重对解无影响）：
```
ATA = Σ_x  CᵀC = [[ Σ LᵀL ,  −Σ LᵀW ],   (6×6)      （= AᵀA，Eq 14 的 Hessian）
                  [ −Σ WᵀL ,   Σ WᵀW ]]
Atb = Σ_x  Cᵀ·(−v_t) = [ −Σ Lᵀv_t ;  Σ Wᵀv_t ]     (6×1)      （b = −∂v/∂t）
```
`G := Σ WᵀW`（3×3）与时间无关（`W` 只依赖位置），可只算一次。解 6×6：
```
[q*; q̇*] = ATA⁻¹ · Atb        （Eigen 用 LDLT）
```
取 `q* = (a,b,c)` 构造观察者 `u(x) = (a − c·y, b + c·x)`，输出 `u` 与 `v − u`。

> `ComputePerTimestepLSQ` 每个时间片**独立**解上面这一个最小二乘，彼此不耦合。重建 `u` 只需 `q`；
> `q̇` 是使 observed time derivative 平方最小的联合未知量（观察者瞬时时间变化率），不进入 `u` 的重建。

---

## 2. Killing mode — 3D 全局 (`KillingLeastSquareOptimization3d`, 本组 2024)

同样最小化 observed time derivative，但 **2024 的写法省略了 `−∂u/∂t` 项**（不引入 `q̇`），只解刚体场
`u = t + w×x` 的 6 个系数——这就是 2D(2026) 与 3D(2024) "写法小有不同"之处。

### 推导（令 observed time derivative 的空间部分为 0）
去掉 `∂u/∂t` 后，令 `∂v/∂t + ∇v·u − ∇u·v = 0`。对 `u=t+w×x`（`∇u=skew(w)`），代入
`∇v·u = J·t − J·skew(x)·w`、`∇u·v = −skew(v)·w`，整理：
```
J·t + (skew(v) − J·skew(x))·w = −∂v/∂t
```

### 逐点系统 + 全局累加
```
A(x) = [ J | skew(v) − J·skew(x) ]   (3×6),  未知量 q = [t; w]
b(x) = −∂v/∂t                         (3×1)

MTM = Σ_{active x} AᵀA   (6×6)        MTb = Σ_{active x} Aᵀb   (6×1)      （Kahan 求和 + 对称化）
q = [t; w] = MTM⁻¹ MTb               （fullPivHouseholderQr + 2 步迭代精化）
```
观察者 `u(x) = t + w×x`（逐点重建整场），`v̂ = v − u`。`mGridFilter` 可选：只在活动点上累加。

---

## 3. Non-killing mode — 2D (`runGunther17Optimization`, Objectivity)

Günther et al. 2017 “Generic Objective Vortices”（奠基工作，C++ 开源、最直接）。**逐点**在邻域内最小二乘，
含刚体速度 + 其时间导数（加速度）自由度。等价于最小化 observed time derivative，但正则化是
**逐点邻域聚合隐式**给出的（而非显式 Killing energy）——每点得到自己的局部最优参考系，整体非刚体。

### 逐点 M（Objectivity，systemSize=6）
`Xp = (−y, x)`，`Vp = (−v_y, v_x)`，`Jxpvp = −J·Xp + Vp`：
```
M(0,·) = [ Jxpvp.x , J(0,0) , J(0,1) , 1 , 0 , Xp.x ]      (2×6)
M(1,·) = [ Jxpvp.y , J(1,0) , J(1,1) , 0 , 1 , Xp.y ]
b = ∂v/∂t
```
列含义：`col0=ω(角速度)`、`col1..2=c(平移速度)`、`col3..4=ċ(线加速度)`、`col5=ω̇(角加速度)`。

### 邻域聚合（Summed-Area-Table）+ 逐点求解
每点先算 `MᵀM`、`Mᵀb`；用 2D SAT（对 `x,y` 前缀和）在 `O(1)` 取 `[iy±U]×[ix±U]` 邻域和：
```
uu = MTM_win⁻¹ · MTb_win          （fullPivHouseholderQr）
v̂(x) = v + (uu1, uu2) − uu0·Xp    →   u = v − v̂ = uu0·Xp − (uu1, uu2)
```
重建只用前 3 个未知量 `ω, c`（加速度项不进入重建）。输出 `_observed_v-u`（`v̂`）与 `_observer_u`。

> 其他 invariance（`Similarity`=8、`Affine`=12、`Displacement`=Taylor）只是换 `M` 的列与重建式，
> 本包只实现最常用的 **Objectivity**。

---

## 4. Non-killing mode — 3D (`GenericLocalOptimization3d`, Objective)

2D 的 3D 版：**逐点**邻域最小二乘，`M` 为 3×12。

### 逐点 M（Objective，systemSize=12）
`F = −J·skew(X) + skew(V)`：
```
M = [ F | J | skew(X) | −I ]      (3×12),  未知量 [ ω(3) | c(3) | ω̇(3) | ċ(3) ]
b = ∂v/∂t
```
（零速度点 `V≈0` 在 C++ 里被跳过。）

### 3D SAT（8 角公式）+ 逐点求解
对 `x,y,z` 三方向前缀和，取 `[±U]³` 立方邻域：
```
uu = MTM_win⁻¹ · MTb_win          （fullPivHouseholderQr）
v̂(x) = V + ω×X + c   （ω=uu[0:3], c=uu[3:6]）   →   u = v − v̂ = −(ω×X + c)
```
输出 `u` 场与 `v̂ = v − u` 场。

---

## 5. Killing vs Non-killing 对照

| 维度 | Killing mode (`killing_2d`/`killing_3d`) | Non-killing (Günther17, `gunther17_*`) |
|------|------------------------------------------|-----------------------------------------|
| 被最小化量 | observed time derivative `L_u(v−u)` | observed time derivative（+ 隐式邻域正则） |
| 观察者约束 | 精确 Killing 基 `Σqⁱeᵢ`（刚体） | 逐点局部最优（整体非刚体） |
| 基 / 自由度 | 2D: 3；3D: 6 | 2D: 6/点；3D: 12/点 |
| `∂u/∂t` 项 (`q̇`) | 2D(2026) 含；3D(2024) 省略 | 含（`ω̇, ċ`） |
| 显式正则化 | 无（`λ_K=λ_S=0`） | Killing energy / proximal，或邻域隐式 |
| 拟合范围 | 每时间片全局一组参数 | 每点一个邻域 |
| `b` (RHS) | `−∂v/∂t` | `+∂v/∂t`（移项符号约定不同，本质同一 observed time derivative=0） |
| 典型用途 | 单一最优参考系 / worldline | 客观涡旋场 (IVD 等) |

---

## 6. Python 复现：`FLowUtils/ReferenceFrameOptimization.py`

数据结构对接 `VectorField2d.py`（`field` 形状 `(T,Y,X,2)`）与 `VectorField3d.py`（`(T,Z,Y,X,3)`）。
`J=∇v` 与 `∂v/∂t` 用中心差分（`numpy.gradient`，按物理 `dx,dy,dz,dt`）。

公开函数：

```python
# --- Killing mode（最小化 observed time derivative，Killing 基，无正则）---
killing_optimization_2d(field, boundary_skip=2, mask=None) -> ReferenceFrameResult   # 含 q̇
killing_optimization_3d(field, mask=None)                  -> ReferenceFrameResult    # 省略 ∂u/∂t

# --- Non-killing mode（Günther 2017，逐点邻域最小二乘）---
gunther17_optimization_2d(field, neighborhood=3, use_sat=True, mask=None) -> ReferenceFrameResult
gunther17_optimization_3d(field, neighborhood=3, use_sat=True, mask=None) -> ReferenceFrameResult
```

`ReferenceFrameResult` 携带 `u_field` / `v_minus_u_field`（同类型 Unsteady*VectorField）与
`params`（Killing mode 逐时间片参数：2D `(a,b,c)`、3D `(t,w)`；Günther mode 为 `None`）。

实现要点：均为正规方程 `AᵀA·x = Aᵀb` 的最小二乘；Killing 2D 复用常量 `G`、跳 2 层边界；
Günther SAT 邻域用四/八角公式；逐点解用带极小 Tikhonov 正则的批量 `solve`（等价 C++ 满秩 QR）。

### GUI 调用（渲染引擎内）
`GuiObjcts/ActiveFieldObject.py` 注册了两个 action，`python main.py` 启动渲染引擎后即可在 ActiveField 面板点击：
`RFO killing (observer u + v-u)` 与 `RFO gunther17 (observer u + v-u)`——对当前 active field 跑优化，
把 observer `u` 与观测场 `v-u` 用 `insertField` 插回场列表可视化（参数 `RFO_neighborhood` 调 Günther 邻域，
按 `getDim()` 自动走 2D/3D）。

---

## 7. 离线可视化对比 (`FLowUtils/ReferenceFrameViz.py`)

不启动 GUI 渲染引擎的快速对比图（matplotlib streamplot），适合远程 / 无桌面时验证结果。

一键（跑 killing + gunther17 并出图）：
```python
from FLowUtils.AnalyticalFlowCreator import rotation_four_center, constant_rotation
from FLowUtils.ReferenceFrameViz import quick_compare_2d

rfc = rotation_four_center((64, 64), 64)
cr  = constant_rotation((64, 64), 64)              # 已知真值 observer（可选）
fig, results = quick_compare_2d(
    rfc, reference=("constant_rotation (ground truth)", cr), save_path="rfc_verify.png")
# 打印: killing params c≈-1; observer temporal-variation killing=0.0000 (恒定旋转)
```

通用（自定义要对比的 observer 场）：
```python
from FLowUtils.ReferenceFrameViz import plot_observer_recovery
from FLowUtils.ReferenceFrameOptimization import killing_optimization_2d, gunther17_optimization_2d
rk = killing_optimization_2d(v); rg = gunther17_optimization_2d(v, neighborhood=4)
plot_observer_recovery(v, {"killing": rk.u_field, "gunther17": rg.u_field},
                       reference=("truth", cr), timesteps=[0, 21, 42], save_path="cmp.png")
```

图布局：行 = 输入 `v` / (reference observer) / 各方法 observer `u`；列 = 若干时间片；流线按速度着色。

**判读**：若真值 observer 是恒定运动（如恒定旋转），正确方法的 observer 行应**各时间列一致**、
且与 reference 行一致（`observer_temporal_variation(...) ≈ 0`）。

> **重要**：`v−u` 只是**瞬时相对速度**（不是真正的 observed field，见第 8 节），在固定欧拉网格上仍随时间
> 移动/旋转（RFC 的 `v−u` 是跟着参考系转的四涡）——正确物理；真正定常的是 pushforward 后的 observed field。
> 可视化验证看 **observer `u`**（应等于真值恒定运动），而非期待 `v−u` 在欧拉网格上静止。3D 场自动取中间 z 切片作图。

---

## 8. 相对速度 `v−u` vs 真正的 observed field（pushforward）

> **易错概念（核心）**：`killing_optimization_*` / `gunther17_*` 输出的 `v − u` **只是每个时刻的瞬时相对速度场**
> （从 lab 系速度里减掉 observer 速度 `u`），**它不是**真正的 observed field。

真正的 observed field 需要**积分 observer 的刚体运动**、再对 `v − u` 做 **pushforward**：

1. 把 observer 场 `u` 当速度**积分**，得到时变刚体变换 `Φ_t`（observed 系 ↔ lab 系）：旋转 `R_t=R(θ_t)`（`θ_t=∫_{t₀}^{t} c_s ds`）+ 平移 `T_t`。
2. observed field 定义在 observed（随动）系的规则网格 `ξ` 上：
```
observed(ξ, t) = R_tᵀ · (v − u)( Φ_t(ξ), t ) ,     Φ_t(ξ) = R_t·ξ + T_t
```
即：observed 网格点 `ξ` → 经 `Φ_t` 映到 lab 位置 `x_lab` 采样相对速度 `(v−u)(x_lab,t)` → 再用 `R_tᵀ` 把矢量**旋转回**随动系。

**为什么 pushforward 才对**：`v−u` 是 lab 系里的瞬时相对速度——即便场"在随动系里 steady"，`v−u` 采样在固定 lab 网格上仍随时间移动/旋转（RFC 的 `v−u` 是跟着参考系转的四涡）。只有把它 pushforward 到随动系（跟着相机刚体运动走 + 旋转矢量），才得到真正定常的 observed field。2D 数学相同：`R_t` 为 2×2 旋转、`θ_t=∫c_t`、`T_t` 由 observer worldline 积分得到。

**C++ 参考**：`WorldlineUtility3D::RFT_Observed_VectorField`（`flow3d/ReferenceFrame3d.cpp`）——`computeReferenceFrameTransformation(u)` 积分 `rft`；对 observed 网格点 `ξ`：`x_lab=Φ_t(ξ)`，采样 `v`、`u`，`R_tᵀ·(v−u)` 存回。2D observer 积分已在 `FLowUtils/KillingObserver2D.py`（`compute_reference_frame_transformation_from_field` / `integrate_reference_frame_rotation` / `Worldline2D`）。

### INR 压缩实验印证（rfc2d / cylinder2d）
把"先算 killing observer、再让 INR 拟合 observed 场（+存每帧 3 个 killing 系数）"与"直接 INR 拟合原场 `v`"对比（同 SIREN/CoordNet 配置，PSNR 在原 `v` 空间；killing 系数极小、几乎不增存储）：

| 场 | baseline `INR(v)` | 拟合**相对速度** `INR(v−u)` | 拟合**真 observed field**（pushforward 到随动系） |
|---|---|---|---|
| **rfc2d** (随动系 intrinsically steady) | 39.18 dB | 36.93 dB (**−2.25**) | **44.67 dB (+5.49)** |
| **cylinder2d** (涡街周期脱落) | 44.56 dB | 46.07 dB (+1.51) | 45.52 dB (+0.96) |

结论：直接拟合**相对速度 `v−u`** 收益不稳定（依赖 `u` 去掉的运动是否是场的大幅值分量：cylinder 去掉大来流 DC 有益，rfc 去掉的旋转太小反而更差）；拟合**真正的 observed field（pushforward）** 在"随动系接近 steady"的 RFC 上大赢（场变得近乎与 `t` 无关，INR 极易拟合）。cylinder 涡街在随动系仍周期脱落、非 steady，故 pushforward 收益有限（且随动系坐标平移累积会扩大坐标范围、稀释分辨率）。

---

## 9. 分片最优参考系 / 区域划分（`FLowUtils/referenceFrameDecompose.py`）

**动机**：单一全局 Killing observer 只对"整场同一刚体运动"的流场有效。很多场（如 cylinder：近尾流慢回流区 vs 下游随来流平移的涡街）**内部有多个流向**，需要**分片**——每个区域一个自己的（时变）Killing observer。

**方法**（自底向上区域合并 → merge tree）：
- 每像素每帧构造 3-DOF Killing 最小二乘 `A·q=b`（`q=(a,b,c)`, `b=−∂v/∂t`），每个 `k×k` leaf 累加**充分统计量** `S=ΣAᵀA, r=ΣAᵀb, e=Σbᵀb`（逐帧）。
- 区域残差（= observed time derivative 能量）`E=Σ_t(e_t − r_tᵀS_t⁻¹r_t)`，**闭式且可累加**：合并两区域 = 相加统计量、O(D³) 解一次。
- Region Adjacency Graph + 优先队列，每步合并"代价 `Δ=E_AB−E_A−E_B` 最小"的相邻区域（Ward 式 linkage），直到 1 个区域，记录 **dendrogram**。之后按需 `cut`。
- 设计决策：完整 dendrogram 事后 cut；observer 3 DOF；静态空间划分（一棵树跨帧累加）。`J=∇v` 用全局有限差分（leaf 太小算不准）。

**接口**：
```python
from FLowUtils.referenceFrameDecompose import decompose_reference_frame_2d
dec = decompose_reference_frame_2d(field, k=2)          # 建 merge tree
print(dec.diag["interpretation"])                       # 诊断（见下）
labels = dec.cut(n_regions=3)                           # 或 cut(cost_threshold=...) -> (Y,X) 分区
obs    = dec.region_observers(labels)                   # {region: (T,3) killing 系数}
u      = dec.observer_field(labels)                     # 分片刚体 observer 场 (UnsteadyVectorField2D)
ns, res = dec.residual_curve(max_regions=20)            # 残差-区域数曲线，找肘部选 cut
```

**诊断**（`dec.diag`）——区分三种情形：
- `global_residual_ratio`（单一全局 observer 剩余非定常比例）小 → **单一全局足够**，无需分区。
- `finest_residual_ratio`（最细分区仍剩比例）大 → **本征非定常**（涡脱落/湍流），任何刚体分区都压不平。
- `decomposition_benefit=(global−finest)/raw`（分区相对原始能多消除的比例）大 → **多个流向**，分区显著有益。

**验证**：合成双流场（左半 +x 平移、右半 +y 平移）→ `cut(2)` 正确分左右（纯度 96.9%）、恢复各区 observer `(u1,0)`/`(0,u2)`；rfc → 诊断"单一全局足够"；cylinder2d → benefit 0.113，`cut` 出**近尾流区**（慢平移+旋转）与**下游涡街/自由流区**（平移≈来流 0.93），residual curve 肘部 n=2–3。

> 局限（未来可扩展）：静态空间划分对**移动结构**会切断（涡漂过区域边界）；硬边界可能割裂连贯结构（某些任务需软/重叠分配）；per-frame 划分、3D 版为后续扩展。
