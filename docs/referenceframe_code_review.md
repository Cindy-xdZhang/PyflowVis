# Reference-Frame 代码审阅 & 实验可复现性报告

本文档系统审阅所有 reference-frame 相关代码，并**诚实诊断**此前实验数字不稳定的根源。
结论先行：**代码与数学经逐项可执行验证无 bug**；此前"同样过程差好几 dB"的假象来自
**(a) 我跨对比时换了变量（未控制）+ (b) 没锁 cuda 确定性 + (c) SIREN 对输入微扰的混沌放大**。

---

## 1. 代码正确性审阅（逐文件）

审阅方法：独立审阅 agent 通读全部代码 + 编写可执行测试逐项验证 + 人工复核关键数学，
交叉对照 `docs/referenceframeOptimization.md`、C++ 参考实现、`VectorField2d/3d` 约定。

### 已验证**正确**的部分（附验证方式）
| 项 | 文件 | 验证 |
|---|---|---|
| Summed-area-table `_windowed_sum_nd`（2ⁿ 角公式、边界钳制、`np.ix_` gather） | ReferenceFrameOptimization.py | 对暴力钳制窗口和，2D/3D 各半径 max 误差 ~1e-14 |
| `np.add.reduceat` 块归约到 leaf（含非整除余块） | referenceFrameDecompose.py | 对暴力 k×k 块和，含 (7,9,2)/(9,9,4) max 误差 ~1e-16 |
| merge-tree：`E[c]` 与重算残差一致、cost=E_AB−E_A−E_B、lazy 删除、union-find、`leaf_rep` | referenceFrameDecompose.py | 双流场 143 次合并全部一致；cut(2) 纯度 100% |
| `observer_field` 掩码 view 赋值 `u0[:,mask]=…` | referenceFrameDecompose.py | 数值验证写回正确 |
| Jacobian 轴约定 `J[i,j]=∂v_i/∂x_j`、`np.gradient` axis、`_skew_field` | 两文件 | 2D axis1=x/axis0=y、3D axis2/1/0；`skew(a)·b=a×b` ✓ |
| 所有 M/A/L/W 矩阵符号 & v̂ 重建式（killing 2D/3D、gunther 2D/3D） | ReferenceFrameOptimization.py | 逐项对照 C++ `ReferenceFrame3d.cpp` 与解析符号检验 |
| pushforward 正/逆变换 | 实验脚本 | perfect-INR round-trip rel-err = **0.0**（完全可逆） |

### 发现并处理的问题
- **#1（真 bug，已修）** `_solve_sym_small` 原用 `np.linalg.lstsq(A,b,rcond=None)`。`rcond=None` 是**绝对**阈值
  (`eps·max(M,N)≈1e-15`)，对秩亏输入（纯刚体/解析场，A 列秩亏）保留近零奇异值 → 最小残差但**范数巨大**解
  （3D Killing 在纯刚体场上参数爆到 ~1e5）。**已改为相对 `rcond=1e-8`** 截断近零方向。主要影响 3D；2D 因
  `−v_t` 右端使最小范数解保持有界，此前 2D 实验未触发。
- **#2（docs 措辞，非代码错，待澄清）** 目标是 observed **time** derivative `L_u(v−u)=∂v/∂t−∂u/∂t+∇v·u−∇u·v`。
  对 **steady** 场（∂v/∂t=0）`u=0` 即全局最优 → observer 不可辨识。因此把 **steady** 的 `constant_rotation`
  (v=(y,−x)) 作**输入**会恢复 c=0（而非 −1）。我此前的 RFC 验证是用 **unsteady** 的 `rotation_four_center`
  作输入（恢复 c≈−0.96，正确），`constant_rotation` 仅作**参考真值**——但相关 docstring/docs 措辞会误导，
  需注明"输入必须非定常，steady 场 observer 不可辨识"。

---

## 2. 实验数字不稳定的根源（诚实诊断）

此前我报过的 rfc `cut=1` 数字 37.70 / 38.60 / 41.95，被我并列成"同样过程差好几 dB"。**真相**：

1. **未控制变量（我的方法论错误）**：这三个数其实是 **epochs=300/600/1000**（PSNR 随 epochs 单调升，欠训练现象）；
   `cylinder cut=1(52.5) vs global warp(45.5)` 是 **m64d4 vs m48d3 网络不同**；`warp(44.67) vs cut=1(41.95)` 还叠加
   **6-DOF vs 3-DOF observer**。每个"差异"背后都换了变量，且没锁确定性。

2. **cuda 训练非确定性**（已消除）：cuda matmul 默认非确定。已加
   `cudnn.deterministic + use_deterministic_algorithms + CUBLAS_WORKSPACE_CONFIG=:4096:8 + 固定 seed`。
   验证：**同一 observer 连训两次，PSNR 差 = 0.000000 dB**（确定性完全生效）。

3. **SIREN 混沌敏感性**（关键，待处理）：CoordNet(omega₀=30) 训练对输入微扰**混沌放大**——
   **observer 加 1e-9 噪声 → PSNR 差 ~4 dB**。global 与 cut=1 独立算 observer，浮点累加顺序差
   `9e-9` → 被放大成 0.68 dB。数学上 global 与 cut=1 是**同一个整场 killing observer**，此差异是浮点噪声，非方法差异。

> **要点**：这不是 pipeline bug（见 §1、§4 铁证），而是 (a) 未控制变量 + (b) 未锁确定性 + (c) SIREN 数值敏感性。
> 前两者已修；(c) 决定了**任何涉及 observer 的 PSNR 对比必须消除浮点噪声并降低 SIREN 敏感性**，否则真实方法差异会被淹没。

---

## 3. 可复现性规程（从此严格遵守）

1. **确定性**：脚本导入即设 cudnn/确定性算法/CUBLAS/seed。同输入 → bit-identical 输出（已验证 0.00 dB）。
2. **严格控制变量**：固定 INR（CoordNet m=64/d=4）、固定 epochs、固定 observer 定义（一阶 3-DOF killing），
   一次只改被测变量（如 cut 数）。**不再报任何未控制变量的数字。**
3. **observer 浮点噪声**：global 与 cut=1 数学同一 → 对比时用 **bit-identical observer**（cut=1 复用 global observer），
   消除 9e-9 累加差带来的伪差异。
4. **INR 选择（关键结论）**：CoordNet(SIREN, ω=30) 训练对这类 pushforward 数据**根本不稳定**——
   实测 ω(10/15/20/30)、lr(3e-4/1e-4)、epochs(300~4000) 全维度扫描，对 observer 的 **1e-9** 扰动
   敏感 **0.7~18 dB**，且 epochs 大时训练发散（PSNR 从 55→35）。根因是 sin 激活的非凸多峰 loss landscape，
   解依赖优化路径。**已改用 Fourier-features + ReLU MLP**（`scratchpad/ff_inr.py`：scale=8, hidden=512,
   n_freq=256, layers=4, ~1.31M params, epochs=4000, lr=1e-3）：同样 1e-9 扰动敏感性仅 **0.13 dB**、单调收敛不发散、
   PSNR 43.9（≈SIREN）。**此 INR 配置固定不变**，用于所有对比。
   > **教训**：INR（尤其高频 SIREN）的训练稳定性/可复现性必须**先验证**（对输入微扰的敏感性 ≪ 待测方法差异），
   > 再用它做任何 PSNR 对比——否则训练噪声会淹没甚至**伪造**方法差异。这是本轮所有"数字混乱"的最深根源之一。

---

## 4. 已验证事实清单（铁证，可复算）
- 输入逐元素一致：warp-logic 与 `fit_region`（同 observer）`max|coords/vals diff| = 2e-7`（float32 精度）。
- pushforward 可逆：perfect-INR round-trip `rel-err = 0.0`。
- 确定性：同 observer 连训两次 `ΔPSNR = 0.000000 dB`。
- SIREN 敏感性：observer `+1e-9` → `ΔPSNR ≈ 4 dB`（epochs=300）。
- SAT / 块归约 / merge-tree / 符号约定：见 §1 表，误差 1e-14~1e-16。

一阶 killing 定义（本项目当前工作模式）：`u(x,y)=(a,b)+c·(−y,x)`，a,b 为 x/y 平移速度、c 为角速度（3 未知量）。
二阶模式额外含 (a,b,c) 的一阶时间导数（共 6 未知量）；当前只用一阶。3D 以后类似拓展。
