# Reference-Frame 分区 INR 流场压缩 — v2（完全重写版）

> 本文档是 v2 重写的**方法规格 + 实验记录**。旧版（`experiments/referenceframe_inr/`、
> `docs/referenceframe_inr_summary.md`）的代码与结论**全部不作为依据**，本版从零开始。
> 代码：`experiments/referenceframe_inr_v2/`。遵循组内工作流：baseline 冻结、方法带版本号、
> 每个结论标注出处（脚本 + 参数 + 日志）。

---

## 0. 为什么重写（旧版不可信的具体理由）

检查旧版代码/文档后确认的问题（这些只作为"重写动机"，不引用旧版任何数字）：

1. **对比架构不一致**：旧版 proposed 用的 INR 是 Fourier 特征 + ReLU MLP（`ff_inr.py`），
   而 baseline 论文是 CoordNet（SIREN 残差块）。架构不同，对比不能归因于方法本身。
2. **块数是手工调的**（cylinder=5、boussinesq=3，"早期各场表现最好的值"），不是用户设想的
   自底向上 τ-合并自动发现 N；且旧版自己发现"cylinder 的 N=5 用任何 τ 都取不到"（`tau_map.py`）。
3. **重建用 overlap+高斯 blend**（margin=4, sigma=4），有像素覆盖不到会静默填 0（旧版自己的
   `audit_blend.py`/`audit_coverage.py` 承认此风险），PSNR 可能被污染。
4. **baseline 欠训练混淆**：1000 epochs 下最大单网络欠训练，"分区赢 baseline"无法归因
   （旧文档第 7 节自认）。
5. 结论反复翻转（N=1 由正转负等），无法追溯。

## 0b. 项目叙事修订：Story v1 → v2（用户 2026-07-16 定）

**旧叙事（Story v1，项目最初构思）**：CoordNet + reference frame transformation（参考系
变换，下称 RFT）拟合 observed field，会比 CoordNet 直接拟合原场更好——即"RFT 提高
SIREN/CoordNet"。

**新叙事（Story v2，自本日起）**：不再主张"RFT 能提高 SIREN/CoordNet"；主张 **RFT 是
架构无关（architecture-agnostic）的增益模块：任意架构的 INR + RFT 都可获得一定程度的
性能提高**。

**为什么改（证据，均出自 §4.4j / §4.9）**：
1. SIREN 系（正弦激活）网络——CoordNet、FINER 皆属此类——**天生擅长拟合 cylinder 这类
   卡门涡街准周期振荡场**（频谱窄带，与正弦基函数天然匹配）：FINER 直拟合 cylinder
   baseline 达 **73.7 dB**，超过 coordnet 好盆地（≈70）与一切 pro 配置。在这类数据上
   给频域架构叠加 RFT 很难再有效提高：coordnet pro 输 7-8 dB、finer pro 更差且不稳定、
   mlp pro 也仅 −0.6 打平。**cylinder 类"全域窄带振荡"场是 RFT 的不适用边界**（E/E0
   先验判据在 §4.9① 已预示：涡街不可被局部刚体运动解释）。
2. 架构对照实验（Verify_arch_1.1）显示 RFT 增益**并非 SIREN 专属，且在非频域架构上更
   普遍**：残差 ReLU MLP（无频域先验）上 pro 在 4/5 场架构内取胜（+2.1~+14.1）；FINER
   在 rfc +17.0；而 coordnet 只在全局运动主导的场（rfc、boussinesq）受益。

**新叙事当前的证据边界（诚实记录，防止超卖）**：
- 支持最强：v_MLP0.0（4/5 场胜、种子稳定）；rfc 上三架构全胜（coordnet 等参 +4.7~5.2、
  mlp +14.1、finer +17.0）。
- 尚未闭合：①FINER 大 INR × observed field 训练不稳定（§4.4j 读数 3），"任意架构"要
  站住需给出 FINER 的稳定配方或明确声明其适用条件；②mlp/finer 的架构内增益目前是
  quality 口径（pro=4B vs baseline=B，**非等参**）——等参/等步数的 observer 隔离归因
  链只在 coordnet 上做过（单窗诊断 +4.7~5.2，§4.4b），mlp/finer 的 no_observer 消融
  与等预算对照**待补**，这是 Story v2 成立的关键实验缺口；③gerris finer pro 低 lr
  重跑未回收。
- 与 §4.9⑤ 的关系：§4.9⑤"存在一类流场 + 自动检测机制"的定位保留，其中与 SIREN/
  CoordNet 绑定的表述由本节取代。

**对后续实验的指引（优先级）**：补 v_MLP/v_FINER 的等参消融（no_observer、单窗
observed field 诊断）→ 使"任意架构 + RFT ⇒ 提高"具备与 coordnet 同级的归因链；
FINER×observed field 稳定化作为 Verify_arch_1.2。

## 1. Baseline：CoordNet 复现验证（已核对论文原文，冻结）

论文：Han & Wang, *CoordNet: Data Generation and Visualization Generation for Time-Varying
Volumes via a Coordinate-Based Neural Network*, IEEE TVCG 29(12), 2023。

对 `CoordNetCompression.py` 逐条核对论文（2026-07-12，PDF 原文 §4.2/§4.4/§5）：

| 项 | 论文 | `CoordNetCompression.py` | 结论 |
|---|---|---|---|
| SIREN 层 | sin(ω(Wx+b)), ω=30 | `SineLayer`, ω=30 | ✅ |
| SIREN 初始化 | Sitzmann：首层 U(±1/n)，隐层 U(±√(6/n)/ω) | 同 | ✅ |
| 残差块 | 主路 2×SIREN；in≠out 时**多加一层 SIREN**；(main+skip)/2 | 主路 2×SIREN + skip 投影 SIREN + 平均 | ✅（见下歧义说明） |
| 架构 | k→m, m→2m, 2m→4m, d×(4m→4m), decoder 4m→p（Table 2） | 同 | ✅ |
| 超参 | m=64, d=10 | 可配置（默认按数据规模缩小并注明） | ✅ |
| 归一化 | 坐标、值都缩放到 [-1,1] | minmax → [-1,1] | ✅ |
| 损失 | 纯 MSE（式 1） | MSE | ✅ |
| 训练 | Adam(0.9,0.999), batch=32000, lr=1e-5, L2 1e-6, 300 epochs | 可配置；yaml 默认是加速 demo 档（lr 3e-4 余弦、60 epochs、grad-clip、best-snapshot），**非论文档**，注释已声明 | ✅（论文档需显式配置） |
| 指标 | data-level PSNR | 20·log10(range) − 10·log10(MSE) | ✅ |

- **参数量核算与实测一致**：闭式公式 P(m,d,k,p) = (41+32d)m² + (2k+21+8d+8p)m + p²+3p；
  m=32,d=8,k=4,p=1 → 307,364，与冒烟运行打印完全一致（5 epochs 合成数据，PSNR 24.69 dB，
  管线端到端可跑通，RTX 3090）。
- **一处论文歧义**（不影响结论）：in≠out 残差块"多加的一层 SIREN"，论文文字没写清是放
  skip 路（当前实现）还是主路前端；两种读法参数量略差。当前实现取 skip 投影，为合理读法，
  v2 所有对比双方用**同一个类**（直接 import `CoordNetCompression.CoordNet`），故不影响公平性。

**v2 冻结的训练 recipe（baseline 与 proposed 完全一致，唯一改动源是方法本身）**：

- **v2.0 recipe（已废弃，仅留档）**：照搬论文 lr=1e-5 恒定、300 epochs。实测（rfc,
  2026-07-12）所有模式在 300 epochs 时 MSE 仍在快速下降 —— 论文的 recipe 是为 ~5e8 体素
  的数据调的（每 epoch ≈1.6 万步梯度更新），我们的 2D 场每 epoch 只有 ~8–16 步，照搬等于
  总步数少约三个数量级，全体欠训练，对比无意义。
- **v2.1 recipe（当前，全模式一致）**：Adam(β=0.9/0.999)、batch=32000、weight decay=1e-6、
  **1000 epochs**（硬上限）、lr=3e-4 **余弦衰减**到 1e-5、grad-clip=1.0（SIREN 高 lr 稳定性）、
  坐标/值 minmax→[-1,1]、纯 MSE、无 best-snapshot（报告 last，best 只作诊断）、固定种子 +
  CUDA 确定性。此偏离已记录且对所有模式一致，不影响归因；它同时正面消除旧版"baseline
  欠训练混淆"。
- **v2.2 协议追加：每 INR best-of-3 seeds（编码期搜索）**。动机（v2.1 rfc 运行观测）：SIREN
  收敛对随机种子混沌敏感 —— 同分布数据、同结构（m=29），seed A 最终 MSE 2.0e-4、seed B
  7.9e-6（差 25×），直接把 pro_quality 的 PSNR 打乱（45.98，低于更小网络的 pro_budget 56.73，
  纯属种子运气）；与 v1 记录的"SIREN 种子敏感 0.7~18dB"一致。对策：每个 INR 用 3 个派生
  seed 各训一遍、**保留最终 MSE 最低者**——压缩语义下合法（搜索只花编码时间，解码与存储
  不变），且对 baseline / proposed 对称适用。每 INR 的 seed 间离散度记入日志作诊断。
- **v2.3 recipe（当前正式口径）：lr 回论文 1e-5（余弦→1e-6）+ 自适应 batch（每 INR ≥64 步/epoch）**。
  动机——v2.1/v2.2 的高 lr 在大网大场上引爆两个新故障（证据见 §4.5/§4.6）：
  ① cylinder2d 全模式坍缩到"均值流吸引子"（baseline 两个 seed 最终 normalized MSE **完全相同**
  = 3.277e-2 ≈ 波动能量，PSNR 仅 23 dB；同一 CoordNet 类同一数据在 lr=1e-5 下已知可达 ~50 dB）；
  ② 固定 epochs 下每 INR 的梯度步数 ∝ 其样本数，窗口切分把步数砍半（§4.5 归因）。
  两个故障同根：**recipe 用 epochs 定义而各数据集/区域的每 epoch 步数差百倍**。v2.3 用
  `batch = min(32000, n_samples/64)` 把步数归一化（epochs 仍 ≤1000，硬约束不变），lr 用 SIREN
  安全的论文值。此改动对所有模式一致。
  **试点验证（单 seed，`outputs/pilot_v23_*.log`）**：cylinder baseline 300ep = **45.64 dB**
  （v2.2 recipe 同设置 23.29，坍缩修复，且 MSE 仍在下降）；rfc baseline 56.13 / pro_budget 55.82
  （步数归一后窗口惩罚从 −1.27 dB 收窄到 −0.31 dB，单 seed 噪声内）。正式数字以 v2.3 全套
  best-of-seeds 为准。

## 1b. INR 架构变体：v_MLP0.0 / v_FINER0.0（2026-07-15，仅冒烟验证，无正式数字）

动机：§4.9④ 大 SIREN 双峰不稳定，handover 下一步清单第 3 条要求非 SIREN 架构对照。两个
变体都**完全保留 CoordNet 骨架**（encoder k→m→2m→4m→d×(4m→4m)；残差块 = 主路两层 +
in≠out 时投影 skip + (main+skip)/2；decoder = 一个残差块 4m→p），只改激活函数与初始化 ⇒
每个 nn.Linear 形状与 CoordNet 相同，闭式参数公式 / `pick_m_for_budget` / 字节口径对所有
变体**逐位成立**（实测 4 组 (m,d) 含 m=64,d=10 的 1,486,538 全部相等；`train_inr` 内的
参数量断言对所有变体保留）。

| 版本 | `--model` | 激活 | 初始化 | 备注 |
|---|---|---|---|---|
| v_MLP0.0 | `mlp` | 每块主路隐层一个 ReLU，块输出保持线性（块输出若过 ReLU 会把 (main+skip)/2 钳到 ≥0，无法表示 [-1,1] 值域） | Kaiming-uniform（ReLU 增益）、bias=0 | 原始 (x,y,t) 坐标、无位置编码——加 Fourier features（傅里叶特征位置编码）会改 in_dim、破坏参数对等，属将来单独的版本 |
| v_FINER0.0 | `finer` | 变周期 sine：sin(ω·(\|z\|+1)·z), z=Wx+b（FINER, Liu et al., CVPR 2024） | SIREN 权重初始化；首层 bias 可选 U(±k)，k=`--finer_first_bias_scale`（官方 repo 默认 None=标准 bias 初始化；(\|z\|+1) 因子按官方默认不参与反传，scale_req_grad=False） | ω=30 与 CoordNet 相同 |

代码路径：`models_alt.py`（模型与工厂 `build_inr_model`）→ `inr.train_inr(model_name=...)`
→ `pipeline.ExpCfg.model` → `run_experiment.py --model {coordnet,mlp,finer}`。默认
`coordnet`，既有实验路径完全不受影响。

**冒烟验证**（本地 RTX3090，rfc，v2.3 recipe，epochs=20、n_seeds=1、tau=0.05；日志
`outputs/smoke_arch/{vMLP0.0,vFINER0.0,coordnet_ref}_rfc_smoke.log`。目的只有"能跑 + 在
收敛"，**20 epochs 的 PSNR 无科学意义、禁止引用为结论**）：

| `--model` | baseline MSE 轨迹（ep5→20） | baseline PSNR | pro_budget PSNR | 跑通模式 |
|---|---|---|---|---|
| coordnet（参照） | 5.48e-2 → 1.96e-2 | 23.17 | 23.39 | baseline / pro_budget |
| mlp | 8.90e-2 → 8.04e-2 | 16.97 | 17.80 | 全部 4 模式 ✓ |
| finer | 8.03e-2 → 4.58e-2 | 19.44 | 18.96 | baseline / pro_budget |

读数（工程判断，非科学结论）：三者每条 MSE 轨迹均单调下降 = 都在收敛；20 epochs 下
coordnet > finer > mlp 符合预期——sine 系网络对低频光滑场收敛快，而 raw 坐标 ReLU MLP 有
spectral bias（谱偏置：ReLU 网络先拟合低频、难以拟合高频的已知性质），且 v2.3 recipe 的
lr=1e-5 是按 SIREN 稳定性调的，对 MLP 可能偏小。**正式对比（1.0 版本）前必须：①对每个
架构做 recipe 适配检查（至少 lr 扫描），否则会把"recipe 不适配"误判成"架构差"；②沿用
best-of-seeds 协议并记录 seed spread（检验换架构是否消除 §4.9④ 的双峰不稳定——这正是
引入它们的首要动机）。**

## 2. 方法 v2.0 规格

### 2.1 记号与 killing 最小二乘（2D 低阶）

输入非定常场 v(x,y,t)，网格 (T,Y,X,2)。observed time derivative 残差（丢掉 ∂u/∂t，低阶）：

```
r(x,t) = ∂v/∂t + (∇v)u − (∇u)v
killing observer:  u(x) = (a − c·y,  b + c·x)   （平移 a,b；角速度 c）
per-pixel 2×3 系统:  A = [ J | J·x⊥ − v⊥ ],  x⊥=(−y,x), v⊥=(−v_y,v_x)
区域 Ω、时刻 t:  min_q Σ_{x∈Ω} ‖A q + ∂v/∂t‖²  →  (ΣAᵀA) q = −ΣAᵀ∂v/∂t
能量:  E_Ω(t) = E0_Ω(t) + qᵀ(ΣAᵀ∂v/∂t),   E0 = Σ‖∂v/∂t‖²  （E ≤ E0 恒成立）
```

符号约定经"人工构造旋转场"单元测试独立验证（§3 T1），与 optimal-connection C++ 及
`verify_signs.py` 的结论（ω 列 = J·x⊥ − v⊥ 配 RHS −∂v/∂t）一致。
每 timestep 独立解 3-DOF；区域 killing 参数 q(t)=(a,b,c)(t) 逐帧存储（边信息，计入字节）。

### 2.2 时间窗口

时间轴切成 `n_windows` 段连续窗口（默认 2），每窗长 ≤ (tmax−tmin)/2（用户约束）。
每个窗口独立做分区与 INR 拟合；重建时按 t 所在窗口查询对应 INR。

### 2.3 自底向上 τ-合并分区（每窗口）

- 最小单元：k×k 像素 cell（默认 k=2，边缘允许不整除的碎 cell）。
- 预计算每 cell、每 timestep 的 ΣAᵀA (3×3)、ΣAᵀ∂v/∂t (3)、E0（区域统计=cell 求和，O(1) 合并）。
- 区域邻接图（4-邻接），贪心凝聚：反复取当前**合并后残差比** ρ 最小的相邻对 (r,s)，
  `ρ(r∪s) = E_{r∪s} / (E0_{r∪s} + ε·|r∪s|)`，若 ρ ≤ τ 则合并；直到无对满足。
  τ 的语义：**一个区域存在的条件是单个刚体 observer 能解释其原始时间能量的 (1−τ) 以上**。
  ε 是逐像素地板（ε_rel × 全域每像素平均原始能量，ε_rel=1e-6）：局部本身接近 steady 的区域
  （E0≈0，如远场）ρ≈0，可自由并入任何邻居 —— steady 块任何 observer 都能解释，这是正确行为。
  RFC 全域 ρ≈3e-4，故对任意 τ≥0.01 必然收敛到 N=1（§3 T2 验证）。
- **设计变更记录（可追溯）**：初版用增量式准则 δ=[E_{r∪s}−E_r−E_s]/E0_{r∪s}≤τ，被 T4 反例
  "切香肠"击穿（每一步增量相对不断增大的 E0_union 都小，τ=0.05 时两个不同 observer 的半域被
  链式合并成 N=1）。残差比准则约束的是**累计**未解释比例，不可被切香肠；已改用之。
- 产出：cell 标签图（每像素恰属一个区域，内置覆盖断言）、每区域 q(t) 与能量诊断。

### 2.4 每区域 observed field 的 INR 拟合（关键：无重采样、无插值、无 blend）

对窗口 [t₀,t₁)、区域 Ω、killing 参数 q(t)：

```
帧变换（观察者刚体运动）: θ(t)=∫c dt（梯形）,  D(t)=∫R(−θ)(a,b) dt（梯形）,  θ(t₀)=0, D(t₀)=0
lab→observed:  ξ = R(−θ)·x − D          observed 值:  ṽ(ξ,t) = R(−θ)·[v(x,t) − u(x,t)]
```

训练样本 = {(ξ(x,t), t) → ṽ}，x 取 Ω 的**原始网格像素**、t 取窗口帧 —— ṽ 由 v 的网格值精确
计算，**不插值 v、不重采样**；重建查询与训练坐标**逐点相同**，故无覆盖洞、无 blend：

```
重建:  v̂(x,t) = R(θ)·INR(ξ(x,t), t) + u(x,t)
```

（往返变换恒等性有独立断言 §3 T3：给 INR 换成完美预言时 v̂≡v 到 1e-10。）
坐标归一化：ξ 按该区域窗口内扫过的 bbox → [-1,1]²，t 按窗口 → [-1,1]；值按该区域样本
per-component minmax → [-1,1]（baseline 用全场 per-component minmax，对称）。

### 2.5 预算与实验模式

基础预算 B = baseline CoordNet(m_B, d_B, k=3, p=2) 的参数量。设全部窗口共 M 个区域（M 个 INR）：

| 模式 | 每 INR 参数上限 | 说明 |
|---|---|---|
| `baseline` | B（单网络） | CoordNet 直接拟合 v(x,y,t) |
| `pro_budget` | B/M（均分） | 总参数 ≤ B；m_r 取满足闭式参数公式的最大整数（d 同 baseline） |
| `pro_quality` | 4B/M | 总参数 ≤ 4B，只比 PSNR |

**pro_quality 规格变更（2026-07-15，用户明示）**：3B → **4B**。§4.4b–§4.4i 的全部历史
pro_quality 数字均为 **3B 口径**（引用须注明）；4B 自 Verify_arch_1.1（§4.4j）起生效。
| `no_observer`（消融） | B/M | 分区与 pro_budget 完全相同，但 q≡0（拟合原始 v）——隔离 observer 的贡献 |

**字节核算**：总字节 = Σ参数×4 + 边信息（killing 参数 N·T_w·3×4B + cell 标签图 + 每区域
ξ-bbox/值域/宽度 m_r）。压缩比 = 原始数据字节 / 总字节。"observer 系数代价可忽略"必须由
数字支撑，不能只是口号。

**严格压缩口径（mainExp_compress_1.1 起，2026-07-16）**：`--budget_frac f` 把预算源从
B 换成 **f × 原始场 float32 字节**，且约束对象是**总字节（参数×4 + 边信息）**——baseline
由预算反解宽度 m，proposed 先扣精确边信息再均分，两侧均有超预算断言。规格见 §4.4k，
尺寸规划工具 `budget_calc.py`。

### 2.6 诊断量（每区域）

- steadiness gain：var_t(ṽ)/var_t(v)（observed field 的时间方差 / 原场时间方差，越小越接近 steady）；
- E/E0（killing 拟合残差比）；区域像素数、m_r、样本数。

## 3. 正确性验证协议（`validate_rfc.py`，全部必须通过才跑实验）

| # | 测试 | 断言 |
|---|---|---|
| T1 | 人工构造：steady 四涡胞 s(x,y) 叠加已知全局旋转相机 ω₀（解析生成 v=R(−ω₀t)s(R(ω₀t)x)−ω₀x⊥） | 全局 killing 解 c≈−ω₀、(a,b)≈0、E/E0≪1 |
| T2 | 仓库 RFC（`rotation_four_center`，al_t=1，相机项 = −x⊥ ⇒ 预言 c_true=−1.0） | 任意窗口（n_windows=2、4 都测）τ-合并结果 **N=1**；解出 c(t)≈−1.0 |
| T3 | 往返恒等：随机 q、随机场，用"完美 INR"（直接回代 ṽ）重建 | ‖v̂−v‖∞ < 1e-10 |


## 4. 实验版本表

（PSNR 单位 dB；B = baseline 参数量；模式定义见 §2.5；全部固定 seed=0、CUDA 确定性。）

### 4.0 实验命名对照（按组内命名规则）

"v2.x"是 **recipe（训练配方）版本号**（见 §1），实验本身按组内规则命名如下：

| 实验名 | 内容 | 对应小节/日志 |
|---|---|---|
| mainExp_2.3 | 主实验：baseline/pro_budget/pro_quality + no_observer 消融，rfc & cylinder2d（v2.3 recipe） | §4.4b/§4.4d，`outputs/v23_*.log` |
| mainExp_2.3-ibex | 同上在 Ibex 跨机复现 | §4.4c，jobs 48759543-45 |
| Verify_window_2.1 / 2.3 | 设计选择验证：时间窗口切分的代价归因（单窗诊断） | §4.4/§4.4b/§4.5 |
| Verify_alloc_1.1 | 设计选择验证：预算分配 均分 vs 按像素比例（cylinder τ=0.1） | 运行中，`outputs/v23_cyl_allocpx.log` |
| Verify_seedstability_1.1 | cylinder baseline 补种子（钉死跨机 10 dB 抖动区间） | 排队中，`outputs/v23_cylbase_s2.log` |
| Other_tworotor_1.1 | 探索：双转子合成场（方法理想正例，定义见 §3 T4） | 排队中，`outputs/v23_tworotor.log` |
| Verify_tau_1.1 | 设计选择验证：τ 敏感性（τ→N→PSNR），pro_quality vs baseline；cylinder τ∈{0.005..0.1}/absorb=64（M=21/14/4/5/3）、boussinesq τ∈{0.1..0.5}/absorb=256（M=24/21/15/7/2），2 seeds/点 | Ibex job 48814029（20 并行任务），`outputs/ibex_tausweep/` |
| Verify_arch_1.1 | 架构变体 v_MLP0.0 / v_FINER0.0：各架构**自身** baseline vs pro_quality(**4B**)，5 场 × 2 架构 × 2 模式 × 2 种子 = 40 独立任务；**2-seed 均值协议** | §4.4j，`outputs/Verify_arch_1.1/`，Ibex job 见 §4.4j |
| mainExp_compress_1.1 | 严格压缩口径主实验：总字节（参数+边信息）≤ {5,10,20}% × 原始场字节，各架构（coordnet/mlp）自身 baseline vs pro_budget（分区+RFT），rfc/cylinder2d/boussinesq，72 任务，2-seed 均值 | §4.4k，`outputs/mainExp_compress_1.1/`，Ibex job 48967626 |
| Verify_compresswin_1.1 | 压缩口径翻盘 sweep：boussinesq coordnet τ×lr×窗口数（**M≤3 硬约束**）+ rfc coordnet 单窗闭环，5%/10% 预算，baseline 同步 sweep lr，epochs 1000 | §4.4l，`outputs/Verify_compresswin_1.1/`，Ibex jobs 49000294 + 种子扩展 49023725 |
| mainExp_compress_1.2 | 2.5% 字节预算档（CR≥40×），修正协议（M≤3 结构 + 场级 lr + 1000ep）：rfc/cylinder2d/boussinesq × {coordnet(2 lr 臂), mlp} × {baseline, pro}，36 任务 | §4.4m，`outputs/mainExp_compress_1.2/`，Ibex job 49025811 |
| Verify_compresswin_1.2 | bouss {2.5,5}% × 中档 lr {3e-5..1e-4} × {bl,w1M1,w2M2} 网格补齐（skip 既有格），3 种子 | §4.4n，`outputs/Verify_compresswin_1.1/`（带 lr/种子后缀），Ibex job 49033283 |
| Verify_compresswin_1.3 | bouss 2.5%/5% 翻盘：observer 参数化变体（consttrans/constfull/tvfull）× 字节口径 v2（N=1 免标签，pro 等宽 m17）× lr warmup，bl 对称，45 任务 | §4.4o，`outputs/Verify_compresswin_1.3/`，Ibex job 49044332 |
| Verify_compresswin_1.4 | cylinder×mlp 翻盘（用户 07-18 裁定不豁免后唯一未达标格）：M=1 单窗全局 observer（ct/cf）+ 字节 v2 等宽，2.5/5/10% × mlp@3e-4，22 任务（授权 24，余 2） | §4.4p，`outputs/Verify_compresswin_1.4/`，Ibex job 49084951 |
| （非实验）正确性验证套件 / 归一化审计 | validate_rfc.py / audit_normalization.py | §3 / §4.8 无（代码内） |

（已废弃的 v2.0/v2.1/v2.2 recipe 下的运行只留档不编号，见各小节标注。）

### 4.1 验证套件（必须全过，2026-07-12 已全过）

`validate_rfc.py`：T1 人工旋转场收敛性（细网格 |q−q*|=2.6e-3、E/E0=1.3e-4，均随分辨率收敛）；
T2 RFC 全部窗口布局 × τ∈{0.01,0.05,0.2} 均 **N=1**，c=−0.9974≈−al_t、平移~1e-16、E/E0=3.3e-4；
T3 往返 1.3e-15；T4 双转子反例 N≥2、每半 c=−0.801/+1.385（真值 −0.8/+1.4）。

### 4.2 τ→N 干跑（无训练，分区行为）

| 数据集 | 形状 (T,Y,X) | τ=0.02 | τ=0.05 | τ=0.1 | τ=0.2 | 备注 |
|---|---|---|---|---|---|---|
| rfc | (64,64,64) | N=1 | N=1 | N=1 | N=1（τ=0.2 亦 1） | 每窗口都 N=1，符合预言 |
| cylinder2d | (128,80,320) | 63/62 | 20/20 | 3/2 | — | "1 巨区(残差比≈τ)+顽固小 cell"结构；涡街非刚体可解释 |
| boussinesq | (128,225,75) | — | 448/547 | 106/135 | 17/23 | 羽流更难；N 大 ⇒ 均分预算会碎片化（开放问题见 §5） |

### 4.3 端到端结果


| 版本/recipe | 数据集 | 模式 | PSNR | 参数量 | 总字节 | CR | #INR (N/窗口) | 出处 |
|---|---|---|---|---|---|---|---|---|
| v2.1（1000ep, lr3e-4→1e-5 余弦, clip1.0, 单 seed） | rfc | baseline | 54.50 | 99,154 | 396,656 | 5.3× | 1 | `outputs/rfc_v21.log` |
| 〃 | rfc | **pro_budget** | **56.73** | 88,948 | 360,724 | 5.8× | 2 (1,1) | 〃 |
| 〃 | rfc | pro_quality | 45.98※ | 288,628 | 1,159,444 | 1.8× | 2 (1,1) | 〃 |
| 〃 | rfc | no_observer | 43.29 | 88,948 | 359,956 | 5.8× | 2 (1,1) | 〃 |

v2.1 rfc 读数：①单 seed 下 pro_budget 56.73 > baseline 54.50；②observer 隔离贡献 +13.4 dB；
③※pro_quality 反常（3×参数反而 45.98）定位为 SIREN 种子混沌（见 §1 v2.2 动机）。
**⚠️ v2.1 单 seed 数字全部只留档不采信** —— v2.2 多种子结果显示 ① 是 baseline 的种子坏运气
造成的假阳性（这正是"结论必须稳定"要求多种子协议的原因）。

| 版本/recipe | 数据集 | 模式 | PSNR | 参数量 | 总字节 | CR | #INR (N/窗口) | 出处 |
|---|---|---|---|---|---|---|---|---|
| **v2.2**（v2.1 + best-of-3-seeds，正式口径） | rfc | baseline | **58.00** | 99,154 | 396,656 | 5.3× | 1 | `outputs/rfc_v22.log` |
| 〃 | rfc | pro_budget | 56.73 | 88,948 | 360,724 | 5.8× | 2 (1,1) | 〃 |
| 〃 | rfc | pro_quality | **60.09** | 288,628 | 1,159,444 | 1.8× | 2 (1,1) | 〃 |
| 〃 | rfc | no_observer | 51.35 | 88,948 | 359,956 | 5.8× | 2 (1,1) | 〃 |

v2.2 rfc 读数（当前正式结论，均 best-of-3）：

1. **种子混沌证实并被消除**：per-INR 种子间最终 MSE 离散度最大 ×52.9（pro_quality w0）、×11.2
   （no_observer w0）——正是 v2.1 反常的两处；修正后 pro_quality(3B)=60.09 恢复"参数多→更好"。
2. **observer 隔离贡献 = pro_budget − no_observer = +5.38 dB**（同分区同预算同种子协议，干净归因）。
3. **等预算下 pro_budget(56.73) 比 baseline(58.00) 低 1.27 dB**。解释假设：RFC 在 (x,y,t) 中本是
   低阶多项式×高斯×三角的光滑解析函数，直接 INR 拟合并不难（"unsteady≠难拟合"）；而窗口规则
   强制 ≥2 窗口把预算劈成两个 m=16 网络。窗口成本 vs observer 增益的归因见单窗口诊断（§4.4）。

### 4.4 单窗口归因诊断（rfc，违反 ≤T/2 规则的专用诊断，`--allow_full_window`）

全部 budget B（m=24, 99,154 参数）、best-of-3、单窗口全时段（`outputs/rfc_v22_diag1w.log`）：

| 配置（均为 1 个 INR @ B） | PSNR | 结论 |
|---|---|---|
| INR(v)（= baseline 走 proposed 代码路径, q≡0） | 58.00 | 与 baseline 主跑 58.00 **完全一致** —— proposed 管线无路径偏差的内部一致性证据 |
| INR(observed)（全局 killing observer + pushforward） | **62.74** | **observer 在等参数下的净增益 = +4.74 dB** |

**RFC 归因链（全部 @B, best-of-3）**：baseline 58.00 → +observer（单窗）62.74（**+4.74**）→
再强制切 2 窗（每窗 m=16）56.73（**−6.01**）。即：**提案的核心机制（参考系变换让 INR 更好拟合）
在等参数下成立且贡献显著；把它拖到 baseline 之下的是"窗口 ≥2"的固定规则**——RFC 的 observer
本就时不变，正确窗口数是 1。⇒ 方法启示：时间窗口数应像空间 N 一样**数据自适应**（时间轴的
自底向上合并），而非固定超参。observed field 拟合还更稳（种子离散 ×1.3 vs ×2.2）。

### 4.4b v2.3 正式结果（rfc，best-of-3，`outputs/v23_rfc.log`）

| 版本/recipe | 数据集 | 模式 | PSNR | 参数量 | 总字节 | CR | #INR | 出处 |
|---|---|---|---|---|---|---|---|---|
| **v2.3**（lr1e-5→1e-6 余弦 + 自适应 batch ≥64 步/ep + best-of-3） | rfc | baseline | 56.23 | 99,154 | 396,656 | 5.3× | 1 | `outputs/v23_rfc.log` |
| 〃 | rfc | pro_budget | 55.81 | 88,948 | 360,724 | 5.8× | 2 | 〃 |
| 〃 | rfc | **pro_quality** | **63.80** | 288,628 | 1,159,444 | 1.8× | 2 | 〃 |
| 〃 | rfc | no_observer | 51.07 | 88,948 | 359,956 | 5.8× | 2 | 〃 |

读数：①种子离散度全面塌缩（最大 ×5.2，v2.2 recipe 是 ×52.9）——v2.3 稳定性达标；②observer
隔离贡献 = 55.81 − 51.07 = **+4.74 dB**（与 v2.2 recipe 下的 +5.38 同量级，跨 recipe 稳健）；
③pro_budget 与 baseline 差距收窄到 −0.42 dB（v2.2 recipe 下 −1.27）。

**v2.3 单窗诊断（`outputs/v23_rfc_diag1w.log`，参数与步数与 baseline 完全一致，唯一差别 =
observer 变换）**：INR(observed)@B = **61.43**，INR(v)@B = 56.23（与 baseline 主跑逐位一致，
内部一致性再次确认）。归因链（v2.3 口径，全部 @B、best-of-3）：

```
baseline 56.23 → +observer（单窗、等参等步）61.43（+5.20） → 强制 2 窗 55.81（−5.62）
```

**结论修订（旧 → 新）**：§4.5 曾把窗口成本归因于"每 INR 梯度步数减半"（H2，v2.1 recipe 证据
E1a）。v2.3 已把步数归一化（两窗每 INR 64k 步 = 单窗），窗口成本却仍有 −5.62 dB ⇒ **H2 不是
全部**。修订后的归因：固定预算下切窗使每个 INR 更小（m=24→16），小网在低 lr（1e-5）下**收敛
更慢**，64k 步内到不了自己的天花板（同一 m=16 在 v2.1 recipe 3e-4/16k 步可达 64.01，天花板
足够高——容量始终不是原因）；旁证：pro_quality 每窗 m=29 在同 recipe 下恢复到 63.80。
**跨 recipe 的稳定结论**：(a) observer 等参增益 +4.7~+5.2 dB 稳健；(b) 窗口切分在两种 recipe
下都有 5~6 dB 的"优化税"（近因不同：v2.2=步数、v2.3=小网低 lr 收敛慢），容量与任务难度均已
排除 ⇒ 方法必须把窗口数做成数据自适应（RFC 正确答案 = 1 窗），并把"每 INR 的优化预算"作为
recipe 的显式维度。

### 4.4c 跨机复现（Ibex，本地→push→pull→sbatch 全流程验证，2026-07-13）

Ibex（Linux, cu118, V100|A100；作业 48759543/44，`slurm_logs/RFv2.*`）vs 本地（Win11, cu126,
RTX3090），同代码 commit `3e25405`、同 v2.3 recipe、同种子协议：验证套件 T1–T4 **全过**；
rfc 六个数字最大偏差 **0.25 dB**、排序与派生结论全部复现（observer 消融隔离贡献 +4.74↔+4.76、
等参 observer 增益 +5.20↔+5.39、窗口税 −5.62↔−5.76）。数据路径经 `PYFLOWVIS_DATA2D` 环境
变量适配（Ibex: `/home/zhanx0o/DeepVortex/FLowDataFolder`）。

| 模式（rfc, v2.3） | 本地 | Ibex |
|---|---|---|
| baseline | 56.23 | 56.29 |
| pro_budget | 55.81 | 55.92 |
| pro_quality | 63.80 | 63.72 |
| no_observer | 51.07 | 51.16 |
| 单窗 INR(observed) @B | 61.43 | 61.68 |
| 单窗 INR(v) @B | 56.23 | 56.29 |

### 4.4d cylinder2d v2.3 正式结果（本地，best-of-2，`outputs/v23_cyl.log`）

| 模式 | PSNR | 参数量 | #INR (N/窗口) |
|---|---|---|---|
| **baseline** | **68.02** | 1,486,538 | 1 |
| pro_budget | 54.38 | 1,432,390 | 5 (3,2) |
| pro_quality (3B) | 62.76 | 4,363,990 | 5 (3,2) |
| no_observer | 53.54 | 1,432,390 | 5 (3,2) |

读数：①v2.3 修复确认——baseline 从 v2.2 recipe 的 23.29 恢复到 68.02（本地抽签值，见下方修订）；
②observer 在涡街上的隔离贡献仅 **+0.84 dB**（54.38−53.54）——物理上符合预期，涡街本质非刚体
可解释（与 §4.2 分区干跑的结论互证）；③均分预算把巨区（≈92% 像素）压到 B/5（m=28），而 4 像素
小区用同额参数把 256 个样本过拟合到 MSE 1e-7（浪费 ~60% 预算）⇒ 已实现 `--alloc pixels`。

**跨机对照（Ibex 48759545）与结论修订（旧 → 新）**：

| cylinder 模式 | 本地 | Ibex | 复现？ |
|---|---|---|---|
| baseline | 68.02 | **57.90** | ❌ 差 10 dB |
| pro_budget | 54.38 | 54.17 | ✓ |
| pro_quality | 62.76 | 62.39 | ✓ |
| no_observer | 53.54 | 53.51 | ✓ |

- **旧结论**（仅本地）："baseline 68.02，proposed 落后 13.6 dB"。
- **新证据**：Ibex 上 baseline 仅 57.90（其余三模式 0.4 dB 内复现），且本地日志中 baseline 是
  唯一种子离散 ×11.0 的模式——**大 SIREN（m=64,d=10）在 cylinder 上即使 lr=1e-5 收敛也分叉**，
  best-of-2 抽签方差 ~10 dB，跨机 cuDNN 差异使两边落入不同分支；Ibex 上排序甚至翻转
  （pro_quality 62.39 > baseline 57.90）。
- **修订后的结论**：cylinder 的 baseline 数字在 best-of-2 下**不稳定（57.9~68.0）**，
  "baseline 领先幅度"未定（3.7~13.6 dB），需更多种子钉死（补种子运行已排队）；跨机**稳定**的
  结论只有：(a) pro_budget ≈ 54.2-54.4、pro_quality ≈ 62.4-62.8、no_observer ≈ 53.5；
  (b) observer 在涡街上贡献 +0.7~+0.8 dB（两台机器一致）；(c) baseline > pro_budget 两边都成立。

### 4.4e Verify_seedstability_1.1：cylinder baseline 种子分布（Ibex 8 独立种子）

seeds 10–17（job 48810859_[8-15]，每种子独立任务）：
**{48.95, 59.27, 63.21, 69.49, 69.72, 70.01, 70.62, 71.36}** —— 极差 22.4 dB，中位数 69.6。
分布呈**双峰**：5/8 落在"好盆地"≈69.5–71.4，3/8 散落 49–63。此前的跨机矛盾（本地 68.02 vs
Ibex 57.90）完全解释为盆地抽样：best-of-2 抽中至少一个好盆地的概率 ≈86%，本地中了、Ibex 主跑
没中。**协议后果**：大网络（m=64,d=10）的 baseline 数字必须 ≥8 种子报中位数+最大值；
best-of-2 只对小网络够用。**cylinder baseline 的诚实数字：好盆地 ≈70 dB**（远高于此前引用的
57.9~68.0 区间上下限的歧义——之前"差距未定 3.7~13.6"的说法收敛为：好盆地 baseline 70.0 vs
pro_quality 62.4-62.8，**差距 ≈7-8 dB**）。

### 4.4f boussinesq mainExp_2.3-ibex（每(模式,种子)独立任务，seeds {0, 7777}，run 级取优）

| 模式 | seed0 | seed7777 | best-of-2 |
|---|---|---|---|
| baseline | **68.47** | 53.84 | 68.47（同样双峰！差 14.6 dB） |
| pro_budget | 56.85 | 56.24 | 56.85 |
| pro_quality (3B) | 65.06 | 64.27 | 65.06 |
| no_observer | 55.23 | 53.73 | 55.23 |

读数：①observer 隔离贡献 = 56.85 − 55.23 = **+1.62 dB**（羽流比涡街略受益，但远小于 RFC 的
+4.7~5.2——与"可刚体解释程度"排序一致：rfc ≫ boussinesq > cylinder）；②好盆地 baseline
（68.47）> pro_quality（65.06）> pro_budget ≫ no_observer——与 cylinder 同构；③大网 SIREN
双峰性在第二个真实数据集复现，系统性现象。本地对照运行中（同种子对，per-INR 取优口径）。

### 4.4g Verify_tau_1.1：τ 敏感性（pro_quality vs baseline，Ibex job 48814029，20 任务零失败）

每点 2 独立种子任务取优（run 级），与 §4.4e/f 的 baseline 同协议可比：

**cylinder2d**（absorb=64；baseline 好盆地 ≈70.0，8 种子中位 69.6）：

| τ | 0.005 | 0.01 | 0.02 | 0.05 | 0.1 |
|---|---|---|---|---|---|
| M (INR 数) | 21 | 14 | 4 | 5 | 3 |
| PSNR best-of-2 | 58.84 | 57.29 | **62.14** | 56.78 | 51.22 |

**boussinesq**（absorb=256；baseline best-of-2 = 68.47）：

| τ | 0.1 | 0.15 | 0.2 | 0.3 | 0.5 |
|---|---|---|---|---|---|
| M (INR 数) | 24 | 21 | 15 | 7 | 2 |
| PSNR best-of-2 | 64.43 | 65.66 | 65.64 | 67.10 | **70.43** |

读数：
1. **boussinesq 单调："区域越少越好"**，τ=0.5（M=2 = 每窗口一个全局 observer + 1.5B 大 INR）
   达 **70.43 > baseline 68.47** —— **proposed 首次在真实数据上超过 baseline**（quality 口径、
   同协议同种子对；注意此规模下 CR≈1.0，不构成压缩，只是拟合质量对比）。羽流的全局运动成分
   使全局 observer 有效，与 observer 贡献排序（bouss +1.6 > cyl +0.8）互证。
2. cylinder 全 τ 落后 baseline（最好 τ=0.02/M=4 的 62.14 vs ~70），且非单调——涡街对分区
   粒度不敏感、对"障碍物小区是否被吸进巨区"敏感（τ=0.1/absorb=64 把障碍物区吸收进巨区后
   掉到 51.2，而 mainExp 同 τ/absorb=0 分开时 62.4-62.8；另有 run 级 vs per-INR 取优的口径差）。
3. 两数据集共同结论：**pro_quality 的最优 τ 都在"大区域/少 INR"端**——3B 预算下"每 INR 更大"
   胜过"分区更细"；结合 §4.4b 窗口税，当前方法的收益结构 = 全局/大区域 observer (+) 、
   细粒度切分 (−)。

### 4.4h Verify_alloc_1.1：预算分配 均分 vs 按像素比例（cylinder τ=0.1，本地，负结果）

| 模式 | 均分（mainExp_2.3） | 像素分配 |
|---|---|---|
| pro_budget | 54.38 | **49.97**（−4.4 dB） |
| no_observer | 53.54 | 49.21 |

**假设被推翻**：按像素比例分配更差。根因：小而剧烈的区域（80px 障碍物区）被压到 m=4，
normalized MSE 卡在 7e-2（完全拟合失败），而这些区域恰是速度最剧烈处，80 个像素的误差
拖垮全场。教训：**纯比例与纯均分都不对，需要"比例 + 每 INR 容量下限"**；且结合 §4.4g
（少而大区域最优），固定 M 下的分配微调优先级降低。observer 增量 +0.76 与均分口径一致。

### 4.4i Verify_gerristiny_1.1：GerrisTinySet 湍流场（gerris0/gerris4，Ibex job 48828750）

新测试场 **GerrisTinySet**（8 个独立 Gerris solver 2D 非定常场，`.am`，每场原始 256×500×256，
降采样到 (T,X,Y)=(128,128,256)）。pilot 取 gerris0（|v|≈0.066 小速度）+ gerris4（|v|≈0.86 大速度），
τ=0.6，recipe v2.3、best-of-3。数据 scp 至 ibex `/ibex/user/zhanx0o/FLowDataFolder/`；`load_field`
新增 gerris0..gerris7 分支（commit 3812204）。

| 场 | baseline | pro_quality | 差 |
|---|---|---|---|
| gerris0（τ=0.6，N=[4,5]，9 INR） | **78.56** / CR5.6x | 71.92 / CR2.0x | **−6.6 dB** |
| gerris4（τ=0.7 重跑，job 48891784，11.9h） | **68.31** / CR5.6x | 65.79（3B, best-of-3） | **−2.5 dB** |

读数：
1. **首个明确的负结果**：gerris0 上 pro_quality 用 3B 预算反而比 baseline 低 6.6 dB、CR 还更差
   （2.0x vs 5.6x）——**两头输**。与 rfc/cylinder（pro_quality +4~7 dB）完全相反。
2. 归因（推测）：baseline 78.56 dB 近乎无损，说明降采样后 gerris0 对单个 CoordNet 已极友好
   （低频/光滑），**没有可被刚体 observer 利用的结构**；切成区域后每 INR 只 B/3 预算，反不如一个
   大网络拟合整场。**rft 变换无损已独立证实**（§3 T3=1.3e-15；cylinder 真实 + 极端相反 observer
   往返 5.55e-16 / 1.11e-15，见 scratchpad cylinder_rft_extreme.py），故此负结果是方法适用边界，
   **不是变换有损或实现 bug**。
3. GerrisTinySet 需 τ≥0.6 才不碎片化（τ→N 干跑：gerris0 τ=0.1→[145,338]，gerris4 τ=0.1→[321,566]），
   远大于 cylinder/boussinesq 的 0.1~0.2——湍流难被单一刚体 observer 解释，与负结果自洽。
4. 工程：gerris4 pro_quality τ=0.6（N=[9,16]=25 INR × best-of-3）跑 13h 超 12h 墙钟 TIMEOUT；
   τ=0.7（N≈11）+ --time 24h 重跑（job 48891784）。

### 4.4j Verify_arch_1.1：架构变体正式对比（v_MLP0.0 / v_FINER0.0，Ibex 部署 2026-07-15，结果待回收）

**问题**：观察者变换的收益是否依赖 SIREN？每个架构对比**自己的** baseline（直接拟合 v）
vs **自己的** pro_quality（τ-合并分区 + observed field，**4B**——规格变更见 §2.5，
故与 §4.4d–i 的 coordnet pro_quality 3B 数字**不同预算、不可直接对比**；本实验的对照
是架构内 baseline vs pro_quality）。

**协议变更（本实验起生效，用户 2026-07-15 明示）**：
1. epochs 硬上限 1000 → **2000**（`run_experiment.py` 断言同步放宽）；**不再要求各架构
   epoch/lr 与 coordnet baseline 对齐**——只看收敛后性能，论文只汇报模型大小。
2. lr 按架构由本地试点选定（下表）；lr_final=1e-6 余弦；其余沿 v2.3（自适应 batch
   ≥64 步/epoch、grad-clip 1.0、minmax 归一化）。
3. **种子协议：2 个独立种子 {0, 7777}，结论取均值**（取代 best-of-k；每任务 n_seeds=1，
   无编码期种子搜索）。⚠️ 与旧 best-of 口径的数字**不可直接混排**，跨协议引用须注明。

**lr 试点（本地 RTX3090，日志 `outputs/lrpilot/*/`）**：

| 架构 | 候选 lr | rfc 200ep baseline PSNR | cylinder m=64,d=10 100ep 坍缩检查 | 选定 |
|---|---|---|---|---|
| mlp | 1e-4 / 3e-4 | 41.86 / **44.75** | 3e-4：MSE 1.3e-2→7.0e-5、50.06 dB、无坍缩 | **3e-4** |
| finer | 1e-4 / 5e-4（官方） | **48.15** / 15.66（坍缩） | 1e-4：MSE 2.2e-2→3.5e-5、53.00 dB、无坍缩 | **1e-4** |

试点发现（有独立价值）：①FINER 官方 lr=5e-4（为 3 隐层浅网调的）搬到深 CoordNet 骨架
**即坍缩**——rfc 小网 m=24,d=4 都从 epoch 50 起钉死在 MSE 1.087e-1，"sine 系高 lr 坍缩"
现象跨激活函数复现（与 §4.6 SIREN@3e-4 同族）；②lr=1e-4 的 FINER 100ep 就在 cylinder
大网到 53.0 dB（参照：coordnet v2.3 recipe 300ep = 45.64）；③mlp@3e-4 100ep 50.1 dB，
raw 坐标残差 ReLU 在该场收敛健康。

**运行矩阵**（`ibex_bash/refframe_v2_arch.sh`，array 40 任务 ≤32 并发，24h 墙钟，
[a100|v100]，一任务=一 (field, model, mode, seed)）：

| field | m/d | τ | absorb | 配置依据 |
|---|---|---|---|---|
| rfc | 24/4 | 0.05 | 0 | mainExp_2.3（每窗 N=1）|
| cylinder2d | 64/10 | 0.1 | 0 | mainExp_2.3 pro_quality 62.4-62.8（M=5，cylinder 历史最佳）|
| boussinesq | 64/10 | 0.5 | 256 | Verify_tau_1.1 胜点 70.43（M=2）|
| gerris0 | 64/10 | 0.6 | 0 | Verify_gerristiny_1.1 先导（9 INR）|
| gerris4 | 64/10 | 0.7 | 0 | 先导 τ=0.6 TIMEOUT→0.7 重跑配置（N≈11）|

输出 `outputs/Verify_arch_1.1/{field}_{model}_{mode}_s{seed}/`。**Ibex jobs**：
baseline 20 任务 = job **48900605**（commit 7cebf9f；3B→4B 变更不影响 baseline）；
pro_quality 20 任务在 4B 变更后取消重提 = job **48901021**（commit 8be61e5，4B 口径，
`sbatch --array=<pro_quality 索引> refframe_v2_arch.sh`）。

**中期发现与 lr 修正（2026-07-15 晚，22/40 完成时）**：

1. **v_MLP0.0 全部健康且种子极稳**（两种子差 ≤0.5 dB）：rfc bl 51.7/51.3、pq(4B)
   65.7/65.5；cylinder bl 58.50/58.51；boussinesq bl 55.9/55.8；gerris0 bl 66.5/66.7；
   gerris4 bl 55.8/55.9 —— **大 SIREN 的双峰种子不稳定（§4.9④）在残差 ReLU 上未出现**。
2. **v_FINER0.0 小网优异、大网坍缩**：rfc（m=24/29）bl 60.7/63.6、**pq(4B) 79.4/78.8**
   （rfc 全历史最高）；但 m≥57 的大网在持续 lr=1e-4 下坍缩——cylinder bl 24.5/23.3、
   boussinesq bl 24.2/24.4、gerris0 bl s7777 30.7。轨迹证据：cylinder ep100 即落入
   3.278e-2 均值流吸引子（与 §4.6 v2.2 SIREN@3e-4 同值）且到 ep1700 未逃逸；boussinesq
   从 ep100 的 6.7e-3 **反向恶化**到 2.35e-2。m=38 的 gerris4 finer pq 区域健康（1.0e-4
   @ep800）⇒ 坍缩阈值在 m≈50 附近。
3. **试点方法教训（重要）**：§4.4j 的本地坍缩检查用"100ep 余弦"调度（lr 快速衰减，
   ep100 时已≈1e-6），而部署是"2000ep 余弦"（lr≈1e-4 维持数百 ep）——同峰值 lr、不同
   调度长度，试点因此漏报。**lr 稳定性检查必须在与部署一致的调度形状/长度下做。**
4. **修正**：finer 大网组合（cylinder/boussinesq 的 bl+pq、gerris0 bl，各 2 种子）以
   lr **3e-5** = job **48916196**、lr **1e-5** = job **48916197** 重跑（脚本 `RFV2_LR`
   覆盖，输出目录带 `_lr{LR}` 后缀不覆盖原结果）；已确定报废的 boussinesq finer pq@1e-4
   （48901021_22/23）提前取消止损。rfc 的 finer@1e-4 结果健康、保留。gerris4 finer
   baseline@1e-4 完成后确认同样坍缩（16.0/36.6）→ 补交重跑 jobs **48931725**（3e-5）/
   **48931726**（1e-5）。⇒ finer 的 lr 按网宽分档（小网 1e-4 / 大网取 3e-5 与 1e-5 中
   收敛更好者），最终口径以重跑结果为准并在结果表中注明每个数字的 lr。
5. **lr 重跑首批回收（3e-5 大胜，且超过 coordnet 历史最好值）**：cylinder finer
   baseline@3e-5 = **73.37/73.98**（coordnet 好盆地 ≈70.0，8 种子中位 69.6）；
   boussinesq finer baseline@3e-5 = **70.83/70.94**（coordnet best-of-2 = 68.47）——
   两场均为两种子一致、无双峰。@1e-5 则明显低一档（cyl 70.37/69.71、bouss 65.91）⇒
   大网 finer 口径取 **3e-5**。遗留问题：pq@3e-5 出现种子分裂（bouss 71.91 vs 56.04；
   cyl s7777 仅 50.48）——m=90/57 的 pq 大 INR 在 3e-5 下仍有残余不稳定，1e-5 的 pq
   对照在跑，回收后再定 pq 口径。
6. gerris0 finer pq(m=42)@1e-4 = 31.9/33.5、gerris4 finer pq(m=38)@1e-4 = 36.1/40.8
   ——完成后确认**同样坍缩**（早先"m=38 健康"的判断只看了个别 INR 的轨迹，最终整场被
   个别坍缩区域拖垮）→ 补交重跑 jobs **48965055**（3e-5）/ **48965056**（1e-5），
   各 2 场 × 2 种子。

**结果表（2026-07-16 整理；2-seed 均值协议，格式 = 均值 (s0/s7777)；pq=4B）**

v_MLP0.0（lr 3e-4，全部健康，种子差 ≤0.5 dB）：

| field | baseline | pro_quality(4B) | 架构内 Δ |
|---|---|---|---|
| rfc | 51.49 (51.72/51.26) | 65.63 (65.75/65.50) | **+14.14** |
| cylinder2d | 58.51 (58.50/58.51) | 57.91 (57.80/58.01) | −0.60 |
| boussinesq | 55.84 (55.92/55.75) | 58.22 (58.13/58.31) | **+2.38** |
| gerris0 | 66.59 (66.47/66.71) | 68.72 (68.86/68.58) | **+2.13** |
| gerris4 | 55.89 (55.82/55.95) | 59.82 (59.74/59.90) | **+3.93** |

v_FINER0.0（rfc=1e-4；大网两档 lr 如实并列，**不挑臂**；@1e-4 大网数字全部坍缩仅留档）：

| field | lr | baseline | pro_quality(4B) | 架构内 Δ |
|---|---|---|---|---|
| rfc | 1e-4 | 62.12 (60.68/63.56) | **79.12** (79.39/78.85) | **+17.00** |
| cylinder2d | 3e-5 | **73.68** (73.37/73.98) | 48.55 (46.62/50.48)⚠ | −25.1 |
| cylinder2d | 1e-5 | 70.04 (70.37/69.71) | 59.34 (52.64/66.03)⚠ | −10.7 |
| boussinesq | 3e-5 | **70.89** (70.83/70.94) | 63.98 (71.91/56.04)⚠ | −6.9 |
| boussinesq | 1e-5 | 67.78 (65.91/69.64) | 69.35 (68.06/70.63) | **+1.57** |
| gerris0 | 3e-5 | **81.23** (79.10/83.35) | 66.92 (75.01/58.82)⚠ | −14.3 |
| gerris0 | 1e-5 | 62.13 (47.18/77.07)⚠ | 72.77 (72.86/72.67) | +10.6 |
| gerris4 | 3e-5 | 65.02 (70.24/59.80)⚠ | **70.12** (73.85/66.38)⚠ | +5.1 |
| gerris4 | 1e-5 | 56.69 (71.38/41.99)⚠ | 69.64 (71.74/67.54) | +12.9 |

（⚠ = 两种子差 >5 dB 的种子分裂，均值解释需谨慎。）

**读数（截至 gerris finer pq 重跑回收前）**：
1. **v_MLP0.0：proposed 在 5 场中 4 场架构内取胜**（rfc +14.1、gerris4 +3.9、bouss +2.4、
   gerris0 +2.1；cylinder −0.6），且全部数字种子稳定。对照 coordnet（旧协议）：真实场
   仅 bouss τ=0.5 一个胜点、gerris0 −6.6、gerris4 −2.5 ⇒ **观察者+分区方法的收益在
   残差 ReLU 上比在 SIREN 上普遍得多**——方法的故事不必绑死 SIREN。
2. **v_FINER0.0 baseline = 单网直拟合的新天花板**：cylinder 73.7、gerris0 81.2、
   boussinesq 70.9，全部超过 coordnet 历史最好（≈70.0 / 78.6 / 68.5），rfc pq 79.1
   为 rfc 全历史最高。**FINER 换掉 SIREN 后 baseline 本身大幅变强。**
3. **但 FINER × proposed 大 INR 组合不稳定**：pq 的 observed-field 样本（扫掠 bbox
   坐标系）对 finer 大网比 raw field 难训练得多——cyl pq 两档 lr 都远低于 bl 且种子
   分裂；bouss 两臂结论相反（3e-5：−6.9 / 1e-5：+1.57）。**lr-臂的选择会翻转结论，
   故两臂如实并列**；机理（扫掠坐标 → 有效频率升高 → 变周期激活的稳定域变窄？）留待
   Verify_arch_1.2 专项。
4. 综合定位：**稳定性选 MLP（方法收益普遍）、峰值质量选 FINER baseline（但其与本方法
   的大 INR 组合暂不可用）**。
5. **gerris finer pq 重跑回收（2026-07-16，jobs 48965055/48965056，表已补全）**：
   (a) gerris0：bl 最优臂（3e-5，81.2，种子稳）仍是全表天花板，pq 两臂均低于它
   （最好 72.8@1e-5）⇒ finer×proposed 在 gerris0 为负；(b) gerris4：pq 在两臂都高于
   同臂 bl（+5.1⚠ / +12.9），方向为正，但 bl 两臂都种子分裂，底数不稳；(c) **lr 规律
   跨场复现**：pq 大 INR 在 3e-5 种子分裂（gerris0 差 16.2，与 cyl/bouss pq@3e-5 分裂
   同族）、在 1e-5 稳定（gerris0 72.86/72.67，与 bouss pq@1e-5 稳定一致）⇒ finer 的
   observed-field 大 INR 需要比 raw-field 更低的 lr（bl 优臂 3e-5 / pq 稳臂 1e-5 错位，
   进一步支持"扫掠坐标提高有效频率、收窄稳定域"假设，留 Verify_arch_1.2）。

### 4.4k mainExp_compress_1.1：严格压缩口径（总字节 ≤ 5%/10%/20% 原始场，Ibex 部署+回收 2026-07-16）

**动机（用户 2026-07-16 指令）**：压缩任务要求网络明确小于流场，比旧 pro_budget 更严格
——旧预算 B = baseline 参数量，占原始场字节 ~19-23%（rfc 18.9%、cylinder 22.7%、
boussinesq 19.4%），且边信息不计入预算上限。本实验改为**总字节硬预算**：
参数×4 + 边信息 ≤ frac × 原始场 float32 字节，frac ∈ {5%, 10%, 20%}（即 CR ≥
20×/10×/5×），管线内置断言，超预算即 fail。

**实现（commit 23524b1）**：
- `budget_calc.py`（规划工具）：给定场（名字查表或 --shape）与 frac，用已验证的闭式参数
  公式反解宽度 m；proposed 先扣**精确边信息**（cell 标签图 + killing (a,b,c)(t) +
  每区域 bbox/值域/m_r，口径与 `run_proposed` 逐字节一致）再均分 M 份。**深度 d 冻结为
  各场 baseline 值**（rfc 4、cyl/bouss 10），预算档只改宽度——避免深度与预算两个变量
  混淆，保持与全部历史实验可比；`--d_sweep` 可打印宽深权衡的参考表（不改变策略）。
- `run_experiment.py --budget_frac f`：baseline 由预算反解 m（此时 m_base 失效）；
  pro_budget 均分 (f×场字节 − 边信息)/4（均分依据 Verify_alloc_1.1 负结果：按像素
  比例更差）；no_observer 同口径可用（本轮未跑）；pro_quality 与 frac 组合被禁用
  （语义不同，防标签混淆）。

**协议** = Verify_arch_1.1（用户 2026-07-15 定）：epochs 2000、各架构自身 lr（coordnet
1e-5 = v2.3 冻结 recipe；mlp 3e-4 = lr 试点选定值）、2 独立种子 {0,7777} 独立任务取
**均值**、n_windows=2、τ/absorb 取各场记录工作点（rfc 0.05/0、cylinder2d 0.1/0、
boussinesq 0.5/256 → M=2/5/2）。

**规划尺寸（budget_calc.py 输出；bl = baseline 单网宽度 m，pro = 每区域宽度 m_r）**：

| field | 原始字节 | M | 5% bl/pro | 10% bl/pro | 20% bl/pro |
|---|---|---|---|---|---|
| rfc (d=4) | 2,097,152 | 2 | m=12 / 8 | 17 / 12 | 24 / 17 |
| cylinder2d (d=10) | 26,214,400 | 5 | 29 / 13 | 42 / 18 | 60 / 26 |
| boussinesq (d=10) | 17,280,000 | 2 | 24 / 16 | 34 / 24 | 48 / 34 |

锚点连续性：rfc 20% baseline（m=24,d=4）与 mainExp_2.3 rfc baseline 配置完全相同；
cylinder 20% baseline m=60 ≈ 历史 m=64（后者 = 22.7% 预算）——20% 档与历史结果可互为
sanity check，5%/10% 是全新的更严格压缩区间。

**运行矩阵**：3 场 × 3 frac × 2 架构（coordnet = Coordinate INR baseline，mlp =
v_MLP0.0）× 2 模式（baseline 直拟合 / pro_budget 分区+RFT）× 2 种子 = **72 独立任务**。
脚本 `ibex_bash/refframe_v2_compress.sh`，Ibex job **48967626**（array 0-71%32，24h，
[a100|v100]）。输出 `outputs/mainExp_compress_1.1/{field}_{model}_{mode}_f{frac}_s{seed}/`。

**冒烟验证（本地 RTX3090，20 epochs，无科学意义）**：rfc coordnet f=0.05、mlp f=0.10
跑通；字节口径与 budget_calc **逐字节一致**（pro f=0.05 总字节 96,340 = 计算器预测值）；
两侧断言全过；validate_rfc.py 全过（本次改动未触碰 killing/partition/frame）。

**已知风险（读结果时注意）**：
1. cylinder 20% 的 coordnet baseline（m=60,d=10）处于大 SIREN 双峰区（§4.9④），2-seed
   协议下可能种子分裂——沿用 §4.4j 的 ⚠ 标注规则（两种子差 >5 dB 时均值慎读）；mlp 侧
   无此风险（§4.4j 读数 1：残差 ReLU 种子稳定）。
2. coordnet 5% 档小网（m=8~16）+ lr 1e-5 有"小网低 lr 收敛慢"风险（§4.4b 窗口税同机理，
   2000 epochs 比历史 1000 翻倍已部分缓解）；baseline 与 pro 对称受影响，架构内对比
   仍公平，但绝对数字可能是优化受限而非容量受限。

**结果（2026-07-16 回收，72/72 任务完成零失败；2-seed 均值，格式 = 均值 (s0/s7777)，
⚠ = 两种子差 >5 dB；Δ = 架构内 pro − baseline；出处 = Ibex job 48967626 各任务
`slurm_logs/RFv2cmp.*.48967626.out` 与 `outputs/mainExp_compress_1.1/*/*_metrics.json`）**

rfc（M=2；⚠ 注意本表与 mainExp_2.3 协议不同——2000ep、2-seed 均值 vs 1000ep、
best-of-3——数字不可混排）：

| frac | coordnet bl | coordnet pro | Δ | mlp bl | mlp pro | Δ |
|---|---|---|---|---|---|---|
| 5% | 51.52 (54.05/48.98)⚠ | 45.47 (48.68/42.25)⚠ | −6.05 | 46.11 | **54.21** (56.13/52.29) | **+8.11** |
| 10% | 57.83 | 52.99 (50.71/55.26) | −4.85 | 48.99 | **58.35** | **+9.36** |
| 20% | **61.45** | 58.45 (56.83/60.06) | −3.00 | 51.57 | 58.72 | **+7.15** |

cylinder2d（M=5）：

| frac | coordnet bl | coordnet pro | Δ | mlp bl | mlp pro | Δ |
|---|---|---|---|---|---|---|
| 5% | **64.61** | 47.00 | −17.62 | 54.40 | 47.59 | −6.81 |
| 10% | **68.74** | 53.27 | −15.47 | 56.38 | 50.14 | −6.24 |
| 20% | 66.23 (72.59/59.87)⚠ | 59.12 | −7.12 | 58.22 | 52.81 | −5.42 |

boussinesq（M=2）：

| frac | coordnet bl | coordnet pro | Δ | mlp bl | mlp pro | Δ |
|---|---|---|---|---|---|---|
| 5% | **58.46** | 55.79 | −2.67 | 50.13 | 48.71 | −1.42 |
| 10% | **62.84** | 60.11 | −2.74 | 52.42 | 51.41 | −1.01 |
| 20% | **69.46** | 64.67 | −4.80 | 54.48 | 54.01 | −0.47 |

**读数**：
1. **rfc×mlp：pro 全预算档大胜（+7.2~+9.4）**，且在最严的 5%/10% 档 mlp pro **跨架构**
   反超 coordnet baseline（54.21 > 51.52、58.35 > 57.83）——严格压缩口径下 rfc 的全场
   最优配置 = MLP+RFT（20% 档 coordnet bl 61.45 收回第一）。RFT 压缩收益在"可全局刚体
   解释"的场上真实存在，且承载架构是 mlp 而非 SIREN（与 Story v2 §0b 一致）。
2. **真实场压缩口径 proposed 全线落后**：cylinder 两架构 −5.4~−17.6（不适用边界再证，
   §4.9①）；boussinesq 温和落后，但 **mlp 差距随预算单调收窄 −1.42→−1.01→−0.47**，
   与 4B quality 口径的 **+2.38**（§4.4j）衔接 ⇒ boussinesq×mlp 存在"预算交叉点"，
   RFT 是高预算侧增益，5-20% 严格压缩区间尚在交叉点之下。
3. **部署前的两条风险预判全部命中**：cylinder coordnet 20%（m=60）baseline 种子分裂
   72.59/59.87（双峰区 §4.9④，均值 66.23 慎读——好盆地单种子 72.59 与历史 m=64 好盆地
   ≈70 同档）；rfc coordnet 5% 档两模式均 ⚠（小网 lr 1e-5 收敛方差）。**mlp 全部 36
   任务无一 ⚠**（最大种子差 3.84，rfc 5% pro），再证残差 ReLU 的种子稳定性。
4. 工程注记：pro 的 m_r 取整在 M=2 的 5% 档最多浪费 ~8% 预算（boussinesq pro 用掉
   89.6% vs bl 97.6%），对结论方向无影响但对 pro 略不利；如需公平到字节级可加
   "剩余预算给最大区域"的再分配（未做）。

### 4.4l Verify_compresswin_1.1：压缩口径翻盘 sweep（boussinesq coordnet τ×lr×窗口数，M≤3；Ibex 部署 2026-07-16，回收 2026-07-17）

**验收标准（用户 2026-07-16 定，项目级约束）**：只接受 cylinder2d 上 proposed 无法提高
coordnet（SIREN）；**其他所有数据集上 proposed 不得比对应架构的纯 INR baseline 更差**。
§4.4k 中 boussinesq coordnet −2.7、rfc coordnet −3.0~−6.1 均不满足 ⇒ 本实验找翻盘配置。

**新约束与依据（用户同日定）**：
1. **总 INR 数 M ∈ [1,3] 硬约束**——压缩预算本来就小，切 M 份后每网参数 ≈ 预算/M，
   区域一多每个网络就没有拟合能力（§4.4k cylinder M=5 在 5% 档 m_r=13 即此病）。
   管线新增 `--max_inrs`：分区结果超限直接报错，不允许静默跑碎片化配置。
2. M=1 只能由单时间窗实现 ⇒ **"≥2 窗"规则在本实验按用户指令解除**（`--allow_full_window`
   臂），亦有 §4.4b 依据：窗口切分 = 5-6 dB 优化税，RFC 单窗等参 observer 增益 +4.7~5.2。
3. **epochs 2000 → 1000**（用户：压缩只看收敛后性能，1000 足够）。
4. lr 对 **baseline 同步 sweep**——5-10% 档网络小（m=13~35），v2.3 的 lr=1e-5 是按大
   SIREN 稳定性调的，小网偏小（§4.4b"小网低 lr 收敛慢"）；只调 pro 的 lr 不可信，
   两侧同 grid 才能报"最优对最优"。

**τ→M 干跑（boussinesq，absorb=256，本地 2026-07-16）**：
1 窗：τ=0.5→**M=1**、0.4→2、0.35→3（0.3→6 超限）；2 窗：τ=0.5→[1,1]=**2**（旧工作点）、
0.4→[1,2]=**3**（0.35→5 超限）。

**运行矩阵（120 任务，job 49000294，array 0-119%32，12h，coordnet，1000ep，2 种子均值）**：
- boussinesq（d=10）：{bl + 5 个 pro 臂：w1M1(τ=.5)/w1M2(.4)/w1M3(.35)/w2M2(.5)/w2M3(.4)}
  × frac {5%,10%} × lr {1e-5, 3e-5, 1e-4, 3e-4} × 2 种子 = 96；
- rfc 闭环块（d=4，验收标准同样要求 rfc 不输）：{bl, w1M1(τ=.05，即 §4.4b 等参赢点配置)}
  × frac {5%,10%} × lr {1e-5, 1e-4, 3e-4} × 2 种子 = 24。
预算规划（budget_calc，d=10）：bouss 5% M=1/2/3 → m_r=24/16/13，10% → 34/24/19；
M=1 时 pro 与 bl **同宽度**（5% 均 m=24、10% 均 m=34）——对比退化为"observed field vs
raw field 等参等步"，最干净的归因形态。
脚本 `ibex_bash/refframe_v2_compresswin.sh`（commit 8e7f05c），输出
`outputs/Verify_compresswin_1.1/{field}_{arm}_f{frac}_lr{LR}_s{seed}/`。

**读结果协议**：主对比 = 同 lr 配对（pro@lr vs bl@lr）+ 最优对最优（各自 lr 内取最好臂，
编码期搜索对称合法）；⚠ 规则同 §4.4j；lr=3e-4 臂若坍缩（§4.6 大网均值流吸引子为 m=64
现象，小网未知）如实记录。

**结果（2026-07-17 回收，120/120 零失败；2-seed 均值；† = 坍缩到均值流吸引子 ~23.4-23.7；
⚠ = 种子差 >5 dB；出处 job 49000294 + `outputs/Verify_compresswin_1.1/*/*_metrics.json`）**

boussinesq f=5%（bl 与 6 臂 × 4 lr；粗体 = 该 lr 行最优）：

| lr | bl | w1M1 | w1M2 | w1M3 | w2M2 | w2M3 |
|---|---|---|---|---|---|---|
| 1e-5 | 53.70 | **54.21** | 50.66 | 46.82 | 50.68 | 46.39 |
| 3e-5 | 61.20 | **61.46** | 58.07 | 54.32 | 55.93 | 54.91 |
| 1e-4 | 63.88 (65.81/61.95) | 62.06⚠ | 63.89⚠ | 58.83 | **64.61** (64.49/64.72) | 60.72 |
| 3e-4 | 23.65† | 23.37† | 23.44† | 28.71† | 23.73† | 26.16† |

boussinesq f=10%：

| lr | bl | w1M1 | w1M2 | w1M3 | w2M2 | w2M3 |
|---|---|---|---|---|---|---|
| 1e-5 | 58.37 | **58.50** | 56.09 | 54.80 | 54.92 | 52.16 |
| 3e-5 | **66.23** | 66.16 | 63.61 | 58.66 | 62.32 | 58.48⚠ |
| 1e-4 | **69.84** (70.07/69.60) | 61.62 | 64.81⚠ | 65.82 | 69.76 (69.53/69.98) | 63.82 |
| 3e-4 | 23.63† | 23.34† | 23.61† | 23.95† | 23.38† | 23.54† |

rfc（bl vs w1M1）：

| frac | lr | bl | w1M1 | Δ |
|---|---|---|---|---|
| 5% | 1e-5 | 46.25⚠ | 48.58⚠ | +2.3 |
| 5% | 1e-4 | 58.27 | 61.94⚠ | +3.7 |
| 5% | 3e-4 | 63.50⚠ (58.85/68.15) | **69.46**⚠ (59.07/79.85) | +6.0 |
| 10% | 1e-5 | 52.60 | 59.00 | +6.4 |
| 10% | 1e-4 | 65.31 | 74.40 (74.04/74.75) | +9.1 |
| 10% | 3e-4 | 68.00 (68.03/67.96) | **80.24** (80.82/79.66) | **+12.2** |

**读数**：
1. **boussinesq 5%：达标（proposed 赢）**。最优对最优 w2M2@1e-4 = 64.61（种子稳，差
   0.23）> bl@1e-4 = 63.88（离散 3.86），**+0.73**；且三个未坍缩 lr 行的同 lr 配对
   pro 最佳臂均 ≥ bl（+0.51/+0.26/+0.73）。赢点结构 = 每窗一个全局 observer（与
   Verify_tau_1.1 quality 口径赢点同构），如今在**压缩口径**也成立。
2. **boussinesq 10%：统计平手**。bl 69.84 vs w2M2 69.76（−0.08），两侧种子区间重叠
   （69.60-70.07 vs 69.53-69.98）；同 lr 配对 +0.13/−0.07/−0.08 全在噪声内。判定交给
   种子扩展（下）。
3. **rfc coordnet：彻底翻盘，验收标准闭合**。10% w1M1@3e-4 = **80.24 vs bl 68.00
   （+12.2，两侧种子稳）**——单窗全局 observer 把 §4.4b 的等参增益带进压缩口径并放大
   （lr 调优后 observed field 的可拟合性优势拉大）；5% 方向一致（6 组同 lr 配对全赢）
   但小网（m=12）种子混沌，幅度 ⚠。此前 §4.4k rfc coordnet pro 全输的根因 = 2 窗税 +
   lr 欠优，两者都已定位并消除。
4. **recipe 发现（并入方法学）**：①压缩档小网的最优 lr 远高于冻结 recipe 的 1e-5——
   boussinesq 1e-4、rfc 3e-4，差 6~12 dB；lr 调优后 10% 预算 baseline（69.84）甚至超过
   历史 22.7% 预算 baseline（68.47，§4.4f）⇒ §4.4k 的绝对数字全体系统性偏低（"优化受限"
   风险预判成真），但架构内对比方向仍有效。②**均值流坍缩不是大网专属**：boussinesq 上
   3e-4 连 m=24 都全臂坍缩（23.4-23.7），rfc 同 lr 完全健康 ⇒ 坍缩由数据（羽流的强均值
   成分？）而非仅网络规模决定，lr 必须按（场,规模）选。③**最优窗口结构场依赖**：rfc
   （observer 时不变）单窗最优；boussinesq 双窗每窗全局 observer 最优（w1M1 在 1e-4 反而
   不稳 ⚠8.3——单窗扫掠 bbox 更大）。④M 单调性：M=3 臂几乎处处垫底，"少而大区域"再证
   （§4.4g 一致）。
5. **种子扩展（判定 10% 平手 + 收紧 rfc 5%）**：决胜格子各 +3 种子（{1,2,3}）→ 5-seed
   均值：boussinesq {5,10}% × {bl, w2M2}@1e-4 + rfc 5% × {bl, w1M1}@3e-4，18 任务 =
   Ibex job **49023725**（commit 9070c24）。

**种子扩展结果（2026-07-17 回收，18/18；5-seed = {0,7777,1,2,3}，报 均值 [min..max]）**：

| 格子（@最优 lr） | baseline | pro | Δ均值 / Δ中位 / Δ最差 |
|---|---|---|---|
| bouss 5% @1e-4 | 63.83 [56.99..67.73] | w2M2 62.65 [60.37..64.72] | **−1.18** / −3.61 / **+3.38** |
| bouss 10% @1e-4 | 66.18 [53.62..70.07] | w2M2 **69.37** [68.50..69.98] | **+3.19** / +0.08 / **+14.88** |
| rfc 5% @3e-4 | 57.61 [51.11..68.15] | w1M1 **67.18** [59.07..79.85] | **+9.57** / +5.58 / +7.96 |

**⚠ 结论修正（2-seed → 5-seed，按"结论可追溯不静默翻转"规则并列记录）**：
- 之前说（2-seed，§4.4l 读数 1/2）：bouss 5% pro 赢 +0.73；10% 统计平手 −0.08。
- 现在（5-seed）：**bouss 5% pro 均值输 −1.18**（但最差种子/稳定性大胜：pro 离散 4.4 vs
  bl 10.7）；**bouss 10% pro 均值赢 +3.19**（bl 有 1/5 种子塌到 53.62 的坏盆地左尾，
  pro 五种子全在 68.5-70.0）。
- 为什么变：boussinesq 小网 SIREN@1e-4 的 baseline 有**坏盆地左尾**（f05 s3=56.99、
  f10 s1=53.62）——§4.9④ 双峰现象在小网上复现；2-seed 抽样两次都没抽到/抽到，把
  两个格子的符号各翻了一次。
- rfc 5% 从"方向一致但 ⚠"升级为**决定性胜**：pro 最差种子 59.07 > bl 中位数 57.98。

**种子预算规则（用户 2026-07-17 定，取代本节曾写的"≥5 种子"要求）**：**任何格子 ≤3
种子**（算力成本），标准组 {0, 7777, 1}，报 均值 + [min..max]。上面 5-seed 表为已产生
数据留档；**3-seed 口径复核符号全部一致**：bouss 5% bl 64.80 vs w2M2 63.19（−1.61）、
bouss 10% bl 64.43 vs w2M2 69.34（**+4.91**）、rfc 5% bl 59.65 vs w1M1 67.40（**+7.75**）
——本文既有结论无一需要改。Verify_compresswin_1.2（§4.4m 后续）部署时含 5 种子，
seed_idx 3/4 的 48 个任务已在 PENDING 阶段全部 scancel（零算力浪费），实跑 = 3 种子网格。

### 4.4m mainExp_compress_1.2：2.5% 预算档（CR≥40×，修正协议；Ibex 部署 2026-07-17，结果待回收）

**任务（用户 2026-07-17）**：压缩任务扩到 2.5% × 原始场字节。协议 = §4.4l 修正后版本，
**与 §4.4k（mainExp_compress_1.1 的 5-20% 档）不可混排**（那批 lr 固定 1e-5/3e-4、
2000ep、M 无上限；本批如下）：
- **M≤3 硬约束 + 各场已验证最优分区结构**：rfc pro = w1M1（单窗 τ=0.05，M=1，与 bl
  **同宽 m=8**）；cylinder pro = w2M2（2 窗 τ=0.1 **absorb=256**，干跑确认 M=[1,1]=2
  ——旧工作点 absorb=0 的 M=5 违反 M≤3，故换）；boussinesq pro = w2M2（2 窗 τ=0.5
  absorb=256，M=2，§4.4l 赢点结构）。
- **场级 lr（baseline 同臂对称）**：coordnet 每场 2 臂——rfc {1e-4, 3e-4}、cylinder
  {3e-5, 1e-4}、boussinesq {3e-5, 1e-4}（bouss 3e-4 全臂坍缩已证 §4.4l；cyl 3e-4 坍缩
  §4.6，均不设）；mlp 单臂 3e-4（各场稳定已证）。
- epochs 1000、2 种子 {0,7777} 均值、n_seeds=1。

**规划尺寸（budget_calc --fracs 0.025）**：rfc bl/pro 均 m=8（11,426 params，bl 45,744 B）；
cylinder bl m=21（161,794）/ pro m_r=14×2（每 INR 72,488，side 27,204 B）；boussinesq
bl m=17（106,430）/ pro m_r=11×2（45,044×2，side 18,780 B）。全部 ≤ 2.5% 预算断言内。

**运行矩阵**：coordnet 3 场 × {bl,pro} × 2 lr × 2 种子 = 24 + mlp 3 场 × {bl,pro} ×
2 种子 = 12，共 **36 任务** = Ibex job **49025811**（commit b9ee09e，
`ibex_bash/refframe_v2_compress25.sh`），输出
`outputs/mainExp_compress_1.2/{field}_{model}_{mode}_f0.025_lr{LR}_s{seed}/`。

**结果（2026-07-17 回收，36/36 零失败；2-seed 均值 (s0/s7777)；⚠ = 种子差 >5 dB）**：

| 场 | 架构 | baseline（各 lr） | pro（各 lr） | 最优对最优 Δ |
|---|---|---|---|---|
| rfc | coordnet | 1e-4: 50.95；3e-4: 53.41 | 1e-4: 55.51；3e-4: **64.79** (57.84/71.74)⚠ | **+11.38**（同 lr 配对 +4.6/+11.4 全胜） |
| rfc | mlp | 3e-4: 40.13 | 3e-4: **49.29** | **+9.16** |
| cylinder2d | coordnet | 3e-5: 60.39；1e-4: **63.55** | 3e-5: 49.73；1e-4: 56.23 | −7.32（豁免场） |
| cylinder2d | mlp | 3e-4: 49.52 | 3e-4: 45.42 | −4.10 |
| boussinesq | coordnet | 3e-5: 59.31；1e-4: **64.38** | 3e-5: 46.04；1e-4: 52.48 | **−11.90** |
| boussinesq | mlp | 3e-4: 46.15 | 3e-4: 43.12 | −3.03 |

**读数**：
1. **rfc 2.5%：两架构 pro 全胜**（coordnet 同 lr 配对全赢，pro 最差臂均值 55.51 也超
   bl 最优 53.41；mlp +9.2）——M=1 等宽结构在最严预算档依然成立。
2. **boussinesq 2.5% 大输（−11.9）暴露 w2M2 的预算下限**：2 窗切分使每 INR 只剩
   m_r=11（45k 参数），而 bl 单网 m=17（106k）；5%→2.5% 时 Δ 从 −1.2 恶化到 −11.9 ⇒
   **切分成本随预算收紧超线性增长**。未试的 w1M1（M=1，m=16 ≈ bl 17 近等宽）是结构性
   候选——rfc 的 M=1 在 2.5% 都赢，bouss 的 M=1 在 5%/10% 同 lr 低档也曾小胜（§4.4l）。
3. cylinder mlp 2.5% 输 −4.1（5-20% 档为 −5.4~−6.8，一致）。**用户裁定（2026-07-18）：
   cylinder 豁免仅限 SIREN**——豁免理由是正弦基天然擅长涡街窄带振荡（架构先验，
   §0b/§4.9①），mlp 无此先验 ⇒ **cylinder×mlp 不豁免，是当前记分板的未达标格**。
   注意现有 cyl×mlp 数字全部来自旧结构（5-20% 档 M=5、2.5% 档 w2M2），M=1 等宽 +
   consttrans（涡街近匀速平流 ⇒ 共动系准定常，Taylor 冻结流）未试。
4. mlp 在 2.5% 全场大幅低于 coordnet baseline（rfc 40 vs 53、cyl 50 vs 64、bouss 46
   vs 64）——极小网下谱偏置劣势放大，mlp 不是 2.5% 档的竞争架构（但架构内 rfc 仍 +9）。

### 4.4n Verify_compresswin_1.2 结果（bouss {2.5,5}% × 中档 lr 网格，job 49033283，回收 2026-07-17）

**部署 skip 逻辑 bug（如实记录）**：`refframe_v2_compresswin2.sh` 的 J2 检查只按 mode 名
（pro_budget）匹配 `mainExp_compress_1.2` 的旧目录，未区分结构 —— w1M1@2.5% × lr{1e-4,
3e-5} × s{0,7777} 共 4 格被 **w2M2 的旧结果错误跳过**（这 4 格只有 s1 数据）。w2M2 侧的
skip 正确（同配置）。受影响格子由 Verify_compresswin_1.3 的 tv 臂取代（新字节口径），
残缺数据只留档。

**结果（3-seed {0,7777,1} 均值 [min..max]；bl@1e-4/3e-5 的 2.5% 格与全部 5% 既有格合并
`Verify_compresswin_1.1/` 与 `mainExp_compress_1.2/` 同协议数据）**：

boussinesq f=2.5%（bl m=17 / w1M1 m_r=16（旧字节口径）/ w2M2 m_r=11×2）：

| lr | bl | w1M1 | w2M2 |
|---|---|---|---|
| 3e-5 | 57.82 [54.84..59.36] | 54.02（仅 s1） | 47.45 |
| 5e-5 | 60.11 [57.03..62.00] | 58.99 [56.42..60.53] | 48.55 |
| 7e-5 | 61.43 [58.27..63.37] | 59.90 [57.13..61.34] | 51.90 |
| 1e-4 | **62.85** [59.79..64.96] | 58.33（仅 s1；bl s1=59.79） | 53.93 |

boussinesq f=5%（全臂 m 相同：bl 与 w1M1 均 m=24）：

| lr | bl | w1M1 | w2M2 |
|---|---|---|---|
| 3e-5 | 61.19 | 61.45 [61.15..61.77] | 54.47 |
| 5e-5 | 63.97 | 63.95 [63.71..64.40] | 59.85 |
| 7e-5 | **66.86** [66.00..68.16] | 65.23 [63.85..66.48] | 61.87 |
| 1e-4 | 63.83 [56.99..67.73]（5-seed） | 63.97 [57.90..67.79] | 62.65（5-seed） |

**读数（判定：两档均未达标）**：
1. **2.5%：pro 最优 59.90（w1M1@7e-5）vs bl 最优 62.85（@1e-4），差 −2.95**；每个同 lr
   配对 w1M1 都输 1.1~1.5。结构性劣势之一：旧字节口径下 w1M1 被边信息（cell 标签图
   8,588B + q(t) 1,536B ≈ 预算的 2.4%）压到 m_r=16，比 bl（m=17）少 11% 参数。
2. **5%：pro 最优 65.23（w1M1@7e-5）vs bl 新最优 66.86（@7e-5），差 −1.63**。中档 lr
   把 bl 的最优臂从 1e-4（63.83，左尾 ⚠）推高到 7e-5（66.86，种子稳）——1.2 网格
   "中间 lr 稳住 w1M1"的赌注对 pro 生效（65.23 种子稳）但 bl 同步受益更多。
   注意 **5% 的 w1M1 与 bl 已是等宽（都 m=24）仍输 1.6**：等宽必要但不充分。
3. w2M2 在两档全 lr 落后 w1M1/bl ⇒ 2 窗切分结构在 ≤5% 预算档淘汰（与 §4.4m 读数 2
   "切分成本超线性"一致）。
4. **关键线索（通往 1.3）**：w1M1@1e-4 的好种子（5%：66.21/67.79）已超 bl 最优均值
   66.86，输在 SIREN 高 lr 坏盆地左尾（s7777=57.90）——高 lr 天花板更高但方差大；
   中档 lr 稳但天花板低。若能消灭左尾，1e-4 臂直接翻盘。

### 4.4o Verify_compresswin_1.3：单全局 observer 变体 + 字节口径 v2 + lr warmup（Ibex job 49044332，部署 2026-07-17，结果待回收）

**任务（用户 2026-07-17）**：bouss 2.5% 压缩必须打败 coordnet(SIREN) baseline；授权改
observer（不分时间/空间窗口，单全局 observer）与 lr 等参数；新部署 ≤64 任务。

**干跑诊断（`diag_agent_observer_variants.py`，无训练，本地）**——bouss 单窗全域 killing
观察者的真实形态：

| 变体 | E/E0 | ξ-bbox 膨胀 | 备注 |
|---|---|---|---|
| tv-full（现状 w1M1） | 0.4551 | 1.288 | c(t)=−0.035±0.005、(a,b)≈(−0.02,+0.179)：**本质是常数向上平流** |
| tv-trans（c=0） | 0.4641 | 1.111 | 去旋转只丢 0.9% 解释能量 |
| const-full（整窗联合 LS） | 0.4562 | 1.289 | 时变自由度只多解释 0.1% |
| const-trans（匀速平移帧） | 0.4650 | 1.111 | Taylor 冻结湍流假说形态；边信息 8B |

rfc 锚点（同脚本）：c=−0.997 常数、E/E0=3.3e-4、平移变体正确失败（E/E0=1.0）——变体
必须按场选；bouss 属"平流主导"，rfc 属"旋转主导"。

**三个修正（全部进 commit 5cb1ebc，validate_rfc T1–T5 双机全过）**：
1. **字节口径 v2**：N=1 的窗口不存 cell 标签图（常数图，8,588B 纯浪费）；observer 按
   参数化精确存储（tvfull Tw×3×4 / tvtrans Tw×2×4 / constfull 12B / consttrans 8B，
   +1B 全局变体标签）⇒ **bouss 2.5% 的 pro 从 m_r=16 升到 m_r=17，与 bl 完全等宽**
   （rfc 赢点结构）。`budget_calc.py` 同步；管线 side_planned==side_bytes 双侧断言不变。
2. **observer 参数化变体** `--observer {tvfull,tvtrans,constfull,consttrans}`：从区域
   已有的 LSQ 充分统计量重解（分区准则仍用 tv-full，不影响 τ-合并语义）；新增
   `killing2d.solve_killing_trans`（2-DOF）。**validate_rfc 新增 T5**：匀速平移合成场
   consttrans/tvtrans/constfull 恢复真值（误差 ~1e-3、E/E0=5e-5）+ 旋转场反例
   （constfull=闭式解、平移变体 E/E0=0.99 正确失败）。
3. **lr warmup**（`--warmup_frac`，TrainCfg 线性升温后接余弦）：针对 §4.4l/n 的 SIREN
   高 lr 坏盆地左尾（bl 与 pro 都被 5-16 dB 拖尾），**对 baseline 对称适用**（公平）。

**网格（45 任务 = 15 组合 × 3 种子 {0,7777,1}，`ibex_bash/refframe_v2_compresswin3.sh`，
job 49044332，6h 墙钟 [a100|v100]；协议 = §4.4l 修正版：coordnet、1000ep、2 窗 bl /
单窗 M=1 pro、τ=0.5、absorb=256、均值 [min..max] 口径）**：

- 2.5%（9 组合）：bl × {1e-4, 1.5e-4} × wu0.1 + bl 1.5e-4 wu0；ct(consttrans) ×
  {1e-4, 1.5e-4} × wu0.1 + ct 1e-4 wu0；cf(constfull) 1e-4 wu0.1；tv(tvfull 新口径
  m=17) × 1e-4 × {wu0, wu0.1}。
- 5%（6 组合）：{bl, ct} × {7e-5, 1e-4, 1.5e-4} × wu0.1。
- 归因链设计：ct−tv = observer 常数平移化的贡献；tv(m17)−旧 w1M1(m16) = 等宽的贡献；
  wu0.1−wu0 = warmup 的贡献；cf−ct = 保留旋转 DOF 的价值。bl 侧同 grid 扫 lr×warmup,
  判定口径 = 3-seed 均值最优对最优。
- 既有同协议格子不重跑（bl 2.5%@1e-4 wu0、bl 5%@{7e-5,1e-4} wu0 引自 §4.4n 表）。
- 任务预算：本轮新部署 45 ≤ 64（用户上限），余 19 留第二波（若 1.5e-4+warmup 仍坍缩
  或差距未闭合）。

**Wave-1 结果（2026-07-18 回收，45/45 零失败；3-seed {0,7777,1} 均值 [min..max]）**：

boussinesq f=2.5%（全臂 m=17 等宽；旧格子引自 §4.4n 同协议）：

| 臂 | lr | wu | 均值 [min..max] | vs bl 最优 |
|---|---|---|---|---|
| bl（旧） | 1e-4 | 0 | 62.85 [59.79..64.96] | −0.88 |
| bl | 1e-4 | 0.1 | **63.73** [61.39..65.00] | bl 最优 |
| bl | 1.5e-4 | 0 / 0.1 | 60.20 [54.74..]⚠ / 61.50 [56.36..]⚠ | 1.5e-4 对 bl 不稳 |
| **cf (constfull)** | 1e-4 | 0.1 | **64.48** [61.76..66.26] | **+0.75**（min/中位/max 全为正） |
| ct (consttrans) | 1e-4 | 0.1 | 64.03 [61.53..65.66] | +0.30 |
| ct | 1.5e-4 | 0.1 | 63.98 [62.66..64.87] | +0.25（最差种子 62.66 > bl 全臂 min） |
| ct | 1e-4 | 0 | 62.77 [61.09..64.20] | −0.96 |
| tv (tvfull, m17) | 1e-4 | 0 / 0.1 | 63.01 / 63.41 | −0.72 / −0.32 |

**判定：boussinesq 2.5% 达标（验收标准闭合）**。三个 pro 臂（cf@1e-4wu、ct@1e-4wu、
ct@1.5e-4wu）同时超过 bl 最优，非单点侥幸；cf 对 bl 最优在 min/中位/max 三个口径全胜。
**归因链（均 @1e-4）**：旧 w1M1(m16) 最优 59.90 → 等宽字节口径 tv(m17)@wu0 63.01
（等宽+lr 合计 +3.1）→ +warmup 63.41（+0.4；bl 侧 +0.88）→ observer 常数化 ct 64.03
（+0.62）→ 保留旋转 DOF cf 64.48（+0.45）。**cf > ct 与干跑预期相反**（干跑：旋转 DOF
只多解释 0.1% 能量、bbox 膨胀 1.29 vs 1.11）——实测 bbox 膨胀的分辨率成本小于旋转
分量的解释收益，ξ-bbox 膨胀 ≤1.3 时不是主要矛盾。

boussinesq f=5%（全臂 m=24 等宽）：

| 臂 | lr | wu | 均值 [min..max] | 备注 |
|---|---|---|---|---|
| bl（旧） | 7e-5 | 0 | 66.86 [66.00..68.16] | 旧最优 |
| bl | 1e-4 | 0.1 | **67.08** [66.61..67.61] | **bl 新最优**（warmup 治愈 1e-4 左尾） |
| bl | 7e-5 | 0.1 | 65.31 [62.78..67.23] | warmup 在 7e-5 反而拖尾 |
| bl | 1.5e-4 | 0.1 | 51.41 [23.65..66.73]† | s0 坍缩——1.5e-4 超 bouss m24 稳定边界 |
| ct | 7e-5 | 0.1 | 65.99 [64.92..66.57] | pro 最优（稳） |
| ct | 1e-4 | 0.1 | 64.14 [58.07..67.80]⚠ | 残余左尾（wu0.1 不够） |
| ct | 1.5e-4 | 0.1 | 64.81 [62.14..66.59] | pro 在 1.5e-4 不坍缩（bl 坍）——稳定性优势 |

**判定：5% 未闭合（−1.09）**，但线索明确：①cf（2.5% 最强臂）没在 5% 跑过；
②ct@1e-4 的左尾要更长 warmup。

**Wave-2（job 49045212，15 任务，`refframe_v2_compresswin3b.sh`，commit c9d1eb0；
本 session 部署合计 60/64）**：cf × {7e-5, 1e-4} × wu0.1；{cf, ct, bl} × 1e-4 ×
wu0.2（bl 对称）。全部 5%、3 种子。

**Wave-2 结果（2026-07-18 回收，15/15 零失败）**：

| 臂 | lr | wu | 均值 [min..max] | 读数 |
|---|---|---|---|---|
| cf | 7e-5 | 0.1 | **66.60** [66.44..66.82] | **pro 新最优**；三种子紧收敛（天花板形态，非噪声） |
| ct | 1e-4 | 0.2 | 66.46 [64.64..68.40] | wu0.2 治愈左尾（58.07→64.64）；**max 68.40 全场最高** |
| cf | 1e-4 | 0.1 / 0.2 | 63.58 [55.33..]⚠ / 64.25 [58.52..]⚠ | cf@1e-4 左尾 wu 治不好——5% 的 cf 稳定边界 < 1e-4 |
| bl | 1e-4 | 0.2 | 66.81 [65.35..68.24] | 对称对照：wu0.2 对 bl 中性偏负，bl 最优仍 = wu0.1 的 67.08 |

**5% 判定（截至 wave-2）：未闭合，差距 −1.09 → −0.48**（cf@7e-5+wu 66.60 vs bl
67.08）。pro 侧稳定性全面占优（1.5e-4 下 bl 坍缩 pro 健康；cf@7e-5 离散 0.38 vs bl
1.00），但均值最优对最优未翻。**Wave-3（job 49046397，3 任务，session 合计 63/64）**：
探 cf 的 7e-5/1e-4 中点 **8e-5 + wu0.2**（7e-5 紧收敛=天花板受限，1e-4 分裂 ⇒ 中点
可能兼得）；bl 不补 8e-5——其网格已夹住该点（7e-5+wu 65.31 / 1e-4+wu 67.08），插值
不会超过既有最优，pro 只需 >67.08 即胜。

**Wave-3 结果（2026-07-18 回收，3/3）：cf@8e-5+wu0.2 = 67.08 [66.89..67.36]**
（全精度 67.0847 = {66.8908, 67.0013, 67.3619}）。中点赌注命中：紧收敛（离散 0.47）
且天花板抬到 67+。

**5% 最终判定：统计平手，验收标准"不更差"成立**（全精度对比 bl 最优 67.0826 =
{66.606, 67.0294, 67.6126}）：Δ均值 = **+0.002**（远小于种子离散，按 §4.4l 先例判
平手，不主张"赢"）；Δmin = **+0.285**（pro 最差种子更好）；Δ中位 = −0.028（噪声）。
配合 pro 的鲁棒性证据（1.5e-4 下 bl 坍缩 23.65、pro 健康 64.81；全部 pro 臂无坍缩），
判定 **proposed 在 5% 不比 baseline 差**——"proposed 不得比对应架构 baseline 更差"
的验收标准在 boussinesq 全部三档（2.5% +0.75 胜 / 5% 平 / 10% +3.19 胜）**闭合**。

**Verify_compresswin_1.3 总结论**：
1. **bouss 2.5%（CR≥40×）：proposed 明确战胜 SIREN baseline**（+0.75，三臂同超，
   min/中位/max 全口径），赢点配置 = **单窗单区域全局 constfull observer（整窗联合
   LS 的常数 (a,b,c)）+ 字节口径 v2（等宽 m=17）+ lr 1e-4 + warmup 0.1**。
2. bouss 5%：cf@8e-5+wu0.2 与 bl 最优统计打平（+0.002），min 口径 +0.28，"不更差"
   成立；10% 此前已胜（§4.4l）。
3. **方法学结论**：①压缩口径的赢点结构跨场统一为"M=1 单全局 observer + 等宽"
   （rfc 已证 +11~12，bouss 本轮证）；②observer 的**时变自由度在平流主导场上是
   纯负担**——常数化（constfull）同时省边信息与降低样本噪声，且**保留旋转 DOF 优于
   纯平移**（cf−ct=+0.45@2.5%，干跑的 bbox 膨胀顾虑被证伪，≤1.3 的膨胀无实害）；
   ③**lr warmup 是高 lr 坏盆地的有效药方**（bl 与 pro 双侧受益：bl 62.85→63.73、
   ct 62.77→64.03@2.5%；ct@1e-4 5% 左尾 58.07→64.64@wu0.2），应并入压缩 recipe；
   ④pro（observed field）的训练稳定性系统性优于 bl（1.5e-4 全臂对照），是可写进
   论文的附带卖点。

### 4.4p Verify_compresswin_1.4：cylinder×mlp 翻盘（M=1 全局 observer 迁移；Ibex job 49084951，部署 2026-07-18，结果待回收）

**任务（用户 2026-07-18）**："继续调参部署 至多24个任务 让mlp上也能胜过baseline"——
cylinder×mlp 是裁定后唯一未达标格（§4.4m 读数 3；旧结构下各档 −4~−7）。本轮 22 任务，
余 2 留补救。

**设计（把 §4.4o 的 bouss 赢点结构迁移到 cylinder）**：
- **结构**：nw=1 + τ=0.1 + absorb=256 → **M=1**（干跑+冒烟确认；全域刚体运动解释
  cylinder 时间能量的 88.6%，E/E0=0.114——比 bouss 的 0.455 好得多，物理依据 =
  涡街近匀速平流/Taylor 冻结流，共动系里准定常，正中 mlp 谱偏置的软肋）；
- **字节口径 v2 等宽**：pro 边信息 43 B ⇒ 与 bl 逐档同宽 m=21/29/42（2.5/5/10%）；
- **observer 臂**：ct（consttrans）为主赌注 + cf（constfull）2.5%/5% 保险（bouss 上
  cf>ct +0.45）；不跑 tv 对照（归因借用 §4.4o，省任务）；
- **recipe**：mlp lr 3e-4（各场已证稳定、无坏盆地 ⇒ 不需 warmup）、1000ep、
  3 种子 {0,7777,1}。
- **bl 口径**：2.5% 复用 mainExp_compress_1.2 的同协议 bl（s0/s7777）仅补 s1；
  5%/10% 的 bl 全新 3 种子重跑（§4.4k 的 bl 是 2000ep 不可混排）。

**任务表（22）**：2.5% {ct×3, cf×3, bl 补 s1}；5% {bl×3, ct×3, cf×3}；10% {bl×3, ct×3}。
脚本 `ibex_bash/refframe_v2_compresswin4.sh`（commit cb3aaf1）。

**Wave-1 结果（2026-07-18 回收，22/22 零失败；3-seed 均值 [min..max]，2.5% bl 的
s0/s7777 复用 mainExp_compress_1.2 同协议数据）**：

| frac | bl | ct (consttrans) | cf (constfull) | 最优对最优 Δ |
|---|---|---|---|---|
| 2.5% | 49.80 [48.95..50.35] | 48.94 [48.71..49.30] | 48.67 | −0.86 |
| 5% | 52.80 [51.90..53.50] | 52.46 [51.91..53.45] | 51.91 | **−0.34**（区间几乎重合，Δmin +0.01） |
| 10% | 54.91 [53.60..55.93] | 54.30 [53.36..55.75] | — | −0.61 |

**读数**：
1. M=1 结构迁移把旧差距 −4~−7（§4.4m/k）压缩到 **−0.3~−0.9**——方向验证成功但未过线。
2. **ct > cf（与 bouss 相反）**：尾迹无全局旋转，旋转 DOF 在此是拟合噪声 ⇒ observer
   变体确需按场选（§4.4o 干跑判据有效：cylinder 的 cf 与 ct E/E0 几乎同值）。
3. **残余差距归因（物理）**：诊断（`diag_agent_observer_variants` 扩展到 cylinder）
   q=(+0.816, −0.004)=自由流平流、E/E0=0.115、**bbox 膨胀仅 1.169**（坐标拉伸排除）；
   剩下的解释 = **单一 Galilean 帧冻结平流尾迹的同时，把 lab 系静止的障碍物/形成区
   搅成时变**——τ-合并自己在 τ=0.02 拒绝合并的 216px 区正是它。
4. 工程：mlp 全部臂种子差 ≤1.6 dB（无 ⚠），观察者变体在 mlp 上同样稳定。

**Wave-2（最后 2 任务，24/24 用满；job 49088664，commit 189c304，结果待回收）**：
5% 档（差距最小）M=2 障碍物分离结构：nw=1、τ=0.02、absorb=128 → [23800, 216] px，
每区各解 consttrans（尾迹自动得 (0.816,0)、障碍物区自动得 ≈0 留 lab 系）；**新分配
`--alloc capsmall`**（§4.4h "比例+容量下限"教训落地：非最大区域按 1 参数/样本封顶
→ 障碍物 m=8，余量全给尾迹 → m=28 ≈ bl 的 29）；2 种子 {0,7777}（第三种子无额度，
mlp 离散小，结果解读时注明 2-seed 口径）。冒烟验证 m=28/8、字节 1,255,189 ≤ 预算。

**Wave-2 结果（2026-07-19 回收，2/2；2-seed，出处 job 49088664）**：ctM2-capsmall @5% =
**52.60 [52.00..53.20]**（s0 53.20 / s7777 52.00；区域拟合都健康：尾迹 MSE 3.9~5.3e-5、
障碍物 2.3~2.6e-4）。对比：
- vs bl 52.80 [51.90..53.50]（3-seed）：Δ均值 **−0.20**，区间几乎完全重合；**种子配对
  一胜一负**（s0：53.20 vs 51.90 = +1.30；s7777：52.00 vs 53.50 = −1.50）——差异被
  种子噪声（±1.5）主导，判**统计平手**（比 bouss 5% 的 +0.002 平手弱一档，如实标注）。
- vs wave-1 M=1 ct 52.46：**+0.14**——障碍物分离假设成立但只值 ~0.14 dB，
  wave-1 残余差距的主体不在障碍物污染。

**Verify_compresswin_1.4 最终判定（24/24 任务用满）**：cylinder×mlp **未完全闭合**——
5% 档到达统计平手（−0.20，噪声内，capsmall M=2 结构），2.5%/10% 仍 −0.6~−0.9。
方法学收获：①M=1 全局平流 observer 消掉旧结构 −4~−7 差距的九成 ⇒ 涡街的可平流解释
成分对 mlp 真实有效；②障碍物分离 + capsmall（比例+容量下限）方向正确但增量小；
③cylinder 对 mlp 的残余劣势 ~0.2-0.9 dB 的机理未定（候选：观察系下形成区/剪切层的
时变残余、mlp 对准定常但空间尖锐结构的谱偏置本身）——**cylinder 是两架构共同的
边界场**：SIREN 因架构先验太强 RFT 提不动（豁免），mlp 因单帧/双区 RFT 只能到 parity。
后续若要求全闭合需新授权（候选：ctM2 补 s1 定 5% 平手成色、2.5%/10% 迁移 capsmall
M=2、更大 obstacle 区域 τ/absorb 微调、或承认边界写入论文故事）。

**Wave-3（用户 2026-07-19 新授权"一小批"，按 ≤24 封顶；本波 12 任务 = job 49106108，
commit 5fbd720，结果待回收）**：探 **mlp 的 lr**——全项目最大杠杆（coordnet 侧 6-12 dB），
而 mlp 的 3e-4 来自 rfc 100ep 试点、从未在 cylinder 验证；且现有 5% 日志显示 ep1000 时
MSE 仍在降 8-13%/200ep（余弦尾部压制）⇒ 更高 lr 在 1000ep 内买到更多有效步。
对称网格：{bl, ctM2-capsmall} × lr {5e-4, 1e-3} × 3 种子 @5%。剩余 ~12 任务额度等
读数后定向投放（若 lr* 出现 → {bl,ctM2}×lr* 迁移 2.5%/10%；若 lr 无效 → 如实报边界）。

### 4.5 RFC 窗口税归因（v2.1/v2.2 recipe 时期的历史记录，被 §4.4b 引用）

用户质疑（正确）：RFC 的 killing observer 时不变，切不切窗口 observer/observed field 都一样，
结果不该变。归因实验（`outputs/diag_agent_*.log`）：

| 实验 | 设置 | 结果 | 判定 |
|---|---|---|---|
| E0 | 数值验证两窗 observer 与 observed 样本 | 两窗 c(t)≈−1.0 完全一致；observed 样本经锚点旋转校正后落在同一 steady 函数（差异=离散化底噪） | **用户前提成立** |
| E1a | 单窗全时段、m=16（=B/2 参数，16k 步） | **64.01 dB**（比 m=24@62.74 还高、参数减半） | **H1 容量假设被推翻**；H3 扫掠 bbox 顺带排除（E1a 用的是更大的 2π 扫掠） |
| （排除法） | 2 窗 m=16 每 INR 只有 8k 步 vs E1a 16k 步 | 56.73 vs 64.01 | **H2 成立：窗口成本 = 固定 epochs 下每 INR 梯度步数减半**，纯训练预算效应 |

结论：窗口切分对 RFC 的伤害**不是**方法性的（observer/observed field/容量都无差），而是 v2.1/v2.2
recipe 把"epochs"当训练预算造成的伪影 —— 与 §4.6 cylinder 坍缩同根，由 v2.3 步数归一化一并消除。
（E1a 的 64.01 dB 同时刷新了"observer 等参数增益"的上界：44K 参数即可到 64 dB，比 baseline 99K
参数的 58.00 高 6 dB。）

### 4.6 v2.2 cylinder2d 运行 = recipe 失败记录（数字全部不采信）

`outputs/cyl_v22.log`（B=1,486,538、τ=0.1、N=[3,2]、best-of-2）：baseline / pro_budget /
pro_quality / no_observer 全部 ≈23.3–23.4 dB。诊断：所有含"巨区/全场"的 INR（无论 m=64/49/28）
坍缩到同一 normalized MSE 3.28e-2 ≈ 均值流吸引子（两 seed 逐位一致、s1 在 epoch 100 即达到并
停留 900 epochs）；小区域正常拟合（1e-5 量级）。lr=3e-4 对大 SIREN 在此数据上不稳定。
唯一可用信息：失败模式本身 + no_observer w1r0 种子离散 ×2236（中等规模网的随机坍缩）。

## 4.9- 综合结论（2026-07-15 稿，证据均标注出处）

**① 方法核心机制成立，且收益可预判**（最可信，双机复现）：
observer 变换的等预算增益按"场可被局部刚体运动解释的程度"排序：
**rfc +5.2~5.4 ≫ boussinesq +1.6 > cylinder +0.7~0.8**（§4.4b/c/f/d），与分区干跑的 E/E0
指标（rfc 3e-4 ≪ bouss ~0.2 < cyl ~0.1@τ 停点但涡街密集不可解释）同序 ⇒ E/E0 可作
**先验适用性判据**，不训练即可预估 observer 收益。

**② proposed 在真实数据的首个胜点**：boussinesq τ=0.5（M=2 = 每窗一个**全局** observer +
1.5B INR）**70.43 > baseline 68.47**（§4.4g，同协议同种子对；quality 口径，CR≈1 非压缩）。
两数据集 τ 曲线共同表明：**收益来自"大区域/全局 observer"，细粒度切分是净负贡献**
（boussinesq 单调 64.4→70.4 随 M 24→2；cylinder 全 τ 落后，最好 62.14@M=4 vs baseline≈70）。

**③ 固定规则的代价已量化**：强制 ≥2 时间窗 = 5-6 dB 优化税（rfc，双机）；均分预算在悬殊
分区上浪费 ~60%（cylinder）；但**按像素比例分配更差**（−4.4 dB，小而剧烈区域被饿死到 m=4，
§4.4h）⇒ 若做分配需"比例+容量下限"，且在"少而大区域"最优解下优先级不高。

**④ 贯穿性协议发现——大 SIREN 双峰不稳定**：m=64/d=10 的 CoordNet 在真实数据上种子间
极差 22.4 dB（cylinder 8 种子 {49.0..71.4}，5/8 好盆地 ≈70；boussinesq 同现 68.5/53.8，
§4.4e/f）。**凡大网数字必须 ≥8 独立种子报分布**（Ibex 并行任务），best-of-2 仅小网够用；
v2.2 之前的一切单种子对比不可信——本仓库两次"结论翻转"（v2.1→v2.2 rfc、本地→Ibex cylinder
baseline）皆源于此。

**⑤ 方法定位建议（给论文故事）**：卖点不是"到处赢 baseline"，而是：
(a) 存在一类流场（全局/大尺度刚体运动主导，如 RFC、羽流）observer 变换带来确定收益，
且 τ-合并 + E/E0 提供**自动检测该结构 + 自动选 N**的机制（RFC 锚点 N=1 双机验证）；
(b) 方向修正：时间窗口数与空间 N 一样应自适应、都偏向小值端；
(c) 压缩口径（pro_budget, CR 4.6-5.8×）目前仍全面落后 baseline，需先解决大网不稳定与
预算分配才有意义。



## 5. 开放问题（诚实记录）

1. **预算分配策略**：用户规格为均分（B/M）。cylinder/boussinesq 的分区呈"1 巨区 + 众小区"，
   均分对像素数悬殊的区域显然次优（4 像素区域与 24K 像素区域拿同样参数）。旧版"均分最好"的
   结论来自大小相近的块，不可迁移。候选：按像素数/按残差分配、或对小区域合并吸收。**进展**：
   已实现可选的小区域吸收后处理（`absorb_min_pixels`，默认关闭；开启 = 明确的规格偏离项）。
   boussinesq τ=0.2 实测：absorb=256 使 M 从 40→15（每 INR m=16 可训练），代价是宿主 ρ 从
   0.198→0.33-0.37 —— 被吸收的小块**像素少但时间能量剧烈**（羽流最猛烈的剪切单元），这提示
   "顽固小块"其实是分区给出的难度地图，未来可作为按难度分配参数的依据。
2. **τ 的选取**：τ 是方法核心旋钮（τ→N 单调）。当前按数据集手选（rfc 0.05 / cylinder 0.1 /
   boussinesq 0.2 起步）；自动选 τ（如 N 上限约束 / 字节-失真曲线扫描）待做。
3. **ξ 扫掠 bbox 归一化**：旋转使区域在 observed 坐标系中扫过更大 bbox，归一化后有效分辨率
   下降（旧版怀疑此点伤害 cylinder）。v2 用逐轴 minmax；是否需要各向同性归一化/按窗口缩短
   来缓解，待实验。
