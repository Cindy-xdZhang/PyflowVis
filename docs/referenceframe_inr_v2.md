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
| gerris4 | 68.31 / CR5.6x | τ=0.6/25INR 首跑 TIMEOUT；τ=0.7 重跑中（job 48891784） | — |

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
baseline 20 任务 = job 48900605（commit 7cebf9f，3B→4B 变更不影响 baseline）；
pro_quality 20 任务在 4B 变更后取消重提 = job（重提后回填）（commit 见 git log，4B 口径）。

### 4.5 "窗口为何伤 RFC"归因实验（agent，v2.1 recipe 下）

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
