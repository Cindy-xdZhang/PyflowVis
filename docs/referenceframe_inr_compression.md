# Reference-Frame 引导的 INR 压缩：实验报告

验证想法：**输入场 → 树形区域划分 → 每区 killing observer → pushforward 得 observed field → 每区一个 INR 拟合**，
是否比直接 `INR(v)` 更省参数 / 更高 PSNR。所有数字**确定性、可复现、可复算**
（脚本与原始日志见 `experiments/referenceframe_inr/`，下文引用的 `rfo_final*.py` / `ff_inr.py` 均在该目录）。

## 1. 方法
- 分区：`referenceFrameDecompose`（自底向上 merge tree，一阶 3-DOF killing）。
- 每区：其 observer `(a,b,c)_t` → 积分刚体运动 → **pushforward** 得该区 observed field `R(−θ)(v−u)`。
- 每区一个 INR 拟合 observed field；重建 = INR → 逆 pushforward + observer。
- 存储 = INR 参数 + 每帧每区 3 个 killing 标量（极小）。baseline = 直接 `INR(v)`。

## 2. 可复现设置（关键，血泪教训）
- **INR 固定 = Fourier-features + ReLU MLP**（`ff_inr.py`；scale=8, hidden=512, n_freq=256, layers=4,
  ~1.31M/网络, lr=1e-3, 梯度裁剪 + 全数据 best-snapshot）。
- **确定性**：`cudnn.deterministic + use_deterministic_algorithms + CUBLAS_WORKSPACE_CONFIG=:4096:8 + 固定 seed`。
- **为什么不用 SIREN**：CoordNet(SIREN, ω=30) 对 observer **1e-9** 扰动敏感 **0.7~18 dB**、训练发散，数字不可靠
  （ω/lr/epochs 全扫过都不行）；FF+ReLU 同扰动敏感性仅 **0.13 dB**。详见 `referenceframe_code_review.md` §3。
- **global ≡ cut=1**：两者独立算 observer 差 9e-9（浮点累加），是**同一个** observer → 用 bit-identical observer 时
  PSNR 完全相同（Δ=0.00 dB，铁证无 bug）。

## 3. 结果

### A. 单区 pushforward vs baseline（**同参数 1.31M，公平**，epochs=4000）
| 场 | baseline `INR(v)` | proposed cut=1(=global) | Δ |
|---|---|---|---|
| rfc | 39.31 | **43.86** | **+4.55** |
| cylinder2d | 45.14 | 46.71 | +1.57 |
| boussinesq | 40.98 | **45.49** | **+4.51** |

→ 单区 pushforward 对 **rfc（随动系 intrinsically steady）** 与 **boussinesq** 大幅有效；对 **cylinder（涡街本征非定常）** 收益小。
（cylinder 单区收益随 epochs 波动 +0.26~1.57，定性稳：小。）

### B. cut=1/3/6（**参数不固定**，PSNR 混容量，epochs=2500）
| 场 | baseline 1.31M | cut=1 1.31M | cut=3 3.94M | cut=6 7.89M |
|---|---|---|---|---|
| cylinder2d | 44.16 | 44.41 | 50.36 | 52.40 |
| boussinesq | 40.98 | 45.49 | 46.39 | 48.69 |

→ PSNR 随 cut 单调升，但参数同步涨 3–6×（cut=6 的 7.89M > cylinder 3.28M 像素，过参数化）。**不能把高 PSNR 归因于分区**。

### C. 固定参数预算（**公平判据**，epochs=2500）
| 预算 | 场 | baseline（1 个大 INR） | proposed（N 个小 INR） | 分区优势 |
|---|---|---|---|---|
| 3.94M | cylinder2d | 49.17 (h=930) | 50.36 (cut=3) | **+1.19（赢）** |
| 3.94M | boussinesq | 48.93 (h=930) | 46.39 (cut=3) | **−2.54（输）** |
| 7.89M | cylinder2d | *未跑 (h=1343)* | 52.40 (cut=6) | *不可比* |
| 7.89M | boussinesq | *未跑 (h=1343)* | 48.69 (cut=6) | *不可比* |

> 7.89M 两行的 baseline **从未训练过**，故 §3-B 里 cut=6 的高 PSNR **不能**与 baseline 比较。
> 若要补，跑 `rfo_final4.py` 的 h=1343。

## 4. 结论
1. **同参数预算下，分区依场而定**：cylinder 分区**赢** +1.19dB，boussinesq 分区**输** 2.54dB。分区不是普遍更好。
2. **decomposition benefit 不预测 INR 压缩的分区价值**：boussinesq benefit=0.43（强多流向）却分区更差。benefit 衡量 observer 空间的多流向度，而 INR 压缩的是场本身。
3. **分区的代价**：边界切断连贯结构（如 boussinesq 羽流蘑菇头跨区）+ 每个小 INR 欠拟合 + 无法共享全局相关性；这些可能盖过 observer 简化的收益。cylinder 的自由流区简单、更"可分"，故分区微赢。
4. **单区 pushforward（无分区）对 intrinsically-steady 场大幅有效**（rfc +4.55dB 同参数），是最稳的正收益。
5. **方法论铁律**：任何 PSNR 对比前必须 ① 锁 cuda 确定性、② 验证 INR 对输入微扰的敏感性 ≪ 待测方法差异、③ 严格控制变量（同 INR/epochs/observer 定义，只改被测量）。否则训练噪声会淹没甚至伪造结论。

## 5. 改进：区域重叠训练 + 加权融合重建（overlap+blend）

针对 §4 结论③（边界切断损失）的工程改进（`rfo_final5.py`）：每区训练时**膨胀 margin=4** 包含
邻域像素，重建时用**高斯距离权重融合**（sigma=4）相邻区预测（不再硬切边界）。固定预算 3.94M、epochs=2500：

| 场 | baseline(h=930) | 硬边界最优 | cut=2 | cut=3 | cut=5 | cut=10 |
|---|---|---|---|---|---|---|
| cylinder2d | 49.17 | cut=3 +1.19 | — | 47.17 (−2.00) | **50.52 (+1.35 赢)** | 49.31 (+0.14) |
| boussinesq | 48.93 | cut=3 −2.54 | 46.22 (−2.71) | **48.51 (−0.42)** | 45.55 (−3.38) | 43.93 (−5.00) |

overlap+blend 对 boussinesq cut=3 净赚 **+2.12dB**（46.39→48.51），证实**边界确实是主要损失来源**。

**结论**：
- **cylinder：在固定参数预算下确实超过 baseline（overlap cut=5 +1.35dB）**——空间可分的场，分区参考系压缩有效。
- boussinesq（连贯结构跨区）：overlap 大幅缩小差距，最好到 cut=3 的 −0.42dB，**未翻盘**；cut=2 更差（−2.71），
  故 cut=3 是其最优，**boussinesq 在任何测过的 cut 上都输 baseline**。
- overlap **依场而定**：相邻区 observer 差异大时（cylinder cut=3）blend 反而有害（−2.0 vs 硬边界 +1.19），差异小时（cut=5）有益。
- **固定预算下有最优 cut**：分太多 → 每区 INR 太小欠拟合（cut=10 普遍变差）；存在中间最优。

## 6. 自适应区域数 `cut_adaptive(tau)`：为什么"调 tau 让所有场都赢"不可行

`cut_adaptive(tau)` ≡ `cut(cost_threshold = tau · raw_td_energy)`，沿 dendrogram 的**合并顺序前缀**、
**首次代价超阈即停**。因此 **tau 只能挑选区域数 N，不能改变分区形状**——扫 tau 严格等价于扫 N。

### A. tau=0.02 实测（`rfo_final6.py`，固定预算 3.94M，epochs=2500，baseline 同轮 in-process 重算）

| 场 | adaptive N | baseline(h=930) | proposed | Δ |
|---|---|---|---|---|
| rfc | 1 | 35.24 | **40.76** | **+5.51（赢）** |
| cylinder2d | 2 | 49.17 | 47.38 (h=640) | −1.79 |
| boussinesq | 4 | 48.93 | 47.95 (h=436) | −0.98 |

### B. tau → N 映射（`tau_map.py`，纯 CPU，直接由 linkage 代价推出）

| tau | cylinder2d → N | boussinesq → N |
|---|---|---|
| 0.0005 | **10（+0.14 赢）** | ~60 |
| 0.001 | 6 | 48 |
| 0.005 | 3（−2.00） | 15 |
| 0.02 | 2（−1.79） | 4（−0.98） |
| 0.04 | 2 | **3（−0.42，bouss 最优）** |
| ≥0.09 | **1** | **1** |

**关键发现**：
1. **cylinder 的 N=5（唯一 +1.35dB 的赢点）对任何 tau 都不可达**。阈值前缀切法要求"该次合并代价 > 之前所有合并代价"
   （running max）才能停在该 N；cylinder 的第 L−5 次合并不满足，阈值扫过时直接从 N=6 跳到 N=4。
   cylinder 可达 N = {1,2,3,4,6,8,9,10}；不可达 N = {5,7,11,…}。boussinesq 不可达 N = {7,10,…}。
2. **两场对 tau 的需求方向相反**：cylinder 要赢需 tau≈0.00045（N=10），该 tau 把 boussinesq 切成 ~60 区
   （每区仅 65k 参数，必然崩溃）；boussinesq 最优需 tau≈0.04（N=3），该 tau 让 cylinder 退化为 N=2（−1.79）。
   **不存在任何统一 tau 使两场同时超过 baseline。**这是 dendrogram 的结构性事实，不是调参不够。
3. tau 用 `raw_td_energy` 归一**并不能**让阈值跨场可迁移：boussinesq 的 benefit=0.43（强多流向）导致合并代价整体偏高，
   同一 tau 下被切得远比 cylinder 碎。
4. **唯一在所有场上都一致的 tau 区间是 tau≥0.09 → N=1**，即方法退化为"全局 observer pushforward、不分区"。

## 6b. 决定性负结果：N=1 在 3.94M 预算下反转（`rfo_final7.py`）

§6 结论 4 曾寄望"N=1 全局 pushforward 处处为正"。**在 baseline 完全同参数（3.94M）、同 epochs（2500）下直接测，
两个真实场都是负的**：

| 场 | baseline(h=930) | proposed N=1(h=930) | Δ |
|---|---|---|---|
| rfc | 35.24 | 40.76 | **+5.52（赢）** |
| cylinder2d | 49.17 | 48.43 | **−0.74（输）** |
| boussinesq | 48.93 | 44.60 | **−4.33（输）** |

（脚本已复现 rfc 的 40.76，证明搬迁后管线无损。）

**这推翻了"observer 变换本身普遍提升压缩"的乐观外推。** 此前 N=1 为正的数字（cylinder +1.57、bouss +4.51）
取自 §3-A，预算只有 **1.31M 且 epochs=4000**——预算与 epochs 都与本表不同，不可直接外推。

**⚠ 未排除的混杂 = epochs（必须先解决再定论）**：本表 proposed 用 2500ep，而大网络（h=930）很可能欠训练。
铁证：boussinesq 的 proposed N=1 从 **1.31M/4000ep 的 45.49 掉到 3.94M/2500ep 的 44.60**——网络更大反而更差，
是欠训练的典型征兆。因此 −4.33 **可能是欠训练假象，而非方法反转**。

**待做（唯一负责任的下一步）**：固定 epochs=4000、预算 3.94M，同时重训 baseline 与 proposed N=1（两场）。
- 若 proposed 追平/超过 → observer 变换在大预算下仍有效，反转是假象；
- 若仍显著为负 → 收益随预算真实消失。**疑似机制**：pushforward 把每帧 warp 到随动系后，坐标点云 bbox 被
  旋转+平移放大，renorm 到 [-1,1] 后有效分辨率下降，固定 `Fourier scale=8` 在更大域上覆盖更少高频；小预算下
  observer 去非定常的收益 > 此损失，大预算下 baseline 吃满高频时此损失反超。若属实，proposed 的 FF scale 需
  **独立于 baseline 重调**才公平（当前 scale=8 是为 baseline 定的）。

## 6c. 自适应参数分配（`rfo_final8.py`）：负结果，均分最优

§7 曾把"每区 INR 大小按复杂度分配"列为最有希望的翻盘方向。**实测否定：均分（uniform）最优，任何非均分都更差。**

**前提（EP≤1000 硬约束）**：用户限定任何实验 ≤1000 epochs。这让 baseline（单个 h930 大网络）**严重欠训练**：
boussinesq baseline 从 @2500ep 的 48.93 掉到 @1000ep 的 **41.02**（−7.9dB）。本节所有数字在 EP=1000 下，
与 §3/§5 的 2500ep 数字**不可比**。

boussinesq（N=3, budget 3.94M, EP=1000, gamma=0.5, overlap+blend margin=4/sigma=4）：

| 配置 | PSNR | vs uniform | vs baseline |
|---|---|---|---|
| baseline（单 INR, h930） | 41.02 | — | — |
| **uniform（均分, 3×h512）** | **45.37** | **基准** | **+4.35** |
| adaptive:pixels（按面积） | 43.61 | −1.76 | +2.60 |
| adaptive:residual（按残差难度） | 44.01 | −1.36 | +2.99 |
| adaptive:respix（残差×面积） | 44.09 | −1.28 | +3.07 |

**结论**：
1. **自适应分配是死路**：把参数从均分挪向大区（pixels/respix）或高残差区（residual）都损失 1.3–1.8dB。
   疑因 observer pushforward 后每区 observed field 都相对好拟合、容量需求接近均匀，从任何区抽参数都是净损失（边际收益递减）。
   boussinesq 是 benefit=0.43（最该受益于分区）的场，它都否定，benefit 更低的 cylinder 无需重复验证。
2. **uniform +4.35 vs baseline 不是假象，是"固定训练预算下分块更易训练"的真实优势**：EP=1000 对 baseline 与
   proposed 同等固定，公平。baseline 是难训的大单网络，proposed 把它拆成易训的小网络——在**有限训练预算**
   （真实压缩部署常态）下，这本身就是更强的压缩能力。但此优势随 EP 增大而缩小（§5 同对比 @2500ep 已是 −0.42）。
3. **⚠ 机制未分离**：+4.35 可能**全来自"分块易训"，与 observer 变换无关**。当前所有 proposed 都带 observer
   pushforward，无对照 → 见 §6d。

## 6d. Observer ablation（`rfo_final8.py --no-observer`）：分离"分块"与"observer"两个收益源

关键对照：同分区 / N / 预算 / EP / overlap，唯一变量是 **observer 有无**（`--no-observer` 令每区 killing
coeff=0，`fit_region` 退化为直接拟合区域原始 v = 纯空间分区、无 pushforward）。

**分解（两场，N=BEST_N, budget 3.94M, EP=1000, uniform 均分）**：

| 场 | baseline | 纯分区(无observer) | 分区+observer | 分块 Δ | **observer Δ** |
|---|---|---|---|---|---|
| boussinesq (benefit 0.43) | 41.02 | 42.98 | 45.37 | +1.96 | **+2.39** |
| cylinder2d (benefit 0.11) | 43.42 | 46.11 | 44.05 | +2.69 | **−2.06** |

**结论（依场而定，observer 非普遍有益）**：
1. **分块（分区 + 小网络）在两场都稳健为正**（+1.96 / +2.69）：把难训的大 INR 拆成易训的小网络，是有限训练预算
   （EP≤1000）下的真实优势。
2. **observer pushforward 的贡献依场而定，不是普遍有益**：boussinesq **+2.39**（有益），cylinder **−2.06**（有害）。
   在 cylinder 上，纯空间分区（无 observer）46.11 反而是**全场最优配置**。
3. **与 decomposition_benefit 关联（可能可预测）**：高 benefit 的 boussinesq（0.43，整体运动可被刚体参考系解释）
   observer 有益；低 benefit 的 cylinder（0.11，涡街本征非定常、刚体 observer 解释不了）observer **有害**——
   疑因 pushforward 的 warp（坐标域旋转+平移放大、重采样）在 observer 帮助小时变成净损失（呼应 §6b 的高频损失机制）。
4. **⚠ 方法论修正**：本文件上一版仅凭 boussinesq 单场得出"observer 核心普遍成立"，被 cylinder 立即反转推翻。
   **单场不足以定论**（本项目铁律再次应验）。observer 只在"场可被刚体参考系有效解释"（高 benefit）时提升压缩。

## 7. 进一步方向（未来）
- ~~参数分配自适应~~ **已试，否定（§6c）**：按残差/像素非均分都不如均分，均分是最优分配。
- **EP–优势曲线**：proposed 相对 baseline 的优势随训练预算单调衰减（@1000ep +4.35 → @2500ep −0.42）。
  找 crossover EP、并论证"有限训练预算压缩"这一场景的合理性，是当前最诚实的立论点（若 §6d ablation 证 observer 有用）。
- **公平性对照**：用更小 batch 增加 @1000ep 的梯度步数，让 baseline 大网络也充分训练，检验 +4.35 是否收缩——
  区分"分块的训练效率优势"与"EP 卡太低的人为放大"。
- **换掉阈值前缀切法**：改用 rate–distortion 代价或肘部检测选 N，可让不可达的 N（如 cylinder N=5）变为可选。
- 融合权重 / margin / 是否 blend 自适应（相邻区 observer 相近才融合，差异大则硬切）；margin=4/sigma=4 目前是拍脑袋取值，未扫过。
- per-frame 划分跟踪移动结构（当前静态空间划分）。
