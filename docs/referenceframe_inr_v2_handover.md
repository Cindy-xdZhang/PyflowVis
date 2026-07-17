# 交接文档：Reference-Frame 分区 INR 压缩 v2（更新至 session 2026-07-17 ~ 07-18）

> 下一个 session 从这里开始读。规格/全部数字/出处在
> [referenceframe_inr_v2.md](referenceframe_inr_v2.md)（下称"主文档"），本文只讲脉络。
> 上一版 handover（session 07-12~15 的 v2 重写与 Verify_arch_1.1 部署）见 git 历史
> `git show 082199b:docs/referenceframe_inr_v2_handover.md` 之前的版本；其"已完成/教训"
> 均已并入主文档 §1-§4.9，本文不再复述。
>
> **本 session（07-17~18）头条**：boussinesq 压缩三档全部闭合——2.5%
> proposed 胜 SIREN baseline +0.75（Verify_compresswin_1.3，主文档 §4.4o），5% 统计
> 平手（+0.002，min +0.28）"不更差"成立，10% 此前已胜。赢点配置 = 单窗单区域
> **constfull 全局常数 observer** + 字节口径 v2 等宽 + lr warmup。

## 0. 用户定的硬规则（违反 = 返工，全部有明示出处）

1. **验收标准（2026-07-16；2026-07-18 补充裁定）**：只接受 cylinder2d 上 proposed 无法
   提高 coordnet（SIREN）；**其他数据集上 proposed 不得比对应架构的纯 INR baseline 更差**。
   **豁免仅限 SIREN**（理由：正弦基天然擅长涡街窄带振荡，架构先验强到 RFT 提不动）；
   **cylinder×mlp 不豁免**——是记分板未达标格（各档 −4~−7，但 M=1 等宽 + consttrans
   结构未试，涡街匀速平流 ⇒ 共动系准定常，对 mlp 可能是大增益；部署待用户授权）。
2. **压缩实验总 INR 数 M ≤ 3**（预算/M 太小则每网无拟合能力）；管线 `--max_inrs 3`
   硬保护，分区超限直接报错。
3. **"每窗长 ≤ T/2（即 ≥2 窗）"规则已解除**——M=1 需要单窗，`--allow_full_window` 转正。
4. **压缩实验 epochs 1000 足够**（全局硬上限 ≤2000 仍在，run_experiment.py 断言）。
5. **种子 ≤3（2026-07-17，"算力不要钱吗"）**：标准组 {0, 7777, 1}，报 均值+[min..max]；
   不允许靠加种子解决分裂——选稳定臂并如实标注。
6. 沿用：结论可追溯不许静默翻转；修订必须新旧并列；大改 killing/partition/frame 后必须
   重跑 validate_rfc.py；epochs 硬上限 2000。

## 1. 本 session 完成的

### 代码（commits 23524b1 → 082199b，全部在 main）
- `budget_calc.py`：按流场字节 × frac 反解 CoordNet 骨架 (m,d)（d 冻结只调宽度；
  proposed 先扣与管线逐字节一致的边信息再均分）。
- `pipeline.py`/`run_experiment.py`：`--budget_frac`（总字节 = 参数×4+边信息 ≤ frac×场
  字节，双侧断言；pro_quality 禁用）；`--max_inrs`。
- Ibex 脚本：`refframe_v2_compress.sh`（5/10/20%）、`refframe_v2_compresswin.sh`
  （τ×lr×窗口 sweep）、`refframe_v2_compresswin_seedext.sh`、`refframe_v2_compress25.sh`
  （2.5%）、`refframe_v2_compresswin2.sh`（bouss 网格补齐，跳过已有格）。

### 实验（全部零失败；出处 = 主文档小节 + outputs 目录 + job 号）
| 实验 | 内容 | job | 状态 |
|---|---|---|---|
| mainExp_compress_1.1（§4.4k） | 5/10/20% × {coordnet,mlp} × {bl,pro} × 3 场，72 任务（旧协议：lr 固定、2000ep、M 无上限） | 48967626 | 已回收 |
| Verify_compresswin_1.1（§4.4l） | bouss τ×lr×窗口 sweep（M≤3）+ rfc 单窗闭环，120 任务 + 种子扩展 18 | 49000294 / 49023725 | 已回收 |
| mainExp_compress_1.2（§4.4m） | 2.5% 档修正协议，36 任务 | 49025811 | 已回收 |
| Verify_compresswin_1.2（§4.4n） | bouss {2.5,5}% × lr{3e-5,5e-5,7e-5,1e-4} × {bl,w1M1,w2M2} × 3 种子 | 49033283 | 已回收（两档均未达标；发现 skip-bug） |
| **Verify_compresswin_1.3（§4.4o）** | observer 变体 + 字节口径 v2 + warmup 三波：wave1 45 任务、wave2 15、wave3 3（合计 63/64 预算） | **49044332 / 49045212 / 49046397** | **已回收：2.5% 胜 +0.75、5% 平手闭合** |
| （附带）Verify_arch_1.1 gerris finer pq 低 lr 重跑回收，§4.4j 表补全 | | 48965055/56 | 已回收 |

### 本 session（07-17~18）新增代码（commits 5cb1ebc → 0b1ed6d）
- `killing2d.solve_killing_trans`（2-DOF 平移 LS）；`pipeline --observer
  {tvfull,tvtrans,constfull,consttrans}`（从区域充分统计量重解，分区准则不变）。
- **字节口径 v2**：N=1 窗口不存 cell 标签图；observer 按参数化精确存储（const 8-12B）
  ⇒ bouss 2.5% pro m_r 16→17 与 bl 等宽。`budget_calc.py` 同步。
- `inr.TrainCfg.warmup_frac` / `--warmup_frac`：线性 lr 升温后接余弦（bl/pro 对称）。
- `validate_rfc.py` 新增 **T5**（平移场恢复 + 旋转场反例），T1–T5 双机全过。
- `diag_agent_observer_variants.py`：observer 变体干跑诊断（E/E0 × bbox 膨胀）。
- Ibex 脚本 `refframe_v2_compresswin3{,b,c}.sh`。

## 2. 当前记分板（对照验收标准；证据 = 主文档 §4.4k/l/m/n/o）

| 场 × coordnet | 2.5% | 5% | 10% |
|---|---|---|---|
| rfc | ✓ +11.4（同 lr 全胜） | ✓ +7.75（3-seed；pro 最差种子 > bl 中位） | ✓ +12.2（种子稳） |
| boussinesq | **✓ +0.75**（cf@1e-4+wu，三臂同超，全口径正，§4.4o） | **✓ 平手 +0.002**（cf@8e-5+wu0.2 67.08 vs bl 67.08；min +0.28；"不更差"成立） | ✓ +4.91（3-seed） |
| cylinder2d | SIREN 豁免（−7.3）/ **mlp 不豁免未达标（−4.1）** | 同左（−6.8） | 同左（−6.2） |

- **rfc + boussinesq 全档闭合**。赢点结构跨场统一：**M=1 单窗单区域全局 observer +
  与 bl 等宽**；lr 场依赖（rfc 3e-4 / bouss 1e-4 或 8e-5）+ warmup。
- **bouss 各档最优 pro 配置**：2.5% = constfull@1e-4+wu0.1（64.48）；5% =
  constfull@8e-5+wu0.2（67.08）；10% = w2M2@1e-4（69.37，§4.4l 旧结构，未用新
  结构重跑——若要统一故事可补 cf 单窗 @10%，见下一步）。
- 唯一未达标格 = **cylinder×mlp**（用户 07-18 裁定不豁免）：各档 −4~−7，但从未用
  "M=1 等宽 + const observer"新结构试过——涡街近匀速平流（Taylor 冻结），共动系下
  准定常，对无频域先验的 mlp 可能是大增益；**部署待用户授权**。
- mlp 在 2.5% 档 baseline 本身远弱于 coordnet（谱偏置），不是压缩口径的竞争架构
  （rfc 架构内仍 +9）。

## 3. 本 session 的方法学发现（写论文/设计实验都要用）

1. **lr 是此前"压缩口径全输"的主因**：小网最优 lr = bouss 1e-4 / rfc 3e-4，比冻结
   recipe 的 1e-5 高 6-12 dB；调对后 10% 预算 baseline（69.8）超过历史 22.7% 预算
   baseline（68.5）⇒ §4.4k 的绝对数字系统性偏低（架构内方向仍有效），论文口径用
   修正协议数字。
2. **均值流坍缩是数据依赖的，不是大网专属**：bouss @3e-4 连 m=17/24 都全臂坍缩
   （23.4-23.7），rfc 同 lr 完全健康 ⇒ lr 必须按（场,规模）选，且稳定性检查要用与
   部署一致的调度长度（§4.4j 教训 3）。
3. **切分成本随预算收紧超线性**：bouss w2M2 5%→2.5% 从 −1.6 恶化到 −11.9 ⇒ 最严预算
   下唯一可行结构是 M=1 无切分（rfc 已证）。
4. **小网 SIREN 也有坏盆地左尾**（bouss@1e-4 bl：5 种子里 1 个塌 5-16 dB）——曾把
   2-seed 符号翻两次（§4.4l 修订块，新旧并列）。在 ≤3 种子规则下的对策：报
   [min..max]、优先选种子稳的臂（pro w2M2/w1M1@低 lr 天然更稳，这本身是卖点：
   **observed field 训练更稳**，§4.4 也见过）。
5. 最优窗口结构场依赖：rfc（observer 时不变）单窗；bouss 双窗（10% 旧结构）→ 本
   session 后修正：**bouss 在 2.5%/5% 的赢点也是单窗**（constfull 常数 observer），
   "双窗更优"只在 10% 未复检。自适应窗口数（§4.9⑤b）仍是正确的方法化方向。

### 本 session（07-17~18）新增发现（证据 = 主文档 §4.4o）
6. **observer 的时变自由度在平流主导场上是纯负担**：bouss 全域 killing 解 = 常数向上
   平流（c≈−0.035、b≈+0.179，时变只多解释 0.1% 能量，干跑 diag）⇒ constfull（整窗
   联合 LS 一个 (a,b,c)）**优于逐帧 tvfull**（+0.62@2.5%），且省 1.5KB 边信息。
7. **保留旋转 DOF 优于纯平移**（cf−ct=+0.45@2.5%）——干跑的"bbox 膨胀成本"顾虑被
   证伪：ξ-bbox 膨胀 ≤1.3 时对 PSNR 无实害，解释能量优先。
8. **字节口径 v2 是免费的等宽升级**：N=1 不存标签图 + const observer 8-12B ⇒ 2.5%
   档 pro m 16→17；等宽是赢的必要条件（5% 等宽仍输过 → 还需 warmup+常数化）。
9. **lr warmup（0.1~0.2×epochs 线性升温）是高 lr 坏盆地的通用药方**：bl 与 pro 双侧
   受益（bl@1e-4 左尾治愈 62.85→63.73@2.5%、67.08@5%；ct@1e-4 5% 左尾 58.07→64.64
   @wu0.2）。**已并入压缩 recipe 推荐**；wu 长度本身是超参（7e-5 下 wu0.1 对 bl 反而
   微负）。
10. **pro（observed field）训练稳定性系统性优于 bl**：1.5e-4 下 bl 坍缩（23.65）而
    全部 pro 臂健康；可写进论文的附带卖点。

## 4. 下一步（按优先级）

1. **cylinder×mlp 翻盘尝试（用户 07-18 裁定后的唯一未达标格，待授权）**：
   {bl, pro-M=1×{constfull,consttrans}} × {2.5%, 5%} × lr 3e-4 × wu{0.1} × 3 种子
   ≈ 24 任务；物理依据：涡街近匀速平流（Taylor 冻结），共动系准定常，对无频域先验的
   mlp 可能是大增益；新工具链（observer 变体+字节口径 v2+warmup）已就绪。
2. **bouss 10% 用新结构复检**（可选统一故事）：cf 单窗 @10% × {8e-5,1e-4} × wu ×3
   ≈ 6-9 任务——若胜过 w2M2 的 69.37，三档赢点统一为"单窗 constfull"。
3. Story v2 归因链缺口（遗留）：mlp/finer 的等参消融（no_observer、单窗 observed
   诊断）；FINER×observed field 稳定化 = Verify_arch_1.2。
4. 把 2.5-20% 的 rate-distortion 曲线统一到修正协议（20% 档目前只有旧协议数字；
   2.5/5% 现在有新结构数字，10% 待复检）。
5. 更多数据集铺 E/E0→增益相关性（gerris 系列已支持）；E/E0 判据可扩展为
   "constfull E/E0"（新变体的先验判据口径）。

## 5. 陷阱清单（新增项在前，旧项仍有效）

- **skip-existing 检查必须匹配完整配置**（observer/结构/窗口数），不能只匹配 mode 名
  ——compresswin2.sh 的 J2 bug 把 w1M1 格子当成已有的 w2M2 跳过（§4.4n）。
- **字节口径 v2（commit 5cb1ebc 起）与旧口径不可混排**：新口径 N=1 免标签图、observer
  按变体存储；旧 w1M1 数字（m=16）是旧口径。对比时以"总字节 ≤ frac×场字节"约束下的
  PSNR 为准（两代口径都满足约束，公平）。
- **lr 稳定性边界按（场,预算档,observer）三元组定**：bouss 5% 的 cf@1e-4 左尾治不好
  （wu0.2 仍 58.5），2.5% 的 cf@1e-4 健康；1.5e-4 对 bl 5% 坍缩、对 ct 健康。
- **种子 ≤3 硬规则**：任何新实验不得超 3 种子；已有 5-seed 数据只作留档（3-seed 复核
  符号全一致，主文档 §4.4l）。
- **不同协议数字不可混排**：§4.4k（lr 固定/2000ep/M 无上限）与 §4.4l/m（场级 lr/
  1000ep/M≤3）是两套口径，引用必须注明；best-of-k 与 2/3-seed 均值也不可混排。
- bouss@1e-4 的 2-seed 符号不可信（左尾）；报区间。
- bouss/cyl @3e-4 坍缩；rfc @3e-4 最优——lr 表：rfc {1e-4,3e-4}、bouss {3e-5..1e-4}、
  cyl {3e-5,1e-4}。
- `--allow_full_window` 已是正式臂（单窗），但 `--max_inrs 3` 必须一起带上。
- cylinder M≤3 需 absorb=256（absorb=0 会给 M=5）。
- Ibex：pull 后核 HEAD+md5；sbatch 脚本保持 LF；python 失败 `|| exit 1`；数据在
  `$HOME/DeepVortex/FLowDataFolder`（gerris 在 `/ibex/user/zhanx0o/FLowDataFolder`）。
- 旧目录 `experiments/referenceframe_inr/`（v1）禁止引用。

## 6. 资产索引

- 主文档：`docs/referenceframe_inr_v2.md`（本 session 新增 §4.4n/§4.4o；§4.5 标题修复）
- 代码：`experiments/referenceframe_inr_v2/`（observer 变体、字节口径 v2、warmup、T5、
  diag_agent_observer_variants.py）
- Ibex 输出：`experiments/referenceframe_inr_v2/outputs/{mainExp_compress_1.1,
  mainExp_compress_1.2, Verify_compresswin_1.1, Verify_compresswin_1.3}/`
  （compresswin_1.2 的格子在 Verify_compresswin_1.1/ 目录、compresswin_1.3 三波都在
  Verify_compresswin_1.3/ 目录，靠 arm/lr/wu/种子后缀区分）；日志
  `slurm_logs/RFv2{cmp,win,wse,c25,win2,win3,win3b,win3c}.*`
- 关键 commits：`23524b1`（严格压缩协议）→ `8e7f05c`（compresswin sweep+max_inrs）
  → `b9ee09e`（2.5% 档）→ `082199b`（1.2 网格）→ `acedc4a`（cyl 豁免裁定）→
  **`5cb1ebc`（observer 变体+字节口径 v2+warmup+T5）** → `9d254e8`/`0b1ed6d`（1.3
  wave-1/2 结果）→ 本 commit（wave-3 + 全档闭合）
- 记忆文件：`referenceframe-inr-compression.md`（已同步到本文状态）
