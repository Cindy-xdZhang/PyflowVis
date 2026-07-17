# 交接文档：Reference-Frame 分区 INR 压缩 v2（更新至 session 2026-07-16 ~ 07-17）

> 下一个 session 从这里开始读。规格/全部数字/出处在
> [referenceframe_inr_v2.md](referenceframe_inr_v2.md)（下称"主文档"），本文只讲脉络。
> 上一版 handover（session 07-12~15 的 v2 重写与 Verify_arch_1.1 部署）见 git 历史
> `git show 082199b:docs/referenceframe_inr_v2_handover.md` 之前的版本；其"已完成/教训"
> 均已并入主文档 §1-§4.9，本文不再复述。

## 0. 用户定的硬规则（违反 = 返工，全部有明示出处）

1. **验收标准（2026-07-16）**：只接受 cylinder2d 上 proposed 无法提高 coordnet（SIREN）；
   **其他数据集上 proposed 不得比对应架构的纯 INR baseline 更差**。
   （cylinder×mlp 是否同豁免未裁定——字面只豁免 SIREN，待问用户。）
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
| Verify_compresswin_1.2（§4.4m 读数 2 后续） | bouss {2.5,5}% × lr{3e-5,5e-5,7e-5,1e-4} × {bl,w1M1,w2M2} × 3 种子，超额种子已 scancel | **49033283** | **在跑，待回收** |
| （附带）Verify_arch_1.1 gerris finer pq 低 lr 重跑回收，§4.4j 表补全 | | 48965055/56 | 已回收 |

## 2. 当前记分板（对照验收标准；证据 = 主文档 §4.4k/l/m）

| 场 × coordnet | 2.5% | 5% | 10% |
|---|---|---|---|
| rfc | ✓ +11.4（同 lr 全胜） | ✓ +7.75（3-seed；pro 最差种子 > bl 中位） | ✓ +12.2（种子稳） |
| boussinesq | ✗ −11.9（w2M2；w1M1 在跑） | ✗ −1.61（3-seed；pro 稳但均值低） | ✓ +4.91（3-seed；bl 有坏盆地左尾，pro 极稳） |
| cylinder2d | 豁免（−7.3） | 豁免 | 豁免 |

- **rfc 全档闭合**，赢点 = **w1M1**（单窗全局 observer，M=1，与 bl 等宽）+ lr 3e-4。
- **bouss 赢点结构 = w2M2**（2 窗每窗全局 observer）+ lr 1e-4，但只在 10% 达标；
  5% 差 −1.6，2.5% 因 2 窗切分（m_r=11 vs bl m=17）大输 → 在跑的 1.2 网格赌两点：
  ①中间 lr（5e-5/7e-5）稳住 w1M1（其 1e-4 好种子 66.2 > bl 均值但分裂）；②2.5% 用
  w1M1 无切分近等宽（m=16 vs bl 17）。
- mlp：rfc 全档大胜（+7~+9），bouss/cyl 全输且 2.5% 档 mlp baseline 本身远弱于
  coordnet（谱偏置），不是压缩口径的竞争架构。

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
5. 最优窗口结构场依赖：rfc（observer 时不变）单窗；bouss 双窗。自适应窗口数（§4.9⑤b）
   仍是正确的方法化方向。

## 4. 下一步（按优先级）

1. **回收 Verify_compresswin_1.2（job 49033283）**：按 3-seed 口径出 bouss 5%/2.5% 的
   最终判定表（同 lr 配对 + 最优对最优）。若 w1M1 中间 lr 仍不达标：向用户汇报"bouss
   压缩口径的诚实边界 = 10%+"，并问是否接受（validated 的 10% 胜点 + rfc 全胜 + §4.9
   的 E/E0 先验判据已构成完整故事）。
2. **cylinder×mlp 豁免问题问用户**（字面标准只豁免 SIREN）。
3. Story v2 归因链缺口（上一 session 遗留）：mlp/finer 的等参消融（no_observer、单窗
   observed 诊断）；FINER×observed field 稳定化 = Verify_arch_1.2（lr 线索已有：pq 大
   INR 3e-5 分裂 / 1e-5 稳，§4.4j 读数 5c）。
4. 把 2.5-20% 的 rate-distortion 曲线统一到修正协议（20% 档目前只有旧协议数字）。
5. 更多数据集铺 E/E0→增益相关性（gerris 系列已支持）。

## 5. 陷阱清单（新增项在前，旧项仍有效）

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

- 主文档：`docs/referenceframe_inr_v2.md`（本 session 新增 §0 硬规则相关、§4.4k/l/m）
- 代码：`experiments/referenceframe_inr_v2/`（budget_calc.py 新增）
- Ibex 输出：`experiments/referenceframe_inr_v2/outputs/{mainExp_compress_1.1,
  mainExp_compress_1.2, Verify_compresswin_1.1}/`（compresswin_1.2 的新格子也写进
  Verify_compresswin_1.1/ 目录，靠 lr/种子后缀区分）；日志 `slurm_logs/RFv2{cmp,win,wse,c25,win2}.*`
- 关键 commits：`23524b1`（严格压缩协议+budget_calc）→ `8e7f05c`（compresswin sweep+max_inrs）
  → `b9ee09e`（2.5% 档）→ `082199b`（1.2 网格+5-seed 修订记录）
- 记忆文件：`referenceframe-inr-compression.md`（已同步到本文状态）
