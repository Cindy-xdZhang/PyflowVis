# 交接文档：Reference-Frame 分区 INR 压缩 v2（session 2026-07-12 ~ 07-15）

> 下一个 session 从这里开始读。详细规格/全部数字/出处在
> [referenceframe_inr_v2.md](referenceframe_inr_v2.md)（下称"主文档"），本文只讲脉络。

## 1. 本 session 的任务

用户对之前 session 的代码与结论**全部不信任**，要求：
1. 重读基础材料（CoordNet 论文 + optimal-connection 的参考系理论文档）；
2. 检验旧 session 复现的 CoordNet baseline（`CoordNetCompression.py`）；
3. **完全重写** proposed 方法（τ-合并区域分区 + 每区域 killing observer + pushforward +
   每区域 INR，对比 baseline），并用 RFC（rotation four center，旋转四涡心解析场——一个
   steady 场叠加全局旋转相机构成的非定常场）做正确性锚点：任意时间窗口必须得到 N=1。
4. 中途追加：部署 Ibex 跨机复现验证本地→集群工作流；τ 敏感性实验；最大化并行（种子拆独立任务）。

## 2. 已完成

### 代码（全部在 `experiments/referenceframe_inr_v2/`，git main 分支）
- `killing2d.py` 2D killing（刚体）observer 最小二乘（独立重推导，cell 级统计 O(1) 合并）；
- `partition.py` 自底向上 τ-合并（**残差比准则** ρ=E/E0≤τ）+ 可选小区域吸收（absorb_min_pixels）；
- `frame.py` 帧积分 + pushforward + 逆变换（训练样本=区域像素的精确 pullback，无插值/无 blend）；
- `inr.py` 训练（v2.3 冻结 recipe + best-of-k seeds）、预算→网宽闭式公式；
- `pipeline.py` 四模式管线（baseline / pro_budget / pro_quality / no_observer 消融）+ 数据加载
  （`PYFLOWVIS_DATA2D` 环境变量指数据目录）；
- `validate_rfc.py` 正确性验证套件（T1 闭式解收敛 / T2 RFC N=1 / T3 往返 1e-15 / T4 双转子反例）；
- `audit_normalization.py` 归一化审计（所有输入链路 [-1,1]、eval==train 坐标）；
- `viz_partition.py` 分区可视化。
- Ibex 脚本：`ibex_bash/refframe_v2_repro.sh`（单任务复现）、`refframe_v2_sweep.sh`、
  `refframe_v2_tausweep.sh`（array 并行，一任务=一(数据集,模式,τ,种子)）。

### 实验（命名按组内规则，主文档 §4.0 对照表；全部可从日志复算）
- **mainExp_2.3**：rfc / cylinder2d / boussinesq 四模式，本地 + Ibex 双机；
- **Verify_window_2.1/2.3**：时间窗口代价归因（单窗诊断）；
- **Verify_seedstability_1.1**：cylinder baseline 8 独立种子分布（Ibex 并行）；
- **Verify_alloc_1.1**：预算均分 vs 按像素比例（负结果）；
- **Verify_tau_1.1**：τ 敏感性 5 点×2 数据集×2 种子（Ibex 20 并行任务）。

### 基础检验
- CoordNet baseline 逐条对论文核对通过（主文档 §1），参数量闭式公式与实测一致；
- 验证套件 + 归一化审计在 Windows/RTX3090/cu126 与 Linux/V100|A100/cu118 双平台全过；
- 本地→Ibex 工作流全链路验证（push→pull→数据路径→sbatch→日志回收）；rfc 六数字双机 ≤0.25 dB。

## 3. 关键发现（详见主文档 §4.9-final，逐条有出处）

**科学结论**：
1. **observer 变换核心机制成立**：等参数等步数增益 rfc +5.2~5.4 dB ≫ boussinesq +1.6 > cylinder
   +0.7~0.8，排序与免训练的 E/E0 指标一致 ⇒ E/E0 是先验适用性判据。RFC N=1 锚点双机验证。
2. **真实数据首个胜点**：boussinesq τ=0.5（每窗一个全局 observer + 1.5B INR）70.43 >
   baseline 68.47（quality 口径）。τ 曲线两数据集一致：**收益来自大区域/全局 observer，
   细粒度切分是净负贡献**。
3. 固定"≥2 时间窗"规则 = 5-6 dB 优化税（非方法本质）；压缩口径（pro_budget）目前全面落后。

**方法学教训（比数字更值钱）**：
4. **大 SIREN 双峰不稳定**：m=64,d=10 的 CoordNet 种子间极差 22.4 dB（好盆地≈70 vs 坏盆地
   49-63），两个真实数据集复现。**大网数字必须 ≥8 独立种子报分布**；本仓库两次"结论翻转"
   （v2.1→v2.2、本地→Ibex baseline）皆源于此。
5. **recipe 必须按"步数"思考而非 epochs**：数据集/区域间每 epoch 步数差百倍——v2.0 照搬论文
   （欠训练 3 个量级）→ v2.1/2.2 高 lr（大网坍缩到均值流吸引子）→ v2.3 自适应 batch
   （每 INR ≥64 步/epoch）+ lr 1e-5 才两头兼顾。
6. 增量式合并准则会被"切香肠"击穿（T4 反例抓到，改残差比准则）；按像素比例分配预算会饿死
   小而剧烈的区域（−4.4 dB，Verify_alloc_1.1 负结果）。

## 4. 下一步（按优先级）

1. **方法修正——自适应窗口数**：时间轴做与空间同构的自底向上合并（RFC 应自动得 1 窗），
   消除 5-6 dB 窗口税。这是把 §4.9 结论转化为方法改进的最直接一步。
2. **把"少而大区域"做成正式方法**：τ-合并作为"该场需要几个 observer"的检测器（取小 M 端），
   而非细分工具；boussinesq 的 70.43 胜点提示"每窗全局 observer"可能是最实用形态。
3. **压缩口径翻盘的前提**：大网稳定性（初始化/训练方案，如 SIREN 频率调度、warmup、或换
   FF+ReLU 对照）+ "比例+容量下限"的预算分配。不解决这两个，pro_budget 无法与 baseline 竞争。
4. **补缺**：Other_tworotor_1.1（双转子理想正例的 INR 实验，队列中断未跑，一条命令可补：
   `python run_experiment.py --field tworotor --tau 0.05 --absorb_min_pixels 64 --m_base 24 --d_base 4 --n_seeds 3`）；
   boussinesq 本地对照的 pro 模式（Ibex 已覆盖，可选）。
5. 更多数据集（beads2d、doublegyre2d、pipedcylinder2d、gerris 系列——`pipeline.load_field`
   已支持 gerris0..7）铺 E/E0→observer 增益的相关性曲线，坐实先验判据。

## 5. 交接注意事项（陷阱清单）

- **凡引用大网络（m≳49）的单次 PSNR 都不可信**，先查该运行的 seed spread（日志里
  `best-of-k ... spread` 行）；对比实验两侧协议必须相同（per-INR 取优 vs run 级取优有差别）。
- 改动 `killing2d/partition/frame` 后**必须重跑 `validate_rfc.py`**；改归一化相关代码后重跑
  `audit_normalization.py`。
- τ 与 absorb_min_pixels 有交互：absorb=256 会把 cylinder 所有 τ 压成 N=1（cylinder 用 64）。
- Ibex：数据在 `~/DeepVortex/FLowDataFolder/`（boussinesq.nc 本 session 已上传）；提交前核对
  **实验用到的每个数据文件**都在；sbatch 脚本保持 LF；种子/模式拆独立 array 任务（用户明确
  要求，最多 32 卡）；python 失败要 `|| exit 1` 否则 Slurm 谎报 COMPLETED。
- 本地长队列用后台链 + 磁盘 done 标记文件；Claude 进程重启会杀死后台链，重启后按 done 文件
  断点续排。
- 旧目录 `experiments/referenceframe_inr/`（v1）与其文档仅作踩坑参考，**禁止引用其数字**。

## 6. 资产索引

- 主文档（规格+全部结果+出处）：`docs/referenceframe_inr_v2.md`
- 代码：`experiments/referenceframe_inr_v2/`；日志：其 `outputs/`（本地）+ Ibex
  `slurm_logs/RFv2*.{out,err}`（jobs 48759543-45, 48810859, 48811329, 48814029）
- 关键 commits：`3e25405`（v2 初版+Ibex 部署）→ `0186ac6`（像素分配）→ `86136f4`（归一化审计）
  → `d3c9418`（Ibex 容错修复）→ `0915a18`（τ sweep）
- 记忆文件：`referenceframe-inr-compression.md`（已同步到 v2.3 状态）
