# Small-Label Morphology Naming：用少量标签做涡形态多分类

> 面向 *Scientific Machine Learning + Geometric Representation Learning* 的涡形态识别
> （发卡涡 / 涡丝 / 涡管 / 脱落涡 / 涡环 …），基于已有的 **FMT / DCT / VAE 轨线几何 encoder**。
> 目标：把"匿名聚类"升级成"有语义命名的多分类 / segmentation"，且只花几十~几百个人工标签。

---

## TL;DR（先说结论）

1. **你要的不是 clustering，是 Generalized Category Discovery (GCD)。** 纯聚类必然给"匿名簇"（你已经踩到这个点）。学术界把"用少量已知类标签给簇命名、同时还能发现新类"这件正事叫 GCD（Vaze CVPR'22 / SimGCD ICCV'23）。你的问题就是 GCD 的一个 domain 实例。
2. **你手上三样东西，恰好对应 label-efficient 学习的三大杠杆**：自监督表征（FMT/DCT）↔ SSL pretrain；可合成的参数化涡（Vatistas + Biot-Savart）↔ synthetic/sim2real 监督；IVD 二值 mask ↔ programmatic weak supervision。**这套组合几乎是为"少标签命名"量身定做的**，你缺的只是把它们串起来的流程。
3. **领域空白 = 你的新意**：流体 ML 现有工作全部是「二分类涡检测」或「零标签几何提取」或「无监督轨线聚类」，**没有一篇做过"learned 多涡型分类"**。这既是机会，也意味着没有现成 benchmark，需要自建评测。
4. **如果只做一件事**：用合成涡给每个形态造 labeled 原型 → 在**冻结**的 FMT embedding 上训一个 nearest-class-mean / ProtoNet 头得到"命名锚点" → 再用 semi-supervised k-means（GCD）把锚点扩散到真实数据、并留出"New/unknown"簇。这条脊柱在 ~10–50 标签/类的预算下就能跑通。

---

## 1. Level-3 方法家族速览

> "Level 3 = small-label morphology naming"：在无/自监督表征之上，用**极少量**人工标签得到**多类**语义。
> 下表按"对你最有用"排序，标签预算和效果均为文献报告值（多数已 fetch 核实）。

| 家族 | 代表作 | 机制一句话 | 标签预算 | 效果锚点 | 对你怎么用 |
|---|---|---|---|---|---|
| **① 自监督 + 线性探针/原型头** | SimCLRv2、BYOL、MAE、**MSN**、DINO/**DINOv2** | 无标签学表征，冻结后只训一个线性/最近类均值头 | 1–10% 标签，或 ~5–50/类 | SimCLRv2 1% 标签→73.9% top-1；MSN ~5/类→72.4%；DINOv2 冻结 kNN 83.5% | **你的主线**。FMT/DCT 就是这里的 encoder，冻结它、在上面挂分类头 |
| **② 少样本度量学习** | **Prototypical Net**、Matching Net、P>M>F | 类=支持样本的均值 embedding，按最近原型分类 | 1–5 shot/类 | ProtoNet miniImageNet 5-way 5-shot 68.2% | 冻结 FMT 上做 nearest-prototype，是最稳的第一个 baseline，兼作 embedding 质量体检 |
| **③ Generalized Category Discovery** | **GCD (Vaze'22)**、**SimGCD**、SelEx | 少量已知类标签 + 大量无标签(含新类) → 半监督 k-means / 参数化分类头，同时命名已知类+发现新类 | 部分类有标签，其余全无 | — | **你的框架**。已知类=合成能造的涡型，New 簇=真实数据里的未知形态 |
| **④ 深度聚类 + 命名** | DeepCluster、**SCAN** | 先无监督聚成语义簇，再每簇给 1 个标签命名 | 每簇 1 个标签 | SCAN CIFAR-10 无标签→88% | 你现在的聚类升级版：加"每簇 1 标"这一步就有语义 |
| **⑤ 冷启动主动学习** | **TypiClust**、BADGE、Core-set | 先聚类，选每簇最"典型"(到簇内平均距离最小)的样本给人标 | ≈ 簇数 | 低预算下稳超随机选样 | **决定"标哪几条轨线"**：每簇挑 1 条典型轨线给专家命名，一标标一片 |
| **⑥ 程序化弱监督** | **Snorkel / data programming** | 把领域启发式写成 labeling functions，产出带噪概率标签，用 label model 去噪 | 0 人工标签，写 N 条规则 | — | 把 IVD 阈值、LAVD、轨线闭合性等**物理判据**写成 LF，批量预标注 |
| **⑦ 半监督一致性/伪标签** | **FixMatch**、Mean Teacher、FlexMatch | 弱增广出伪标签，高置信才保留，逼强增广版一致 | 极少，如 4/类 | FixMatch CIFAR-10 40 标签(4/类)→88.6% | 有少量种子标签后，用它把无标签真实数据吃进来 |
| **⑧ 弱监督分割** | One-Thing-One-Click、10×-fewer-labels | 稀疏点/涂鸦标注 + 自训练标签传播 | ~0.02% 点 | 逼近全监督 | 3D 形态 **segmentation**：点几条种子轨线，沿 embedding 传播 |

**关键取舍（已核实）**：在 ~10–50 标签/类这个预算下，**冻结特征 + kNN/ProtoNet/线性头** 是最稳、最有竞争力的做法；**整体 fine-tune 会过拟合**（MAE/MSN 都有此证据）。所以先冻结 FMT 跑原型头，别急着端到端微调。

---

## 2. 领域现状与空白（关键发现）

流体/科学可视化里，"涡 + 机器学习"已有三条线，但**都不做多涡型命名**：

- **二分类涡检测（learned）**：Berenjkoub, Chen & Günther 2020（CNN 学涡边界，训练自合成 Vatistas/Lamb–Oseen 涡）、Deng 等 CNN 涡识别、以及你们自己的 **VortexTransformer (Zhang, Rautek & Hadwiger 2025)**。输出是"涡/非涡"或涡边界，**不是类型**。
- **零标签几何/拓扑提取（无学习）**：LAVD 客观涡提取（Haller 2016）、发卡涡自动提取（arXiv 2606.05229, 2025，基于 λ₂ + merge-tree + 骨架，**零标签、只提发卡一种**）。精确但每种形态都要手写一套判据，不可扩展。
- **无监督拉格朗日轨线聚类**：Coherent Structure Coloring（Schlueter-Kuck & Dabiri 2017）、Hadjighasem 等谱聚类涡检测。**这条线最贴近你**——它证明了"无监督就能从轨线分出 coherent/incoherent（≈涡/非涡二分类）"，正好为你"聚类已能分涡/非涡"的观察背书。但它给的还是**匿名 coherent set**，没有"这是发卡/涡管"的语义。

海洋/天文侧有成熟的"少标签命名"先例可借：
- **EddyNet (Lguensat 2018)**：U-Net 逐像素 3 类 {非涡, 反气旋, 气旋}，是"多类涡分割"最接近的模板。
- **Zoobot (Walmsley 2022)** galaxy morphology 基础模型：预训练后**几百个新标签即可 fine-tune**，一个标签就能做相似检索——这正是"foundation encoder + 少标签命名"范式在科学形态学上的成功样板，和你 FMT+少标签的路线同构。

> **结论**：**"用轨线几何 embedding 做 learned 多涡型分类"目前是空白**。你不是在补一个已解决问题，而是把 CV 里成熟的 GCD/SSL/weak-supervision 迁到一个没人做过的 domain。**这既是论文卖点，也意味着评测基准要自己搭。**

---

## 3. 推荐流水线（针对你的资源，分阶段）

> 记号：✅=你基本已有；🔨=需要搭。整体是一条 **synthetic-anchored GCD** 管线。

**Stage 0 — 表征（✅ 已有）**
在**全部无标签**轨线簇上自监督训练 FMT/DCT/VAE。这一步等价于 SSL pretrain，**不用动**。先做体检：冻结 embedding + kNN，看"涡/非涡"二分类是否 clean（对标 CSC 的无监督可分性）。若不 clean，先修表征再谈命名。

**Stage 1 — 合成命名锚点（🔨 核心，最高杠杆）**
用你能合成的参数化涡造 **labeled 原型库**：
- 2D：单 Vatistas 涡、co-rotating / counter-rotating 涡对、剪切层、脱落涡列（Kármán 街）。
- 3D：涡管/roller、发卡涡、涡环、涡丝/worm——用 **Biot–Savart 核线** 定形状 + **Vatistas 截面** 定强度分布，采样出速度场→积分轨线→喂 FMT。
每个形态→一批 labeled embedding。在**冻结** FMT 上训 nearest-class-mean / ProtoNet 头。**这一步直接消灭"簇无意义"问题**：它给了每个形态一个有名字的 anchor。

**Stage 2 — GCD 迁到真实数据（🔨）**
把 Stage 1 的合成原型当作 GCD 的"已知类"，混入真实 LBM/CFD 的无标签轨线，跑 **semi-supervised k-means（Vaze'22）** 或 **SimGCD 参数化头**：已知类被命名、同时开放 **"New/unknown" 簇** 容纳合成库覆盖不到的真实形态。这一步同时解决 **sim2real domain gap**（合成锚点 + 真实无标签一起学）和**新类发现**。

**Stage 3 — 冷启动人工标注预算（🔨，决定"标哪几条"）**
对真实数据的 New 簇 + 低置信样本，用 **TypiClust** 在 FMT embedding 上每簇挑 1 条**最典型**轨线给专家。专家一次命名=命名一整簇。**几十个标签就能覆盖整个 taxonomy**，且比随机/uncertainty 选样在低预算下更稳。

**Stage 4 — 弱监督融合（🔨，放大标签）**
把物理判据写成 **Snorkel 式 labeling functions** 批量预标注，再用 label model 去噪：
- IVD > 阈值 → 涡区（你已有）；
- 轨线在共动系里闭合 + 高 LAVD → 涡管/roller；
- 轨线对反平行、周期性脱落 → shedding；
- 展向涡量主导 + 骨架呈 Λ 形 → 发卡。
产出的概率标签 + Stage 3 的少量真标 → 用 **FixMatch** 把海量无标签真实轨线吃进来。

**Stage 5 — Segmentation（🔨，可选进阶）**
要 2D/3D 稠密形态分割时，当作 **point/scribble 弱监督**：专家点几条种子轨线，沿 FMT embedding 做标签传播 / One-Thing-One-Click 自训练，得到形态 segmentation。

**2D vs 3D 的现实**：**发卡/涡管/涡环/涡丝本质是 3D 现象**，2D 里能命名的主要是"单涡 vs 涡对 vs 脱落列"。你认为务实的分工是：**2D 用来打通并验证整条管线（Stage 0–4）**，**3D LBM（圆柱扰流 Re=6400、F22）才是多形态命名的主战场**。

---

## 4. 落地建议与坑

- **别整体微调**：~几十标签/类下冻结 FMT + 原型/线性头才是对的；端到端 fine-tune 会过拟合（有 MAE/MSN 佐证）。
- **合成-真实分布差**是最大风险：合成涡太"干净"。缓解=Stage 2 的 GCD 让真实无标签参与，+ 对合成涡加噪/加背景剪切做 domain randomization。
- **类别极不均衡**：脱落涡海量、涡环稀有。ProtoNet/最近类均值对不均衡比 softmax 头鲁棒；采样时按类平衡。
- **评测没有 GT**：① 合成集留出 held-out 报 accuracy（上界体检）；② 真实数据靠专家抽检 + 报 cluster purity / GCD 的 ACC（匈牙利匹配）；③ 和纯几何判据（λ₂、LAVD、发卡提取器）做交叉一致性。
- **命名可复现性**：固定合成原型库为"类定义的锚"，避免每次聚类簇序号漂移导致语义不稳定。
- **先验证再扩张**：先在 2D 上证明"合成锚点 + GCD 能命名 co-rotating/counter-rotating/shedding"，再上 3D 多形态。

---

## 5. 参考文献

**① SSL + 少标签探针**
- Chen et al., *Big Self-Supervised Models are Strong Semi-Supervised Learners* (SimCLRv2), NeurIPS 2020. arXiv:2006.10029
- Grill et al., *BYOL*, NeurIPS 2020. arXiv:2006.07733
- He et al., *Masked Autoencoders (MAE)*, CVPR 2022. arXiv:2111.06377
- Assran et al., *Masked Siamese Networks (MSN)*, ECCV 2022. arXiv:2204.07141
- Caron et al., *DINO*, ICCV 2021. arXiv:2104.14294 ｜ Oquab et al., *DINOv2*, TMLR 2024. arXiv:2304.07193

**② 少样本度量学习**
- Snell et al., *Prototypical Networks*, NeurIPS 2017. arXiv:1703.05175
- Vinyals et al., *Matching Networks*, NeurIPS 2016. arXiv:1606.04080
- Hu et al., *P>M>F*, CVPR 2022. arXiv:2204.07305

**③ GCD / 深度聚类**
- Vaze et al., *Generalized Category Discovery*, CVPR 2022. arXiv:2201.02609
- Wen et al., *SimGCD*, ICCV 2023. arXiv:2211.11727
- Van Gansbeke et al., *SCAN*, ECCV 2020. arXiv:2005.12320 ｜ Caron et al., *DeepCluster*, ECCV 2018. arXiv:1807.05520

**④ 主动学习 / 弱监督 / 半监督**
- Hacohen et al., *TypiClust*, ICML 2022. arXiv:2202.02794
- Ash et al., *BADGE*, ICLR 2020. arXiv:1906.03671 ｜ Sener & Savarese, *Core-set*, ICLR 2018. arXiv:1708.00489
- Ratner et al., *Snorkel / Data Programming*, NeurIPS 2016 / VLDB 2017. arXiv:1605.07723
- Sohn et al., *FixMatch*, NeurIPS 2020. arXiv:2001.07685 ｜ Zhang et al., *FlexMatch*, NeurIPS 2021. arXiv:2110.08263
- Liu et al., *One Thing One Click*, CVPR 2021. arXiv:2104.02246 ｜ Xu & Lee, *10× fewer labels point cloud*, CVPR 2020. arXiv:2004.04091

**⑤ 科学形态学案例 + 流体涡**
- Walmsley et al., *Zoobot / Galaxy morphology*, MNRAS 2022. arXiv:2110.12735
- Lguensat et al., *EddyNet*, IGARSS 2018. arXiv:1711.03954
- Schlueter-Kuck & Dabiri, *Coherent Structure Coloring*, J. Fluid Mech. 2017 ｜ Hadjighasem et al., *Spectral clustering Lagrangian vortex*, Phys. Rev. E 2016. arXiv:1506.02258
- Haller et al., *LAVD objective vortices*, J. Fluid Mech. 2016
- Berenjkoub, Chen & Günther, *Vortex boundary CNN*, 2020
- *Hairpin Vortices Extraction in TBL*（几何法，零标签）, arXiv:2606.05229, 2025
- Zhang, Rautek & Hadwiger, *VortexTransformer*, 2025（你们自己的工作）
