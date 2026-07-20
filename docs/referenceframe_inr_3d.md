# Reference-Frame 分区 INR 压缩 — 3D 扩展（deltaWing）

> 启动:2026-07-20,用户指令"现在进一步开始做 3d",数据
> `flowData3D/deltaWing_mag0_3reesampled.nc`。
> 2D 项目(方法、协议、全部教训)见 [referenceframe_inr_v2.md](referenceframe_inr_v2.md)
> (已收官:三场九格记分板全绿);本文档只记录 3D 新增部分,2D 的规则默认沿用
> (epochs ≤2000、种子 ≤3 {0,7777,1}、M ≤3、字节口径 = 参数×4+边信息 ≤ frac×场字节、
> 结论可追溯不许静默翻转)。

## 0. 数据

`deltaWing_mag0_3reesampled.nc`:非定常 3D 向量场,变量 u/v/w + 坐标 x/y/z/t。
维度 (T, Z, Y, X) = (171, 55, 314, 55),float32,全量 162,424,350 点 × 3 分量
× 4 B ≈ **1.81 GiB**(约为 2D cylinder 的 50 倍样本数)。物理:三角翼绕流
(前缘涡为主结构)。

## 1. 代码(`experiments/referenceframe_inr_3d/`)

v2 的维度直推,**v2 目录保持冻结**,3D 代码单独成目录;INR 训练器/归一化/
架构注册表(coordnet=SIREN / mlp / finer)直接 `import` 自 v2 的 `inr.py`
(其本身维度通用:k = coords.shape[1], p = values.shape[1],参数闭式公式
`coordnet_num_params(m, d, k, p)` 对 k=4,p=3 成立且有 torch 断言)。

| 模块 | 内容 | 与 2D 的差异 |
|---|---|---|
| `killing3d.py` | 6-DOF Killing LSQ(cell 充分统计量):u(x) = t_vec + w×x,A 列 = [J e_i \| J(e_i×x) − e_i×v] | 2D 是它的 wz 切片(e_z×x = x⊥);cell 变 k³ 体素;AtA (T,nCz,nCy,nCx,6,6),内存 ≈ T·nCells·288 B(k_cell 默认 4) |
| `frame3d.py` | 观察系积分 + exact pullback + 逆变换:dR/dt = ŵR 逐步 Rodrigues 累乘(非交换),D = ∫Rᵀt_vec ds 梯形;xi = Rᵀx − D,vtil = Rᵀ(v−u),v = R vtil + u | 2D 的 θ 累加是 w=(0,0,c) 特例(T5 逐位锚定);推导 D 的方程:x = Rξ+b ⇒ t_vec = ḃ−w×b,D:=Rᵀb ⇒ Ḋ = Rᵀt_vec |
| `partition3d.py` | τ-merge(残差比准则不变) | cell 6-邻接;标签 3D 展开;absorb 单位 = 体素 |
| `synth3d.py` | 合成场:斜轴旋转相机(闭式解 w = −ω0 n, t_vec = c0×w)、平移场、双转子(异轴异速) | 闭式解是 2D true_killing_params 的 n=e_z 推广 |
| `validate_rft3d.py` | T1-T5 验证套件(见 §2) | 新增 T5 = 2D 嵌入锚点 |
| `pipeline3d.py` | 数据加载(netCDF4 直读 + stride 降采样)、预算、observer 变体(tvfull/tvtrans/constfull/consttrans,6/3 DOF)、字节口径、训练/重建/PSNR | coords (x,y,z,tn) k=4 p=3;边信息:bbox 6+6 浮点、observer 24/12 B(const 变体);新增 `max_steps_per_epoch`(见 §3 规模协议) |
| `run_experiment3d.py` | CLI(默认 n_windows=1 允许单窗 = 2D 赢点结构) | `--stride_t/--stride_xyz/--t_max` 数据降采样并打进 field 名 |

## 2. 验证套件(validate_rft3d.py,训练实验前必须全过)

- **T1** 斜轴旋转相机制造场(轴 n=(0.3,0.5,0.8) 归一、c0 偏心):killing 解
  收敛到闭式 (t_vec, w);E/E0→0;exact/solved q 的 pushforward 对所有 t 恒为
  steady 图案 s(xi)。
- **T2** 平移场(Taylor 冻结):consttrans/constfull 恢复 c_vec、E/E0≪1;
  反例 = 旋转场上纯平移必须失败(E/E0≈1)。
- **T3** oracle 往返:随机场+随机 6-DOF q+散乱 mask,pushforward→逆变换
  逐位还原(<1e-10)。
- **T4** 双转子反例(左绕 e_z ω=0.8、右绕 e_x ω=−1.4,轴和速率都不同):
  τ-merge 不许并成 N=1,每半 w 恢复真值。
- **T5** **2D 嵌入锚点**(本套件的核心保险):z-复制 2D 场(vz=0,zs 对称)
  下,killing3d 的 (tx,ty,wz) 必须 == killing2d 的 (a,b,c)(<1e-6)、面外
  DOF≈0、E/E0 与 2D 一致(<1e-9);integrate_frame_3d(wz-only) 与 2D θ/D
  逐位一致(<1e-9);单 z 层 pullback 样本 (xi, vtil) 与 2D 逐分量一致。
  ⇒ 3D 实现被钉死到已双机验证的 2D 冻结实现上。

结果:(待填,首跑进行中)

## 3. 规模协议(3D 专属,防止 2D recipe 直搬爆算力)

全量场 162M 样本/窗,v2 recipe 的"每 epoch 全扫"= 每 epoch ~5000 步
(batch 32k),1000ep = 5M 步,不可行(2D cylinder 是 ~100 步/epoch)。
新协议旋钮 `--max_steps_per_epoch S`(默认 0 = 关):

- S>0 时训练集 = 网格的固定随机子采样(S×batch_size 个样本,种子可控),
  **评估始终全网格**;epoch 含义变为"固定 S 步",与 v2 的
  min_steps_per_epoch(下限)对称地设上限。
- 使用它的运行必须在结果里注明 S(cfg 全量落盘,自动可追溯);与不带 cap 的
  数字不混排。
- 数据侧降采样(--stride_t/--stride_xyz)是另一条路:场名自动带 `_s{t}x{xyz}`
  后缀,视为**独立数据集变体**,禁止跨 stride 混排数字。

## 4. 实验记录

### 4.1 Verify_rft3d_1.1:验证套件 + deltaWing 冒烟(进行中)

- 验证套件首跑:(待填)
- deltaWing 降采样冒烟(本地,分区行为 + 管线全链路 + 字节断言):(待填)
- 全量 E/E0 干跑(deltaWing 可被全局/少区域刚体运动解释的程度,直接决定
  RFT 3D 的收益预期,对应 2D 的先验判据):(待填)
