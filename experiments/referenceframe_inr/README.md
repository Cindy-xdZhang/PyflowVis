# Reference-frame 引导的 INR 压缩：实验脚本

结论与数据见 [`docs/referenceframe_inr_compression.md`](../../docs/referenceframe_inr_compression.md)；
代码审阅与可复现性规程见 [`docs/referenceframe_code_review.md`](../../docs/referenceframe_code_review.md)。

被测方法依赖 `FLowUtils/ReferenceFrameOptimization.py` 与 `FLowUtils/referenceFrameDecompose.py`。
这些脚本原本散落在临时 scratchpad 目录，已归档至此；`logs/` 是它们当时的原始输出。

## 核心组件
| 文件 | 作用 |
|---|---|
| `ff_inr.py` | Fourier-features + ReLU MLP（**固定 INR**，取代 SIREN；见 code_review §3） |
| `rfo_decompose_inr.py` | `fit_region`（单区 pushforward → INR → 逆变换）、`vpsnr`、确定性设置、`DEV` |
| `rfo_final5.py` | `proposed_overlap`：区域重叠训练 + 高斯权重融合重建（margin/sigma） |

## 实验入口（按时间顺序，后者取代前者）
| 文件 | 问题 | 状态 |
|---|---|---|
| `rfo_final2.py` | 单区 pushforward vs baseline，同参数 1.31M | 完成 → `logs/final2.log` |
| `rfo_final3.py` | cut=1/3/6，参数不固定 | 完成（PSNR 混了容量，不可直接比） → `logs/final3.log` |
| `rfo_final4.py` | baseline 在 3.94M 预算下的 PSNR（h=930） | 完成 → `logs/final4.log` |
| `rfo_final5.py` | overlap+blend，固定预算 3.94M，cut=3/5/10 | 完成 → `logs/final5.log` |
| `cut2_test.py` | boussinesq cut=2 补测 | 完成（−2.71dB） → `logs/cut2.log` |
| `rfo_final6.py` | `cut_adaptive(tau)` 自动选 N，tau=0.02 | 完成 → `logs/final6.log` |
| `tau_map.py` | tau→N 映射（纯 CPU，不训练） | 完成 → `tau_map.json` |
| `rfo_final7.py` | **直接指定 N**（绕开 tau 的可达性限制）；`--regions 1` = 全局 pushforward 不分区 | 见下 |

`rfo_final7.py` 是当前入口。它把 final6 混在一起的两件事拆开：*评估哪个 N* 与 *tau 如何映射到 N*。
必要性见 docs §6：阈值前缀切法无法到达某些 N（cylinder 的 N=5 恰好不可达，而那是它唯一的赢点）。

```bash
cd experiments/referenceframe_inr
python tau_map.py                                              # tau -> N，秒级，先看这个
python rfo_final7.py --fields rfc --regions 1                  # 冒烟：应复现 40.76dB / +5.51
python rfo_final7.py --fields cylinder2d,boussinesq --regions 1   # 决定性实验：同预算全局 pushforward
python rfo_final7.py --fields cylinder2d --regions 5 --margin 8   # margin 扫描
```

baseline（3.94M / 2500ep / h=930）默认走 `BASELINE_CACHE`，取自 final6 的 in-process 数值；
`--recompute-baseline` 可强制重训。缓存是安全的：确定性已验证（同配置两次训练 ΔPSNR=0.00dB，
且 final4 与 final6 独立跑出相同的 49.17 / 48.93）。

## 审阅 / 验证脚本
`verify_*.py`（符号约定、高阶模式、单调性、残差）、`audit_*.py`（blend 覆盖、round-trip、归一化确定性）、
`sensitivity.py` / `robust_test.py` / `omega_test.py`（SIREN 对 observer 微扰的混沌敏感性 → 换成 FF+ReLU 的依据）、
`test_decompose.py` / `test_rfo.py` / `test_rfc.py`（merge-tree、SAT、observer 求解正确性）。

## 数据
`cylinder2d.nc` / `boussinesq.nc` 从 `OneDrive - KAUST\WorkingInProcess\FLowVisAssets\flowData2D`
加载（路径在脚本里硬编码），分别重采样到 `(128,320,80)` 与 `(128,75,225)`；`rfc` = 解析场
`rotation_four_center((64,64),64)`。
