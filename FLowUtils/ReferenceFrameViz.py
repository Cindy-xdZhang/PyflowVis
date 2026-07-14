"""
FLowUtils/ReferenceFrameViz.py

参考系优化结果的**离线可视化**（matplotlib streamplot 对比图），不依赖 GUI 渲染引擎，
适合远程 / 无桌面时快速对比 killing / gunther17 恢复的 observer 场是否正确。

配合 FLowUtils/ReferenceFrameOptimization.py 使用，详见 docs/referenceframeOptimization.md。

典型用法：
    from FLowUtils.AnalyticalFlowCreator import rotation_four_center, constant_rotation
    from FLowUtils.ReferenceFrameViz import quick_compare_2d
    rfc = rotation_four_center((64, 64), 64)
    cr  = constant_rotation((64, 64), 64)          # 已知真值 observer（可选）
    quick_compare_2d(rfc, reference=("constant_rotation (ground truth)", cr),
                     save_path="rfc_verify.png")

判读：若真值 observer 是恒定运动（如恒定旋转），则好的方法恢复出的 observer 行应
**各时间列一致**，且与 reference 行一致。（注意：observed 场 v-u 在固定欧拉网格上可能仍随
时间旋转——参考系的定常性是在随动系里，不是欧拉点上，所以验证看 observer u 而非 v-u。）
"""
import numpy as np

from .VectorField2d import UnsteadyVectorField2D
from .VectorField3d import UnsteadyVectorField3D


def _to_txy2(field) -> np.ndarray:
    """取场数据为 (T,Y,X,2) numpy；3D 场取中间 z 切片的 (vx,vy) 两个平面内分量。"""
    data = field.field
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    data = np.asarray(data, np.float64)
    if data.ndim == 5:               # (T,Z,Y,X,3) -> 中间 z 切片, 取平面内分量
        z = data.shape[1] // 2
        data = data[:, z, :, :, :2]
    return data                      # (T,Y,X,2)


def _coords_xy(field):
    if field.getDim() == 3:
        gi = np.asarray(field.gridInterval); bmin = np.asarray(field.domainMinBoundary)
        x = bmin[0] + gi[0] * np.arange(field.Xdim)
        y = bmin[1] + gi[1] * np.arange(field.Ydim)
    else:
        x = field.domainMinBoundary[0] + field.gridInterval[0] * np.arange(field.Xdim)
        y = field.domainMinBoundary[1] + field.gridInterval[1] * np.arange(field.Ydim)
    return np.asarray(x, float), np.asarray(y, float)


def plot_observer_recovery(input_field, observers, reference=None, timesteps=None,
                           save_path=None, title=None, show=False, density=1.1):
    """画 observer 恢复对比图（streamplot，按速度大小着色）。

    行：输入 v / (reference observer) / 各方法 observer u ；  列：若干时间片。

    Args:
        input_field: 输入场 v (UnsteadyVectorField2D/3D)。
        observers:   dict {label: observer 场 u}，如 {"killing": rk.u_field, "gunther17": rg.u_field}。
        reference:   可选真值 observer 场，或 (label, 场) 元组。
        timesteps:   时间索引列表，默认 3 个均匀取样。
        save_path:   给了就保存 png。
        title:       图标题。
        show:        True 则 plt.show()（交互）。
        density:     streamplot 流线密度。
    Returns:
        matplotlib Figure。
    """
    import matplotlib
    if save_path is not None and not show:
        try:
            matplotlib.use("Agg")     # 无桌面环境保存 png 用 Agg
        except Exception:
            pass
    import matplotlib.pyplot as plt

    x, y = _coords_xy(input_field)
    vI = _to_txy2(input_field)
    T = vI.shape[0]
    if timesteps is None:
        timesteps = [0, T // 2, T - 1] if T >= 3 else list(range(T))

    rows = [("input  v", vI)]
    if reference is not None:
        rlabel, rfield = reference if isinstance(reference, tuple) else ("reference observer", reference)
        rows.append((rlabel, _to_txy2(rfield)))
    for label, uf in observers.items():
        rows.append((f"{label}  (observer u)", _to_txy2(uf)))

    nrows, ncols = len(rows), len(timesteps)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.6 * nrows), squeeze=False)
    tmin, tmax = input_field.getMinTime(), input_field.getMaxTime()
    for r, (label, data) in enumerate(rows):
        for c, ti in enumerate(timesteps):
            ax = axes[r][c]
            U = data[ti, :, :, 0]; V = data[ti, :, :, 1]; spd = np.hypot(U, V)
            ax.streamplot(x, y, U, V, color=spd, cmap="viridis",
                          density=density, linewidth=0.7, arrowsize=0.7)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlim(x[0], x[-1]); ax.set_ylim(y[0], y[-1])
            if r == 0:
                tphys = tmin + (tmax - tmin) * ti / max(1, T - 1)
                ax.set_title(f"t = {tphys:.2f}", fontsize=10)
            if c == 0:
                ax.set_ylabel(label, fontsize=9)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97] if title else None)
    if save_path is not None:
        fig.savefig(save_path, dpi=110)
        print(f"[ReferenceFrameViz] saved {save_path}")
    if show:
        plt.show()
    return fig


def observer_temporal_variation(observer_field) -> float:
    """observer 场随时间的变化量（逐点跨时间 std 的空间均值）；恒定运动 observer 应 ≈ 0。"""
    u = _to_txy2(observer_field)
    if u.shape[0] < 3:
        return float(np.mean(np.std(u, axis=0)))
    return float(np.mean(np.std(u[1:-1], axis=0)))


def quick_compare_2d(field, reference=None, neighborhood=4, timesteps=None,
                     save_path=None, title=None, show=False):
    """一键：对 field 跑 killing_2d + gunther17_2d，画 observer 恢复对比图。

    Args:
        field:       输入场 v (UnsteadyVectorField2D)。
        reference:   可选真值 observer 场，或 (label, 场) 元组（如 constant_rotation(...)）。
        neighborhood: Günther 邻域半径。
        save_path/title/show/timesteps: 见 plot_observer_recovery。
    Returns:
        (fig, {"killing": ReferenceFrameResult, "gunther17": ReferenceFrameResult})
    """
    from .ReferenceFrameOptimization import killing_optimization_2d, gunther17_optimization_2d
    rk = killing_optimization_2d(field)
    rg = gunther17_optimization_2d(field, neighborhood=neighborhood)
    observers = {"killing": rk.u_field, "gunther17": rg.u_field}
    if title is None:
        title = "Reference-frame observer recovery (streamplot, speed-colored)"
    fig = plot_observer_recovery(field, observers, reference=reference, timesteps=timesteps,
                                 save_path=save_path, title=title, show=show)
    if rk.params is not None and len(rk.params) > 2:
        print(f"[ReferenceFrameViz] killing params mean over t = {np.round(rk.params[1:-1].mean(0), 4)}")
    print(f"[ReferenceFrameViz] observer temporal-variation (0=constant):  "
          f"killing={observer_temporal_variation(rk.u_field):.4f}  "
          f"gunther17={observer_temporal_variation(rg.u_field):.4f}")
    return fig, {"killing": rk, "gunther17": rg}


def plot_decomposition(dec, cuts=(2, 3, 4, 6), timestep=None, save_path=None, show=False,
                       wide_aspect=1.8):
    """可视化区域划分（referenceFrameDecompose 的结果）：|v| 底图 + 每个 cut 的分区叠加。

    **自适应长宽比**：按场的 X/Y 决定布局并用 aspect="equal" 保持像素比例——
    宽场（X/Y ≥ wide_aspect，如 cylinder）panel 竖排；方/竖场（rfc/boussinesq）panel 横排。
    避免把方形/竖直场横向拉伸压扁。

    Args:
        dec:          ReferenceFrameDecomposition（有 .field/.cut/.region_residuals/.diag）。
        cuts:         要展示的区域数列表。
        timestep:     底图 |v| 的时间片（默认中间帧）。
        save_path:    给了就保存 png。
        wide_aspect:  X/Y ≥ 此值视为宽场（panel 竖排），否则方/竖场（panel 横排）。
    Returns:
        matplotlib Figure。
    """
    import matplotlib
    if save_path is not None and not show:
        try:
            matplotlib.use("Agg")
        except Exception:
            pass
    import matplotlib.pyplot as plt

    data = _to_txy2(dec.field)                          # (T,Y,X,2)
    T, Y, X, _ = data.shape
    ti = T // 2 if timestep is None else int(timestep)
    speed = np.hypot(data[ti, :, :, 0], data[ti, :, :, 1])

    panels = [("|v| (t=%d)" % ti, None)] + [("cut n=%d" % n, n) for n in cuts]
    npn = len(panels)

    # 布局方向 + figsize：panel 保持真实 X:Y 比例（aspect="equal"），总图大小受控
    if X / max(Y, 1) >= wide_aspect:                    # 宽场 -> panel 竖排
        nrows, ncols = npn, 1
        total_h = min(14.0, 2.4 * npn)
        ph = total_h / npn; pw = ph * X / max(Y, 1)
        figsize = (min(pw, 14.0) + 0.4, total_h + 0.7)
    else:                                              # 方/竖场 -> panel 横排
        nrows, ncols = 1, npn
        total_w = min(16.0, 3.6 * npn)
        pw = total_w / npn; ph = pw * Y / max(X, 1)
        figsize = (total_w + 0.3, min(ph, 12.0) + 0.9)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, (title, n) in zip(axes, panels):
        if n is None:
            ax.imshow(speed, origin="lower", cmap="viridis", aspect="equal")
        else:
            labels = dec.cut(n_regions=n)
            nreg = len(np.unique(labels))
            totres = sum(dec.region_residuals(labels).values())
            ax.imshow(speed, origin="lower", cmap="gray", aspect="equal")
            ax.imshow(labels, origin="lower", cmap="tab10", alpha=0.5, aspect="equal", vmin=0, vmax=9)
            title = f"cut n={n}  (regions={nreg})  res={totres:.2g}"
        ax.set_title(title, fontsize=8); ax.set_xticks([]); ax.set_yticks([])

    d = dec.diag
    fig.suptitle(f"reference-frame decomposition  |  benefit={d['decomposition_benefit']:.3f}  "
                 f"global_res_ratio={d['global_residual_ratio']:.3f}\n{d['interpretation']}", fontsize=9)
    if save_path is not None:
        fig.savefig(save_path, dpi=130)
        print(f"[ReferenceFrameViz] saved {save_path}")
    if show:
        plt.show()
    return fig


__all__ = ["plot_observer_recovery", "observer_temporal_variation", "quick_compare_2d",
           "plot_decomposition"]
