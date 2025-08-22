import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


import torch.nn.functional as F

def compute_curvature_2d(pathline):
    """
    计算一条2D pathline的逐点曲率
    pathline: ndarray, shape [steps, 3]，最后一维为(x, y, t)
    返回:
        curvature: shape [steps]，逐点曲率
    """
    x = pathline[:, 0]
    y = pathline[:, 1]
    t = pathline[:, 2]

    dt = np.mean(np.diff(t))  # 假设时间步均匀

    # 一阶导 (速度)
    dx = np.gradient(x, dt)
    dy = np.gradient(y, dt)

    # 二阶导 (加速度)
    ddx = np.gradient(dx, dt)
    ddy = np.gradient(dy, dt)

    # 曲率公式
    numerator = np.abs(dx * ddy - dy * ddx)
    denominator = (dx**2 + dy**2) ** 1.5
    curvature = np.zeros_like(numerator)
    mask = denominator > 1e-8
    curvature[mask] = numerator[mask] / denominator[mask]

    return curvature

def visualize_curvature(pathline, curvature):
    """
    可视化 pathline 及其曲率
    """
    x = pathline[:, :,0]
    y = pathline[:,:, 1]
    t = pathline[:,:, 2]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左图：轨迹，颜色表示曲率大小
    sc = axes[0].scatter(x, y, c=curvature, cmap='viridis')
    axes[0].plot(x, y, 'k--', alpha=0.5)
    axes[0].set_title("Pathline with curvature coloring")
    plt.colorbar(sc, ax=axes[0], label="Curvature")

    # 右图：曲率随时间的变化
    axes[1].plot(t, curvature, '-o')
    axes[1].set_title("Curvature vs Time")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("Curvature")

    plt.tight_layout()
    plt.show()





def _first_second_derivative_nonuniform(y: torch.Tensor, t: torch.Tensor):
    """
    y, t: [B, S]
    返回一阶导 y', 二阶导 y''（中心差分，非均匀时间步；边界复制邻近值）
    """
    # 邻近时间步
    dt_f = t[:, 2:] - t[:, 1:-1]    # forward   [B, S-2]
    dt_b = t[:, 1:-1] - t[:, :-2]   # backward  [B, S-2]

    # 前/后向斜率
    v_f = (y[:, 2:]   - y[:, 1:-1]) / dt_f
    v_b = (y[:, 1:-1] - y[:, :-2])  / dt_b

    # 一阶导：加权平均（非均匀栅格的中心差分）
    y1 = (dt_b * v_f + dt_f * v_b) / (dt_f + dt_b)              # [B, S-2]
    # 二阶导：斜率差的中心差分
    y2 = 2.0 * (v_f - v_b) / (dt_f + dt_b)                      # [B, S-2]

    # 边界：复制邻近内点（避免爆点）
    y1 = torch.cat([y1[:, :1], y1, y1[:, -1:]], dim=1)          # [B, S]
    y2 = torch.cat([y2[:, :1], y2, y2[:, -1:]], dim=1)          # [B, S]
    return y1, y2


def compute_curvature_batch(pathlines: torch.Tensor, eps: float = 1e-12):
    """
    计算 batch pathlines 的逐点曲率 (2D 曲率)
    pathlines: [B, S, 3] 或 [B, 1, S, 3]，最后一维为 (x, y, t)
    返回:
        curvature: [B, S]
    """
    # 兼容 [B,1,S,3]
    if pathlines.ndim == 4:
        pathlines = pathlines[:, 0, :, :]  # -> [B, S, 3]

    # 用 double 提升数值稳定性
    # pathlines = pathlines.to(torch.float64)

    x = pathlines[..., 0]   # [B, S]
    y = pathlines[..., 1]   # [B, S]
    t = pathlines[..., 2]   # [B, S]

    # 一阶/二阶导（对非均匀 t）
    dx, ddx = _first_second_derivative_nonuniform(x, t)
    dy, ddy = _first_second_derivative_nonuniform(y, t)

    # 曲率 k = |x' y'' - y' x''| / ( (x'^2 + y'^2)^(3/2) )
    num = torch.abs(dx * ddy - dy * ddx)
    den = (dx * dx + dy * dy).clamp_min(eps).pow(1.5)
    kappa = num / den
    return kappa.to(pathlines.dtype)

def visualize_one_pathline(pathline: torch.Tensor, curvature: torch.Tensor, idx: int = 0):
    """
    可视化 batch 中某一条 pathline 的轨迹与曲率
    pathline: [B, steps, 3]
    curvature: [B, steps]
    idx: 要可视化的轨迹索引
    """
    x = pathline[idx, :, 0].cpu().numpy()
    y = pathline[idx, :, 1].cpu().numpy()
    t = pathline[idx, :, 2].cpu().numpy()
    c = curvature[idx].cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左图：轨迹，颜色显示曲率
    sc = axes[0].scatter(x, y, c=c, cmap='viridis')
    axes[0].plot(x, y, 'k--', alpha=0.5)
    axes[0].set_title("Pathline with curvature coloring")
    plt.colorbar(sc, ax=axes[0], label="Curvature")

    # 右图：曲率随时间的变化
    axes[1].plot(t, c, '-o')
    axes[1].set_title("Curvature vs Time")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("Curvature")

    # 关闭 offset，让坐标轴直接显示 1.xxx
    axes[1].ticklabel_format(useOffset=False, style='plain')
    axes[1].yaxis.set_major_formatter(FormatStrFormatter('%.6f'))
    
    plt.tight_layout()
    plt.show()

import math
# ====== 解析真值曲率 ======
def true_kappa_circle(R, S):
    return torch.full((1, S), 1.0 / R, dtype=torch.float64)

def true_kappa_ellipse(a, b, t):  # t:[1,S]
    denom = (a*a*(torch.sin(t)**2) + b*b*(torch.cos(t)**2))**1.5
    return (a*b) / denom

def true_kappa_parabola(a, x):    # y = a x^2, x:[1,S]
    return (2*abs(a)) / (1 + (2*a*x)**2)**1.5

def true_kappa_sine(A, w, x):     # y = A sin(w x), x:[1,S]
    num  = abs(A) * (w**2) * torch.abs(torch.sin(w*x))
    denom = (1 + (A*w*torch.cos(w*x))**2)**1.5
    return num / denom

def true_kappa_line(S):           # 直线曲率=0
    return torch.zeros((1, S), dtype=torch.float64)

# ====== 构造测试曲线，输出 batch:[B,S,3]；解析曲率:[B,S] ======
def make_tests(S=200, nonuniform=False, dtype=torch.float64, device="cpu"):
    batch = []
    truths = []
    names  = []

    # 0) 圆 R=1
    t = torch.linspace(0.0, 2*math.pi, steps=S+1, dtype=dtype, device=device)[:-1].unsqueeze(0)  # [1,S]
    if nonuniform:
        # 给 param t 加一点非均匀扰动（仍单调）
        dt = torch.diff(t, dim=1)
        dt = dt * (1.0 + 0.2*torch.randn_like(dt))
        t  = torch.cat([t[:, :1], t[:, :1] + torch.cumsum(dt, dim=1)], dim=1)
    R = 1.0
    x = R*torch.cos(t); y = R*torch.sin(t)
    batch.append(torch.stack([x,y,t], dim=-1))                     # [1,S,3]
    truths.append(true_kappa_circle(R, S))
    names.append("circle(R=1)")

    # 1) 椭圆 a=2,b=1
    t = torch.linspace(0.0, 2*math.pi, steps=S+1, dtype=dtype, device=device)[:-1].unsqueeze(0)
    if nonuniform:
        dt = torch.diff(t, dim=1)
        dt = dt * (1.0 + 0.2*torch.randn_like(dt))
        t  = torch.cat([t[:, :1], t[:, :1] + torch.cumsum(dt, dim=1)], dim=1)
    a, b = 2.0, 1.0
    x = a*torch.cos(t); y = b*torch.sin(t)
    batch.append(torch.stack([x,y,t], dim=-1))
    truths.append(true_kappa_ellipse(a,b,t))
    names.append("ellipse(a=2,b=1)")

    # 2) 抛物线 y = a x^2, x∈[-2,2]
    x = torch.linspace(-2.0, 2.0, steps=S, dtype=dtype, device=device).unsqueeze(0)
    if nonuniform:
        dx = torch.diff(x, dim=1)
        dx = dx * (1.0 + 0.2*torch.randn_like(dx))
        x  = torch.cat([x[:, :1], x[:, :1] + torch.cumsum(dx, dim=1)], dim=1)
    a = 0.5
    y = a * x**2
    t = x.clone()   # 这里把 t 当作 x
    batch.append(torch.stack([x,y,t], dim=-1))
    truths.append(true_kappa_parabola(a, x))
    names.append("parabola(y=0.5 x^2)")

    # 3) 正弦波 y = A sin(w x), x∈[0,2π]
    x = torch.linspace(0.0, 2*math.pi, steps=S, dtype=dtype, device=device).unsqueeze(0)
    if nonuniform:
        dx = torch.diff(x, dim=1)
        dx = dx * (1.0 + 0.2*torch.randn_like(dx))
        x  = torch.cat([x[:, :1], x[:, :1] + torch.cumsum(dx, dim=1)], dim=1)
    A, w = 0.8, 2.0
    y = A*torch.sin(w*x)
    t = x.clone()
    batch.append(torch.stack([x,y,t], dim=-1))
    truths.append(true_kappa_sine(A,w,x))
    names.append("sine(y=0.8 sin(2x))")

    # 4) 直线 y = m x + c
    x = torch.linspace(-1.0, 1.0, steps=S, dtype=dtype, device=device).unsqueeze(0)
    if nonuniform:
        dx = torch.diff(x, dim=1)
        dx = dx * (1.0 + 0.2*torch.randn_like(dx))
        x  = torch.cat([x[:, :1], x[:, :1] + torch.cumsum(dx, dim=1)], dim=1)
    m, c = 0.3, -0.2
    y = m*x + c
    t = x.clone()
    batch.append(torch.stack([x,y,t], dim=-1))
    truths.append(true_kappa_line(S))
    names.append("line(y=0.3x-0.2)")

    batch  = torch.cat(batch,  dim=0)  # [B,S,3]
    truths = torch.cat(truths, dim=0)  # [B,S]
    return batch, truths, names

# ====== 跑测试并可视化 ======
def run_curvature_tests(S=200, nonuniform=False, device="cpu", visualize=True):
    batch, truths, names = make_tests(S=S, nonuniform=nonuniform, device=device)
    # 用 float64 计算再转回
    curv = compute_curvature_batch(batch.to(torch.float64)).to(batch.dtype)

    # 指标
    with torch.no_grad():
        mae = torch.mean(torch.abs(curv.to(torch.float64) - truths), dim=1)
        maxe = torch.max(torch.abs(curv.to(torch.float64) - truths), dim=1).values

    print("\n=== Curvature test results ===")
    for i, name in enumerate(names):
        print(f"{i}. {name:22s} | MAE={mae[i].item():.6e}  MaxE={maxe[i].item():.6e}")

    if visualize:
        # 任选几条画图（使用你已有的可视化函数）
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FormatStrFormatter
        for i in range(len(names)):
            print(f"\nVisualizing: {names[i]}")
            visualize_one_pathline(batch, curv, idx=i)
    return batch, curv, truths, names

# ====== 直接运行 ======
if __name__ == "__main__":
    run_curvature_tests(S=128, nonuniform=False, device="cpu", visualize=True)
    # 也可以再测非均匀采样：
    # run_curvature_tests(S=128, nonuniform=True, device="cpu", visualize=True)
