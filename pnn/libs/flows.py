from FLowUtils.VectorField2d import *
from FLowUtils.flowDatasetUtils.NetCDF_AmiraLoader import *
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors, cm
import numpy as np
from tqdm import trange, tqdm
import torch
import torch.nn.functional as F

def to_array_if_valid(lst, expected_shape=None):
    ok=0# should be 4
    for line in lst:
        # logger.info(f"len(line):{len(line)},expected_shape[1]:{expected_shape[1]}")
        if len(line)==expected_shape[1]:# all have the max steps points
            
            ok+=1
    
    if ok==expected_shape[0]:
        try:
            arr = np.array(lst)
            # logger.info(f"succeeded.")
        except ValueError:
            # logger.info(f"Ignore list..")
            # ragged（不规则）列表无法转换
            return None
    else:
        return None

    return arr

def compute_streamline_euler(
    steady_field,       # numpy array of shape (H, W, 2)
    start_pos,          # (x, y) in physical space
    domain_min,         # [xmin, ymin]
    domain_max,         # [xmax, ymax]
    Xdim, Ydim,
    step_size=0.01,
    max_steps=1000,
    normalized=False,
    localized=False
):
    """
    计算 streamline：在固定时刻的速度场中进行欧拉积分。
    """
    def is_inside_domain(x, y):
        return (domain_min[0] <= x <= domain_max[0]) and (domain_min[1] <= y <= domain_max[1])
    
    def get_vector_at(x_phys, y_phys):
        # 坐标映射到 grid space
        x_grid = (x_phys - domain_min[0]) / (domain_max[0] - domain_min[0]) * (Xdim - 1)
        y_grid = (y_phys - domain_min[1]) / (domain_max[1] - domain_min[1]) * (Ydim - 1)
        return bilinear_interpolate(steady_field, x_grid, y_grid)

    original=(0.0,0.0)
    loc_points=[original]
    
    nor_start_points=((start_pos[0]-domain_min[0])/(domain_max[0]-domain_min[0]),(start_pos[1]-domain_min[1])/(domain_max[1]-domain_min[1]))
    points = [start_pos]
    x, y = start_pos
    
    # ignore the start pos where the velocity is zero
    vx0, vy0 = get_vector_at(x, y)
    if vx0==0.0 and vy0==0.0:
        points_dict={
            'points':points,
            'loc_points':loc_points
        }
        return points_dict
    
    for _ in range(max_steps):
        if not is_inside_domain(x, y):
            break
        vx, vy = get_vector_at(x, y)
        x += vx * step_size
        y += vy * step_size
        
        cur_x=x
        cur_y=y
        
        if normalized:#
            cur_x=(cur_x-domain_min[0])/(domain_max[0]-domain_min[0])
            cur_y=(cur_y-domain_min[1])/(domain_max[1]-domain_min[1])
            if localized:
                cur_x=cur_x-nor_start_points[0]
                cur_y=cur_y-nor_start_points[1]
            # logger.info(f"localized: x:{x},y:{y}")
        points.append((x, y))
        loc_points.append((cur_x,cur_y))
    points_dict={
        'points':points,
        'loc_points':loc_points
    }
    return points_dict

def compute_multiple_streamlines(
    steady_field,       # numpy array of shape (H, W, 2)
    start_pos,          # (x, y) in physical space
    domain_min,         # [xmin, ymin]
    domain_max,         # [xmax, ymax]
    Xdim, Ydim,
    step_size=0.01,
    max_steps=1000,
    offset_dist: float = 0.5,
    normalized=False,
    localized=False
):
    """
    计算 streamline：在固定时刻的速度场中进行欧拉积分。
    """
    def is_inside_domain(x, y):
        return (domain_min[0] <= x <= domain_max[0]) and (domain_min[1] <= y <= domain_max[1])
    
    def get_vector_at(x_phys, y_phys):
        # 坐标映射到 grid space
        x_grid = (x_phys - domain_min[0]) / (domain_max[0] - domain_min[0]) * (Xdim - 1)
        y_grid = (y_phys - domain_min[1]) / (domain_max[1] - domain_min[1]) * (Ydim - 1)
        return bilinear_interpolate(steady_field, x_grid, y_grid)

    """
    Compute streamlines from five starting points:
      - original start_pos
      - ±offset_dist along x
      - ±offset_dist along y
    """
    # 五个偏移量：原点，x±，y±
    offsets = [
        (0.0, 0.0),
        ( offset_dist, 0.0),
        (-offset_dist, 0.0),
        (0.0,  offset_dist),
        (0.0, -offset_dist),
    ]

    streamlines = []
    loc_streamlines=[]
    for dx, dy in offsets:
        sp = (start_pos[0] + dx, start_pos[1] + dy)
        pts = compute_streamline_euler(
            steady_field,
            start_pos=sp,
            domain_min=domain_min,
            domain_max=domain_max,
            Xdim=Xdim,
            Ydim=Ydim,
            step_size=step_size,
            max_steps=max_steps,
            normalized=normalized,
            localized=localized
        )
        streamlines.append(pts['points'])
        loc_streamlines.append(pts['loc_points'])

    streamlines_dict={
        'streamlines':streamlines,
        'loc_streamlines':loc_streamlines
    }
    return streamlines_dict

def visualize_multiple_line(field: UnsteadyVectorField2D, streamline_point_list, time_index=0):
    slice_data = field.getSlice(time_index).field  # shape: (H, W, 2)
    X, Y = np.meshgrid(
        np.linspace(field.domainMinBoundary[0], field.domainMaxBoundary[0], field.Xdim),
        np.linspace(field.domainMinBoundary[1], field.domainMaxBoundary[1], field.Ydim)
    )
    U = slice_data[:, :, 0]
    V = slice_data[:, :, 1]

    # 下采样箭头展示
    step = 5
    X_down = X[::step, ::step]
    Y_down = Y[::step, ::step]
    U_down = U[::step, ::step]
    V_down = V[::step, ::step]

    plt.figure(figsize=(8, 6))
    plt.quiver(X_down, Y_down, U_down, V_down, color='gray', alpha=0.5, scale=100)

    
    for i in range(len(streamline_point_list)):
        streamline_points = np.array(streamline_point_list[i])
        plt.plot(streamline_points[:, 0], streamline_points[:, 1], 'b-', linewidth=1)
        # plt.scatter(streamline_points[0, 0], streamline_points[0, 1], c='green', label='Start')
        # plt.scatter(streamline_points[-1, 0], streamline_points[-1, 1], c='blue', label='End')

    plt.legend()
    plt.title(f"Streamline at t = {field.getTime(time_index):.2f}")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

def visualize_streamlines_labels(field: UnsteadyVectorField2D, streamline_point, labes,time_index=0,nerbos=5,max_step=100,scale=50,title="Test"):
    
    # streamline_point： tensor in cpu() Nx505x3
    # labes: N,
    
    slice_data = field.getSlice(time_index).field  # shape: (H, W, 2)
    X, Y = np.meshgrid(
        np.linspace(field.domainMinBoundary[0], field.domainMaxBoundary[0], field.Xdim),
        np.linspace(field.domainMinBoundary[1], field.domainMaxBoundary[1], field.Ydim)
    )
    U = slice_data[:, :, 0]
    V = slice_data[:, :, 1]

    # 下采样箭头展示
    step = 5
    X_down = X[::step, ::step]
    Y_down = Y[::step, ::step]
    U_down = U[::step, ::step]
    V_down = V[::step, ::step]

    plt.figure(figsize=(8, 6))
    # the laeger scale, the smaller arrow
    plt.quiver(X_down, Y_down, U_down, V_down, color='gray', alpha=0.5, scale=scale)
    
    N,pN,_=streamline_point.shape
    colors = {-1:'pink',0: 'red', 1: 'g',2:'b',3:'y'}
    for i in range(N):
        streamline=streamline_point[i,:,0:2].numpy()# Nx2
        # print(f"labes({i}):{labes[i]},streamline:{streamline.shape}")
        for j in range(5):
            # logger.info(f"colors[labes[i]]；{colors[labes[i]]}")
            plt.plot(streamline[0+j*(max_step+1):(j+1)*(max_step+1)-1,0 ], streamline[0+j*(max_step+1):(j+1)*(max_step+1)-1,1 ], colors[labes[i]], linewidth=1)
    
    plt.legend()
    plt.title(f"{title}")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

def multi_lines_vis_fast(
    field, streamline_points_list, labels_list,
    time_index=0, nerbos=5, max_step=100, scale=50,
    title=("Test",), layout=(1,2),
    stride_steps=1,          # 轨迹点下采样步长（>1 可进一步提速）
    max_quiver=64,           # 每个维度的最大箭头数（控制稀疏 quiver）
    show_quiver=True,        # 是否叠加稀疏 quiver
    cmap_bg="binary",         # 背景 colormap
):
    """
    高效多轨迹可视化：
    - 所有轨迹用 LineCollection 一次性绘制
    - 背景用 imshow 显示速度幅值（可选叠加稀疏 quiver）
    - 支持对轨迹点做 stride 下采样
    """
    rows, cols = layout
    fig, axes = plt.subplots(rows, cols, figsize=(16*cols, 6*(rows+0.5)),constrained_layout=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.ravel()

    nfig = min(len(streamline_points_list), len(axes))
    xmin, ymin = field.domainMinBoundary[0:2]
    xmax, ymax = field.domainMaxBoundary[0:2]

    # 背景网格（一次性创建）
    XX, YY = np.meshgrid(
        np.linspace(xmin, xmax, field.Xdim),
        np.linspace(ymin, ymax, field.Ydim)
    )
    ori_nerbos=nerbos
    
    # === 背景：幅值图像（更快） ===
    slice_data = field.getSlice(time_index).field  # (H, W, 2)
    speed = np.hypot(slice_data[...,0], slice_data[...,1])
    
    # 稳健的颜色范围（去极端值）
    vmin, vmax = np.nanpercentile(speed, (2, 98))
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    # 统一 colormap 对象（可以是字符串或 plt.get_cmap(...)）
    cmap = plt.get_cmap(cmap_bg)
    # 用于共享 colorbar 的 mappable
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # 一些版本需要
    for fi in range(nfig):
        ax = axes[fi]
        sl_points = streamline_points_list[fi]   # 形如 [N, P, D]
        labs      = np.asarray(labels_list[fi])  # [N]

        nerbos=ori_nerbos
        if sl_points.shape[1]!=nerbos*(max_step+1):
            nerbos=4
            # logger.error(f"set nerbos as {nerbos}.")
        
        
        ax.imshow(
            speed, origin='lower', cmap=cmap_bg, alpha=1, interpolation='nearest',
            extent=[xmin, xmax, ymin, ymax], rasterized=True
        )

        # 可选：稀疏 quiver（只画很少的箭头）
        if show_quiver:
            m = np.nanpercentile(speed, 95)          # 典型最大速度
            L = 0.06 * (xmax - xmin)                 # 希望典型箭头长度=域宽的 ~6%
            scale = m / L   * 10
            # logger.info(f"scale:{scale}")
            MAX=field.Xdim if field.Xdim>field.Ydim else field.Ydim
            x_max_quiver=int(max_quiver*field.Xdim/MAX)
            y_max_quiver=int(max_quiver*field.Ydim/MAX)
            sx = max(1, field.Xdim // x_max_quiver)
            sy = max(1, field.Ydim // y_max_quiver)
            U = slice_data[::sy, ::sx, 0]
            V = slice_data[::sy, ::sx, 1]
            ax.quiver(
                XX[::sy, ::sx], YY[::sy, ::sx], U, V,
                angles='xy', scale_units='xy', scale=scale,
                pivot='mid',
                headwidth=5, headlength=10, headaxislength=10,  # 小巧的箭头头部（单位：points）
                linewidth=0.5,                  # 细一点的箭杆
                color='grey', alpha=0.5
            )

        # === 轨迹数据准备 ===
        # torch -> numpy（一次性）
        if isinstance(sl_points, torch.Tensor):
            sl_points = sl_points.detach().cpu().numpy()

        # sl_points: [N, P, D]，其中 P 期望是 nerbos*(max_step+1)
        N, P, D = sl_points.shape
        # 你的原判断里写成了 streamline_point.shape[0]，应为第二维 P：
        expected_P = nerbos * (max_step + 1)
        if P != expected_P:
            # 也支持另一种组织（每条线已经是 (max_step+1)），就尝试自动推断
            # 若你的输入本就为 [N*nerbos, max_step+1, D]，那下面这步可以跳过
            pass

        # 把 [N, nerbos*(max_step+1), D] → [N*nerbos, max_step+1, D]
        lines = sl_points.reshape(N, nerbos, max_step+1, D).reshape(N*nerbos, max_step+1, D)

        # 轨迹点下采样（提速 + 减少过密绘制）
        if stride_steps > 1:
            lines = lines[:, ::stride_steps, :]

        # === LineCollection 一次性绘制 ===
        segs = lines[:, :, :2]  # [L, Lp, 2]，每行是一个 polyline
        lc = LineCollection(segs, linewidths=1.0, antialiased=False, rasterized=True)

        # 按标签上色
        # 颜色映射（按需改）
        # cmap_label = {-1:'pink', 0:'r', 1:'g', 2:'b', 3:'y'}
        cmap_label = {
                -1: 'pink',
                0: '#1f77b4',  1: '#aec7e8',
                2: '#ff7f0e',  3: '#ffbb78',
                4: '#2ca02c',  5: '#98df8a',
                6: '#d62728',  7: '#ff9896',
                8: '#9467bd',  9: '#c5b0d5',
                10: '#8c564b', 11: '#c49c94',
                12: '#e377c2', 13: '#f7b6d2',
                14: '#7f7f7f', 15: '#c7c7c7',
                16: '#bcbd22', 17: '#dbdb8d',
                18: '#17becf', 19: '#9edae5',
            }
        line_colors = np.array([cmap_label.get(int(lbl), 'k') for lbl in np.repeat(labs, nerbos)])
        lc.set_colors(line_colors)

        ax.add_collection(lc)

        # === 外观设置 ===
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        if fi < len(title):
            ax.set_title(title[fi])
        ax.set_xlabel('X'); ax.set_ylabel('Y')
        ax.set_aspect('equal')
        ax.grid(False)  # 关掉网格加速

    # 隐藏多余子图
    for j in range(nfig, len(axes)):
        axes[j].axis('off')

    # 全局性能优化的一些 rc 设置（可选）
    import matplotlib as mpl
    mpl.rcParams['path.simplify'] = True
    mpl.rcParams['path.simplify_threshold'] = 0.1
    mpl.rcParams['agg.path.chunksize'] = 10000  # 长折线分块渲染
    
    # ---- 循环结束后：添加“共享”颜色条 ----
    cbar = fig.colorbar(sm, ax=axes, location='right', pad=0.01, fraction=0.03)
    cbar.set_label('|v|', rotation=270, labelpad=12)
    # 可选：在两端标注
    cbar.ax.text(0.5, -0.02, 'min', ha='center', va='top', transform=cbar.ax.transAxes)
    cbar.ax.text(0.5,  1.02, 'max', ha='center', va='bottom', transform=cbar.ax.transAxes)
    
    
    # plt.tight_layout()
    plt.show()
    
    
def multi_points_vis_fast(
    field, streamline_points_list, labels_list,
    time_index=None,time=None, nerbos=5, max_step=100, scale=50,
    title=("Test",), layout=(1,2),
    max_quiver=64, show_quiver=True,
    cmap_bg="binary",
    pick="all_starts",          # 'seed' | 'all_starts' | ('line', j) | ('step', k)
    point_size=50 # DL: point size, larger make bigger points
):
    """
    高效“点”可视化（一次性scatter）：
      - 背景：共享一张幅值图 + 共享 colorbar
      - 取点模式：
          'seed'       : 每组只取第0条线的起点（常作为采样点）
          'all_starts' : 每组的所有起点（nerbos或4个）
          ('line', j)  : 每组取第 j 条线的起点
          ('step', k)  : 每条线取第 k 个时间步位置（0..max_step），组内全部线
    """

    if time_index is not None:
        slice_data = field.getSlice(time_index).field
    elif time is not None:
        slice_data = field.getSliceAtPhysicalTime(time).field  # (H,W,2)
    else:
        raise ValueError("time_index or time must be provided")

    rows, cols = layout
    fig, axes = plt.subplots(rows, cols, figsize=(16*cols, 6*(rows+0.5)),
                             constrained_layout=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.ravel()
    nfig = min(len(streamline_points_list), len(axes))

    xmin, ymin = field.domainMinBoundary[0:2]
    xmax, ymax = field.domainMaxBoundary[0:2]

    # 背景与归一化（一次）
    speed = np.hypot(slice_data[...,0], slice_data[...,1])
    vmin, vmax = np.nanpercentile(speed, (2, 98))
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_bg)

    # 半像素扩展，避免竖条缝
    dx = (xmax - xmin) / field.Xdim
    dy = (ymax - ymin) / field.Ydim
    extent = [xmin - 0.5*dx, xmax + 0.5*dx, ymin - 0.5*dy, ymax + 0.5*dy]

    # 共享 colorbar 的 mappable
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    # 稀疏 quiver 预计算
    if show_quiver:
        MAX = max(field.Xdim, field.Ydim)
        x_max_quiver = int(max_quiver*field.Xdim/MAX)
        y_max_quiver = int(max_quiver*field.Ydim/MAX)
        sx = max(1, field.Xdim // x_max_quiver)
        sy = max(1, field.Ydim // y_max_quiver)
        XX, YY = np.meshgrid(
            np.linspace(xmin, xmax, field.Xdim),
            np.linspace(ymin, ymax, field.Ydim)
        )
        U = slice_data[::sy, ::sx, 0]
        V = slice_data[::sy, ::sx, 1]
        # 自适应箭头尺度
        m = np.nanpercentile(np.hypot(U, V), 95)
        L = 0.06 * (xmax - xmin)
        q_scale = (m / L) * 10 if m > 0 else 1.0

    # 颜色映射
    # cmap_label = {-1:'pink', 0:'r', 1:'g', 2:'b', 3:'y'}
    cmap_label = {
    -1: 'pink',
     0: '#1f77b4',  1: '#aec7e8',
     2: '#ff7f0e',  3: '#ffbb78',
     4: '#2ca02c',  5: '#98df8a',
     6: '#d62728',  7: '#ff9896',
     8: '#9467bd',  9: '#c5b0d5',
    10: '#8c564b', 11: '#c49c94',
    12: '#e377c2', 13: '#f7b6d2',
    14: '#7f7f7f', 15: '#c7c7c7',
    16: '#bcbd22', 17: '#dbdb8d',
    18: '#17becf', 19: '#9edae5',
}

    for fi in range(nfig):
        ax = axes[fi]
        sl_points = streamline_points_list[fi]   # [N, P, D]
        labs      = np.asarray(labels_list[fi])  # [N]

        if isinstance(sl_points, torch.Tensor):
            sl_points = sl_points.detach().cpu().numpy()
        pts,N,k,L,D = None,None,None,None,None
        if sl_points.ndim == 3:
            N, P, D = sl_points.shape
            L = max_step + 1
            # 推断每组线数 k（可能是 5 或 4 或 1）
            if P == L:
                k = 1
            elif P % L == 0:
                k = P // L
            else:
                raise ValueError(f"P={P} not divisible by L={L}")
            pts = sl_points.reshape(N, k, L, D)
        elif sl_points.ndim == 4:
            N, k, L, D = sl_points.shape
            pts = sl_points
         

        # 背景
        ax.imshow(speed, origin='lower', cmap=cmap, norm=norm,
                  interpolation='none', extent=extent, alpha=1.0, rasterized=True)

        if show_quiver:
            ax.quiver(XX[::sy, ::sx], YY[::sy, ::sx], U, V,
                      angles='xy', scale_units='xy', scale=q_scale, pivot='mid',
                      headwidth=3, headlength=4, headaxislength=3,
                      linewidth=0.5, color='grey', alpha=0.5)

        # ---- 选点逻辑（一次性取出并扁平化） ----
       

        if pick == 'seed':
            # 每组 第0条线 的 起点
            sel = pts[:, 0, 0, :2]                  # [N,2]
            color_list = [cmap_label.get(int(l), 'k') for l in labs]
        elif pick == 'all_starts':
            # 每组 所有线 的 起点
            sel = pts[:, :, 0, :2].reshape(-1, 2)   # [N*k,2]
            color_list = [cmap_label.get(int(l), 'k') for l in np.repeat(labs, k)]
        elif isinstance(pick, tuple) and pick[0] == 'line':
            j = int(pick[1])                        # 指定第 j 条线的起点
            if not (0 <= j < k):
                raise ValueError(f"line index j={j} out of [0,{k-1}]")
            sel = pts[:, j, 0, :2]                  # [N,2]
            color_list = [cmap_label.get(int(l), 'k') for l in labs]
        elif isinstance(pick, tuple) and pick[0] == 'step':
            t_idx = int(pick[1])                    # 指定时间步（0..L-1）
            t_idx = max(0, min(L-1, t_idx))
            sel = pts[:, :, t_idx, :2].reshape(-1, 2)  # [N*k,2]
            color_list = [cmap_label.get(int(l), 'k') for l in np.repeat(labs, k)]
        else:
            raise ValueError(f"Unknown pick mode: {pick}")

        x = sel[:, 0].ravel()
        y = sel[:, 1].ravel()

        sc = ax.scatter(x, y, c=color_list, s=point_size, marker='o',
                        edgecolors='none', alpha=0.9)
        sc.set_rasterized(True)  # 海量点导出更快

        # 外观
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        if fi < len(title): ax.set_title(title[fi])
        ax.set_xlabel('X'); ax.set_ylabel('Y')
        ax.set_aspect('equal')
        ax.grid(False)

    # 隐藏多余子图
    for j in range(nfig, len(axes)):
        axes[j].axis('off')

    # 共享 colorbar（不与子图重叠）
    cbar = fig.colorbar(sm, ax=axes, location='right', pad=0.01, fraction=0.03, aspect=30)
    cbar.set_label('|v|', rotation=270, labelpad=12)

    plt.show()


    
def multi_lines_vis(field: UnsteadyVectorField2D, streamline_points:list, labes:list,time_index=0,nerbos=5,max_step=100,scale=100,title=["Test"],layout=[1,2]):
    """
    Function to draw multi lines
    :
        latout indicates the row and column
    
    """
    fig, axes = plt.subplots(layout[0], layout[1], figsize=(16*layout[1], 6*(layout[0]+2)))  # row and column
    Nfig=len(streamline_points)
    
    for fi in range(Nfig):
            # 画背景的速度场
        slice_data = field.getSlice(time_index).field  # (H, W, 2)
        X, Y = np.meshgrid(
            np.linspace(field.domainMinBoundary[0], field.domainMaxBoundary[0], field.Xdim),
            np.linspace(field.domainMinBoundary[1], field.domainMaxBoundary[1], field.Ydim)
        )
        U = slice_data[:, :, 0]
        V = slice_data[:, :, 1]

        step = 5
        axes[fi].quiver(
            X[::step, ::step], Y[::step, ::step],
            U[::step, ::step], V[::step, ::step],
            color='gray', alpha=0.5, scale=scale
        )
        streamline_point=streamline_points[fi]
        label=labes[fi]
        N,pN,_=streamline_point.shape
        colors = {-1:'pink',0: 'red', 1: 'g',2:'b',3:'y'}
        for i in range(N):
            streamline=streamline_point[i,:,0:2].numpy()# Nx2
            if streamline_point.shape[0]==nerbos*(max_step+1):# we have 5 lines
                for j in range(5):
                    axes[fi].plot(streamline[0+j*(max_step+1):(j+1)*(max_step+1)-1,0 ], streamline[0+j*(max_step+1):(j+1)*(max_step+1)-1,1 ], colors[label[i]], linewidth=1,alpha=0.5)
            else:# we have 4 lines
                for j in range(4):
                    axes[fi].plot(streamline[0+j*(max_step+1):(j+1)*(max_step+1)-1,0 ], streamline[0+j*(max_step+1):(j+1)*(max_step+1)-1,1 ], colors[label[i]], linewidth=1,alpha=0.5)
            
        axes[fi].set_title(title[fi])
        axes[fi].set_xlabel('X')
        axes[fi].set_ylabel('Y')
        axes[fi].set_aspect('equal')
        axes[fi].grid(True)
    plt.tight_layout()
    plt.show()

def multi_points_vis(field: UnsteadyVectorField2D, streamline_points:list, labes:list,time_index=0,nerbos=5,max_step=100,scale=100,title=["Test"],layout=[1,2]):
    """
    Function to draw multi lines
    :
        latout indicates the row and column
    
    """
    fig, axes = plt.subplots(layout[0], layout[1], figsize=(16*layout[1], 6*(layout[0]+2)))  # row and column
    Nfig=len(streamline_points)
    
    for fi in trange(Nfig):
            # 画背景的速度场
        slice_data = field.getSlice(time_index).field  # (H, W, 2)
        X, Y = np.meshgrid(
            np.linspace(field.domainMinBoundary[0], field.domainMaxBoundary[0], field.Xdim),
            np.linspace(field.domainMinBoundary[1], field.domainMaxBoundary[1], field.Ydim)
        )
        U = slice_data[:, :, 0]
        V = slice_data[:, :, 1]

        step = 5
        axes[fi].quiver(
            X[::step, ::step], Y[::step, ::step],
            U[::step, ::step], V[::step, ::step],
            color='gray', alpha=0.5, scale=scale
        )
        streamline_point=streamline_points[fi]
        label=labes[fi]
        N,pN,_=streamline_point.shape
        colors = {-1:'pink',0: 'red', 1: 'g',2:'b',3:'y'}
        # 决定这组数据里是 5 条线还是 4 条线
        n_lines = nerbos if pN == nerbos*(max_step+1) else (nerbos-1)
        
        # 遍历每条记录（可能是多个样本），只取第一个点
        pts_xy = streamline_point[:, :, 0:2].numpy()  # [N,pN, 2]
        pts_xy=pts_xy.reshape(N,n_lines,-1,2)
        x0=pts_xy[:,0,0:1,0] #N
        y0=pts_xy[:,0,0:1,1] #N
        # print(f"x0:{len(x0)}")
        # print(f"y0:{len(y0)}")

        # 3) 映射到颜色列表
        color_list = [colors[l] for l in label]#N
        # print(f"color_list:{len(color_list)}")
        axes[fi].scatter(x0, y0,
                        color=color_list,
                        s=50,        # 点的大小
                        alpha=0.8,
                        edgecolors='none',
                        linewidth=0.5)
                
        axes[fi].set_title(title[fi])
        axes[fi].set_xlabel('X')
        axes[fi].set_ylabel('Y')
        axes[fi].set_aspect('equal')
        axes[fi].grid(True)
    plt.tight_layout()
    plt.show()



## now we write the functions for pathlines
def compute_pathline_euler(
    field: UnsteadyVectorField2D,
    start_pos,               # (x, y) physical coordinates
    start_time,              # float
    target_time,             # float
    step_size=0.01,          # time step size
    max_steps=1000,
    normalized=False,
    localized=False
):
    """
    Compute a pathline using Euler integration in an unsteady vector field.

    Returns:
        points_dict with:
            - 'points': list of (x, y, t)
            - 'loc_points': list of normalized (x, y)
    """

    def is_inside_domain(x, y):
        return (field.domainMinBoundary[0] <= x <= field.domainMaxBoundary[0]) and \
               (field.domainMinBoundary[1] <= y <= field.domainMaxBoundary[1])

    def get_vector_at(x_phys, y_phys, time):
        x_grid = (x_phys - field.domainMinBoundary[0]) / (field.domainMaxBoundary[0] - field.domainMinBoundary[0]) * (field.Xdim - 1)
        y_grid = (y_phys - field.domainMinBoundary[1]) / (field.domainMaxBoundary[1] - field.domainMinBoundary[1]) * (field.Ydim - 1)
        t_index = int((time - field.tmin) / field.timeInterval)
        return field.getBilinearInterpolateVector(x_grid, y_grid, t_index)

    # Determine time step direction
    dt = step_size if target_time >= start_time else -step_size
    x, y = start_pos
    t = start_time

    # Normalized coordinate preparation
    nor_start = (
        (x - field.domainMinBoundary[0]) / (field.domainMaxBoundary[0] - field.domainMinBoundary[0]),
        (y - field.domainMinBoundary[1]) / (field.domainMaxBoundary[1] - field.domainMinBoundary[1])
    )

    points = [(x, y, t)]
    nor_time=(t-field.tmin)/(field.tmax-field.tmin)
    loc_points = [(0.0, 0.0,nor_time)]  # local path always starts from origin

    vx0, vy0 = get_vector_at(x, y,t)
    if vx0==0.0 and vy0==0.0 or vx0<=0.000001 and vy0<=0.000001 :
        points_dict={
            'points':points,
            'loc_points':loc_points
        }
        return points_dict
    
    for _ in range(max_steps):
        if (dt > 0 and t >= target_time) or (dt < 0 and t <= target_time):
            break
        if not is_inside_domain(x, y):
            break

        vx, vy = get_vector_at(x, y, t)

        # Euler integration
        x_new = x + vx * dt
        y_new = y + vy * dt
        t_new = t + dt

        # Normalization
        x_loc = x_new
        y_loc = y_new
        if normalized:
            x_loc = (x_new - field.domainMinBoundary[0]) / (field.domainMaxBoundary[0] - field.domainMinBoundary[0])
            y_loc = (y_new - field.domainMinBoundary[1]) / (field.domainMaxBoundary[1] - field.domainMinBoundary[1])
            if localized:
                x_loc -= nor_start[0]
                y_loc -= nor_start[1]

        points.append((x_new, y_new, t_new))
        
        nor_time=(t_new-field.tmin)/(field.tmax-field.tmin)
        loc_points.append((x_loc, y_loc,nor_time))

        # Update state
        x, y, t = x_new, y_new, t_new

    return {
        'points': points,       # (x, y, t)
        'loc_points': loc_points
    }


def compute_pathline_rk4(
    field: UnsteadyVectorField2D,
    start_pos,               # (x, y) physical coordinates
    start_time,              # float
    target_time,             # float
    step_size=0.01,          # time step size
    max_steps=1000,
    normalized=False,
    localized=False
):
    """
    Compute a pathline using 4th-order Runge–Kutta integration in an unsteady vector field.

    Returns:
        {
            'points':    list of (x, y, t),
            'loc_points': list of (x_loc, y_loc, t_norm)
        }
    """

    def is_inside_domain(x, y):
        return (field.domainMinBoundary[0] <= x <= field.domainMaxBoundary[0]) and \
               (field.domainMinBoundary[1] <= y <= field.domainMaxBoundary[1])

    def get_vector_at(x_phys, y_phys, time):
        # 物理坐标转格点坐标并插值
        xg = (x_phys - field.domainMinBoundary[0]) / (field.domainMaxBoundary[0] - field.domainMinBoundary[0]) * (field.Xdim - 1)
        yg = (y_phys - field.domainMinBoundary[1]) / (field.domainMaxBoundary[1] - field.domainMinBoundary[1]) * (field.Ydim - 1)
        t_idx = int((time - field.tmin) / field.timeInterval)
        return field.getBilinearInterpolateVector(xg, yg, t_idx)

    def rk4_step(x, y, t, dt):
        # k1
        vx1, vy1 = get_vector_at(x,      y,      t)
        # k2
        x2, y2, t2 = x + vx1*dt/2, y + vy1*dt/2, t + dt/2
        vx2, vy2 = get_vector_at(x2,     y2,     t2)
        # k3
        x3, y3, t3 = x + vx2*dt/2, y + vy2*dt/2, t + dt/2
        vx3, vy3 = get_vector_at(x3,     y3,     t3)
        # k4
        x4, y4, t4 = x + vx3*dt,   y + vy3*dt,   t + dt
        vx4, vy4 = get_vector_at(x4,     y4,     t4)

        dx = (vx1 + 2*vx2 + 2*vx3 + vx4) * dt / 6
        dy = (vy1 + 2*vy2 + 2*vy3 + vy4) * dt / 6
        return x + dx, y + dy

    # 设定时间步长方向
    dt = step_size if target_time >= start_time else -step_size
    x, y = start_pos
    t = start_time

    # 归一化与局部化初始值
    nor_start = (
        (x - field.domainMinBoundary[0]) / (field.domainMaxBoundary[0] - field.domainMinBoundary[0]),
        (y - field.domainMinBoundary[1]) / (field.domainMaxBoundary[1] - field.domainMinBoundary[1])
    )

    points = [(x, y, t)]
    t_norm = (t - field.tmin) / (field.tmax - field.tmin)
    loc_points = [(0.0, 0.0, t_norm)]

    # 如果初始速度几乎为零，直接返回
    vx0, vy0 = get_vector_at(x, y, t)
    if abs(vx0) < 1e-6 and abs(vy0) < 1e-6:
        return {'points': points, 'loc_points': loc_points}

    # 主循环
    for _ in range(max_steps):
        # 达到目标时间或出界时停止
        if (dt > 0 and t >= target_time) or (dt < 0 and t <= target_time):
            break
        if not is_inside_domain(x, y):
            break

        # RK4 计算下一步位置
        x_new, y_new = rk4_step(x, y, t, dt)
        t_new = t + dt

        # 归一化与局部化
        x_loc, y_loc = x_new, y_new
        if normalized:
            x_loc = (x_new - field.domainMinBoundary[0]) / (field.domainMaxBoundary[0] - field.domainMinBoundary[0])
            y_loc = (y_new - field.domainMinBoundary[1]) / (field.domainMaxBoundary[1] - field.domainMinBoundary[1])
            if localized:
                x_loc -= nor_start[0]
                y_loc -= nor_start[1]

        points.append((x_new, y_new, t_new))
        t_norm = (t_new - field.tmin) / (field.tmax - field.tmin)
        loc_points.append((x_loc, y_loc, t_norm))

        x, y, t = x_new, y_new, t_new

    return {
        'points': points,
        'loc_points': loc_points
    }


def compute_multiple_pathlines(
    steady_field,       # numpy array of shape (H, W, 2)
    start_pos,          # (x, y, t_start) in physical space
    target_time,
    step_size=0.01,
    max_steps=1000,
    offset_dist: float = 0.5,
    normalized=False,
    localized=False,
    line_method="Euler"
):
    """
    calculate streamlines
    """

    """
    Compute pathline from five starting points:
      - original start_pos
      - ±offset_dist along x
      - ±offset_dist along y
    """
    # 5 offset：origins，x±，y±
    offsets = [
        (0.0, 0.0),
        ( offset_dist, 0.0),
        (-offset_dist, 0.0),
        (0.0,  offset_dist),
        (0.0, -offset_dist),
    ]

    t_start=start_pos[2]
    streamlines = []
    loc_streamlines=[]
    for dx, dy in offsets:
        sp = (start_pos[0] + dx, start_pos[1] + dy)
        if line_method=="Euler":
            pts = compute_pathline_euler(
                steady_field,
                start_pos=sp,
                start_time=t_start,
                target_time=target_time,
                step_size=step_size,
                max_steps=max_steps,
                normalized=normalized,
                localized=localized
            )
        elif line_method=="RK4":
            pts = compute_pathline_rk4(
                steady_field,
                start_pos=sp,
                start_time=t_start,
                target_time=target_time,
                step_size=step_size,
                max_steps=max_steps,
                normalized=normalized,
                localized=localized
            )
        streamlines.append(pts['points'])
        loc_streamlines.append(pts['loc_points'])
    
    streamlines_dict={
        'streamlines':streamlines,
        'loc_streamlines':loc_streamlines
    }
    # logger.info(f"streamlines_dict:{streamlines_dict}")
    return streamlines_dict

def visualize_pathlines_labels(field: UnsteadyVectorField2D, streamline_point, labes,time_index=0,nerbos=5,max_step=100,scale=100):
    
    # streamline_point： tensor in cpu() Nx505x3
    # labes: N,
    
    slice_data = field.getSlice(time_index).field  # shape: (H, W, 2)
    X, Y = np.meshgrid(
        np.linspace(field.domainMinBoundary[0], field.domainMaxBoundary[0], field.Xdim),
        np.linspace(field.domainMinBoundary[1], field.domainMaxBoundary[1], field.Ydim)
    )
    U = slice_data[:, :, 0]
    V = slice_data[:, :, 1]

    # 下采样箭头展示
    step = 5
    X_down = X[::step, ::step]
    Y_down = Y[::step, ::step]
    U_down = U[::step, ::step]
    V_down = V[::step, ::step]

    plt.figure(figsize=(8, 6))
    # the laeger scale, the smaller arrow
    plt.quiver(X_down, Y_down, U_down, V_down, color='gray', alpha=0.5, scale=scale)
    
    N,pN,_=streamline_point.shape
    colors = {0: 'r', 1: 'g'}
    for i in range(N):
        streamline=streamline_point[i,:,0:2].numpy()# Nx2
        # print(f"labes({i}):{labes[i]},streamline:{streamline.shape}")
        for j in range(5):
            plt.plot(streamline[0+j*(max_step+1):(j+1)*(max_step+1)-1,0 ], streamline[0+j*(max_step+1):(j+1)*(max_step+1)-1,1 ], colors[labes[i]], linewidth=1)
    
    plt.legend()
    plt.title(f"Streamline at t = {field.getTime(time_index):.2f}")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

def visualize_pathlines(field: UnsteadyVectorField2D, streamline_points, time_index=0,nerbos=5,max_step=100):
    slice_data = field.getSlice(time_index).field  # shape: (H, W, 2)
    X, Y = np.meshgrid(
        np.linspace(field.domainMinBoundary[0], field.domainMaxBoundary[0], field.Xdim),
        np.linspace(field.domainMinBoundary[1], field.domainMaxBoundary[1], field.Ydim)
    )
    U = slice_data[:, :, 0]
    V = slice_data[:, :, 1]

    # 下采样箭头展示
    step = 5
    X_down = X[::step, ::step]
    Y_down = Y[::step, ::step]
    U_down = U[::step, ::step]
    V_down = V[::step, ::step]

    plt.figure(figsize=(8, 6))
    plt.quiver(X_down, Y_down, U_down, V_down, color='gray', alpha=0.5, scale=100)

    N,pN,_=streamline_points.shape
    for i in range(N):
        streamline=streamline_points[i,:,0:2].numpy()# Nx2
        # print(f"labes({i}):{labes[i]},streamline:{streamline.shape}")
        for j in range(5):
            plt.plot(streamline[0+j*(max_step+1):(j+1)*(max_step+1)-1,0 ], streamline[0+j*(max_step+1):(j+1)*(max_step+1)-1,1 ], linewidth=1)

    plt.legend()
    plt.title(f"Streamline at t = {field.getTime(time_index):.2f}")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

def pcdGen(samples,vectorfield,line_method="Euler"):
    """
    Function to generate the pointclouds
    
    input: 
        configs: params for lines generation
        vectorfield: the 2D vector field
    
    output: list of point clouds
        all_lines: the original line in original space
        loc_all_lines: the lines in localized and normalized space
    """
    logger.info(f"Generating pcds...")
    slice = vectorfield.getSlice(samples.time_slice)  # get one slice
    grid_interval=vectorfield.gridInterval
    
    all_line=[]
    loc_all_line=[]
    
    points_ori=None
    points_tsfm=None
    
    if samples.sample_area=="full":
        if samples.line_type=="s_line":
            # sample the whole vector field
            interval_scale=samples.interval_scale
            x_sample=int((vectorfield.domainMaxBoundary[0]-vectorfield.domainMinBoundary[0])/(vectorfield.gridInterval[0]*interval_scale))# x grid interval
            y_sample=int((vectorfield.domainMaxBoundary[1]-vectorfield.domainMinBoundary[1])/(vectorfield.gridInterval[1]*interval_scale))# y grid interval
            
            x_setp_array=np.array([grid_interval[0]*interval_scale,0])
            y_setp_array=np.array([0,grid_interval[1]*interval_scale])
            start_from=np.array(vectorfield.domainMinBoundary)#vectorfield.domainMinBoundary
            dt=vectorfield.timeInterval*samples.dt_scale
            max_steps=samples.line_steps
            for x in range(x_sample):
                cur_pos=start_from+x*x_setp_array
                for y in range(y_sample):
                    now_pos=cur_pos+y*y_setp_array
                    streamline_dict = compute_multiple_streamlines(
                        slice.field,
                        start_pos=now_pos,  # 物理坐标
                        domain_min=vectorfield.domainMinBoundary,
                        domain_max=vectorfield.domainMaxBoundary,
                        Xdim=vectorfield.Xdim,
                        Ydim=vectorfield.Ydim,
                        step_size=dt,
                        max_steps=max_steps,
                        offset_dist=vectorfield.gridInterval[0]*samples.offset_scale,
                        normalized=samples.normalized,
                        localized=samples.localized
                    )
                    
                    nerbos=samples.sample_nerbors
                    target_shape = (nerbos, max_steps+1, 2)
                    safe_arr = to_array_if_valid(streamline_dict['streamlines'], expected_shape=target_shape)
                    loc_safe_arr=to_array_if_valid(streamline_dict['loc_streamlines'], expected_shape=target_shape)
                    if safe_arr is not None  and loc_safe_arr is not None :

                        streamline_points=safe_arr.reshape([1,-1,2]) # sampled 
                        all_line.append(streamline_points)
                        
                        
                        streamline_points=loc_safe_arr.reshape([1,-1,2])
                        loc_all_line.append(streamline_points)
                    else:
                        continue
            np_points_in_ori_spaces=np.array(all_line)
            logger.info(f"points_in_ori_spaces:{np_points_in_ori_spaces.shape}")
            np_points_in_trsf_spaces=np.array(loc_all_line)
            logger.info(f"points_in_trsf_spaces:{np_points_in_trsf_spaces.shape}")
            
            points_ori=torch.from_numpy(np_points_in_ori_spaces).reshape(np_points_in_ori_spaces.shape[0],-1,np_points_in_ori_spaces.shape[-1])
            ones = points_ori.new_ones(*points_ori.shape[:-1], 1)  # shape = [4, 1]
            points_ori=torch.cat([points_ori,ones],dim=2).cuda().float()
            logger.info(f"points_ori:{points_ori.shape}")
            
            points_tsfm=torch.from_numpy(np_points_in_trsf_spaces).reshape(np_points_in_trsf_spaces.shape[0],-1,np_points_in_ori_spaces.shape[-1])
            ones = points_tsfm.new_ones(*points_tsfm.shape[:-1], 1)  # shape = [4, 1]
            points_tsfm=torch.cat([points_tsfm,ones],dim=2).cuda().float()
            logger.info(f"points_tsfm:{points_tsfm.shape}")
        if samples.line_type=="p_line":
            # todo get pathlnes
            interval_scale=samples.interval_scale
            x_sample=int((vectorfield.domainMaxBoundary[0]-vectorfield.domainMinBoundary[0])/(vectorfield.gridInterval[0]*interval_scale))# x grid interval
            y_sample=int((vectorfield.domainMaxBoundary[1]-vectorfield.domainMinBoundary[1])/(vectorfield.gridInterval[1]*interval_scale))# y grid interval
            
            x_setp_array=np.array([grid_interval[0]*interval_scale,0])
            y_setp_array=np.array([0,grid_interval[1]*interval_scale])
            start_from=np.array(vectorfield.domainMinBoundary)#vectorfield.domainMinBoundary
            dt=vectorfield.timeInterval*samples.dt_scale
            max_steps=samples.line_steps
            for x in range(x_sample):
                cur_pos=start_from+x*x_setp_array
                for y in range(y_sample):
                    now_pos=cur_pos+y*y_setp_array
                    streamline_dict=compute_multiple_pathlines(
                        steady_field=vectorfield,       # numpy array of shape (H, W, 2)
                        start_pos=(now_pos[0],now_pos[1],float(samples.t_start*(vectorfield.tmax-vectorfield.tmin))),          # (x, y, t_start) in physical space
                        target_time=samples.t_target*(vectorfield.tmax-vectorfield.tmin),
                        step_size=dt,
                        max_steps=max_steps,
                        offset_dist=vectorfield.gridInterval[0]*samples.offset_scale,
                        normalized=samples.normalized,
                        localized=samples.localized,
                        line_method=line_method
                    )
                    nerbos=samples.sample_nerbors
                    target_shape = (nerbos, max_steps+1, 3)
                    # logger.info(f"streamline_dict['streamlines']:{len(streamline_dict['streamlines']),len(streamline_dict['streamlines'][0])}")
                    safe_arr = to_array_if_valid(streamline_dict['streamlines'], expected_shape=target_shape)
                    # logger.info(f"safe_arr:{safe_arr.shape}")
                    loc_safe_arr=to_array_if_valid(streamline_dict['loc_streamlines'], expected_shape=target_shape)
                    if safe_arr is not None  and loc_safe_arr is not None :

                        streamline_points=safe_arr.reshape([1,-1,safe_arr.shape[-1]]) # sampled 
                        all_line.append(streamline_points)
                        
                        
                        streamline_points=loc_safe_arr.reshape([1,-1,loc_safe_arr.shape[-1]])
                        loc_all_line.append(streamline_points)
                    else:
                        continue
            np_points_in_ori_spaces=np.array(all_line)
            logger.info(f"points_in_ori_spaces:{np_points_in_ori_spaces.shape}")
            np_points_in_trsf_spaces=np.array(loc_all_line)
            logger.info(f"points_in_trsf_spaces:{np_points_in_trsf_spaces.shape}")
            
            points_ori=torch.from_numpy(np_points_in_ori_spaces).reshape(np_points_in_ori_spaces.shape[0],-1,np_points_in_ori_spaces.shape[-1]).cuda().float()
            logger.info(f"points_ori:{points_ori.shape}")
            
            points_tsfm=torch.from_numpy(np_points_in_trsf_spaces).reshape(np_points_in_trsf_spaces.shape[0],-1,np_points_in_ori_spaces.shape[-1]).cuda().float()
            logger.info(f"points_tsfm:{points_tsfm.shape}")
    if samples.sample_area=="points":# start from some specific points
        if samples.line_type=="p_line":
            # todo get pathlnes
            interval_scale=samples.interval_scale
            x_sample=int((vectorfield.domainMaxBoundary[0]-vectorfield.domainMinBoundary[0])/(vectorfield.gridInterval[0]*interval_scale))# x grid interval
            y_sample=int((vectorfield.domainMaxBoundary[1]-vectorfield.domainMinBoundary[1])/(vectorfield.gridInterval[1]*interval_scale))# y grid interval
            
            x_setp_array=np.array([grid_interval[0]*interval_scale,0])
            y_setp_array=np.array([0,grid_interval[1]*interval_scale])
            start_from=np.array(vectorfield.domainMinBoundary)#vectorfield.domainMinBoundary
            dt=vectorfield.timeInterval*samples.dt_scale
            max_steps=samples.line_steps
            for i in range(len(samples.start_points)):
                now_pos=np.array(samples.start_points[i])
                
                streamline_dict=compute_multiple_pathlines(
                    steady_field=vectorfield,       # numpy array of shape (H, W, 2)
                    start_pos=(now_pos[0],now_pos[1],float(samples.t_start*(vectorfield.tmax-vectorfield.tmin))),          # (x, y, t_start) in physical space
                    target_time=samples.t_target*(vectorfield.tmax-vectorfield.tmin),
                    step_size=dt,
                    max_steps=max_steps,
                    offset_dist=vectorfield.gridInterval[0]*samples.offset_scale,
                    normalized=samples.normalized,
                    localized=samples.localized,
                    line_method=line_method
                )
                nerbos=samples.sample_nerbors
                target_shape = (nerbos, max_steps+1, 3)
                # logger.info(f"streamline_dict['streamlines']:{len(streamline_dict['streamlines']),len(streamline_dict['streamlines'][0])}")
                safe_arr = to_array_if_valid(streamline_dict['streamlines'], expected_shape=target_shape)
                # logger.info(f"safe_arr:{safe_arr.shape}")
                loc_safe_arr=to_array_if_valid(streamline_dict['loc_streamlines'], expected_shape=target_shape)
                if safe_arr is not None  and loc_safe_arr is not None :

                    streamline_points=safe_arr.reshape([1,-1,safe_arr.shape[-1]]) # sampled 
                    all_line.append(streamline_points)
                    
                    
                    streamline_points=loc_safe_arr.reshape([1,-1,loc_safe_arr.shape[-1]])
                    loc_all_line.append(streamline_points)
                else:
                    continue
            np_points_in_ori_spaces=np.array(all_line)
            logger.info(f"points_in_ori_spaces:{np_points_in_ori_spaces.shape}")
            np_points_in_trsf_spaces=np.array(loc_all_line)
            logger.info(f"points_in_trsf_spaces:{np_points_in_trsf_spaces.shape}")
            
            points_ori=torch.from_numpy(np_points_in_ori_spaces).reshape(np_points_in_ori_spaces.shape[0],-1,np_points_in_ori_spaces.shape[-1]).cuda().float()
            logger.info(f"points_ori:{points_ori.shape}")
            
            points_tsfm=torch.from_numpy(np_points_in_trsf_spaces).reshape(np_points_in_trsf_spaces.shape[0],-1,np_points_in_ori_spaces.shape[-1]).cuda().float()
            logger.info(f"points_tsfm:{points_tsfm.shape}")
    if samples.sample_area=="areas":# start from some specific areas
        if samples.line_type=="p_line":
            interval_scale=samples.interval_scale
            x_sample=int((vectorfield.domainMaxBoundary[0]-vectorfield.domainMinBoundary[0])/(vectorfield.gridInterval[0]*interval_scale))# x grid interval
            y_sample=int((vectorfield.domainMaxBoundary[1]-vectorfield.domainMinBoundary[1])/(vectorfield.gridInterval[1]*interval_scale))# y grid interval
            
            x_setp_array=np.array([grid_interval[0]*interval_scale,0])
            y_setp_array=np.array([0,grid_interval[1]*interval_scale])
            start_from=np.array(vectorfield.domainMinBoundary)#vectorfield.domainMinBoundary
            dt=vectorfield.timeInterval*samples.dt_scale
            max_steps=samples.line_steps
            
            bboxes=np.array(samples.bbox)
            logger.info(f"bboxes:{bboxes.shape}")
            if bboxes.shape[0]%2!=0:
                logger.error(f"Wrong bboxing setting")
              
            for x in range(x_sample):
                cur_pos=start_from+x*x_setp_array
                for y in range(y_sample):
                    
                    """                    def is_inside_domain(x, y):
                        return (field.domainMinBoundary[0] <= x <= field.domainMaxBoundary[0]) and \
                            (field.domainMinBoundary[1] <= y <= field.domainMaxBoundary[1])"""
                    
                    now_pos=cur_pos+y*y_setp_array
                    
                    if bboxes[0][0]<=now_pos[0]<=bboxes[1][0] and bboxes[0][1]<=now_pos[1]<=bboxes[1][1]:
                        
                        streamline_dict=compute_multiple_pathlines(
                            steady_field=vectorfield,       # numpy array of shape (H, W, 2)
                            start_pos=(now_pos[0],now_pos[1],float(samples.t_start*(vectorfield.tmax-vectorfield.tmin))),          # (x, y, t_start) in physical space
                            target_time=samples.t_target*(vectorfield.tmax-vectorfield.tmin),
                            step_size=dt,
                            max_steps=max_steps,
                            offset_dist=vectorfield.gridInterval[0]*samples.offset_scale,
                            normalized=samples.normalized,
                            localized=samples.localized,
                            line_method=line_method
                        )
                        nerbos=samples.sample_nerbors
                        target_shape = (nerbos, max_steps+1, 3)
                        # logger.info(f"streamline_dict['streamlines']:{len(streamline_dict['streamlines']),len(streamline_dict['streamlines'][0])}")
                        safe_arr = to_array_if_valid(streamline_dict['streamlines'], expected_shape=target_shape)
                        # logger.info(f"safe_arr:{safe_arr.shape}")
                        loc_safe_arr=to_array_if_valid(streamline_dict['loc_streamlines'], expected_shape=target_shape)
                        if safe_arr is not None  and loc_safe_arr is not None :

                            streamline_points=safe_arr.reshape([1,-1,safe_arr.shape[-1]]) # sampled 
                            all_line.append(streamline_points)
                            
                            
                            streamline_points=loc_safe_arr.reshape([1,-1,loc_safe_arr.shape[-1]])
                            loc_all_line.append(streamline_points)
                        else:
                            continue
                    else:
                        continue
            np_points_in_ori_spaces=np.array(all_line)
            logger.info(f"points_in_ori_spaces:{np_points_in_ori_spaces.shape}")
            np_points_in_trsf_spaces=np.array(loc_all_line)
            logger.info(f"points_in_trsf_spaces:{np_points_in_trsf_spaces.shape}")
            
            points_ori=torch.from_numpy(np_points_in_ori_spaces).reshape(np_points_in_ori_spaces.shape[0],-1,np_points_in_ori_spaces.shape[-1]).cuda().float()
            logger.info(f"points_ori:{points_ori.shape}")
            
            points_tsfm=torch.from_numpy(np_points_in_trsf_spaces).reshape(np_points_in_trsf_spaces.shape[0],-1,np_points_in_ori_spaces.shape[-1]).cuda().float()
            logger.info(f"points_tsfm:{points_tsfm.shape}") 
    return points_ori,points_tsfm

def sampleLines(line_sample,sample_nerbors,points):
    """
    Function to samples the lines but save the head and tail
    input:
        points: N x nerborsxsteps x points demensions
    
    output: 
        sampled points
    """
    points_ori=points.reshape(points.shape[0],sample_nerbors,-1,points.shape[-1])

    # 1. get head and tail
    first = points_ori[:, :, 0:1, :]     # (B, C, 1, D)
    last  = points_ori[:, :, -1:, :]     # (B, C, 1, D)

    # 2. sample in [1, N-1)
    mid   = points_ori[:, :, 1:-1:line_sample, :]  # (B, C, M, D)，M ≈ (N-2)/step

    # 3. merge
    points_ori = torch.cat([first, mid, last], dim=2)  # (B, C, M+2, D)
    points_ori=points_ori.reshape(points_ori.shape[0],-1,points_ori.shape[-1])
    return points_ori

@torch.no_grad()
def resample_to_fixed_count(points: torch.Tensor,
                                     valid_steps: torch.Tensor,
                                     K: int) -> torch.Tensor:
    """
    points:      [M, Nmax, D]    每条线最多 Nmax 点（超出的为占位）
    valid_steps: [M]             第 i 条线真实点数 n_i (>=1)
    K:           目标点数（所有线最终统一为 K 点）
    return:      [M, K, D]
    """
    device, dtype = points.device, points.dtype
    M, Nmax, D = points.shape
    n = valid_steps.to(torch.long).clamp_min(1)  # [M]

    # 用最后一个有效点“填充”超出部分，避免脏数据
    idxs = torch.arange(Nmax, device=device).view(1, -1).expand(M, -1)      # [M,Nmax]
    last_idx = (n - 1).view(-1, 1).expand(-1, Nmax)                          # [M,Nmax]
    take_idx = torch.where(idxs < n.view(-1, 1), idxs, last_idx)             # [M,Nmax]
    P = points.gather(1, take_idx.unsqueeze(-1).expand(-1, -1, D))           # [M,Nmax,D]

    # 在“索引域” [0, n_i-1] 上等分取 K 个位置
    if K == 1:
        s = torch.zeros(M, 1, device=device, dtype=dtype)
    else:
        s = torch.linspace(0, 1, K, device=device, dtype=dtype).unsqueeze(0) * (n - 1).unsqueeze(1).to(dtype)  # [M,K]

    s0 = s.floor().to(torch.long).clamp_max(Nmax - 1)  # 左索引
    s1 = (s0 + 1).clamp_max(Nmax - 1)                  # 右索引
    alpha = (s - s0.to(dtype)).unsqueeze(-1)           # [M,K,1]

    P0 = P.gather(1, s0.unsqueeze(-1).expand(-1, -1, D))  # [M,K,D]
    P1 = P.gather(1, s1.unsqueeze(-1).expand(-1, -1, D))  # [M,K,D]

    Pk = (1 - alpha) * P0 + alpha * P1                    # 线性插值（按索引）
    return Pk






@torch.no_grad()
def temporal_downsamplePathlineCrossPrimitiveWithValidLength(points: torch.Tensor,valid_length: torch.Tensor,
                                     K: int) -> torch.Tensor:
    """
    Resample along the time dimension to a fixed number of steps K according to the maximum valid length of each group:
    - Must keep the first point index=0
    - Must keep the last valid point of each line (given by valid_length)
    - The remaining K-2 indices are randomly sampled in [1, L_max_group-1) (the same for all lines in the group)

    Args:
      - points: shape [groups, lines_per_group, timesteps, D]
      - valid_length: shape [groups, lines_per_group], valid length of each pathline
      - K: target number of sampled steps (>=2)

    Returns:
      - Tensor, shape [groups, lines_per_group, K, D]
    """
    if K < 2:
        raise ValueError("K must be >= 2 to keep both head and tail")
    if points.dim() < 3:
        raise ValueError("points must have at least 3 dims: [groups, lines_per_group, timesteps, ...]")

    device = points.device
    dtype = torch.long
    G, L = valid_length.shape[0], valid_length.shape[1]
    T = points.shape[2]

    # Clamp valid length to [1, T]
    vl = valid_length.to(device=device)
    vl = torch.clamp(vl, min=1, max=T)
    last_idx_per_line = (vl - 1).to(dtype)

    # Maximum valid length per group (not exceeding T)
    max_len_per_group = vl.max(dim=1).values  # [G]

    # Construct random middle indices for each group (shape [G, K-2], empty if K<=2)
    num_mid = max(K - 2, 0)
    if num_mid > 0:
        mids_by_group = torch.zeros((G, num_mid), device=device, dtype=dtype)
        for g in range(G):
            Lg = int(max_len_per_group[g].item())
            if Lg > 2:
                # Randomly sample from [1, Lg-1)
                idx_mid = torch.randint(1, Lg - 1, (num_mid,), device=device, dtype=dtype)
                idx_mid, _ = torch.sort(idx_mid)
                mids_by_group[g] = idx_mid
            else:
                # No available middle indices, keep as 0 (allow duplication with head)
                pass
    else:
        mids_by_group = None

    # Assemble selection indices: first column=0; middle columns=group-wise random; last column=last valid point of each line
    sel_idx = torch.zeros((G, L, K), device=device, dtype=dtype)
    if num_mid > 0:
        sel_idx[:, :, 1:-1] = mids_by_group[:, None, :].expand(G, L, num_mid)
    sel_idx[:, :, -1] = last_idx_per_line

    # Gather along the time dimension
    expand_sel = sel_idx.unsqueeze(-1).expand(-1, -1, -1, points.shape[-1])
    out = torch.gather(points, dim=2, index=expand_sel)
    return out



def compute_features_in_chunks(points_batch, point_nn, chunk_size=50, desc="Extracting Features"):
    all_features = []
    B = points_batch.shape[0]  # batch size
    num_chunks = (B + chunk_size - 1) // chunk_size  # 向上取整计算chunk数

    for start in tqdm(range(0, B, chunk_size), desc=desc, total=num_chunks):
        end = min(start + chunk_size, B)
        chunk = points_batch[start:end].permute(0, 2, 1)
        with torch.no_grad():  # 避免记录计算图，节省显存
            features = point_nn(chunk)
        all_features.append(features)

    return torch.cat(all_features, dim=0)
