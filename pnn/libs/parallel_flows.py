"""
Here we develop some code to sample calculate the infomation from the flow

"""
import torch
import torch.nn.functional as F
import logging
import numpy as np
from tqdm.auto import tqdm
@torch.no_grad()
def batched_pathlines(
    field_tensor,          # [T, H, W, 2] float32 (GPU)
    domain_min,            # (xmin, ymin)
    domain_max,            # (xmax, ymax)
    tmin, tmax,            # floats
    time_interval,         # Δt between frames
    starts_xy,             # [N, 2] in physical coord
    t_start,               # scalar float,physcial time
    t_target,              # scalar float,physcial time
    dt,                    # scalar float
    max_steps,             # int
    offsets_size=0.1,
    method="rk4",        # or "rk4"
    time_interp="linear"  # "nearest" or "linear"
):
    device = field_tensor.device
    dtype  = field_tensor.dtype

    # 1) seeds [N*5, 2]
    # 这段代码的作用是：对于每一个起始点 starts_xy，生成5个偏移点（包括原点和四个方向的偏移），用于后续生成一组交叉路径线（cross pathlines）。
    # offsets 定义了5个二维偏移量，分别是(0,0)、(+offsets_size,0)、(-offsets_size,0)、(0,+offsets_size)、(0,-offsets_size)，即原点、左右、上下。
    offsets = ((0,0),(+offsets_size,0),(-offsets_size,0),(0,+offsets_size),(0,-offsets_size))
    # offs 是 shape 为 [5,2] 的张量，存储上述5个偏移量，放在和 field_tensor 相同的设备和数据类型上。
    offs = torch.tensor(offsets, device=device, dtype=dtype)   # [5,2]
    # N 是起始点的数量
    N = starts_xy.shape[0]
    # 通过广播 starts_xy[:,None,:] + offs[None,:,:]，对每个起始点加上5个偏移，得到 [N,5,2]，再 reshape 成 [N*5,2]，即每个起始点扩展为5个偏移点
    seeds = (starts_xy[:,None,:] + offs[None,:,:]).reshape(-1,2).to(device)  # [M,2], M=N*5

    # 2) time direction
    dt = dt if t_target >= t_start else -dt
    dt= torch.tensor(dt,device=device,dtype=dtype)
    T, H, W, _ = field_tensor.shape

    # 3) helpers ---------------------------------------------------------------
    def phys_to_grid(xy):
        gx = (xy[...,0]-domain_min[0])/(domain_max[0]-domain_min[0])*(W-1)
        gy = (xy[...,1]-domain_min[1])/(domain_max[1]-domain_min[1])*(H-1)
        return torch.stack([gx, gy], dim=-1)

    def time_to_index(t):
        return torch.clamp(((t - tmin)/time_interval).round().long(), 0, T-1)

    def sample_velocity(xy, tcur):
        gxy = phys_to_grid(xy)
        gx  = torch.clamp(gxy[...,0], 0, W-1)
        gy  = torch.clamp(gxy[...,1], 0, H-1)
        x0 = gx.floor().long();  x1 = torch.clamp(x0+1, max=W-1)
        y0 = gy.floor().long();  y1 = torch.clamp(y0+1, max=H-1)
        wx = gx - x0.float();    wy = gy - y0.float()
        w00 = (1-wx)*(1-wy); w01 = (1-wx)*wy
        w10 = wx*(1-wy);     w11 = wx*wy

        if time_interp == "nearest":
            ti = time_to_index(tcur)                    # [M]
            V  = field_tensor[ti, ...]                  # [M,H,W,2]
            ar = torch.arange(V.shape[0], device=device)
            v00 = V[ar, y0, x0]; v01 = V[ar, y1, x0]
            v10 = V[ar, y0, x1]; v11 = V[ar, y1, x1]
            return (w00[...,None]*v00 + w01[...,None]*v01 +
                    w10[...,None]*v10 + w11[...,None]*v11)
        else:
            t_idx_f = torch.clamp(((tcur - tmin)/time_interval).floor().long(), 0, T-1)
            t_idx_c = torch.clamp(t_idx_f+1, 0, T-1)
            alpha   = ((tcur - (tmin + t_idx_f.float()*time_interval)) / time_interval).clamp(0,1)

            V0 = field_tensor[t_idx_f]; V1 = field_tensor[t_idx_c]
            ar = torch.arange(V0.shape[0], device=device)
            v00_0 = V0[ar, y0, x0]; v01_0 = V0[ar, y1, x0]; v10_0 = V0[ar, y0, x1]; v11_0 = V0[ar, y1, x1]
            v00_1 = V1[ar, y0, x0]; v01_1 = V1[ar, y1, x0]; v10_1 = V1[ar, y0, x1]; v11_1 = V1[ar, y1, x1]

            v0 = (w00[...,None]*v00_0 + w01[...,None]*v01_0 +
                  w10[...,None]*v10_0 + w11[...,None]*v11_0)
            v1 = (w00[...,None]*v00_1 + w01[...,None]*v01_1 +
                  w10[...,None]*v10_1 + w11[...,None]*v11_1)
            return (1-alpha)[...,None]*v0 + alpha[...,None]*v1

    # 4) buffers ---------------------------------------------------------------
    M = seeds.shape[0]
    P = torch.full((M, max_steps, 3), float('nan'), device=device, dtype=dtype)
    alive = torch.ones((M,), dtype=torch.bool, device=device)

    # 记录“最后写入的步索引”（从0起），以及物理弧长
    last_step = torch.zeros((M,), dtype=torch.int32, device=device)  # 初始已写入 step=0

    x = seeds.clone()                      # [M,2]
    t = torch.full((M,), t_start, device=device, dtype=dtype)


    # 写入 step 0
    P[:,0,:2] = x
    P[:,0,  2] = t 

    # 初速近零者直接标记为死亡
    v0 = sample_velocity(x, t)
    alive &= (v0.abs().max(dim=1).values > 1e-6)

    # 5) main loop -------------------------------------------------------------
    for k in range(max_steps-1):
        if not alive.any(): break

        idx_alive = torch.where(alive)[0]
        x_prev = x[idx_alive]                        # 保存上一位置用于弧长
        v = sample_velocity(x_prev, t[idx_alive])

        if method.lower() == "euler":
            x_new = x_prev + v*dt
        elif method.lower() == "rk4":
            k1 = v
            k2 = sample_velocity(x_prev + 0.5*dt*k1, t[idx_alive] + 0.5*dt)
            k3 = sample_velocity(x_prev + 0.5*dt*k2, t[idx_alive] + 0.5*dt)
            k4 = sample_velocity(x_prev + dt*k3,     t[idx_alive] + dt)
            x_new = x_prev + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

        t_new = t[idx_alive] + dt

        # 越界/到时检测
        in_x = (x_new[:,0] >= domain_min[0]) & (x_new[:,0] <= domain_max[0])
        in_y = (x_new[:,1] >= domain_min[1]) & (x_new[:,1] <= domain_max[1])
        still_in = in_x & in_y
        time_ok  = ((dt>0) & (t_new < t_target)) | ((dt<0) & (t_new > t_target))
        still_alive_now = still_in & time_ok

        # 写入 step k+1
        x[idx_alive] = x_new
        t[idx_alive] = t_new
        P[idx_alive, k+1, :2] = x_new
        P[idx_alive, k+1,  2] = t_new 

        # 1) 更新最后步索引
        last_step[idx_alive] = k + 1

        # 更新存活标记
        alive[idx_alive[~still_alive_now]] = False


    # 有效点数（包含 step0），即“最终长度（按点数）”
    valid_steps = (last_step + 1).to(torch.int32)   # [M]

    return P, valid_steps

def gen_starts(vectorfield, interval_scale, device="cuda", dtype=torch.float32, include_end=True):
    xmin, ymin = vectorfield.domainMinBoundary
    xmax, ymax = vectorfield.domainMaxBoundary
    gx, gy     = vectorfield.gridInterval
    step_x = float(gx) * interval_scale
    step_y = float(gy) * interval_scale

    nx = int(torch.floor(torch.tensor((xmax - xmin)/step_x)).item()) + 1
    ny = int(torch.floor(torch.tensor((ymax - ymin)/step_y)).item()) + 1

    xs = torch.arange(nx, device=device, dtype=dtype) * step_x + xmin
    ys = torch.arange(ny, device=device, dtype=dtype) * step_y + ymin

    if include_end:
        if xs[-1] < xmax - 1e-12:
            xs = torch.cat([xs, torch.tensor([xmax], device=device, dtype=dtype)])
        if ys[-1] < ymax - 1e-12:
            ys = torch.cat([ys, torch.tensor([ymax], device=device, dtype=dtype)])

    # meshgrid & 拼成 [N,2]
    YY, XX = torch.meshgrid(ys, xs, indexing="ij")    # "ij" 或者用 "xy" 都行，下面按列/行展平
    starts_xy = torch.stack([XX.reshape(-1), YY.reshape(-1)], dim=1)  # [N,2]
    return starts_xy

def filter_starts_by_bboxes(starts_xy, bboxes, device="cuda"):
    """
    starts_xy: [N,2] (torch, GPU)
    bboxes:    numpy 或 torch, 形状 [K,2,2]，[[xmin,ymin],[xmax,ymax]]
    返回: 过滤后的 starts_xy
    """
    if not torch.is_tensor(bboxes):
        bboxes = torch.from_numpy(bboxes)
    bboxes = bboxes.to(device=starts_xy.device, dtype=starts_xy.dtype)  # [K,2,2]

    x = starts_xy[:, 0][:, None]  # [N,1]
    y = starts_xy[:, 1][:, None]  # [N,1]

    xmin = bboxes[:, 0, 0][None, :]  # [1,K]
    ymin = bboxes[:, 0, 1][None, :]
    xmax = bboxes[:, 1, 0][None, :]
    ymax = bboxes[:, 1, 1][None, :]

    in_box = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)  # [N,K]
    keep  = in_box.any(dim=1)  # [N]
    return starts_xy[keep]

def iter_starts_batches(vectorfield,
                        interval_scale,
                        rows_per_batch=64,
                        device="cuda",
                        dtype=torch.float32,
                        include_end=True,
                        bboxes=None):
    """
    逐批(y方向分块)生成起点:
      - 每次 yield 一个 starts_xy: [Nb, 2] (GPU tensor)
      - 可选: bboxes 过滤(形状 [K,2,2], [[xmin,ymin],[xmax,ymax]])
    """
    xmin, ymin,_ = vectorfield.domainMinBoundary
    xmax, ymax,_ = vectorfield.domainMaxBoundary
    gx, gy     = vectorfield.gridInterval

    step_x = float(gx) * interval_scale
    step_y = float(gy) * interval_scale

    nx = int(torch.floor(torch.tensor((xmax - xmin)/step_x)).item()) + 1
    ny = int(torch.floor(torch.tensor((ymax - ymin)/step_y)).item()) + 1

    xs = torch.arange(nx, device=device, dtype=dtype) * step_x + xmin
    ys = torch.arange(ny, device=device, dtype=dtype) * step_y + ymin

    if include_end:
        if xs[-1] < xmax - 1e-12:
            xs = torch.cat([xs, torch.tensor([xmax], device=device, dtype=dtype)])
        if ys[-1] < ymax - 1e-12:
            ys = torch.cat([ys, torch.tensor([ymax], device=device, dtype=dtype)])

    # 预处理 bbox
    if bboxes is not None:
        if not torch.is_tensor(bboxes):
            bboxes = torch.from_numpy(np.asarray(bboxes))
        bboxes = bboxes.to(device=device, dtype=dtype)  # [K,2,2]
        xmin_b = bboxes[:, 0, 0][None, :]
        ymin_b = bboxes[:, 0, 1][None, :]
        xmax_b = bboxes[:, 1, 0][None, :]
        ymax_b = bboxes[:, 1, 1][None, :]

    ny = ys.numel()
    for y0 in range(0, ny, rows_per_batch):
        y1 = min(ny, y0 + rows_per_batch)
        YY, XX = torch.meshgrid(ys[y0:y1], xs, indexing="ij")    # [r, nx]
        starts = torch.stack([XX.reshape(-1), YY.reshape(-1)], dim=1)  # [r*nx, 2]

        if bboxes is not None:
            x = starts[:, 0][:, None]  # [Nb,1]
            y = starts[:, 1][:, None]
            in_box = (x >= xmin_b) & (x <= xmax_b) & (y >= ymin_b) & (y <= ymax_b)  # [Nb,K]
            keep = in_box.any(dim=1)
            starts = starts[keep]

        if starts.numel() == 0:
            continue

        yield starts

def run_pathlines_batched(samples, vectorfield, line_method="Euler",
                          rows_per_batch=4, device="cuda"):
    # 统一准备
    field_tensor = torch.from_numpy(vectorfield.field).to(device)  # [T,H,W,2]
    t_start  = float(samples.t_start * (vectorfield.tmax - vectorfield.tmin)+vectorfield.tmin)
    t_target = float(samples.t_target * (vectorfield.tmax - vectorfield.tmin)+vectorfield.tmin)
    dt       =  samples.dt
    max_steps = samples.line_steps
    offset_dist = vectorfield.gridInterval[0] * samples.offset_scale

    all_P = []
    all_valid = []

    # 选择是否需要 bbox
    bboxes = None
    if samples.sample_area == "areas":
        bboxes = np.array(samples.bbox)
    # total_seeds=int((vectorfield.field.shape[1]/samples.interval_scale)*(vectorfield.field.shape[2]/samples.interval_scale))
    # with tqdm(total=total_seeds, desc="pathline seeds", unit="seed", leave=False) as pbar:
        # 按批生成
    p_row=0
    full_row=vectorfield.field.shape[1]+1
    for starts_xy in iter_starts_batches(
        vectorfield, samples.interval_scale,
        rows_per_batch=rows_per_batch, device=device, dtype=torch.float32,
        include_end=True, bboxes=bboxes
    ):
        # 空批跳过
        if starts_xy.numel() == 0:
            continue

        P_b, valid_b = batched_pathlines(
            field_tensor=field_tensor,
            domain_min=vectorfield.domainMinBoundary,
            domain_max=vectorfield.domainMaxBoundary,
            tmin=vectorfield.tmin, tmax=vectorfield.tmax,
            time_interval=vectorfield.timeInterval,
            starts_xy=starts_xy,                 # 这一批的起点
            t_start=t_start, t_target=t_target,
            dt=dt, max_steps=max_steps,
            offsets_size=offset_dist,
            method=line_method,
            time_interp="nearest"
        )

        all_P.append(P_b)
        all_valid.append(valid_b)
        # 关键：按“本批起点数”推进
        p_row+=rows_per_batch
        logging.info(f"\r processing row:{p_row}/{full_row}")
            # pbar.update(starts_xy.shape[0])
    if not all_P:
        # 没有任何起点（可能bbox过小），返回空张量
        return (torch.empty(0, max_steps+1, 3, device=device),
                torch.empty(0, dtype=torch.int32, device=device))

    P = torch.cat(all_P, dim=0)            # [sum(M_b), max_steps+1, 3]
    valid_steps = torch.cat(all_valid, 0)  # [sum(M_b)]
    return P.float(), valid_steps


def ParaPcdGen(samples,vectorfield,line_method="Euler",device="cuda"):
    """
    Function to generate the pointclouds
    
    input: 
        configs: params for lines generation
        vectorfield: the 2D vector field
    
    output: list of point clouds
        all_lines: the original line in original space
        loc_all_lines: the lines in localized and normalized space
    """
    logging.info(f"Generating pcds...")
    grid_interval=vectorfield.gridInterval
    
    P=None

    dt=vectorfield.timeInterval*samples.dt_scale
    t_start=float(samples.t_start*(vectorfield.tmax-vectorfield.tmin))+vectorfield.tmin
    t_target=samples.t_target*(vectorfield.tmax-vectorfield.tmin)+vectorfield.tmin
    max_steps=samples.line_steps
    offset_dist=vectorfield.gridInterval[0]*samples.offset_scale
    
    starts_xy=None
    # 1) 一次性生成起点
    if samples.sample_area == "full":
        starts_xy = gen_starts(vectorfield, samples.interval_scale, device="cuda")
    elif samples.sample_area == "areas":
        starts_xy_all = gen_starts(vectorfield, samples.interval_scale, device="cuda")
        bboxes = np.array(samples.bbox)  # 或 torch.tensor(...)
        starts_xy = filter_starts_by_bboxes(starts_xy_all, bboxes)
    elif samples.sample_area == "points":
        starts_xy = torch.tensor(samples.start_points, device="cuda", dtype=torch.float32)
    else:
        raise ValueError("Unknown sample_area")
    
    
    if samples.line_type=="p_line":
        
        P, valid_steps=batched_pathlines(
            field_tensor=torch.from_numpy(vectorfield.field).to(device),          # [T, H, W, 2] float32 (GPU)
            domain_min=vectorfield.domainMinBoundary,            # (xmin, ymin)
            domain_max=vectorfield.domainMaxBoundary,            # (xmax, ymax)
            tmin=vectorfield.tmin, tmax=vectorfield.tmax,            # floats
            time_interval=vectorfield.timeInterval,         # Δt between frames
            starts_xy=starts_xy,             # [N, 2] in physical coord
            t_start=t_start,               # scalar float
            t_target=t_target,              # scalar float
            dt=dt,             # scalar float
            max_steps=max_steps,             # int
            offsets_size=offset_dist,
            method=line_method,        # or "rk4"
            time_interp="nearest"  # "nearest" or "linear"
        )
            
    return P.float(), valid_steps