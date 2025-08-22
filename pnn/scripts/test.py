import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator

# === Step 1: 加载 NetCDF 数据 ===
ds = Dataset(r'D:\Projects\FlowVis\Point-NN\data\vectorField\doublegyre2d.nc')
u = ds.variables['u'][:]        # shape: (T, Y, X)
v = ds.variables['v'][:]        # shape: (T, Y, X)
x = ds.variables['xdim'][:]     # shape: (X,)
y = ds.variables['ydim'][:]     # shape: (Y,)
t = ds.variables['tdim'][:]     # shape: (T,)
print(f"t:{t.shape}")
# === Step 2: 定义速度插值函数 ===
def get_velocity_at(x_pos, y_pos, t_index):
    u_interp = RegularGridInterpolator((y, x), u[t_index], bounds_error=False, fill_value=0)
    v_interp = RegularGridInterpolator((y, x), v[t_index], bounds_error=False, fill_value=0)
    return u_interp((y_pos, x_pos)), v_interp((y_pos, x_pos))

# === Step 3: 初始化参数 ===
start_pos = (1.0, 0.5)
dt = ((t[-1]-t[0])/len(t))*0.2
print(f"dt:{dt}")
max_steps = 5000
x_traj, y_traj = [start_pos[0]], [start_pos[1]]
t_index = 0

# === Step 4: 可视化设置 ===
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(x[0], x[-1])
ax.set_ylim(y[0], y[-1])
ax.set_aspect('equal')
ax.set_title('Animated Pathline with Dynamic Velocity Field')
ax.set_xlabel('X')
ax.set_ylabel('Y')

# 背景流场网格
X, Y = np.meshgrid(x, y)

# 初始时刻稀疏采样速度场
quiver = ax.quiver(X[::5, ::5], Y[::5, ::5],
                   u[0][::5, ::5], v[0][::5, ::5],
                   color='gray', alpha=0.4, scale=50)

# 动画元素
line, = ax.plot([], [], 'r-', linewidth=2, label='Pathline')
point, = ax.plot([], [], 'bo', markersize=6, label='Current Point')
ax.plot([start_pos[0]], [start_pos[1]], 'go', label='Start Point')

# === Step 5: 动画更新函数 ===
def update(frame):
    global t_index, x_traj, y_traj

    if t_index >= len(t) - 1:
        return line, point, quiver

    # 当前物理位置
    x_curr = x_traj[-1]
    y_curr = y_traj[-1]

    # 当前速度
    vx, vy = get_velocity_at(x_curr, y_curr, t_index)
    x_next = x_curr + vx * dt
    y_next = y_curr + vy * dt

    # 边界检查
    if not (x[0] <= x_next <= x[-1]) or not (y[0] <= y_next <= y[-1]):
        return line, point, quiver

    # 更新 pathline 点
    x_traj.append(x_next)
    y_traj.append(y_next)

    # 时间更新
    t_real = t[t_index] + dt
    t_index = int((t_real - t[0]) / (t[1] - t[0]))

    # 更新 pathline 曲线和当前点
    line.set_data(x_traj, y_traj)
    point.set_data([x_next], [y_next])

    # 更新速度场箭头
    U_frame = u[t_index][::5, ::5]
    V_frame = v[t_index][::5, ::5]
    quiver.set_UVC(U_frame, V_frame)

    return line, point, quiver

# === Step 6: 启动动画 ===
ani = FuncAnimation(fig, update, frames=max_steps, interval=1, blit=True, repeat=False)
plt.legend()
plt.tight_layout()
plt.show()
