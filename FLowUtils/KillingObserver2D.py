

import vtk
import numpy as np
from typing import Callable, List, Tuple
from .VectorField2d import UnsteadyVectorField2D, SteadyVectorField2D
from .flowlineIntegral import (
	pathline_integration_one_direction_2D,
	compute_pathline_2D,
)


class TimeDependentKilling2D(UnsteadyVectorField2D):
	"""
	时间依赖的 2D Killing 观察者场。
	定义如下:
	- 平移速度: (a(t), b(t)) 
	- 刚体角速度: c(t) 绕 z 轴，速度场的旋转部分为 c(t) * (-y, x)
	因此观测者场为: v(x,y,t) = (a(t), b(t)) + c(t) * (-y, x)
	
	注意: 该类是“解析型”的，不使用栅格数据 self.field。
	仅复用基类的网格/时间元信息以便路径积分使用 tmin/tmax 和域。
	"""
	def __init__(
		self,
		Xdim:int,
		Ydim:int,
		time_steps:int,
		domainMinBoundary:list=[-2.0,-2.0,0.0],
		domainMaxBoundary:list=[2.0,2.0,2.0],
		a:Callable[[float], float]=lambda t: 0.0,
		b:Callable[[float], float]=lambda t: 0.0,
		c:Callable[[float], float]=lambda t: 0.0,
	):
		# 初始化网格与时间元信息
		UnsteadyVectorField2D.__init__(self, Xdim, Ydim, time_steps, domainMinBoundary, domainMaxBoundary)
		# 标记字段不用数据张量
		self.field = np.zeros((time_steps, Ydim, Xdim, 2), dtype=np.float32)
		self._a = a
		self._b = b
		self._c = c

	def getSlice(self, timeSlice:int) -> SteadyVectorField2D:
		# 提供一张基于解析表达式采样的切片（采样在栅格点上）
		sf = SteadyVectorField2D(self.Xdim, self.Ydim, self.domainMinBoundary, [self.domainMaxBoundary[0], self.domainMaxBoundary[1], 0.0])
		Y, X = self.Ydim, self.Xdim
		out = np.zeros((Y, X, 2), dtype=np.float32)
		t = float(self.getPhysicalTime(int(np.clip(timeSlice, 0, max(0, self.time_steps-1)))))
		for iy in range(Y):
			for ix in range(X):
				px, py = self.convert_grid_pos_2_physical_pos(ix, iy)
				vx, vy = self._eval(px, py, t)
				out[iy, ix, 0] = vx
				out[iy, ix, 1] = vy
		sf.field = out
		return sf


	def _eval(self, x:float, y:float, t:float) -> Tuple[float, float]:
		a_t = float(self._a(t))
		b_t = float(self._b(t))
		c_t = float(self._c(t))
		# 旋转部分 c(t)*(-y, x)
		vx = a_t + (-c_t) * y
		vy = b_t + ( c_t) * x
		return vx, vy

	def get_vector(self, posX:float, posY:float, time:float) -> np.ndarray:
		vx, vy = self._eval(posX, posY, time)
		return np.array([vx, vy], dtype=np.float32)


class Worldline2D:
	"""
	存储路径线与沿线瞬时角速度（标量 c(t)）。
	pathline: List[ (pos3d(np.float32[3]), time(float)) ]
	inst_rotation: List[float] 与 pathline 等长
	"""
	def __init__(self):
		self.pathline: List[Tuple[np.ndarray, float]] = []
		self.inst_rotation: List[float] = []
		self.tmin_tmax: Tuple[float, float] | None = None

	def set_time_range_from_path(self):
		if len(self.pathline) == 0:
			self.tmin_tmax = None
			return
		ts = [t for _, t in self.pathline]
		self.tmin_tmax = (min(ts), max(ts))

	def number_of_steps(self) -> int:
		return len(self.pathline)


def compute_worldline_2d(vector_field: UnsteadyVectorField2D, start_pos3d: np.ndarray, t0: float, stepSize: float, observation_t_min: float | None=None, observation_t_max: float | None=None, maxIterations:int=5000, method:str="RK4") -> Worldline2D:
	"""
	积分观察者场得到 worldline（路径线 + 沿线瞬时角速度）。
	- 前后向拼接，与 C++ 逻辑一致
	- 瞬时角速度: omega = 0.5*(du/dx_y - dv/dy_x) = 0.5*(dvdx_y - dvdy_x) 对于 2D，
	  也即 z 分量的角速度标量。
	"""
	# 路径线（双向）
	min_time = observation_t_min if observation_t_min is not None else vector_field.getMinTime()
	max_time = observation_t_max if observation_t_max is not None else vector_field.getMaxTime()
	forward = pathline_integration_one_direction_2D(vector_field, start_pos3d, t0, max_time, stepSize, maxIterations, method)
	backward = pathline_integration_one_direction_2D(vector_field, start_pos3d, t0, min_time, stepSize, maxIterations, method)
	backward = backward[::-1]
	full_path = backward + forward[1:]

	wl = Worldline2D()
	wl.pathline = full_path
	wl.set_time_range_from_path()

	# 计算沿线瞬时角速度（标量）
	inst = []
	for (p3, t) in full_path:
		x, y = float(p3[0]), float(p3[1])
		omega = instantaneous_angular_velocity_z(vector_field, x, y, t)
		inst.append(float(omega))
	wl.inst_rotation = inst
	return wl


def instantaneous_angular_velocity_z(vector_field: UnsteadyVectorField2D, x: float, y: float, t: float) -> float:
	"""
	以二阶中心差分计算 2D 场的角速度 z 分量: 0.5*(du/dy - dv/dx)，
	与 C++ 中 0.5*(dvdx(1) - dvdy(0)) 等价。
	"""
	dY = (vector_field.domainMaxBoundary[1] - vector_field.domainMinBoundary[1]) / float(vector_field.Ydim - 1);
	dX = (vector_field.domainMaxBoundary[0] - vector_field.domainMinBoundary[0]) / float(vector_field.Xdim - 1);
	# du/dy
	v_y_plus = vector_field.get_vector(x, y + dY, t)
	v_y_minus = vector_field.get_vector(x, y - dY, t)
	du_dy = (v_y_plus[0] - v_y_minus[0]) / (2.0 * dY)
	# dv/dx
	v_x_plus = vector_field.get_vector(x + dX, y, t)
	v_x_minus = vector_field.get_vector(x - dX, y, t)
	dv_dx = (v_x_plus[1] - v_x_minus[1]) / (2.0 * dX)
	omega = 0.5 * (du_dy - dv_dx)
	return float(omega)


class ReferenceFrameTransformation2D:
	"""参考系变换: 保存 worldline、观测时刻、积分旋转(标量 theta 序列)。"""
	def __init__(self):
		self.worldline: Worldline2D | None = None
		self.observation_time: float = 0.0
		self.integrated_rotation: List[float] = []


def compute_reference_frame_transformation_from_field(vector_field: UnsteadyVectorField2D, start_pos3d: np.ndarray, t0: float, stepSize: float, observation_time: float, maxIterations:int=5000, method:str="RK4") -> ReferenceFrameTransformation2D:
	"""
	先计算 worldline，再基于观测时刻积分旋转（标量）。
	"""
	rl = ReferenceFrameTransformation2D()
	rl.worldline = compute_worldline_2d(vector_field, start_pos3d, t0, stepSize, None, None, maxIterations, method)
	rl.observation_time = observation_time
	integrate_reference_frame_rotation(rl.worldline, observation_time, rl)
	return rl


def integrate_reference_frame_rotation(worldline: Worldline2D, observation_time: float, rf: ReferenceFrameTransformation2D):
	"""
	数值积分 c(t) 沿路径对观测时刻向前向后累积：
	- 找到路径中小于 observation_time 的分界索引 k
	- 后向步长使用 -dt，前向使用 +dt；dt 取均匀步长估计 (tmax-tmin)/(N-1)
	- 结果 integrated_rotation 与路径长度一致。
	"""
	rf.observation_time = observation_time
	rf.integrated_rotation = []
	if worldline is None or worldline.number_of_steps() == 0 or worldline.tmin_tmax is None:
		return

	tmin, tmax = worldline.tmin_tmax
	N = worldline.number_of_steps()
	if N < 2 or not (tmin < tmax):
		return
	dt = (tmax - tmin) / float(max(1, N - 1))

	# 找分界点（第一个 >= obs 的索引）
	k = 0
	for idx, (_, tt) in enumerate(worldline.pathline):
		if tt < observation_time:
			k += 1
		else:
			break
	# 若全在过去，则挪回一步，保证有前向
	if k == N:
		k = N - 1

	theta = [0.0] * N
	theta[k] = 0.0
	inst = worldline.inst_rotation

	# backward accumulate (use -dt with inst at i-1)
	for i in range(k, 0, -1):
		omega = float(inst[i - 1])
		theta[i - 1] = theta[i] + (-dt) * omega
	# forward accumulate (use +dt with inst at i+1)
	for i in range(k, N - 1):
		omega = float(inst[i + 1])
		theta[i + 1] = theta[i] + (+dt) * omega

	rf.integrated_rotation = theta


__all__ = [
	"TimeDependentKilling2D",
	"Worldline2D",
	"compute_worldline_2d",
	"instantaneous_angular_velocity_z",
	"ReferenceFrameTransformation2D",
	"compute_reference_frame_transformation_from_field",
	"integrate_reference_frame_rotation",
]




