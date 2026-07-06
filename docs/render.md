# 渲染引擎与相机逻辑 (render.md)

本文档记录 PyflowVis 的 OpenGL 渲染管线、相机 (`CameraObject.py`) 的工作原理，并诊断"无论如何调整相机都存在强透视畸变"的问题。

---

## 1. 整体架构

这是一个基于 **pygame + PyOpenGL + imgui** 的 2D/3D 流场可视化引擎。

- [`VisualizationEngine.py`](../GuiObjcts/VisualizationEngine.py) — 单例引擎，创建 GL 上下文、跑主循环 `MainLoop()`。
- [`Object.py`](../GuiObjcts/Object.py) — 所有可视对象的基类，提供"变量(uniform)系统"和 `getScope()`。
- [`CameraObject.py`](../GuiObjcts/CameraObject.py) — 相机，产出 `viewMat` / `projMat`。
- [`shaderManager.py`](../GuiObjcts/shaderManager.py) — 编译 shader、把 uniform 上传到 GPU。
- [`VertexArrayObject.py`](../GuiObjcts/VertexArrayObject.py) — VAO/VBO 封装，每个对象自带 `modelMat`。
- `assets/shaders/*.glsl` — 顶点/几何/片元 shader。

### 主循环 (`VisualizationEngine.MainLoop`)
```
while running:
    handle_events()                     # 处理 pygame 事件（含 resize）
    glClear(COLOR | DEPTH | STENCIL)
    scene.render_all()                  # 渲染所有 3D 对象
    imgui.new_frame(); scene.drawGui(); imgui.render()   # 叠加 GUI
    pygame.display.flip()
```

---

## 2. Uniform "作用域(scope)"系统

每个 `Object` 持有一组命名变量（`persistentProperties` / `nonPersistentProperties`），通过 `getScope()` 暴露成一个 `dict`。

渲染一个对象时（见 `VertexArrayObject.render` / `PlanarManifold.render`）：
```python
self.material.apply([self.parentScene, self.cameraObject, self])
```
`ShaderProgram.setUniformScope` 把这几个对象的 scope **按顺序合并**成一个大 dict（后者覆盖前者），再逐个 `glUniform*` 上传。因此 uniform 来源为：

| uniform | 来源对象 | 类型 |
|---|---|---|
| `projMat`, `viewMat` | **Camera** (`getScope` 被重写为返回 `MVPVariables`) | `glm.mat4` |
| `modelMat`, `color`, 标量场纹理等 | 具体对象自身 | 多为 `numpy` |
| 场景级公共变量 | `parentScene` | — |

> Camera 重写了 `getScope()`：它**只**返回 `{"viewMat", "projMat"}`，而不是普通变量字典。

---

## 3. 相机 (`CameraObject.py`)

相机状态由 4 个持久变量描述：
- `position` — 相机世界坐标（默认 `(0,0,10)`）
- `targetDirection` — 初始视线方向（默认指向原点，即 `-Z`）
- `up` — 上方向（默认 `(0,1,0)`）
- `rotation_matrix` — arcball 累积旋转（4×4，初始单位阵）

### 3.1 视图矩阵 `get_view_matrix()`
```python
target_dir_new = (rotation_matrix @ target_dir)   # 旋转视线方向
up_dir_new     = (rotation_matrix @ up)            # 旋转上方向
target_new     = position + target_dir_new
return glm.lookAt(position, target_new, up_dir_new)
```
`rotation_matrix` 是纯旋转（由 Rodrigues 公式构造、乘法累积），所以视图矩阵始终是刚体变换 —— **不会**引入任何错切/拉伸。

### 3.2 投影矩阵 `get_projection_matrix()`
```python
fov  = self.getValue("fov")                     # 默认 60°
dist = norm(position)
near = max(0.5, dist * 0.1)                      # 近平面随距离自动变化
far  = max(100.0, dist * 10)                     # 远平面随距离自动变化

if projectionMode == "orthographic":            # ← 新增：正交模式
    half_h = dist * tan(radians(fov)/2)          # 在焦距处与透视取景对齐
    half_w = half_h * aspect_ratio
    return glm.ortho(-half_w, half_w, -half_h, half_h, near, far)
return glm.perspective(radians(fov), aspect_ratio, near, far)
```
- 通过 `projectionMode`（GUI 下拉框，`["perspective", "orthographic"]`）切换透视/正交。
  - **正交模式彻底消除透视前缩**（详见第 6 节）。半高用 `dist * tan(fov/2)` 推算，使切换时焦距处的取景大小不跳变；`fov` 在正交模式下退化为"缩放"控制。
- `aspect_ratio = width / height`，默认 `1600/1200 = 1.333`。
- `near` / `far` 由相机到原点的距离自动推算，用户**无法**直接设定。

### 3.3 交互控制
| 操作 | 方法 | 效果 |
|---|---|---|
| 左键拖拽 | `handle_mouse_move` (arcball) | 累积到 `rotation_matrix`，旋转视线 |
| 滚轮 | `zoom` | 改变 `fov`，**夹在 [10, 50]**（注意初始值 60 超出上限，一旦滚动就回不到 60） |
| WASD/方向键 | `pan` | 沿相机右/上方向平移 `position`，每次仅 **0.1** |
| Q/E | `pan` | 沿视线前后平移，每次仅 **0.1** |
| reset position | `resetCamera` | 恢复到初始 `(0,0,10)` / fov 60 / 单位旋转 |

---

## 4. Shader 与矩阵上传约定 (`shaderManager.py`)

顶点 shader（`simple_vertex.glsl` 等）统一是标准 MVP：
```glsl
gl_Position = projMat * (viewMat * modelMat) * vec4(aPos, 1);
```

`__setUniform` 对 `mat4` 的处理是本仓库一个**已知坑点**：
- `glm.mat4`（列主序，OpenGL 原生）→ `glUniformMatrix4fv(..., GL_FALSE, value_ptr)`。
- `numpy.ndarray`（行主序，数学约定）→ `glUniformMatrix4fv(..., GL_TRUE, ...)`，**转置后上传**。

> 代码注释里特别强调：如果把行主序的 numpy 矩阵用 `GL_FALSE` 上传，平移分量会被错误地塞进 W 行，导致**"类似透视的强烈畸变"**。当前代码对 `viewMat`/`projMat`(glm) 和 `modelMat`(numpy) 的处理都是**正确**的。

---

## 5. 窗口缩放 (resize) 一致性

`pygame.VIDEORESIZE` 时：
- `EventRegistrar.handle_events` → `glViewport(0,0,w,h)`（同时 `set_mode` 重建窗口）。
- `Camera.eventCallBacks` → `update_window_size(w,h)` → 更新 `aspect_ratio`。

两者保持一致，所以**正常情况下不会出现纵横比拉伸**。

---

## 6. 诊断：为什么"总是有很强的透视畸变"

### 6.1 结论先行
**渲染管线在数学上是正确的，没有矩阵 bug。** 用户看到的"透视畸变"是**透视投影固有的前缩(foreshortening)**，其根本原因是：

> **（修复前）引擎只有透视相机，没有正交相机；而前缩的强弱由 `场景尺寸 / 相机距离` 之比和 `fov` 决定 —— 跟 `near`/`far` 完全无关。**

> **（现状）已加入正交投影切换，见 6.4；切到 `orthographic` 即可彻底消除前缩。**

### 6.2 数值证据
用相机里**真实的** glm 代码复算（默认 fov=60, aspect=1.333, 默认场 domain `[-2,2]²`，平面在 z=0）：

**正对平面**时，4×4 正方形的四角投影到 NDC：
```
world(-2,-2,0) -> ndc(-0.260,-0.346)      world(2,-2,0) -> ndc(+0.260,-0.346)
world(-2,+2,0) -> ndc(-0.260,+0.346)      world(2,+2,0) -> ndc(+0.260,+0.346)
```
四角完全对称、w 全部 =10 → 在 1600×1200 视口上是一个**正方形**，**零畸变**。说明正对时管线毫无问题。

**倾斜平面**（模拟 arcball 旋转 50°）时，近边/远边长度比（=前缩强度）：

| 投影 | 相机距离 | top/bottom 边长比 | 含义 |
|---|---|---|---|
| 透视 | z=10（远） | **1.36** | 轻微前缩 |
| 透视 | z=4（近） | **2.24** | **强烈前缩 = 强透视畸变** |
| 正交 | z=10 | **1.00** | 完全无前缩 |

→ **相机越靠近平面（或场越大），前缩越夸张**；正交投影则恒为 1.00。

### 6.3 为什么"调 fov / near / far / 移动相机"都治不好
- **`near` / `far`**：对前缩**零影响**，只影响裁剪和深度精度。用户调它当然没用。
- **`fov`**：单独原地调 fov 只是"变焦放大"，并不会"压平"画面。要压平必须**同时把相机拉远 + 收窄 fov**（长焦效果去逼近正交）。而且 `zoom` 把 fov 夹在 [10,50]，无法更广。
- **移动相机**：`pan` 每次只动 0.1，没有快速 dolly；很难真正退到足够远。一旦相机离平面较近（例如恢复了上次保存的近距视角，或 Q 键推近过），就会"卡"在强透视里出不来。
- 默认 **fov=60° 偏广**，本身就会放大透视感。

### 6.4 修复进展与后续方向
1. ✅ **已实现：正交投影切换** —— 在 `CameraObject` 加了 `projectionMode` 选项变量（GUI 下拉框 `perspective` / `orthographic`），正交模式用 `glm.ortho` 渲染，**根治透视前缩**。半高按 `dist*tan(fov/2)` 推算，切换时取景不跳变。数值验证：倾斜 50° 平面，透视近/远边比 1.36 → 正交 **1.00**。
   - 默认仍是 `perspective`（保持原行为不突变）；若希望 2D 场默认正交，把 `create_variable_callback("projectionMode", ...)` 的列表首项改为 `"orthographic"` 即可。
2. （后续）把默认 `fov` 调小（如 35–45°），透视模式下的透视感更弱。
3. （后续）提供独立的"相机距离/ dolly"控制（而非 0.1/次的 pan），并放开 fov 上限。
4. （可选）让 `near`/`far` 可配置，解耦 `position`。

> 注：6.3 中"用户分不清变焦与压平"这一点是基于现象的**推测**；若实际畸变来自其它非默认状态（例如加载了大 domain 的数据集、或 `restore` 恢复了一个很近/很偏的相机），结论 6.1（缺正交 + 透视固有前缩）依然成立。
