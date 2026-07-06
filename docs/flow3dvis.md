# 3D 流体可视化：现状审阅、性能诊断与改造路线

> 引擎：`pygame` + `PyOpenGL`(GL 4.6 core) + `imgui`。所有可视化对象都是 `GuiObjcts/` 下继承 `Object` 的
> 单例场景成员。本文审阅现有 3D 流可视化能力，**定位"基本不可用"的性能根因**，并给出分阶段改造方案。
> 结论先行：**vector glyph 是首要瓶颈（完全没有 GPU instancing，逐箭头在 Python 里烘焙几何，且重建发生在绘制路径内）**；
> 流线/路径线的**绘制**其实高效，瓶颈在 **3D 积分是纯 Python 串行**；**三维标量场可视化完全缺失**（无 `ScalarField3D`、无 iso-surface、无 DVR）。
>
> **实现进度（截至最新）**：Phase 1 实例化 glyph ✅、Phase 2 批量并行 3D 流线（~20×）✅、Phase 3 iso-surface ✅、Phase 4 DVR ✅ 均已完成。
> **验证**：全部 shader 编译链接通过；四个对象都用**离屏 GL 集成测试**（隐藏窗口→渲染到 FBO→读回像素）确认能正确渲染、无 GL 错误、且参数（sampling/isoFraction/densityScale）真正驱动输出；数值逻辑另有解析真值单测。建议仍在真实 UI 里过一遍确认交互手感。剩 Phase 5（`Basic3DFlow` 开箱包 + 3D 交互）未做。

---

## 0. TL;DR

| 能力 | 现状 | 主要问题 | 优先级 |
|---|---|---|---|
| Vector glyph (2D/3D) | 有，**已改 GPU 实例化** | ~~无 instancing，Python 逐箭头烘焙~~ → **Phase 1 已完成**（单模板 + 每实例缓冲 + `glDrawArraysInstanced`） | **P0 ✅** |
| Streamline / Pathline | 有，**3D 已批量并行** | 绘制本就OK；~~3D 积分纯 Python 串行~~ → **Phase 2 已完成**（numba `prange` 批量积分 + numpy 缓冲，实测 **~20× 加速**） | **P1 ✅** |
| 3D 标量场 iso-surface | **已实现** | **Phase 3 已完成**（`IsoSurfaceObject`：从活动 3D 场派生 Q/λ₂/涡量/幅值/IVD + Marching Cubes） | **P1 ✅** |
| 3D 标量场 DVR (体渲染) | **已实现** | **Phase 4 已完成**（`VolumeRenderObject`：3D 纹理 + 代理立方体片元 ray-marching + 1D colormap 当 transfer function） | **P2 ✅** |
| 3D 流"开箱包" | ❌ 无 | `buildWorkLoads("Basic3DFlow")` 是空的 `pass` | P2 |

---

## 1. 渲染引擎架构（改造前必须理解的机制）

- **主循环** [`VisualizationEngine.MainLoop`](../GuiObjcts/VisualizationEngine.py:81)：`while running: handle_events → glClear → scene.render_all() → imgui → flip`。
  **无帧率限制、无 vsync 无关计时**；每帧对场景里每个对象调用一次 `render()`。因此**任何放进 `render()` 的重活都会直接拖垮帧率**。
- **对象/材质系统**：`Material.apply([scene, camera, obj])` → `ShaderProgram.setUniformScope(...)`
  [`shaderManager.py:319`](../GuiObjcts/shaderManager.py:319)。它把 scene / camera / object 各自 `getScope()` 返回的变量字典合并，
  **按名字匹配**逐个 `glUniform*`。所以：**要给 shader 传一个 uniform，只需某个对象里有同名变量**。
  - Camera 提供 `projMat`、`viewMat`；对象提供 `modelMat`、`color` 等。
  - `checkUniforms()` 会对"shader 里有但没被赋值"的 active uniform 报 warning——**新 shader 的每个 uniform 必须被某个 scope 对象提供**。
- **顶点属性约定**：`location 0 = vec3 位置`，`location 1 = vec2 texcoord`（`VertexArrayObject.init`）；
  flowline 用 `loc0=vec3 pos, loc1=vec2 (time, attrib)`。
- **几何缓冲**（`VertexArrayObject`）：几何以 **Python list** 累积（`self.vertex_geometry` 等），`commit()` 时
  `np.array(list, float32)` 再 `glBufferData(GL_STATIC_DRAW)`，`render()` 用 `glDrawElements`。
  → 对**静态**几何没问题，对**每帧重建**的几何是灾难（见 §3.1）。
- **3D 数据模型已就绪**：[`UnsteadyVectorField3D`](../FLowUtils/VectorField3d.py:200) 存 `field: (T, Z, Y, X, 3) float32`，
  配 numba 四线性插值 `_quadrilinear_interpolate_njit`。**这个内存布局可直接上传为 GL 3D 纹理**（DVR/GPU 采样用得上）。

---

## 2. 现有 3D 流可视化能力清单

1. **Vector glyph** [`vectorGlyphObject.py`](../GuiObjcts/vectorGlyphObject.py)：`VertexArrayVectorGlyph`。
   **Phase 1 重写后**：numba 插值 (`_interpolate_2d/3d_vectorized_core_numba`) 只产出"每实例" `(pos, vec)` 数组，
   由 `_rebuild_instances` 上传到 instance VBO，`render()` 一次 `glDrawArraysInstanced` 画完全部箭头（模板来自 `_build_arrow_template`）。
   *（改造前*的问题分析见 §3.1，作为设计动机保留。）
2. **Streamline / Pathline** [`FlowLineRenderObject.py`](../GuiObjcts/FlowLineRenderObject.py)：2D/3D 都有。
   - 绘制：单条交错 VBO（5 float/顶点 = pos3 + (time, attrib)），一次 `glMultiDrawArrays(GL_LINE_STRIP_ADJACENCY,...)`
     [`FlowLineRenderObject.py:320`](../GuiObjcts/FlowLineRenderObject.py:320)，几何着色器把线段扩成**屏幕对齐 ribbon**
     [`flowline_geometry.glsl`](../assets/shaders/flowline_geometry.glsl)。这一路**已经是高效的 GPU 做法**。
   - 积分：2D pathline 有 CUDA/CPU 自动后端 (`compute_pathline_2D_auto`)；**3D 与 2D streamline 走纯 Python `list(map(...))`**。
3. **数据模型**：`UnsteadyVectorField3D` ✅；`ActiveFieldObj.insertField` 已接受 3D 向量场。
4. **缺失**：
   - **无 3D 流开箱包**：`buildWorkLoads("Basic3DFlow")` 是 `pass`，3D 场只能像 [`main.py:162`](../main.py:162) 注释那样手动 `insertField` 塞进 2D 场景。
   - **无任何 3D 标量场可视化**：没有 `ScalarField3D`（只有 `ScalarField2D`）；`insertScalarField` 被 `@typechecked` 限死为 `ScalarField2D`
     [`ActiveFieldObject.py:215`](../GuiObjcts/ActiveFieldObject.py:215)；标量算子 `MAGNITUDE/CURL/Q_CRITERION/LAMBDA2/IVD` 只对 2D。
     → iso-surface、DVR 曾从零开始；**iso-surface 已由 Phase 3 补上（`FLowUtils/ScalarField3d.py` 自带 3D 标量派生，绕开 2D 管线）；DVR 仍待做（Phase 4）**。

---

## 3. 性能诊断（"基本不可用"的根因）

### 3.1 Vector glyph —— 首要瓶颈（P0，**改造前**分析；Phase 1 已修复）

问题不在插值（numba 已经快），而在**几何生成与上传**：

1. **完全没有 GPU instancing。** `_batch_generate_glyphs` 用 **Python for 循环**遍历每个采样点，
   对每个箭头调 `_generate_single_arrow` 用 numpy 逐个算 `norm/cross`、拼 Python list，再
   `all_vertices.extend(...)` + `[idx + offset for idx in arrow_indices]`。复杂度 **O(N·segments) 的纯 Python 工作**，N 是箭头数。
   3D 密采样下 N 轻松到 10⁵–10⁶（`(domain/sampling)³`），**每个箭头 6 个三角形全在 CPU 烘焙成一整块 buffer**。
2. **重建发生在绘制路径里。** `render()` 每帧被调；一旦 `dirty` 就
   `erase()` → 重新插值 → Python 烘焙 → `commit()`。而**任何参数改动（scale/radius/sampling）或时间前进都会把 `dirty=True`**
   （[`ActiveFieldObject.timedirtyCallBack`](../GuiObjcts/ActiveFieldObject.py:33)）。**播放动画 = 每帧全量 CPU 重建 → 帧被阻塞 → 无交互。**
3. **`commit()` 每次全量重传。** [`commit`](../GuiObjcts/VertexArrayObject.py:134) 把上百万元素的 Python list `np.array(...)` 化，再
   `glBufferData`（整块重新分配，而非 `glBufferSubData`/orphaning）。
4. **附带的正确性问题**：`_generate_single_arrow` 先 `direction/=norm` 再算 `velocity_mag=norm(direction)`（恒为 1），**箭头长度丢失了速度信息**；
   `textureCoords=[]` 后又拼接 circle 的 texcoord，导致顶点数/纹理数不一致。

> **修法（P0，本次已实现）**：改为 **GPU 实例化**——上传一个箭头模板网格（一次），外加一个"每实例"缓冲
> `(pos.xyz, vec.xyz)`，顶点着色器按方向构基、按速度缩放、平移到采样点，`glDrawElementsInstanced` 一次画完。
> 重建从"O(N·segments) Python 烘焙 + 上百万浮点上传"降为"**numba 插值 + N×6 浮点的小上传**"，无 CPU 几何烘焙。

### 3.2 Streamline / Pathline（P1，**改造前**分析；Phase 2 已修复 3D 积分）

- **绘制路径 OK**：一次 `glMultiDrawArrays` + geometry shader 扩 ribbon，不用动。
- **瓶颈是积分**：[`pathline_integration_one_direction_2D`](../FLowUtils/flowlineIntegral.py:10) 是**纯 Python for 循环**，RK4 每步 4 次插值、
  `path.append((pos.copy(), t))` 逐步拷贝 numpy。**3D 流线/路径线走 `list(map(compute_*_3D, args_list))` 串行**，
  每个 seed 一条，Nseeds × 5000 步 × 4 插值全在 Python 层 → 多 seed 时秒级卡顿。
- **建缓冲慢**：[`MappingFlowlineAsRenderingVAO`](../GuiObjcts/FlowLineRenderObject.py:462) 逐顶点 append 一个 `[x,y,z,t,label]` Python list，再 `np.array`。
- **视觉**：`flowline_geometry.glsl` 生成的是**屏幕对齐 ribbon**（`viewingDirection=vec3(0,0,-1)` 硬编码），3D 下不是有光照的真三维 tube——够用但可作为后续升级项。

> **修法（P1）**：把 3D 积分**批量化/并行化**（numba `prange` 或 torch 批积分，复用 2D 已有的 CUDA 后端思路），
> 用 numpy 直接组装交错缓冲（去掉逐顶点 Python append）。可选：真 3D tube 着色。

### 3.3 结构性问题

- 主循环无帧率上限；重活放在 `render()` 内；用一个 `dirty` 位把"参数变更"和"每帧"混为一谈。
- 建议后续：把"数据重建"与"绘制"分离；仅在数据真变时重建，绘制永远只发 GPU draw call。

---

## 4. 改造路线（分阶段）

- **Phase 1（P0）✅ 已完成**：GPU 实例化 vector glyph。新增 `assets/shaders/glyph_instanced_{vertex,fragment}.glsl`，
  在 `shaderManager` 注册 `glyphInstancedMat`，重写 `VertexArrayVectorGlyph`（`_build_arrow_template` + `_rebuild_instances` + `glDrawArraysInstanced`）。
  numba 插值核已用 `prange` 真并行。已通过离屏 GL 集成测试（渲染出 1413px、sampling 联动）+ 数值单测验证。
- **Phase 2（P1）✅ 已完成**：3D 流线/路径线积分**批量并行化**。`flowlineIntegral.py` 新增 numba `prange` 批量积分核
  (`compute_streamlines_3D_batch` / `compute_pathlines_3D_batch`，RK4/Euler；其它方法自动回退串行)，
  `FlowLineRenderObject` 新增 `MappingBatchedFlowlineAsRenderingVAO` 用 numpy 直接组装 VBO（循环按线数而非顶点数）。
  数值上与串行严格一致（路径线误差 0.0、流线 3e-6）；实测 150 条流线 ~52 万点 **15.7s → 0.76s（~20×）**；离屏渲染确认出线（8005 顶点）。
  *（仍待办*：2D streamline 仍串行；真 3D tube 光照仍是屏幕对齐 ribbon。）
- **Phase 3（P1）✅ 已完成**：三维标量场 **iso-surface**。新增 `FLowUtils/ScalarField3d.py`（magnitude/vorticity/Q/λ₂/IVD 单切片派生 + `ScalarField3D` holder + `marching_cubes_world` 包 skimage），
  `GuiObjcts/IsoSurfaceObject.py`（自有 pos+normal VAO+EBO，两级 dirty：标量体/网格；`isoFraction∈[0,1]` 滑条；`scalarOperation` 下拉），
  新 shader `iso_surface_{vertex,fragment}.glsl`（双面 Lambert），注册 `isoSurfaceMat`，加入 `buildWorkLoads`。
  解析真值验证：刚体旋转 **涡量=2 / Q=1 / λ₂=−1 精确**，MC 球面顶点在球上、法线径向；离屏渲染确认出面且 isoFraction 联动。**自包含**：直接从活动 3D 速度场派生标量（经 `velocity_slice` 直接取切片，绕开有既有 bug 的 `getSlice`），不依赖 2D 标量管线；活动场非 3D 时不绘制。
- **Phase 4（P2）✅ 已完成**：**DVR 体渲染**。新增 `GuiObjcts/VolumeRenderObject.py`（标量体传为 `GL_R32F` 3D 纹理，代理立方体前面剔除→背面光栅，片元里 `inverse(viewMat)` 求相机→射线-盒相交→沿射线采样→front-to-back 预乘合成）+ shader `dvr_{vertex,fragment}.glsl`，注册 `dvrMat`，复用 `TextureManager` 的 `colorMaps1Darray` 当 transfer function（`colorMap` 下拉 + `densityScale`/`numSteps` 可调），加入 `buildWorkLoads`。
  立方体绕序（全部朝外 CCW）已单测验证；离屏渲染确认体渲染填充（44k px）且 densityScale 联动；标量派生复用 Phase 3。**默认 `draw=False`**（覆盖式渲染，深度测试关，在 Objects 菜单显式开启）；仅活动场为 3D 时绘制。**局限**：作为半透明 overlay 不与其它 3D 物体做深度交叠（Phase 4 够用，后续可加 depth-aware 合成）。
- **Phase 5（P2）**：`Basic3DFlow` 开箱包 + 3D 相机/裁剪面交互。

### 关键技术设计要点
- **Instanced glyph**：模板箭头沿 +Z、单位长；每实例属性 `loc2=iPos(vec3)`、`loc3=iVec(vec3)`，`glVertexAttribDivisor=1`；
  顶点着色器由 `dir=normalize(iVec)` 构正交基，长度 ∝ `|iVec|`，颜色按速度归一化。uniform 全部由 glyph/camera 提供（`projMat/viewMat/modelMat/color/scale/radius/height/maxSpeed`）。
- **Iso-surface**：标量体 → MC → `(vertices, normals, indices)`；用现有材质约定（`loc0=pos, loc1` 可放 normal）加一个带 Lambert 光照的 fragment。阈值可交互（仅阈值变才重算 MC）。
- **DVR**：`glTexImage3D(GL_R32F)` 上传标量体；渲染代理 box 的背面，片元里从相机出发 ray-march，累积 `front-to-back` alpha；transfer function 用 `GL_TEXTURE_1D_ARRAY` 采样。早停 + 步长/最大步数做 uniform 以平衡质量/性能。

---

## 5. 验证方式（本轮实际采用）
- **离屏 GL 集成测试（关键手段）**：用 `pygame.display.set_mode(size, OPENGL | HIDDEN)` 开**隐藏窗口**拿到真 GL 上下文，
  headless 起完整引擎、灌入合成 3D 涡场、逐对象渲染到默认 framebuffer，再 `glReadPixels` 统计**非背景像素覆盖** + 查 `glGetError`。
  这样即使看不到 UI，也能验证"是否真的画出来了 / 有没有 GL 错误 / 参数是否驱动输出"。本轮四个对象据此全部验证通过，
  并借此抓出并修复了 iso/DVR 踩到的 `getSlice` bug（glyph 1413px、iso 2308px、DVR 44k px、flowline 8005 顶点，均 `glError=0`）。
- **数值逻辑**用解析真值单测：插值 vs numpy 参考、批量积分 vs 串行（路径线误差 0）、涡量/Q/λ₂ vs 刚体旋转、MC vs 解析球面。
- 以上均为**临时测试，验证后已删除**；最终仍建议在真实 UI `python main.py` 目测帧率与交互手感。

---

## 6. 真实 UI 复查：问题定位与修复（2026-07-06）

> 在真实 `python main.py`（活动场 = `tornado3d.nc`：128³ **steady** 场，domain `[-10,10]³`，`time_steps=1, tmin=tmax=0`）里逐项复查，发现三个可用性问题并修复。
> **关键教训：验证靠离屏截图（二进制），不靠日志文本**——本轮排查中终端/工具输出多次被污染并伪造数据，唯有 `glReadPixels` 存出的 PNG 可信。

### 6.1 打开体渲染 → 所有 imgui 窗口消失、程序失控（P0，已修）

**现象**：勾选 VolumeRender 的 `draw` 后整屏变成一片不透明淡蓝，GUI 全不可见、无法操作。

**先排除**：DVR **不崩溃、不慢**——离屏实测 DVR-on **75–81 fps**（比 DVR-off 的 38 fps 还快），`glError=0`，`render_all` 无异常。

**根因 = texture unit 泄漏污染 imgui**：DVR 的 material 含 `sampler3D volumeTex` + `sampler1DArray colorMaps1Darray`；[`set_sampler_uniform`](../GuiObjcts/shaderManager.py:298) 用 `glActiveTexture(GL_TEXTURE0+n)`+`glBindTexture` 绑定后**从不恢复**，渲染完 active unit 停在非 0、unit 0 上残留 `GL_TEXTURE_3D`。而 pygame 的 imgui 是**固定管线** renderer，假设 unit 0 = 自己的 2D 字体图集 → 整个 GUI 无法绘制成 DVR 淡蓝。逐状态探针证实：只关 depth test **无效**（仍糊屏），只 `glActiveTexture(GL_TEXTURE0)`+解绑即恢复（imgui 区域 RGB 从 `[150,169,201]` 回到 `[20,23,28]`）。
> iso-surface / glyph 不触发,是因为它们的 material **不含任何 sampler 纹理**；DVR 是第一个用 3D 纹理的对象故最先暴露。`flowline`/`colormap` 用的 `builtIn` 1D-array 也有同样隐患，被下面的全局兜底一并覆盖。

**修复**：
- [`VisualizationEngine.MainLoop`](../GuiObjcts/VisualizationEngine.py:90)：每帧 `render_all` 后、imgui 前，`glActiveTexture(GL_TEXTURE0)` + 解绑 `GL_TEXTURE_3D`/`GL_TEXTURE_1D_ARRAY`（**治本**，覆盖所有对象的纹理泄漏）。
- [`VolumeRenderObject.render`](../GuiObjcts/VolumeRenderObject.py:133)：结尾解绑自己绑的 3D/1D-array 纹理（对象自洽，与它已恢复 depth/blend/cull 一致）。

### 6.2 体渲染糊屏：铺满不透明雾（已修）

即便 GUI 正常，DVR 也会把整个 domain ray-march 成几乎不透明的一坨盖住场景——相机默认 `(0,0,10)` 正贴 domain 的 `z=10` 面，代理立方体铺满视野，加上 `densityScale=20` 过高（`a = sn·density·dt` 很快 clamp 到 1）。**修复**：[`densityScale` 默认 `20→4`](../GuiObjcts/VolumeRenderObject.py:48)。仍是 depth-test-off 的半透明 overlay（Phase 4 已知局限），不与其它 3D 物体深度交叠；彻底解决需 depth-aware 合成。

### 6.3 3D streamline / pathline 不显示（streamline 已修；pathline 是数据性质）

- **streamline**：积分核本就正常（离屏实测单种子 603 点、12 种子 `[439,226,462,…]`）。真因是**种子进错组**——[`randomReseeding`](../GuiObjcts/Indicator.py:80) 只填 `SeedingGroup0`（`activeSeedingGroup` 默认 0），而 [`update_streamline`](../GuiObjcts/FlowLineRenderObject.py:326) 读 `SeedingGroup1` → 默认空 → 无线。附带 2D-only 遗留：`z` 硬编码 0、[`denseReseeding`](../GuiObjcts/Indicator.py:209) 是空 `pass`、鼠标右键只命中 z=0 的 `plane`。**修复**：播种支持 3D（z 按 domain 分布）+ **同时填 group0/group1**（pathline 用 0、streamline 用 1）+ 实现 `denseReseeding`（规则网格，3D 第三轴 cap 到 6 防爆炸）。修复后实测 `vertex_count=8021`（12 条线）。
- **pathline**：`tornado3d` 是 **steady** 场，时间不前进 → 积分第一步即终止，只剩种子 1 点，`<2` 被 [`MappingBatchedFlowlineAsRenderingVAO`](../GuiObjcts/FlowLineRenderObject.py:557) 丢弃。**非 bug**（steady 场没有有意义的 pathline）；仅在 [`update_pathline`](../GuiObjcts/FlowLineRenderObject.py:411) 加日志提示，不再静默消失。要测 pathline 需换 unsteady 3D 场。

### 6.4 验证（本轮）
离屏起真实引擎 + 灌 `tornado3d` + 跑完整帧（含上述修复），`glReadPixels` **在 flip 之前**读回存 PNG：DVR-on 时全部 GUI 窗口 + 强制测试窗口清晰可见、背景半透明淡蓝；streamline `vertex_count=8021`；两个种子组各 12 点、streamline 组 z∈`[-9.71,8.53]`（真 3D 分布）；`densityScale` 默认 4。
