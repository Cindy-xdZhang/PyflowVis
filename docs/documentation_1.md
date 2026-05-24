# Developer Instructions 

This document provides **in-depth developer-oriented documentation** for PyFlowVis, focusing on:

- **Engine–plugin architecture & GUI object development**
- **High-performance vector field visualization on CUDA**
- **How computation backends (Numba/C++/CUDA) fit into the engine**
- **How to extend or customize the system as a developer**

---

## 1. Overall Architecture (Developer View)

PyFlowVis is a **hybrid Python/C++/CUDA** framework organised into three main layers:

- **Front-end GUI (Python + ImGui)**  
  - Implemented in Python using an ImGui binding.  
  - All user-facing controls (panels, sliders, buttons, color pickers, etc.) are defined as **“GUI Objects”** which are Python classes derived from a common base (e.g. `Object`).

- **Rendering Backend (PyOpenGL)**  
  - Responsible for all screen-space rendering (vector glyphs, pathlines, scalar fields, etc.).  
  - Receives geometry and textures that are prepared by the computation backends.

- **Computation Backends (Numba/C++/CUDA)**  
  - **Numba + @njit**: fast Python-based CPU implementations for prototyping and fallback.  
  - **C++ via PyBind11**: performance-critical CPU-side algorithms (e.g. LIC renderer).  
  - **CUDA kernels**: GPU-accelerated integrators and flow diagnostics (pathlines, FTLE, optimal reference frames, etc.).

At runtime, the GUI objects:

1. Expose parameters via ImGui widgets.  
2. Trigger computation backends (Numba/C++/CUDA) when parameters change or when the user interacts.  
3. Push resulting data (e.g. trajectories, scalar fields, corelines) to the renderer for visualization.

---

## 2. Engine–Plugin Architecture

### 2.1 Core Concepts

PyFlowVis uses an **engine–plugin system**:

- The **engine** (`VisualizationEngine.py`) is responsible for:
  - Maintaining a registry of **objects/plugins**.
  - Setting up the ImGui frame, handling events, and dispatching draw/update calls.
  - Managing shared resources (e.g. loaded flow fields, OpenGL contexts).

- A **plugin / GUI object** is:
  - A Python class derived from a base class like `Object`.
  - Registered into the engine so that it appears as an ImGui panel.
  - Responsible for its own:
    - Parameters and state (using `create_variable` / `create_variable_gui`).
    - Actions (`addAction`) that may trigger computations or I/O.
    - Rendering hooks, if it draws its own geometry.

You can look at `GuiObjcts/VisualizationEngine.py` to understand how:

- Objects are instantiated and registered.
- Each frame, the engine:
  - Iterates over registered objects.
  - Renders their panels and processes their actions.

For a concrete example, see `GuiObjcts/PlanarManifold.py`, which demonstrates:

- How to define a custom object.
- How to connect GUI controls to backend logic.

### 2.2 Defining a Custom GUI Object

A minimal example (based on the snippet in `readme.md`) looks like:

```python
class GuiTest(Object):
    def __init__(self):
        super().__init__("GuiTest")
        
        # Example of GUI-bound variables
        self.create_variable_gui("boolean_var", True, False, {'widget': 'checkbox'})
        self.create_variable_gui("checkbox_int", 1, False, {'widget': 'checkbox'})
        self.create_variable_gui("input_int", 1, False, {'widget': 'input'})
        self.create_variable_gui("slider_float", 0.5, False,
                                 {'widget': 'slider_float', 'min': 0.0, 'max': 1.0})
        self.create_variable_gui("color_vec3", (255.0, 0.0, 0.0), False,
                                 {'widget': 'color_picker'})
        self.create_variable("input_vec4", [1, 1, 1, 1])
        self.create_variable_gui("default_vec4", (255, 0, 0, 0))

        # Vector/array types
        self.create_variable_gui("input_ivec3", (255, 0, 0), False, {'widget': 'input'})
        self.create_variable_gui("ivecn", (0, 0, 1, 1, 0, 2))
        self.create_variable_gui("vecn", (255, 0, 0, 0, 0, 0))

        # Plot example
        self.create_variable_gui("float_array_var_plot", [0.1, 0.2, 0.3, 0.4, 0.2], False,
                                 {'widget': 'plot_lines'})         
        self.create_variable_gui("string_var", "Hello ImGui", False, {'widget': 'input'})
        self.create_variable_gui("string_var2", "Hello ImGui", False)
        
        # Actions
        self.addAction("reload NoiseImage", lambda obj: print("reload image")) 

        # Hierarchical dictionary (nested data structure)
        testDictionary = {
            "a": 1,
            "array0": [0.1, 0.2, 0.3, 0.4, 0.2],
            "StepSize2": 3.0,
            "sonDictionary": {
                "son_a": 11,
                "array1": [0.3, 0.2, 0.3],
                "gradSondict": {
                    "gradSon_b": 22,
                    "gradVec": [1, 2, 3]
                }
            }
        }
        self.create_variable("testDictionary", testDictionary, False)
```

**Key APIs:**

- `create_variable(name, value, ...)`  
  - Registers a variable into the object without necessarily creating a GUI widget.  
  - Useful for internal state, precomputed results, etc.

- `create_variable_gui(name, value, is_readonly, config_dict)`  
  - Registers a variable **and** automatically binds it to an ImGui widget.  
  - `config_dict["widget"]` selects widget type: `'checkbox'`, `'input'`, `'slider_float'`, `'color_picker'`, `'plot_lines'`, etc.  
  - (Implementation-specific) additional keys in `config_dict` control range, layout, etc.

- `addAction(label, callback)`  
  - Adds a button or menu entry labeled `label`.  
  - When pressed, calls `callback(self)` (or similar signature depending on implementation).  
  - Typically used to trigger computations (CUDA kernels, C++ routines, etc.).

### 2.3 Registering Your Object into the Engine

While the exact registration pattern may vary, a common approach is:

1. Import your object class in a central registration file or in `VisualizationEngine.py`.  
2. Instantiate it and pass it to the engine’s object registry.

Pseudo-code (for illustration):

```python
from GuiObjcts.PlanarManifold import PlanarManifold
from GuiObjcts.GuiTest import GuiTest

def register_objects(engine):
    engine.add_object(PlanarManifold())
    engine.add_object(GuiTest())
```

Once registered, the engine will:

- Create an ImGui window for each object.
- Call its update/draw functions each frame.

---

## 3. High-Performance CUDA Vector Field Visualization

PyFlowVis accelerates several computationally expensive operations using custom CUDA kernels, as summarised in `readme.md`:

- **Streamline / Pathline integration**
- **Flow map & FTLE computation**
- **Optimal reference frame optimization**

This section explains the design philosophy, data flow, and extension strategies.

### 3.1 Supported Algorithms (CUDA Side)

1. **Pathline / Streamline Integration**
   - Integration of trajectories through the vector field using:
     - **Euler** (first-order) and **RK4** (fourth-order) integrators.  
     - GPU implementations for massive numbers of trajectories.
   - Multiple backends:
     - **CPU (Numba)**: Euler, RK4, RK5 with `@njit` acceleration.  
     - **CUDA kernels**: Euler and RK4.  
     - **Differentiable solvers** via `torchdiffeq` (dopri5, dopri8, bosh3, fehlberg2, adaptive_heun) for gradient-based tasks.

2. **FTLE (Finite-Time Lyapunov Exponent)**
   - 2D FTLE is implemented as a CUDA kernel (see `assets/cuda_kernal/FTLE_CUDA.cu`).  
   - Computes sensitivity of trajectories to small perturbations in initial positions.

3. **Optimal Reference Frame Optimization**
   - Implements the algorithm from **“Generic Objective Vortices for Flow Visualization” (Günther et al., 2017)**.  
   - Provides objective vortices in unsteady 2D/3D fields using CUDA-based optimization.

All of these share some common patterns:

- Vector field data is uploaded to the GPU (often as textures or structured arrays).  
- Seeds / grid samples are passed to CUDA kernels.  
- Kernels perform integration / optimization and write output into GPU buffers.  
- Output is either:
  - Read back to CPU for further processing, or  
  - Passed directly to the renderer (e.g. as vertex buffers, textures).

### 3.2 Data Layout and Memory Access

To achieve high performance, the CUDA kernels are designed with:

- **Coalesced global memory access**
  - Vector field samples and trajectory arrays are laid out such that consecutive threads access consecutive memory locations.

- **Shared memory usage**
  - Frequently reused data (e.g. local region of the flow field, intermediate Jacobians) is cached in **shared memory** to reduce global memory traffic.

- **Warp-aware scheduling**
  - Threads are grouped so that warps (32 threads) perform similar work (e.g. integrating a similar number of steps, accessing nearby cells).  
  - Helps reduce branch divergence and improves overall occupancy.

In practice:

- Spatial indices (i, j, k) are mapped to linear indices with a stride consistent with the global memory layout.  
- Seeds are often arranged to be **spatially contiguous** so that warps process nearby seeds.

### 3.3 Load Balancing for Trajectory Integration

Trajectory integration has **irregular workloads**:

- Paths may leave the domain early or require fewer steps.  
- Some regions may require more adaptive time stepping (in differentiable solvers).

PyFlowVis mitigates this with:

- **Chunked integration**  
  - Instead of integrating each path in a single long loop, work is broken into chunks (e.g. several fixed-size steps).  
  - This allows balancing work across warps more efficiently.

- **Early termination handling**  
  - Seeds that exit the domain or meet convergence criteria are flagged and skipped in subsequent chunks.  

The goal is to keep **SM occupancy high** and avoid having many idle threads.

### 3.4 Example: FTLE CUDA Kernel

Although the full code is in `assets/cuda_kernal/FTLE_CUDA.cu`, conceptually the steps are:

1. For each grid point \( x_0 \):
   - Integrate the flow map \( \phi_{t_0}^{t_0 + T}(x_0) \) for a fixed time horizon \( T \).

2. Approximate the **Cauchy–Green strain tensor**:
   - Sample neighboring trajectories to approximate the Jacobian \( D\phi \).  
   - Compute \( C = (D\phi)^\top (D\phi) \).

3. Compute FTLE:
   - Let \( \lambda_{\max} \) be the largest eigenvalue of \( C \).  
   - FTLE is \( \sigma = \frac{1}{|T|} \log \sqrt{\lambda_{\max}} \).

The kernel parallelizes this over grid points, making FTLE computation feasible for
large resolutions.

---

## 4. Backends and Fallback Logic

As mentioned in `readme.md`, PyFlowVis is tested on **Windows 10/11** with CUDA **11.8** and **12.6**.  
Before using CUDA, you should run `misc/TestPyCUDA.py`:

- If CUDA is available and correctly configured, **GPU implementations** will be used.  
- If CUDA is unavailable or fails, the system **falls back to CPU (Numba/C++ implementations)**.

Typical flow:

1. Python GUI object collects user parameters and seeds.  
2. Python layer decides which backend to call:
   - CUDA kernel (via `pycuda`, `cupy`, or custom bindings).  
   - C++ function (via PyBind11).  
   - Numba-accelerated Python function.
3. Results are wrapped into a form suitable for the renderer.

As a developer, you can:

- Start with a **Numba** implementation for correctness and readability.  
- Move hotspots into **C++** or **CUDA** once profiling indicates bottlenecks.

---

## 5. Integrating C++/CUDA Modules via PyBind11

For performance-critical operations (like the LIC renderer), PyFlowVis uses **PyBind11**.

- C++ sources live under `CppProjects/PybindCppModules`.  
- `LicRenderer.py` in `FLowUtils/` (note: check the exact path in the repo) demonstrates:
  - How to expose a C++ class/function to Python.  
  - How to manage GPU/CPU buffers.  
  - How to integrate with the rest of the engine.

### 5.1 Typical Workflow

1. **Create C++ source** in `CppProjects/PybindCppModules/YourModule/`.  
2. **Write CMake configuration** to build a Python extension module:
   - Link against CUDA if needed.  
   - Link against required third-party libraries (e.g. VTK, Eigen).
3. **Expose functions/classes** using PyBind11:
   - Define a `PYBIND11_MODULE` block.  
   - Wrap your C++ functions in a clean Python API.
4. **Build the module** via CMake:
   - From within `CppProjects/`:
     - `git submodule update --init`  
     - `mkdir build && cd build`  
     - `cmake .. -B .`  
     - Build using your preferred generator (e.g. Visual Studio).
5. **Use the module in Python**:
   - Import as `import your_module_name`.  
   - Integrate it into a GUI object or a utility module.

---

## 6. Developer Best Practices

### 6.1 Profiling & Optimization

- Use **Nsight Compute** to profile CUDA kernels:
  - Identify memory bottlenecks and divergence.  
  - Tune block sizes, grid dimensions, and shared memory usage.

- For Python/Numba:
  - Use built-in profilers (`cProfile`, `line_profiler`).  
  - Use Numba diagnostics to check for nopython mode and type stability.

### 6.2 Code Organization

- Keep **GUI logic** (ImGui panels, user interaction) in `GuiObjcts`.  
- Keep **algorithmic code** (integrators, FTLE, vortex detection) in dedicated modules:
  - Python/Numba modules for prototypes.  
  - C++/CUDA modules for final, high-performance implementations.

- When possible, enforce a clear separation:
  - GUI → high-level orchestration, parameter setting.  
  - Backend modules → heavy numerical computation only.

### 6.3 Extending the System

To add a new visualization or analysis technique:

1. **Design the user-facing controls**  
   - New ImGui panel? New sliders/checkboxes?  
   - Implement a new `Object` subclass in `GuiObjcts/`.

2. **Implement the algorithm**  
   - Start in Python/Numba for correctness.  
   - Once stable and necessary, port critical parts to C++/CUDA.

3. **Connect GUI to backend**  
   - From the `Object` subclass, call your backend functions when variables or actions change.  
   - Update OpenGL resources (textures, VBOs) for rendering.

4. **Test across backends**  
   - Verify CPU and CUDA produce consistent results (within numerical tolerance).  
   - Ensure CPU fallback works when CUDA is disabled.

---

## 7. Practical Setup Notes

- **Environment**  
  - Install GUI-related dependencies:
    - `pip install -r requirements_gui.txt`
  - For VortexTransformer and deep learning components:
    - `pip install -r requirements.txt`

- **CUDA & Drivers**  
  - Tested on Windows 10/11 with CUDA 11.8 and 12.6.  
  - Always run `misc/TestPyCUDA.py` after changing drivers or CUDA toolkit versions.

- **Building C++/CUDA Projects**  
  - From `CppProjects/`:
    - `git submodule update --init`  
    - `mkdir build && cd build`  
    - `cmake .. -B .`  
    - Open generated solution (e.g. `FlowGenerator.sln`) in Visual Studio to build.

---

