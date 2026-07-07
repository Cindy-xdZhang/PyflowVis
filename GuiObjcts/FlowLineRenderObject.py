from GuiObjcts.VertexArrayObject import *
from .VisualizationEngine import getEngine
from GuiObjcts.shaderManager import *
from FLowUtils.VectorField2d import *
from FLowUtils.VectorField3d import *
from OpenGL import GL as gl 
import ctypes
import numpy as np
from FLowUtils.ScalarField2d import *
from FLowUtils.flowlineIntegral import *
from FLowUtils.KillingObserver2D import compute_reference_frame_transformation_from_field, ReferenceFrameTransformation2D

from GuiObjcts.ActiveFieldObject import *







class _LineBuffer:
    """One GL_LINE_STRIP_ADJACENCY multi-draw buffer (vertex layout: pos(x,y,z) + (time, attrib)).
    Pathlines and streamlines each own one so they render independently and never overwrite or
    erase each other (they used to share a single VBO)."""

    def __init__(self):
        self.vertex_count = 0
        self.offsets = np.array([], dtype=np.uint32)
        self.sizes = np.array([], dtype=np.uint32)
        self.vao_id = gl.glGenVertexArrays(1)
        self.vbo_id = gl.glGenBuffers(1)
        gl.glBindVertexArray(self.vao_id)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_id)
        stride = 5 * 4  # 5 float32 = 20 bytes
        gl.glEnableVertexAttribArray(0)  # layout(location=0) in vec3 in_position
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(1)  # layout(location=1) in vec2 in_attribs(time, attrib)
        gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(3 * 4))
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)

    def upload_vertices(self, vertex_attributes):
        va = np.ascontiguousarray(vertex_attributes, dtype=np.float32)
        self.vertex_count = len(va)
        gl.glBindVertexArray(self.vao_id)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_id)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, va.nbytes, va if va.size else None, gl.GL_STATIC_DRAW)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)

    def set_indices(self, offsets, sizes):
        self.offsets = np.array(offsets, dtype=np.uint32)
        self.sizes = np.array(sizes, dtype=np.uint32)

    def erase(self):
        self.vertex_count = 0
        self.offsets = np.array([], dtype=np.uint32)
        self.sizes = np.array([], dtype=np.uint32)
        gl.glBindVertexArray(self.vao_id)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_id)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, 0, None, gl.GL_DYNAMIC_DRAW)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)

    def draw(self):
        if self.vertex_count == 0 or len(self.offsets) == 0:
            return
        gl.glBindVertexArray(self.vao_id)
        gl.glMultiDrawArrays(gl.GL_LINE_STRIP_ADJACENCY, self.offsets, self.sizes, len(self.offsets))
        gl.glBindVertexArray(0)


class FlowLineObject(Object):
    def __init__(self):
        super().__init__("Flowline")
        self.engine = getEngine()
        self.parentScene=getScene()
        assert self.parentScene is not None,  "scene is not set"

        self.pathline_dirty = True
        self.streamline_dirty = True

        
        self.__initDynamicTypeGLContext__()
        #  material
        self.create_variable("modelMat",np.eye(4,dtype=np.float32),False,False)
        self.material = Material("flowlineMaterial",  "flowlineMat",texture0="builtIn")
        self.setMaterial(self.material)


        def dirtyCallBack(obj):
            if self.getValue("pathline_active"):
                setattr(self, "pathline_dirty", True)
            if self.getValue("streamline_active"):
                    setattr(self, "streamline_dirty", True)

        self.create_variable("streamline_active",True,True)
        self.create_variable("pathline_active",True,True)
        self.create_variable_gui("lineWeight",0.1,True, {'widget': 'slider_float', 'min': 0.0, 'max': 5.0})
        self.create_variable("zOffset", 0.0,True)
        self.create_variable_gui("uplifting",0.1,True, {'widget': 'slider_float', 'min': 0.0, 'max': 5.0})
        self.create_variable("colorMap",self.engine.getBuiltInTextureNames(),True)
        self.create_variable_gui("ColorCodingAttribute", ["attrib0", "attrib1"], True)
        self.create_variable("maxIteration", 5000,True)

        self.create_variable_callback("integrator", ["RK4","Euler","RK5","dopri5","dopri8","bosh3","fehlberg2","adaptive_heun"],dirtyCallBack,True)#euler,rk4
        #neural ODE solver method:
        # dopri8 Runge-Kutta of order 8 of Dormand-Prince-Shampine.
        # dopri5 Runge-Kutta of order 5 of Dormand-Prince-Shampine [default].
        # bosh3 Runge-Kutta of order 3 of Bogacki-Shampine.
        # fehlberg2 Runge-Kutta-Fehlberg of order 2.
        # adaptive_heun Runge-Kutta of order 2.
        # https://github.com/rtqichen/torchdiffeq

        self.create_variable_callback("stepSize", 0.01 ,dirtyCallBack,True)

        getEngine().eventRegister.registerChannelEvent("seeding_changed", lambda : dirtyCallBack(self))
        

        #########################################################
        # for observer-relative transformation (UI variables)
        #########################################################
        self.create_variable("InputFieldMaxTime", 0.0, False, False)
        self.create_variable("InputFieldMinTime", 0.0, False, False)
        self.create_variable("DefaultObseverMaxTime", 0.0, False, False)
        self.create_variable("DefaultObseverMinTime", 0.0, False, False)
        self.create_variable_callback("transformationMode", ["none", "observed", "inverse"], lambda obj: None, True, True)
        # create SSBO for worldline/integratedC (binding=6 in shader)
        self.ssbo0_id = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, self.ssbo0_id)
        gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER, 0, None, gl.GL_DYNAMIC_DRAW)
        gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, 6, self.ssbo0_id)
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, 0)
        self.create_variable("SSBOBufferId0",  self.ssbo0_id, False, False)
        self.create_variable("ssbo0_length",  int(0), False, False)
        self.create_variable("worldlineStartPos0", (0.0, 0.0), False, False)

        def change_reference_frame_action():
           
            vector_field:UnsteadyVectorField2D = self.activeFieldWidget.getActiveObserverField()
            if vector_field is None or vector_field.getDim() != 2:
                print("Changing reference frame is only implemented for 2D active field.")
                return
            indicator = self.parentScene.getObject("Indicator") if self.parentScene.hasObject("Indicator") else None
            if indicator is None:
                return
            seeds = indicator.getValue("SeedingGroup0")
            if not seeds:
                return
            start_pos3d, t0 = seeds[-1]  
            step_size = self.getValue("stepSize")
            obs_time = self.activeFieldWidget.time()
            try:
                rf:ReferenceFrameTransformation2D = compute_reference_frame_transformation_from_field(
                    vector_field=vector_field,
                    start_pos3d=start_pos3d,
                    t0=t0,
                    stepSize=step_size,
                    observation_time=obs_time,
                    maxIterations=self.getValue("maxIteration"),
                    method=self.getOptionValue("integrator")
                )
                self.__setFrameTransformationBuffers(rf.worldline.pathline, rf.integrated_rotation, start_pos3d, obs_time)
            except Exception as e:
                print(f"[ReferenceFrame] Failed: {e}")
        def reset_reference_frame_action():
            self.__resetFrameTransformationBuffers()
        self.addAction("Change reference frame by integrating observer field", lambda obj: change_reference_frame_action())
        self.addAction("Reset reference frame", lambda obj: reset_reference_frame_action())

        

        

        def pathline_integrate_torch_action():
            from FLowUtils.neuralVectorFieldODE import integrate_pathline2D_torch,UnsteadyVectorField2D_Torch
            actFieldWidget = self.parentScene.getObject("ActiveField")
            time_current = actFieldWidget.time()
            vector_field:UnsteadyVectorField2D = actFieldWidget.getActiveField()
            if vector_field is None:
                return
            self.indicatorObject=self.parentScene.getObject("Indicator") if self.parentScene.hasObject("Indicator") else None
            if self.indicatorObject is None:
                return
            
            seeds_data = self.indicatorObject.getValue("SeedingGroup0")
            if not seeds_data:
                self.pathline_buf.erase()
                self.pathline_dirty = False
                return

            start_positions = np.array([p[0][:2] for p in seeds_data], dtype=np.float32)

            vf_torch = UnsteadyVectorField2D_Torch(
                data_tensor=torch.from_numpy(vector_field.getDataAsNumpy()).float(),
                domain_min=vector_field.domainMinBoundary,
                domain_max=vector_field.domainMaxBoundary,
                t_min=vector_field.tmin,
                t_max=vector_field.tmax
            )
            
            time_min = vector_field.getMinTime()
            time_max = vector_field.getMaxTime()
            step_size = self.getValue("stepSize")
            method = self.getOptionValue("integrator")
            if method not in ["dopri5","dopri8","bosh3","fehlberg2","adaptive_heun"]:
                method = "dopri5"

            forward_cache = integrate_pathline2D_torch(
                vf_torch, start_positions, time_current, time_max, step_size, method
            )
            backward_cache = integrate_pathline2D_torch(
                vf_torch, start_positions, time_current, time_min, step_size, method
            )

            pathline_cache = []
            for i in range(len(start_positions)):
                bwd_path = backward_cache[i][::-1]
                fwd_path = forward_cache[i][1:]
                full_path = bwd_path + fwd_path
                pathline_cache.append(full_path)
                
            self.MappingFlowlineAsRenderingVAO(pathline_cache, self.pathline_buf)
            self.pathline_dirty = False
        self.addAction("pathline_integrate_torch", pathline_integrate_torch_action)

    def setMaterial(self,material) -> None:
        self.material=material

    def erase(self) -> None:
        # Clear both line buffers (kept for any external caller; internal code erases per-type).
        self.pathline_buf.erase()
        self.streamline_buf.erase()


    def postInit(self):
        super().postInit()
        self.activeFieldWidget = self.parentScene.getObject("ActiveField")
        if self.activeFieldWidget is None:
            logging.getLogger().error("No ActiveField object found in scene, return")
            return
  
    def __initDynamicTypeGLContext__(self):
        # Pathlines and streamlines each get their OWN buffer, so they render independently and
        # never overwrite/erase one another. Previously they shared one VBO, so clearing one
        # seeding group (or a pathline erase on empty seeds) wiped the other's lines.
        self.pathline_buf = _LineBuffer()
        self.streamline_buf = _LineBuffer()


    def __resetFrameTransformationBuffers(self):
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, self.ssbo0_id)
        gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER, 0, None, gl.GL_DYNAMIC_DRAW)
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, 0)
        self.setValue("ssbo0_length",  int(0), callback=False)
        self.setValue("worldlineStartPos0", (0.0, 0.0), callback=False)
        self.updateOptionValue("transformationMode", "none")

    def __setFrameTransformationBuffers(self, pathline, integratedRotation, reference_point, observationTime:float):
        """
        Python 版本：将 worldline 与 integratedRotation 写入 SSBO(binding=6)，并设置相关 UI 变量。
        - pathline: [(pos3d, t), ...]
        - integratedRotation: [theta_i]
        - reference_point: (x0, y0)/(x0, y0, z0)
        """
        if pathline is None or integratedRotation is None or len(pathline) != len(integratedRotation) or len(pathline) == 0:
            print("[setFrameTransformationBuffers] invalid inputs")
            return
        worldlinetimeMin = float(pathline[0][1])
        worldlinetimeMax = float(pathline[-1][1])
        activeFieldTimeMin = float(self.activeFieldWidget.getActiveField().getMinTime())
        activeFieldTimeMax = float(self.activeFieldWidget.getActiveField().getMaxTime())

        # pack vec4(x, y, theta, time)
        buf = np.zeros((len(pathline), 4), dtype=np.float32)
        for i, (p3, t) in enumerate(pathline):
            buf[i, 0] = float(p3[0])
            buf[i, 1] = float(p3[1])
            buf[i, 2] = float(integratedRotation[i])
            buf[i, 3] = float(t)
        # upload SSBO
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, self.ssbo0_id)
        gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER, buf.nbytes, buf, gl.GL_DYNAMIC_DRAW)
        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, 0)
        # update variables
        self.updateOptionValue("transformationMode", "observed")
        self.setValue("DefaultObseverMaxTime", worldlinetimeMax, callback=False)
        self.setValue("DefaultObseverMinTime", worldlinetimeMin, callback=False)
        self.setValue("InputFieldMaxTime", activeFieldTimeMax, callback=False)
        self.setValue("InputFieldMinTime",  activeFieldTimeMin, callback=False)
        # record counts for shader uniform (we set at draw time)
        self.setValue("ssbo0_length", int(buf.shape[0]), callback=False)
        self.setValue("worldlineStartPos0", (float(reference_point[0]), float(reference_point[1])), callback=False)

    def setFrameTransformation(self, instaneousC, worldline, timerange, observationTime:float):
        """
        Python 版本：从瞬时角速度与世界线构建 integratedRotation，然后调用 setFrameTransformationBuffers。
        - instaneousC: [omega_i]
        - worldline: [(pos3d, t), ...]
        - timerange: (tmin, tmax)
        """
        if timerange[0] >= timerange[1]:
            print("[setFrameTransformation] Time range error.")
            return
        if not worldline or not instaneousC or len(worldline) != len(instaneousC):
            print("[setFrameTransformation] invalid worldline/instC inputs")
            return
        min_obs_time = worldline[0][1]
        max_obs_time = worldline[-1][1]
        N = len(instaneousC)
        dt = (max_obs_time - min_obs_time) / float(max(1, N - 1))
        k = 0
        for (_, tt) in worldline:
            if tt < observationTime:
                k += 1
            else:
                break
        if k == N:
            k = N - 1
        integratedRotation = [0.0] * N
        integratedRotation[k] = 0.0
        # backward
        for i in range(k, 0, -1):
            angular_c = float(instaneousC[i - 1])
            integratedRotation[i - 1] = integratedRotation[i] + (-dt) * angular_c
        # forward
        for i in range(k, N - 1):
            angular_c = float(instaneousC[i + 1])
            integratedRotation[i + 1] = integratedRotation[i] + (+dt) * angular_c
        reference_point = (float(worldline[k][0][0]), float(worldline[k][0][1]))
        self.__setFrameTransformationBuffers(worldline, integratedRotation, reference_point, observationTime)


    def render(self):
        #update if dirty
        if self.getValue("pathline_active"):
            self.update_pathline()
        if self.getValue("streamline_active"):
            self.update_streamline()

        draw_path = self.getValue("pathline_active") and self.pathline_buf.vertex_count > 0
        draw_stream = self.getValue("streamline_active") and self.streamline_buf.vertex_count > 0
        if not (draw_path or draw_stream):
            return
        # One material/uniform setup; pathline and streamline share the shader but their own VAOs.
        if self.material is not None:
            self.material.apply([self.parentScene, self.cameraObject, self])
        if draw_path:
            self.pathline_buf.draw()
        if draw_stream:
            self.streamline_buf.draw()
        gl.glUseProgram(0)

    def update_streamline(self,  attribs2=None):
        if not hasattr(self, 'streamline_dirty') or not self.streamline_dirty:
            return

        self.indicatorObject=self.parentScene.getObject("Indicator") if self.parentScene.hasObject("Indicator") else None
        if self.indicatorObject is None:
            return

        actFieldWidget = self.parentScene.getObject("ActiveField")
        if actFieldWidget is None:
            return

        vector_field:UnsteadyVectorField3D = actFieldWidget.getActiveField()
        if vector_field is None :
            return

        time = actFieldWidget.time()
        seeds = self.indicatorObject.getValue("SeedingGroup1")
        # No streamline seeds -> clear ONLY the streamline buffer; pathlines are untouched.
        if not seeds:
            self.streamline_buf.erase()
            self.streamline_dirty = False
            return

        step_size = self.getValue("stepSize")
        max_iteration = self.getValue("maxIteration")
        method = self.getOptionValue("integrator")

        args_list = [
            (vector_field, pos3d, time, step_size, max_iteration, method)
            for pos3d, _ in seeds
        ]

        if vector_field.getDim() == 3:
            # Fast path: numba parallel batch integration + numpy VBO assembly.
            seeds_xyz = np.array([np.asarray(pos3d, dtype=np.float32)[:3] for pos3d, _ in seeds], dtype=np.float32)
            batched = compute_streamlines_3D_batch(vector_field, seeds_xyz, time, step_size, max_iteration, method)
            if batched is not None:
                self.MappingBatchedFlowlineAsRenderingVAO(*batched, self.streamline_buf)
            else:  # unsupported integrator (e.g. RK5) -> serial fallback
                self.streamline_cache = list(map(compute_streamline_3D, args_list))
                self.MappingFlowlineAsRenderingVAO(self.streamline_cache, self.streamline_buf)
        else:
            self.streamline_cache = list(map(compute_streamline_2D, args_list))
            self.MappingFlowlineAsRenderingVAO(self.streamline_cache, self.streamline_buf)

        self.streamline_dirty = False


    def update_pathline(self,   attribs2=None):
        # the vertex attribe is fixed as
        # layout(location = 0) in vec3 in_position(x,y,z);
        # layout(location = 1) in vec2 in_attribs(time,addtional_attibs);
        # this is exactly what the vertexarrayobject now support( )

        if not hasattr(self, 'pathline_dirty') or not self.pathline_dirty:
            return
        self.indicatorObject=self.parentScene.getObject("Indicator") if self.parentScene.hasObject("Indicator") else None
        if self.indicatorObject is None:
            return
        actFieldWidget = self.parentScene.getObject("ActiveField")
        time=actFieldWidget.time()
        vector_field:UnsteadyVectorField2D = actFieldWidget.getActiveField()  
        if vector_field is None:
            return

        method = self.getOptionValue("integrator")
        if method in ["dopri5","dopri8","bosh3","fehlberg2","adaptive_heun"]:
          #callNeuralODESolver
            self.runAction("pathline_integrate_torch")
        else:
            #call my classical ODE solver
            seeds=self.indicatorObject.getValue("SeedingGroup0") # [(pos3D, time), ...]
            number_of_pathlines = len(seeds)
            if number_of_pathlines == 0:
                self.pathline_buf.erase()
                self.pathline_dirty = False
                return

            min_time = vector_field.getMinTime()
            max_time = vector_field.getMaxTime()
            #get integration and rendering paramter from gui
            step_size=self.getValue("stepSize")
            max_iteration = self.getValue("maxIteration")
            args_list = [
                    (vector_field, pos3d, t0, min_time, max_time, step_size, max_iteration, method)
                    for pos3d, t0 in seeds
                ]
    
            if vector_field.getDim()==2:
                self.pathline_cache = compute_pathline_2D_auto(args_list)
                self.MappingFlowlineAsRenderingVAO(self.pathline_cache, self.pathline_buf)
            elif vector_field.getDim()==3:
                # Fast path: numba parallel batch integration + numpy VBO assembly.
                seeds_xyzt = np.array(
                    [[float(pos3d[0]), float(pos3d[1]), float(pos3d[2]), float(t0)] for pos3d, t0 in seeds],
                    dtype=np.float64)
                batched = compute_pathlines_3D_batch(vector_field, seeds_xyzt, min_time, max_time, step_size, max_iteration, method)
                if batched is not None:
                    self.MappingBatchedFlowlineAsRenderingVAO(*batched, self.pathline_buf)
                else:  # unsupported integrator -> serial fallback
                    self.pathline_cache = list(map(compute_pathline_3D, args_list))
                    self.MappingFlowlineAsRenderingVAO(self.pathline_cache, self.pathline_buf)
            self.pathline_dirty = False

    def __numpy_vec3Arrays_to_pathlines2D(self, numpy_pathlines:np.ndarray):
        """
        numpy_pathline: np.ndarray, shape=(B,L, 3), float32
        """
        
        assert numpy_pathlines.ndim == 3 and numpy_pathlines.shape[2] == 3
        paths = []
        for i in range(numpy_pathlines.shape[0]):
            path = []
            for j in range(numpy_pathlines.shape[1]):
                pos = np.array(numpy_pathlines[i, j][:2])
                time = numpy_pathlines[i, j][2]
                path.append((pos, time))
            paths.append(path)
        return paths



    def RendExternalPathline(self, external_pathline, labels=None):
        """
        Instead of integrating the pathline, we just render the external pathline directly.
        external_pathline: List[np.ndarray, float]
        """
        self.pathline_cache=self.__numpy_vec3Arrays_to_pathlines2D(external_pathline)
        # convert labels from 1D array to list
        if labels is not None:
            labels = labels.tolist()
        self.MappingFlowlineAsRenderingVAO(self.pathline_cache, self.pathline_buf, labels)
        self.pathline_dirty = False
   


        

    def MappingFlowlineAsRenderingVAO(self, pathline_cache, buf, label_appending=None):
        """
        pathline_cache: List[],每个元素是一条 pathline, pathline 是 (pos3d(x,y,z), t) 的 list
        or
        pathline_cache: List[np.ndarray, float],每个元素是一条 pathline, pathline 是 (pos2d(x,y), t) 的 list
        label_appending: 可选,label of pathline transfer to attrib2(float) of vertex
        """
        # 获取时间范围
        if not pathline_cache or not pathline_cache[0]:
            buf.erase()
            return
        # 计算全局最小最大时间
        all_times = [t for path in pathline_cache for (_, t) in path]
        min_time = min(all_times)
        max_time = max(all_times)
        inverse_time_range = 1.0 / (max_time - min_time) if max_time > min_time else 1.0
        
        pathline_offset_indices = []
        pathline_sizes = []
        offset_counter = 0
        pathline_vertices = []
        PathlineIs3d=pathline_cache[0][0][0].shape[-1] == 3
        appendLabel=label_appending is not None and len(label_appending) == len(pathline_cache)
        if appendLabel:
            label_min_value=min(label_appending)
            label_max_value=max(label_appending)
            label_range=label_max_value-label_min_value
            label_inverse_range=1.0/label_range


        if PathlineIs3d:
            for lineID, path in enumerate(pathline_cache):
                # Skip empty pathlines
                if not path or len(path) < 2:
                    continue
                # Duplicate the first and last vertex for adjacency
                first_vertex = path[0]
                last_vertex = path[-1]
                # Build the expanded path: [first, ...all..., last]
                expanded_path = [first_vertex] + path + [last_vertex]
                # Record offset and size
                pathline_offset_indices.append(offset_counter)
                size = len(expanded_path)
                pathline_sizes.append(size)
                offset_counter += size
                # Add vertices

                for pos3d, t in expanded_path:
                    attrib0_time = (t - min_time) * inverse_time_range
                    pathline_vertices.append([
                        pos3d[0], pos3d[1], pos3d[2],
                        attrib0_time, 0.0 if not appendLabel else (label_appending[lineID]-label_min_value)*label_inverse_range
                    ])
        else:
            for lineID, path in enumerate(pathline_cache):
                # Skip empty pathlines
                if not path or len(path) < 2:
                    continue
                # Duplicate the first and last vertex for adjacency
                first_vertex = path[0]
                last_vertex = path[-1]
                # Build the expanded path: [first, ...all..., last]
                expanded_path = [first_vertex] + path + [last_vertex]
                # Record offset and size
                pathline_offset_indices.append(offset_counter)
                size = len(expanded_path)
                pathline_sizes.append(size)
                offset_counter += size
                # Add vertices

                for pos3d, t in expanded_path:
                    attrib0_time = (t - min_time) * inverse_time_range
                    pathline_vertices.append([
                        pos3d[0], pos3d[1],0.0,
                        attrib0_time, 0.0 if not appendLabel else (label_appending[lineID]-label_min_value)*label_inverse_range
                    ])
        
        if not pathline_vertices:
            buf.erase()
            return
        buf.upload_vertices(np.array(pathline_vertices, dtype=np.float32))
        buf.set_indices(pathline_offset_indices, pathline_sizes)

    def MappingBatchedFlowlineAsRenderingVAO(self, out_pos: np.ndarray, out_len: np.ndarray, buf):
        """Build the render VBO directly from the padded batch-integrator output.

        out_pos: (N, maxLen, 4) = (x, y, z, t); out_len: (N,) valid point count per line.
        Same vertex layout / adjacency as MappingFlowlineAsRenderingVAO ([first]+pts+[last],
        per-vertex (x, y, z, normalized_time, attrib2=0)), but assembled with numpy — the loop
        is O(#lines), not O(#vertices), so long dense 3D lines no longer stall on Python appends.
        """
        N = int(out_pos.shape[0])
        valid = []
        tmin_all = np.inf
        tmax_all = -np.inf
        for i in range(N):
            L = int(out_len[i])
            if L >= 2:
                tt = out_pos[i, :L, 3]
                tmn = float(tt.min())
                tmx = float(tt.max())
                if tmn < tmin_all:
                    tmin_all = tmn
                if tmx > tmax_all:
                    tmax_all = tmx
                valid.append((i, L))
        if not valid:
            buf.erase()
            return
        inv = 1.0 / (tmax_all - tmin_all) if tmax_all > tmin_all else 1.0

        verts_list = []
        offsets = []
        sizes = []
        off = 0
        for i, L in valid:
            seg = out_pos[i, :L, :]
            exp = np.empty((L + 2, 5), dtype=np.float32)
            exp[1:L + 1, 0:3] = seg[:, 0:3]
            exp[1:L + 1, 3] = (seg[:, 3] - tmin_all) * inv
            exp[1:L + 1, 4] = 0.0
            exp[0] = exp[1]          # duplicate first vertex for LINE_STRIP_ADJACENCY
            exp[L + 1] = exp[L]      # duplicate last vertex
            verts_list.append(exp)
            offsets.append(off)
            sizes.append(L + 2)
            off += L + 2

        buf.upload_vertices(np.concatenate(verts_list, axis=0))
        buf.set_indices(offsets, sizes)
        
        



    
        
