from GuiObjcts.VertexArrayObject import *
from .VisualizationEngine import getEngine
from GuiObjcts.shaderManager import *
from FLowUtils.VectorField2d import *
from FLowUtils.VectorField3d import *
from OpenGL import GL as gl 
import ctypes



def pathline_integration_one_direction_2D(
    vectorField: UnsteadyVectorField2D,
    start_pos:np.ndarray[np.float32,3],
    timeStart:float,
    timeEnd:float,
    stepSize:float=0.01,
    maxIterations:int=5000,
    NumericalMethod:str="RK4"
):
    """
    Integrate a pathline in one direction through an unsteady vector field.
    Args:
        vectorField: UnsteadyVectorField2D 
        start_pos: [x, y,z=0] initial position
        timeStart: float, start time
        timeEnd: float, end time
        stepSize: float, integration time step
        maxIterations: int, maximum number of steps
        NumericalMethod: str, "RK4" or "Euler"
    Returns:
        List of (position, time) tuples
    """
    pos_3d = np.array(start_pos, dtype=np.float32)
    t = timeStart
    path = [(pos_3d.copy(), t)]
    direction = 1 if timeEnd > timeStart else -1
    stepSize = abs(stepSize) * direction
    for i in range(maxIterations):
        if (direction > 0 and t >= timeEnd) or (direction < 0 and t <= timeEnd) or not vectorField.IsInside(pos_3d):
            break
        if NumericalMethod == "RK4":
            v1 = vectorField.get_vector(pos_3d[0], pos_3d[1], t)
            v2 = vectorField.get_vector(pos_3d[0] + 0.5 * stepSize * v1[0], pos_3d[1] + 0.5 * stepSize * v1[1], t + 0.5 * stepSize)
            v3 = vectorField.get_vector(pos_3d[0] + 0.5 * stepSize * v2[0], pos_3d[1] + 0.5 * stepSize * v2[1], t + 0.5 * stepSize)
            v4 = vectorField.get_vector(pos_3d[0] + stepSize * v3[0], pos_3d[1] + stepSize * v3[1], t + stepSize)
            delta = (stepSize / 6.0) * (v1 + 2 * v2 + 2 * v3 + v4)
        elif NumericalMethod == "RK5":
            k1 = vectorField.get_vector(pos_3d[0], pos_3d[1], t)
            p2 = pos_3d[:2] + stepSize * (1/4 * k1)
            k2 = vectorField.get_vector(p2[0], p2[1], t + 1/4 * stepSize)
            p3 = pos_3d[:2] + stepSize * (3/32 * k1 + 9/32 * k2)
            k3 = vectorField.get_vector(p3[0], p3[1], t + 3/8 * stepSize)
            p4 = pos_3d[:2] + stepSize * (1932/2197 * k1 - 7200/2197 * k2 + 7296/2197 * k3)
            k4 = vectorField.get_vector(p4[0], p4[1], t + 12/13 * stepSize)
            p5 = pos_3d[:2] + stepSize * (439/216 * k1 - 8 * k2 + 3680/513 * k3 - 845/4104 * k4)
            k5 = vectorField.get_vector(p5[0], p5[1], t + 1 * stepSize)
            p6 = pos_3d[:2] + stepSize * (-8/27 * k1 + 2 * k2 - 3544/2565 * k3 + 1859/4104 * k4 - 11/40 * k5)
            k6 = vectorField.get_vector(p6[0], p6[1], t + 1/2 * stepSize)
            delta = stepSize * (16/135 * k1 + 6656/12825 * k3 + 28561/56430 * k4 - 9/50 * k5 + 2/55 * k6)
        elif NumericalMethod == "Euler":
            v = vectorField.get_vector(pos_3d[0], pos_3d[1], t)
            delta = stepSize * v
        else:
            raise ValueError(f"Unknown NumericalMethod: {NumericalMethod}")
    
        delta_pos_3d = np.array([delta[0], delta[1], 0.0], dtype=np.float32)
        pos_3d = pos_3d + delta_pos_3d
        t = t + stepSize
        path.append((pos_3d.copy(), t))
    return path
    
def pathline_integration_one_direction_3D(
    vectorField: UnsteadyVectorField3D,
    start_pos:np.ndarray[np.float32,3],
    timeStart:float,
    timeEnd:float,
    stepSize:float=0.01,
    maxIterations:int=5000,
    NumericalMethod:str="RK4"):
    """
    Integrate a pathline in one direction through an unsteady vector field.
    Args:
        vectorField: UnsteadyVectorField3D 
        start_pos: [x, y,z=0] initial position
        timeStart: float, start time
        timeEnd: float, end time
        stepSize: float, integration time step
        maxIterations: int, maximum number of steps
        NumericalMethod: str, "RK4" or "Euler"
    Returns:
        List of (position, time) tuples
    """
    pos_3d = np.array(start_pos, dtype=np.float32)
    t = timeStart
    path = [(pos_3d.copy(), t)]
    direction = 1 if timeEnd > timeStart else -1
    stepSize = abs(stepSize) * direction
    for i in range(maxIterations):
        if (direction > 0 and t >= timeEnd) or (direction < 0 and t <= timeEnd) or not vectorField.IsInside(pos_3d):
            break
        if NumericalMethod == "RK4":
            v1 = vectorField.get_vector(pos_3d[0], pos_3d[1],pos_3d[2], t)
            v2 = vectorField.get_vector(pos_3d[0] + 0.5 * stepSize * v1[0], pos_3d[1] + 0.5 * stepSize * v1[1],pos_3d[2] + 0.5 * stepSize * v1[2], t + 0.5 * stepSize)
            v3 = vectorField.get_vector(pos_3d[0] + 0.5 * stepSize * v2[0], pos_3d[1] + 0.5 * stepSize * v2[1],pos_3d[2] + 0.5 * stepSize * v2[2], t + 0.5 * stepSize)
            v4 = vectorField.get_vector(pos_3d[0] + stepSize * v3[0], pos_3d[1] + stepSize * v3[1],pos_3d[2] + stepSize * v3[2], t + stepSize)
            delta = (stepSize / 6.0) * (v1 + 2 * v2 + 2 * v3 + v4)
        elif NumericalMethod == "RK5":
            k1 = vectorField.get_vector(pos_3d[0], pos_3d[1], pos_3d[2], t)
            p2 = pos_3d + stepSize * (1/4 * k1)
            k2 = vectorField.get_vector(p2[0], p2[1], p2[2], t + 1/4 * stepSize)
            p3 = pos_3d + stepSize * (3/32 * k1 + 9/32 * k2)
            k3 = vectorField.get_vector(p3[0], p3[1], p3[2], t + 3/8 * stepSize)
            p4 = pos_3d + stepSize * (1932/2197 * k1 - 7200/2197 * k2 + 7296/2197 * k3)
            k4 = vectorField.get_vector(p4[0], p4[1], p4[2], t + 12/13 * stepSize)
            p5 = pos_3d + stepSize * (439/216 * k1 - 8 * k2 + 3680/513 * k3 - 845/4104 * k4)
            k5 = vectorField.get_vector(p5[0], p5[1], p5[2], t + 1 * stepSize)
            p6 = pos_3d + stepSize * (-8/27 * k1 + 2 * k2 - 3544/2565 * k3 + 1859/4104 * k4 - 11/40 * k5)
            k6 = vectorField.get_vector(p6[0], p6[1], p6[2], t + 1/2 * stepSize)
            delta = stepSize * (16/135 * k1 + 6656/12825 * k3 + 28561/56430 * k4 - 9/50 * k5 + 2/55 * k6)
        elif NumericalMethod == "Euler":
            v = vectorField.get_vector(pos_3d[0], pos_3d[1],pos_3d[2], t)
            delta = stepSize * v
        else:
            raise ValueError(f"Unknown NumericalMethod: {NumericalMethod}")
    
        delta_pos_3d = np.array([delta[0], delta[1], delta[2]], dtype=np.float32)
        pos_3d = pos_3d + delta_pos_3d
        t = t + stepSize
        path.append((pos_3d.copy(), t))
    return path


def streamline_integration_one_direction_2D(
    vectorField: UnsteadyVectorField2D,
    start_pos,
    time,
    stepSize=0.01,
    maxIterations=5000,
    NumericalMethod="RK4",
    direction="forward"
):
    """
    Integrate a streamline in one direction through an unsteady vector field at a fixed time.
    Args:
        vectorField: UnsteadyVectorField2D instance
        start_pos: [x, y] initial position
        time: float, fixed time for streamline
        stepSize: float, integration step size
        maxIterations: int, maximum number of steps
        NumericalMethod: str, "RK4" or "Euler"
        direction: str, "forward" or "backward"
    Returns:
        List of (position, time) tuples
    """
    pos = np.array(start_pos, dtype=np.float32)
    path = [(pos.copy(), time)]
    step_direction = 1 if direction == "forward" else -1
    abs_step_size = abs(stepSize) * step_direction
    
    for i in range(maxIterations):
        if NumericalMethod == "RK4":
            v1 = vectorField.get_vector(pos[0], pos[1], time)
            v2 = vectorField.get_vector(pos[0] + 0.5 * abs_step_size * v1[0], pos[1] + 0.5 * abs_step_size * v1[1], time)
            v3 = vectorField.get_vector(pos[0] + 0.5 * abs_step_size * v2[0], pos[1] + 0.5 * abs_step_size * v2[1], time)
            v4 = vectorField.get_vector(pos[0] + abs_step_size * v3[0], pos[1] + abs_step_size * v3[1], time)
            delta = (abs_step_size / 6.0) * (v1 + 2 * v2 + 2 * v3 + v4)
        elif NumericalMethod == "RK5":
            t = time
            k1 = vectorField.get_vector(pos[0], pos[1], t)
            p2 = pos + abs_step_size * (1/4 * k1)
            k2 = vectorField.get_vector(p2[0], p2[1], t)
            p3 = pos + abs_step_size * (3/32 * k1 + 9/32 * k2)
            k3 = vectorField.get_vector(p3[0], p3[1], t)
            p4 = pos + abs_step_size * (1932/2197 * k1 - 7200/2197 * k2 + 7296/2197 * k3)
            k4 = vectorField.get_vector(p4[0], p4[1], t)
            p5 = pos + abs_step_size * (439/216 * k1 - 8 * k2 + 3680/513 * k3 - 845/4104 * k4)
            k5 = vectorField.get_vector(p5[0], p5[1], t)
            p6 = pos + abs_step_size * (-8/27 * k1 + 2 * k2 - 3544/2565 * k3 + 1859/4104 * k4 - 11/40 * k5)
            k6 = vectorField.get_vector(p6[0], p6[1], t)
            delta = abs_step_size * (16/135 * k1 + 6656/12825 * k3 + 28561/56430 * k4 - 9/50 * k5 + 2/55 * k6)
        elif NumericalMethod == "Euler":
            v = vectorField.get_vector(pos[0], pos[1], time)
            delta = abs_step_size * v
        else:
            raise ValueError(f"Unknown NumericalMethod: {NumericalMethod}")
        
        pos = pos + delta
        path.append((pos.copy(), time))
        
        # Check if we've gone out of bounds (optional safety check)
        if (pos[0] < vectorField.domainMinBoundary[0] or pos[0] > vectorField.domainMaxBoundary[0] or
            pos[1] < vectorField.domainMinBoundary[1] or pos[1] > vectorField.domainMaxBoundary[1]):
            break
    
    return path


def streamline_integration_one_direction_3D(
    vectorField: UnsteadyVectorField3D,
    start_pos,
    time,
    stepSize=0.01,
    maxIterations=5000,
    NumericalMethod="RK4",
    direction="forward"
):
    """
    Integrate a streamline in one direction through an unsteady 3D vector field at a fixed time.
    """
    pos_3d = np.array(start_pos, dtype=np.float32)
    path = [(pos_3d.copy(), time)]
    step_direction = 1 if direction == "forward" else -1
    abs_step_size = abs(stepSize) * step_direction

    for i in range(maxIterations):
        if not vectorField.IsInside(pos_3d):
            break

        if NumericalMethod == "RK4":
            v1 = vectorField.get_vector(pos_3d[0], pos_3d[1], pos_3d[2], time)
            v2_pos = pos_3d + 0.5 * abs_step_size * v1
            v2 = vectorField.get_vector(v2_pos[0], v2_pos[1], v2_pos[2], time)
            v3_pos = pos_3d + 0.5 * abs_step_size * v2
            v3 = vectorField.get_vector(v3_pos[0], v3_pos[1], v3_pos[2], time)
            v4_pos = pos_3d + abs_step_size * v3
            v4 = vectorField.get_vector(v4_pos[0], v4_pos[1], v4_pos[2], time)
            delta = (abs_step_size / 6.0) * (v1 + 2 * v2 + 2 * v3 + v4)
        elif NumericalMethod == "RK5":
            t = time
            k1 = vectorField.get_vector(pos_3d[0], pos_3d[1], pos_3d[2], t)
            p2 = pos_3d + abs_step_size * (1/4 * k1)
            k2 = vectorField.get_vector(p2[0], p2[1], p2[2], t)
            p3 = pos_3d + abs_step_size * (3/32 * k1 + 9/32 * k2)
            k3 = vectorField.get_vector(p3[0], p3[1], p3[2], t)
            p4 = pos_3d + abs_step_size * (1932/2197 * k1 - 7200/2197 * k2 + 7296/2197 * k3)
            k4 = vectorField.get_vector(p4[0], p4[1], p4[2], t)
            p5 = pos_3d + abs_step_size * (439/216 * k1 - 8 * k2 + 3680/513 * k3 - 845/4104 * k4)
            k5 = vectorField.get_vector(p5[0], p5[1], p5[2], t)
            p6 = pos_3d + abs_step_size * (-8/27 * k1 + 2 * k2 - 3544/2565 * k3 + 1859/4104 * k4 - 11/40 * k5)
            k6 = vectorField.get_vector(p6[0], p6[1], p6[2], t)
            delta = abs_step_size * (16/135 * k1 + 6656/12825 * k3 + 28561/56430 * k4 - 9/50 * k5 + 2/55 * k6)
        elif NumericalMethod == "Euler":
            v = vectorField.get_vector(pos_3d[0], pos_3d[1], pos_3d[2], time)
            delta = abs_step_size * v
        else:
            raise ValueError(f"Unknown NumericalMethod: {NumericalMethod}")

        pos_3d += delta
        path.append((pos_3d.copy(), time))

    return path


def compute_pathline_2D(args):
        vector_field, pos3d, t0, min_time, max_time, step_size, max_iteration, method = args
        forward = pathline_integration_one_direction_2D(vector_field, pos3d, t0, max_time, step_size, max_iteration, method)
        backward = pathline_integration_one_direction_2D(vector_field, pos3d, t0, min_time, step_size, max_iteration, method)
        backward = backward[::-1]
        full_path = backward + forward[1:]
        return full_path

def compute_pathline_3D(args):
    vector_field, pos3d, t0, min_time, max_time, step_size, max_iteration, method = args
    forward = pathline_integration_one_direction_3D(vector_field, pos3d, t0, max_time, step_size, max_iteration, method)
    backward = pathline_integration_one_direction_3D(vector_field, pos3d, t0, min_time, step_size, max_iteration, method)
    backward = backward[::-1]
    full_path = backward + forward[1:]
    return full_path

def compute_streamline_3D(args):
    vector_field, pos3d, time, step_size, max_iteration, method = args
    forward = streamline_integration_one_direction_3D(vector_field, pos3d, time, step_size, max_iteration, method, "forward")
    backward = streamline_integration_one_direction_3D(vector_field, pos3d, time, step_size, max_iteration, method, "backward")
    backward = backward[::-1]
    full_path = backward + forward[1:]
    return full_path









class FlowLineObject(Object):
    def __init__(self):
        super().__init__("flowline")
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

        

        def pathline_integrate_torch_action():
            from FLowUtils.neuralVectorFieldODE import integrate_pathline2D_torch,UnsteadyVectorField2D_Torch
            actFieldWidget = self.parentScene.getObject("ActiveField")
            time_current = actFieldWidget.time()
            vector_field:UnsteadyVectorField2D = actFieldWidget.getActiveField()
            if vector_field is None:
                return
            self.indicatorObject=self.parentScene.getObject("indicator") if self.parentScene.hasObject("indicator") else None
            if self.indicatorObject is None:
                return
            
            seeds_data = self.indicatorObject.getValue("SeedingGroup0")
            if not seeds_data:
                self.erase()
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
                
            self.MappingFlowlineAsRenderingVAO(pathline_cache)
            self.pathline_dirty = False
        self.addAction("pathline_integrate_torch", pathline_integrate_torch_action)

    def setMaterial(self,material) -> None:
        self.material=material

    def erase(self) -> None:
        self.vertex_count = 0
        self.mOffsetIndices = []
        self.mDrawSizes = []
        # Bind VAO and VBO
        gl.glBindVertexArray(self.vao_id)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_id)
        # Upload empty data to clear buffer
        gl.glBufferData(gl.GL_ARRAY_BUFFER, 0, None, gl.GL_DYNAMIC_DRAW)
        # Unbind
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)

        
    def __initDynamicTypeGLContext__(self):
        self.vertex_count = 0
        self.mOffsetIndices = []
        self.mDrawSizes = []

        self.vao_id = gl.glGenVertexArrays(1)
        self.vbo_id = gl.glGenBuffers(1)  # vertex buffer
        gl.glBindVertexArray(self.vao_id)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_id)#link vao to vbo
        # the vertex attribe is fixed as
        # layout(location = 0) in vec3 in_position(x,y,z);
        # layout(location = 1) in vec2 in_attribs(time,addtional_attibs);
        stride = 5 * 4  # 5 float32 = 20 bytes
        # layout(location = 0) in vec3 in_positions;
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(0))
        # layout(location = 1) in vec2 in_attribs;
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(3 * 4))
        
        
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)


 


    def render(self):
        #update if dirty
        if self.getValue("pathline_active"):
            self.update_pathline()
        if self.getValue("streamline_active"):
            self.update_streamline()
        
        if self.vertex_count == 0:
            return
        # Bind VAO
        if self.material is not None:
            self.material.apply([self.parentScene, self.cameraObject,self])

        gl.glBindVertexArray(self.vao_id)  
        gl.glMultiDrawArrays(gl.GL_LINE_STRIP_ADJACENCY, self.mOffsetIndices, self.mDrawSizes, len(self.mOffsetIndices))
 
        # Unbind VAO
        gl.glBindVertexArray(0)
        gl.glUseProgram(0)

    def update_streamline(self,  attribs2=None):
        if not hasattr(self, 'streamline_dirty') or not self.streamline_dirty:
            return

        self.indicatorObject=self.parentScene.getObject("indicator") if self.parentScene.hasObject("indicator") else None
        if self.indicatorObject is None:
            return

        actFieldWidget = self.parentScene.getObject("ActiveField")
        if actFieldWidget is None:
            return

        vector_field:UnsteadyVectorField3D = actFieldWidget.getActiveField()
        if vector_field is None or not isinstance(vector_field, UnsteadyVectorField3D):
            return

        time = actFieldWidget.time()
        seeds = self.indicatorObject.getValue("SeedingGroup1")
        
        step_size = self.getValue("stepSize")
        max_iteration = self.getValue("maxIteration")
        method = self.getValue("integrator")

        args_list = [
            (vector_field, pos3d, time, step_size, max_iteration, method)
            for pos3d, _ in seeds
        ]
        
        if vector_field.getDim() == 3:
            self.streamline_cache = list(map(compute_streamline_3D, args_list))
            self.MappingFlowlineAsRenderingVAO(self.streamline_cache)

        self.streamline_dirty = False


    def update_pathline(self,   attribs2=None):
        # the vertex attribe is fixed as
        # layout(location = 0) in vec3 in_position(x,y,z);
        # layout(location = 1) in vec2 in_attribs(time,addtional_attibs);
        # this is exactly what the vertexarrayobject now support( )

        if not hasattr(self, 'pathline_dirty') or not self.pathline_dirty:
            return
        self.indicatorObject=self.parentScene.getObject("indicator") if self.parentScene.hasObject("indicator") else None
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
                self.pathline_cache = list(map(compute_pathline_2D, args_list))
            elif vector_field.getDim()==3:
                self.pathline_cache = list(map(compute_pathline_3D, args_list))

            self.MappingFlowlineAsRenderingVAO(self.pathline_cache)
            self.pathline_dirty = False



    def updateMultiDrawIndices(self,line_offset_indices,line_sizes):
        #const std::vector<GLint>& offsetIndices, const std::vector<GLsizei>& sizes
        self.mOffsetIndices = np.array(line_offset_indices, dtype=np.uint32)
        self.mDrawSizes = np.array(line_sizes, dtype=np.uint32)


    def updateVertexAttributesBuffer(self, vertex_attributes):
        """
        vertex_attributes: np.ndarray, shape=(N, attr_dim), float32
        """
        self.vertex_count = len(vertex_attributes)
        gl.glBindVertexArray(self.vao_id)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_id)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertex_attributes.nbytes, vertex_attributes, gl.GL_STATIC_DRAW)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)

    def MappingFlowlineAsRenderingVAO(self, pathline_cache, scalar_field_appending=None):
        """
        pathline_cache: List[List[Tuple[np.ndarray, float]]]
            每个元素是一条 pathline，pathline 是 (pos3d, t) 的 list
        scalar_field_appending: 可选，支持 None 或有 get_value(pos, t) 方法的对象
        """
        # 获取时间范围
        if not pathline_cache or not pathline_cache[0]:
            return
        # 计算全局最小最大时间
        all_times = [t for path in pathline_cache for (_, t) in path]
        min_time = min(all_times)
        max_time = max(all_times)
        inverse_time_range = 1.0 / (max_time - min_time) if max_time > min_time else 1.0
        
        ADJACENCY_SIZE = 2
        pathline_offset_indices = []
        pathline_sizes = []
        offset_counter = 0
        pathline_vertices = []
        posIs3d=False
        if pathline_cache[0][0][0].shape[-1] == 3:
            for path in pathline_cache:
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
                        attrib0_time, 0.0
                    ])
        else:
            for path in pathline_cache:
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
                        attrib0_time, 0.0
                    ])
        
        if not pathline_vertices:
            return
        self.updateVertexAttributesBuffer(np.array(pathline_vertices, dtype=np.float32))
        self.updateMultiDrawIndices(pathline_offset_indices, pathline_sizes)
        
        



    
        
