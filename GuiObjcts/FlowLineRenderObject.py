from GuiObjcts.VertexArrayObject import *
from .VisualizationEngine import getEngine
from GuiObjcts.shaderManager import *
from FLowUtils.VectorField2d import *
from FLowUtils.VectorField3d import *
from OpenGL import GL as gl 
import ctypes
from OpenGL.arrays import GLintArray, GLsizeiArray

def pathline_integration_one_direction(
    vectorField: UnsteadyVectorField2D|UnsteadyVectorField3D,
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
        vectorField: UnsteadyVectorField2D or UnsteadyVectorField3Dinstance
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
        if (direction > 0 and t >= timeEnd) or (direction < 0 and t <= timeEnd):
            break
        if NumericalMethod == "RK4":
            v1 = vectorField.get_vector(pos_3d[0], pos_3d[1], t)
            v2 = vectorField.get_vector(pos_3d[0] + 0.5 * stepSize * v1[0], pos_3d[1] + 0.5 * stepSize * v1[1], t + 0.5 * stepSize)
            v3 = vectorField.get_vector(pos_3d[0] + 0.5 * stepSize * v2[0], pos_3d[1] + 0.5 * stepSize * v2[1], t + 0.5 * stepSize)
            v4 = vectorField.get_vector(pos_3d[0] + stepSize * v3[0], pos_3d[1] + stepSize * v3[1], t + stepSize)
            delta = (stepSize / 6.0) * (v1 + 2 * v2 + 2 * v3 + v4)
        elif NumericalMethod == "Euler":
            v = vectorField.get_vector(pos_3d[0], pos_3d[1], t)
            delta = stepSize * v
        else:
            raise ValueError(f"Unknown NumericalMethod: {NumericalMethod}")
        #make sure the delta_pos_3d is a 3D vector
        if isinstance(vectorField, UnsteadyVectorField3D):
            delta_pos_3d = np.array([delta[0], delta[1], delta[2]], dtype=np.float32)
        else   :
            delta_pos_3d = np.array([delta[0], delta[1], 0.0], dtype=np.float32)

        pos_3d = pos_3d + delta_pos_3d
        t = t + stepSize
        path.append((pos_3d.copy(), t))
    return path
    

def pathline_integration_double_direction(
    vectorField: UnsteadyVectorField2D,
    start_pos,
    timeStart,
    timeEnd,
    stepSize=0.01,
    maxIterations=5000,
    NumericalMethod="RK4"
):
    """
    Integrate a pathline in both forward and backward directions through an unsteady vector field.
    Args:
        vectorField: UnsteadyVectorField2D instance
        start_pos: [x, y] initial position
        timeStart: float, start time
        timeEnd: float, end time
        stepSize: float, integration time step
        maxIterations: int, maximum number of steps
        NumericalMethod: str, "RK4" or "Euler"
    Returns:
        List of (position, time) tuples, with backward points first, then forward points
    """
    # Integrate backward from timeStart to timeEnd
    backward_path = Pathline_integration_one_direction(
        vectorField, start_pos, timeEnd, timeStart, stepSize, maxIterations, NumericalMethod
    )
    
    # Integrate forward from timeStart to timeEnd
    forward_path = Pathline_integration_one_direction(
        vectorField, start_pos, timeStart, timeEnd, stepSize, maxIterations, NumericalMethod
    )
    
    # Combine paths: backward (reversed) + forward
    # Remove duplicate start point from forward path
    full_path = backward_path[::-1] + forward_path[1:]
    return full_path


def streamline_integration_one_direction(
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


def streamline_integration_double_direction(
    vectorField: UnsteadyVectorField2D,
    start_pos,
    time,
    stepSize=0.01,
    maxIterations=5000,
    NumericalMethod="RK4"
):
    """
    Integrate a streamline in both forward and backward directions through an unsteady vector field at a fixed time.
    Args:
        vectorField: UnsteadyVectorField2D instance
        start_pos: [x, y] initial position
        time: float, fixed time for streamline
        stepSize: float, integration step size
        maxIterations: int, maximum number of steps
        NumericalMethod: str, "RK4" or "Euler"
    Returns:
        List of (position, time) tuples, with backward points first, then forward points
    """
    # Integrate backward
    backward_path = streamline_integration_one_direction(
        vectorField, start_pos, time, stepSize, maxIterations, NumericalMethod, "backward"
    )
    
    # Integrate forward
    forward_path = streamline_integration_one_direction(
        vectorField, start_pos, time, stepSize, maxIterations, NumericalMethod, "forward"
    )
    
    # Combine paths: backward (reversed) + forward
    # Remove duplicate start point from forward path
    full_path = backward_path[::-1] + forward_path[1:]
    return full_path
    


class FlowLineObject(Object):
    def __init__(self):
        super().__init__("flowline")
        self.engine = getEngine()

   
        self.pathline_dirty = True
        self.streamline_dirty = True

        self.active_seeding_group = 0
        self.SeedingGroup0=[]# a list of (pos3D,time), for 2D, pos.Z is 0 used for pathlines
        self.SeedingGroup1=[]# a list of (pos3D,time), for 2D, pos.Z is 0 used for streamlines

 
        
        self.__initDynamicTypeGLContext__()
        #  material
        self.create_variable("modelMat",np.eye(4,dtype=np.float32),False)
        self.material = Material("flowlineMaterial",  "flowlineMat",texture0="builtIn")
        self.setMaterial(self.material)

        self.create_variable("streamline_active",False)
        self.create_variable("pathline_active",True)
        self.create_variable("lineWeight", 0.1,True)
        self.create_variable("zOffset", 0.0,True)
        self.create_variable("colorMap",self.engine.getBuiltInTextureNames())
        self.create_variable("maxIteration", 5000,True)
        self.create_variable("integrator", "RK4",True)#euler,rk4
        self.create_variable("stepSize", 0.01  ,True)
        # self.create_variable("flowlineGroupID", self.flowlineGroupID)

    def setMaterial(self,material) -> None:
        self.material=material

    def erase(self) -> None:
        self.vertex_geometry  = []
        self.vertex_tex_coords  = []
        self.indices = []
        self.vetex_count=0
        self._buffer_dirty = True
        self.commit()
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
        pass


    def update_pathline(self,   attribs2=None):
        # the vertex attribe is fixed as
        # layout(location = 0) in vec3 in_position(x,y,z);
        # layout(location = 1) in vec2 in_attribs(time,addtional_attibs);
        # this is exactly what the vertexarrayobject now support( )

        if not hasattr(self, 'pathline_dirty') or not self.pathline_dirty:
            return
        actFieldWidget = self.parentScene.getObject("ActiveField")
        time=actFieldWidget.time()
        vector_field:UnsteadyVectorField2D = actFieldWidget.getActiveField()  
        if vector_field is None:
            return

        
        self.SeedingGroup0=[(np.array([0.0,0.1,0.0]),time), (np.array([0.0,0.9,0.0]),time)]
        seeds = self.SeedingGroup0  # [(pos3D, time), ...]
        number_of_pathlines = len(seeds)

        start_time =time
        min_time = vector_field.getMinTime()
        max_time = vector_field.getMaxTime()
        #get integration and rendering paramter from gui
        colorMapID=self.getValue("colorMap")
        lineWeight=self.getValue("lineWeight")
        zOffset=self.getValue("zOffset")
        step_size=self.getValue("stepSize")
        max_iteration = self.getValue("maxIteration")
        method= self.getValue("integrator")

        self.pathline_cache = []
        for pos3d, t0 in seeds:
            forward = pathline_integration_one_direction(vector_field,
                pos3d, t0, max_time, step_size, max_iteration, method
            )
            backward =pathline_integration_one_direction( vector_field,
                pos3d, t0, min_time, step_size, max_iteration, method
            )
            backward = backward[::-1]
            full_path = backward + forward[1:]
            self.pathline_cache.append(full_path)

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
        
        if not pathline_vertices:
            return
        self.updateVertexAttributesBuffer(np.array(pathline_vertices, dtype=np.float32))
        self.updateMultiDrawIndices(pathline_offset_indices, pathline_sizes)
        
        











    
        

  