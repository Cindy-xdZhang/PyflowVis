
from OpenGL import GL as gl
from GuiObjcts.Object import Object
import numpy as np    
from shaderManager import getShaderManager,ShaderProgram
from functools import wraps

class VertexArrayObject(Object):
    def __init__(self,name):
        super().__init__(name)
        # Create a new VAO and bind it
        self.vao_id = gl.glGenVertexArrays(1)

        # Initialize lists to store vertex and element data
        self.vertex_geometry  = []
        self.vetex_count=0
        self.indices = []
        self.vertex_tex_coords  = []

        # Create placeholders for VBO and EBO
        self.vbo_ids = gl.glGenBuffers(2)#vertex buffer and texture buffer
        self.ebo_id = gl.glGenBuffers(1)
        self.init()
        self.material=None
        self.shader_program=None
        self.create_variable("modelMat",np.eye(4,dtype=np.float32),False)
        
    def setMaterial(self,material):
        self.material=material
        sm=getShaderManager()
        if  self.material.shader_name in sm.shaders:
            self.shader_program = sm.get_program(self.material.shader_name)

    def init(self):
        gl.glBindVertexArray(self.vao_id)
        # Bind element buffer object
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo_id)

        # Bind VBO for geometry (attribute 0)
        gl.glEnableVertexAttribArray(0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_ids[0])
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        # Bind VBO for texture coordinates (attribute 1)
        gl.glEnableVertexAttribArray(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_ids[1])
        gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, None)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)

    def appendVertexGeometry(self, vertex_data, index_data,texture_date):
        # Append new vertex and index data
        self.appendVertexGeometryNoCommit(vertex_data, index_data,texture_date)
        self.commit()

    def erase(self):
        self.vertex_geometry  = []
        self.vetex_count=0
        self.indices = []
        self.vertex_tex_coords  = []
        self.commit()
    def appendVertexGeometryNoCommit(self, vertex_data, index_data,texture_date):
        # Append new vertex and index data
        self.vertex_geometry.extend(vertex_data)
        self.indices.extend(index_data)
        self.vertex_tex_coords.extend(texture_date)
        self.vetex_count+=len(vertex_data)//3

    def appendTriangleWithoutCommit(self,pos0, pos1, pos2):
        # Assuming pos0, pos1, pos2 are numpy arrays or can be converted into numpy arrays
        geometryVerts =[pos0[0], pos0[1], pos0[2], pos1[0], pos1[1], pos1[2], pos2[0], pos2[1], pos2[2]]

        previoussize=self.vetex_count
        # Assuming a simple triangle, elements (indices) would be straightforward
        elements = [previoussize, previoussize+1,previoussize+ 2]
        
        # Define texture coordinates for the triangle
        texCoords = [0.0, 0.0,1.0, 0.0,1.0, 1.0]
        
        # Here you should append these to your geometry system
        # For example purposes, this might look like:
        self.appendVertexGeometryNoCommit(geometryVerts, elements, texCoords)

    def commit(self) -> None:
        #! TODO: add multiple vertex,tex,index list,and a flattern function to support add multiple objects in multi-threads
        vertex_geometry = np.array(self.vertex_geometry, dtype=np.float32)
        vertex_tex_coords = np.array(self.vertex_tex_coords, dtype=np.float32)
        indices = np.array(self.indices, dtype=np.uint32)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_ids[0])
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertex_geometry.nbytes, vertex_geometry, gl.GL_STATIC_DRAW)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_ids[1])
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertex_tex_coords.nbytes, vertex_tex_coords, gl.GL_STATIC_DRAW)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo_id)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_STATIC_DRAW)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)



    def draw(self):
        # Bind VAO
        if self.shader_program is not None:
            cameraObject=self.parentScene.getObject("Camera")
            self.shader_program.setUniformScope([cameraObject,self])
            self.shader_program.Use()
        gl.glBindVertexArray(self.vao_id)  
        # Draw elements
        gl.glDrawElements(gl.GL_TRIANGLES, len(self.indices), gl.GL_UNSIGNED_INT, None)

        # Unbind VAO
        gl.glBindVertexArray(0)
        gl.glUseProgram(0)

    def appendCircleWithoutCommit(self,centerPos:np.ndarray[np.float32,3], normal:np.ndarray[np.float32,3], radius:float, segments:int):
        # Calculate orthogonal vectors to the normal for circle's plane
        orth0 = np.array([normal[1],-normal[0],normal[2]])
        if orth0[0]*orth0[0]+orth0[1]*orth0[1]< 0.001:
            orth0 = np.array([normal[2],normal[1],normal[0]])
        orth0 = orth0 / np.linalg.norm(orth0)
        orth1 = np.cross(normal, orth0)
        orth1 = orth1 / np.linalg.norm(orth1)
        orth2 = np.cross(normal, orth1)
        orth2 = orth2 / np.linalg.norm(orth2)

        # Initialize lists for vertices, textures, and indices
        firstPos=centerPos+orth2*radius
        geometryVerts = [centerPos[0] ,centerPos[1] ,centerPos[2],firstPos[0],firstPos[1],firstPos[2]]      
        textureCoords = [0.0,0.0,0.0,1.0]
        elements=[]
        # Generate vertices around the circle
        for i in range(1, segments ):
            angle = i * 2 * np.pi / segments
            s=np.sin(angle)
            c=np.cos(angle)

            v = s * orth1 + c* orth2
            vCirc = centerPos + v * radius
            geometryVerts.extend([vCirc[0],vCirc[1],vCirc[2]])
    
            textureCoords.extend([(s+ 1.0) * 0.5, (c + 1.0) * 0.5])
            elements.extend([0, i, i + 1])
        # Connect the last segment to the first
        elements.extend([0,segments, 1])
        self.appendVertexGeometryNoCommit(geometryVerts, elements, textureCoords)

    def appendConeWithoutCommit(self,centerPos:np.ndarray[np.float32,3], direction:np.ndarray[np.float32,3], radius:float, height:float, segments:int):
        direction=direction / np.linalg.norm(direction)
        startPos = centerPos - direction * height * 0.5

        # Generating orthogonal vectors
        orth0 = np.array([direction[1],-direction[0],direction[2]])
        if orth0[0]*orth0[0]+orth0[1]*orth0[1]< 0.001:
            orth0 = np.array([direction[2],direction[1],direction[0]])
        orth0 = orth0 / np.linalg.norm(orth0)

        orth1 = np.cross(direction, orth0)
        orth1 = orth1 / np.linalg.norm(orth1)
        orth2 = np.cross(direction, orth1)
        orth2 = orth2 / np.linalg.norm(orth2)


        # Append cone vertices
        vTop = startPos + direction * height
        lastV = orth2
        textureCoords=[]
        temporayVertex=[None]*9*segments
        for i in range(1,segments):
            angle = i * 2 * np.pi / segments
            s=np.sin(angle)
            c=np.cos(angle)
            v = s * orth1 + c * orth2
            vBottom = startPos + v * radius
            lastVBottom = startPos + lastV * radius
          
            lastV = v
            # self.appendTriangleWithoutCommit(lastVBottom, vBottom, vTop) is replaced by the following code:

            temporayVertex[(i-1)*9:i*9-1] =[lastVBottom[0], lastVBottom[1], lastVBottom[2], vBottom[0], vBottom[1], vBottom[2], vTop[0], vTop[1], vTop[2]]
            # previoussize=indexCount
            # temporayIndex[(i-1)*3:i*3-1]=[previoussize, previoussize+1,previoussize+ 2]
            # indexCount+=3
            # texCoords = [0.0, 0.0,1.0, 0.0,1.0, 1.0]
            # textureCoords.extend([0.0, 0.0,1.0, 0.0, 1.0, 1.0])

        
        v=orth2
        vBottom = startPos + v * radius
        lastVBottom = startPos + lastV * radius
        temporayVertex=temporayVertex[0:9*segments]
        temporayVertex[-9:] =[lastVBottom[0], lastVBottom[1], lastVBottom[2], vBottom[0], vBottom[1], vBottom[2], vTop[0], vTop[1], vTop[2]]
        temporayIndex=list(range(self.vetex_count,self.vetex_count+ 3*segments))#3*segments


        self.appendVertexGeometryNoCommit(temporayVertex, temporayIndex, textureCoords)
        # self.appendTriangleWithoutCommit(lastVBottom, vBottom, vTop)
         #    put caps to the cylinder
        self.appendCircleWithoutCommit(startPos, direction, radius, segments)




def call_on_dirty(func):
    """
    Decorator to skip calling the function if the parameters have not changed.
    """
    func._call_signature = None

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Create a tuple representing the current call signature
        current_call_signature = (args, tuple(kwargs.items()))

        # Check if the call signature has changed since the last call
        if current_call_signature != func._call_signature:
            # Update the call signature
            func._call_signature = current_call_signature
            # Call the original function
            return func(*args, **kwargs)
        else:
            # Skip calling the function
            pass

    return wrapper


class VertexArrayVectorGlyph(VertexArrayObject):
    def __init__(self, name):
        super().__init__(name)
        def dirtyCallBack(obj) -> None:
            obj.dirty=True
        self.create_variable_callback("scale",1.0,dirtyCallBack,False,1.0)
        self.create_variable_callback("segments",10,dirtyCallBack,False,10)
        self.create_variable_callback("radius",0.01,dirtyCallBack,False,0.01)
        self.create_variable_callback("height",0.1,dirtyCallBack,False,0.1)
        self.dirty = True



    def updateVectorGlyph(self,vector_field, time: float=0.0, position=(0, 0, 0), scale=1.0):
        """
        Draw vector glyphs representing a vector field interpolated between two time steps.

        :param vector_field: A VectorField2D object representing the vector field.
        :param time: The specific time to interpolate the vector field at.
        :param position: The position where to start drawing the vector field.
        :param scale: Scale factor for drawing glyphs.
        """
        if vector_field is None or self.dirty==False:
            return
    
        # Calculate the interpolation index
        time_idx = (time - vector_field.domainMinBoundary[2]) / vector_field.timeInterval
        lower_idx = int(np.floor(time_idx))
        upper_idx = int(np.ceil(time_idx))
        alpha = time_idx - lower_idx

        # Ensure indices are within the bounds of the vector field time steps
        lower_idx = max(0, min(vector_field.time_steps - 1, lower_idx))
        upper_idx = max(0, min(vector_field.time_steps - 1, upper_idx))

        # Get the two time slices of the vector field
        lower_field = vector_field.field[lower_idx].detach().numpy()
        upper_field = vector_field.field[upper_idx].detach().numpy()

        # Interpolate between the two time slices
        interpolated_field = (1 - alpha) * lower_field + alpha * upper_field
        radius=self.getValue("radius")
        hight=self.getValue("height")
        segments=self.getValue("segments")
        self.erase()
        for y in range(interpolated_field.shape[0]):
            for x in range(interpolated_field.shape[1]):
                vx, vy = interpolated_field[y, x,:]  # Extract the vector components
                posX,posY=x * vector_field.gridInterval[0]+vector_field.domainMinBoundary[0], y * vector_field.gridInterval[1]+vector_field.domainMinBoundary[1]
                direction = np.array([vx, vy, 0.0],dtype=np.float32)  # Create a vector from the components
                # Draw the vector glyph            
                self.appendConeWithoutCommit(np.array([posX,posY,0.0],dtype=np.float32),direction, radius, hight, segments)
        self.dirty=False
          
        

    






def createPlane(gridSize, domainSize) -> [list, list, list]:
    """
    Generate a grid of vertices, indices, and texture coordinates for a plane.

    Args:
        gridSize (list): A list of integers [Xdim, Ydim] representing the number of grid cells in the x and y directions.
        domainSize (list): A list of floats [Xmin, Ymin, Xmax, Ymax] representing the domain size.

    Returns:
        list: A list of vertex data in the format [x, y, z, u, v] where (x, y, z) are the vertex coordinates and (u, v) are the texture coordinates.
        list: A list of indices for the triangle vertices forming the grid plane.
        list: A list of texture coordinates in the format [u, v] for each vertex.
    """
    Xdim, Ydim = gridSize
    Xmin, Ymin, Xmax, Ymax = domainSize

    vertices = []
    indices = []
    textures = []

    dx = float(Xmax - Xmin) / float(Xdim-1)  # Horizontal spacing between vertices
    dy = float(Ymax - Ymin) / float(Xdim-1)  # Vertical spacing between vertices

    # Generate vertices and texture coordinates
    for y in range(Ydim ):
        for x in range(Xdim ):
            vx = Xmin + x * dx  # Vertex coordinate
            vy = Ymin + y * dy
            tx = x / Xdim  # Texture coordinate
            ty = y / Ydim
            vertices.extend([vx, vy, 0])  # Assuming z = 0 for a flat plane
            textures.extend([tx, ty])

    # Generate indices
    for y in range(Ydim):
        for x in range(Xdim):
            indexLL=y * (Xdim ) + x
            indexUL=(y+1) * (Xdim ) + x
            indexUR=(y+1)* (Xdim ) + x+1
            indexLR=y * (Xdim ) + x+1

    
            indices.extend([
                indexLL, indexUL, indexUR,
               indexUR , indexLR,  indexLL
            ])

    return vertices, indices, textures


def create_cube():
    # Cube vertex positions
    # 8 vertices, each vertex with 3 coordinates (x, y, z)
    vertices = [
        # Front face
        -1.0, -1.0,  1.0,
         1.0, -1.0,  1.0,
         1.0,  1.0,  1.0,
        -1.0,  1.0,  1.0,
        # Back face
        -1.0, -1.0, -1.0,
        -1.0,  1.0, -1.0,
         1.0,  1.0, -1.0,
         1.0, -1.0, -1.0,
    ]
  

    # Cube texture coordinates
    # 4 vertices per face, each texture coordinate with 2 values (u, v)
    tex_coords = [
        # Front face
        0.0, 0.0,
        1.0, 0.0,
        1.0, 1.0,
        0.0, 1.0,
        # Back face (Same as above, adjust according to actual texture)
        0.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
        1.0, 0.0,
    ]

    # Cube indices (two triangles form one face)
    # 6 faces, each face with 2 triangles, each triangle with 3 indices
    indices = [
        # Front face
        0, 1, 2, 2, 3, 0,
        # Right face
        1, 7, 6, 6, 2, 1,
        # Back face
        7, 4, 5, 5, 6, 7,
        # Left face
        4, 0, 3, 3, 5, 4,
        # Bottom face
        4, 7, 1, 1, 0, 4,
        # Top face
        3, 2, 6, 6, 5, 3,
    ]

    return vertices, tex_coords, indices
