
from OpenGL import GL as gl
from GuiObjcts.Object import Object
import numpy as np    
from shaderManager import getShaderManager,ShaderProgram


class VertexArrayObject(Object):
    def __init__(self,name):
        super().__init__(name)
        # Create a new VAO and bind it
        self.vao_id = gl.glGenVertexArrays(1)

        # Initialize lists to store vertex and element data
        self.vertex_geometry  = []
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

    def appendVertexGeometryNoCommit(self, vertex_data, index_data,texture_date):
        # Append new vertex and index data
        self.vertex_geometry.extend(vertex_data)
        self.indices.extend(index_data)
        self.vertex_tex_coords.extend(texture_date)

    def commit(self) -> None:
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
    # Scale down the cube size
    for i in range(len(vertices)):
        vertices[i] *= 0.1

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
