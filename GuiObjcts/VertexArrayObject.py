
from OpenGL import GL as gl
from .Object import *
import numpy as np    
from GuiObjcts.shaderManager import getShaderManager,ShaderProgram,Material
from functools import wraps


class VertexArrayObject(Object):
    def __init__(self,name):
        super().__init__(name)
        # Create a new VAO and bind it
        self.vao_id = gl.glGenVertexArrays(1)
        self.vertex_geometry:list=[] 
        self.vertex_tex_coords:list=[]
        self.indices=[]
        self.vetex_count=0

        # Create placeholders for VBO and EBO
        self.vbo_ids = gl.glGenBuffers(2)#vertex buffer and texture buffer
        self.ebo_id = gl.glGenBuffers(1)
        self.init()
        self.material:Material=None
        self.create_variable("modelMat",np.eye(4,dtype=np.float32),False)
        
        # Performance optimization flags
        self._buffer_dirty = True
        self._last_vertex_count = 0
        self._last_index_count = 0
  
    def setMaterial(self,material) -> None:
        self.material=material

    def init(self) -> None:
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

    def erase(self) -> None:
        self.vertex_geometry  = []
        self.vertex_tex_coords  = []
        self.indices = []
        self.vetex_count=0
        self._buffer_dirty = True
        self.commit()
        
    def appendVertexGeometryNoCommit(self, vertex_data, index_data,texture_date) -> None:
        # Append new vertex and index data
        self.vertex_geometry.extend(vertex_data)
        index_data_flattern=[index+self.vetex_count for index in index_data]
        self.indices.extend(index_data_flattern)
        self.vertex_tex_coords.extend(texture_date)
        self.vetex_count+=len(vertex_data)//3
        self._buffer_dirty = True
        
    def appendVertexGeometryBatch(self, vertex_data_list, index_data_list, texture_data_list) -> None:
        """Batch append multiple geometry chunks for better performance"""
        total_vertices = 0
        total_indices = 0
        total_tex_coords = 0
        
        # Pre-calculate total sizes
        for vertices, indices, tex_coords in zip(vertex_data_list, index_data_list, texture_data_list):
            total_vertices += len(vertices)
            total_indices += len(indices)
            total_tex_coords += len(tex_coords)
        
        # Pre-allocate lists with known size
        self.vertex_geometry = [None] * total_vertices
        self.indices = [None] * total_indices
        self.vertex_tex_coords = [None] * total_tex_coords
        
        # Fill in the data
        v_offset = 0
        i_offset = 0
        t_offset = 0
        
        for vertices, indices, tex_coords in zip(vertex_data_list, index_data_list, texture_data_list):
            # Copy vertex data
            v_end = v_offset + len(vertices)
            self.vertex_geometry[v_offset:v_end] = vertices
            v_offset = v_end
            
            # Copy index data with offset
            i_end = i_offset + len(indices)
            for i in range(len(indices)):
                self.indices[i_offset + i] = indices[i] + self.vetex_count
            i_offset = i_end
            
            # Copy texture data
            t_end = t_offset + len(tex_coords)
            self.vertex_tex_coords[t_offset:t_end] = tex_coords
            t_offset = t_end
            
            self.vetex_count += len(vertices) // 3
        
        self._buffer_dirty = True

    def appendTriangleWithoutCommit(self,pos0, pos1, pos2):
        # Assuming pos0, pos1, pos2 are numpy arrays or can be converted into numpy arrays
        geometryVerts =[pos0[0], pos0[1], pos0[2], pos1[0], pos1[1], pos1[2], pos2[0], pos2[1], pos2[2]]

   
        # Assuming a simple triangle, elements (indices) would be straightforward
        elements = [0, 1, 2]
        
        # Define texture coordinates for the triangle
        texCoords = [0.0, 0.0,1.0, 0.0,1.0, 1.0]
        
        # Here you should append these to your geometry system
        # For example purposes, this might look like:
        self.appendVertexGeometryNoCommit(geometryVerts, elements, texCoords)



    def commit(self) -> None:
  
        # self.flattern()
        vertex_geometry = np.array(self.vertex_geometry, dtype=np.float32)
        vertex_tex_coords = np.array(self.vertex_tex_coords, dtype=np.float32)
        indices = np.array(self.indices, dtype=np.uint32)

        # Check if buffers need updating
        current_vertex_count = len(vertex_geometry)
        current_index_count = len(indices)
        
        if (self._buffer_dirty or 
            current_vertex_count != self._last_vertex_count or 
            current_index_count != self._last_index_count):
            
            # Update vertex buffer
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_ids[0])
            gl.glBufferData(gl.GL_ARRAY_BUFFER, vertex_geometry.nbytes, vertex_geometry, gl.GL_STATIC_DRAW)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

            # Update texture coordinate buffer
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_ids[1])
            gl.glBufferData(gl.GL_ARRAY_BUFFER, vertex_tex_coords.nbytes, vertex_tex_coords, gl.GL_STATIC_DRAW)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

            # Update index buffer
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo_id)
            gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_STATIC_DRAW)
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
            
            # Update tracking variables
            self._last_vertex_count = current_vertex_count
            self._last_index_count = current_index_count
            self._buffer_dirty = False



    def render(self):
        # Bind VAO
        if self.material is not None:
            self.material.apply([self.parentScene, self.cameraObject,self])

        gl.glBindVertexArray(self.vao_id)  
        # Draw elements
        gl.glDrawElements(gl.GL_TRIANGLES, len(self.indices), gl.GL_UNSIGNED_INT, None)

        # Unbind VAO
        gl.glBindVertexArray(0)
        gl.glUseProgram(0)
        
    def _generate_circle_geometry(self, centerPos, normal, radius, segments):
        """Generate circle geometry similar to appendCircleWithoutCommit but return data instead of appending"""
        # Calculate orthogonal vectors to the normal for circle's plane
        orth0 = np.array([normal[1], -normal[0], normal[2]])
        if orth0[0]*orth0[0] + orth0[1]*orth0[1] < 0.001:
            orth0 = np.array([normal[2], normal[1], normal[0]])
        orth0 = orth0 / np.linalg.norm(orth0)
        orth1 = np.cross(normal, orth0)
        orth1 = orth1 / np.linalg.norm(orth1)
        orth2 = np.cross(normal, orth1)
        orth2 = orth2 / np.linalg.norm(orth2)

        # Initialize lists for vertices, textures, and indices
        firstPos = centerPos + orth2 * radius
        geometryVerts = [centerPos[0], centerPos[1], centerPos[2], firstPos[0], firstPos[1], firstPos[2]]      
        textureCoords = [0.0, 0.0, 0.0, 1.0]
        elements = []
        
        # Generate vertices around the circle
        for i in range(1, segments):
            angle = i * 2 * np.pi / segments
            s = np.sin(angle)
            c = np.cos(angle)

            v = s * orth1 + c * orth2
            vCirc = centerPos + v * radius
            geometryVerts.extend([vCirc[0], vCirc[1], vCirc[2]])
    
            textureCoords.extend([(s + 1.0) * 0.5, (c + 1.0) * 0.5])
            elements.extend([0, i, i + 1])
            
        # Connect the last segment to the first
        elements.extend([0, segments, 1])
        
        return geometryVerts, elements, textureCoords
    
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
        velocity_mag=np.linalg.norm(direction)
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
        vTop = startPos + direction * height*velocity_mag
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
            temporayVertex[(i-1)*9:i*9-1] =[lastVBottom[0], lastVBottom[1], lastVBottom[2], vBottom[0], vBottom[1], vBottom[2], vTop[0], vTop[1], vTop[2]]

        
        v=orth2
        vBottom = startPos + v * radius
        lastVBottom = startPos + lastV * radius
        temporayVertex=temporayVertex[0:9*segments]
        temporayVertex[-9:] =[lastVBottom[0], lastVBottom[1], lastVBottom[2], vBottom[0], vBottom[1], vBottom[2], vTop[0], vTop[1], vTop[2]]
        temporayIndex=list(range(0,3*segments))#3*segments


        self.appendVertexGeometryNoCommit(temporayVertex, temporayIndex, textureCoords)
        # self.appendTriangleWithoutCommit(lastVBottom, vBottom, vTop)
         #    put caps to the cylinder
        self.appendCircleWithoutCommit(startPos, direction, radius, segments)




    def appendCylinderWithoutCommit(self, centerPos, direction, radius, height, segments):
        # Normalize the direction vector
        direction = direction / np.linalg.norm(direction)

        # Calculate orthogonal vectors to create the circular base
        orthoVector1 = np.array([-direction[1], direction[0], 0])
        if np.dot(orthoVector1, orthoVector1) < 1e-6:  # Handle the case where direction is parallel to z-axis
            orthoVector1 = np.array([0, -direction[2], direction[1]])
        orthoVector1 /= np.linalg.norm(orthoVector1)
        orthoVector2 = np.cross(direction, orthoVector1)
        orthoVector2 /= np.linalg.norm(orthoVector2)

        # Calculate vertices for the top and bottom circles
        circleVertices = []
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            dx = np.cos(angle)
            dy = np.sin(angle)
            vertexBase = centerPos - 0.5 * height * direction + radius * (dx * orthoVector1 + dy * orthoVector2)
            vertexTop = centerPos + 0.5 * height * direction + radius * (dx * orthoVector1 + dy * orthoVector2)
            circleVertices.extend(vertexBase.tolist())
            circleVertices.extend(vertexTop.tolist())

        # Calculate indices for the side surfaces
        indices = []
        for i in range(segments):
            baseIndex = 2 * i
            topIndex = baseIndex + 1
            nextBaseIndex = (2 * ((i + 1) % segments))
            nextTopIndex = nextBaseIndex + 1
            
            # Each quad on the side of the cylinder is made of two triangles
            indices.extend([baseIndex, nextBaseIndex, nextTopIndex, nextTopIndex, topIndex, baseIndex])

        # Texture coordinates (Optional)
        # For simplicity, we can map the texture linearly around the circumference and along the height
        texCoords = []
        for i in range(segments):
            texCoords.extend([i / segments, 0])  # Bottom vertex
            texCoords.extend([i / segments, 1])  # Top vertex

        # Append to the class attributes
        self.appendVertexGeometryNoCommit(circleVertices, indices, texCoords)

    def appendArrowWithoutCommit(self, shaftBasePos:np.ndarray[np.float32,3], direction:np.ndarray[np.float32,3], shaftRadius:float, shaftHeight:float, coneHeight:float, coneRadius:float, segments:int):
        direction = direction / np.linalg.norm(direction)
        cylinderCenterPos=shaftBasePos+direction * shaftHeight * 0.5
        # Append the cylinder (shaft of the arrow)
        self.appendCylinderWithoutCommit(cylinderCenterPos, direction, shaftRadius, shaftHeight, segments)
        
        # Calculate the position for the base of the cone (tip of the arrow)
        coneBasePos = shaftBasePos + direction * (shaftHeight)
        coneCenterPos=coneBasePos+direction * coneHeight * 0.5
        # Append the cone (tip of the arrow)
        self.appendConeWithoutCommit(coneCenterPos, direction, coneRadius, coneHeight, segments)


    
class CoordinateSystem(Object):
                       
    def __init__(self):
        sceneArg:Scene=getScene()
        assert isinstance(sceneArg, Scene._original_class)
        super().__init__("CoordinateSystem")
        self.setGuiVisibility(False)
        Xaxis=VertexArrayObject("Xaxis")
        Yaxis=VertexArrayObject("Yaxis")
        Zaxis=VertexArrayObject("Zaxis")
        Xaxis.appendArrowWithoutCommit(np.array([0,0,0],dtype=np.float32),np.array([1,0,0],dtype=np.float32),0.025,2.0, 0.2, 0.05, 16)
        Yaxis.appendArrowWithoutCommit(np.array([0,0,0],dtype=np.float32),np.array([0,1,0],dtype=np.float32),0.025,2.0, 0.2, 0.05, 16)
        Zaxis.appendArrowWithoutCommit(np.array([0,0,0],dtype=np.float32),np.array([0,0,1],dtype=np.float32),0.025,2.0, 0.2, 0.05, 16)
        self.Vaos=[Xaxis,Yaxis,Zaxis] 
        defaultMat =getShaderManager().getDefautlMaterial()
        self.parentScene=sceneArg
        self.color=[ [1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]
        for  i,axis in enumerate(self.Vaos):
            axis.create_variable("color",self.color[i])
            axis.parentScene=self.parentScene
            axis.setMaterial(defaultMat)
            axis.commit()
       
    def postInit(self):
        super().postInit()
        for axis in self.Vaos:
            axis.cameraObject=self.cameraObject

    def drawGui(self):
        pass
    def render(self):
       for axis in self.Vaos:
            axis.render()
   
        
    
        






def createPlane(gridSize, domainBoundaryMin,domainBoundaryMax) -> [list, list, list]:
    """
    Generate a grid of vertices, indices, and texture coordinates for a plane.

    Args:
        gridSize (list): A list of integers [Xdim, Ydim] representing the number of grid cells in the x and y directions.
        domainBoundaryMin (list): A list of floats [Xmin, Ymin] representing the domain size.
        domainBoundaryMax (list): A list of floats [Xmax, Ymax] representing the domain size.

    Returns:
        list: A list of vertex data in the format [x, y, z, u, v] where (x, y, z) are the vertex coordinates and (u, v) are the texture coordinates.
        list: A list of indices for the triangle vertices forming the grid plane.
        list: A list of texture coordinates in the format [u, v] for each vertex.
    """
    Xdim, Ydim = gridSize
    Xmin,  Ymin = domainBoundaryMin
    Xmax, Ymax =domainBoundaryMax

    vertices = []
    indices = []
    textures = []

    dx = float(Xmax - Xmin) / float(Xdim-1)  # Horizontal spacing between vertices
    dy = float(Ymax - Ymin) / float(Ydim-1)  # Vertical spacing between vertices

    # Generate vertices and texture coordinates
    for y in range(Ydim ):
        for x in range(Xdim ):
            vx = Xmin + x * dx  # Vertex coordinate
            vy = Ymin + y * dy
            tx = float(x) / float(Xdim - 1.0) 
            ty = float(y) / float(Ydim - 1.0) 
            vertices.extend([vx, vy, 0])  # Assuming z = 0 for a flat plane
            textures.extend([tx, ty])

    # Generate indices
    for y in range(Ydim-1):
        for x in range(Xdim-1):
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
