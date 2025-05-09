
from OpenGL import GL as gl
from GuiObjcts.Object import Object,Scene
import numpy as np    
from shaderManager import getShaderManager,ShaderProgram
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
        self.material=None
        self.create_variable("modelMat",np.eye(4,dtype=np.float32),False)
  
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
        self.commit()
        
    def appendVertexGeometryNoCommit(self, vertex_data, index_data,texture_date) -> None:
        # Append new vertex and index data
        self.vertex_geometry.extend(vertex_data)
        index_data_flattern=[index+self.vetex_count for index in index_data]
        self.indices.extend(index_data_flattern)
        self.vertex_tex_coords.extend(texture_date)
        self.vetex_count+=len(vertex_data)//3

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



    # def flattern(self)->None:
    #     self.vertex_geometry = []
    #     self.vertex_tex_coords = []
    #     self.indices = []
    #     indiceOffset=0
    #     for submesh in self.submeshes:
    #         self.vertex_geometry.extend(submesh.vertex_geometry)
    #         self.vertex_tex_coords.extend(submesh.vertex_tex_coords)
    #         indices_offset=[idx+indiceOffset for idx in submesh.indices]
    #         self.indices.extend(indices_offset)
    #         indiceOffset+=len(submesh.vertex_geometry)//3

        

    def commit(self) -> None:
  
        # self.flattern()
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



    def render(self):

        # Bind VAO
        if self.material is not None:
            self.material.shader_program.setUniformScope([self.parentScene, self.cameraObject,self])
            self.material.apply()

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
    def __init__(self, name="vectorGlyph"):
        super().__init__(name)
        def dirtyCallBack(obj) -> None:
            obj.dirty=True
        self.create_variable_callback("scale",1.0,dirtyCallBack,False,1.0)
        self.create_variable_callback("segments",10,dirtyCallBack,False,10)
        self.create_variable_callback("radius",0.01,dirtyCallBack,False,0.01)
        self.create_variable_callback("height",0.1,dirtyCallBack,False,0.1)
        self.create_variable_callback("sampling",0.5,dirtyCallBack,True,0.1)

        self.create_variable_gui("color",(0.2,-.2,0.2),False,{'widget': 'color_picker'})
        self.dirty = True

    def render(self):
        if self.dirty==True:
            actFieldWidget=self.parentScene.getObject("ActiveField")
            self.updateVectorGlyph(actFieldWidget.getActiveField(), actFieldWidget.time())
        super().render()
        return 

    def updateVectorGlyph(self,vector_field, time: float=0.0):
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
        time_idx = (time - vector_field.tmin) / vector_field.timeInterval
        lower_idx = int(np.floor(time_idx))
        upper_idx = int(np.ceil(time_idx))
        alpha = time_idx - lower_idx

        # Ensure indices are within the bounds of the vector field time steps
        lower_idx = max(0, min(vector_field.time_steps - 1, lower_idx))
        upper_idx = max(0, min(vector_field.time_steps - 1, upper_idx))

        # Get the two time slices and convert to numpy if needed
        lower_field = vector_field.field[lower_idx]
        upper_field = vector_field.field[upper_idx]
        # Convert PyTorch tensors to numpy if necessary
        if hasattr(lower_field, 'detach'):  # Check if it's a PyTorch tensor
            lower_field = lower_field.detach().numpy()
            upper_field = upper_field.detach().numpy()

        # Interpolate between the two time slices
        interpolated_field = (1 - alpha) * lower_field + alpha * upper_field
        radius=self.getValue("radius")
        hight=self.getValue("height")
        segments=self.getValue("segments")
        scale=self.getValue("scale")
        sampling_distance=max(self.getValue("sampling"),0.0001)
        # Calculate number of samples in each direction
        num_samples_x = int((vector_field.domainMaxBoundary[0]-vector_field.domainMinBoundary[0]) / sampling_distance)
        num_samples_y = int((vector_field.domainMaxBoundary[1]-vector_field.domainMinBoundary[1]) / sampling_distance)
        
        self.erase()
        for y in range(num_samples_y):
            for x in range(num_samples_x):
                # Calculate actual position
                posX = vector_field.domainMinBoundary[0] + x * sampling_distance
                posY = vector_field.domainMinBoundary[1] + y * sampling_distance
                # Convert position to grid coordinates for interpolation
                grid_x = (posX - vector_field.domainMinBoundary[0]) / vector_field.gridInterval[0]
                grid_y = (posY - vector_field.domainMinBoundary[1]) / vector_field.gridInterval[1]
                
                # Get interpolated vector at this position
                x_idx = int(grid_x)
                y_idx = int(grid_y)
                
                # Skip if outside field bounds
                if (x_idx >= interpolated_field.shape[1] - 1 or 
                    y_idx >= interpolated_field.shape[0] - 1 or 
                    x_idx < 0 or y_idx < 0):
                    continue
                
                # Bilinear interpolation weights
                fx = grid_x - x_idx
                fy = grid_y - y_idx
                
                # Get vectors at surrounding grid points
                v00 = interpolated_field[y_idx, x_idx]
                v10 = interpolated_field[y_idx, x_idx + 1]
                v01 = interpolated_field[y_idx + 1, x_idx]
                v11 = interpolated_field[y_idx + 1, x_idx + 1]
                
                # Bilinear interpolation
                vx = (1 - fx) * (1 - fy) * v00[0] + fx * (1 - fy) * v10[0] + \
                     (1 - fx) * fy * v01[0] + fx * fy * v11[0]
                vy = (1 - fx) * (1 - fy) * v00[1] + fx * (1 - fy) * v10[1] + \
                     (1 - fx) * fy * v01[1] + fx * fy * v11[1]
                vx=vx*scale
                vy=vy*scale
                direction = np.array([vx, vy, 0.0], dtype=np.float32)
                self.appendConeWithoutCommit(np.array([posX, posY, 0.0], dtype=np.float32),
                                          direction, radius, hight, segments)
        self.commit()
        self.dirty=False
          
        
class CoordinateSystem(Object):
                       
    def __init__(self, sceneArg:Scene):
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
       
   
    def drawGui(self):
        pass
    def render(self):
       for axis in self.Vaos:
            axis.cameraObject=self.cameraObject
            axis.render()
   
        
    
        






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
