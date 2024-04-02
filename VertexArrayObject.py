
from OpenGL import GL as gl
from GuiObjcts.Object import Object
    

class VertexArrayObject(Object):
    def __init__(self):
        # Create a new VAO and bind it
        self.vao_id = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao_id)

        # Initialize lists to store vertex and element data
        self.vertices = []
        self.indices = []

        # Create placeholders for VBO and EBO
        self.vbo_id = gl.glGenBuffers(1)
        self.ebo_id = gl.glGenBuffers(1)

    def appendVertexGeometry(self, vertex_data, index_data):
        # Append new vertex and index data
        self.vertices.extend(vertex_data)
        self.indices.extend(index_data)

    def commit(self):
        # Bind VAO
        gl.glBindVertexArray(self.vao_id)

        # Bind and set VBO
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_id)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, np.array(self.vertices, dtype=np.float32), gl.GL_STATIC_DRAW)

        # Bind and set EBO
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo_id)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, np.array(self.indices, dtype=np.uint32), gl.GL_STATIC_DRAW)

        # Specify the layout of the vertex data
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 6 * 4, None)
        gl.glEnableVertexAttribArray(0)
        
        # Unbind VAO
        gl.glBindVertexArray(0)

    def draw(self):
        # Bind VAO
        gl.glBindVertexArray(self.vao_id)
        # Draw elements
        gl.glDrawElements(gl.GL_TRIANGLES, len(self.indices), gl.GL_UNSIGNED_INT, None)
        # Unbind VAO
        gl.glBindVertexArray(0)

    def createPlane(self, gridSize, domainSize):
        self.vertices.clear()  # Clear any existing vertices
        self.indices.clear()  # Clear any existing indices
        
        dx = domainSize[0] / gridSize[0]  # Horizontal space between vertices
        dy = domainSize[1] / gridSize[1]  # Vertical space between vertices

        # Generate vertices
        for y in range(gridSize[1] + 1):
            for x in range(gridSize[0] + 1):
                # Vertex coordinates
                vx = x * dx - domainSize[0] / 2  # Shift so that the center of the plane is at (0,0)
                vy = y * dy - domainSize[1] / 2
                # Texture coordinates
                tx = x / gridSize[0]
                ty = y / gridSize[1]
                self.vertices.extend([vx, vy, tx, ty])

        # Generate indices for the grid cells
        for y in range(gridSize[1]):
            for x in range(gridSize[0]):
                # Calculate index of the square's first vertex
                start = y * (gridSize[0] + 1) + x
                self.indices.extend([
                    start, start + 1, start + gridSize[0] + 1,
                    start + 1, start + gridSize[0] + 2, start + gridSize[0] + 1
                ])

