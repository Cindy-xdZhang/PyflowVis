from OpenGL.GL import *
from OpenGL.GLU import *
import OpenGL.GL as gl
import pygame
from fileMonitor import FileMonitor

# General Shader class for compiling GLSL shaders
class Shader:
    def __init__(self, file_path, shader_type):
        # Create shader object
        self.shader_id = gl.glCreateShader(shader_type)
        # Read shader source
        with open(file_path, 'r') as file:
            gl.glShaderSource(self.shader_id, file.read())
        # Compile shader
        gl.glCompileShader(self.shader_id)
        # Check for compilation errors
        if not gl.glGetShaderiv(self.shader_id, gl.GL_COMPILE_STATUS):
            error = gl.glGetShaderInfoLog(self.shader_id).decode()
            print(f'Error compiling {file_path}: {error}')
            raise RuntimeError("Shader compilation error")
      # Returns the shader id
    def get_id(self):
        return self.shader_id

# Class to link vertex and fragment shaders into a shader program
class ShaderProgram:
    def __init__(self, key_name,vertex_shader_path, fragment_shader_path):
        self.key_name = key_name
        self.vertex_shader_path = vertex_shader_path
        self.fragment_shader_path = fragment_shader_path
        self.file_manager = FileMonitor([vertex_shader_path, fragment_shader_path])
        self.needReload = False
        self.compile_and_link()       
        
    def compile_and_link(self):
        # Create program object
        self.program_id = gl.glCreateProgram()
        # Create and attach vertex shader
        vertex_shader = Shader(self.vertex_shader_path, gl.GL_VERTEX_SHADER)
        # Create and attach fragment shader
        fragment_shader = Shader(self.fragment_shader_path, gl.GL_FRAGMENT_SHADER)
        # Attach shaders
        gl.glAttachShader(self.program_id, vertex_shader.get_id())
        gl.glAttachShader(self.program_id, fragment_shader.get_id())
        # Link program
        gl.glLinkProgram(self.program_id)
        # Check for linking errors
        if not gl.glGetProgramiv(self.program_id, gl.GL_LINK_STATUS):
            error = gl.glGetProgramInfoLog(self.program_id).decode()
            print(f'Error linking program: {error}')
            self.program_id = None
            raise RuntimeError("Program linking error")
        else:
            # Get and store the location of uniform variables
            self.uniform_locations = self._get_uniform_locations()
            print(f'Shader program {self.key_name} created with ID {self.program_id}')
        # Delete shaders as they're linked into our program now and no longer necessary
        gl.glDeleteShader(vertex_shader.get_id())
        gl.glDeleteShader(fragment_shader.get_id())
        
    def check_for_updates(self):
        updated_files = self.file_manager.update_files()
        if updated_files:
            print(f"Shader files updated: {updated_files}")
            self.needReload = True
            
    def _get_uniform_locations(self):
        # Activate the program to retrieve uniforms
        self.useShaderProgram()
        
        # Initialize an empty dictionary to hold uniform locations
        uniform_locations = {}
        
        # Get the number of active uniforms and the max name length
        num_uniforms = gl.glGetProgramiv(self.program_id, gl.GL_ACTIVE_UNIFORMS)
        max_name_length = gl.glGetProgramiv(self.program_id, gl.GL_ACTIVE_UNIFORM_MAX_LENGTH)
        
        # Iterate through all the active uniforms
        for i in range(num_uniforms):
            # Retrieve the name of the i-th uniform
            name, size, uniform_type = gl.glGetActiveUniform(self.program_id, i)
            
            # Retrieve the location (address) of the uniform
            location = gl.glGetUniformLocation(self.program_id, name)
            
            # Store the location in the dictionary using the name as the key
            uniform_locations[name] = location
        
        # Return to the caller with the dictionary of uniform locations
        return uniform_locations

    def setUniform(self, name, value, type="float"):
        self.useShaderProgram()        
        if name not in self.uniform_locations:
            print(f"Warning: Uniform {name} does not exist in the shader program.")
            return
        location = self.uniform_locations[name]
        # Float
        if type == "float":
            gl.glUniform1f(location, value)
        # Integer
        elif type == "int":
            gl.glUniform1i(location, value)
        # Vector2
        elif type == "vec2":
            gl.glUniform2fv(location, 1, value)
        # Vector3
        elif type == "vec3":
            gl.glUniform3fv(location, 1, value)
        # Vector4
        elif type == "vec4":
            gl.glUniform4fv(location, 1, value)
        # 2x2 Matrix
        elif type == "mat2":
            gl.glUniformMatrix2fv(location, 1, GL_FALSE, value)
        # 3x3 Matrix
        elif type == "mat3":
            gl.glUniformMatrix3fv(location, 1, GL_FALSE, value)
        # 4x4 Matrix
        elif type == "mat4":
            gl.glUniformMatrix4fv(location, 1, GL_FALSE, value)
        else:
            print(f"Warning: Uniform type {type} is not supported.")
    
    def setUnforms(self, uniforms:dict):
        for name, value in uniforms.items():
            self.setUniform(name, value)  
            
    def useShaderProgram(self):
        self.check_for_updates()
        if self.needReload:
            print("Reloading shaders due to file changes.")
            self.compile_and_link()
            self.needReload = False
        gl.glUseProgram(self.program_id)

# Class for managing multiple shader programs
class ShaderManager:
    def __init__(self):
        self.shaders = {}  # Stores all shader programs

    # Adds a shader program with a given key
    def add_shader_program(self, key, vertex_shader_path, fragment_shader_path):
        self.shaders[key] = ShaderProgram(key,vertex_shader_path, fragment_shader_path)

    # Uses the shader program specified by the given key
    def use_program(self, key:str):
        if key in self.shaders:
            self.shaders[key].use()
        else:
            print(f'Shader program {key} not found.')
            
    def get_program(self, key:str):
        if key in self.shaders:
            return self.shaders[key]
        else:
            print(f'Shader program {key} not found.')


def setup_opengl():
    pygame.init()
    size = (800, 600)
    pygame.display.set_mode(size,  pygame.DOUBLEBUF | pygame.OPENGL| pygame.RESIZABLE)
    print("OpenGL initialized: Version", glGetString(GL_VERSION).decode())
    # Now it's safe to call OpenGL functions; create shaders, buffers, etc.
    #program_id = glCreateProgram()
    
    
    
def test_shader_manager():
    """Test the ShaderManager class."""
    setup_opengl()
    shader_manager = ShaderManager()
    # Add a basic shader program
    shader_manager.add_shader_program('basic', 'shaders/simple_vertex.glsl', 'shaders/simple_fragment.glsl')
    # Use the basic shader program in rendering loop
    shader_program=shader_manager.get_program('basic')
    shader_program.setUniform('myFloat', 0.5)
    shader_program.setUniform('myInt', 20, 'int')
    shader_program.setUniform('myVec2', [0.5, 0.5], 'vec2')
    shader_program.setUniform('myVec3', [0.5, 0.5, 0.5], 'vec3')
    shader_program.setUniform('myVec4', [0.5, 0.5, 0.5, 1.0], 'vec4')
    shader_program.setUniform('myMat2', [1.0, 0.0, 0.0, 1.0], 'mat2')
    shader_program.setUniform('myMat3', [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], 'mat3')
    shader_program.setUniform('myMat4', [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0], 'mat4')
        
if __name__ == '__main__':
    test_shader_manager()