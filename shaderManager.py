from OpenGL.GL import *
from OpenGL.GLU import *
import OpenGL.GL as gl
import pygame
from fileMonitor import FileMonitor
from GuiObjcts.Object import Object,singleton
import glm
import logging
logger=logging.getLogger()

def create_texture(image_data):
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image_data.shape[0],image_data.shape[1], 0, GL_RGB, GL_UNSIGNED_BYTE, image_data)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    return texture_id

@singleton
class TextureManager:
    def __init__(self,name):
        self.name=name
        self.textures = {}
        self.file_manager = FileMonitor()
        
    def add_texture(self, key, image_path):
        self.textures[key] = create_texture(image_path)
        self.file_manager.add_file(image_path)
        
    def get_texture(self, key):
        return self.textures[key]
    
    def check_for_updates(self):
        updated_files = self.file_manager.update_files()
        if updated_files:
            print(f"Texture files updated: {updated_files}")
            for key, image_path in updated_files.items():
                self.textures[key] = create_texture(image_path)

def getTextureManager() -> TextureManager:
    return TextureManager("TextureManager")

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
            logger.error(f'Error compiling {file_path}: {error}')
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
        self.program_id = None
        self.vertex_shader_id = None
        self.fragment_shader_id= None
        self._compile_and_link()       
    
    def _compile_and_link(self):
        # Create program object
        self.program_id = gl.glCreateProgram() if self.program_id is None else self.program_id
        if self.vertex_shader_id is not None:
            gl.glDetachShader(self.program_id,self.vertex_shader_id)
        if self.fragment_shader_id is not None:
            gl.glDetachShader(self.program_id,self.fragment_shader_id)

        # Create and attach vertex shader
        vertex_shader = Shader(self.vertex_shader_path, gl.GL_VERTEX_SHADER)
        self.vertex_shader_id = vertex_shader.get_id()
        # Create and attach fragment shader
        fragment_shader = Shader(self.fragment_shader_path, gl.GL_FRAGMENT_SHADER)
        self.fragment_shader_id =fragment_shader.get_id()
        # Attach shaders
        gl.glAttachShader(self.program_id, vertex_shader.get_id())
        gl.glAttachShader(self.program_id, fragment_shader.get_id())
        # Link program
        gl.glLinkProgram(self.program_id)
        # Check for linking errors
        if not gl.glGetProgramiv(self.program_id, gl.GL_LINK_STATUS):
            error = gl.glGetProgramInfoLog(self.program_id).decode()
            logger.error(f'Error linking program {self.program_id}-{self.key_name}: {error}')
            self.program_id = None
            raise RuntimeError("Program linking error")
        else:
            # Get and store the location of uniform variables
            self.uniform_locations = self._get_uniform_locations()
            logger.info(f'Shader program {self.key_name} created with ID {self.program_id}')
            # Delete shaders as they're linked into our program now and no longer necessary
            gl.glUseProgram(0)

    def deleteProgram(self):
            if self.program_id >0:
                gl.glDeleteProgram(self.program_id)
            if self.vertex_shader.get_id() >0:
                gl.glDeleteShader(self.vertex_shader.get_id())
            if self.fragment_shader.get_id() >0:
                gl.glDeleteShader(self.fragment_shader.get_id())
            

    def check_for_updates(self):
        updated_files = self.file_manager.update_files()
        if updated_files:
            print(f"Shader files updated: {updated_files}")
            self.needReload = True
            
    def _get_uniform_locations(self) -> dict:
        # Initialize an empty dictionary to hold uniform locations
        uniform_locations = {}

        # Get the number of active uniforms and the max name length
        num_uniforms = gl.glGetProgramiv(self.program_id, gl.GL_ACTIVE_UNIFORMS)
        max_name_length = gl.glGetProgramiv(self.program_id, gl.GL_ACTIVE_UNIFORM_MAX_LENGTH)

        # Iterate through all the active uniforms
        for i in range(num_uniforms):
            # Retrieve the name, size, and type of the i-th uniform
            name, size, uniform_type = gl.glGetActiveUniform(self.program_id, i)
            name = str(name.decode("utf-8"))

            # Retrieve the location (address) of the uniform
            location = gl.glGetUniformLocation(self.program_id, name)

            # Store the location, size, and type in the dictionary using the name as the key
            uniform_locations[name] = (location, size, uniform_type,False)

        return uniform_locations

    def setUniform(self, name, value):
        if name not in self.uniform_locations.keys():
            # logger.warning(f" Uniform {name} does not exist in the shader program.")
            return
        location, size, uniform_type,hasBeenSet = self.uniform_locations[name]
        # Set the uniform based on the type
        if uniform_type == gl.GL_FLOAT:
            gl.glUniform1f(location, value)
        elif uniform_type == gl.GL_INT:
            gl.glUniform1i(location, value)
        elif uniform_type == gl.GL_FLOAT_VEC2:
            gl.glUniform2fv(location, 1, value)
        elif uniform_type == gl.GL_FLOAT_VEC3:
            gl.glUniform3fv(location, 1, value)
        elif uniform_type == gl.GL_FLOAT_VEC4:
            gl.glUniform4fv(location, 1, value)
        elif uniform_type == gl.GL_FLOAT_MAT2:
            matrix_data = glm.value_ptr(value) if value.__class__==glm.mat2 else value
            gl.glUniformMatrix2fv(location, 1, gl.GL_FALSE, matrix_data)
        elif uniform_type == gl.GL_FLOAT_MAT3:
            matrix_data = glm.value_ptr(value) if value.__class__==glm.mat3 else value
            gl.glUniformMatrix3fv(location, 1, gl.GL_FALSE, matrix_data)
        elif uniform_type == gl.GL_FLOAT_MAT4:
            #only support glm.mat4
            matrix_data = glm.value_ptr(value) if value.__class__==glm.mat4 else value
            gl.glUniformMatrix4fv(location, 1, gl.GL_FALSE,  matrix_data)

        else:
            print(f"Warning: Uniform type {uniform_type} is not supported.")
        self.uniform_locations[name]=(location, size, uniform_type,True)

    def setUniformScope(self, uniformsScopeObjects ):
        """set the uniforms for the shader program from the scope of the objects

        Args:
            uniformsScopeObjects (list of [Object or dict]): the later object overwrite the former object's uniforms
        """
        self.Use()
        if uniformsScopeObjects is None:
            return
        dict0={}
        for obj in uniformsScopeObjects:
            uniforms=obj.getScope() 
            dict0.update(uniforms)
        self.setUnforms(dict0)
        self.checkUniforms()

    def checkUniforms(self):
        #check all active uniforms has been set
        for name, value in self.uniform_locations.items():
            if not value[3]:
                logger.warning(f"Shader program {self.key_name} 's Uniform {name} has not been set.")

        
    def setUnforms(self, uniforms:dict):
        for name, value in uniforms.items():
            self.setUniform(name, value)  
            
    def Use(self):
        self.check_for_updates()
        if self.needReload:
            logger.info("Reloading shader {self.key_name} due to file changes.")
            self._compile_and_link()
            self.needReload = False
        gl.glUseProgram(self.program_id)


        
class Material:
    def __init__(self,name:str, shader_name:str,texture0=None,texture1=None):
        """material is collection of shdader,texture,uniforms for rendering an object

        Args:
            shader_name (string): the key to retrieve the shader program from the shader manager
            texture0 (string): texture  name to retrieve the shader program from the shader manager;
            texture1 (string): texture name to retrieve the shader program from the shader manager;
            (maximum 2 texture)
        """
        self.name=name
        self.shader_name = shader_name
        self.uniforms = {}
        self.texture0 = texture0
        self.texture1= texture1
    
    def append_uniform(self, name, value):
        self.uniforms[name] = value

    def apply(self):
        sm=getShaderManager()
      
        if self.shader_name in sm.shaders:
            shader_program = sm.get_program(self.shader_name)
            shader_program.Use()
  
        #! TODO: use texture
        # tm=getTextureManager()
        #  if self.texture0 is not None:
        #     gl.glActiveTexture(gl.GL_TEXTURE0)
        #     gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        #     gl.glUniform1i(gl.glGetUniformLocation(self.shader_program, "texture1"), 0)
    


@singleton
class ShaderManager(Object):
    """Class for managing multiple shader programs
    """
    def __init__(self,name):
        super().__init__(name)
        self.shaders = {}  # Stores all shader programs
   
    # Adds a shader program with a given key
    def add_shader_program(self, key, vertex_shader_path, fragment_shader_path):
        self.shaders[key] = ShaderProgram(key,vertex_shader_path, fragment_shader_path)
      
    # Uses the shader program specified by the given key
    def use_program(self, key:str):
        if key in self.shaders:
            self.shaders[key].Use()
        else:
            logger.error(f'Shader program {key} not found.')
            
    def get_program(self, key:str):
        if key in self.shaders:
            return self.shaders[key]
        else:
            logger.error(f'Shader program {key} not found.')
    def getDefautlMaterial(self):
        return Material("defaultMaterial","simpleColor")

def getShaderManager() -> ShaderManager:
    return ShaderManager("ShaderManager")


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
    shader_manager = ShaderManager("test_sm")
    # Add a basic shader program
    shader_manager.add_shader_program('basic', 'shaders/simple_vertex.glsl', 'shaders/simple_fragment.glsl')
    # Use the basic shader program in rendering loop
    shader_program=shader_manager.get_program('basic')
    shader_program.setUniform('myFloat', 0.5)
   

    

if __name__ == '__main__':
    test_shader_manager()