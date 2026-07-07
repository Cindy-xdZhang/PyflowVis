from OpenGL.GL import *
from OpenGL.GLU import *
import OpenGL.GL as gl
import pygame
from misc.fileMonitor import FileMonitor
from misc.ImageLoader import ImageLoader
from GuiObjcts.Object import Object,singleton
import glm
import logging
logger=logging.getLogger()
import numpy as np
import os

def create_texture(image_data,data_type=gl.GL_RGB):
    texture_id = glGenTextures(1)
    # Convert the Pillow Image to a NumPy array
    image_array = np.array(image_data)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexImage2D(GL_TEXTURE_2D, 0, data_type, image_array.shape[0], image_array.shape[1], 0, data_type, GL_UNSIGNED_BYTE, image_array)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    return texture_id

#! todo: scalar field &color map
def load_images_from_folder(folder_path,extension="png"):

    globalImageLoader=ImageLoader()
    image_dict = {}
    for filename in os.listdir(folder_path):
        lowerCaseFileName=str(filename).lower()
        if lowerCaseFileName.endswith(".png"):
            file_path = os.path.join(folder_path, filename)
            image = globalImageLoader.load_image(file_path)

            image_dict[filename] = np.array(image,dtype=np.float32)/255.0
    return image_dict


def init_color_maps_texture_array() -> tuple[int, list[str], list[np.ndarray]]:
    # Using 1D texture array for color maps
    u_texture_array_index = glGenTextures(1)
    glBindTexture(GL_TEXTURE_1D_ARRAY, u_texture_array_index)
    glTexParameteri(GL_TEXTURE_1D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_1D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_1D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    path = "assets/colormaps/"
    colorMapImages=load_images_from_folder(path,"png")
    #regularize color maps image to same size 
    target_width = 1024
    resized_color_maps=[]
    for  color_map in colorMapImages.values():
        width:int=color_map.shape[1]
        resized_color_map = np.zeros((target_width, 3))
        for i in range(target_width):        
            fraction:float=float(i)/float(target_width)
            indexFloor:int=int(np.floor(fraction*(width-1))) 
            indexCeil:int=int(np.ceil(fraction*(width-1)))
            alpha=float(fraction*(width-1)-float(indexFloor)) 
            color0=color_map[0,indexFloor,:]
            color1=color_map[0,indexCeil,:]
            resized_color_map[i,:]=color0*(1-alpha)+color1*alpha
        resized_color_maps.append(resized_color_map)


    glTexStorage2D(GL_TEXTURE_1D_ARRAY, 1, GL_RGB32F, target_width, len(resized_color_maps))

    
    for i, color_map in enumerate(resized_color_maps):
        color_map_array = np.array(color_map)
        if color_map_array.flags['C_CONTIGUOUS']:
             # The array is in row-major order
            glTexSubImage2D(GL_TEXTURE_1D_ARRAY, 0, 0, i, target_width, 1, GL_RGB, GL_FLOAT, color_map_array)
        else:
            # The array is in column-major order, so we need to transpose it
            color_map_array = color_map_array.T
            glTexSubImage2D(GL_TEXTURE_1D_ARRAY, 0, 0, i, target_width, 1, GL_RGB, GL_FLOAT, color_map_array)

    glBindTexture(GL_TEXTURE_1D_ARRAY, 0)
    color_map_names=list(colorMapImages.keys())
    # Also hand back the CPU-side resampled colormaps (each (target_width,3), float) so callers
    # such as the DVR transfer-function LUT can bake colormap colors on the CPU.
    return u_texture_array_index, color_map_names, resized_color_maps


@singleton
class TextureManager:
    def __init__(self,name):
        self.name=name
    
        #initBuiltInTextures()
        self.BuiltItextureIDarray, self.BuiltInTextuireimageNames, self.builtin_colormaps_cpu= init_color_maps_texture_array()
        #additional textures
        self.texturesOpenGLID= {}
        


    def add_imageTexture(self, key, image_data):
        self.texturesOpenGLID[key] = create_texture(image_data)
    def get_texture(self, key:str=None):
        return self.texturesOpenGLID[key]
    


    def getBuiltInTextureNames(self):
        return self.BuiltInTextuireimageNames  
    
    def getBuiltInTextureIdArray(self):
        return self.BuiltItextureIDarray

    def getBuiltInColormapCPU(self, index:int):
        """Return the CPU-side resampled colormap (target_width,3) float32 for the given built-in
        colormap index (aligned with getBuiltInTextureNames()), used to bake DVR transfer-function
        colors on the CPU. Returns None if unavailable."""
        maps = getattr(self, "builtin_colormaps_cpu", None)
        if not maps:
            return None
        if index < 0 or index >= len(maps):
            index = 0
        return np.asarray(maps[index], dtype=np.float32)


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
    def __init__(self, key_name,vertex_shader_path, fragment_shader_path,geometry_shader_path=None):
        self.key_name = key_name
        self.vertex_shader_path = vertex_shader_path
        self.fragment_shader_path = fragment_shader_path
        self.geometry_shader_path = geometry_shader_path
        self.file_manager = FileMonitor([vertex_shader_path, fragment_shader_path,geometry_shader_path]) if geometry_shader_path is not None else FileMonitor([vertex_shader_path, fragment_shader_path])

        self.needReload = False
        self.program_id = None
        self.vertex_shader_id = None
        self.fragment_shader_id= None
        self.geometry_shader_id= None
        self.activeTextureUnitCounter=0
        self._compile_and_link()       
    
    def _compile_and_link(self):
        # Create program object
        self.program_id = gl.glCreateProgram() if self.program_id is None else self.program_id
        if self.vertex_shader_id is not None:
            gl.glDetachShader(self.program_id,self.vertex_shader_id)
        if self.fragment_shader_id is not None:
            gl.glDetachShader(self.program_id,self.fragment_shader_id)
        if self.geometry_shader_id is not None:
            gl.glDetachShader(self.program_id,self.geometry_shader_id)



      
        # Create and attach vertex shader
        vertex_shader = Shader(self.vertex_shader_path, gl.GL_VERTEX_SHADER)
        self.vertex_shader_id = vertex_shader.get_id()
        # Create and attach fragment shader
        fragment_shader = Shader(self.fragment_shader_path, gl.GL_FRAGMENT_SHADER)
        self.fragment_shader_id =fragment_shader.get_id()

    


        # Attach shaders
        gl.glAttachShader(self.program_id, vertex_shader.get_id())
        gl.glAttachShader(self.program_id, fragment_shader.get_id())
        if self.geometry_shader_path is not None:
            geometry_shader = Shader(self.geometry_shader_path, gl.GL_GEOMETRY_SHADER)
            self.geometry_shader_id = geometry_shader.get_id()
            gl.glAttachShader(self.program_id, self.geometry_shader_id)
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
            self.uniform_locations,self.uniform_Flags = self._get_uniform_locations()
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
        if self.geometry_shader_path is not None:
            gl.glDeleteShader(self.geometry_shader_id)
            

    def check_for_updates(self):
        updated_files = self.file_manager.update_files()
        if updated_files:
            print(f"Shader files updated: {updated_files}")
            self.needReload = True
            
    def _get_uniform_locations(self) -> dict:
        # Initialize an empty dictionary to hold uniform locations
        uniform_locations = {}
        uniform_flag :dict[str,bool]= {}

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
            uniform_locations[name] = (location, size, uniform_type)
            uniform_flag[name]=False

        return uniform_locations,uniform_flag

    def __setUniform(self, name, value):
        
        if name not in self.uniform_locations.keys():
            # logger.warning(f" Uniform {name} does not exist in the shader program.")
            return
        setSuccess=True
        location, size, uniform_type = self.uniform_locations[name]
        # Set the uniform based on the type
        if uniform_type == gl.GL_FLOAT:
            gl.glUniform1f(location, value)
        elif uniform_type == gl.GL_INT:
            gl.glUniform1i(location, value)
        elif uniform_type == gl.GL_UNSIGNED_INT:
            gl.glUniform1u(location, value)
        elif uniform_type == gl.GL_FLOAT_VEC2:
            gl.glUniform2fv(location, 1, value)
        elif uniform_type == gl.GL_FLOAT_VEC3:
            gl.glUniform3fv(location, 1, value)
        elif uniform_type == gl.GL_FLOAT_VEC4:
            gl.glUniform4fv(location, 1, value)
        elif uniform_type == gl.GL_FLOAT_MAT2:
            # glm matrices are column-major already; numpy matrices are almost always row-major.
            # If we feed a row-major 2x2 with transpose=GL_FALSE, OpenGL will interpret it as column-major
            # which effectively transposes (and can break transforms when used as part of a larger pipeline).
            if value.__class__ == glm.mat2:
                gl.glUniformMatrix2fv(location, 1, gl.GL_FALSE, glm.value_ptr(value))
            elif isinstance(value, np.ndarray):
                mat = np.ascontiguousarray(value, dtype=np.float32)
                gl.glUniformMatrix2fv(location, 1, gl.GL_TRUE, mat)
            else:
                gl.glUniformMatrix2fv(location, 1, gl.GL_FALSE, value)
        elif uniform_type == gl.GL_FLOAT_MAT3:
            if value.__class__ == glm.mat3:
                gl.glUniformMatrix3fv(location, 1, gl.GL_FALSE, glm.value_ptr(value))
            elif isinstance(value, np.ndarray):
                mat = np.ascontiguousarray(value, dtype=np.float32)
                gl.glUniformMatrix3fv(location, 1, gl.GL_TRUE, mat)
            else:
                gl.glUniformMatrix3fv(location, 1, gl.GL_FALSE, value)
        elif uniform_type == gl.GL_FLOAT_MAT4:
            # glm.mat4 is column-major, compatible with OpenGL when transpose=GL_FALSE.
            # Many objects in this repo store modelMat as numpy (row-major). For row-major matrices,
            # uploading with transpose=GL_FALSE will transpose them implicitly and may push translation into W,
            # causing severe perspective-like distortion. Upload numpy mats with transpose=GL_TRUE.
            if value.__class__ == glm.mat4:
                gl.glUniformMatrix4fv(location, 1, gl.GL_FALSE, glm.value_ptr(value))
            elif isinstance(value, np.ndarray):
                mat = np.ascontiguousarray(value, dtype=np.float32)
                gl.glUniformMatrix4fv(location, 1, gl.GL_TRUE, mat)
            else:
                gl.glUniformMatrix4fv(location, 1, gl.GL_FALSE, value)
        elif uniform_type in [gl.GL_SAMPLER_1D, gl.GL_SAMPLER_2D, gl.GL_SAMPLER_3D,
                         gl.GL_SAMPLER_1D_ARRAY, gl.GL_SAMPLER_2D_ARRAY]:
            setSuccess=self.set_sampler_uniform(location,uniform_type, name, value)
        else:
            setSuccess=False
            logger.warning(f"Warning: Uniform type {uniform_type} is not supported.")
        self.uniform_Flags[name]=setSuccess

    def set_sampler_uniform(self, location,uniform_type, name, value) -> bool:
        
        gl.glActiveTexture(gl.GL_TEXTURE0 + self.activeTextureUnitCounter)
        
        if uniform_type == gl.GL_SAMPLER_1D:
            gl.glBindTexture(gl.GL_TEXTURE_1D, value)
        elif uniform_type == gl.GL_SAMPLER_2D:
            gl.glBindTexture(gl.GL_TEXTURE_2D, value)
        elif uniform_type == gl.GL_SAMPLER_3D:
            gl.glBindTexture(gl.GL_TEXTURE_3D, value)
        elif uniform_type == gl.GL_SAMPLER_1D_ARRAY:
            gl.glBindTexture(gl.GL_TEXTURE_1D_ARRAY, value)
        elif uniform_type == gl.GL_SAMPLER_2D_ARRAY:
            gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, value)
        else:
            logger.warning(f" Uniform type {uniform_type} is not supported by set_sampler_uniform.")
            return False
        gl.glUniform1i(location, self.activeTextureUnitCounter)
        self.activeTextureUnitCounter += 1
        return True

    def setUniformScope(self, uniformsScopeObjects ):
        """set the uniforms for the shader program from the scope of the objects

        Args:
            uniformsScopeObjects (list of [Object or dict]): the later object overwrite the former object's uniforms
        """
        self.Use()
        if uniformsScopeObjects is None:
            return
         
        dict0={}
        for itemObj in uniformsScopeObjects:
            if itemObj is not None:
                uniformScope=itemObj.getScope() 
                dict0.update(uniformScope)
        # dict0 = {k: v for itemObj in uniformsScopeObjects if itemObj is not None for k, v in itemObj.getScope().items()}
        self.__setUnforms(dict0)
        self.checkUniforms()

    def checkUniforms(self):
        #check all active uniforms has been set
        for name, value in self.uniform_Flags.items():
            if not value:
                logger.warning(f"Shader program {self.key_name} 's Uniform {name} has not been set.")
                if name=="projMat" or name=="viewMat":
                    logger.warning(f"Shader program {self.key_name} 's  Camera Object is not set,maybe you forget to call super().postInit()?")

        
    def __setUnforms(self, uniforms:dict):
        self.activeTextureUnitCounter=0
        for name, value in uniforms.items():
            self.__setUniform(name, value)  
            
    def Use(self):
        self.check_for_updates()
        if self.needReload:
            logger.info("Reloading shader {self.key_name} due to file changes.")
            self._compile_and_link()
            self.needReload = False
        gl.glUseProgram(self.program_id)




@singleton
class ShaderManager(Object):
    """Class for managing multiple shader programs
    """
    def __init__(self,name):
        super().__init__(name)
        self.shaders = {}  # Stores all shader programs
        #compile built-in shader programs
        self.add_shader_program("monoColor","assets/shaders/simple_vertex.glsl","assets/shaders/simple_fragment.glsl")
        self.add_shader_program("colormapMat","assets/shaders/simple_vertex.glsl","assets/shaders/colorMap_fragment.glsl")
        self.add_shader_program("flowlineMat","assets/shaders/flowline_vertex.glsl","assets/shaders/flowline_fragment.glsl","assets/shaders/flowline_geometry.glsl")
        self.add_shader_program("glyphInstancedMat","assets/shaders/glyph_instanced_vertex.glsl","assets/shaders/glyph_instanced_fragment.glsl")
        self.add_shader_program("isoSurfaceMat","assets/shaders/iso_surface_vertex.glsl","assets/shaders/iso_surface_fragment.glsl")
        self.add_shader_program("dvrMat","assets/shaders/dvr_vertex.glsl","assets/shaders/dvr_fragment.glsl")
        self.add_shader_program("generalMeshMat","assets/shaders/general_mesh_vertex.glsl","assets/shaders/general_mesh_fragment.glsl")
        self.add_shader_program("planarManifoldMat","assets/shaders/planarManifold_vertex.glsl","assets/shaders/planarManifold_fragment.glsl")
        self.add_shader_program("corelineMat","assets/shaders/coreline_vertex.glsl","assets/shaders/coreline_fragment.glsl")
   
    # Adds a shader program with a given key
    def add_shader_program(self, key, vertex_shader_path, fragment_shader_path,geometry_shader_path=None):
        self.shaders[key] = ShaderProgram(key,vertex_shader_path, fragment_shader_path,geometry_shader_path)
      
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

    def getMaterial(self,material_name:str=None):
        if material_name is None or material_name=="defaultMaterial":
            return Material("defaultMaterial","monoColor")
        elif material_name=="colormapMat":
            return Material("coloprMapMaterial","colormapMat",texture0="builtIn")
        else:
            return Material(material_name)
        

def getShaderManager() -> ShaderManager:
    return ShaderManager("ShaderManager")


    
def test_shader_manager():
    """Test the ShaderManager class."""
    pygame.init()
    size = (800, 600)
    pygame.display.set_mode(size,  pygame.DOUBLEBUF | pygame.OPENGL| pygame.RESIZABLE)
    print("OpenGL initialized: Version", glGetString(GL_VERSION).decode())
    shader_manager = ShaderManager("test_sm")
    # Add a basic shader program
    shader_manager.add_shader_program('basic', 'shaders/simple_vertex.glsl', 'shaders/simple_fragment.glsl')
    # Use the basic shader program in rendering loop
    shader_program=shader_manager.get_program('basic')
    shader_program.setUniform('myFloat', 0.5)
   

            
class Material(Object):
    def __init__(self,material_name:str, shader_program_name:str=None ,texture0:str=None,texture1:str=None):
        """material is collection of shdader,texture for rendering an object

        Args:
            shader_program_name(string): the name of shader program from the shader manager
            texture0 (string): texture  name to retrieve the shader program from the shader manager;
            texture1 (string): texture name to retrieve the shader program from the shader manager;
            (maximum 2 texture)
        """
        super().__init__(material_name)
        self.shader_name=shader_program_name
        sm=getShaderManager()
        if self.shader_name is not None and self.shader_name in sm.shaders:
            shader_program = sm.get_program(self.shader_name)
            self.shader_program=shader_program
        if self.shader_program is None:
            logger.error(f"Shader program {self.shader_name} not found.")
            raise ValueError(f"Shader program {self.shader_name} not found.")

        tm=getTextureManager()
        if texture0 is not None:
            if  texture0=="builtIn":
                self.create_variable("colorMaps1Darray",getTextureManager().getBuiltInTextureIdArray())
            else:
                self.texture0 = texture0
                self.texture0_id = tm.get_texture(self.texture0) if self.texture0 in sm.shaders else None




    def apply(self,UniformObjectScope):
        """apply the material (shader program+texture +uniforms) to the current rendering context
            However, currently only the shader program is useful.
        """
        UniformObjectScope.append(self)
        self.shader_program.setUniformScope(UniformObjectScope)
        # self.shader_program.Use()
        # if self.texture0_id is not None and self.texture0 >0:
        #     gl.glActiveTexture(gl.GL_TEXTURE0)
        #     gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture0_id)
        #     gl.glUniform1i(gl.glGetUniformLocation(self.shader_program, "texture0"), 0)
    


if __name__ == '__main__':
    test_shader_manager()