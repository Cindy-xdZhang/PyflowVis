import pygame
import OpenGL.GL as gl
from OpenGL.GL import *
from OpenGL.GLU import *
import imgui
from imgui.integrations.pygame import PygameRenderer
import numpy as np
import logging
from GuiObjcts  import *
from EventRegistrar import EventRegistrar
from train_vector_field import *
from flowCreator import *
from functools import wraps
from VertexArrayObject import *
from shaderManager import *

def draw_on_dirty(func):
    """Decorator to skip drawing if the parameters have not changed.(wip)"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not hasattr(wrapper, "last_call_signature"):
            wrapper.last_call_signature = None

        current_call_signature = (args, tuple(sorted(kwargs.items())))
        
        if current_call_signature == wrapper.last_call_signature:
            print("Skipping drawing, parameters have not changed.")
            return
        else:
            wrapper.last_call_signature = current_call_signature
            return f
        
def screen_to_world(x, y, width, height, modelview, projection, viewport):    
    y = height - y  # OpenGL's y axis  is reversed of pygame's y axis
    z = gl.glReadPixels(x, y, 1, 1, gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT)
    return gluUnProject(x, y, z, modelview, projection, viewport)

class GuiTest(Object):
    def __init__(self):
        super().__init__("GuiTest")
        
        self.create_variable_gui("boolean_var", True, False,{'widget': 'checkbox'})
        self.create_variable_gui("boolean_var_default", True, False)
        self.create_variable_gui("checkbox_int",1,False,{'widget': 'checkbox'})
        self.create_variable_gui("input_int",1,False, {'widget': 'input'})
        self.create_variable_gui("default_int",1,False)
        self.create_variable_gui("slider_float",0.5,False, {'widget': 'slider_float', 'min': 0.0, 'max': 1.0})
        self.create_variable_gui("slider_float",0.5,False, {'widget': 'input'})
        self.create_variable_gui("default_float",0.5,False) 
        self.create_variable_gui("color_vec3", (255.0, 0.0, 0.0), False,{'widget': 'color_picker'})
        self.create_variable_gui("input_ivec3", (255, 0, 0), False,{'widget': 'input'})
        self.create_variable_gui("default_ivec3", (255, 0, 0), False)
        self.create_variable_gui("color_vec4",[1.0,1.0,1.0,1.0],False)
        self.appendGuiCustomization(ValueGuiCustomization("color_vec4","vec4",{'widget': 'color_picker'}) )
        
        self.create_variable("input_vec4", [1, 1, 1, 1])        
        self.create_variable_gui("default_vec4", (255, 0, 0,0))
        
        self.create_variable_gui("ivecn", (255, 0, 0,0,0,0))
        self.create_variable_gui("vecn", (255, 0, 0,0,0,0))
        self.create_variable_gui("float_array_var_plot", [0.1, 0.2, 0.3, 0.4,0.2], False,{'widget': 'plot_lines'})         
        self.create_variable_gui("string_var", "Hello ImGui", False,{'widget': 'input'})
        self.create_variable_gui("string_var2", "Hello ImGui", False)
        
        self.addAction("reload NoiseImage", lambda object: globalImageLoader.load_image(object.getValue("NoiseImage")) )
        testDictionary = {"a": 1, "b": 2, "StepSize2": 3.0,"sonDictionary":{"son_a": 11, "gradSondict": {"gradSon_b":22 }}}
        self.create_variable("testDictionary",testDictionary,True)
   
   
        



class Renderable:
    def __init__(self, image_data,x=0.0,y=0.0,z=0.0):
        self.width, self.height = image_data.shape[1]/10, image_data.shape[0]/10
        self.texture_id = create_texture(image_data)
        self.x = x
        self.y = y
        self.z = z
        self.name="renderable"

   
    def check_collision(self, point):
        half_width, half_height = self.width / 2, self.height / 2
        min_x, max_x = self.x - half_width, self.x+ half_width
        min_y, max_y = self.y- half_height, self.y + half_height
        if min_x <= point[0] <= max_x and min_y <= point[1] <= max_y:
            self.selected = True
            logging.debug(f"click select success")
        else:
            self.selected = False

    def draw(self):
    
        glPushMatrix()  # Save the current matrix state
        glTranslatef(self.x, self.y, self.z)  # Move the object to its position

        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 0.0); glVertex3f(-self.width / 2,  -self.height / 2,0)
        glTexCoord2f(1.0, 0.0); glVertex3f(self.width / 2,-self.height / 2,0)
        glTexCoord2f(1.0, 1.0); glVertex3f(self.width / 2, self.height / 2,0)
        glTexCoord2f(0.0, 1.0); glVertex3f(-self.width / 2,  self.height / 2,0)
        glEnd()
        glPopMatrix()  # Restore the matrix state
        
    def eventCallBacks(self,event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 2:# mid button
            x, y = event.pos
            modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
            projection = glGetDoublev(GL_PROJECTION_MATRIX)
            viewport = glGetIntegerv(GL_VIEWPORT)
            world_coordinates = screen_to_world(x, y, viewport[2], viewport[3], modelview, projection, viewport)
            self.check_collision(world_coordinates[:3])
            


                    
                    



from ImageLoader import ImageLoader
globalImageLoader=ImageLoader()





def drawVectorGlyph(vector_field, time: float=0.0, position=(0, 0, 0), scale=1.0):
    """
    Draw vector glyphs representing a vector field interpolated between two time steps.

    :param vector_field: A VectorField2D object representing the vector field.
    :param time: The specific time to interpolate the vector field at.
    :param position: The position where to start drawing the vector field.
    :param scale: Scale factor for drawing glyphs.
    """
    if vector_field is None:
        return
    gl.glPushMatrix()  # Save the current matrix state
    gl.glTranslatef(*position)  # Move to the specified position

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

    for y in range(interpolated_field.shape[0]):
        for x in range(interpolated_field.shape[1]):
            vx, vy = interpolated_field[y, x,:]  # Extract the vector components
            gl.glPushMatrix()  # Save the matrix state
            posX,posY=x * vector_field.gridInterval[0]+vector_field.domainMinBoundary[0], y * vector_field.gridInterval[1]+vector_field.domainMinBoundary[1]
            gl.glTranslatef(posX,posY, 0.1)  # Position the glyph
            
            # Calculate the angle of the vector
            angle = np.arctan2(vy, vx) * 180 / np.pi
            
            gl.glRotatef(angle, 0, 0, 1)  # Rotate the glyph to match the vector direction
            gl.glScalef(scale, scale, 1)  # Scale the glyph
            
            # Draw the glyph as a simple line for now
            gl.glBegin(gl.GL_LINES)
            gl.glColor3f(1.0, 0.0, 0.0) 
            gl.glVertex2f(0, 0)
            gl.glColor3f(0.0, 1.0, 0.0) 
            gl.glVertex2f(0.1, 0)  # Draw line in the direction of the vector
            gl.glEnd()
    
            gl.glPopMatrix()  # Restore the matrix state
    gl.glColor3f(1.0, 1.0, 1.0) 
    gl.glPopMatrix()  # Restore the original matrix state
 
  

def main():
    pygame.init()
    size = (800, 600)
    pygame.display.set_mode(size,  pygame.DOUBLEBUF | pygame.OPENGL| pygame.RESIZABLE)
    # Configure logging to display all messages
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')    


    # Set up OpenGL context
    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glEnable(gl.GL_TEXTURE_2D)  # Enable texture mapping
    gl.glClearColor(0.1, 0.1, 0.1, 1)

    # Setup ImGui context and the pygame renderer for ImGui
    imgui.create_context()
    imgui.get_io().display_size = size
    imgui.get_io().fonts.get_tex_data_as_rgba32()
    impl=PygameRenderer()
     # Assuming you have your LIC texture data ready
    lic_texture_data = np.random.rand(100, 100, 3)*128   # Use random data as an example
    lic_texture_data = lic_texture_data.astype('uint8')
    
    renderable_object = Renderable(lic_texture_data,5,2,-5)
    
    shaderManager=getShaderManager()
    shaderManager.add_shader_program("simpleColor","shaders/simple_vertex.glsl","shaders/simple_fragment.glsl")
    defaultMat=Material(name="simpleColorMat",shader_name="simpleColor")

    # all the objects in the scene
    scene=Scene("DefaultScene")
    # renderParamterPage=LicParameter("LicParameter")
    camera = Camera(45.0, (0, 0, 5), (0, 0, 0),size[0],size[1])
    eventRegister=EventRegistrar(impl)
    actFieldWidget=ActiveField()
    scene.add_object(actFieldWidget)
    scene.add_object(MainUICommand("mainCommandUI"))
    scene.add_object(GuiTest())
    scene.add_object(camera)
    # eventRegister.register(lambda event: renderable_object.eventCallBacks(event))
    eventRegister.register(lambda event: camera.eventCallBacks(event))
    eventRegister.register(lambda event: actFieldWidget.eventCallBacks(event))
    eventRegister.register(lambda event: scene.save_state_all() if event.type == pygame.KEYDOWN and event.key == pygame.K_F3 else None)
            
      
   

    # scene.add_object(camera)
    scene.restore_state_all()
    args={}
    # device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    device=torch.device("cpu")
    args['device'] = device    
    args["epochs"]=500
    
    vectorField2d= rotation_four_center((16,16),16)
    # resUfield=train_pipeline(vectorField2d,args)
    # actFieldWidget.insertField("rfc",vectorField2d)
    # actFieldWidget.insertField("Result field",resUfield)
    plane=VertexArrayObject("plane")
    plane.setGuiVisibility(False)
    vertices, indices, textures= createPlane([32,32],[-2.0,-2.0,2.0,2.0])
    plane.appendVertexGeometry(vertices, indices, textures)
    plane.setMaterial(defaultMat)
    scene.add_object(plane)

    clock = pygame.time.Clock()
    while eventRegister.running:
        eventRegister.handle_events()
        # Rendering
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        camera.apply_projection()
        camera.apply_view()
        
       

        imgui.new_frame()
        scene.drawGui()
        imgui.render()
        imgui.end_frame()
        
        scene.draw_all()
        # renderable_object.draw()

        impl.render(imgui.get_draw_data())

        drawVectorGlyph(actFieldWidget.getActiveField(), actFieldWidget.time())

        pygame.display.flip()
        pygame.time.wait(10)# Limit to 60 frames per second
     
     
    # scene.save_state_all()
    impl.shutdown()
    pygame.quit()

if __name__ == "__main__":
    main()