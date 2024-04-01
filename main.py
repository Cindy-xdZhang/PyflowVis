import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
import imgui
from imgui.integrations.pygame import PygameRenderer
import numpy as np
import logging
from Object import *
from mainCommandUI import MainUICommand
from EventRegistrar import EventRegistrar
from train_vector_field import *
from flowCreator import *

def screen_to_world(x, y, width, height, modelview, projection, viewport):    
    y = height - y  # OpenGL's y axis  is reversed of pygame's y axis
    z = glReadPixels(x, y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT)
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
   
   
        
 


class Camera(Object):
    def __init__(self, fov, position, target,width, height):
        super().__init__("Camera")
        self.fov = fov
        # self.position = position
        self.init_position = position
        self.target = target
        self.width = width
        self.height = height
        self.last_mouse_pos = None
        self.mouse_down = False
        self.rotation_matrix = np.eye(4)  # Initialize as identity matrix for rotation
        self.create_variable("position",position,False)
        
        
        self.addAction("reset position", lambda object: object.resetCamera() )
    def resetCamera(self):
        self.position = self.init_position
        self.rotation_matrix = np.eye(4)
        self.fov = 45.0
    def update_windowSize(self, width, height):
        """Update the perspective projection based on new width and height."""
        self.width = width
        self.height = height
    

    def apply_projection(self):
        """Apply the perspective projection."""
        width= self.width 
        height= self.height
        aspect_ratio = width / height
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.fov, aspect_ratio, 0.1, 200.0)
        glMatrixMode(GL_MODELVIEW)
        


    def apply_view(self):
        """Apply the camera view."""
        glLoadIdentity()
        position=list(self.getValue("position")) 
        position.append(1)
        positionNew = np.dot( self.rotation_matrix, np.array(position) ) [:3]
        gluLookAt(
            *positionNew,  # Camera position
            *self.target,    # Look-at target
            0.0, 1.0, 0.0    # Up vector
        )
        
    def handle_mouse_move(self, x, y,up=False):
        """Handle the mouse movement to rotate the camera around the target."""
        if up==True:
            self.last_mouse_pos = None
            return
        if self.last_mouse_pos is None:
            self.last_mouse_pos = (x, y)
            return

        dx, dy = x - self.last_mouse_pos[0], y - self.last_mouse_pos[1]
        self.last_mouse_pos = (x, y)

        # Convert mouse movement to rotation angle
        sensitivity = 0.0025  # Adjust this value based on your preference
        angle_x = dy * sensitivity
        angle_y = dx * sensitivity

        # Update rotation_matrix based on mouse movement
        rotation_x = np.array([
            [1, 0, 0, 0],
            [0, np.cos(angle_x), -np.sin(angle_x), 0],
            [0, np.sin(angle_x), np.cos(angle_x), 0],
            [0, 0, 0, 1]
        ])
        
        rotation_y = np.array([
            [np.cos(angle_y), 0, np.sin(angle_y), 0],
            [0, 1, 0, 0],
            [-np.sin(angle_y), 0, np.cos(angle_y), 0],
            [0, 0, 0, 1]
        ])

        # Apply the rotations
        self.rotation_matrix = np.dot(rotation_y, np.dot(rotation_x, self.rotation_matrix))
    


    def zoom(self, direction):
        """Zoom the camera in/out."""
        old_fov = self.fov
        if direction == 'in' and self.fov > 10:
            self.fov -= 0.50
            logging.debug(f"Zoom in: FOV changed from {old_fov} to {self.fov}")
        elif direction == 'out' and self.fov < 120:
            self.fov += 0.50
            logging.debug(f"Zoom in: FOV changed from {old_fov} to {self.fov}")
            
      
    def eventCallBacks(self,event):
        if event.type == pygame.MOUSEBUTTONDOWN:
               # Zoom in
            if event.button == 4:
                self.zoom('in')
            # Zoom out
            elif event.button == 5:
                self.zoom('out')
            elif event.button == 1:  # Left mouse button
                self.mouse_down = True
        elif event.type == pygame.MOUSEMOTION and self.mouse_down and not(imgui.is_any_item_hovered() or imgui.is_any_item_active()):  # Only rotate when the left button is down
            x, y = event.pos  # Use relative motion for smoother rotation
            self.handle_mouse_move(x, y)
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:  # Left mouse button
            self.mouse_down = False
            x, y = event.pos  # Use relative motion for smoother rotation
            self.handle_mouse_move(x, y,up=True)
        elif event.type == pygame.VIDEORESIZE:
            self.update_windowSize(event.w, event.h)
       
            
                






class Renderable:
    def __init__(self, image_data,x=0.0,y=0.0,z=0.0):
        self.width, self.height = image_data.shape[1]/10, image_data.shape[0]/10
        self.texture_id = self.create_texture(image_data)
        self.x = x
        self.y = y
        self.z = z

   
    def check_collision(self, point):
        half_width, half_height = self.width / 2, self.height / 2
        min_x, max_x = self.x - half_width, self.x+ half_width
        min_y, max_y = self.y- half_height, self.y + half_height
        if min_x <= point[0] <= max_x and min_y <= point[1] <= max_y:
            self.selected = True
            logging.debug(f"click select success")
        else:
            self.selected = False

    def create_texture(self, image_data):
        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image_data.shape[0],image_data.shape[1], 0, GL_RGB, GL_UNSIGNED_BYTE, image_data)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        return texture_id

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
            


class ActiveField(Object):
    def __init__(self):
        super().__init__("ActiveField")
        self.active_vector_field_name="rotation_four_center"
        self.pause=False
        self.create_variable_gui("time",0.0,False, {'widget': 'input'})
        self.create_variable_gui("animationSpeed",0.01,False, {'widget': 'input'})

    def time(self):
        return self.getValue("time")

    def eventCallBacks(self,event):        
        time=self.getValue("time")
        if self.pause==False and  time<2*np.pi:
            time=time+self.getValue("animationSpeed")
            self.setValue("time",time)
        elif self.pause==False:
            self.pause=True

        if event.type == pygame.KEYDOWN and event.key == pygame.K_F11:
                time=0.0 if time>= 2*np.pi and self.pause==True else time
                self.pause = not self.pause
                self.setValue("time",time)
        
        



                    
                    



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
            gl.glTranslatef(posX,posY, 0)  # Position the glyph
            
            # Calculate the angle of the vector
            angle = np.arctan2(vy, vx) * 180 / np.pi
            
            gl.glRotatef(angle, 0, 0, 1)  # Rotate the glyph to match the vector direction
            gl.glScalef(scale, scale, 1)  # Scale the glyph
            
            # Draw the glyph as a simple line for now
            gl.glBegin(gl.GL_LINES)
            gl.glColor3f(1.0, 0.0, 0.0) 
            gl.glVertex2f(0, 0)
            gl.glColor3f(0.0, 1.0, 0.0) 
            gl.glVertex2f(1, 0)  # Draw line in the direction of the vector
            gl.glEnd()
       
            gl.glPopMatrix()  # Restore the matrix state
    
    gl.glPopMatrix()  # Restore the original matrix state
    
    
def main():
    pygame.init()
    size = (800, 600)
    pygame.display.set_mode(size,  pygame.DOUBLEBUF | pygame.OPENGL| pygame.RESIZABLE)
    # Configure logging to display all messages
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')    


    # Set up OpenGL context
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_TEXTURE_2D)  # Enable texture mapping
    glClearColor(0.1, 0.1, 0.1, 1)

    # Setup ImGui context and the pygame renderer for ImGui
    imgui.create_context()
    imgui.get_io().display_size = size
    imgui.get_io().fonts.get_tex_data_as_rgba32()
    impl=PygameRenderer()
     # Assuming you have your LIC texture data ready
    lic_texture_data = np.random.rand(100, 100, 3)*128   # Use random data as an example
    lic_texture_data = lic_texture_data.astype('uint8')
    
    # renderable_object = Renderable(lic_texture_data,2,2,-20)
    
    # all the objects in the scene
    scene=Scene("DefaultScene")
    # renderParamterPage=LicParameter("LicParameter")
    camera = Camera(45.0, (0, 0, 5), (0, 0, 0),size[0],size[1])
    eventRegister=EventRegistrar(impl)
    actFieldWidget=ActiveField()
    scene.add_object(actFieldWidget)
    scene.add_object(MainUICommand("mainCommandUI"))
    scene.add_object(GuiTest())

    # eventRegister.register(lambda event: renderable_object.eventCallBacks(event))
    eventRegister.register(lambda event: camera.eventCallBacks(event))
    eventRegister.register(lambda event: actFieldWidget.eventCallBacks(event))
    eventRegister.register(lambda event: scene.save_state_all() if event.type == pygame.KEYDOWN and event.key == pygame.K_F3 else None)
            
      
   
    # scene.add_object(renderParamterPage)
    # scene.add_object(camera)
    scene.restore_state_all()
    
    
    vectorField2d= constant_rotation((16,16),16)
    resUfield=train_pipeline(vectorField2d)
    clock = pygame.time.Clock()
    while eventRegister.running:
        eventRegister.handle_events()
        # Rendering
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        camera.apply_projection()
        camera.apply_view()
        

        imgui.new_frame()
        scene.drawGui()
        imgui.render()
        imgui.end_frame()
        
        scene.draw_all()

        impl.render(imgui.get_draw_data())

        drawVectorGlyph(resUfield, actFieldWidget.time())

        pygame.display.flip()
        pygame.time.wait(10)# Limit to 60 frames per second
     
     
    # scene.save_state_all()
    impl.shutdown()
    pygame.quit()

if __name__ == "__main__":
    main()