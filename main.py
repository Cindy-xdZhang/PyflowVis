import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
import imgui
from imgui.integrations.pygame import PygameRenderer
import numpy as np
import logging
from Object import *
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



class EventRegistrar:
    def __init__(self):
        self.event_actions = {}

    def register(self, event, action):
        if event not in self.event_actions:
            self.event_actions[event] = [action]
        else:
            self.event_actions[event].append(action)

    def handle_events(self):
        for event in pygame.event.get():
            actions = self.event_actions.get(event.type)
            if actions:
                for action in actions:
                    action(event)



from ImageLoader import ImageLoader
globalImageLoader=ImageLoader()

class LicParameter(Object):
    def __init__(self,name):
        super().__init__(name)
        self.create_variable("StepSize",0.01,True,0.01)
        self.create_variable("MaximumStepSize",100000,True,1) 
        self.create_variable("NoiseImage","assets//noise//512x512.png",True)
  

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
    impl = PygameRenderer()

     # Assuming you have your LIC texture data ready
    lic_texture_data = np.random.rand(100, 100, 3)*128   # Use random data as an example
    lic_texture_data = lic_texture_data.astype('uint8')
    
    renderable_object = Renderable(lic_texture_data,2,2,-20)
    
    # all the objects in the scene
    scene=Scene("DefaultScene")
    renderParamterPage=LicParameter("LicParameter")
    camera = Camera(45.0, (0, 0, 5), (0, 0, 0),size[0],size[1])
    eventRegister=EventRegistrar()
    
    scene.add_object(GuiTest())
    # scene.add_object(renderParamterPage)
    # scene.add_object(camera)
    scene.restore_state_all()
    
    clock = pygame.time.Clock()
    running = True
    mouse_down=False
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN :
                if event.button == 1:  # Left mouse button
                    mouse_down = True
                # Zoom in
                if event.button == 4:
                    camera.zoom('in')
                # Zoom out
                elif event.button == 5:
                    camera.zoom('out')
                elif event.button == 2:# mid button
                    x, y = event.pos
                    modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
                    projection = glGetDoublev(GL_PROJECTION_MATRIX)
                    viewport = glGetIntegerv(GL_VIEWPORT)
                    world_coordinates = screen_to_world(x, y, viewport[2], viewport[3], modelview, projection, viewport)
                    renderable_object.check_collision(world_coordinates[:3])
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left mouse button
                    mouse_down = False
                    x, y = event.pos  # Use relative motion for smoother rotation
                    camera.handle_mouse_move(x, y,up=True)
            elif event.type == pygame.MOUSEMOTION:
                if mouse_down and not(imgui.is_any_item_hovered() or imgui.is_any_item_active()):  # Only rotate when the left button is down
                    x, y = event.pos  # Use relative motion for smoother rotation
                    camera.handle_mouse_move(x, y)
            elif event.type == pygame.VIDEORESIZE:
                # Update the window size
                screen = pygame.display.set_mode((event.w, event.h), pygame.DOUBLEBUF | pygame.OPENGL | pygame.RESIZABLE)
                camera.update_windowSize(event.w, event.h)
             # Pass the pygame events to the ImGui Pygame renderer if we need imgui react(map pygame key to ImGui key etc.)
            impl.process_event(event)
      
                
       

       

        # Rendering
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        camera.apply_projection()
        camera.apply_view()
      

        imgui.new_frame()
        scene.DrawGui()
        imgui.render()
        imgui.end_frame()
        
        scene.draw_all()
        renderable_object.draw()
        impl.render(imgui.get_draw_data())
    

        pygame.display.flip()
        pygame.time.wait(10)# Limit to 60 frames per second
     
    scene.save_state()
    impl.shutdown()
    pygame.quit()

if __name__ == "__main__":
    main()