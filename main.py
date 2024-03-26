import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
import imgui
from imgui.integrations.pygame import PygameRenderer
import numpy as np
import logging

def screen_to_world(x, y, width, height, modelview, projection, viewport):    
    y = height - y  # OpenGL's y axis  is reversed of pygame's y axis
    z = glReadPixels(x, y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT)
    return gluUnProject(x, y, z, modelview, projection, viewport)

class Camera:
    def __init__(self, fov, position, target,width, height):
        self.fov = fov
        self.position = position
        self.target = target
        self.width = width
        self.height = height

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
        gluLookAt(
            *self.position,  # Camera position
            *self.target,    # Look-at target
            0.0, 1.0, 0.0    # Up vector
        )
    


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
    camera = Camera(45.0, (0, 0, 5), (0, 0, 0),size[0],size[1])

    # Setup ImGui context and the pygame renderer for ImGui
    imgui.create_context()
    imgui.get_io().display_size = size
    imgui.get_io().fonts.get_tex_data_as_rgba32()
    impl = PygameRenderer()

     # Assuming you have your LIC texture data ready
    lic_texture_data = np.random.rand(100, 100, 3)*128   # Use random data as an example
    lic_texture_data = lic_texture_data.astype('uint8')
    
    renderable_object = Renderable(lic_texture_data,2,2,-20)
    
    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN :
                # Zoom in
                if event.button == 4:
                    camera.zoom('in')
                # Zoom out
                elif event.button == 5:
                    camera.zoom('out')
                elif event.button == 1:# left button
                    x, y = event.pos
                    modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
                    projection = glGetDoublev(GL_PROJECTION_MATRIX)
                    viewport = glGetIntegerv(GL_VIEWPORT)
                    world_coordinates = screen_to_world(x, y, viewport[2], viewport[3], modelview, projection, viewport)
                    renderable_object.check_collision(world_coordinates[:3])

            elif event.type == pygame.VIDEORESIZE:
                # Update the window size
                screen = pygame.display.set_mode((event.w, event.h), pygame.DOUBLEBUF | pygame.OPENGL | pygame.RESIZABLE)
                camera.update_windowSize(event.w, event.h)
      
                
            # Pass the pygame events to the ImGui Pygame renderer
            impl.process_event(event)

       

        # Rendering
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        camera.apply_projection()
        camera.apply_view()
      


        #  Start new ImGui frame
        imgui.new_frame()
        # Example ImGui window
        if imgui.begin("Your GUI Window"):
            imgui.text("Hello, world!")
            # You can add more GUI elements here as needed
        imgui.end()
        imgui.render()
        imgui.end_frame()
        renderable_object.draw()
        impl.render(imgui.get_draw_data())
     




        pygame.display.flip()
        pygame.time.wait(10)# Limit to 60 frames per second
     
    impl.shutdown()
    pygame.quit()

if __name__ == "__main__":
    main()