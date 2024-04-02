import OpenGL.GL as gl
from OpenGL.GLU  import *
from .Object import Object
import numpy as np
import pygame
from .mainCommandUI import getlogger
import imgui

logger=getlogger()


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
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gluPerspective(self.fov, aspect_ratio, 0.1, 200.0)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        


    def apply_view(self):
        """Apply the camera view."""
        gl.glLoadIdentity()
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
            logger.debug(f"Zoom in: FOV changed from {old_fov} to {self.fov}")
        elif direction == 'out' and self.fov < 120:
            self.fov += 0.50
            logger.debug(f"Zoom in: FOV changed from {old_fov} to {self.fov}")
            
      
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
       
            
                



