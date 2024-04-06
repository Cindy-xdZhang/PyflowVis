import OpenGL.GL as gl
from OpenGL.GLU  import *
from .Object import Object
import numpy as np
import pygame
from .mainCommandUI import getlogger
import imgui
import glm

logger=getlogger()

def screen_to_arcball(x, y, width, height):
    px = 1.0 - (x * 2.0) / width
    py = (y * 2.0) / height - 1.0
    distance = px * px + py * py
    if distance <= 1.0:
        return glm.vec3(px, py, glm.sqrt(1.0 - distance))
    else:
        return glm.normalize(glm.vec3(px, py, 0))
    
def glm_mat4_to_np_array(glm_mat):
    return np.array([
        [glm_mat[0][0], glm_mat[1][0], glm_mat[2][0], glm_mat[3][0]],
        [glm_mat[0][1], glm_mat[1][1], glm_mat[2][1], glm_mat[3][1]],
        [glm_mat[0][2], glm_mat[1][2], glm_mat[2][2], glm_mat[3][2]],
        [glm_mat[0][3], glm_mat[1][3], glm_mat[2][3], glm_mat[3][3]]
    ], dtype=np.float32)

class Camera(Object):
    def __init__(self, fov, position, center,up,width, height):
        super().__init__("Camera")
        #cache value for resetting the camera
        self.init_fov= fov
        self.init_position = np.array(position,dtype=np.float32)
        self.init_targetDirection =  np.array(center,dtype=np.float32)-  self.init_position
        self.init_up = glm.vec3(up)
        

        self.create_variable("position", np.array(position,dtype=np.float32),False)
        self.fov = fov
        self.targetDirection   =self.init_targetDirection
        self.up = glm.vec3(up)
        self.width = width
        self.height = height
        self.aspect_ratio = width / height
        self.last_mouse_pos = None
        self.mouse_down = False
        self.rotation_matrix = np.eye(4, dtype=np.float32) # Initialize as identity matrix for rotation
     
        
        self.addAction("z positive", lambda object: object.look_at_z_positive() )
        self.addAction("z negative", lambda object: object.look_at_z_negative() )
        self.addAction("reset position", lambda object: object.resetCamera() )

    def resetCamera(self):
        self.fov = self.init_fov
        self.target = self.init_targetDirection
        self.up = self.init_up
        self.setValue("position", self.init_position)
        self.rotation_matrix = np.eye(4, dtype=np.float32)
        
    def getScope(self ):
        """overide the getScope method to return view and projection matrix"""
        AllVariables=  {}
        viewMat= self.get_view_matrix()
        projMat=self.get_projection_matrix()
        AllVariables["viewMat"]=viewMat 
        AllVariables["projMat"]=projMat
        return AllVariables

    def get_view_matrix(self):
        """Get the view matrix."""
        pos= self.getValue("position")
        targetDirection=self.targetDirection
        targetDirection = np.append(targetDirection, 1)
        targetDirectionNew = np.dot( self.rotation_matrix.transpose(), np.array(targetDirection) ) [:3]
        targetNew = pos + targetDirectionNew
        targetNew = glm.vec3(targetNew)
        return glm.lookAt(pos, targetNew, self.up)

    def get_projection_matrix(self):
        return glm.perspective( glm.radians(self.fov), self.aspect_ratio, 0.1, 100.0)

   

    def update_window_size(self, width, height):
        """Update the window size and recalculate the projection matrix."""
        self.width = width
        self.height = height
        self.aspect_ratio = width / height
        # self.projection_matrix = self.get_projection_matrix()

    def look_at_z_positive(self):
        """Adjust the camera to look at the Z positive direction."""
        # pos= self.getValue("position")
        self.targetDirection = np.array([0, 0, 1])

    def look_at_z_negative(self):
        """Adjust the camera to look at the Z negative direction."""
        # pos= self.getValue("position")
        self.targetDirection  =  np.array([0, 0, -1])

    def resetCamera(self):
        """Reset the camera to the initial state."""
        self.fov = self.init_fov
        self.updateValue("position", self.init_position)
        self.targetDirection = self.init_targetDirection
        self.rotation_matrix = np.eye(4, dtype=np.float32)

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
      
  
    
    def pan(self, dx:float, dy:float,dz:float):
        """
        Pans the camera based on horizontal (dx) and vertical (dy) input values.
        """
        def normalize_vector(v):
            norm = np.linalg.norm(v)
            if norm == 0:
                return v
            return v / norm
        # Calculate the right vector as the cross product of the target direction and the up vector
        up_vec = np.array([ self.up.x,  self.up.y,  self.up.z], dtype=np.float32)
        right = np.cross(self.targetDirection,  up_vec )
        right=normalize_vector(right) 

        # Calculate the actual movement vectors
        right_movement = right * dx
        up_movement = glm.normalize(self.up) * dy
        z_movement = normalize_vector(self.targetDirection) * dz
        
        # Convert movement vectors from glm to numpy for calculations
        up_movement_np = np.array([up_movement.x, up_movement.y, up_movement.z], dtype=np.float32)
        
        # Update the position and target direction based on the movements
        self.updateValue("position", self.getValue("position") + right_movement + up_movement_np+z_movement)



    def zoom(self, direction):
        """Zoom the camera in/out."""
        old_fov = self.fov
        if direction == 'in' and self.fov > 10:
            self.fov -= 1.0
        elif direction == 'out' and self.fov < 120:
            self.fov += 1.0
        # self.projection_matrix = self.get_projection_matrix()
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
            self.update_window_size(event.w, event.h)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT or event.key == pygame.K_a:
                self.pan(-0.1, 0,0)  # Pan left
            elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                self.pan(0.1, 0,0)  # Pan right
            elif event.key == pygame.K_UP or event.key == pygame.K_w:
                self.pan(0, 0.1,0)  # Pan up
            elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                self.pan(0, -0.1,0)  # Pan down
            elif event.key == pygame.K_q:
                self.pan(0, 0, 0.1)  # Pan forward (Q)
            elif event.key == pygame.K_e:
                self.pan(0, 0, -0.1)  # Pan backward (E)

