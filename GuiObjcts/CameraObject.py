import OpenGL.GL as gl
from OpenGL.GLU  import *
from .Object import Object
import numpy as np
import pygame
import imgui
import glm
import logging
logger=logging.getLogger()

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
    def __init__(self, fov, position, center, up, width, height):
        super().__init__("Camera")
        # cache value for resetting the camera
        self.init_fov = fov
        self.init_position = np.array(position, dtype=np.float32)
        self.init_targetDirection = (np.array(center, dtype=np.float32) - self.init_position)
        self.init_targetDirection = self.init_targetDirection / np.linalg.norm(self.init_targetDirection)
        self.init_up =up/np.linalg.norm(up)

        # persist to load/save camera configuration
        self.create_variable_callback("position", np.array(position, dtype=np.float32), lambda x: self.updateMVPVariables(), True)
        self.create_variable_callback("fov", fov, lambda x: self.updateMVPVariables(), True)
        # Projection mode toggle (combo box in GUI). "orthographic" removes perspective
        # foreshortening entirely, which is what you usually want for 2D flow fields.
        self.create_variable_callback("projectionMode", ["perspective", "orthographic"], lambda x: self.updateMVPVariables(), True, True)
        self.create_variable_callback("targetDirection", np.array(self.init_targetDirection, dtype=np.float32), lambda x: self.updateMVPVariables(), True)
        self.create_variable_callback("up", np.array(up, dtype=np.float32), lambda x: self.updateMVPVariables(), True)
        self.create_variable("rotation_matrix", np.eye(4, dtype=np.float32), True,False)

 
        self.width = width
        self.height = height
        self.aspect_ratio = width / height
        self.last_mouse_pos = None
        self._last_arcball_vec = None
        self.mouse_down = False

        self.addAction("z positive", lambda object: object.look_at_z_positive())
        self.addAction("z negative", lambda object: object.look_at_z_negative())
        self.addAction("reset position", lambda object: object.resetCamera())
        self.MVPVariables = {"viewMat": None, "projMat": None}
        self.updateMVPVariables()

    def OnRestoreState(self):
        self.updateMVPVariables()

    def resetCamera(self):
        self.updateValue("fov", self.init_fov)
        self.updateValue("targetDirection", self.init_targetDirection)
        self.updateValue("up", np.array(self.init_up, dtype=np.float32))
        self.setValue("position", self.init_position)
        self.updateValue("rotation_matrix", np.eye(4, dtype=np.float32))
        self.updateMVPVariables()

    def updateMVPVariables(self):
        self.MVPVariables["viewMat"] = self.get_view_matrix()
        self.MVPVariables["projMat"] = self.get_projection_matrix()

    def getScope(self):
        """override the getScope method to return view and projection matrix"""
        return self.MVPVariables

    def get_view_matrix(self):
        """Get the view matrix."""
        pos = np.array(self.getValue("position"), dtype=np.float32)

        # Apply camera rotation (column-vector convention): v' = R @ v
        rotation_matrix = np.array(self.getValue("rotation_matrix"), dtype=np.float32)
        target_dir = np.array(self.getValue("targetDirection"), dtype=np.float32)
        up_dir = np.array(self.getValue("up"), dtype=np.float32)

        target4 = np.array([target_dir[0], target_dir[1], target_dir[2], 0.0], dtype=np.float32)
        up4 = np.array([up_dir[0], up_dir[1], up_dir[2], 0.0], dtype=np.float32)
        target_dir_new = (rotation_matrix @ target4)[:3]
        up_dir_new = (rotation_matrix @ up4)[:3]

        # Robust normalize (avoid NaNs)
        td_norm = np.linalg.norm(target_dir_new)
        if td_norm > 1e-8:
            target_dir_new = target_dir_new / td_norm
        up_norm = np.linalg.norm(up_dir_new)
        if up_norm > 1e-8:
            up_dir_new = up_dir_new / up_norm

        target_new = pos + target_dir_new
        return glm.lookAt(glm.vec3(*pos), glm.vec3(*target_new), glm.vec3(*up_dir_new))
    
    def get_projection_matrix(self):
        fov = self.getValue("fov")
        # 根据相机位置动态调整近平面
        pos = self.getValue("position")
        distance_to_origin = np.linalg.norm(pos)
        near_plane = max(0.5, distance_to_origin * 0.1)  # 动态近平面
        far_plane = max(100.0, distance_to_origin * 10)    # 动态远平面

        if self.getOptionValue("projectionMode") == "orthographic":
            # Match the on-screen scale of the perspective view at the focal distance
            # (assumed to be the scene center near the origin, consistent with the
            # near/far heuristic above) so toggling does not jump the framing.
            # half_height = focal_distance * tan(fov/2); ortho has no foreshortening.
            half_height = max(distance_to_origin * np.tan(np.radians(fov) * 0.5), 1e-3)
            half_width = half_height * self.aspect_ratio
            return glm.ortho(-half_width, half_width, -half_height, half_height, near_plane, far_plane)

        return glm.perspective(glm.radians(fov), self.aspect_ratio, near_plane, far_plane)
    


    def update_window_size(self, width, height):
        """Update the window size and recalculate the projection matrix."""
        self.width = width
        self.height = height
        self.aspect_ratio = width / height
        self.updateMVPVariables()

    def look_at_z_positive(self):
        """Adjust the camera to look at the Z positive direction."""
        targetDirection = np.array([0, 0, 1])
        self.updateValue("targetDirection", targetDirection)
        self.updateMVPVariables()

    def look_at_z_negative(self):
        """Adjust the camera to look at the Z negative direction."""
        targetDirection = np.array([0, 0, -1])
        self.updateValue("targetDirection", targetDirection)
        self.updateMVPVariables()

    def handle_mouse_move(self, x, y, up=False):
        """Handle the mouse movement to rotate the camera around the target."""
        if up is True:
            self.last_mouse_pos = None
            self._last_arcball_vec = None
            return
        if self.last_mouse_pos is None:
            self.last_mouse_pos = (x, y)
            self._last_arcball_vec = screen_to_arcball(x, y, self.width, self.height)
            return

        self.last_mouse_pos = (x, y)
        v0 = self._last_arcball_vec
        v1 = screen_to_arcball(x, y, self.width, self.height)
        self._last_arcball_vec = v1

        # Arcball: rotate from v0 to v1 around axis = v0 x v1
        axis = glm.cross(v0, v1)
        axis_len = glm.length(axis)
        if axis_len < 1e-7:
            return
        dot01 = float(glm.dot(v0, v1))
        dot01 = max(-1.0, min(1.0, dot01))
        angle = float(np.arccos(dot01))

        ax = np.array([axis.x, axis.y, axis.z], dtype=np.float32) / float(axis_len)
        c = float(np.cos(angle))
        s = float(np.sin(angle))
        C = 1.0 - c
        x0, y0, z0 = float(ax[0]), float(ax[1]), float(ax[2])
        r3 = np.array([
            [c + x0*x0*C,     x0*y0*C - z0*s, x0*z0*C + y0*s],
            [y0*x0*C + z0*s,  c + y0*y0*C,    y0*z0*C - x0*s],
            [z0*x0*C - y0*s,  z0*y0*C + x0*s, c + z0*z0*C   ],
        ], dtype=np.float32)
        r4 = np.eye(4, dtype=np.float32)
        r4[:3, :3] = r3

        # Compose rotation in camera local space: R_new = R_current @ R_inc
        rotation_matrix = np.array(self.getValue("rotation_matrix"), dtype=np.float32)
        rotation_matrix = rotation_matrix @ r4
        self.updateValue("rotation_matrix", rotation_matrix)

        self.updateMVPVariables()

    def pan(self, dx: float, dy: float, dz: float):
        """
        Pans the camera based on horizontal (dx) and vertical (dy) input values.
        """
        def _norm(v: np.ndarray) -> np.ndarray:
            n = np.linalg.norm(v)
            return v if n < 1e-8 else (v / n)

        rotation_matrix = np.array(self.getValue("rotation_matrix"), dtype=np.float32)
        forward0 = np.array(self.getValue("targetDirection"), dtype=np.float32)
        up0 = np.array(self.getValue("up"), dtype=np.float32)

        f4 = np.array([forward0[0], forward0[1], forward0[2], 0.0], dtype=np.float32)
        u4 = np.array([up0[0], up0[1], up0[2], 0.0], dtype=np.float32)
        forward = _norm((rotation_matrix @ f4)[:3])
        up_vec = _norm((rotation_matrix @ u4)[:3])
        right = _norm(np.cross(forward, up_vec))

        right_movement = right * dx
        up_movement = up_vec * dy
        z_movement = forward * dz

        new_position = np.array(self.getValue("position") + right_movement + up_movement + z_movement, dtype=np.float32)
        self.updateValue("position", new_position )
        self.updateMVPVariables()

    def zoom(self, direction):
        """Zoom the camera in/out."""
        fov = self.getValue("fov")
        if direction == 'in' and fov > 10:
            self.updateValue("fov", fov - 1.0)
        elif direction == 'out' and fov < 50:
            self.updateValue("fov", fov + 1.0)
        self.updateMVPVariables()

    def eventCallBacks(self, event):
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
            self.handle_mouse_move(x, y, up=True)
        elif event.type == pygame.VIDEORESIZE:
            self.update_window_size(event.w, event.h)
        if not(imgui.is_any_item_hovered() or imgui.is_any_item_active()):
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                self.pan(-0.1, 0, 0)  # Pan left
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                self.pan(0.1, 0, 0)  # Pan right
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                self.pan(0, 0.1, 0)  # Pan up
            if keys[pygame.K_DOWN] or keys[pygame.K_s]:
                self.pan(0, -0.1, 0)  # Pan down
            if keys[pygame.K_q]:
                self.pan(0, 0, 0.1)  # Pan forward
            if keys[pygame.K_e]:
                self.pan(0, 0, -0.1)  # Pan backward



