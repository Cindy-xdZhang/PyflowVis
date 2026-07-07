import OpenGL.GL as gl
from OpenGL.GLU  import *
from .Object import Object
import numpy as np
import pygame
import imgui
import glm
import logging
logger=logging.getLogger()


def _rotation_about_axis(axis, angle):
    """Rodrigues rotation matrix (3x3) for a right-handed rotation of `angle`
    radians about a (not necessarily unit) `axis`."""
    a = np.asarray(axis, dtype=np.float32)
    n = float(np.linalg.norm(a))
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)
    x, y, z = (a / n)
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    C = 1.0 - c
    return np.array([
        [c + x*x*C,   x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, c + z*z*C  ],
    ], dtype=np.float32)


def _normalize(v, fallback=None):
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return np.asarray(fallback, dtype=np.float32) if fallback is not None else v
    return v / n


class Camera(Object):
    """Orbit ("turntable") camera.

    The camera always looks at a pivot point ``center`` from ``position`` (the
    eye), with ``up`` kept level.  Dragging orbits the eye around the pivot
    (yaw about the world-up axis + pitch about the camera-right axis) so the
    scene never leaves the view, and the horizon never rolls.  Axis-view
    buttons snap the eye onto a principal axis, so you can always return to a
    clean front/top/side view.

    Only ``viewMat`` / ``projMat`` (via :meth:`getScope`) and the
    ``get_*_matrix`` helpers are consumed by the rest of the engine, so the
    internal parameterization here is free to change.
    """

    def __init__(self, fov, position, center, up, width, height):
        super().__init__("Camera")
        # ---- cached values for reset ----
        self.init_fov = float(fov)
        self.init_position = np.array(position, dtype=np.float32)
        self.init_center = np.array(center, dtype=np.float32)
        self.init_up = _normalize(up, fallback=[0.0, 1.0, 0.0])
        # Fixed vertical reference the turntable yaws about. Kept constant so
        # horizontal drags always spin around the same world axis.
        self._world_up = np.array(self.init_up, dtype=np.float32)

        # ---- persistent / GUI-exposed state ----
        self.create_variable_callback("position", np.array(position, dtype=np.float32), lambda x: self.updateMVPVariables(), True)
        self.create_variable_callback("center", np.array(center, dtype=np.float32), lambda x: self.updateMVPVariables(), True)
        self.create_variable_callback("up", np.array(self.init_up, dtype=np.float32), lambda x: self.updateMVPVariables(), True)
        self.create_variable_callback("fov", float(fov), lambda x: self.updateMVPVariables(), True)
        # Projection mode toggle (combo box in GUI). "orthographic" removes perspective
        # foreshortening entirely, which is what you usually want for 2D flow fields.
        self.create_variable_callback("projectionMode", ["perspective", "orthographic"], lambda x: self.updateMVPVariables(), True, True)

        self.width = width
        self.height = height
        self.aspect_ratio = width / height
        self.last_mouse_pos = None
        self.mouse_down = False

        # ---- axis-aligned view presets ----
        # Semantics: put the eye ON the given axis and look back at ``center``
        # (i.e. "z positive" == camera on +Z looking toward -Z == facing the
        # z-axis).  ``up`` is chosen so the horizon stays sensible.
        self.addAction("x positive", lambda o: o.set_axis_view(( 1,  0,  0), (0, 1,  0)))
        self.addAction("x negative", lambda o: o.set_axis_view((-1,  0,  0), (0, 1,  0)))
        self.addAction("y positive", lambda o: o.set_axis_view(( 0,  1,  0), (0, 0, -1)))
        self.addAction("y negative", lambda o: o.set_axis_view(( 0, -1,  0), (0, 0,  1)))
        self.addAction("z positive", lambda o: o.set_axis_view(( 0,  0,  1), (0, 1,  0)))
        self.addAction("z negative", lambda o: o.set_axis_view(( 0,  0, -1), (0, 1,  0)))
        self.addAction("reset position", lambda o: o.resetCamera())

        self.MVPVariables = {"viewMat": None, "projMat": None}
        self.updateMVPVariables()

    def OnRestoreState(self):
        self.updateMVPVariables()

    def resetCamera(self):
        self.setValue("fov", float(self.init_fov), callback=False)
        self.setValue("position", np.array(self.init_position, dtype=np.float32), callback=False)
        self.setValue("center", np.array(self.init_center, dtype=np.float32), callback=False)
        self.setValue("up", np.array(self.init_up, dtype=np.float32), callback=False)
        self.updateMVPVariables()

    def updateMVPVariables(self):
        self.MVPVariables["viewMat"] = self.get_view_matrix()
        self.MVPVariables["projMat"] = self.get_projection_matrix()

    def getScope(self):
        """override the getScope method to return view and projection matrix"""
        return self.MVPVariables

    def get_view_matrix(self):
        """Get the view matrix (looks from ``position`` at ``center``)."""
        eye = np.array(self.getValue("position"), dtype=np.float32)
        center = np.array(self.getValue("center"), dtype=np.float32)
        up = np.array(self.getValue("up"), dtype=np.float32)

        forward = _normalize(center - eye, fallback=[0.0, 0.0, -1.0])
        # Guard against a degenerate up (parallel to the view direction).
        if float(np.linalg.norm(np.cross(forward, _normalize(up)))) < 1e-6:
            up = self._world_up
            if float(np.linalg.norm(np.cross(forward, up))) < 1e-6:
                up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        return glm.lookAt(glm.vec3(*eye), glm.vec3(*center), glm.vec3(*up))

    def get_projection_matrix(self):
        fov = self.getValue("fov")
        # Near/far scale with how far the eye is from the pivot it orbits.
        eye = np.array(self.getValue("position"), dtype=np.float32)
        center = np.array(self.getValue("center"), dtype=np.float32)
        distance = float(np.linalg.norm(eye - center))
        near_plane = max(0.5, distance * 0.1)
        far_plane = max(100.0, distance * 10)

        if self.getOptionValue("projectionMode") == "orthographic":
            # Match the on-screen scale of the perspective view at the pivot
            # distance so toggling does not jump the framing.
            # half_height = distance * tan(fov/2); ortho has no foreshortening.
            half_height = max(distance * np.tan(np.radians(fov) * 0.5), 1e-3)
            half_width = half_height * self.aspect_ratio
            return glm.ortho(-half_width, half_width, -half_height, half_height, near_plane, far_plane)

        return glm.perspective(glm.radians(fov), self.aspect_ratio, near_plane, far_plane)

    def update_window_size(self, width, height):
        """Update the window size and recalculate the projection matrix."""
        self.width = width
        self.height = height
        self.aspect_ratio = width / height
        self.updateMVPVariables()

    def set_axis_view(self, axis, up):
        """Place the eye on ``axis`` (through the pivot) looking back at
        ``center``, keeping the current orbit distance. Gives an exact,
        repeatable front/top/side view."""
        center = np.array(self.getValue("center"), dtype=np.float32)
        eye = np.array(self.getValue("position"), dtype=np.float32)
        distance = float(np.linalg.norm(eye - center))
        if distance < 1e-6:
            distance = float(np.linalg.norm(self.init_position - self.init_center)) or 1.0

        axis = _normalize(axis, fallback=[0.0, 0.0, 1.0])
        new_eye = center + axis * distance
        self.setValue("position", np.array(new_eye, dtype=np.float32), callback=False)
        self.setValue("up", _normalize(up, fallback=[0.0, 1.0, 0.0]), callback=False)
        self.updateMVPVariables()

    def orbit(self, dx, dy):
        """Orbit the eye around ``center`` by a mouse delta (pixels).

        Turntable convention (no quaternions, no roll):
          * horizontal drag -> yaw about the world-up axis
          * vertical drag   -> pitch about the camera-right axis, clamped away
            from the poles so the view never flips
        A fresh, level ``up`` is rebuilt every step, so the horizon stays flat
        and returning near a principal axis is easy.
        """
        center = np.array(self.getValue("center"), dtype=np.float32)
        eye = np.array(self.getValue("position"), dtype=np.float32)
        up_stored = np.array(self.getValue("up"), dtype=np.float32)
        world_up = self._world_up

        offset = eye - center
        dist = float(np.linalg.norm(offset))
        if dist < 1e-8:
            return

        # Rotation per pixel: a full-window-height drag ~= 180 degrees.
        k = np.pi / float(max(self.height, 1))
        yaw = dx * k
        pitch = dy * k

        # 1) yaw about the world-up axis
        offset = _rotation_about_axis(world_up, yaw) @ offset

        # 2) pitch about the camera-right axis, clamped in polar angle so the
        #    eye never crosses (or sits exactly on) a pole.
        forward = _normalize(-offset)
        right = np.cross(forward, world_up)
        if float(np.linalg.norm(right)) < 1e-6:      # at a pole: use last up
            right = np.cross(forward, up_stored)
        right = _normalize(right, fallback=[1.0, 0.0, 0.0])

        cos_polar = float(np.clip(np.dot(offset, world_up) / dist, -1.0, 1.0))
        polar = float(np.arccos(cos_polar))          # 0 at +world_up, pi at -world_up
        POLE = np.radians(2.0)
        new_polar = float(np.clip(polar + pitch, POLE, np.pi - POLE))
        offset = _rotation_about_axis(right, new_polar - polar) @ offset

        # 3) rebuild a level up (kills any accumulated roll / drift)
        forward = _normalize(-offset)
        right = np.cross(forward, world_up)
        if float(np.linalg.norm(right)) < 1e-6:
            up = up_stored
        else:
            right = _normalize(right)
            up = _normalize(np.cross(right, forward))

        self.setValue("position", np.array(center + offset, dtype=np.float32), callback=False)
        self.setValue("up", np.array(up, dtype=np.float32), callback=False)
        self.updateMVPVariables()

    def handle_mouse_move(self, x, y, up=False):
        """Orbit the camera from relative mouse motion."""
        if up is True:
            self.last_mouse_pos = None
            return
        if self.last_mouse_pos is None:
            self.last_mouse_pos = (x, y)
            return

        lx, ly = self.last_mouse_pos
        dx = x - lx
        dy = y - ly
        self.last_mouse_pos = (x, y)
        if dx == 0 and dy == 0:
            return
        self.orbit(dx, dy)

    def pan(self, dx: float, dy: float, dz: float):
        """Translate the whole rig (eye and pivot together) along the current
        screen right / up / forward axes, so panning slides the view without
        rotating it."""
        center = np.array(self.getValue("center"), dtype=np.float32)
        eye = np.array(self.getValue("position"), dtype=np.float32)
        up = np.array(self.getValue("up"), dtype=np.float32)

        forward = _normalize(center - eye, fallback=[0.0, 0.0, -1.0])
        right = _normalize(np.cross(forward, up), fallback=[1.0, 0.0, 0.0])
        up_vec = _normalize(np.cross(right, forward), fallback=[0.0, 1.0, 0.0])

        move = right * dx + up_vec * dy + forward * dz
        self.setValue("position", np.array(eye + move, dtype=np.float32), callback=False)
        self.setValue("center", np.array(center + move, dtype=np.float32), callback=False)
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
