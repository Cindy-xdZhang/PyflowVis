from .Object import *
import numpy as np
import glm
import pygame
from OpenGL.GL import *
from OpenGL.GLU import gluUnProject
from .VisualizationEngine import getEngine

def manual_gluUnProject(winX, winY, winZ, modelview, projection, viewport):
    """
    Manual implementation of gluUnProject function to convert screen coordinates to world coordinates,
    used to solve the inconsistency between glm matrices and OpenGL matrices
    """
    # Convert screen coordinates to normalized device coordinates (NDC)
    # OpenGL's Y axis is opposite to screen coordinates, need to flip
    x = (winX - viewport[0]) / viewport[2] * 2.0 - 1.0
    y = (winY - viewport[1]) / viewport[3] * 2.0 - 1.0
    z = 2.0 * winZ - 1.0
    # Create homogeneous coordinates
    point = glm.vec4(x, y, z, 1.0)
    
    # Calculate the inverse of MVP matrix
    # Note: OpenGL uses column-major order, so matrix multiplication order is projection * modelview
    mvp = projection * modelview
    mvp_inv = glm.inverse(mvp)
    
    # Apply inverse transformation
    world_point = mvp_inv * point
    # Return world coordinates (first three components), perform perspective division
    return np.array([world_point.x / world_point.w, 
                    world_point.y / world_point.w, 
                    world_point.z / world_point.w])

def screen_to_world_ray(x, y, modelview, projection)->tuple[np.ndarray[np.float32,3],np.ndarray[np.float32,3]]:
    # Use glm matrices directly, don't convert to NumPy
    viewport = glGetIntegerv(GL_VIEWPORT)
    y_opengl = viewport[3] - y - 1
    
    # Use manually implemented gluUnProject
    # 0.0 = near clipping plane (in front of camera)
    # 1.0 = far clipping plane (behind camera)
    # These two points define the ray starting from the camera
    near = manual_gluUnProject(x, y_opengl, 0.0, modelview, projection, viewport)
    far = manual_gluUnProject(x, y_opengl, 1.0, modelview, projection, viewport)
    
    ray_origin = np.array(near)
    ray_dir = np.array(far) - np.array(near)
    ray_dir = ray_dir / np.linalg.norm(ray_dir)
    return ray_origin.astype(np.float32), ray_dir.astype(np.float32)


@singleton
class Indicator(Object):
    """
    Indicator object for handling 3D picking and seeding in the visualization.
    Supports multiple seeding groups and robust state management.
    """
    def __init__(self):
        super().__init__("Indicator")
        

        def notifySeedingChanged(obj):
            getEngine().eventRegister.notifyEvent("seeding_changed")

        self.addAction("clear seeding", lambda x: self.clearSeeding())
        self.addAction("dense reseeding", lambda x: self.denseReseeding())
        self.addAction("random reseeding", lambda x: self.randomReseeding())

        # Seeding groups: each is a list of (pos, time)
        self.create_variable("SeedingCountPerAxis", 10, True)

        self.create_variable("keepSeeding", False, True)
        self.create_variable("activeSeedingGroup", 0, True)
        self.create_variable_callback("SeedingGroup0", list([]), notifySeedingChanged,False)
        self.create_variable_callback("SeedingGroup1", list([]), notifySeedingChanged,False)
        self.last_indicator_pos = None
        self.last_indicator_time = None

        
    def _reseeding_bounds(self, field, margin):
        """Domain bounds shrunk inwards by a fractional `margin`; returns (lo3, hi3, is3d).
        For a 2D field the z extent is degenerate, so callers pin z=0 there."""
        dmin = np.asarray(field.domainMinBoundary, dtype=np.float32)
        dmax = np.asarray(field.domainMaxBoundary, dtype=np.float32)
        span = dmax - dmin
        return dmin + margin * span, dmax - margin * span, field.getDim() == 3

    def randomReseeding(self):
        activeFieldWidget = self.getParentScene().getObject("ActiveField")
        if activeFieldWidget is None:
            return
        field = activeFieldWidget.getActiveField()
        if field is None:
            return
        n = int(self.getValue("SeedingCountPerAxis"))
        lo, hi, is3d = self._reseeding_bounds(field, 0.01)
        xs = np.random.uniform(lo[0], hi[0], n)
        ys = np.random.uniform(lo[1], hi[1], n)
        # A 2D field lives on z=0; only a 3D field gets seeds spread through the z extent.
        zs = np.random.uniform(lo[2], hi[2], n) if is3d else np.zeros(n)
        seeds = [np.array([x, y, z], dtype=np.float32) for x, y, z in zip(xs, ys, zs)]

        time = self.getParentScene().getTime()
        keep = self.getValue("keepSeeding")
        # Populate BOTH groups: pathline reads SeedingGroup0, streamline reads SeedingGroup1.
        # Filling only the active group left streamlines (group 1) empty by default -> nothing drawn.
        self.SetIndicators(seeds, time, 0, keep_seeding=keep)
        self.SetIndicators(seeds, time, 1, keep_seeding=keep)

  

    def eventCallBacks(self,event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 3:  # Right mouse button
                seeding_plane=self.getParentScene().getObject("plane")
                hit,hit_pos=self.handleMouseRayIntersection(event.pos, seeding_plane)
                if hit:
                    time=self.getParentScene().getTime()
                    groupIdtoOperate=self.getValue("activeSeedingGroup")
                    keepSeeding=self.getValue("keepSeeding")
                    self.SetIndicator(hit_pos, time, group=groupIdtoOperate, keep_seeding=keepSeeding)


    def SetIndicators(self, pos3D, time, group: int = 0, keep_seeding: bool = False):
        """
        pos3D: list of pos3d(3)
        """
        if group == 0:
            SeedingGroup0=self.getValue("SeedingGroup0")
            if keep_seeding:
                for pos in pos3D:
                    SeedingGroup0.append((pos, time))
            else:
                SeedingGroup0 = [(pos, time) for pos in pos3D]
            self.setValue("SeedingGroup0", SeedingGroup0)
        elif group == 1:
            SeedingGroup1=self.getValue("SeedingGroup1")
            if keep_seeding:
                for pos in pos3D:
                    SeedingGroup1.append((pos, time))
            else:
                SeedingGroup1 = [(pos, time) for pos in pos3D]
            self.setValue("SeedingGroup1", SeedingGroup1)
        else:
            raise ValueError("group must be 0 or 1")
        getEngine().eventRegister.notifyEvent("seeding_changed")

    def SetIndicator(self, pos3D, time, group: int = 0, keep_seeding: bool = False):
        """
        Set the indicator position and time, and update the seeding group.
        :param pos3D: 3D position (iterable of 3 floats)
        :param time: float, current time
        :param group: int, 0 or 1, which seeding group to update
        :param keep_seeding: if True, append; else, replace
        """
        pos3D = np.array(pos3D, dtype=np.float32)
        self.last_indicator_pos = pos3D
        self.last_indicator_time = time

        if group == 0:
            SeedingGroup0=self.getValue("SeedingGroup0")
            if keep_seeding:
                SeedingGroup0.append((pos3D, time))
            else:
                SeedingGroup0 = [(pos3D, time)]
            self.setValue("SeedingGroup0", SeedingGroup0)
        elif group == 1:
            SeedingGroup1=self.getValue("SeedingGroup1")
            if keep_seeding:
                SeedingGroup1.append((pos3D, time))
            else:
                SeedingGroup1 = [(pos3D, time)]
            self.setValue("SeedingGroup1", SeedingGroup1)
        else:
            raise ValueError("group must be 0 or 1")
        
        getEngine().eventRegister.notifyEvent("seeding_changed")




    def GetLastIndicator(self):
        """
        Get the current indicator position and time.
        :return: (np.ndarray, float) or (None, None)
        """
        return self.last_indicator_pos, self.last_indicator_time

    def handleMouseRayIntersection(self, p, targetObject):
        # p: (x, y) 屏幕坐标
        width, height = self.cameraObject.width, self.cameraObject.height

        ray_origin, ray_dir = screen_to_world_ray(p[0], p[1], self.cameraObject.get_view_matrix(), self.cameraObject.get_projection_matrix())


        hit,hit_pos=targetObject.intersect_ray(ray_origin, ray_dir)
        return hit,hit_pos

    



    def clearSeeding(self):
        """
        Clear all seeding groups and indicator state.
        """
        
        self.setValue("SeedingGroup0", [])
        self.setValue("SeedingGroup1", [])
        self.last_indicator_pos = None
        self.last_indicator_time = None
        getEngine().eventRegister.notifyEvent("seeding_changed")

    def denseReseeding(self):
        activeFieldWidget = self.getParentScene().getObject("ActiveField")
        if activeFieldWidget is None:
            return
        field = activeFieldWidget.getActiveField()
        if field is None:
            return
        n = max(2, int(self.getValue("SeedingCountPerAxis")))
        lo, hi, is3d = self._reseeding_bounds(field, 0.05)
        xs = np.linspace(lo[0], hi[0], n)
        ys = np.linspace(lo[1], hi[1], n)
        if is3d:
            nz = min(n, 6)  # cap the 3rd axis so the count stays ~ n*n*6, not n**3
            zs = np.linspace(lo[2], hi[2], nz)
            gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')
            seeds = [np.array([x, y, z], dtype=np.float32)
                     for x, y, z in zip(gx.ravel(), gy.ravel(), gz.ravel())]
        else:
            gx, gy = np.meshgrid(xs, ys, indexing='ij')
            seeds = [np.array([x, y, 0.0], dtype=np.float32)
                     for x, y in zip(gx.ravel(), gy.ravel())]
        time = self.getParentScene().getTime()
        self.SetIndicators(seeds, time, 0, keep_seeding=False)
        self.SetIndicators(seeds, time, 1, keep_seeding=False)