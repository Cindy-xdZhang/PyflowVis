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
    def __init__(self, name,camera):
        super().__init__(name)
        self.camera=camera

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

        
    def randomReseeding(self):
        activeFieldWidget=self.getParentScene().getObject("ActiveField")
        if activeFieldWidget is None:
            return
        activeVectorField=activeFieldWidget.getActiveField()
        if activeVectorField is None:
            return
        seeingCountPerAxis=self.getValue("SeedingCountPerAxis")
        xrange = activeVectorField.domainMaxBoundary[0] - activeVectorField.domainMinBoundary[0]
        yrange = activeVectorField.domainMaxBoundary[1] - activeVectorField.domainMinBoundary[1]
        xmin, ymin, _ = activeVectorField.domainMinBoundary+0.01*np.array([xrange, yrange, 0])
        xmax, ymax, _ = activeVectorField.domainMaxBoundary-0.01*np.array([xrange, yrange, 0])
        xrange=xmax-xmin
        yrange=ymax-ymin
        rx=np.random.rand(seeingCountPerAxis)
        ry=np.random.rand(seeingCountPerAxis)
        xs=xmin+rx*xrange
        ys=ymin+ry*yrange
        # 将每个种子点表示为一个np.array3的列表
        randomSeeds = [np.array([x, y, 0.0], dtype=np.float32) for x, y in zip(xs, ys)]

        time=self.getParentScene().getTime()
        groupIdtoOperate=self.getValue("activeSeedingGroup")
        self.SetIndicators(randomSeeds, time, groupIdtoOperate, keep_seeding=self.getValue("keepSeeding"))

  

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
        width, height = self.camera.width, self.camera.height

        ray_origin, ray_dir = screen_to_world_ray(p[0], p[1], self.camera.get_view_matrix(), self.camera.get_projection_matrix())


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
        pass