from .Object import *
import numpy as np
import glm
import pygame
from OpenGL.GL import *
from OpenGL.GLU import gluUnProject
from .VisualizationEngine import getEngine

# def screen_to_world(x, y, width, height, modelview, projection, viewport):    
#     y = height - y  # OpenGL's y axis  is reversed of pygame's y axis
#     z = gl.glReadPixels(x, y, 1, 1, gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT)
#     return gluUnProject(x, y, z, modelview, projection, viewport)

def screen_to_world_ray(x, y, modelview, projection)->tuple[np.ndarray[np.float32,3],np.ndarray[np.float32,3]]:
    # 获取OpenGL当前矩阵和视口
    # modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
    # projection = glGetDoublev(GL_PROJECTION_MATRIX)
    modelview = np.array(modelview.to_list(), dtype=np.float64)
    projection = np.array(projection.to_list(), dtype=np.float64)
    viewport = glGetIntegerv(GL_VIEWPORT)
    y_opengl = viewport[3] - y - 1
    near = gluUnProject(x, y_opengl, 0.0, modelview, projection, viewport)
    far = gluUnProject(x, y_opengl, 1.0, modelview, projection, viewport)
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

        # Seeding groups: each is a list of (pos, time)
        self.create_variable("keepSeeding", False, True)
        self.create_variable("activeSeedingGroup", 0, True)
        self.create_variable_callback("SeedingGroup0", list([]), notifySeedingChanged,False)
        self.create_variable_callback("SeedingGroup1", list([]), notifySeedingChanged,False)
        self.last_indicator_pos = None
        self.last_indicator_time = None

        

  

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
        # elif event.type == pygame.MOUSEMOTION:
        #     if event.buttons[0] == 1:  # Left mouse button
        #         self.SetIndicator(event.pos, time, group=0, keep_seeding=True)

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