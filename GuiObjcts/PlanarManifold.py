from GuiObjcts.VertexArrayObject import *
from .VisualizationEngine import getEngine
from GuiObjcts.shaderManager import *
import numpy as np

class PlanarManifold(VertexArrayObject):
    def __init__(self, Xdim, Ydim,domainMinBoundary:list=[-2.0,-2.0],domainMaxBoundary:list=[2.0,2.0]):
        super().__init__(f"plane")
        self.engine=getEngine()
        self.Xdim= Xdim
        self.Ydim = Ydim
        # Initialize the scalar field parameters with random values, considering the time dimension
        self.Scalarfield = []
        self.domainMinBoundary=domainMinBoundary
        self.domainMaxBoundary=domainMaxBoundary
        self.gridInterval = [(domainMaxBoundary[0]-domainMinBoundary[0])/(Xdim-1),(domainMaxBoundary[1]-domainMinBoundary[1])/(Ydim-1)]
        self.create_plane_mesh()
        colormapMat=Material("planarManifoldMaterial","colormapMat",texture0="builtIn")
        self.setMaterial(colormapMat)
        self.create_variable("colorMap",self.engine.getBuiltInTextureNames())
        self.create_variable("attributeBounds",(0.0,1.0))
        self.z=0.0

    def create_plane_mesh(self):
        self.vertices, self.indices, self.textures= createPlane([self.Xdim,self.Ydim],self.domainMinBoundary,self.domainMaxBoundary)
        self.appendVertexGeometry(self.vertices,  self.indices,  self.textures)
    
    def intersect_ray(self, ray_origin:np.ndarray[np.float32,3], ray_dir:np.ndarray[np.float32,3]):
        """
        Intersect a ray with the z=self.z plane and check if the intersection is within the domain.
        :param ray_origin: np.ndarray[np.float32,3], ray origin
        :param ray_dir: np.ndarray[np.float32,3], ray direction (should be normalized)
        :return: (hit: bool, hit_pos: tuple or None)
        """
        # Plane: z = self.z
        if abs(ray_dir[2]) < 1e-8:
            return False, None  # Parallel to plane
        t = (self.z - ray_origin[2]) / ray_dir[2]
        if t < 0:
            return False, None  # Intersection behind origin
        hit_pos = ray_origin + ray_dir * t
        x, y, z = float(hit_pos[0]), float(hit_pos[1]), float(hit_pos[2])
        # Check if (x, y) is within domain
        xmin, ymin = self.domainMinBoundary
        xmax, ymax = self.domainMaxBoundary
        if xmin <= x <= xmax and ymin <= y <= ymax:
            return True, (x, y, self.z)
        return False, None

        






