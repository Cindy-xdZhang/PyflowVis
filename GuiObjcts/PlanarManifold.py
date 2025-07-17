from GuiObjcts.VertexArrayObject import *
from .VisualizationEngine import getEngine
from GuiObjcts.shaderManager import *
import numpy as np
from FLowUtils.VectorField2d import *
import OpenGL.GL as gl
from FLowUtils.ScalarField2d import *

class PlanarManifold(VertexArrayObject):
    def __init__(self, Xdim, Ydim,domainMinBoundary:list=[-2.0,-2.0],domainMaxBoundary:list=[2.0,2.0]):
        super().__init__(f"plane")
        self.engine=getEngine()
        self.Xdim= Xdim
        self.Ydim = Ydim
        self.domainMinBoundary=domainMinBoundary
        self.domainMaxBoundary=domainMaxBoundary
        self.gridInterval = [(domainMaxBoundary[0]-domainMinBoundary[0])/(Xdim-1),(domainMaxBoundary[1]-domainMinBoundary[1])/(Ydim-1)]
        self.create_plane_mesh()
        colormapMat=Material("planarManifoldMaterial","planarManifoldMat",texture0="builtIn")
        self.setMaterial(colormapMat)
        self.create_variable("colorMap",self.engine.getBuiltInTextureNames(),True)
        self.create_variable("attributeBounds",[-1.0,1.0])
        self.create_variable("scalarFieldMinTime", 0.0,False,False)
        self.create_variable("scalarFieldMaxTime", 0.0,False,False)
        self.create_variable("scalarAttributeTexture", int(-1),False,False)
        self.z=0.0
        self.Scalarfield=None
        self.actFieldObject=None

    def postInit(self):
        super().postInit()
        self.actFieldObject=self.parentScene.getObject("ActiveField") if self.parentScene.hasObject("ActiveField") else None
        if self.actFieldObject is None:
            logging.getLogger().warning("No ActiveField object found in scene, return")
            return
    


    def setScalarField(self, Scalarfield: ScalarField2D, bUpdateRange=True):
     
        self.Scalarfield = Scalarfield
        # Clean up previous texture if exists
        texture_id=self.getValue("scalarAttributeTexture")
        if texture_id != -1:
            gl.glDeleteTextures([texture_id])

        # Extract data from ScalarField2D
        if hasattr(Scalarfield, 'getDataAsNumpy'):
            data = Scalarfield.getDataAsNumpy()
        else:
            data = Scalarfield.field.detach().cpu().numpy() if hasattr(Scalarfield.field, 'detach') else Scalarfield.field
        # data shape: (T, Y, X)
        if data is None:
            raise ValueError("ScalarField2D data is None!")
        

        time_steps, Ydim, Xdim = data.shape
        attributeBuffer3D = data.astype(np.float32)
        minAttrib = float(np.min(attributeBuffer3D))
        maxAttrib = float(np.max(attributeBuffer3D))

        # Upload as 3D texture
        texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_3D, texture_id)
        gl.glTexImage3D(
            gl.GL_TEXTURE_3D, 0, gl.GL_R32F,
            Xdim, Ydim, time_steps, 0,
            gl.GL_RED, gl.GL_FLOAT, attributeBuffer3D
        )
        gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_WRAP_R, gl.GL_CLAMP_TO_EDGE)
        gl.glBindTexture(gl.GL_TEXTURE_3D, 0)
        self.updateValue("scalarAttributeTexture", int(texture_id))
        self.updateValue("attributeBounds", (minAttrib, maxAttrib))
        self.updateValue("scalarFieldMinTime", float(Scalarfield.getMinTime()))
        self.updateValue("scalarFieldMaxTime", float(Scalarfield.getMaxTime()))
       

    def render(self):
        self.material.apply([self.parentScene, self.cameraObject,self.actFieldObject,self])
        # gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL);
        gl.glBindVertexArray(self.vao_id)  
        gl.glDrawElements(gl.GL_TRIANGLES, len(self.indices), gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)
        gl.glUseProgram(0)







    def create_plane_mesh(self):
        self.vertices, self.indices, self.textures= createPlane([2,2],self.domainMinBoundary,self.domainMaxBoundary)
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

        






