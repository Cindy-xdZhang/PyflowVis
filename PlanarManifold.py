from VertexArrayObject import *
from VisualizationEngine import getEngine
from shaderManager import *

class PlanarManifold(VertexArrayObject):
    def __init__(self, Xdim, Ydim,domainMinBoundary:list=[-2.0,-2.0],domainMaxBoundary:list=[2.0,2.0]):
        super().__init__(f"plane_{Xdim}_{Ydim}")
        self.engine=getEngine()
        self.Xdim= Xdim
        self.Ydim = Ydim
        # Initialize the scalar field parameters with random values, considering the time dimension
        self.Scalarfield = []
        self.domainMinBoundary=domainMinBoundary
        self.domainMaxBoundary=domainMaxBoundary
        self.gridInterval = [(domainMaxBoundary[0]-domainMinBoundary[0])/(Xdim-1),(domainMaxBoundary[1]-domainMinBoundary[1])/(Ydim-1)]
        self.create_plane_mesh()
        colormapMat=Material("colormapMat","colormapMat")
        self.setMaterial(colormapMat)
        self.create_variable("colorMap",self.engine.getTextureNames())
        self.create_variable("attributeBounds",(0.0,1.0))

    

    def create_plane_mesh(self,):
        self.vertices, self.indices, self.textures= createPlane([32,32],[-2,-2,2,2])
        self.appendVertexGeometry(self.vertices,  self.indices,  self.textures)
    

    