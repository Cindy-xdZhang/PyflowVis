from GuiObjcts.VertexArrayObject import *
from .VisualizationEngine import getEngine
from GuiObjcts.shaderManager import *
import numpy as np
from FLowUtils.VectorField2d import *
import OpenGL.GL as gl
from FLowUtils.ScalarField2d import *

class PlanarManifold(VertexArrayObject):
    def __init__(self,Div:int=8,):
        super().__init__(f"plane")
        self.engine=getEngine()
        self.DimDiv=Div   
        self.colormapMat0=Material("planarManifoldMaterial","planarManifoldMat",texture0="builtIn")
        self.colormapMat1=getShaderManager().getMaterial("colormapMat")
        self.setMaterial(self.colormapMat1)# when there is no scalar field, use the colormapMat0 WILL CAUSE WIRED IMGUI BUGS
        
        self.create_variable("colorMap",self.engine.getBuiltInTextureNames(),True)
        self.create_variable("attributeBounds",[-1.0,1.0])
        self.create_variable("scalarFieldMinTime", 0.0,False,False)
        self.create_variable("scalarFieldMaxTime", 0.0,False,False)
        self.create_variable("scalarAttributeTexture", int(-1),False,False)
        self.z=0.0
        self.Scalarfield=None
        self.actFieldObject=None


        def updatePlaneGeometry(self):
            self.fitPlaneWithActiveField()
          
        
        self.create_variable_callback("axis",["X","Y","Z"],updatePlaneGeometry,True,True)
        self.create_variable_callback_with_gui_customization("NormalizedPosition",0.5,updatePlaneGeometry,True,{"widget":"slider_float","min":0.0,"max":1.0})

        getEngine().eventRegister.registerChannelEvent("activefield_changed", lambda: self.fitPlaneWithActiveField())



    def postInit(self):
        super().postInit()
        self.actFieldObject=self.parentScene.getObject("ActiveField") if self.parentScene.hasObject("ActiveField") else None
        if self.actFieldObject is None:
            logging.getLogger().warning("No ActiveField object found in scene, return")
            return

   
    def fitPlaneWithActiveField(self):
        vectorField = self.actFieldObject.getActiveField()
        if vectorField is None:
            return

        grid_size = [int(vectorField.Xdim/self.DimDiv)+1, int(vectorField.Ydim/self.DimDiv)+1]
        self.point_on_plane = None
        self.normal = None

        if vectorField.getDim() == 2:
            v_min = vectorField.domainMinBoundary
            v_max = vectorField.domainMaxBoundary
            self.point_on_plane = np.array([v_min[0], v_min[1], 0.0], dtype=np.float32)
            self.normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            self.domain_min_box = np.array([v_min[0], v_min[1], 0.0], dtype=np.float32) # Give it some depth for the box
            self.domain_max_box = np.array([v_max[0], v_max[1], 0.0], dtype=np.float32)
            self.erase()
            self.vertices, self.indices, self.textures = createPlaneSimple(
                grid_size,
                self.domain_min_box,
                self.domain_max_box
            )
        elif vectorField.getDim() == 3:
            self.domain_min_box = np.array(vectorField.domainMinBoundary, dtype=np.float32)
            self.domain_max_box = np.array(vectorField.domainMaxBoundary, dtype=np.float32)
            axis = self.getOptionValue("axis")
            
            if axis == "X":
                x_phys = self.domain_min_box[0] + self.getValue("NormalizedPosition") * (self.domain_max_box[0] - self.domain_min_box[0])
                self.point_on_plane = np.array([x_phys, self.domain_min_box[1], self.domain_min_box[2]], dtype=np.float32)
                self.normal = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            elif axis == "Y":
                y_phys = self.domain_min_box[1] + self.getValue("NormalizedPosition") * (self.domain_max_box[1] - self.domain_min_box[1])
                self.point_on_plane = np.array([self.domain_min_box[0], y_phys, self.domain_min_box[2]], dtype=np.float32)
                self.normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            elif axis == "Z":
                z_phys = self.domain_min_box[2] + self.getValue("NormalizedPosition") * (self.domain_max_box[2] - self.domain_min_box[2])
                self.point_on_plane = np.array([self.domain_min_box[0], self.domain_min_box[1], z_phys], dtype=np.float32)
                self.normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            self.erase()
            self.vertices, self.indices, self.textures = createPlane(
                grid_size,
                self.point_on_plane,
                self.normal,
                self.domain_min_box,
                self.domain_max_box
            )
        
        self.appendVertexGeometry(self.vertices,  self.indices,  self.textures)


            


        



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
        self.setMaterial(self.colormapMat0)


    def render(self):
        self.material.apply([self.parentScene, self.cameraObject,self.actFieldObject,self])
        # gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL);
        gl.glBindVertexArray(self.vao_id)  
        gl.glDrawElements(gl.GL_TRIANGLES, len(self.indices), gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)
        gl.glUseProgram(0)
        gl.glActiveTexture(gl.GL_TEXTURE0) 
        gl.glBindTexture(gl.GL_TEXTURE_3D, 0)



    
    def intersect_ray(self, ray_origin:np.ndarray[np.float32,3], ray_dir:np.ndarray[np.float32,3]):
        """
        Intersect a ray with the plane defined by self.normal and self.point_on_plane.
        The intersection point is then checked to be within the domain's bounding box.
        :param ray_origin: np.ndarray[np.float32,3], ray origin
        :param ray_dir: np.ndarray[np.float32,3], ray direction (should be normalized)
        :return: (hit: bool, hit_pos: tuple or None)
        """
        # Plane equation: dot(normal, X - point_on_plane) = 0
        # Ray equation: X = ray_origin + t * ray_dir
        
        # Ensure normal is a numpy array
        normal = np.asarray(self.normal, dtype=np.float32)
        ray_dir = np.asarray(ray_dir, dtype=np.float32)

        denom = np.dot(normal, ray_dir)
        
        # Check if the ray is parallel to the plane
        if abs(denom) < 1e-8:
            return False, None

        p0_l0 = np.asarray(self.point_on_plane, dtype=np.float32) - np.asarray(ray_origin, dtype=np.float32)
        t = np.dot(p0_l0, normal) / denom
        
        # Intersection is behind the ray's origin
        if t < 0:
            return False, None
            
        hit_pos = ray_origin + t * ray_dir
        
        # Check if the hit position is within the domain bounding box
        # Add a small tolerance for floating point comparisons
        epsilon = 1e-5
        min_b, max_b = self.domain_min_box, self.domain_max_box
        if (min_b[0] - epsilon <= hit_pos[0] <= max_b[0] + epsilon and
            min_b[1] - epsilon <= hit_pos[1] <= max_b[1] + epsilon and
            min_b[2] - epsilon <= hit_pos[2] <= max_b[2] + epsilon):
            return True, tuple(hit_pos)
            
        return False, None

        






