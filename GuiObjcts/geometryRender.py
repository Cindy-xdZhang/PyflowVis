import numpy as np
from OpenGL import GL as gl
from GuiObjcts.VertexArrayObject import VertexArrayObject
from GuiObjcts.shaderManager import Material
from .VisualizationEngine import getEngine
from .Object import getScene
from FLowUtils.vtk_coreline import *
from FLowUtils.VectorField2d import UnsteadyVectorField2D
from FLowUtils.VectorField3d import UnsteadyVectorField3D
from .varational_coreline import execute_varational_coreline_extractor

#enum for instance type
InstanceTypes=[
     "curve_2D", #curves that will uplift by time
     "curve_3D", #curves that will not  uplift by time
     "surface" #surface that will not uplift by time
 ]



class GeometryRenderObject(VertexArrayObject):
    """
    Renderable object for displaying geometric primitives (curves and surfaces),
    such as corelines or iso-surfaces extracted from VTK or other sources.
    Supports caching, UI-driven selection, and per-geometry rendering properties.
    Now supports rendering multiple curves and surfaces at once.
    Geometry is committed to VAO at add time, not per-frame.
    For curves, a polyline of cylinders is used, and geometry is rebuilt if line width changes.
    """
    def __init__(self):
        super().__init__("geometryRender")
        self.engine = getEngine()
        self.parentScene=getScene()
        assert self.parentScene is not None, "scene is not set"

        # Caches for curves and surfaces
        # Each value is a dict with geometry, render params, 'visible' flag, and VAO
        self.cached_curves = {}   # key: name, value: dict
        self.cached_surfaces = {} # key: name, value: dict
        # UI state
        self.selected_geometry = None  # ("curve" or "surface", name)

        # Default rendering properties
        self.default_curve_color = [1.0, 0.0, 0.0]
        self.default_curve_linewidth = 0.20
        self.default_surface_colormap = "roma"

        self.monoColorMaterial=Material("monoColorMat","monoColor")
        self.corelineMaterial=Material("corelineMat","corelineMat")

        self.colormapMaterial=Material("colormapMat","colormapMat",texture0="builtIn")

        # Variables for UI
        self.create_variable("activeGeometry",[],False,True)

        self.create_variable_gui("color", self.default_curve_color, True,{'widget': 'color_picker'})
        self.create_variable("colormap", self.default_surface_colormap, True)
        self.create_variable("linewidth",0.05,True)
        self.create_variable("segments", 8, True,True)
        self.create_variable("visible", True, True,True)
        self.create_variable("coreline_method", ["ivd","jacobian","q_criterion","simple"], True)
        self.create_variable("coreline_threshold", 0.1, True)
        self.create_variable("coreline_length_filter", 10, True)

        self.addAction("add curve",lambda obj:obj.add_curve("test",np.array([[-1,0,0], 
                                                                              [0.05,0.00,0], 
                                                                              [0.1,0.00,0.05],
                                                                              [0.15,0.11,0.05],
                                                                              [0.2,0,0]])))
        self.addAction("delete geometry",lambda obj:obj.deleteGeometry(obj.getOptionValue("activeGeometry")))


        self.create_variable("weight_A", 1.0, True,True)
        self.create_variable("weight_D", 1.0, True,True)
        self.create_variable("weight_R", 1.0, True,True)
        self.create_variable("max_RK4_iter", 10, True)
        self.create_variable("ds", 0.05, True,True)

        def extract_coreline2d_vtk(obj:GeometryRenderObject) -> None:
            """extract coreline from active field"""
            fieldname=self.actFieldObject.getActiveFieldName()
            time=self.actFieldObject.time()
            vector_field:UnsteadyVectorField2D = self.actFieldObject.getActiveField()  
            if vector_field is None:
                return

            method=obj.getOptionValue("coreline_method")
            threshold=obj.getValue("coreline_threshold")
            resultName=f"vortex_coreline__{fieldname}_{method}"
            if resultName in obj.cached_curves:
                del obj.cached_curves[resultName]
            curves = extract_vortex_corelines_2d_unsteady(
                vector_field,
                method,
                threshold
            )
            if curves:
                obj.add_curve(f"vortex_coreline__{fieldname}_{method}", curves)
            else:
                print("No vortex corelines found with current parameters")
        
        def extract_coreline3d_vtk(obj:GeometryRenderObject) -> None:
            """extract coreline from active field"""
            fieldname=self.actFieldObject.getActiveFieldName()
            time=self.actFieldObject.time()
            vector_field:UnsteadyVectorField3D = self.actFieldObject.getActiveField()  
            if vector_field is None:
                return
            lengthFilter=obj.getValue("coreline_length_filter")
            threshold=obj.getValue("coreline_threshold")
            method='vtk'
            resultName=f"vortex_coreline__{fieldname}_{method}"
            if resultName in obj.cached_curves:
                del obj.cached_curves[resultName]
            curves = extract_vortex_corelines_3d_vtk(vector_field, time, lengthFilter, threshold)
            if curves:
                obj.add_curve(f"vortex_coreline3d__{fieldname}_{method}", curves,segments=self.getValue("segments"),instanceType="curve_3D")
            else:
                print("No vortex corelines found with current parameters")

        def extract_coreline3d_variantional(obj:GeometryRenderObject) -> None:
            import torch
            """extract coreline from active field"""
            fieldname=self.actFieldObject.getActiveFieldName()
            time=self.actFieldObject.time()
            vector_field:UnsteadyVectorField3D = self.actFieldObject.getActiveField()  
            if vector_field is None:
                return
            method='variantional'
            resultName=f"vortex_coreline__{fieldname}_{method}"
            if resultName in obj.cached_curves.keys():
                del obj.cached_curves[resultName]
            weight_A=obj.getValue("weight_A")
            weight_D=obj.getValue("weight_D")
            weight_R=obj.getValue("weight_R")
            seeds_data = self.indicatorObject.getValue("SeedingGroup1")
            #initialize q as seeds[0], qdot as vector value at seeds[0]
            q0,time=seeds_data[0][0],seeds_data[0][1]
            q_dot0=vector_field.get_vector(q0[0],q0[1],q0[2],time)
            q_dot0=q_dot0/np.linalg.norm(q_dot0)
            q_dot0=q_dot0.reshape(1,-1)
            q0=q0.reshape(1,-1)
            q_dot0=torch.from_numpy(q_dot0)

            timeWindow=0.0
            config = {
                'weight_A': weight_A,
                'weight_D': weight_D,
                'weight_R': weight_R,
                't0': time,
                't_T': time + timeWindow,
                'time_steps': 10, # Reduced for performance during testing
                'max_RK4_iter': obj.getValue("max_RK4_iter"),
                'ds': obj.getValue("ds"),
            }
            curves = execute_varational_coreline_extractor(vector_field, q0,q_dot0,config)
            curves=np.array(curves)
            if curves.shape[0]>0:
                obj.add_curve(resultName, curves,segments=self.getValue("segments"),instanceType="curve_3D")    

        self.addAction("extract coreline",lambda obj:extract_coreline2d_vtk(obj))
        self.addAction("extract coreline3d",lambda obj:extract_coreline3d_vtk(obj))
        self.addAction("extract coreline3d_variantional",lambda obj:extract_coreline3d_variantional(obj))


        def updateGeometrycb(obj:GeometryRenderObject) -> None:
            activeGeometry=obj.getOptionValue("activeGeometry")
            if activeGeometry is not None:
                obj.update_curve_properties(activeGeometry)
        self.addAction("update geometry",lambda obj:updateGeometrycb(obj))

    def deleteGeometry(self,name):
        if name in self.cached_curves:
            del self.cached_curves[name]
        if name in self.cached_surfaces:
            del self.cached_surfaces[name]
        optionValue = self.getValue("activeGeometry")
        if optionValue is not None and name in optionValue:
            optionValue.remove(name)
            self.updateValue("activeGeometry",optionValue)
            if len(optionValue)>0:
                self.updateOptionValue("activeGeometry",optionValue[0])
    

    def postInit(self):
        super().postInit()
        self.actFieldObject=self.parentScene.getObject("ActiveField") if self.parentScene.hasObject("ActiveField") else None
        self.flowlineObject=self.parentScene.getObject("flowline") if self.parentScene.hasObject("flowline") else None
        self.indicatorObject=self.parentScene.getObject("indicator") if self.parentScene.hasObject("indicator") else None

    
    def add_curve(self, name, points, segments=8,instanceType:str="curve_2D"):
        """
        Add one or more curves as polylines of cylinders to the cache and commit geometry to VAOs.
        Args:
            name (str): Base name for the curves
            points (list or np.ndarray): List of Nx3 point arrays, or single Nx3 array
            segments (int): Number of segments for the cylinders
        """
        # Convert single curve points to list format
        if isinstance(points, np.ndarray) and len(points.shape) == 2:
            points = [points]

        curves_base_name = name 
        counter = 1
        result_name=name
        while result_name in self.cached_curves:
            result_name = f"{curves_base_name}_{counter}"
            counter += 1

        vao = VertexArrayObject(result_name)
        vao.create_variable("color", self.getValue("color"), False, False)

        # Convert instanceType from string to int index
        instanceType_int = InstanceTypes.index(instanceType)
        vao.create_variable("instanceType", instanceType_int, False, False)
        
        radius=self.getValue("linewidth")
        for curve_idx, curve_points in enumerate(points):
            curve_points = np.array(curve_points, dtype=np.float32)

            # For each consecutive pair of points, create a cylinder segment
            for i in range(len(curve_points) - 1):
                p0 = curve_points[i]
                p1 = curve_points[i+1]
                direction = p1 - p0
                length = np.linalg.norm(direction)
                if length < 1e-6:
                    continue
                direction = direction / length
                center = p0 + 0.5 * direction * length
                vao.appendCylinderWithoutCommit(center, direction, radius, length, segments)
            
        vao.commit()
        
        self.cached_curves[result_name] = {
            "points": points,
            "linewidth": radius,
            "visible": True,
            "vao": vao,
            "material": self.corelineMaterial,
            "segments": segments,
        }
        # Update active geometry list
        optionValue = self.getValue("activeGeometry")
        if optionValue is None:
            optionValue = [result_name]
        elif result_name not in optionValue:
            optionValue.append(result_name)

        self.updateValue("activeGeometry",optionValue)
        self.updateOptionValue("activeGeometry",result_name)
   

    def update_curve_properties(self, name):
        """
        Update the geometry of a cached curve (polyline of cylinders) when the line width changes.
        """
        if name not in self.cached_curves:
            return
        entry = self.cached_curves[name]
        vao:VertexArrayObject = entry["vao"]
        newLinewidth=self.getValue("linewidth")
        newSegments=self.getValue("segments")
        newColor=self.getValue("color")
        vao.setValue("color", newColor,False)
        entry["visible"]=self.getValue("visible")
        needGeometryRebuild=newLinewidth!=entry["linewidth"] or newSegments!=entry["segments"]
        entry["linewidth"]=newLinewidth
        entry["segments"]=newSegments
        if needGeometryRebuild:
            points =entry["points"]
            radius = newLinewidth
            vao.erase()
            for curve_idx, curve_points in enumerate(points):
                curve_points = np.array(curve_points, dtype=np.float32)
                # For each consecutive pair of points, create a cylinder segment
                for i in range(len(curve_points) - 1):
                    p0 = curve_points[i]
                    p1 = curve_points[i+1]
                    direction = p1 - p0
                    length = np.linalg.norm(direction)
                    if length < 1e-6:
                        continue
                    direction = direction / length
                    center = p0 + 0.5 * direction * length
                    vao.appendCylinderWithoutCommit(center, direction, radius, length, newSegments)
            
            vao.commit()


    # Surface logic can be restored as before if needed
    def render(self):
        """
        Render all visible curves and surfaces using their cached VAOs.
        """
        # Render all visible curves
        for name, curve in self.cached_curves.items():
            if curve.get("visible", True):
                self._render_curve(curve)
        # Render all visible surfaces (uncomment if needed)
        # for name, surface in self.cached_surfaces.items():
        #     if surface.get("visible", True):
        #         self._render_surface(surface)

    def _render_curve(self, curve):
        """
        Render a curve as a polyline of cylinders using its VAO.
        """
        vao = curve["vao"]
        # Set color uniform (assume material uses it)
        if curve["material"] is not None:
            curve["material"].apply([self.parentScene, self.cameraObject, self.flowlineObject, self,vao])
            gl.glBindVertexArray(vao.vao_id)
            gl.glDrawElements(gl.GL_TRIANGLES, len(vao.indices), gl.GL_UNSIGNED_INT, None)
            gl.glBindVertexArray(0)
            gl.glUseProgram(0)

    def _render_surface(self, surface):
        """
        Render a surface as a triangle mesh using its VAO.
        """
        colormap = surface["colormap"]
        opacity = surface["opacity"]
        vao = surface["vao"]
        # Set colormap and opacity uniforms (assume material uses them)
        # if self.material is not None:
        #     self.material.setUniform("colormap", colormap)
        #     self.material.setUniform("opacity", opacity)
        #     self.material.apply([self.parentScene, self.cameraObject, self])
        gl.glBindVertexArray(vao.vao_id)
        gl.glDrawElements(gl.GL_TRIANGLES, len(vao.indices), gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)
