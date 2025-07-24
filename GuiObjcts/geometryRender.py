import numpy as np
from OpenGL import GL as gl
from GuiObjcts.VertexArrayObject import VertexArrayObject
from GuiObjcts.shaderManager import Material
from .VisualizationEngine import getEngine, getScene

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
        self.colormapMaterial=Material("colormapMat","colormapMat",texture0="builtIn")

        # Variables for UI
        self.create_variable("activeGeometry",[],False,True)

        self.create_variable_gui("color", self.default_curve_color, True,{'widget': 'color_picker'})
        self.create_variable("colormap", self.default_surface_colormap, True)
        self.create_variable("linewidth", self.default_curve_linewidth, True)
        self.create_variable("segments", 8, True,True)
        self.create_variable("visible", True, True,True)

        self.addAction("add curve",lambda obj:obj.add_curve("curve1",np.array([[-1,0,0], 
                                                                              [0.05,0.00,0], 
                                                                              [0.1,0.00,0.05],
                                                                              [0.15,0.11,0.05],
                                                                              [0.2,0,0]])))

        def updateGeometrycb(obj:GeometryRenderObject) -> None:
            activeGeometry=obj.getOptionValue("activeGeometry")
            if activeGeometry is not None:
                obj.update_curve_properties(activeGeometry)
  

        self.addAction("update geometry",lambda obj:updateGeometrycb(obj))



    def add_curve(self, name, points, segments=8):
        """
        Add a curve as a polyline of cylinders to the cache and commit its geometry to a VAO.
        Args:
            name (str): Unique name for the curve
            points (np.ndarray): Nx3 array of 3D points
            color (list): RGB color
            linewidth (float): Cylinder radius (line width)
            visible (bool): Whether to render this curve
            segments (int): Number of segments for the cylinder
        """
        # Check for name conflicts
        counter = 1
        base_name=name
        while name in self.cached_curves:
            name = f"{base_name}_{counter}"
            counter += 1

        points = np.array(points, dtype=np.float32)
        radius = self.default_curve_linewidth
        vao = VertexArrayObject(name)

    
        # For each consecutive pair of points, create a cylinder segment
        for i in range(len(points) - 1):
            p0 = points[i]
            p1 = points[i+1]
            direction = p1 - p0
            length = np.linalg.norm(direction)
            if length < 1e-6:
                continue
            direction = direction / length
            center = p0 + 0.5 * direction * length
            vao.appendCylinderWithoutCommit(center, direction, radius, length, segments)
        vao.commit()
        vao.create_variable("color", self.default_curve_color, False,False)
        self.cached_curves[name] = {
            "points": points,
            "linewidth": radius,
            "visible": True,
            "vao": vao,
            "material": self.monoColorMaterial,
        }
        optionValue=self.getValue("activeGeometry")
        if optionValue is None:
            optionValue=[name]
        else:
            optionValue.append(name)
        self.updateValue("activeGeometry",optionValue)
        self.updateOptionValue("activeGeometry",name)

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
            # For each consecutive pair of points, create a cylinder segment
            for i in range(len(points) - 1):
                p0 = points[i]
                p1 = points[i+1]
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
            curve["material"].apply([self.parentScene, self.cameraObject, self,vao])
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
