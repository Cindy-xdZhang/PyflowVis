"""Iso-surface rendering of a scalar derived from the active 3D vector field (Phase 3).

Self-contained: the object derives its own scalar volume (Q / lambda2 / vorticity /
magnitude / IVD) from the active 3D velocity field, runs marching cubes at an iso level,
and renders a Lambert-shaded triangle mesh. Two dirty levels keep it interactive:
  - scalar_dirty: recompute the (Z,Y,X) volume (on operation / time change) — heavier;
  - mesh_dirty:   re-run marching cubes at the current iso level — cheap, for slider drags.
Renders nothing unless a 3D vector field is the active field.
"""
from .VertexArrayObject import *
from OpenGL import GL as gl
import ctypes
import numpy as np
import logging

from FLowUtils.ScalarField3d import compute_scalar_slice_3D, marching_cubes_world, velocity_slice


class IsoSurfaceObject(VertexArrayObject):
    def __init__(self, name="IsoSurface"):
        super().__init__(name)
        self.scalar_dirty = True
        self.mesh_dirty = True
        self._last_time = None
        self._last_op = None
        self._volume = None
        self._vmin = 0.0
        self._vmax = 1.0
        self.index_count = 0

        def scalar_cb(obj) -> None:
            obj.scalar_dirty = True
        def mesh_cb(obj) -> None:
            obj.mesh_dirty = True

        self.create_variable_callback("draw", True, mesh_cb, True)
        # Which scalar to iso-surface. Vortex-oriented defaults first.
        self.create_variable_gui("scalarOperation", ["Q_CRITERION", "LAMBDA2", "VORTICITY", "MAGNITUDE", "IVD"], False)
        self.addCallback("scalarOperation", scalar_cb)
        self.create_variable_callback_with_gui_customization(
            "isoFraction", 0.5, mesh_cb, False, {'widget': 'slider_float', 'min': 0.0, 'max': 1.0})
        self.create_variable_gui("color", (0.85, 0.35, 0.25), False, {'widget': 'color_picker'})
        # Read-outs of the current scalar range (so the user can interpret isoFraction).
        self.create_variable("scalarMin", 0.0, False)
        self.create_variable("scalarMax", 1.0, False)

        self.material = Material("isoSurfaceMaterial", "isoSurfaceMat")
        self.setMaterial(self.material)

        # Own VAO: aPos(loc0) + aNormal(loc1) interleaved (stride 24) + element buffer.
        self.iso_vao = gl.glGenVertexArrays(1)
        self.iso_vbo = gl.glGenBuffers(1)
        self.iso_ebo = gl.glGenBuffers(1)
        gl.glBindVertexArray(self.iso_vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.iso_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, 0, None, gl.GL_DYNAMIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 24, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 24, ctypes.c_void_p(12))
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.iso_ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, 0, None, gl.GL_DYNAMIC_DRAW)
        gl.glBindVertexArray(0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)

    def postInit(self):
        super().postInit()
        self.actFieldObject = self.parentScene.getObject("ActiveField") if self.parentScene.hasObject("ActiveField") else None

    @staticmethod
    def _nearest_time_index(field, t):
        ti_f = field.timeInterval
        if ti_f and ti_f != 0:
            return int(np.clip(round((t - field.tmin) / ti_f), 0, field.time_steps - 1))
        return 0

    def _recompute_scalar(self, field, t, op):
        self.scalar_dirty = False
        self._last_time = t
        self._last_op = op
        ti = self._nearest_time_index(field, t)
        try:
            data = velocity_slice(field, ti)  # (Z,Y,X,3)
            if data is None:
                self._volume = None
                self.index_count = 0
                return
            vol = compute_scalar_slice_3D(op, data, field.gridInterval)
        except Exception as e:
            logging.getLogger().error(f"[IsoSurface] scalar computation failed: {e}")
            self._volume = None
            self.index_count = 0
            return
        self._volume = vol
        self._vmin = float(np.min(vol))
        self._vmax = float(np.max(vol))
        self.setValue("scalarMin", self._vmin, callback=False)
        self.setValue("scalarMax", self._vmax, callback=False)
        self.mesh_dirty = True

    def _recompute_mesh(self, field):
        self.mesh_dirty = False
        if self._volume is None or self._vmax <= self._vmin:
            self.index_count = 0
            return
        frac = float(self.getValue("isoFraction"))
        level = self._vmin + frac * (self._vmax - self._vmin)
        eps = 1e-6 * (self._vmax - self._vmin) + 1e-12
        level = min(max(level, self._vmin + eps), self._vmax - eps)

        result = marching_cubes_world(self._volume, level, field.gridInterval, field.domainMinBoundary)
        if result is None:
            self.index_count = 0
            return
        verts, normals, faces = result
        interleaved = np.empty((verts.shape[0], 6), dtype=np.float32)
        interleaved[:, 0:3] = verts
        interleaved[:, 3:6] = normals
        indices = np.ascontiguousarray(faces.reshape(-1), dtype=np.uint32)

        gl.glBindVertexArray(self.iso_vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.iso_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, interleaved.nbytes, interleaved, gl.GL_DYNAMIC_DRAW)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.iso_ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_DYNAMIC_DRAW)
        gl.glBindVertexArray(0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
        self.index_count = int(indices.size)

    def render(self):
        if self.actFieldObject is None:
            return
        field = self.actFieldObject.getActiveField()
        if field is None or field.getDim() != 3:
            self.index_count = 0  # iso-surface is a 3D-only feature
            return
        t = self.actFieldObject.time()
        op = self.getOptionValue("scalarOperation")
        if t != self._last_time or op != self._last_op:
            self.scalar_dirty = True
        if self.scalar_dirty:
            self._recompute_scalar(field, t, op)
        if self.mesh_dirty:
            self._recompute_mesh(field)
        if self.index_count == 0:
            return
        if self.material is not None:
            self.material.apply([self.parentScene, self.cameraObject, self])
        gl.glBindVertexArray(self.iso_vao)
        gl.glDrawElements(gl.GL_TRIANGLES, self.index_count, gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)
        gl.glUseProgram(0)
