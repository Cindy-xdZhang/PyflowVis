"""Direct volume rendering (DVR) of a scalar derived from the active 3D vector field (Phase 4).

Uploads the scalar volume as a GL_R32F 3D texture and ray-marches it in the fragment
shader over a proxy cube, using the engine's built-in 1D colormap array as the transfer
function. Self-contained (derives its own scalar); draws nothing unless a 3D field is active.
"""
from .VertexArrayObject import *
from .shaderManager import getTextureManager
from OpenGL import GL as gl
import numpy as np
import logging

from FLowUtils.ScalarField3d import compute_scalar_slice_3D, velocity_slice


# Unit cube [0,1]^3, 12 triangles wound CCW outward (so GL_FRONT culling keeps back faces).
_CUBE_VERTS = [0, 0, 0,  1, 0, 0,  1, 1, 0,  0, 1, 0,
               0, 0, 1,  1, 0, 1,  1, 1, 1,  0, 1, 1]
_CUBE_TEX = [0.0, 0.0] * 8
_CUBE_INDICES = [0, 2, 1, 0, 3, 2,   # -z
                 4, 5, 6, 4, 6, 7,   # +z
                 0, 1, 5, 0, 5, 4,   # -y
                 3, 6, 2, 3, 7, 6,   # +y
                 0, 4, 7, 0, 7, 3,   # -x
                 1, 2, 6, 1, 6, 5]   # +x


class VolumeRenderObject(VertexArrayObject):
    def __init__(self, name="VolumeRender"):
        super().__init__(name)
        self.scalar_dirty = True
        self._last_time = None
        self._last_op = None
        self._has_volume = False

        def scalar_cb(obj) -> None:
            obj.scalar_dirty = True

        # DVR composites as a translucent overlay (depth-test off), so default it OFF —
        # the user enables it from the Objects menu when they want volume rendering.
        self.create_variable("draw", False, True)
        self.create_variable_gui("scalarOperation", ["MAGNITUDE", "VORTICITY", "Q_CRITERION", "LAMBDA2", "IVD"], False)
        self.addCallback("scalarOperation", scalar_cb)
        colormap_names = list(getTextureManager().getBuiltInTextureNames())
        if not colormap_names:
            colormap_names = ["default"]
        self.create_variable_gui("colorMap", colormap_names, False)
        self.create_variable_gui("densityScale", 4.0, False, {'widget': 'slider_float', 'min': 0.0, 'max': 200.0})
        self.create_variable_gui("numSteps", 128, False, {'widget': 'input'})
        # Uniforms filled in on each scalar recompute.
        self.create_variable("volMin", [-1.0, -1.0, -1.0], False, False)
        self.create_variable("volMax", [1.0, 1.0, 1.0], False, False)
        self.create_variable("scalarMin", 0.0, False)
        self.create_variable("scalarMax", 1.0, False)

        # 3D texture holding the scalar volume; id exposed as the sampler3D uniform.
        self.volume_tex = int(gl.glGenTextures(1))
        gl.glBindTexture(gl.GL_TEXTURE_3D, self.volume_tex)
        gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_WRAP_R, gl.GL_CLAMP_TO_EDGE)
        gl.glBindTexture(gl.GL_TEXTURE_3D, 0)
        self.create_variable("volumeTex", self.volume_tex, False, False)

        self.material = Material("volumeRenderMaterial", "dvrMat", texture0="builtIn")
        self.setMaterial(self.material)

        # Proxy cube geometry (uploaded once via the base VAO).
        self.appendVertexGeometry(_CUBE_VERTS, _CUBE_INDICES, _CUBE_TEX)

    def postInit(self):
        super().postInit()
        self.actFieldObject = self.parentScene.getObject("ActiveField") if self.parentScene.hasObject("ActiveField") else None

    @staticmethod
    def _nearest_time_index(field, t):
        ti = field.timeInterval
        if ti and ti != 0:
            return int(np.clip(round((t - field.tmin) / ti), 0, field.time_steps - 1))
        return 0

    def _recompute_volume(self, field, t, op):
        self.scalar_dirty = False
        self._last_time = t
        self._last_op = op
        try:
            data = velocity_slice(field, self._nearest_time_index(field, t))
            if data is None:
                self._has_volume = False
                return
            vol = np.ascontiguousarray(compute_scalar_slice_3D(op, data, field.gridInterval), dtype=np.float32)
        except Exception as e:
            logging.getLogger().error(f"[VolumeRender] scalar computation failed: {e}")
            self._has_volume = False
            return
        Z, Y, X = vol.shape
        gl.glBindTexture(gl.GL_TEXTURE_3D, self.volume_tex)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glTexImage3D(gl.GL_TEXTURE_3D, 0, gl.GL_R32F, X, Y, Z, 0, gl.GL_RED, gl.GL_FLOAT, vol)
        gl.glBindTexture(gl.GL_TEXTURE_3D, 0)
        dmin = field.domainMinBoundary
        dmax = field.domainMaxBoundary
        self.setValue("volMin", [float(dmin[0]), float(dmin[1]), float(dmin[2])], callback=False)
        self.setValue("volMax", [float(dmax[0]), float(dmax[1]), float(dmax[2])], callback=False)
        self.setValue("scalarMin", float(vol.min()), callback=False)
        self.setValue("scalarMax", float(vol.max()), callback=False)
        self._has_volume = True

    def render(self):
        if self.actFieldObject is None:
            return
        field = self.actFieldObject.getActiveField()
        if field is None or field.getDim() != 3:
            self._has_volume = False
            return
        t = self.actFieldObject.time()
        op = self.getOptionValue("scalarOperation")
        if t != self._last_time or op != self._last_op:
            self.scalar_dirty = True
        if self.scalar_dirty:
            self._recompute_volume(field, t, op)
        if not self._has_volume:
            return

        # DVR pass: composite the premultiplied ray-march result over the scene.
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_ONE, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glCullFace(gl.GL_FRONT)
        super().render()
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glDisable(gl.GL_BLEND)
        gl.glEnable(gl.GL_DEPTH_TEST)
        # Unbind the sampler textures apply() bound (3D volume on unit 0, 1D-array colormap on
        # unit 1). Leaving a 3D texture on unit 0 corrupts the fixed-pipeline imgui renderer,
        # which blanks the whole GUI. (MainLoop also does a global reset; this keeps DVR self-contained.)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_3D, 0)
        gl.glBindTexture(gl.GL_TEXTURE_1D_ARRAY, 0)
