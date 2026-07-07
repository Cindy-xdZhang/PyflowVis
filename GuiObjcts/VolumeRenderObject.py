"""Direct volume rendering (DVR) of a scalar derived from the active 3D vector field.

Reworked to mirror the optimal-connection C++ engine's ObjectVolumeRendering / DVR.frag:

  * the scalar volume is uploaded as a GL_R16F 3D texture holding RAW values; normalization to
    [0,1] happens in the shader via the ``attributeBounds`` (data min/max) uniform, so animating
    the field only updates a vec2 instead of re-normalizing the whole volume;
  * a coarse per-brick (min,max) GL_RG16F 3D texture (8^3 voxels + 1-voxel halo) drives
    empty-space skipping;
  * the transfer function is a real 1D RGBA texture (color from the selected engine colormap +
    an independently editable opacity curve), sampled once per step as texture(uTransferFunc,scalar);
  * ray marching is object-space, view-independent (voxel-bound step + uMaxSteps cap), jittered,
    with step-size opacity correction and central-difference gradient Blinn-Phong shading;
  * the camera world position is passed as the ``camPos`` uniform (no per-fragment inverse);
  * it composites depth-aware (depth-test on, depth-write off) as premultiplied alpha.

Self-contained (derives its own scalar); draws nothing unless a 3D vector field is active. An
imgui opacity-curve editor (the Python counterpart of the C++ TransferFunctionWidget) is drawn in
the object's GUI window.
"""
from .VertexArrayObject import *
from .shaderManager import getTextureManager
from OpenGL import GL as gl
import numpy as np
import logging
import imgui
import glm

from FLowUtils.ScalarField3d import compute_scalar_slice_3D, velocity_slice


# Unit cube [0,1]^3, 12 triangles wound CCW outward (so GL_FRONT culling keeps the back faces —
# exactly one fragment per pixel, from the far side of the box).
_CUBE_VERTS = [0, 0, 0,  1, 0, 0,  1, 1, 0,  0, 1, 0,
               0, 0, 1,  1, 0, 1,  1, 1, 1,  0, 1, 1]
_CUBE_TEX = [0.0, 0.0] * 8
_CUBE_INDICES = [0, 2, 1, 0, 3, 2,   # -z
                 4, 5, 6, 4, 6, 7,   # +z
                 0, 1, 5, 0, 5, 4,   # -y
                 3, 6, 2, 3, 7, 6,   # +y
                 0, 4, 7, 0, 7, 3,   # -x
                 1, 2, 6, 1, 6, 5]   # +x

_BRICK_SIZE = 8   # voxels per empty-space-skipping brick (matches the C++ buildMinMaxBrickTexture)
_BRICK_HALO = 1   # 1-voxel halo so trilinear sampling at brick edges stays covered
_TF_RES = 256     # transfer-function LUT resolution
_SHADING_LABELS = ["No Shading", "Shading (low)", "Shading (med)", "Shading (high)"]


class VolumeRenderObject(VertexArrayObject):
    def __init__(self, name="VolumeRender"):
        super().__init__(name)
        self.scalar_dirty = True
        self._last_time = None
        self._last_op = None
        self._has_volume = False
        self._lut_dirty = True
        self._last_colormap = None
        self._drag_idx = None
        # Opacity transfer function control points (t in [0,1], alpha in [0,1]), kept sorted by t.
        self.opacity_points = [(0.0, 0.0), (1.0, 1.0)]

        def scalar_cb(obj) -> None:
            obj.scalar_dirty = True

        # DVR composites as a translucent overlay; default it OFF — the user enables it from the
        # Objects menu when they want volume rendering.
        self.create_variable("draw", False, True)
        self.create_variable_gui("scalarOperation", ["MAGNITUDE", "VORTICITY", "Q_CRITERION", "LAMBDA2", "IVD"], False)
        self.addCallback("scalarOperation", scalar_cb)
        colormap_names = list(getTextureManager().getBuiltInTextureNames())
        if not colormap_names:
            colormap_names = ["default"]
        self.create_variable_gui("colorMap", colormap_names, False)

        # Ray-marching parameters (names match the DVR.frag uniforms; option index -> int uniform).
        self.create_variable_gui("uShadingMode", list(_SHADING_LABELS), False)
        # uDensityScale is a real extinction coefficient (1.0 = neutral, matching the C++ default):
        # the transfer-function alpha is the base opacity per reference step; values >1 thicken the
        # volume, <1 make it more translucent.
        self.create_variable_gui("uDensityScale", 1.0, False, {'widget': 'slider_float', 'min': 0.0, 'max': 20.0})
        self.create_variable_gui("uSampleStepFactor", 0.5, False, {'widget': 'slider_float', 'min': 0.1, 'max': 2.0})
        self.create_variable_gui("uMaxSteps", 1024, False, {'widget': 'input'})
        self.create_variable_gui("lightStrength", 1.0, False, {'widget': 'slider_float', 'min': 0.0, 'max': 3.0})
        self.create_variable_gui("enableEmptySkip", True, False)
        # DVR aligns with the C++ engine by compositing depth-aware (test on, write off) so opaque
        # geometry occludes the volume. Turn this off to fall back to a pure always-on-top overlay.
        self.create_variable_gui("depthTest", True, False)
        self.addAction("reset transfer function", lambda obj: obj._reset_opacity())

        # Invisible uniform carriers filled per-frame / per-recompute.
        self.create_variable("uEnableEmptySkip", 1, False, False)
        self.create_variable("camPos", [0.0, 0.0, 0.0], False, False)
        self.create_variable("uVolumeBoundsMin", [-1.0, -1.0, -1.0], False, False)
        self.create_variable("uVolumeBoundsMax", [1.0, 1.0, 1.0], False, False)
        self.create_variable("attributeBounds", [0.0, 1.0], False, False)

        # 3D texture holding the raw scalar volume (R16F); id exposed as the sampler3D uniform.
        self.volume_tex = int(gl.glGenTextures(1))
        gl.glBindTexture(gl.GL_TEXTURE_3D, self.volume_tex)
        gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_WRAP_R, gl.GL_CLAMP_TO_EDGE)
        gl.glBindTexture(gl.GL_TEXTURE_3D, 0)
        self.create_variable("uVolumeTex", self.volume_tex, False, False)

        # Coarse per-brick (min,max) texture (RG16F, NEAREST) for empty-space skipping.
        self.minmax_tex = int(gl.glGenTextures(1))
        gl.glBindTexture(gl.GL_TEXTURE_3D, self.minmax_tex)
        gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_WRAP_R, gl.GL_CLAMP_TO_EDGE)
        gl.glBindTexture(gl.GL_TEXTURE_3D, 0)
        self.create_variable("uMinMaxTex", self.minmax_tex, False, False)

        # Transfer-function LUT (RGBA16F 1D, LINEAR); id exposed as the sampler1D uniform.
        self.tf_tex = int(gl.glGenTextures(1))
        gl.glBindTexture(gl.GL_TEXTURE_1D, self.tf_tex)
        gl.glTexParameteri(gl.GL_TEXTURE_1D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_1D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_1D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glBindTexture(gl.GL_TEXTURE_1D, 0)
        self.create_variable("uTransferFunc", self.tf_tex, False, False)

        # Scene-depth copy (GL_TEXTURE_2D) for depth-aware compositing: each frame the opaque depth
        # buffer is copied here so the ray can be clamped to the nearest opaque surface.
        self.depth_tex = int(gl.glGenTextures(1))
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.depth_tex)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_COMPARE_MODE, gl.GL_NONE)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_DEPTH_COMPONENT24, 1, 1, 0,
                        gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT, None)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        self._depth_w = 1
        self._depth_h = 1
        self.create_variable("uSceneDepth", self.depth_tex, False, False)
        self.create_variable("uViewport", [1.0, 1.0], False, False)
        self.create_variable("uInvViewProj", glm.mat4(1.0), False, False)
        self.create_variable("uUseSceneDepth", 1, False, False)

        self.material = Material("volumeRenderMaterial", "dvrMat")
        self.setMaterial(self.material)

        # Proxy cube geometry (uploaded once via the base VAO).
        self.appendVertexGeometry(_CUBE_VERTS, _CUBE_INDICES, _CUBE_TEX)

    def postInit(self):
        super().postInit()
        self.actFieldObject = self.parentScene.getObject("ActiveField") if self.parentScene.hasObject("ActiveField") else None

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _nearest_time_index(field, t):
        ti = field.timeInterval
        if ti and ti != 0:
            return int(np.clip(round((t - field.tmin) / ti), 0, field.time_steps - 1))
        return 0

    def _reset_opacity(self):
        self.opacity_points = [(0.0, 0.0), (1.0, 1.0)]
        self._lut_dirty = True

    @staticmethod
    def _build_minmax_bricks(vol, B=_BRICK_SIZE, halo=_BRICK_HALO):
        """Downsample the (Z,Y,X) volume into per-brick (min,max) with a `halo`-voxel margin.
        Returns (data (bZ,bY,bX,2) float32, dims (bX,bY,bZ))."""
        Z, Y, X = vol.shape
        bZ = (Z + B - 1) // B
        bY = (Y + B - 1) // B
        bX = (X + B - 1) // B
        out = np.empty((bZ, bY, bX, 2), dtype=np.float32)
        for bz in range(bZ):
            z0 = max(0, bz * B - halo); z1 = min(Z, (bz + 1) * B + halo)
            for by in range(bY):
                y0 = max(0, by * B - halo); y1 = min(Y, (by + 1) * B + halo)
                for bx in range(bX):
                    x0 = max(0, bx * B - halo); x1 = min(X, (bx + 1) * B + halo)
                    block = vol[z0:z1, y0:y1, x0:x1]
                    out[bz, by, bx, 0] = float(block.min())
                    out[bz, by, bx, 1] = float(block.max())
        return out, (bX, bY, bZ)

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

        # Raw scalar volume as R16F (normalization is done in the shader via attributeBounds).
        gl.glBindTexture(gl.GL_TEXTURE_3D, self.volume_tex)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glTexImage3D(gl.GL_TEXTURE_3D, 0, gl.GL_R16F, X, Y, Z, 0, gl.GL_RED, gl.GL_FLOAT, vol)
        gl.glBindTexture(gl.GL_TEXTURE_3D, 0)

        # Per-brick (min,max) for empty-space skipping.
        try:
            bricks, (bX, bY, bZ) = self._build_minmax_bricks(vol)
            gl.glBindTexture(gl.GL_TEXTURE_3D, self.minmax_tex)
            gl.glTexImage3D(gl.GL_TEXTURE_3D, 0, gl.GL_RG16F, bX, bY, bZ, 0, gl.GL_RG, gl.GL_FLOAT,
                            np.ascontiguousarray(bricks, dtype=np.float32))
            gl.glBindTexture(gl.GL_TEXTURE_3D, 0)
        except Exception as e:
            logging.getLogger().warning(f"[VolumeRender] brick build failed (empty-space skip disabled): {e}")

        dmin = field.domainMinBoundary
        dmax = field.domainMaxBoundary
        self.setValue("uVolumeBoundsMin", [float(dmin[0]), float(dmin[1]), float(dmin[2])], callback=False)
        self.setValue("uVolumeBoundsMax", [float(dmax[0]), float(dmax[1]), float(dmax[2])], callback=False)
        vmin = float(vol.min())
        vmax = float(vol.max())
        if vmax <= vmin:
            vmax = vmin + 1e-6
        self.setValue("attributeBounds", [vmin, vmax], callback=False)
        self._has_volume = True

    # ------------------------------------------------------------- transfer fn

    def _current_colormap_cpu(self):
        tm = getTextureManager()
        names = list(tm.getBuiltInTextureNames())
        sel = self.getOptionValue("colorMap")
        idx = names.index(sel) if sel in names else 0
        return tm.getBuiltInColormapCPU(idx)

    @staticmethod
    def _sample_opacity(pts, t):
        if not pts:
            return 0.0
        if t <= pts[0][0]:
            return pts[0][1]
        if t >= pts[-1][0]:
            return pts[-1][1]
        for i in range(len(pts) - 1):
            t0, a0 = pts[i]
            t1, a1 = pts[i + 1]
            if t0 <= t <= t1:
                if t1 <= t0:
                    return a1
                w = (t - t0) / (t1 - t0)
                return a0 * (1.0 - w) + a1 * w
        return pts[-1][1]

    def _rebuild_tf_lut(self):
        cmap = self._current_colormap_cpu()
        f = np.linspace(0.0, 1.0, _TF_RES)
        lut = np.zeros((_TF_RES, 4), dtype=np.float32)
        if cmap is not None and cmap.shape[0] > 0:
            ci = np.round(f * (cmap.shape[0] - 1)).astype(int)
            lut[:, 0:3] = cmap[ci]
        else:
            lut[:, 0:3] = f[:, None]
        pts = sorted(self.opacity_points, key=lambda p: p[0])
        lut[:, 3] = np.array([self._sample_opacity(pts, float(tt)) for tt in f], dtype=np.float32)

        gl.glBindTexture(gl.GL_TEXTURE_1D, self.tf_tex)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glTexImage1D(gl.GL_TEXTURE_1D, 0, gl.GL_RGBA16F, _TF_RES, 0, gl.GL_RGBA, gl.GL_FLOAT,
                        np.ascontiguousarray(lut, dtype=np.float32))
        gl.glBindTexture(gl.GL_TEXTURE_1D, 0)
        self._lut_dirty = False

    def _update_scene_depth(self):
        """Copy the current (opaque) depth buffer into uSceneDepth; return [width, height]."""
        vp = gl.glGetIntegerv(gl.GL_VIEWPORT)
        x, y, w, h = int(vp[0]), int(vp[1]), int(vp[2]), int(vp[3])
        if w <= 0 or h <= 0:
            return [float(max(w, 1)), float(max(h, 1))]
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.depth_tex)
        if (w, h) != (self._depth_w, self._depth_h):
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_DEPTH_COMPONENT24, w, h, 0,
                            gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT, None)
            self._depth_w, self._depth_h = w, h
        gl.glCopyTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, x, y, w, h)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        return [float(w), float(h)]

    # ------------------------------------------------------------------ render

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

        # Rebuild the transfer-function LUT on colormap change or opacity edits.
        cm = self.getOptionValue("colorMap")
        if cm != self._last_colormap:
            self._last_colormap = cm
            self._lut_dirty = True
        if self._lut_dirty:
            self._rebuild_tf_lut()

        # Per-frame uniforms sourced on the CPU.
        if self.cameraObject is not None:
            try:
                p = self.cameraObject.getValue("position")
                self.setValue("camPos", [float(p[0]), float(p[1]), float(p[2])], callback=False)
            except Exception:
                pass
        self.setValue("uEnableEmptySkip", 1 if self.getValue("enableEmptySkip") else 0, callback=False)

        # Depth-aware compositing: copy the opaque depth buffer + inverse view-projection so the
        # ray-march can be clamped to the nearest opaque surface. DVR is drawn last (after all opaque
        # geometry), so the depth buffer is complete here.
        use_depth = bool(self.getValue("depthTest"))
        self.setValue("uUseSceneDepth", 1 if use_depth else 0, callback=False)
        if use_depth:
            self.setValue("uViewport", self._update_scene_depth(), callback=False)
            if self.cameraObject is not None:
                try:
                    cs = self.cameraObject.getScope()
                    self.setValue("uInvViewProj", glm.inverse(cs["projMat"] * cs["viewMat"]), callback=False)
                except Exception:
                    pass

        # DVR pass: composite the premultiplied ray-march over the scene. Depth test is OFF — occlusion
        # is done in the shader by clamping the ray to the scene depth (so geometry intersecting the box
        # no longer truncates the whole ray); depth write OFF; back faces only (front-cull).
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDepthMask(gl.GL_FALSE)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_ONE, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glCullFace(gl.GL_FRONT)
        super().render()
        gl.glCullFace(gl.GL_BACK)
        gl.glDisable(gl.GL_CULL_FACE)
        gl.glDisable(gl.GL_BLEND)
        gl.glDepthMask(gl.GL_TRUE)
        gl.glEnable(gl.GL_DEPTH_TEST)
        # Unbind the sampler textures apply() bound (3D volume/minmax, 1D TF, 2D scene depth).
        # Leaving a non-2D texture on unit 0 corrupts the fixed-pipeline imgui renderer.
        for unit in (0, 1, 2, 3):
            gl.glActiveTexture(gl.GL_TEXTURE0 + unit)
            gl.glBindTexture(gl.GL_TEXTURE_3D, 0)
            gl.glBindTexture(gl.GL_TEXTURE_1D, 0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glActiveTexture(gl.GL_TEXTURE0)

    # --------------------------------------------------------------- GUI / TF editor

    def drawGui(self):
        if not self.getValue("GuiVisible"):
            return
        _, opened = imgui.begin(self.name, self.getValue("GuiVisible"))
        self.DrawPropertiesInGui(self.persistentProperties)
        self.DrawPropertiesInGui(self.nonPersistentProperties)
        self.DrawActionButtons()
        imgui.separator()
        try:
            self._draw_tf_editor()
        except Exception as e:
            logging.getLogger().warning(f"[VolumeRender] transfer-function editor failed: {e}")
        imgui.end()
        self.updateValue("GuiVisible", opened)

    def _draw_tf_editor(self):
        """imgui opacity-curve editor (the Python counterpart of the C++ TransferFunctionWidget):
        left-drag to move a point, left-click on empty space to add one, right-click a point to
        remove it; a colormap ramp is drawn underneath."""
        imgui.text("Opacity TF  (L-drag move | L-click add | R-click remove)")
        avail = imgui.get_content_region_available()
        W = max(float(avail.x), 80.0)
        H = 120.0
        origin = imgui.get_cursor_screen_pos()
        x0, y0 = float(origin.x), float(origin.y)
        imgui.invisible_button("dvr_tf_canvas", W, H)
        hovered = imgui.is_item_hovered()
        dl = imgui.get_window_draw_list()

        bg = imgui.get_color_u32_rgba(0.12, 0.12, 0.14, 1.0)
        border = imgui.get_color_u32_rgba(0.5, 0.5, 0.55, 1.0)
        grid = imgui.get_color_u32_rgba(0.25, 0.25, 0.28, 1.0)
        line_col = imgui.get_color_u32_rgba(1.0, 0.75, 0.2, 1.0)
        dl.add_rect_filled(x0, y0, x0 + W, y0 + H, bg)
        for gi in range(1, 4):
            gx = x0 + W * gi / 4.0
            gy = y0 + H * gi / 4.0
            dl.add_line(gx, y0, gx, y0 + H, grid, 1.0)
            dl.add_line(x0, gy, x0 + W, gy, grid, 1.0)
        dl.add_rect(x0, y0, x0 + W, y0 + H, border)

        pts = sorted(self.opacity_points, key=lambda p: p[0])
        self.opacity_points = pts

        def S(t, a):
            return (x0 + t * W, y0 + (1.0 - a) * H)

        def to_norm(px, py):
            t = (px - x0) / W
            a = 1.0 - (py - y0) / H
            return min(max(t, 0.0), 1.0), min(max(a, 0.0), 1.0)

        for i in range(len(pts) - 1):
            sx0, sy0 = S(*pts[i])
            sx1, sy1 = S(*pts[i + 1])
            dl.add_line(sx0, sy0, sx1, sy1, line_col, 2.0)

        io = imgui.get_io()
        mx, my = float(io.mouse_pos.x), float(io.mouse_pos.y)
        nearest, nd = None, 1e9
        for i, (t, a) in enumerate(pts):
            sx, sy = S(t, a)
            d = abs(mx - sx) + abs(my - sy)
            if d < nd:
                nd, nearest = d, i
        for i, (t, a) in enumerate(pts):
            sx, sy = S(t, a)
            hot = (i == nearest and nd < 12.0)
            col = imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0) if hot else line_col
            dl.add_circle_filled(sx, sy, 5.0, col)

        # Interaction.
        if hovered and imgui.is_mouse_clicked(1) and nearest is not None and nd < 12.0:
            if 0 < nearest < len(pts) - 1:
                del self.opacity_points[nearest]
                self._lut_dirty = True
        elif hovered and imgui.is_mouse_clicked(0):
            if nearest is not None and nd < 12.0:
                self._drag_idx = nearest
            else:
                t, a = to_norm(mx, my)
                self.opacity_points.append((t, a))
                self.opacity_points = sorted(self.opacity_points, key=lambda p: p[0])
                self._lut_dirty = True
                self._drag_idx = None
        if self._drag_idx is not None and imgui.is_mouse_down(0):
            i = self._drag_idx
            if 0 <= i < len(self.opacity_points):
                t, a = to_norm(mx, my)
                if i == 0:
                    t = 0.0
                elif i == len(self.opacity_points) - 1:
                    t = 1.0
                else:
                    lo = self.opacity_points[i - 1][0] + 1e-3
                    hi = self.opacity_points[i + 1][0] - 1e-3
                    t = min(max(t, lo), hi)
                self.opacity_points[i] = (t, a)
                self._lut_dirty = True
        if imgui.is_mouse_released(0):
            self._drag_idx = None

        # Colormap ramp underneath.
        imgui.dummy(0.0, 2.0)
        rorigin = imgui.get_cursor_screen_pos()
        rx, ry = float(rorigin.x), float(rorigin.y)
        rH = 16.0
        imgui.invisible_button("dvr_tf_ramp", W, rH)
        cmap = self._current_colormap_cpu()
        strips = 64
        for s in range(strips):
            fr = s / float(strips - 1)
            if cmap is not None and cmap.shape[0] > 0:
                c = cmap[int(round(fr * (cmap.shape[0] - 1)))]
                cr, cg, cb = float(c[0]), float(c[1]), float(c[2])
            else:
                cr = cg = cb = fr
            cu = imgui.get_color_u32_rgba(cr, cg, cb, 1.0)
            dl.add_rect_filled(rx + W * s / strips, ry, rx + W * (s + 1) / strips, ry + rH, cu)
        dl.add_rect(rx, ry, rx + W, ry + rH, border)
