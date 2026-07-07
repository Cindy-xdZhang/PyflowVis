"""Centralized general-geometry render service, the Python counterpart of optimal-connection's
``GeneralMeshRenderer`` (src/core/buitin_gui/GeneralGeometryRender.*).

Other modules obtain the single instance via ``scene.getObject("GeneralMeshRenderer")`` and push
geometry into it with the ``insert_*`` API instead of writing their own renderer; the service owns
the VAOs, the Blinn-Phong/colormap material, tag-based grouping, and the per-frame draw loop.

Three named collections mirror the C++ maps:
  * ``curves``     — polylines rendered as lit tubes (parallel-transport frames, arc-length UV);
  * ``surfaces``   — triangle meshes (position/normal/uv), optionally transparent;
  * ``prototypes`` — spheres / arrows / custom primitives (indexed or soup geometry).

Design note vs. C++: the C++ engine uses two buffer paths (a struct-interleaved
``VertexArrayObjectGeneric`` for curves/surfaces and a separate index-geometry
``VertexArrayObjectPrototying`` *without* normals for prototypes). Here everything uses ONE
interleaved position+normal+uv layout so the single Blinn-Phong material lights every renderable
(including spheres/arrows) correctly. Lighting is a headlight (this engine has no scene-light
uniforms — see general_mesh_fragment.glsl). Only the core service is implemented; ray-picking, the
transform gizmo, PBR/STL/animation, and uplifting materials are intentionally left out.
"""
import ctypes
import logging

import imgui
import numpy as np
from OpenGL import GL as gl

from .VertexArrayObject import VertexArrayObject
from .Object import Object
from .shaderManager import Material, getTextureManager

# ── Default group tags (mirror the C++ GROUP_* constants) ────────────────────
GROUP_CURVES = "curves"
GROUP_SURFACES = "surfaces"
GROUP_PROTOTYPES = "prototypes"
GROUP_SPHERES = "spheres"
GROUP_ARROWS = "arrows"
GROUP_CORELINE_SURFACES = "coreline surfaces"
GROUP_TRANSPARENT = "transparent"

_VERT_STRIDE = 32  # bytes: position(3f) + normal(3f) + uv(2f)


# =============================================================================
# Geometry helpers (ported from GeneralGeometryRender.cpp anonymous namespace).
# Every helper returns an (M, 8) float32 "soup": [px,py,pz, nx,ny,nz, u,v] per vertex.
# =============================================================================

def _safe_normalize(v, fallback=(0.0, 0.0, 1.0)):
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return np.asarray(fallback, dtype=np.float32)
    return v / n


def _ortho_basis(direction):
    """Return (d, u, v): a right-handed orthonormal frame with d along `direction`."""
    d = _safe_normalize(direction, (0.0, 0.0, 1.0))
    ref = np.array([0.0, 1.0, 0.0], np.float32) if abs(float(d[1])) < 0.9 else np.array([1.0, 0.0, 0.0], np.float32)
    u = _safe_normalize(np.cross(ref, d))
    v = _safe_normalize(np.cross(d, u))
    return d, u, v


def _build_tangents(P):
    n = len(P)
    T = np.zeros((n, 3), np.float32)
    for i in range(n):
        if i == 0:
            t = P[1] - P[0]
        elif i == n - 1:
            t = P[n - 1] - P[n - 2]
        else:
            t = P[i + 1] - P[i - 1]
        T[i] = _safe_normalize(t, (1.0, 0.0, 0.0))
    return T


def _build_transport_frames(T):
    """Rotation-minimizing frames via Rodrigues parallel transport (no twist)."""
    n = len(T)
    N = np.zeros((n, 3), np.float32)
    B = np.zeros((n, 3), np.float32)
    if n == 0:
        return N, B
    ref = np.array([0.0, 1.0, 0.0], np.float32)
    if abs(float(np.dot(T[0], ref))) > 0.9:
        ref = np.array([1.0, 0.0, 0.0], np.float32)
    N[0] = _safe_normalize(np.cross(ref, T[0]), (0.0, 0.0, 1.0))
    B[0] = _safe_normalize(np.cross(T[0], N[0]), (0.0, 1.0, 0.0))
    N[0] = _safe_normalize(np.cross(B[0], T[0]), N[0])
    for i in range(1, n):
        t0, t1 = T[i - 1], T[i]
        axis = np.cross(t0, t1)
        s = float(np.linalg.norm(axis))
        c = float(np.dot(t0, t1))
        if s < 1e-6:
            N[i], B[i] = N[i - 1], B[i - 1]
        else:
            a = axis / s
            Ni = N[i - 1] * c + np.cross(a, N[i - 1]) * s + a * float(np.dot(a, N[i - 1])) * (1.0 - c)  # Rodrigues
            Ni = _safe_normalize(Ni, N[i - 1])
            Bi = _safe_normalize(np.cross(t1, Ni), B[i - 1])
            Ni = _safe_normalize(np.cross(Bi, t1), Ni)
            N[i], B[i] = Ni, Bi
    return N, B


def _default_uvs(P):
    """U = normalized cumulative arc length, V = 0."""
    n = len(P)
    s = np.zeros(n, np.float32)
    for i in range(1, n):
        s[i] = s[i - 1] + float(np.linalg.norm(P[i] - P[i - 1]))
    total = s[-1] if s[-1] > 1e-8 else 1.0
    uv = np.zeros((n, 2), np.float32)
    uv[:, 0] = s / total
    return uv


def tube_soup(P, radius=0.02, segments=8, uvs=None):
    """Sweep a circle of `radius` along polyline P -> a triangle soup with radial normals."""
    P = np.asarray(P, dtype=np.float32)
    if len(P) < 2 or segments < 3 or radius <= 0.0:
        return np.zeros((0, 8), np.float32)
    if uvs is None:
        uvs = _default_uvs(P)
    else:
        uvs = np.asarray(uvs, np.float32)
    T = _build_tangents(P)
    N, B = _build_transport_frames(T)
    two_pi = 2.0 * np.pi
    out = np.empty(((len(P) - 1) * segments * 6, 8), np.float32)
    w = 0
    for i in range(len(P) - 1):
        p0, p1 = P[i], P[i + 1]
        n0, b0, n1, b1 = N[i], B[i], N[i + 1], B[i + 1]
        uv0, uv1 = uvs[i], uvs[i + 1]
        for k in range(segments):
            kn = (k + 1) % segments
            a0 = two_pi * k / segments
            a1 = two_pi * kn / segments
            c0, s0 = np.cos(a0), np.sin(a0)
            c1, s1 = np.cos(a1), np.sin(a1)
            o00 = radius * (c0 * n0 + s0 * b0)
            o01 = radius * (c1 * n0 + s1 * b0)
            o10 = radius * (c0 * n1 + s0 * b1)
            o11 = radius * (c1 * n1 + s1 * b1)
            for p, off, uv in ((p0, o00, uv0), (p1, o10, uv1), (p1, o11, uv1),
                               (p0, o00, uv0), (p1, o11, uv1), (p0, o01, uv0)):
                nrm = _safe_normalize(off)
                out[w, 0:3] = p + off
                out[w, 3:6] = nrm
                out[w, 6:8] = uv
                w += 1
    return out


_ICO_CACHE = {}


def _icosphere(recursion):
    """Unit icosphere -> (verts (V,3) float32 on the unit sphere, faces (F,3) int)."""
    if recursion in _ICO_CACHE:
        return _ICO_CACHE[recursion]
    t = (1.0 + np.sqrt(5.0)) / 2.0
    base = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
            (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
            (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    verts = [list(_safe_normalize(v)) for v in base]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
             (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
             (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
             (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    for _ in range(int(recursion)):
        mid = {}

        def midpoint(a, b):
            key = (a, b) if a < b else (b, a)
            if key in mid:
                return mid[key]
            m = list(_safe_normalize((np.asarray(verts[a]) + np.asarray(verts[b])) * 0.5))
            verts.append(m)
            idx = len(verts) - 1
            mid[key] = idx
            return idx

        new_faces = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new_faces
    result = (np.asarray(verts, np.float32), np.asarray(faces, np.int32))
    _ICO_CACHE[recursion] = result
    return result


def sphere_soup(center, radius=0.02, recursion=2, uv=(0.0, 0.0)):
    """Icosphere at `center` with smooth (unit-normal) shading -> triangle soup."""
    uverts, faces = _icosphere(recursion)
    center = np.asarray(center, np.float32)
    out = np.empty((len(faces) * 3, 8), np.float32)
    w = 0
    for a, b, c in faces:
        for vi in (a, b, c):
            n = uverts[vi]
            out[w, 0:3] = center + radius * n
            out[w, 3:6] = n
            out[w, 6:8] = uv
            w += 1
    return out


def _ring(center, u, v, radius, segments):
    return [center + radius * (np.cos(2.0 * np.pi * k / segments) * u + np.sin(2.0 * np.pi * k / segments) * v)
            for k in range(segments)]


def _flat_soup(tris, uv=(0.0, 0.0)):
    """Triangle list [(p0,p1,p2), ...] -> soup with per-triangle (flat) normals."""
    out = np.empty((len(tris) * 3, 8), np.float32)
    w = 0
    for a, b, c in tris:
        a = np.asarray(a, np.float32)
        b = np.asarray(b, np.float32)
        c = np.asarray(c, np.float32)
        n = _safe_normalize(np.cross(b - a, c - a))
        for p in (a, b, c):
            out[w, 0:3] = p
            out[w, 3:6] = n
            out[w, 6:8] = uv
            w += 1
    return out


def _cylinder_tris(base, direction, radius, height, segments):
    d, u, v = _ortho_basis(direction)
    top = base + d * height
    rb = _ring(base, u, v, radius, segments)
    rt = _ring(top, u, v, radius, segments)
    tris = []
    for k in range(segments):
        kn = (k + 1) % segments
        tris.append((rb[k], rb[kn], rt[kn]))
        tris.append((rb[k], rt[kn], rt[k]))
        tris.append((base, rb[kn], rb[k]))   # bottom cap
        tris.append((top, rt[k], rt[kn]))    # top cap
    return tris


def _cone_tris(base, direction, radius, height, segments):
    d, u, v = _ortho_basis(direction)
    apex = base + d * height
    rb = _ring(base, u, v, radius, segments)
    tris = []
    for k in range(segments):
        kn = (k + 1) % segments
        tris.append((rb[k], rb[kn], apex))   # side
        tris.append((base, rb[kn], rb[k]))   # base cap
    return tris


def arrow_soup(origin, direction, total_height, stem_radius, tip_radius, segments=8, uv=(0.0, 0.0)):
    """Arrow = cylinder stem + cone tip (tip = 1/3 of total height), from `origin` along `direction`."""
    d = _safe_normalize(direction, (0.0, 0.0, 1.0))
    origin = np.asarray(origin, np.float32)
    cone_h = total_height * 0.33333
    cyl_h = total_height - cone_h
    tris = _cylinder_tris(origin, d, stem_radius, cyl_h, segments)
    tris += _cone_tris(origin + d * cyl_h, d, tip_radius, cone_h, segments)
    return _flat_soup(tris, uv)


# =============================================================================
# Renderable: one drawable object (own VAO + per-object uniforms + tags).
# =============================================================================

class _Renderable(Object):
    """A single drawable owned by the renderer. Interleaved position+normal+uv VAO, plus the
    per-object shader uniforms (modelMat / baseColor / colorMap) fed through getScope()."""

    def __init__(self, name):
        super().__init__(name)
        self.tags = []
        self.visible = True
        self.vertex_count = 0
        self.index_count = 0

        self.vao = int(gl.glGenVertexArrays(1))
        self.vbo = int(gl.glGenBuffers(1))
        self.ebo = int(gl.glGenBuffers(1))
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, 0, None, gl.GL_DYNAMIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, _VERT_STRIDE, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, _VERT_STRIDE, ctypes.c_void_p(12))
        gl.glEnableVertexAttribArray(2)
        gl.glVertexAttribPointer(2, 2, gl.GL_FLOAT, gl.GL_FALSE, _VERT_STRIDE, ctypes.c_void_p(24))
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, 0, None, gl.GL_DYNAMIC_DRAW)
        gl.glBindVertexArray(0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)

        # Per-object shader uniforms (name-matched by the material scope).
        self.create_variable("modelMat", np.eye(4, dtype=np.float32), False, False)
        self.create_variable("baseColor", [0.85, 0.4, 0.25, 1.0], False, False)
        self.create_variable("colorMap", -1, False, False)
        self._accum = []
        self._centroid = np.zeros(3, dtype=np.float32)   # object-space center, for transparency sort

    def upload(self, verts, indices=None):
        verts = np.ascontiguousarray(verts, dtype=np.float32)
        if verts.ndim != 2 or verts.shape[1] != 8:
            verts = verts.reshape(-1, 8) if verts.size else np.zeros((0, 8), np.float32)
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, verts.nbytes, verts if verts.size else None, gl.GL_DYNAMIC_DRAW)
        self.vertex_count = int(verts.shape[0])
        self._centroid = (verts[:, 0:3].mean(axis=0).astype(np.float32)
                          if verts.shape[0] > 0 else np.zeros(3, dtype=np.float32))
        if indices is not None and len(indices) > 0:
            idx = np.ascontiguousarray(indices, dtype=np.uint32).reshape(-1)
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
            gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, gl.GL_DYNAMIC_DRAW)
            self.index_count = int(idx.size)
        else:
            self.index_count = 0
        gl.glBindVertexArray(0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)

    # Prototype-style accumulation (mirrors appendXxx + commit on the C++ prototyping VAO).
    def append(self, soup):
        if soup is not None and len(soup) > 0:
            self._accum.append(np.asarray(soup, np.float32))

    def commit(self):
        soup = np.concatenate(self._accum, axis=0) if self._accum else np.zeros((0, 8), np.float32)
        self._accum = []
        self.upload(soup)

    def set_color(self, rgba):
        r, g, b = float(rgba[0]), float(rgba[1]), float(rgba[2])
        a = float(rgba[3]) if len(rgba) > 3 else 1.0
        self.setValue("baseColor", [r, g, b, a], callback=False)
        self.setValue("colorMap", -1, callback=False)

    def use_colormap(self, index):
        self.setValue("colorMap", int(index), callback=False)

    def draw(self):
        if self.vertex_count == 0 and self.index_count == 0:
            return
        gl.glBindVertexArray(self.vao)
        if self.index_count > 0:
            gl.glDrawElements(gl.GL_TRIANGLES, self.index_count, gl.GL_UNSIGNED_INT, None)
        else:
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.vertex_count)
        gl.glBindVertexArray(0)

    def world_centroid(self):
        """World-space center of the geometry (modelMat applied to the object-space centroid),
        used to depth-sort transparent renderables back-to-front."""
        m = np.asarray(self.getValue("modelMat"), dtype=np.float32)
        c = np.array([self._centroid[0], self._centroid[1], self._centroid[2], 1.0], dtype=np.float32)
        return (m @ c)[:3]

    def dispose(self):
        try:
            gl.glDeleteVertexArrays(1, [self.vao])
            gl.glDeleteBuffers(1, [self.vbo])
            gl.glDeleteBuffers(1, [self.ebo])
        except Exception:
            pass


# =============================================================================
# The service.
# =============================================================================

class GeneralMeshRenderObject(VertexArrayObject):
    def __init__(self, name="GeneralMeshRenderer"):
        super().__init__(name)
        self.curves = {}
        self.surfaces = {}
        self.prototypes = {}
        self.registered_groups = []

        # One Blinn-Phong/colormap material for every renderable; texture0="builtIn" feeds the
        # sampler1DArray "colorMaps1Darray".
        self.material = Material("generalMeshMaterial", "generalMeshMat", texture0="builtIn")
        self.setMaterial(self.material)

        # camPos uniform (world-space camera position), refreshed each frame for the headlight.
        self.create_variable("camPos", [0.0, 0.0, 0.0], False, False)

        # UI defaults used by the insert_* helpers / demo.
        colormap_names = list(getTextureManager().getBuiltInTextureNames()) or ["default"]
        self.create_variable_gui("defaultColor", (0.85, 0.4, 0.25), False, {'widget': 'color_picker'})
        self.create_variable_gui("defaultColormap", colormap_names, False)
        self.create_variable_gui("lineRadius", 0.03, False, {'widget': 'slider_float', 'min': 0.001, 'max': 0.2})
        self.create_variable_gui("cylinderSegments", 8, False, {'widget': 'input'})
        self.create_variable_gui("sphereRadius", 0.05, False, {'widget': 'slider_float', 'min': 0.005, 'max': 0.5})

        self.addAction("add demo geometry", lambda obj: obj._add_demo())
        self.addAction("clear all", lambda obj: obj.clear_all())

    def postInit(self):
        super().postInit()

    # ------------------------------------------------------------- utilities

    def _colormap_index(self, colormap):
        names = list(getTextureManager().getBuiltInTextureNames())
        if isinstance(colormap, (int, np.integer)):
            return int(max(0, min(int(colormap), len(names) - 1))) if names else 0
        if names and colormap in names:
            return names.index(colormap)
        return 0

    def _default_rgba(self):
        c = self.getValue("defaultColor")
        return [float(c[0]), float(c[1]), float(c[2]), 1.0]

    def _register_tags(self, tags):
        for t in tags:
            if t and t not in self.registered_groups:
                self.registered_groups.append(t)

    def _apply_color(self, renderable, color, colormap):
        if colormap is not None:
            renderable.use_colormap(self._colormap_index(colormap))
        elif color is not None:
            renderable.set_color(color)
        else:
            renderable.set_color(self._default_rgba())

    def _put(self, collection, name, renderable):
        old = collection.get(name)
        if old is not None:
            old.dispose()
        collection[name] = renderable
        self._register_tags(renderable.tags)

    # ------------------------------------------------------------- insert API

    def insert_curves(self, name, polylines, tags=None, color=None, colormap=None, radius=None, segments=None):
        """Insert one or more polylines rendered as lit tubes. `polylines` is a list of (Ni,3)
        arrays (or a single (N,3) array). With `colormap` set, color follows arc-length uv.x."""
        if isinstance(polylines, np.ndarray) and polylines.ndim == 2:
            polylines = [polylines]
        radius = float(self.getValue("lineRadius")) if radius is None else float(radius)
        segments = int(self.getValue("cylinderSegments")) if segments is None else int(segments)
        soups = []
        for P in polylines:
            P = np.asarray(P, np.float32)
            if len(P) >= 2:
                soups.append(tube_soup(P, radius, segments))
        r = _Renderable(name)
        r.tags = [GROUP_CURVES] + list(tags or [])
        if soups:
            r.upload(np.concatenate(soups, axis=0))
        self._apply_color(r, color, colormap)
        self._put(self.curves, name, r)
        return r

    def insert_surface(self, name, vertices, normals, faces=None, uvs=None, tags=None,
                       color=None, colormap=None, transparent=False):
        """Insert a triangle mesh. If `faces` (F,3) is given it is drawn indexed; otherwise
        `vertices`/`normals` are treated as an ordered triangle soup."""
        vertices = np.asarray(vertices, np.float32)
        normals = np.asarray(normals, np.float32)
        uvs = np.zeros((len(vertices), 2), np.float32) if uvs is None else np.asarray(uvs, np.float32)
        inter = np.concatenate([vertices, normals, uvs], axis=1).astype(np.float32)
        r = _Renderable(name)
        r.tags = [GROUP_SURFACES] + list(tags or [])
        if transparent and GROUP_TRANSPARENT not in r.tags:
            r.tags.append(GROUP_TRANSPARENT)
        r.upload(inter, indices=(None if faces is None else np.asarray(faces, np.uint32)))
        self._apply_color(r, color, colormap)
        self._put(self.surfaces, name, r)
        return r

    def get_or_create_prototype(self, name, tags=None):
        """Return the prototype renderable named `name` (creating it if absent). Append geometry
        with append_sphere/append_arrow/append(soup) and finish with commit()."""
        r = self.prototypes.get(name)
        if r is None:
            r = _Renderable(name)
            r.tags = [GROUP_PROTOTYPES] + list(tags or [])
            r.set_color(self._default_rgba())
            self._put(self.prototypes, name, r)
        else:
            for t in (tags or []):
                if t and t not in r.tags:
                    r.tags.append(t)
            self._register_tags(r.tags)
        return r

    def insert_sphere_prototypes(self, name, positions, radius=None, recursion=2, tags=None,
                                 color=None, colormap=None, uvs=None):
        """Create (or replace) a prototype object with one icosphere per position."""
        radius = float(self.getValue("sphereRadius")) if radius is None else float(radius)
        r = _Renderable(name)
        r.tags = [GROUP_PROTOTYPES, GROUP_SPHERES] + list(tags or [])
        for i, c in enumerate(positions):
            uv = uvs[i] if (uvs is not None and i < len(uvs)) else (0.0, 0.0)
            r.append(sphere_soup(np.asarray(c, np.float32), radius, recursion, uv))
        r.commit()
        self._apply_color(r, color, colormap)
        self._put(self.prototypes, name, r)
        return r

    def insert_arrow_prototypes(self, name, positions, directions, scale=0.4, stem_radius=0.015,
                                tip_radius=0.05, segments=8, tags=None, color=None):
        """Create (or replace) a prototype object with one arrow per (position, direction)."""
        r = _Renderable(name)
        r.tags = [GROUP_PROTOTYPES, GROUP_ARROWS] + list(tags or [])
        for i, origin in enumerate(positions):
            d = directions[i]
            total = float(scale) * float(np.linalg.norm(np.asarray(d, np.float32)))
            if total < 1e-8:
                continue
            r.append(arrow_soup(origin, d, total, stem_radius, tip_radius, segments))
        r.commit()
        self._apply_color(r, color, None)
        self._put(self.prototypes, name, r)
        return r

    # ------------------------------------------------------------- tag / color queries

    def find(self, name):
        for coll in (self.curves, self.surfaces, self.prototypes):
            if name in coll:
                return coll[name]
        return None

    def get_names_by_tag(self, tag):
        out = []
        for coll in (self.curves, self.surfaces, self.prototypes):
            for nm, r in coll.items():
                if tag in r.tags:
                    out.append(nm)
        return out

    def update_base_color(self, name, color):
        r = self.find(name)
        if r is not None:
            r.set_color(color)

    def update_base_color_group(self, group, color):
        for nm in self.get_names_by_tag(group):
            self.find(nm).set_color(color)

    def remove(self, name):
        for coll in (self.curves, self.surfaces, self.prototypes):
            r = coll.pop(name, None)
            if r is not None:
                r.dispose()

    def clear_all(self):
        for coll in (self.curves, self.surfaces, self.prototypes):
            for r in coll.values():
                r.dispose()
            coll.clear()

    def set_object_visible(self, name, visible):
        r = self.find(name)
        if r is not None:
            r.visible = bool(visible)

    def set_group_visible(self, tag, visible):
        for nm in self.get_names_by_tag(tag):
            r = self.find(nm)
            if r is not None:
                r.visible = bool(visible)

    def is_group_visible(self, tag):
        names = self.get_names_by_tag(tag)
        return bool(names) and all(self.find(nm).visible for nm in names)

    def remove_group(self, tag):
        for nm in self.get_names_by_tag(tag):
            self.remove(nm)

    # ------------------------------------------------------------- render

    def render(self):
        if not (self.curves or self.surfaces or self.prototypes):
            return
        if self.cameraObject is not None:
            try:
                p = self.cameraObject.getValue("position")
                self.setValue("camPos", [float(p[0]), float(p[1]), float(p[2])], callback=False)
            except Exception:
                pass

        # Save the GL state we touch (mirror of the C++ ScopedBlendDepthCullState).
        prev_cull = bool(gl.glIsEnabled(gl.GL_CULL_FACE))
        prev_blend = bool(gl.glIsEnabled(gl.GL_BLEND))

        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthMask(gl.GL_TRUE)
        gl.glDisable(gl.GL_CULL_FACE)   # two-sided lighting handles back faces
        gl.glDisable(gl.GL_BLEND)

        self._draw_collection(self.curves)
        self._draw_collection(self.surfaces, allow_transparent=True)
        self._draw_collection(self.prototypes)

        gl.glUseProgram(0)
        gl.glBindVertexArray(0)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_1D_ARRAY, 0)

        # Restore state.
        gl.glDepthMask(gl.GL_TRUE)
        if prev_blend:
            gl.glEnable(gl.GL_BLEND)
        else:
            gl.glDisable(gl.GL_BLEND)
        if prev_cull:
            gl.glEnable(gl.GL_CULL_FACE)
        else:
            gl.glDisable(gl.GL_CULL_FACE)

    def _draw_collection(self, collection, allow_transparent=False):
        renderables = [r for r in collection.values() if r.visible]
        if not renderables:
            return

        def _is_transparent(r):
            return allow_transparent and (GROUP_TRANSPARENT in r.tags or GROUP_CORELINE_SURFACES in r.tags)

        opaque = [r for r in renderables if not _is_transparent(r)]
        transparent = [r for r in renderables if _is_transparent(r)]

        # Opaque first, with depth writes on so they occlude correctly.
        for r in opaque:
            self.material.apply([self.parentScene, self.cameraObject, self, r])
            r.draw()

        if not transparent:
            return

        # Transparent next: draw back-to-front (farthest first) so src-over alpha blending
        # composites in the right order. Depth writes off, depth test still on.
        eye = None
        if self.cameraObject is not None:
            try:
                p = self.cameraObject.getValue("position")
                eye = np.array([float(p[0]), float(p[1]), float(p[2])], dtype=np.float32)
            except Exception:
                eye = None
        if eye is not None:
            def _sq_dist_to_eye(r):
                d = r.world_centroid() - eye
                return float(np.dot(d, d))
            transparent.sort(key=_sq_dist_to_eye, reverse=True)

        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glDepthMask(gl.GL_FALSE)
        for r in transparent:
            self.material.apply([self.parentScene, self.cameraObject, self, r])
            r.draw()
        gl.glDisable(gl.GL_BLEND)
        gl.glDepthMask(gl.GL_TRUE)

    # ------------------------------------------------------------- GUI / tag manager

    def drawGui(self):
        if not self.getValue("GuiVisible"):
            return
        _, opened = imgui.begin(self.name, self.getValue("GuiVisible"))
        self.DrawPropertiesInGui(self.persistentProperties)
        self.DrawPropertiesInGui(self.nonPersistentProperties)
        self.DrawActionButtons()
        imgui.separator()
        imgui.text("Tag / group manager")
        try:
            self._draw_tag_manager()
        except Exception as e:
            logging.getLogger().warning(f"[GeneralMeshRenderer] tag manager failed: {e}")
        imgui.end()
        self.updateValue("GuiVisible", opened)

    def _draw_tag_manager(self):
        """Group renderables by tag; per group: visibility toggle (all members), recolor, delete;
        per member: visibility toggle + delete. This is the Python tag/group management surface."""
        groups = {}
        for coll in (self.curves, self.surfaces, self.prototypes):
            for nm, r in coll.items():
                for tg in r.tags:
                    groups.setdefault(tg, []).append((nm, r))
        if not groups:
            imgui.text_disabled("(empty - click 'add demo geometry', or insert_* from a module)")
            return

        remove_objs = []
        remove_groups = []
        for tag in sorted(groups.keys()):
            members = groups[tag]
            all_vis = all(r.visible for _, r in members)
            changed, val = imgui.checkbox("##grpvis_" + tag, all_vis)
            if changed:
                for _, r in members:
                    r.visible = bool(val)
            imgui.same_line()
            node_open = imgui.tree_node(f"{tag}  ({len(members)})##grp_" + tag)
            imgui.same_line()
            if imgui.small_button("color##" + tag):
                self.update_base_color_group(tag, self._default_rgba())
            imgui.same_line()
            if imgui.small_button("del##" + tag):
                remove_groups.append(tag)
            if node_open:
                for nm, r in members:
                    cv, rv = imgui.checkbox(f"{nm}##obj_{tag}_{nm}", r.visible)
                    if cv:
                        r.visible = bool(rv)
                    imgui.same_line()
                    if imgui.small_button(f"x##del_{tag}_{nm}"):
                        remove_objs.append(nm)
                imgui.tree_pop()

        for nm in remove_objs:
            self.remove(nm)
        for tag in remove_groups:
            self.remove_group(tag)

    # ------------------------------------------------------------- demo

    def _add_demo(self):
        """Insert a small showcase (colormap tube + solid tube + spheres + arrows) so the service
        can be verified without another module driving it."""
        t = np.linspace(0.0, 4.0 * np.pi, 80)
        helix = np.stack([0.6 * np.cos(t), 0.6 * np.sin(t), np.linspace(-1.0, 1.0, 80)], axis=1).astype(np.float32)
        self.insert_curves("demo_helix", [helix], colormap=self.getOptionValue("defaultColormap"),
                           radius=0.03, segments=12)
        line = np.array([[-1.0, -1.0, 0.0], [-0.3, 0.2, 0.3], [0.4, 0.5, -0.2], [1.0, 1.0, 0.5]], np.float32)
        self.insert_curves("demo_curve", [line], color=[0.2, 0.8, 0.4, 1.0], radius=0.03, segments=10)
        pts = np.array([[-0.6, 0.6, 0.0], [0.6, 0.6, 0.0], [0.0, -0.6, 0.3]], np.float32)
        self.insert_sphere_prototypes("demo_spheres", pts, radius=0.08, recursion=2, color=[0.9, 0.3, 0.3, 1.0])
        self.insert_arrow_prototypes("demo_arrows", pts, [[0, 1, 0], [0, 1, 0], [1, 0, 0]],
                                     scale=0.4, color=[0.3, 0.5, 0.95, 1.0])
        logging.getLogger().info("[GeneralMeshRenderer] demo geometry inserted.")
