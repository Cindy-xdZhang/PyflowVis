from .VertexArrayObject import *
from numba import njit, prange
from OpenGL import GL as gl
import ctypes


def _build_arrow_template(segments: int = 12, shaft_len: float = 0.55, shaft_r: float = 0.30, head_r: float = 0.6):
    """Build a unit arrow template pointing along +Z (z in [0,1]).

    Returns a (V, 6) float32 array of flat-shaded triangle-soup vertices, each row
    being [pos.x, pos.y, pos.z, nrm.x, nrm.y, nrm.z]. Drawn once per instance via
    glDrawArraysInstanced; the vertex shader orients/scales it per sample.
    """
    ang = np.linspace(0.0, 2.0 * np.pi, segments + 1)
    cs = np.cos(ang).astype(np.float32)
    sn = np.sin(ang).astype(np.float32)
    tip = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    rows = []
    for i in range(segments):
        c0, s0, c1, s1 = cs[i], sn[i], cs[i + 1], sn[i + 1]
        # --- shaft (cylinder side), smooth radial normals ---
        b0 = (shaft_r * c0, shaft_r * s0, 0.0)
        b1 = (shaft_r * c1, shaft_r * s1, 0.0)
        t0 = (shaft_r * c0, shaft_r * s0, shaft_len)
        t1 = (shaft_r * c1, shaft_r * s1, shaft_len)
        n0 = (c0, s0, 0.0)
        n1 = (c1, s1, 0.0)
        rows.append((*b0, *n0)); rows.append((*b1, *n1)); rows.append((*t1, *n1))
        rows.append((*b0, *n0)); rows.append((*t1, *n1)); rows.append((*t0, *n0))
        # --- cone head, flat per-face normal ---
        cb0 = np.array([head_r * c0, head_r * s0, shaft_len], dtype=np.float32)
        cb1 = np.array([head_r * c1, head_r * s1, shaft_len], dtype=np.float32)
        fn = np.cross(cb1 - cb0, tip - cb0)
        ln = float(np.linalg.norm(fn))
        fn = (fn / ln) if ln > 1e-12 else np.array([0.0, 0.0, 1.0], dtype=np.float32)
        fnt = (float(fn[0]), float(fn[1]), float(fn[2]))
        rows.append((float(cb0[0]), float(cb0[1]), float(cb0[2]), *fnt))
        rows.append((float(cb1[0]), float(cb1[1]), float(cb1[2]), *fnt))
        rows.append((float(tip[0]), float(tip[1]), float(tip[2]), *fnt))
    return np.array(rows, dtype=np.float32)


@njit(parallel=True, fastmath=True)
def _interpolate_2d_vectorized_core_numba(
    field_lower_3d, field_upper_3d,
    x_idx, y_idx, fx, fy, alpha,
    output_vectors  # Pre-allocated (num_samples_y, num_samples_x, 3)
):
    """Numba core for 2D vector field interpolation (bilinear in space, linear in time)."""
    num_samples_y, num_samples_x = x_idx.shape
    for i in prange(num_samples_y):
        for j in range(num_samples_x):
            x_i = x_idx[i, j]
            y_i = y_idx[i, j]
            f_x = fx[i, j]
            f_y = fy[i, j]

            # 4 spatial corners, lower time step
            v000 = field_lower_3d[y_i, x_i]
            v100 = field_lower_3d[y_i, x_i + 1]
            v010 = field_lower_3d[y_i + 1, x_i]
            v110 = field_lower_3d[y_i + 1, x_i + 1]
            # upper time step
            v001 = field_upper_3d[y_i, x_i]
            v101 = field_upper_3d[y_i, x_i + 1]
            v011 = field_upper_3d[y_i + 1, x_i]
            v111 = field_upper_3d[y_i + 1, x_i + 1]

            # interpolate in x
            c00 = v000 * (1.0 - f_x) + v100 * f_x
            c10 = v010 * (1.0 - f_x) + v110 * f_x
            c01 = v001 * (1.0 - f_x) + v101 * f_x
            c11 = v011 * (1.0 - f_x) + v111 * f_x

            # interpolate in y
            c0 = c00 * (1.0 - f_y) + c10 * f_y
            c1 = c01 * (1.0 - f_y) + c11 * f_y

            # interpolate in time
            output_vectors[i, j, 0] = c0[0] * (1.0 - alpha) + c1[0] * alpha
            output_vectors[i, j, 1] = c0[1] * (1.0 - alpha) + c1[1] * alpha
            output_vectors[i, j, 2] = c0[2] * (1.0 - alpha) + c1[2] * alpha


@njit(parallel=True, fastmath=True)
def _interpolate_3d_vectorized_core_numba(
    field_lower, field_upper,
    x_idx, y_idx, z_idx, fx, fy, fz, alpha,
    output_vectors  # Pre-allocated (num_samples_z, num_samples_y, num_samples_x, 3)
):
    """Numba core for 3D vector field interpolation (trilinear in space, linear in time)."""
    num_samples_z, num_samples_y, num_samples_x = x_idx.shape
    for k in prange(num_samples_z):
        for i in range(num_samples_y):
            for j in range(num_samples_x):
                x_i = x_idx[k, i, j]
                y_i = y_idx[k, i, j]
                z_i = z_idx[k, i, j]
                f_x = fx[k, i, j]
                f_y = fy[k, i, j]
                f_z = fz[k, i, j]

                # 8 spatial corners, field indexed [Z, Y, X, C]
                v0000 = field_lower[z_i,     y_i,     x_i]
                v1000 = field_lower[z_i,     y_i,     x_i + 1]
                v0100 = field_lower[z_i,     y_i + 1, x_i]
                v1100 = field_lower[z_i,     y_i + 1, x_i + 1]
                v0010 = field_lower[z_i + 1, y_i,     x_i]
                v1010 = field_lower[z_i + 1, y_i,     x_i + 1]
                v0110 = field_lower[z_i + 1, y_i + 1, x_i]
                v1110 = field_lower[z_i + 1, y_i + 1, x_i + 1]
                v0001 = field_upper[z_i,     y_i,     x_i]
                v1001 = field_upper[z_i,     y_i,     x_i + 1]
                v0101 = field_upper[z_i,     y_i + 1, x_i]
                v1101 = field_upper[z_i,     y_i + 1, x_i + 1]
                v0011 = field_upper[z_i + 1, y_i,     x_i]
                v1011 = field_upper[z_i + 1, y_i,     x_i + 1]
                v0111 = field_upper[z_i + 1, y_i + 1, x_i]
                v1111 = field_upper[z_i + 1, y_i + 1, x_i + 1]

                # interpolate in x
                c000 = v0000 * (1.0 - f_x) + v1000 * f_x
                c010 = v0100 * (1.0 - f_x) + v1100 * f_x
                c100 = v0010 * (1.0 - f_x) + v1010 * f_x
                c110 = v0110 * (1.0 - f_x) + v1110 * f_x
                c001 = v0001 * (1.0 - f_x) + v1001 * f_x
                c011 = v0101 * (1.0 - f_x) + v1101 * f_x
                c101 = v0011 * (1.0 - f_x) + v1011 * f_x
                c111 = v0111 * (1.0 - f_x) + v1111 * f_x

                # interpolate in y
                c00 = c000 * (1.0 - f_y) + c010 * f_y
                c10 = c100 * (1.0 - f_y) + c110 * f_y
                c01 = c001 * (1.0 - f_y) + c011 * f_y
                c11 = c101 * (1.0 - f_y) + c111 * f_y

                # interpolate in z
                c0 = c00 * (1.0 - f_z) + c10 * f_z
                c1 = c01 * (1.0 - f_z) + c11 * f_z

                # interpolate in time
                output_vectors[k, i, j, 0] = c0[0] * (1.0 - alpha) + c1[0] * alpha
                output_vectors[k, i, j, 1] = c0[1] * (1.0 - alpha) + c1[1] * alpha
                output_vectors[k, i, j, 2] = c0[2] * (1.0 - alpha) + c1[2] * alpha


class VertexArrayVectorGlyph(VertexArrayObject):
    """Vector-field arrow glyphs rendered with GPU instancing.

    One arrow template mesh lives on the GPU; each frame we only (re)upload a small
    per-instance buffer of (position, vector) rows and issue a single
    glDrawArraysInstanced. Field sampling stays on the numba-accelerated cores below;
    there is no per-arrow CPU geometry baking anymore.
    """

    def __init__(self, name="vectorGlyph"):
        super().__init__(name)
        # Rebuild the per-instance buffer only when a parameter/time actually changes.
        self.dirty = True

        def dirtyCallBack(obj) -> None:
            obj.dirty = True
        self.create_variable_callback("draw", True, dirtyCallBack, True)
        self.create_variable_callback("scale", 1.0, dirtyCallBack, False)
        self.create_variable_callback("radius", 0.02, dirtyCallBack, False)
        self.create_variable_callback("height", 0.2, dirtyCallBack, False)
        self.create_variable_callback_with_gui_customization("sampling", 1.5, dirtyCallBack, False, {'widget': 'slider_float', 'min': 0.01, 'max': 5.0})
        self.create_variable_gui("color", (0.2, 0.6, 0.9), False, {'widget': 'color_picker'})
        # Feeds the fragment colour ramp; refreshed on every instance rebuild.
        self.create_variable("maxSpeed", 0.0, False, False)

        # GPU-instanced arrow material (one template mesh, N per-instance transforms).
        self.material = Material("glyphInstancedMaterial", "glyphInstancedMat")
        self.setMaterial(self.material)

        # ---- Instanced GL resources ----
        template = _build_arrow_template()
        self.template_vertex_count = int(template.shape[0])
        self.instance_count = 0
        self.g_vao = gl.glGenVertexArrays(1)
        self.g_template_vbo = gl.glGenBuffers(1)
        self.g_instance_vbo = gl.glGenBuffers(1)
        gl.glBindVertexArray(self.g_vao)
        # Template: aPos(loc0) + aNormal(loc1), stride 24 bytes, per-vertex (divisor 0).
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.g_template_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, template.nbytes, template, gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 24, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 24, ctypes.c_void_p(12))
        # Per-instance: iPos(loc2) + iVec(loc3), stride 24 bytes, per-instance (divisor 1).
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.g_instance_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, 0, None, gl.GL_DYNAMIC_DRAW)
        gl.glEnableVertexAttribArray(2)
        gl.glVertexAttribPointer(2, 3, gl.GL_FLOAT, gl.GL_FALSE, 24, ctypes.c_void_p(0))
        gl.glVertexAttribDivisor(2, 1)
        gl.glEnableVertexAttribArray(3)
        gl.glVertexAttribPointer(3, 3, gl.GL_FLOAT, gl.GL_FALSE, 24, ctypes.c_void_p(12))
        gl.glVertexAttribDivisor(3, 1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)

    def postInit(self):
        super().postInit()
        self.actFieldObject = self.parentScene.getObject("ActiveField") if self.parentScene.hasObject("ActiveField") else None

    def _vectorized_interpolation_2DVectorField(self, vector_field, time, sampling_distance):
        """Fully vectorized field interpolation for all sample points with 3-channel output"""
        # Calculate time interpolation
        time_idx = (time - vector_field.tmin) / vector_field.timeInterval if vector_field.timeInterval > 0 else 0
        lower_idx = int(np.floor(time_idx))
        upper_idx = int(np.ceil(time_idx))
        alpha = time_idx - lower_idx

        lower_idx = max(0, min(vector_field.time_steps - 1, lower_idx))
        upper_idx = max(0, min(vector_field.time_steps - 1, upper_idx))

        # Calculate sample grid
        num_samples_x = int((vector_field.domainMaxBoundary[0] - vector_field.domainMinBoundary[0]) / sampling_distance)
        num_samples_y = int((vector_field.domainMaxBoundary[1] - vector_field.domainMinBoundary[1]) / sampling_distance)

        # Create position grid
        x_positions = np.linspace(vector_field.domainMinBoundary[0],
                                  vector_field.domainMaxBoundary[0], num_samples_x)
        y_positions = np.linspace(vector_field.domainMinBoundary[1],
                                  vector_field.domainMaxBoundary[1], num_samples_y)

        # Create 2D position grids
        pos_x, pos_y = np.meshgrid(x_positions, y_positions, indexing='ij')

        # Convert to grid coordinates
        grid_x = (pos_x - vector_field.domainMinBoundary[0]) / vector_field.gridInterval[0]
        grid_y = (pos_y - vector_field.domainMinBoundary[1]) / vector_field.gridInterval[1]

        # Get integer indices
        x_idx = np.floor(grid_x).astype(int)
        y_idx = np.floor(grid_y).astype(int)

        # Calculate interpolation weights
        fx = grid_x - x_idx
        fy = grid_y - y_idx

        # Clip indices to valid range
        x_idx = np.clip(x_idx, 0, vector_field.Xdim - 2)
        y_idx = np.clip(y_idx, 0, vector_field.Ydim - 2)

        # Get field data for both time steps
        field_lower = vector_field.field[lower_idx]
        field_upper = vector_field.field[upper_idx]

        # Ensure 3D vectors by padding 2D data
        if field_lower.shape[-1] == 2:
            field_lower_3d = np.zeros((field_lower.shape[0], field_lower.shape[1], 3))
            field_lower_3d[..., :2] = field_lower
            field_upper_3d = np.zeros((field_upper.shape[0], field_upper.shape[1], 3))
            field_upper_3d[..., :2] = field_upper
        else:
            field_lower_3d = field_lower
            field_upper_3d = field_upper

        # Pre-allocate output and run the Numba core
        vectors = np.empty((num_samples_y, num_samples_x, 3), dtype=np.float32)
        _interpolate_2d_vectorized_core_numba(
            field_lower_3d, field_upper_3d,
            x_idx, y_idx, fx, fy, np.float32(alpha),
            vectors
        )

        # Create position grid - ensure 3D positions
        positions = np.stack([pos_x, pos_y, np.zeros_like(pos_x, dtype=np.float32)], axis=-1)
        vectors_reshaped = vectors.reshape(-1, 3)
        positions_reshaped = positions.reshape(-1, 3).astype(np.float32)
        return positions_reshaped, vectors_reshaped

    def _vectorized_interpolation_3DVectorField(self, vector_field, time, sampling_distance):
        """Fully vectorized field interpolation for all sample points in 3D."""
        # Calculate time interpolation
        if vector_field.timeInterval > 0:
            time_idx = (time - vector_field.tmin) / vector_field.timeInterval
        else:
            time_idx = 0.0
        lower_idx = int(np.floor(time_idx))
        upper_idx = int(np.ceil(time_idx))
        alpha = time_idx - lower_idx
        lower_idx = max(0, min(vector_field.time_steps - 1, lower_idx))
        upper_idx = max(0, min(vector_field.time_steps - 1, upper_idx))

        # Calculate sample grid
        num_samples_x = int((vector_field.domainMaxBoundary[0] - vector_field.domainMinBoundary[0]) / sampling_distance) if vector_field.domainMaxBoundary[0] > vector_field.domainMinBoundary[0] else 1
        num_samples_y = int((vector_field.domainMaxBoundary[1] - vector_field.domainMinBoundary[1]) / sampling_distance) if vector_field.domainMaxBoundary[1] > vector_field.domainMinBoundary[1] else 1
        num_samples_z = int((vector_field.domainMaxBoundary[2] - vector_field.domainMinBoundary[2]) / sampling_distance) if vector_field.domainMaxBoundary[2] > vector_field.domainMinBoundary[2] else 1

        if num_samples_x <= 0 or num_samples_y <= 0 or num_samples_z <= 0:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

        # Create position grid (ensure float32)
        x_positions = np.linspace(vector_field.domainMinBoundary[0], vector_field.domainMaxBoundary[0], num_samples_x, dtype=np.float32)
        y_positions = np.linspace(vector_field.domainMinBoundary[1], vector_field.domainMaxBoundary[1], num_samples_y, dtype=np.float32)
        z_positions = np.linspace(vector_field.domainMinBoundary[2], vector_field.domainMaxBoundary[2], num_samples_z, dtype=np.float32)
        pos_x, pos_y, pos_z = np.meshgrid(x_positions, y_positions, z_positions, indexing='ij')

        # Convert to grid coordinates
        grid_x = (pos_x - vector_field.domainMinBoundary[0]) / vector_field.gridInterval[0] if vector_field.gridInterval[0] != 0 else np.zeros_like(pos_x, dtype=np.float32)
        grid_y = (pos_y - vector_field.domainMinBoundary[1]) / vector_field.gridInterval[1] if vector_field.gridInterval[1] != 0 else np.zeros_like(pos_y, dtype=np.float32)
        grid_z = (pos_z - vector_field.domainMinBoundary[2]) / vector_field.gridInterval[2] if vector_field.gridInterval[2] != 0 else np.zeros_like(pos_z, dtype=np.float32)

        # Get integer indices
        x_idx = np.floor(grid_x).astype(np.int32)
        y_idx = np.floor(grid_y).astype(np.int32)
        z_idx = np.floor(grid_z).astype(np.int32)

        # Calculate interpolation weights (ensure float32)
        fx = (grid_x - x_idx).astype(np.float32)
        fy = (grid_y - y_idx).astype(np.float32)
        fz = (grid_z - z_idx).astype(np.float32)

        # Clip indices to valid range
        x_idx = np.clip(x_idx, 0, vector_field.Xdim - 2).astype(np.int32)
        y_idx = np.clip(y_idx, 0, vector_field.Ydim - 2).astype(np.int32)
        z_idx = np.clip(z_idx, 0, vector_field.Zdim - 2).astype(np.int32)

        # Get field data for both time steps (ensure numpy float32; field may be a torch tensor)
        field_lower = vector_field.field[lower_idx]
        field_upper = vector_field.field[upper_idx]
        if hasattr(field_lower, "detach"):
            field_lower = field_lower.detach().cpu().numpy()
            field_upper = field_upper.detach().cpu().numpy()
        field_lower = np.ascontiguousarray(field_lower, dtype=np.float32)
        field_upper = np.ascontiguousarray(field_upper, dtype=np.float32)

        # Pre-allocate output and run the Numba core
        vectors = np.empty((num_samples_z, num_samples_y, num_samples_x, 3), dtype=np.float32)
        _interpolate_3d_vectorized_core_numba(
            field_lower, field_upper,
            x_idx, y_idx, z_idx, fx, fy, fz, np.float32(alpha),
            vectors
        )

        # Reshape to (N, 3)
        positions = np.stack([pos_x, pos_y, pos_z], axis=-1)
        vectors_reshaped = vectors.reshape(-1, 3)
        positions_reshaped = positions.reshape(-1, 3).astype(np.float32)
        return positions_reshaped, vectors_reshaped

    def _rebuild_instances(self, vector_field, time):
        """Interpolate the field and upload one (pos, vec) row per sample to the
        instance VBO. No CPU geometry baking — the arrow mesh lives on the GPU."""
        sampling_distance = max(self.getValue("sampling"), 1e-4)
        if vector_field.getDim() == 2:
            positions, vectors = self._vectorized_interpolation_2DVectorField(vector_field, time, sampling_distance)
        else:
            positions, vectors = self._vectorized_interpolation_3DVectorField(vector_field, time, sampling_distance)

        if positions is None or len(positions) == 0:
            self.instance_count = 0
            self.setValue("maxSpeed", 0.0, callback=False)
            return

        speeds = np.linalg.norm(vectors, axis=1)
        mask = speeds > 1e-8
        n = int(np.count_nonzero(mask))
        if n == 0:
            self.instance_count = 0
            self.setValue("maxSpeed", 0.0, callback=False)
            return

        instance_data = np.empty((n, 6), dtype=np.float32)
        instance_data[:, 0:3] = positions[mask]
        instance_data[:, 3:6] = vectors[mask]

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.g_instance_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, instance_data.nbytes, instance_data, gl.GL_DYNAMIC_DRAW)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

        self.instance_count = n
        self.setValue("maxSpeed", float(speeds[mask].max()), callback=False)

    def render(self):
        if self.actFieldObject is None:
            return
        vector_field = self.actFieldObject.getActiveField()
        if vector_field is None:
            return
        if self.dirty:
            self._rebuild_instances(vector_field, self.actFieldObject.time())
            self.dirty = False
        if self.instance_count == 0:
            return
        if self.material is not None:
            self.material.apply([self.parentScene, self.cameraObject, self])
        gl.glBindVertexArray(self.g_vao)
        gl.glDrawArraysInstanced(gl.GL_TRIANGLES, 0, self.template_vertex_count, self.instance_count)
        gl.glBindVertexArray(0)
        gl.glUseProgram(0)
