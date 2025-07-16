from .VertexArrayObject import *

class VertexArrayVectorGlyph(VertexArrayObject):
    def __init__(self, name="vectorGlyph"):
        super().__init__(name)
        # Performance optimization flags
        self.dirty = True    

        def dirtyCallBack(obj) -> None:
            obj.dirty=True
        self.create_variable_callback("draw",True,dirtyCallBack,True)
        self.create_variable_callback("scale",1.0,dirtyCallBack,False)
        self.create_variable_callback("segments",10,dirtyCallBack,False)
        self.create_variable_callback("radius",0.01,dirtyCallBack,False)
        self.create_variable_callback("height",0.1,dirtyCallBack,False)
        self.create_variable_callback_with_gui_customization("sampling",0.5,dirtyCallBack,False, {'widget': 'slider_float', 'min': 0.0, 'max': 1.0})
        self.create_variable_gui("color",(0.2,-.2,0.2),False,{'widget': 'color_picker'})

        
        
    
    def _vectorized_interpolation_2DVectorField(self, vector_field, time, sampling_distance):
        """Fully vectorized field interpolation for all sample points with 3-channel output"""
        # Calculate time interpolation
        time_idx = (time - vector_field.tmin) / vector_field.timeInterval
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
            # Pad 2D vectors with zeros to make them 3D
            field_lower_3d = np.zeros((field_lower.shape[0], field_lower.shape[1], 3))
            field_lower_3d[..., :2] = field_lower
            field_upper_3d = np.zeros((field_upper.shape[0], field_upper.shape[1], 3))
            field_upper_3d[..., :2] = field_upper
        else:
            field_lower_3d = field_lower
            field_upper_3d = field_upper
        
        # Vectorized trilinear interpolation
        # Get all 8 corner values for each sample point
        v000 = field_lower_3d[y_idx, x_idx]  # (num_samples_y, num_samples_x, 3)
        v100 = field_lower_3d[y_idx, x_idx + 1]
        v010 = field_lower_3d[y_idx + 1, x_idx]
        v110 = field_lower_3d[y_idx + 1, x_idx + 1]
        
        v001 = field_upper_3d[y_idx, x_idx]
        v101 = field_upper_3d[y_idx, x_idx + 1]
        v011 = field_upper_3d[y_idx + 1, x_idx]
        v111 = field_upper_3d[y_idx + 1, x_idx + 1]
        
        # Reshape interpolation weights for broadcasting
        fx_reshaped = fx[..., np.newaxis]  # (num_samples_y, num_samples_x, 1)
        fy_reshaped = fy[..., np.newaxis]  # (num_samples_y, num_samples_x, 1)
        
        # Trilinear interpolation (fully vectorized)
        # Interpolate in x direction for both time steps
        c00 = v000 * (1 - fx_reshaped) + v100 * fx_reshaped
        c10 = v010 * (1 - fx_reshaped) + v110 * fx_reshaped
        c01 = v001 * (1 - fx_reshaped) + v101 * fx_reshaped
        c11 = v011 * (1 - fx_reshaped) + v111 * fx_reshaped
        
        # Interpolate in y direction for both time steps
        c0 = c00 * (1 - fy_reshaped) + c10 * fy_reshaped
        c1 = c01 * (1 - fy_reshaped) + c11 * fy_reshaped
        
        # Interpolate in time direction
        vectors = c0 * (1 - alpha) + c1 * alpha  # (num_samples_y, num_samples_x, 3)
        
        # Create position grid - ensure 3D positions
        positions = np.stack([pos_x, pos_y, np.zeros_like(pos_x)], axis=-1)  # (num_samples_y, num_samples_x, 3)
        
        return positions, vectors
    
    def _batch_generate_glyphs(self, positions, vectors, scale, radius, height, segments):
        """Batch generate glyph geometry using direct arrow generation"""
        # Filter out zero vectors
        vector_magnitudes = np.linalg.norm(vectors, axis=-1)
        valid_mask = vector_magnitudes > 1e-6
        
        if not np.any(valid_mask):
            return
            
        valid_positions = positions[valid_mask]
        valid_vectors = vectors[valid_mask]
        valid_magnitudes = vector_magnitudes[valid_mask]
        
        # Pre-allocate arrays for batch processing
        all_vertices = []
        all_indices = []
        all_tex_coords = []
        
        # Generate arrows directly for each valid position
        for i in range(len(valid_positions)):
            pos = valid_positions[i]
            direction = valid_vectors[i]
            magnitude = valid_magnitudes[i]
            
            # Scale the arrow by magnitude
            scaled_direction = direction * scale
            scaled_height = height * magnitude * scale
            
            # Generate arrow geometry directly
            arrow_vertices, arrow_indices, arrow_tex_coords = self._generate_single_arrow(
                pos, scaled_direction, radius, scaled_height, segments
            )
            
            # Add to batch with proper index offset
            vertex_offset = len(all_vertices) // 3
            all_vertices.extend(arrow_vertices)
            all_indices.extend([idx + vertex_offset for idx in arrow_indices])
            all_tex_coords.extend(arrow_tex_coords)
        
        return all_vertices, all_indices, all_tex_coords
    
    def _generate_single_arrow(self, base_pos, direction, radius, height, segments):
        """Generate geometry for a single arrow using cone-like approach similar to appendConeWithoutCommit"""
        # Normalize direction
        direction = direction / np.linalg.norm(direction)
        
        # Calculate velocity magnitude for scaling
        velocity_mag = np.linalg.norm(direction)
        direction = direction / np.linalg.norm(direction)
        startPos = base_pos - direction * height * 0.5

        # Generating orthogonal vectors (same as appendConeWithoutCommit)
        orth0 = np.array([direction[1], -direction[0], direction[2]])
        if orth0[0]*orth0[0] + orth0[1]*orth0[1] < 0.001:
            orth0 = np.array([direction[2], direction[1], direction[0]])
        orth0 = orth0 / np.linalg.norm(orth0)

        orth1 = np.cross(direction, orth0)
        orth1 = orth1 / np.linalg.norm(orth1)
        orth2 = np.cross(direction, orth1)
        orth2 = orth2 / np.linalg.norm(orth2)

        # Generate cone vertices (similar to appendConeWithoutCommit)
        vTop = startPos + direction * height * velocity_mag
        lastV = orth2
        textureCoords = []
        temporaryVertex = [None] * 9 * segments
        
        for i in range(1, segments):
            angle = i * 2 * np.pi / segments
            s = np.sin(angle)
            c = np.cos(angle)
            v = s * orth1 + c * orth2
            vBottom = startPos + v * radius
            lastVBottom = startPos + lastV * radius
          
            lastV = v
            temporaryVertex[(i-1)*9:(i*9)] = [
                lastVBottom[0], lastVBottom[1], lastVBottom[2], 
                vBottom[0], vBottom[1], vBottom[2], 
                vTop[0], vTop[1], vTop[2]
            ]

        # Handle the last segment
        v = orth2
        vBottom = startPos + v * radius
        lastVBottom = startPos + lastV * radius
        temporaryVertex = temporaryVertex[0:9*segments]
        temporaryVertex[-9:] = [
            lastVBottom[0], lastVBottom[1], lastVBottom[2], 
            vBottom[0], vBottom[1], vBottom[2], 
            vTop[0], vTop[1], vTop[2]
        ]
        temporaryIndex = list(range(0, 3*segments))

        # Generate circle base (similar to appendCircleWithoutCommit)
        circle_vertices, circle_indices, circle_tex_coords = self._generate_circle_geometry(
            startPos, direction, radius, segments
        )
        
        # Combine cone and circle geometry
        all_vertices = temporaryVertex + circle_vertices
        all_indices = temporaryIndex + [idx + len(temporaryVertex)//3 for idx in circle_indices]
        all_tex_coords = textureCoords + circle_tex_coords
        
        return all_vertices, all_indices, all_tex_coords
    

    def updateVectorGlyphOptimized(self, vector_field, time: float = 0.0):
        """Optimized version using vectorized operations and geometry instancing"""
        if vector_field is None or self.dirty == False:
            return
            
        # Get parameters
        radius = self.getValue("radius")
        height = self.getValue("height")
        segments = self.getValue("segments")
        scale = self.getValue("scale")
        sampling_distance = max(self.getValue("sampling"), 0.0001)
        
        # Vectorized interpolation
        positions, vectors = self._vectorized_interpolation_2DVectorField(vector_field, time, sampling_distance)
        
        # Batch generate glyphs
        result = self._batch_generate_glyphs(positions, vectors, scale, radius, height, segments)
        
        if result is None:
            self.erase()
            return
            
        all_vertices, all_indices, all_tex_coords = result
        
        # Update geometry
        self.erase()
        self.appendVertexGeometryNoCommit(all_vertices, all_indices, all_tex_coords)
        self.commit()
        self.dirty = False
    
    def render(self):
        if self.dirty == True:
            actFieldWidget = self.parentScene.getObject("ActiveField")
            # Use optimized version
            self.updateVectorGlyphOptimized(actFieldWidget.getActiveField(), actFieldWidget.time())
        super().render()
        return 

    def updateVectorGlyph(self,vector_field, time: float=0.0):
        """
        Draw vector glyphs representing a vector field interpolated between two time steps.

        :param vector_field: A VectorField2D object representing the vector field.
        :param time: The specific time to interpolate the vector field at.
        :param position: The position where to start drawing the vector field.
        :param scale: Scale factor for drawing glyphs.
        """
        if vector_field is None or self.dirty==False:
            return
    
        # Calculate the interpolation index
        time_idx = (time - vector_field.tmin) / vector_field.timeInterval
        lower_idx = int(np.floor(time_idx))
        upper_idx = int(np.ceil(time_idx))
        alpha = time_idx - lower_idx

        # Ensure indices are within the bounds of the vector field time steps
        lower_idx = max(0, min(vector_field.time_steps - 1, lower_idx))
        upper_idx = max(0, min(vector_field.time_steps - 1, upper_idx))

     
        radius=self.getValue("radius")
        hight=self.getValue("height")
        segments=self.getValue("segments")
        scale=self.getValue("scale")
        sampling_distance=max(self.getValue("sampling"),0.0001)
        # Calculate number of samples in each direction
        num_samples_x = int((vector_field.domainMaxBoundary[0]-vector_field.domainMinBoundary[0]) / sampling_distance)
        num_samples_y = int((vector_field.domainMaxBoundary[1]-vector_field.domainMinBoundary[1]) / sampling_distance)
        
        self.erase()
        for y in range(num_samples_y):
            for x in range(num_samples_x):
                # Calculate actual position
                posX = vector_field.domainMinBoundary[0] + x * sampling_distance
                posY = vector_field.domainMinBoundary[1] + y * sampling_distance
                # Convert position to grid coordinates for interpolation
                grid_x = (posX - vector_field.domainMinBoundary[0]) / vector_field.gridInterval[0]
                grid_y = (posY - vector_field.domainMinBoundary[1]) / vector_field.gridInterval[1]
                
                # Get interpolated vector at this position
                x_idx = int(grid_x)
                y_idx = int(grid_y)
                
                # Skip if outside field bounds
                if (x_idx >= vector_field.Xdim - 1 or 
                    y_idx >= vector_field.Ydim - 1 or 
                    x_idx < 0 or y_idx < 0):
                    continue
                
                # Bilinear interpolation weights
                fx = grid_x - x_idx
                fy = grid_y - y_idx
                
                # Get vectors at surrounding grid points for both time steps
                v000 = vector_field.data[lower_idx][y_idx, x_idx]
                v100 = vector_field.data[lower_idx][y_idx, x_idx + 1] 
                v010 = vector_field.data[lower_idx][y_idx + 1, x_idx]
                v110 = vector_field.data[lower_idx][y_idx + 1, x_idx + 1]
                
                v001 = vector_field.data[upper_idx][y_idx, x_idx]
                v101 = vector_field.data[upper_idx][y_idx, x_idx + 1]
                v011 = vector_field.data[upper_idx][y_idx + 1, x_idx]
                v111 = vector_field.data[upper_idx][y_idx + 1, x_idx + 1]
                
                # Trilinear interpolation
                c00 = v000 * (1 - fx) + v100 * fx
                c10 = v010 * (1 - fx) + v110 * fx
                c01 = v001 * (1 - fx) + v101 * fx
                c11 = v011 * (1 - fx) + v111 * fx
                
                c0 = c00 * (1 - fy) + c10 * fy
                c1 = c01 * (1 - fy) + c11 * fy
                
                v = c0 * (1 - alpha) + c1 * alpha
                
                vx = v[0]
                vy = v[1]
                vx=vx*scale
                vy=vy*scale
                direction = np.array([vx, vy, 0.0], dtype=np.float32)
                self.appendConeWithoutCommit(np.array([posX, posY, 0.0], dtype=np.float32),
                                          direction, radius, hight, segments)
        self.commit()
        self.dirty=False
          
    