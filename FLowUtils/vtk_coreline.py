

import vtk
import numpy as np
from .VectorField2d import UnsteadyVectorField2D
from .VectorField3d import UnsteadyVectorField3D







#implement the coreline of the vector field 2d and 3d






# Assuming your vector field classes are available
# from your_module import SteadyVectorField2D, UnsteadyVectorField2D, SteadyVectorField3D, UnsteadyVectorField3D

def create_vtk_image_data(field_data, domain_min, domain_max):
    """
    Creates a vtkImageData object from a NumPy array representing a vector field
    and its domain boundaries.

    Args:
        field_data (np.ndarray): The vector field data.
                                    Shape for 2D: (Ydim, Xdim, 2) or (Time, Ydim, Xdim, 2)
                                    Shape for 3D: (Zdim, Ydim, Xdim, 3) or (Time, Zdim, Ydim, Xdim, 3)
                                    Assumes the last dimension represents vector components (Vx, Vy[, Vz]).
        domain_min (list): [xmin, ymin(, zmin)] Minimum coordinates of the domain.
        domain_max (list): [xmax, ymax(, zmax)] Maximum coordinates of the domain.

    Returns:
        vtk.vtkImageData: The VTK image data object populated with the vector field.
    """
    field_data = np.asarray(field_data) # Ensure it's a NumPy array
    dims = field_data.shape
    
    # Determine if this is 3D data
    is_3d = dims[-1] == 3
    
    # Determine spatial dimensions
    if is_3d:
        if len(dims) == 4:  # (Z, Y, X, 3) - steady 3D
            nx, ny, nz = dims[2], dims[1], dims[0]
        else:  # (Time, Z, Y, X, 3) - unsteady 3D
            nx, ny, nz = dims[3], dims[2], dims[1]
    else:  # 2D
        if len(dims) == 3:  # (Y, X, 2) - steady 2D
            nx, ny = dims[1], dims[0]
            nz = 1
        else:  # (Time, Y, X, 2) - unsteady 2D
            nx, ny = dims[2], dims[1]
            nz = 1

    # Create vtkImageData
    image_data = vtk.vtkImageData()
    image_data.SetDimensions(nx, ny, nz)

    # Set spacing
    dx = (domain_max[0] - domain_min[0]) / max(1, nx - 1)
    dy = (domain_max[1] - domain_min[1]) / max(1, ny - 1)
    dz = (domain_max[2] - domain_min[2]) / max(1, nz - 1) if is_3d else 1.0
    image_data.SetSpacing(dx, dy, dz)

    # Set origin
    image_data.SetOrigin(domain_min[0], domain_min[1], domain_min[2] if is_3d else 0.0)

    # Flatten the field data for VTK
    if is_3d:
        vectors_flat = field_data.reshape(-1, 3)  # Shape: (nx*ny*nz, 3)
    else:
        # For 2D, pad with 0 for Z component to make it 3D for VTK
        vectors_flat_2d = field_data.reshape(-1, 2)  # Shape: (nx*ny, 2)
        vectors_flat = np.pad(vectors_flat_2d, ((0, 0), (0, 1)), mode='constant', constant_values=0)  # Shape: (nx*ny, 3)

    # Create VTK array for vectors
    vectors_vtk = vtk.vtkFloatArray()
    vectors_vtk.SetNumberOfComponents(3)
    vectors_vtk.SetNumberOfTuples(vectors_flat.shape[0])
    vectors_vtk.SetName("Vectors")  # Important: Name used by vtkStreamTracer

    # Copy data
    for i in range(vectors_flat.shape[0]):
        vectors_vtk.SetTuple3(i, vectors_flat[i, 0], vectors_flat[i, 1], vectors_flat[i, 2])

    # Assign the vector data to the image data
    image_data.GetPointData().SetVectors(vectors_vtk)

    return image_data





def _compute_gradient_2d(field):
    """Compute gradient of 2D field using central differences."""
    grad_y = np.zeros_like(field)
    grad_x = np.zeros_like(field)

    # Central differences for interior points
    if field.shape[0] > 2:
        grad_y[1:-1, :] = (field[2:, :] - field[:-2, :]) / 2.0
    if field.shape[1] > 2:
        grad_x[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / 2.0

    # Forward/backward differences for boundaries
    if field.shape[0] > 1:
        grad_y[0, :] = field[1, :] - field[0, :]
        grad_y[-1, :] = field[-1, :] - field[-2, :]
    if field.shape[1] > 1:
        grad_x[:, 0] = field[:, 1] - field[:, 0]
        grad_x[:, -1] = field[:, -1] - field[:, -2]

    return grad_x, grad_y

def _detect_critical_points(u, v, method, threshold):
    """Detect critical points using specified method. Returns list of (x_idx, y_idx) tuples."""
    if not u.size or not v.size:
        return []

    if method == 'simple':
        # Simple zero velocity detection
        mask = (np.abs(u) < threshold) & (np.abs(v) < threshold)
        y_idx, x_idx = np.where(mask)
        return list(zip(x_idx, y_idx)) # Return as (x, y) index tuples
    elif method == 'ivd':
        du_dx, du_dy = _compute_gradient_2d(u)
        dv_dx, dv_dy = _compute_gradient_2d(v)

        # Compute vorticity (ivd here)
        curl_norm =  np.sqrt((dv_dx - du_dy)**2)

        # Subtract average vorticity
        avg_curl_norm = np.mean(curl_norm)
        ivd = curl_norm - avg_curl_norm

        # Generate mask where ivd > threshold
        vel_mag = np.sqrt(u**2 + v**2)
        mask = (ivd > 0.1*threshold) & (vel_mag < threshold)
        y_idx, x_idx = np.where(mask)
        return list(zip(x_idx, y_idx)) # Return as (x, y) index tuples
    elif method == 'jacobian':
        # Jacobian-based detection
        du_dx, du_dy = _compute_gradient_2d(u)
        dv_dx, dv_dy = _compute_gradient_2d(v)

        # Compute vorticity (out-of-plane component)
        omega = dv_dx - du_dy

        # Find points where velocity is near zero and vorticity is positive
        vel_mag = np.sqrt(u**2 + v**2)
        # Handle potential division by zero or invalid values
        mask = (vel_mag < threshold) & (omega > 0)
        # Ensure no NaNs affect the mask
        mask = mask & np.isfinite(omega) & np.isfinite(vel_mag)

        y_idx, x_idx = np.where(mask)
        return list(zip(x_idx, y_idx)) # Return as (x, y) index tuples

    elif method == 'q_criterion':
        # Q-criterion for 2D
        du_dx, du_dy = _compute_gradient_2d(u)
        dv_dx, dv_dy = _compute_gradient_2d(v)

        # Compute Q-criterion for 2D
        # Q = 0.5 * (Omega_ij * Omega_ij - S_ij * S_ij)
        # For 2D: Omega12 = 0.5 * (dv_dx - du_dy), S11 = du_dx, S22 = dv_dy, S12 = 0.5 * (du_dy + dv_dx)
        omega12 = 0.5 * (dv_dx - du_dy)
        s11 = du_dx
        s22 = dv_dy
        s12 = 0.5 * (du_dy + dv_dx)

        Q = 0.5 * (omega12**2 - (s11**2 + s22**2 + 2*s12**2))

        # Find points where Q > 0 and velocity is small
        vel_mag = np.sqrt(u**2 + v**2)
        # Handle potential division by zero or invalid values
        mask = (Q > 0) & (vel_mag < threshold)
        # Ensure no NaNs affect the mask
        mask = mask & np.isfinite(Q) & np.isfinite(vel_mag)

        y_idx, x_idx = np.where(mask)
        return list(zip(x_idx, y_idx)) # Return as (x, y) index tuples

    else:
        raise ValueError(f"Unknown method: {method}")

def _euclidean_distance_2d(point1, point2):
    """Calculate Euclidean distance between two 2D points (x, y)."""
    if len(point1) < 2 or len(point2) < 2:
        raise ValueError("Points must have at least 2 coordinates (x, y)")
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def _track_critical_points_across_time(points_per_time, max_distance, min_line_length):
    """
    Track critical points across time steps to form continuous corelines.
    This implementation tracks both forward and backward from each seed point
    to build complete, non-fragmented corelines.

    points_per_time: List[List[Tuple[x, y, t]]] - Physical coordinates
    """
    if not points_per_time:
        return []

    all_lines = []
    T = len(points_per_time)

    # A mutable copy of points to track availability.
    available_points = [list(points) for points in points_per_time]

    def _find_closest_point(point, candidates):
        """Helper to find the closest point in a list of candidates."""
        min_dist = float('inf')
        closest_point = None
        current_pos = point[:2]
        for candidate in candidates:
            try:
                dist = _euclidean_distance_2d(current_pos, candidate[:2])
                if dist < min_dist:
                    min_dist = dist
                    closest_point = candidate
            except (ValueError, IndexError):
                continue
        return closest_point, min_dist

    for t in range(T):
        # Iterate over a copy, as we modify the original list.
        for seed_point in list(available_points[t]):
            if seed_point not in available_points[t]:
                continue  # Already used in another line

            # --- Start a new line from this seed point ---
            available_points[t].remove(seed_point)

            # --- Track Forward ---
            forward_path = []
            current_point = seed_point
            for fwd_t in range(t, T - 1):
                closest, dist = _find_closest_point(current_point, available_points[fwd_t + 1])
                if closest and dist <= max_distance:
                    forward_path.append(closest)
                    available_points[fwd_t + 1].remove(closest)
                    current_point = closest
                else:
                    break  # End of forward path

            # --- Track Backward ---
            backward_path = []
            current_point = seed_point
            for bwd_t in range(t, 0, -1):
                closest, dist = _find_closest_point(current_point, available_points[bwd_t - 1])
                if closest and dist <= max_distance:
                    backward_path.append(closest)
                    available_points[bwd_t - 1].remove(closest)
                    current_point = closest
                else:
                    break  # End of backward path

            # --- Combine and Store Line ---
            # backward_path is [t-1, t-2, ...], so reverse it.
            full_line = list(reversed(backward_path)) + [seed_point] + forward_path
            if len(full_line) >= min_line_length:
                all_lines.append(full_line)

    return all_lines



def extract_vortex_corelines_2d_unsteady(vector_field, method='jacobian', threshold=1e-1,
                                       max_distance=None, min_line_length=4):
    """
    Extract vortex corelines from an unsteady 2D vector field as a list of 3D curves.

    This function detects critical points (vortex centers) at each time step and
    connects them across time to form 3D corelines in spacetime (x, y, t).

    Args:
        vector_field (UnsteadyVectorField2D): The input unsteady 2D vector field.
        method (str): Method for critical point detection ('jacobian', 'q_criterion', 'simple')
        threshold (float): Threshold for detecting critical points
        max_distance (float): Maximum distance for tracking points across time steps
        min_line_length (int): Minimum number of points required for a valid coreline

    Returns:
        list[np.ndarray]: A list where each element is an Nx3 numpy array representing
                          a coreline with columns [x, y, t].
    """

    # --- Input Validation based on IDiscreteField2D structure ---
    required_attrs = ['getDataAsNumpy', 'Xdim', 'Ydim', 'time_steps', 'domainMinBoundary', 'domainMaxBoundary', 'tmin', 'tmax']
    for attr in required_attrs:
        if not hasattr(vector_field, attr):
            raise ValueError(f"Vector field missing required attribute: {attr}")

    data = vector_field.getDataAsNumpy()
    if data is None or len(data.shape) != 4 or data.shape[3] != 2:
        raise ValueError("Expected 4D field data with shape (T, Y, X, 2)")

    T = vector_field.time_steps
    Y = vector_field.Ydim
    X = vector_field.Xdim

    # Validate dimensions match data
    if not (data.shape[0] == T and data.shape[1] == Y and data.shape[2] == X):
        raise ValueError("Field dimensions do not match data shape.")

    # Set default max_distance if not provided
    if max_distance is None:
        rangeX =(vector_field.domainMaxBoundary[0] - vector_field.domainMinBoundary[0]) 
        rangeY = (vector_field.domainMaxBoundary[1] - vector_field.domainMinBoundary[1]) 
        max_distance =  0.1*max(rangeX, rangeY)
        min_distance =  0.01*min(rangeX, rangeY)

    # Collect critical points for each time step
    all_core_points_per_time = []

    for t in range(T):
        u = data[t, :, :, 0]
        v = data[t, :, :, 1]

        # Start with initial threshold and gradually decrease until points are found
        critical_points_grid = []
        # If  no points found, try increasing threshold-> automatically update threshold to make sure at least one point per timestep
        current_threshold = threshold
        critical_points_grid = _detect_critical_points(u, v, method, current_threshold)
        while len(critical_points_grid) == 0 and current_threshold <= 1.0:
            current_threshold *= 2
            critical_points_grid = _detect_critical_points(u, v, method, current_threshold)

        # Convert to physical coordinates including time
        physical_coords = []
        for x_idx, y_idx in critical_points_grid:
            # Convert grid indices to physical x, y
            x_phys, y_phys = vector_field.convert_grid_pos_2_physical_pos(float(x_idx), float(y_idx))
            has_added_before=False

            #remove the points that are too close to each other,pick the one with the lowest velocity
            for i in range(len(physical_coords)):
                if _euclidean_distance_2d(physical_coords[i], (x_phys, y_phys)) <= min_distance :
                    has_added_before=True
                    #if the new point has lower velocity, replace the old point
                    if  np.linalg.norm(vector_field.get_vector_at_grid(x_idx,y_idx,t)) < np.linalg.norm(vector_field.get_vector(physical_coords[i][0],physical_coords[i][1],t_phys)):
                        t_phys = vector_field.getPhysicalTime(t)
                        normalized_t_phys = (t_phys - vector_field.tmin) / (vector_field.tmax - vector_field.tmin)
                        physical_coords[i]=(x_phys, y_phys, normalized_t_phys)
                    break
                    

            if not has_added_before:
                t_phys = vector_field.getPhysicalTime(t)
                normalized_t_phys = (t_phys - vector_field.tmin) / (vector_field.tmax - vector_field.tmin)
                physical_coords.append((x_phys, y_phys, normalized_t_phys))



        if len(physical_coords) == 0:
            print(f"Warning: No critical points found at time step {t} even after threshold adjustment")
        all_core_points_per_time.append(physical_coords)

    # Check if any critical points were found
    total_points = sum(len(points) for points in all_core_points_per_time)
    if total_points < T-1:
        print("Warning: Very few critical points detected across all time steps")


 

    # Track critical points across time to form corelines
    tracked_lines = _track_critical_points_across_time(
        all_core_points_per_time, max_distance, min_line_length
    )

    # --- Convert tracked lines to list of Nx3 numpy arrays ---
    curves_list = []
    for line in tracked_lines:
        if len(line) >= 2: # Ensure we have at least a point pair
            # Convert list of tuples [(x1,y1,t1), ...] to Nx3 numpy array
            curve_array = np.array(line, dtype=np.float32) # Shape (N, 3)
            curves_list.append(curve_array)

    return curves_list # This is the format directly usable by add_curve








def extract_corelines_3d_unsteady(vector_field, seed_points, time_slice):
    """
    Extracts corelines from an unsteady 3D vector field at a specific time slice.

    Args:
        vector_field (UnsteadyVectorField3D): The input unsteady 3D vector field.
        seed_points (list of tuples): List of (x, y, z) seed point coordinates.
        time_slice (int): The time slice to extract corelines from.

    Returns:
        vtk.vtkPolyData: The extracted streamlines at the specified time.
    """
    # --- 1. Prepare VTK Data ---
    if not hasattr(vector_field, 'getDataAsNumpy'):
        raise ValueError("Vector field must have getDataAsNumpy method")
    
    field_data = vector_field.getDataAsNumpy()
    
    if len(field_data.shape) != 5:  # (Time, Z, Y, X, 3)
        raise ValueError("Expected 5D field data for unsteady 3D field")
    
    # Extract the specific time slice
    field_data_slice = field_data[time_slice]  # Shape: (Z, Y, X, 3)
    
    vtk_data = create_vtk_image_data(
        field_data_slice,
        vector_field.domainMinBoundary,
        vector_field.domainMaxBoundary
    )

    # --- 2. Setup VTK Streamline Pipeline ---
    # Seed points
    seed_points_vtk = vtk.vtkPoints()
    for point in seed_points:
        seed_points_vtk.InsertNextPoint(point[0], point[1], point[2])

    seed_polydata = vtk.vtkPolyData()
    seed_polydata.SetPoints(seed_points_vtk)

    # Streamline tracer
    stream_tracer = vtk.vtkStreamTracer()
    stream_tracer.SetInputData(vtk_data)
    stream_tracer.SetSourceData(seed_polydata)
    stream_tracer.SetIntegrationDirectionToBoth()
    stream_tracer.SetMaximumPropagation(200.0)
    stream_tracer.SetInitialIntegrationStep(0.05)
    stream_tracer.SetIntegratorTypeToRungeKutta45()
    stream_tracer.SetIntegrationStepUnit(2)  # CELL_LENGTH_UNIT

    # --- 3. Execute and Return ---
    stream_tracer.Update()
    streamlines = stream_tracer.GetOutput()

    return streamlines

def vtk_polydata_to_points(vtk_polydata):
    """
    Converts VTK polydata streamlines to a numpy array of points.
    
    Args:
        vtk_polydata (vtk.vtkPolyData): VTK polydata containing streamlines
        
    Returns:
        np.ndarray: Nx3 array of points representing all streamlines concatenated
    """
    if vtk_polydata.GetNumberOfPoints() == 0:
        return np.array([], dtype=np.float32).reshape(0, 3)
    
    # Get all points from the polydata
    points = vtk_polydata.GetPoints()
    num_points = points.GetNumberOfPoints()
    
    # Convert to numpy array
    points_array = np.zeros((num_points, 3), dtype=np.float32)
    for i in range(num_points):
        point = points.GetPoint(i)
        points_array[i] = point
    
    return points_array


def vtk_polydata_to_separate_curves(vtk_polydata):
    """
    Converts VTK polydata streamlines to a list of separate curves.
    
    Args:
        vtk_polydata (vtk.vtkPolyData): VTK polydata containing streamlines
        
    Returns:
        list: List of numpy arrays, each representing a separate streamline curve
    """
    if vtk_polydata.GetNumberOfCells() == 0:
        return []
    
    curves = []
    points = vtk_polydata.GetPoints()
    
    # Iterate through each cell (line) in the polydata
    for i in range(vtk_polydata.GetNumberOfCells()):
        cell = vtk_polydata.GetCell(i)
        if cell.GetCellType() == vtk.VTK_LINE:
            # Get point IDs for this line
            point_ids = cell.GetPointIds()
            num_points_in_line = point_ids.GetNumberOfIds()
            
            # Extract points for this line
            line_points = np.zeros((num_points_in_line, 3), dtype=np.float32)
            for j in range(num_points_in_line):
                point_id = point_ids.GetId(j)
                point = points.GetPoint(point_id)
                line_points[j] = point
            
            curves.append(line_points)
    
    return curves


# --- Example Usage for Unsteady Vector Fields ---

# --- 2D Unsteady Examples ---
# my_unsteady_2d_field = UnsteadyVectorField2D(Xdim=50, Ydim=50, domainMinBoundary=[-1, -1], domainMaxBoundary=[1, 1], time_steps=100, tmin=0, tmax=2*np.pi)
# # ... populate my_unsteady_2d_field.field with data ...
# seeds_2d = [(0, 0), (1, 1), (-1, 0.5)] # List of (x,y) tuples

# Extract streamlines at a specific time slice
# streamlines_at_time = extract_corelines_2d_unsteady(my_unsteady_2d_field, seeds_2d, time_slice=50)

# Extract spacetime corelines (3D trajectories through space and time)
# Option 1: With specific seed points
# spacetime_corelines = extract_corelines_2d_unsteady_spacetime(my_unsteady_2d_field, seeds_2d)

# Option 2: Without seed points (uses uniform grid)
# spacetime_corelines = extract_corelines_2d_unsteady_spacetime(my_unsteady_2d_field)
# This gives you 3D curves where z-axis represents time

# --- 3D Unsteady Examples ---
# my_unsteady_3d_field = UnsteadyVectorField3D(Xdim=30, Ydim=30, Zdim=30, time_steps=100, tmin=0, tmax=2*np.pi, ...)
# # ... populate my_unsteady_3d_field.field with data ...
# seeds_3d = [(0, 0, 0), (1, 1, 1)] # List of (x,y,z) tuples

# Extract streamlines at a specific time slice
# streamlines_at_time = extract_corelines_3d_unsteady(my_unsteady_3d_field, seeds_3d, time_slice=50)

