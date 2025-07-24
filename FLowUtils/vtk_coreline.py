from .VectorField2d import UnsteadyVectorField2D
from .VectorField3d import UnsteadyVectorField3D









#implement the coreline of the vector field 2d and 3d






import vtk
import numpy as np
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
    is_3d = len(dims) in [4, 5] and dims[-1] == 3
    is_time_dependent = len(dims) in [4, 5] and dims[0] > 1 # Simplified check

    # Determine spatial dimensions
    if is_3d:
        if is_time_dependent:
            nx, ny, nz = dims[2], dims[3], dims[1] # (Time, Z, Y, X, 3)
        else:
            nx, ny, nz = dims[1], dims[2], dims[0] # (Z, Y, X, 3)
    else: # 2D
        if is_time_dependent:
            nx, ny = dims[2], dims[1] # (Time, Y, X, 2)
            nz = 1
        else:
            nx, ny = dims[1], dims[0] # (Y, X, 2)
            nz = 1

    # Create vtkImageData
    image_data = vtk.vtkImageData()
    image_data.SetDimensions(nx, ny, nz)

    # Set spacing
    dx = (domain_max[0] - domain_min[0]) / max(1, nx - 1)
    dy = (domain_max[1] - domain_min[1]) / max(1, ny - 1)
    dz = (domain_max[2] - domain_min[2]) / max(1, nz - 1) if is_3d else 1.0
    image_data.SetSpacing(dx, dy, dz)

    # Set origin (assuming node-centered data at domain_min)
    image_data.SetOrigin(domain_min[0], domain_min[1], domain_min[2] if is_3d else 0.0)

    # Flatten the spatial part of the field data for VTK
    # VTK expects data in (point/cell, component) format
    if is_time_dependent:
       # For time-dependent, we work with a single time slice (index 0 here)
       # You might want to pass the specific time slice's data instead
       if is_3d:
           vectors_flat = field_data[0].reshape(-1, 3) # Shape: (nx*ny*nz, 3)
       else:
           # Pad 2D vectors with 0 for Z component for VTK
           vectors_flat_2d = field_data[0].reshape(-1, 2) # Shape: (nx*ny, 2)
           vectors_flat = np.pad(vectors_flat_2d, ((0, 0), (0, 1)), mode='constant', constant_values=0) # Shape: (nx*ny, 3)
    else: # Steady field
        if is_3d:
            vectors_flat = field_data.reshape(-1, 3) # Shape: (nx*ny*nz, 3)
        else:
             # Pad 2D vectors with 0 for Z component for VTK
            vectors_flat_2d = field_data.reshape(-1, 2) # Shape: (nx*ny, 2)
            vectors_flat = np.pad(vectors_flat_2d, ((0, 0), (0, 1)), mode='constant', constant_values=0) # Shape: (nx*ny, 3)

    # Create VTK array for vectors
    vectors_vtk = vtk.vtkFloatArray()
    vectors_vtk.SetNumberOfComponents(3)
    vectors_vtk.SetNumberOfTuples(vectors_flat.shape[0])
    vectors_vtk.SetName("Vectors") # Important: Name used by vtkStreamTracer

    # Copy data
    for i in range(vectors_flat.shape[0]):
        vectors_vtk.SetTuple3(i, vectors_flat[i, 0], vectors_flat[i, 1], vectors_flat[i, 2])

    # Assign the vector data to the image data
    image_data.GetPointData().SetVectors(vectors_vtk)

    return image_data

def extract_corelines_2d(vector_field, seed_points):
    """
    Extracts streamlines (corelines) from a 2D vector field using VTK.

    Args:
        vector_field (SteadyVectorField2D or UnsteadyVectorField2D): The input 2D vector field.
        seed_points (list of tuples): List of (x, y) seed point coordinates.

    Returns:
        vtk.vtkPolyData: The extracted streamlines.
    """
    # --- 1. Prepare VTK Data ---
    # Get the field data (ensure it's NumPy)
    if hasattr(vector_field, 'getDataAsNumpy'):
        field_data = vector_field.getDataAsNumpy()
    else:
        field_data = np.asarray(vector_field.field)

    # Handle time-dependent fields: Get the first time slice or a specific one
    if len(field_data.shape) == 4: # (Time, Y, X, 2)
        # Example: Use the first time slice. You might want to make this configurable.
        field_data = field_data[0]

    vtk_data = create_vtk_image_data(
        field_data,
        vector_field.domainMinBoundary,
        vector_field.domainMaxBoundary
    )

    # --- 2. Setup VTK Streamline Pipeline ---
    # Seed points
    seed_points_vtk = vtk.vtkPoints()
    for point in seed_points:
        # Add z=0 for 2D points in VTK
        seed_points_vtk.InsertNextPoint(point[0], point[1], 0.0)

    seed_polydata = vtk.vtkPolyData()
    seed_polydata.SetPoints(seed_points_vtk)

    # Streamline tracer
    stream_tracer = vtk.vtkStreamTracer()
    stream_tracer.SetInputData(vtk_data)
    stream_tracer.SetSourceData(seed_polydata)
    stream_tracer.SetIntegrationDirectionToForward() # or BACKWARD or BOTH
    stream_tracer.SetIntegratorTypeToRungeKutta45() # Integration method
    stream_tracer.SetMaximumPropagation(100.0) # Adjust as needed
    stream_tracer.SetInitialIntegrationStep(0.01) # Adjust as needed
    stream_tracer.SetIntegrationStepUnit(2) # CELL_LENGTH_UNIT
    # stream_tracer.SetTerminalSpeed(1e-12) # Stop if speed is too low
    # stream_tracer.SetMaximumNumberOfSteps(1000) # Limit steps

    # --- 3. Execute and Return ---
    stream_tracer.Update()
    streamlines = stream_tracer.GetOutput()

    return streamlines


def extract_corelines_3d(vector_field, seed_points):
    """
    Extracts streamlines (corelines) from a 3D vector field using VTK.

    Args:
        vector_field (StadyVectorField3D or UnsteadyVectorField3D): The input 3D vector field.
        seed_points (list of tuples): List of (x, y, z) seed point coordinates.

    Returns:
        vtk.vtkPolyData: The extracted streamlines.
    """
    # --- 1. Prepare VTK Data ---
    # Get the field data (ensure it's NumPy)
    if hasattr(vector_field, 'getDataAsNumpy'):
        field_data = vector_field.getDataAsNumpy()
    elif hasattr(vector_field, 'field'):
         # Fallback for classes like SteadyVectorField3D that store field directly
        field_data = np.asarray(vector_field.field)
    else:
        raise AttributeError("Vector field object must have 'getDataAsNumpy' method or 'field' attribute.")

    # Handle time-dependent fields: Get the first time slice or a specific one
    if len(field_data.shape) == 5: # (Time, Z, Y, X, 3)
        # Example: Use the first time slice. You might want to make this configurable.
        field_data = field_data[0]

    vtk_data = create_vtk_image_data(
        field_data,
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
    stream_tracer.SetIntegrationDirectionToBoth() # or Forward or Backward
    stream_tracer.SetIntegratorTypeToRungeKutta45() # Integration method
    stream_tracer.SetMaximumPropagation(200.0) # Adjust as needed
    stream_tracer.SetInitialIntegrationStep(0.05) # Adjust as needed
    stream_tracer.SetIntegrationStepUnit(2) # CELL_LENGTH_UNIT
    # stream_tracer.SetTerminalSpeed(1e-12)
    # stream_tracer.SetMaximumNumberOfSteps(2000)

    # --- 3. Execute and Return ---
    stream_tracer.Update()
    streamlines = stream_tracer.GetOutput()

    return streamlines

# --- Example Usage (Conceptual) ---
# Assuming you have instances of your vector field classes

# --- 2D Example ---
# my_2d_field = SteadyVectorField2D(Xdim=50, Ydim=50, ...)
# # ... populate my_2d_field.field with data ...
# seeds_2d = [(0, 0), (1, 1), (-1, 0.5)] # List of (x,y) tuples
# streamlines_2d_vtk = extract_corelines_2d(my_2d_field, seeds_2d)
# # You can now process or visualize streamlines_2d_vtk using VTK

# --- 3D Example ---
# my_3d_field = SteadyVectorField3D(Xdim=30, Ydim=30, Zdim=30, ...)
# # ... populate my_3d_field.field with data ...
# seeds_3d = [(0, 0, 0), (1, 1, 1)] # List of (x,y,z) tuples
# streamlines_3d_vtk = extract_corelines_3d(my_3d_field, seeds_3d)
# # You can now process or visualize streamlines_3d_vtk using VTK

