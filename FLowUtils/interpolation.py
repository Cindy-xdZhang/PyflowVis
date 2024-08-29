import numpy as np

def bilinear_interpolate(vector_field, x, y):
    """
    Perform bilinear interpolation for a 2D vector field.

    Parameters:
    - vector_field: np.ndarray of shape (Ydim, Xdim, 2), the 2D vector field.
    - x, y: float, the fractional coordinates at which to interpolate the vector.

    Returns:
    - interpolated_vector: The interpolated vector at position (x, y).
    """
    
    # Ensure x, y are within the bounds of the vector field
    x = np.clip(x, 0, vector_field.shape[1] - 1)
    y = np.clip(y, 0, vector_field.shape[0] - 1)

    # Get the integer parts of x, y
    x0 = int(x)
    y0 = int(y)

    # Ensure that we don't go out of bounds in the interpolation
    x1 = min(x0 + 1, vector_field.shape[1] - 1)
    y1 = min(y0 + 1, vector_field.shape[0] - 1)

    # Calculate the fractional parts of x, y
    tx = x - x0
    ty = y - y0

    # Get the vectors at the corner points
    v00 = vector_field[y0, x0,:]
    v01 = vector_field[y0, x1,:]
    v10 = vector_field[y1, x0,:]
    v11 = vector_field[y1, x1,:]

    # Perform bilinear interpolation
    a = v00 * (1 - tx) + v01 * tx
    b = v10 * (1 - tx) + v11 * tx
    interpolated_vector = a * (1 - ty) + b * ty

    return interpolated_vector