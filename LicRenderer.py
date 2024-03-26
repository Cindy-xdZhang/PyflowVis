import scipy.ndimage as ndimage

def lic_noise(size):
    """
    Generate a random noise background for LIC visualization.

    Args:
        size: Tuple of int, size of the noise image (nx, ny).

    Returns:
        A 2D numpy array representing the noise image.
    """
    nx, ny = size
    noise = np.random.rand(nx, ny)
    return noise

def lic_convolution(noise, vx, vy, num_steps=10, step_length=1.0):
    """
    Apply Line Integral Convolution (LIC) to a noise image using a vector field.

    Args:
        noise: 2D numpy array, initial noise image.
        vx, vy: 2D numpy arrays, components of the vector field.
        num_steps: Int, number of steps to integrate along the vector field.
        step_length: Float, length of each integration step.

    Returns:
        A 2D numpy array representing the LIC image.
    """
    nx, ny = noise.shape
    lic_image = np.zeros_like(noise)

    for i in range(nx):
        for j in range(ny):
            x, y = float(i), float(j)
            intensity = 0.0

            # Forward integration
            for _ in range(num_steps):
                if 0 <= int(x) < nx and 0 <= int(y) < ny:
                    intensity += noise[int(x), int(y)]
                    dx = vx[int(x), int(y)]
                    dy = vy[int(x), int(y)]
                    x += dx * step_length
                    y += dy * step_length

            # Backward integration
            x, y = float(i), float(j)  # Reset to start position
            for _ in range(num_steps):
                if 0 <= int(x) < nx and 0 <= int(y) < ny:
                    intensity += noise[int(x), int(y)]
                    dx = -vx[int(x), int(y)]
                    dy = -vy[int(x), int(y)]
                    x += dx * step_length
                    y += dy * step_length

            # Normalize the intensity
            lic_image[i, j] = intensity / (2 * num_steps + 1)

    return lic_image

# Test function calls
# Comment these out for the final script
# noise = lic_noise((100, 100))
# lic_image = lic_convolution(noise, vx_linear, vy_linear)
