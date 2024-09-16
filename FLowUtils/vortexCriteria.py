from PIL import Image
import os
import numpy as np
from .VectorField2d import UnsteadyVectorField2D

def computeQcriterion(vecfieldData, SpatialGridIntervalX, SpatialGridIntervalY):
    """
    Computes the Q criterion for a 2D steady vector field slice.

    Parameters:
        vecfieldData (numpy.ndarray): 3D vector field data, shape [ W, H, 2].
        SpatialGridIntervalX (float): Spatial grid interval in the X direction.
        SpatialGridIntervalY (float): Spatial grid interval in the Y direction.

    Returns:
        numpy.ndarray: Q criterion array, shape [W, H].
    """
    W, H, _ = vecfieldData.shape
    Q = np.zeros((W, H))
    inverse_grid_interval_x = 1.0 / SpatialGridIntervalX
    inverse_grid_interval_y = 1.0 / SpatialGridIntervalY

    for y in range(1, H - 1):
        for x in range(1, W - 1):
            du_dx = (vecfieldData[  y,x + 1] - vecfieldData[ y,x - 1]) * 0.5 * inverse_grid_interval_x
            dv_dy = (vecfieldData[ y + 1 ,x ] - vecfieldData[y - 1,x ]) * 0.5 * inverse_grid_interval_y

            gradient = np.array([[du_dx[0], du_dx[1]],
                                 [dv_dy[0], dv_dy[1]]])

            S = 0.5 * (gradient + gradient.T)
            Omega = 0.5 * (gradient - gradient.T)

            Q[y, x] = 0.5 * (np.linalg.norm(Omega, 'fro')**2 - np.linalg.norm(S, 'fro')**2)

    return Q

def computeIVD(vecfieldData, SpatialGridIntervalX, SpatialGridIntervalY):
    """
    Computes the Instantaneous Vorticity Deviation (IVD) for a 2D steady vector field slice.

    Parameters:
        vecfieldData (numpy.ndarray): 3D vector field data, shape [ W, H, 2].
        SpatialGridIntervalX (float): Spatial grid interval in the X direction.
        SpatialGridIntervalY (float): Spatial grid interval in the Y direction.

    Returns:
        numpy.ndarray: IVD array, shape [W, H].
    """
    W, H, _ = vecfieldData.shape
    IVD = np.zeros((W, H))
    inverse_grid_interval_x = 1.0 / SpatialGridIntervalX
    inverse_grid_interval_y = 1.0 / SpatialGridIntervalY

    curlField = np.zeros((W, H))

    for y in range(1, H - 1):
        for x in range(1, W - 1):
            dv_dx = (vecfieldData[ y,x + 1 ] - vecfieldData[y,x - 1 ]) * 0.5 * inverse_grid_interval_x
            du_dy = (vecfieldData[ y + 1,x ] - vecfieldData[y - 1,x ]) * 0.5 * inverse_grid_interval_y
            curlField[y, x] = dv_dx[1] - du_dy[0]

    averageCurl = np.mean(curlField[1:-1, 1:-1])

    for y in range(1, H - 1):
        for x in range(1, W - 1):
            vorticity =curlField[y,x]
            IVD[y,x] = np.abs(vorticity - averageCurl)

    return IVD

def scalar_to_color(scalar, min_scalar, max_scalar):
    """
    Map a scalar value to an RGB color using a simple linear color mapping.
    
    Parameters:
    - scalar: float, the scalar value to map.
    - min_scalar: float, the minimum scalar value in the data.
    - max_scalar: float, the maximum scalar value in the data.
    
    Returns:
    - color: tuple, the RGB color corresponding to the scalar value.
    """
    normalized_scalar = (scalar - min_scalar) / (max_scalar - min_scalar)  # Normalize the scalar to [0, 1]
    # Define a simple linear colormap from blue to red
    blue = (51, 255, 255)
    red = (255, 0, 0)
    r = int(blue[0] * (1 - normalized_scalar) + red[0] * normalized_scalar)
    g = int(blue[1] * (1 - normalized_scalar) + red[1] * normalized_scalar)
    b = int(blue[2] * (1 - normalized_scalar) + red[2] * normalized_scalar)
    return np.array( [r, g, b,255],dtype=np.uint8)

def saveCriteriaPicture(scalarField, filename, upSample=1.0):
    """
    Saves a 2D scalar field as a PNG image with a blue-to-red colormap.

    Parameters:
        scalarField (numpy.ndarray): The scalar field array of shape (Ydim, Xdim).
        filename (str): The filename to save the PNG image.
        upSample (float): Upsampling factor to resize the image. Default is 1.0 (no scaling).
    """
    # Create the directory if it does not exist
    folder = os.path.dirname(filename)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created directory: {folder}")
    Ydim,Xdim=scalarField.shape[0:2]
    minV,maxV=scalarField.min(),scalarField.max()

    colored_image=np.zeros((Ydim,Xdim,4),dtype=np.uint8)
    for y in range(0,Ydim):
        for x in range(0,Xdim):
            color=scalar_to_color(scalarField[y][x],minV,maxV)
            colored_image[y,x,:]=color

    colored_image = (colored_image[..., :3] ).astype(np.uint8)  # Drop alpha channel and scale to [0, 255]

    # Convert to image
    image = Image.fromarray(colored_image)

    # Apply upsampling if needed
    if upSample != 1.0:
        new_size = (int(image.width * upSample), int(image.height * upSample))
        image = image.resize(new_size, Image.NEAREST)
    
    # Save the image
    image.save(filename)


def referenceFrameReconstruct(abc,abcDot,inputfield:UnsteadyVectorField2D):
    """
    referenceFrameReconstruct is suffer from inputfield has limited domain size, and doesn't have analytical expression for point out side of its domain.
    """
    dt=inputfield.timeInterval
    # Initial values
    theta=0.0        
    theta_t = [0.0]
    angular_velocity = abc[2]  # abc is a numpy array of shape (3,)
    angular_velocities = [angular_velocity]
    

    translation_c=np.array([0.0, 0.0])
    translation_c_t = [np.array([0.0, 0.0])]
    translation_cdot = np.array([abc[0], abc[1]])  # translation velocity
    velocities = [translation_cdot]
    translation_cdotdot = np.array([abcDot[0], abcDot[1]])  # acceleration
    Q_tlist= [ np.array([
            [1.0, 0],
            [0, 1.0]
        ])]
    # Integrate rotation and translation
    for i in range(1, inputfield.time_steps):
        theta += dt * angular_velocity
        theta_t.append(theta)
        angular_velocity += dt * abcDot[2]
        angular_velocities.append(angular_velocity)
        Q_tlist.append( np.array([
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]]) )
        translation_c += dt * translation_cdot
        translation_c_t.append(translation_c)
        translation_cdot += dt * translation_cdotdot
        velocities.append(translation_cdot)

    #reconstruct:
    reconstructField=UnsteadyVectorField2D(16,16,5,[-2,-2],[2,2],tmin=0,tmax=0.7853981633974483)
    reconstructField.field=np.zeros([5,16,16, 2],dtype=np.float32)
    
    for t in range(0, inputfield.time_steps):
        # Rotation matrix Q_t based on theta
        theta=theta_t[t]
        Q_t = Q_tlist[t]
        Q_t_transpose = Q_t.T
        angular_velocity=angular_velocities[t]
        # Compute spin tensor (anti-symmetric matrix of angular velocity)
        spin_tensor = np.array([
            [0.0, angular_velocity],
            [-angular_velocity, 0.0]
        ])
        # Compute Q_dot
        Q_dot = np.dot(Q_t, spin_tensor)
        # Translation velocity at this time step
        translation_velocity = velocities[t]
        for y in range(0, inputfield.Ydim):
            for x in range(0, inputfield.Xdim):
                pos_x=np.array([inputfield.domainMinBoundary[0]+x*inputfield.gridInterval[0],inputfield.domainMinBoundary[1]+y*inputfield.gridInterval[1]])
                # Transformed position xStar
                x_star = np.dot(Q_t, pos_x) + translation_c_t[t]
                # Get the analytical vector from the input field at xStar and time t
                #x_star is physical coordinate, need convert to floating indices
                x_star_floatIndex_x=float(x_star[0]-inputfield.domainMinBoundary[0])/inputfield.gridInterval[0]
                x_star_floatIndex_y=float(x_star[1]-inputfield.domainMinBoundary[1])/inputfield.gridInterval[1]
                v_star_xstar = inputfield.getBilinearInterpolateVector(x_star_floatIndex_x,x_star_floatIndex_y,t)

                # Compute the velocity at the original position
                v_at_pos = np.dot(Q_t_transpose, (v_star_xstar - np.dot(Q_dot, pos_x) - translation_velocity))
                reconstructField.field[t][y][x]=v_at_pos
    return reconstructField


if __name__ == '__main__':
    pass