import numpy as np
from .VectorField2d import *

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

def LICAlgorithm(texture:np.ndarray, vecfield: SteadyVectorField2D, resultImageSizeX, resultImageSizeY,stepSize:float, MaxIntegrationSteps:int):
    """
    A simplified LIC algorithm to visualize the flow of a 2D vector field slice. 
    texture as same size as the vector field slice, vecfield is the vector field slice,
    resultImageSizeX and resultImageSizeY are the size of the output image, 
    stepSize is the step size for integration, and MaxIntegrationSteps is the maximum number of integration steps to take.
    """
    if texture.ndim == 2:
        Ydim, Xdim = texture.shape
        texture = texture[:, :, np.newaxis]  # Add a dummy channel dimension
    else:
        Ydim, Xdim, _ = texture.shape

    output_texture = np.zeros((resultImageSizeY, resultImageSizeX), dtype=np.float32)
    vecfieldData=vecfield.field
    domainRangeX=vecfield.domainMaxBoundary[0]-vecfield.domainMinBoundary[0]
    domainRangeY=vecfield.domainMaxBoundary[1]-vecfield.domainMinBoundary[1]

    inverse_grid_interval_x=1/float(vecfield.gridInterval[0])
    inverse_grid_interval_y=1/float(vecfield.gridInterval[1])
    for y in range(resultImageSizeY):
        for x in range(resultImageSizeX):

            ratioX=float(x)/float(resultImageSizeX)
            ratioY=float(y)/float(resultImageSizeY)
            accum_value = 0.0
            accum_count = 0
            
            # Trace forward
            #pos (x,y)
            pos = np.array([ratioX* domainRangeX+vecfield.domainMinBoundary[0], ratioY* domainRangeY+vecfield.domainMinBoundary[1]], dtype=np.float32)

            for _ in range(MaxIntegrationSteps):
                floatIndexX=(pos[0]-vecfield.domainMinBoundary[0])*inverse_grid_interval_x
                floatIndexY=(pos[1]-vecfield.domainMinBoundary[1])*inverse_grid_interval_y
                if not (0 <= floatIndexX < Xdim and 0 <= floatIndexY < Ydim):
                    break  # Stop if we move outside the texture bounds

                accum_value += bilinear_interpolate(texture, floatIndexX, floatIndexY)
                accum_count += 1
                vec =bilinear_interpolate(vecfieldData,  floatIndexX, floatIndexY)
                pos += vec * stepSize
                
            # Trace backward
            pos = np.array([ratioX* domainRangeX+vecfield.domainMinBoundary[0], ratioY* domainRangeY+vecfield.domainMinBoundary[1]], dtype=np.float32)
            for _ in range(MaxIntegrationSteps):
                floatIndexX=(pos[0]-vecfield.domainMinBoundary[0])*inverse_grid_interval_x
                floatIndexY=(pos[1]-vecfield.domainMinBoundary[1])*inverse_grid_interval_y
                if not (0 <= floatIndexX < Xdim and 0 <= floatIndexY < Ydim):
                    break  # Stop if we move outside the texture bounds

                accum_value += bilinear_interpolate(texture, floatIndexX, floatIndexY)
                accum_count += 1
                vec =bilinear_interpolate(vecfieldData,  floatIndexX, floatIndexY)

                pos -= vec * stepSize
            
            # Compute the average value along the path
            if accum_count > 0:
                output_texture[y, x] = accum_value / accum_count
    
    return output_texture

import matplotlib.pyplot as plt
def LICImage_OFFLINE_RENDERING(vecfield: UnsteadyVectorField2DNp|UnsteadyVectorField2D, timeSlice=0,stepSize=0.01, MaxIntegrationSteps=128):
    """
    Render a steady 2D vector field as an LIC image and save to a PNG file.
    """
    # Step 1: Initialize a texture for the LIC process, often random noise
    texture = np.random.rand(vecfield.Ydim, vecfield.Xdim)
    # Detach the tensor, move it to CPU, and convert to NumPy
    VecFieldSlice=vecfield.getSlice(timeSlice)
    # Step 2: Prepare your LIC implementation here. This is a placeholder for
    # the process of integrating along the vector field to modify the texture.
    # You'll need to replace this with your actual LIC algorithm.
    lic_result = LICAlgorithm(texture, VecFieldSlice, 128,128,stepSize, MaxIntegrationSteps)
    
    # Step 3: Normalize the LIC result for visualization
    lic_normalized = (lic_result - np.min(lic_result)) / (np.max(lic_result) - np.min(lic_result))
    
    # Step 4: Convert to an image and save
    plt.imshow(lic_normalized, cmap='gray')
    plt.axis('off')  # Optional: Remove axis for a cleaner image
    plt.savefig("vector_field_lic.png", bbox_inches='tight', pad_inches=0)

@typechecked
def LicRenderingUnsteady(field:UnsteadyVectorField2DNp,licImageSize:int,timeStepSKip:int=2,saveFolder:str="./"):
    #typecheck field type and field is not None    
    Xdim,Ydim,time_steps=field.Xdim,field.Ydim,field.time_steps
    texture = np.random.rand(Xdim, Ydim)    
    for i in range(0, time_steps, timeStepSKip):
        print(f"Processing time step {i}")
        steadyVectorField2D = field.getSlice(i)
        lic_result=LICAlgorithm(  texture  ,steadyVectorField2D ,licImageSize,licImageSize,0.005,256)
        lic_normalized =255* (lic_result - np.min(lic_result)) / (np.max(lic_result) - np.min(lic_result))
        #  Convert to an image and save
        plt.imshow(lic_normalized, cmap='gray')
        plt.axis('off')  # Optional: Remove axis for a cleaner image
        save_name=f"vector_field_lic_{i}.png"
        savePath=os.path.join(saveFolder,save_name)
        plt.savefig(savePath, bbox_inches='tight', pad_inches=0)

