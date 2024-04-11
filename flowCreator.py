import numpy as np
import numexpr as ne
from VectorField2d import VectorField2D
import numpy as np
import torch
import matplotlib.pyplot as plt

class AnalyticalFlowCreator:
    def __init__(self, expression_x, expression_y, grid_size, time_steps=10,domainBoundaryMin=(-2.0,-2.0,0.0),domainBoundaryMax=(2.0,2.0,2*np.pi), parameters=None):
        """
        Initialize the analytical flow creator.

        :param expression_x: String, the mathematical expression for the x-component of the flow.
        :param expression_y: String, the mathematical expression for the y-component of the flow.
        :param grid_size: Tuple, the size of the grid on which to evaluate the flow.(Xdim,Ydim)
        :param time_steps: Int, the number of time steps for the flow field.
        :param parameters: Dictionary, additional parameters to be used in the expression.
        """
        self.expression_x = expression_x
        self.expression_y = expression_y
        self.Xdim=grid_size[0]
        self.Ydim=grid_size[0]
        self.time_steps = time_steps
        self.parameters = parameters if parameters is not None else {}
        # Adding a time dimension
        self.t = np.linspace(0, 2*np.pi, time_steps)
        self.domainBoundaryMin=domainBoundaryMin
        self.domainBoundaryMax=domainBoundaryMax
        # Generating grid for x and y dimensions
        self.x, self.y = np.meshgrid(np.linspace(domainBoundaryMin[0], domainBoundaryMax[0], grid_size[0]), np.linspace(domainBoundaryMin[1], domainBoundaryMax[1], grid_size[1]))
        
        
    def create_flow_field(self):
        """
        Create the flow field based on the mathematical expressions.

        :return: Two numpy arrays representing the x and y components of the flow field.
        """
        # Initializing empty arrays for vx and vy with an additional time dimension
        data=np.zeros((self.time_steps, self.Ydim, self.Xdim,2))
        local_dict = {'x': self.x, 'y': self.y}
        local_dict.update(self.parameters)  # Add additional parameters to the dictionary
        for i, t in enumerate(self.t):
            local_dict['t'] = t
            vx_time_slice_i= ne.evaluate(self.expression_x, local_dict=local_dict)
            vy_time_slice_i= ne.evaluate(self.expression_y, local_dict=local_dict)
            data[i,:,:,0]=vx_time_slice_i
            data[i,:,:,1]=vy_time_slice_i

        vectorField2d=VectorField2D(self.Xdim, self.Ydim, self.time_steps,self.domainBoundaryMin,self.domainBoundaryMax)
        vectorField2d.setInitialVectorField(data)
        return vectorField2d

    def update_parameters(self, new_parameters):
        """
        Update the parameters used in the mathematical expressions.

        :param new_parameters: Dictionary, the new parameters to be updated.
        """
        self.parameters.update(new_parameters)


def constant_rotation(grid_size,timestep,domainBoundaryMin=(-2.0,-2.0,0.0),domainBoundaryMax=(2.0,2.0,2*np.pi),scale=1.0):
    """
    Create a constant rotation flow field.

    :param scale: Float, the scale of the rotation.
    :return: Two numpy arrays representing the x and y components of the flow field.
    """
    flow_creator = AnalyticalFlowCreator('-y', 'x', grid_size=grid_size,time_steps=timestep,domainBoundaryMin=domainBoundaryMin,domainBoundaryMax=domainBoundaryMax)
    vectorField2d= flow_creator.create_flow_field()
    if scale!=1.0:
        vectorField2d.field*=scale
    return vectorField2d

def rotation_four_center(grid_size,timestep,domainBoundaryMin=(-2.0,-2.0,0.0),domainBoundaryMax=(2.0,2.0,2*np.pi),scale=1.0):
    """
    Create a constant rotation flow field.

    :param scale: Float, the scale of the rotation.
    :return: Two numpy arrays representing the x and y components of the flow field.
    """

    expression_u="exp(-y * y - x * x) * (al_t * y * exp(y * y + x * x) - 6.0 * scale * cos(al_t * t) * sin(al_t * t) * y * y * y + (12.0 * scale * (cos(al_t * t) * cos(al_t * t)) - 6.0 * scale) * x * y * y + (6.0 * scale * cos(al_t * t) * sin(al_t * t) * x * x + 6.0 * scale * cos(al_t * t) * sin(al_t * t)) * y + (3.0 * scale - 6.0 * scale * (cos(al_t * t) * cos(al_t * t))) * x)"
    expression_v="-exp(-y * y - x * x) * (al_t * x * exp(y * y + x * x) - 6.0 * scale * cos(al_t * t) * sin(al_t * t) * x * y * y + ((12.0 * scale * (cos(al_t * t) * cos(al_t * t)) - 6.0 * scale) * x * x - 6.0 * scale * (cos(al_t * t) * cos(al_t * t)) + 3.0 * scale) * y + 6.0 * scale * cos(al_t * t) * sin(al_t * t) * x * x * x - 6.0 * scale * cos(al_t * t) * sin(al_t * t) * x)"

    flow_creator = AnalyticalFlowCreator(expression_u,expression_v,grid_size=grid_size,time_steps=timestep,domainBoundaryMin=domainBoundaryMin,domainBoundaryMax=domainBoundaryMax, parameters={'al_t': 1.0, 'scale': 1.0})
    vectorField2d= flow_creator.create_flow_field()
    if scale!=1.0:
        vectorField2d.field*=scale
    return vectorField2d


def test_analytical_flow_creator():
    flow_creator = AnalyticalFlowCreator('cos(y)', '-cos(x)', grid_size=(200, 200))
    vectorField2d= flow_creator.create_flow_field()
    parameters = {'a': 0.5, 'b': 1.5}
    flow_creator = AnalyticalFlowCreator('a*sin(x)*cos(y)', '-b*cos(x)*sin(y)', grid_size=(200, 200), parameters=parameters)
    vectorField2d = flow_creator.create_flow_field()

    new_parameters = {'a': 1.0, 'b': 2.0}
    flow_creator.update_parameters(new_parameters)
    vectorField2d= flow_creator.create_flow_field()

    flow_creator = AnalyticalFlowCreator('x / (x**2 + y**2)', 'y / (x**2 + y**2)', grid_size=(200, 200))
    vectorField2d = flow_creator.create_flow_field()
    vectorField2d= rotation_four_center((32,32),32)


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


def LICAlgorithm(texture, vecfield: VectorField2D,timeSlice, stepSize, MaxIntegrationSteps):
    """
    A simplified LIC algorithm to visualize the flow of a 2D vector field.
    """
    with torch.no_grad():
        vecfieldData=vecfield.field.detach().cpu().numpy()[timeSlice,:,:,:]
        
        Ydim, Xdim,_ = texture.shape
        output_texture = np.zeros_like(texture)



        for y in range(Ydim):
            for x in range(Xdim):
                accum_value = 0.0
                accum_count = 0
                
                # Trace forward
                #pos (x,y)
                pos = np.array([x* vecfield.gridInterval[0]+vecfield.domainMinBoundary[0], y * vecfield.gridInterval[1]+vecfield.domainMinBoundary[1]], dtype=np.float32)

                for _ in range(MaxIntegrationSteps):
                    if not (0 <= pos[0] < Ydim and 0 <= pos[1] < Xdim):
                        break  # Stop if we move outside the texture bounds
                    accum_value += bilinear_interpolate(texture, pos[0], pos[1])
                    accum_count += 1
                    vec =bilinear_interpolate(vecfieldData, pos[0], pos[1])
                    pos += vec * stepSize
                    
                # Trace backward
                pos = np.array([y, x], dtype=np.float32)
                for _ in range(MaxIntegrationSteps):
                    if not (0 <= pos[0] < Ydim and 0 <= pos[1] < Xdim):
                        break
                    accum_value += bilinear_interpolate(texture, pos[0], pos[1])
                    accum_count += 1
                    vec =bilinear_interpolate(vecfieldData, pos[0], pos[1])
                    pos -= vec * stepSize
                
                # Compute the average value along the path
                if accum_count > 0:
                    output_texture[y, x] = accum_value / accum_count
        
        return output_texture




def LICImage_OFFLINE_RENDERING(vecfield: VectorField2D, timeSlice=0,stepSize=0.01, MaxIntegrationSteps=128):
    """
    Render a steady 2D vector field as an LIC image and save to a PNG file.
    """
    # Step 1: Initialize a texture for the LIC process, often random noise
    texture = np.random.rand(vecfield.Ydim, vecfield.Xdim,1)
    # Detach the tensor, move it to CPU, and convert to NumPy

    # Step 2: Prepare your LIC implementation here. This is a placeholder for
    # the process of integrating along the vector field to modify the texture.
    # You'll need to replace this with your actual LIC algorithm.
    lic_result = LICAlgorithm(texture, vecfield, 0,stepSize, MaxIntegrationSteps)
    
    # Step 3: Normalize the LIC result for visualization
    lic_normalized = (lic_result - np.min(lic_result)) / (np.max(lic_result) - np.min(lic_result))
    
    # Step 4: Convert to an image and save
    plt.imshow(lic_normalized, cmap='gray')
    plt.axis('off')  # Optional: Remove axis for a cleaner image
    plt.savefig("vector_field_lic.png", bbox_inches='tight', pad_inches=0)

def myTest():
    vecfield=rotation_four_center((512,512),2)
    LICImage_OFFLINE_RENDERING(vecfield, 0,0.1, 128)



if __name__ == '__main__':
    myTest()

    