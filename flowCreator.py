import numpy as np
import numexpr as ne
import tqdm
from numpy import pi
from VectorField2d import VectorField2D,SteadyVectorField2D
import numpy as np
import matplotlib.pyplot as plt

class AnalyticalFlowCreator:
    def __init__(self, grid_size, time_steps=10,domainBoundaryMin=(-2.0,-2.0,0.0),domainBoundaryMax=(2.0,2.0,2*np.pi), parameters=None):
        """
        Initialize the analytical flow creator.

        :param expression_x: String, the mathematical expression for the x-component of the flow.
        :param expression_y: String, the mathematical expression for the y-component of the flow.
        :param grid_size: Tuple, the size of the grid on which to evaluate the flow.(Xdim,Ydim)
        :param time_steps: Int, the number of time steps for the flow field.
        :param parameters: Dictionary, additional parameters to be used in the expression.
        """
        # self.expression_x = expression_x
        # self.expression_y = expression_y
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
        
    def setExperssioin(self,expression_x,expression_y):
        self.expression_x = expression_x
        self.expression_y = expression_y

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
    flow_creator = AnalyticalFlowCreator( grid_size=grid_size,time_steps=timestep,domainBoundaryMin=domainBoundaryMin,domainBoundaryMax=domainBoundaryMax)
    flow_creator.setExperssioin('-y', 'x')
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

    flow_creator = AnalyticalFlowCreator(grid_size=grid_size,time_steps=timestep,domainBoundaryMin=domainBoundaryMin,domainBoundaryMax=domainBoundaryMax, parameters={'al_t': 1.0, 'scale': 1.0})
    flow_creator.setExperssioin(expression_u, expression_v)
    vectorField2d= flow_creator.create_flow_field()
    if scale!=1.0:
        vectorField2d.field*=scale
    return vectorField2d


def test_analytical_flow_creator():
    flow_creator = AnalyticalFlowCreator( grid_size=(200, 200))    
    flow_creator.setExperssioin('cos(y)', '-cos(x)')
    vectorField2d= flow_creator.create_flow_field()
    parameters = {'a': 0.5, 'b': 1.5}
    flow_creator = AnalyticalFlowCreator( grid_size=(200, 200), parameters=parameters)
    flow_creator.setExperssioin('a*sin(x)*cos(y)', '-b*cos(x)*sin(y)')
    vectorField2d = flow_creator.create_flow_field()
    new_parameters = {'a': 1.0, 'b': 2.0}
    flow_creator.update_parameters(new_parameters)
    vectorField2d= flow_creator.create_flow_field()
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




def LICImage_OFFLINE_RENDERING(vecfield: VectorField2D, timeSlice=0,stepSize=0.01, MaxIntegrationSteps=128):
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

def LICAlgorithmTest():
    vecfield=rotation_four_center((128,128),2)
    LICImage_OFFLINE_RENDERING(vecfield, 0,0.002, 128)


class ObserverReferenceTransformation:
    def __init__(self, timeStep=1) -> None:
        self.timeStep=timeStep
        #Q(t)*x+c(t)
        self.Q_t=[np.identity(2) ]*timeStep
        self.c_t=[0]*timeStep

    def GetPushForward(self, timeStep) -> np.ndarray:
        return self.Q_t[timeStep]

    def GetTransformation(self, timeStep) :
        return [self.Q_t[timeStep],self.c_t[timeStep]] 
    

class SteadyVastistasVelocityGenerator() :
    def __init__(self, grid_size, time_steps=10,domainBoundaryMin=(-2.0,-2.0,0.0),domainBoundaryMax=(2.0,2.0,2*np.pi), parameters=None):
        self.Xdim=grid_size[0]
        self.Ydim=grid_size[0]
        self.domainBoundaryMin=domainBoundaryMin
        self.domainBoundaryMax=domainBoundaryMax
        self.parameters = parameters if parameters is not None else {}
        # Adding a time dimension
        # self.time_steps = time_steps
        # self.t = np.linspace(0, 2*np.pi, time_steps)
        
    def VastistasVo(r, Si, rc, n):
        """
        Calculate velocity v(x) at point x according to equation (1).
        
        Parameters:
        r (float): distance from the origin.
        Si (numpy.ndarray): Matrix defining the base shape.
        rc (float): Radius with maximum velocity.
        n (float): Parameter controlling the shape of the velocity profile.
        
        Returns:
        float: Velocity at point x.
        """
        v0_r = r / (2*np.pi*(rc**2) * ((r/rc)**(2*n) + 1)**(1/n))
        return  v0_r
    def Vastistas(x, Si, rc, n):
        """
        Calculate velocity v(x) at point x according to equation (1).
        
        Parameters:
        x (numpy.ndarray): Point (x, y) at which to calculate velocity.
        Si (numpy.ndarray): Matrix defining the base shape.
        rc (float): Radius with maximum velocity.
        n (float): Parameter controlling the shape of the velocity profile.
        
        Returns:
        float: Velocity at point x.
        """
        r = np.linalg.norm(x)
        v0_r = r / (2*np.pi*(rc**2) * ((r/rc)**(2*n) + 1)**(1/n))
        return np.dot(Si, x) * v0_r


    def create_flow_field_slice(self) -> np.ndarray:
        pass
        # # Initializing empty arrays for vx and vy with an additional time dimension
        # data=np.zeros((self.Ydim, self.Xdim,2))
        # for i, t in enumerate(self.t):
        #     data[i,:,:,0]=vx_time_slice_i
        #     data[i,:,:,1]=vy_time_slice_i
        # return data
    
    # def generate(self)   -> VectorField2D:
    #     return self.create_flow_field()



def myTest():
    pass






if __name__ == '__main__':
    LICAlgorithmTest()

    