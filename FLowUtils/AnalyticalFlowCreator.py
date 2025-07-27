import numpy as np
import numexpr as ne
import tqdm
from numpy import pi
from .VectorField2d import UnsteadyVectorField2D

import numpy as np


class AnalyticalFlowCreator:
    def __init__(self, expression_x, expression_y, parameters=None):
        """
        Initialize the analytical flow creator.

        :param expression_x: String, the mathematical expression for the x-component of the flow.
        :param expression_y: String, the mathematical expression for the y-component of the flow.
        :param parameters: Dictionary, additional parameters to be used in the expression.
        """
        self.expression_x = expression_x
        self.expression_y = expression_y
        self.parameters = parameters if parameters is not None else {}

    def create_flow_field(self, grid_size, time_steps, domainBoundaryMin=(-2.0,-2.0,0.0), domainBoundaryMax=(2.0,2.0,2*np.pi)):
        """
        Create the flow field based on the mathematical expressions.

        :param grid_size: Tuple, the size of the grid on which to evaluate the flow.(Xdim,Ydim)
        :param time_steps: Int, the number of time steps for the flow field.
        :param domainBoundaryMin: Tuple, minimum boundaries for (x,y,t)
        :param domainBoundaryMax: Tuple, maximum boundaries for (x,y,t)
        :return: UnsteadyVectorField2D object
        """
        Xdim = grid_size[0]
        Ydim = grid_size[1]
        
        # Adding a time dimension
        t = np.linspace(0, 2*np.pi, time_steps)
        
        # Generating grid for x and y dimensions
        x, y = np.meshgrid(
            np.linspace(domainBoundaryMin[0], domainBoundaryMax[0], Xdim), 
            np.linspace(domainBoundaryMin[1], domainBoundaryMax[1], Ydim)
        )
        
        # Initializing empty arrays for vx and vy with an additional time dimension
        data = np.zeros((time_steps, Ydim, Xdim, 2))
        local_dict = {'x': x, 'y': y}
        local_dict.update(self.parameters)  # Add additional parameters to the dictionary
        
        for i, t_val in enumerate(t):
            local_dict['t'] = t_val
            vx_time_slice_i = ne.evaluate(self.expression_x, local_dict=local_dict)
            vy_time_slice_i = ne.evaluate(self.expression_y, local_dict=local_dict)
            data[i, :, :, 0] = vx_time_slice_i
            data[i, :, :, 1] = vy_time_slice_i

        vectorField2d = UnsteadyVectorField2D(Xdim, Ydim, time_steps, domainBoundaryMin, domainBoundaryMax)
        vectorField2d.field = data
        return vectorField2d

    def update_parameters(self, new_parameters):
        """
        Update the parameters used in the mathematical expressions.

        :param new_parameters: Dictionary, the new parameters to be updated.
        """
        self.parameters.update(new_parameters)


def constant_rotation(grid_size, timestep, domainBoundaryMin=(-2.0,-2.0,0.0), domainBoundaryMax=(2.0,2.0,2*np.pi), scale=1.0):
    """
    Create a constant rotation flow field.

    :param grid_size: Tuple, the size of the grid (Xdim, Ydim)
    :param timestep: Int, number of time steps
    :param domainBoundaryMin: Tuple, minimum boundaries for (x,y,t)
    :param domainBoundaryMax: Tuple, maximum boundaries for (x,y,t)
    :param scale: Float, the scale of the rotation.
    :return: UnsteadyVectorField2D object
    """
    flow_creator = AnalyticalFlowCreator('-y', 'x')
    vectorField2d = flow_creator.create_flow_field(grid_size, timestep, domainBoundaryMin, domainBoundaryMax)
    if scale != 1.0:
        vectorField2d.field *= scale
    return vectorField2d

def rotation_four_center(grid_size, timestep, domainBoundaryMin=(-2.0,-2.0,0.0), domainBoundaryMax=(2.0,2.0,2*np.pi), scale=1.0):
    """
    Create a rotation flow field with four centers.

    :param grid_size: Tuple, the size of the grid (Xdim, Ydim)
    :param timestep: Int, number of time steps
    :param domainBoundaryMin: Tuple, minimum boundaries for (x,y,t)
    :param domainBoundaryMax: Tuple, maximum boundaries for (x,y,t)
    :param scale: Float, the scale of the rotation.
    :return: UnsteadyVectorField2D object
    """
    expression_u = "exp(-y * y - x * x) * (al_t * y * exp(y * y + x * x) - 6.0 * scale * cos(al_t * t) * sin(al_t * t) * y * y * y + (12.0 * scale * (cos(al_t * t) * cos(al_t * t)) - 6.0 * scale) * x * y * y + (6.0 * scale * cos(al_t * t) * sin(al_t * t) * x * x + 6.0 * scale * cos(al_t * t) * sin(al_t * t)) * y + (3.0 * scale - 6.0 * scale * (cos(al_t * t) * cos(al_t * t))) * x)"
    expression_v = "-exp(-y * y - x * x) * (al_t * x * exp(y * y + x * x) - 6.0 * scale * cos(al_t * t) * sin(al_t * t) * x * y * y + ((12.0 * scale * (cos(al_t * t) * cos(al_t * t)) - 6.0 * scale) * x * x - 6.0 * scale * (cos(al_t * t) * cos(al_t * t)) + 3.0 * scale) * y + 6.0 * scale * cos(al_t * t) * sin(al_t * t) * x * x * x - 6.0 * scale * cos(al_t * t) * sin(al_t * t) * x)"

    flow_creator = AnalyticalFlowCreator(expression_u, expression_v, parameters={'al_t': 1.0, 'scale': 8.0})
    vectorField2d = flow_creator.create_flow_field(grid_size, timestep, domainBoundaryMin, domainBoundaryMax)
    if scale != 1.0:
        vectorField2d.field *= scale
    return vectorField2d

def beadsFLow(grid_size, timestep, domainBoundaryMin=(-2.0,-2.0,0.0), domainBoundaryMax=(2.0,2.0,2*np.pi), scale=1.0):
    """
    Create a beads flow field based on the beads2d problem.
    
    :param grid_size: Tuple, the size of the grid (Xdim, Ydim)
    :param timestep: Int, number of time steps
    :param domainBoundaryMin: Tuple, minimum boundaries for (x,y,t)
    :param domainBoundaryMax: Tuple, maximum boundaries for (x,y,t)
    :param scale: Float, scaling factor for the field
    :return: UnsteadyVectorField2D object
    """
    expression_u = "-1.0 * (y - 1.0 / 3.0 * sin(t)) - (x - 1.0 / 3.0 * cos(t))"
    expression_v = "(x - 1.0 / 3.0 * cos(t)) - (y - 1.0 / 3.0 * sin(t))"
    
    flow_creator = AnalyticalFlowCreator(expression_u, expression_v)
    vectorField2d = flow_creator.create_flow_field(grid_size, timestep, domainBoundaryMin, domainBoundaryMax)
    
    if scale != 1.0:
        vectorField2d.field *= scale
    return vectorField2d

# def test_analytical_flow_creator():
#     flow_creator = AnalyticalFlowCreator('cos(y)', '-cos(x)')    
#     vectorField2d = flow_creator.create_flow_field((200, 200), 10)
#     
#     parameters = {'a': 0.5, 'b': 1.5}
#     flow_creator = AnalyticalFlowCreator('a*sin(x)*cos(y)', '-b*cos(x)*sin(y)', parameters=parameters)
#     vectorField2d = flow_creator.create_flow_field((200, 200), 10)
#     
#     new_parameters = {'a': 1.0, 'b': 2.0}
#     flow_creator.update_parameters(new_parameters)
#     vectorField2d = flow_creator.create_flow_field((200, 200), 10)
#     
#     vectorField2d = rotation_four_center((32,32), 32)









    