import numpy as np
import numexpr as ne
import tqdm
from numpy import pi
from .VectorField2d import UnsteadyVectorField2D

import numpy as np


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
        local_dict.update(self.parameters)  # Add additional parameters to the dictionary1
        for i, t in enumerate(self.t):
            local_dict['t'] = t
            vx_time_slice_i= ne.evaluate(self.expression_x, local_dict=local_dict)
            vy_time_slice_i= ne.evaluate(self.expression_y, local_dict=local_dict)
            data[i,:,:,0]=vx_time_slice_i
            data[i,:,:,1]=vy_time_slice_i

        vectorField2d=UnsteadyVectorField2D(self.Xdim, self.Ydim, self.time_steps,self.domainBoundaryMin,self.domainBoundaryMax)
        vectorField2d.field=data
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

    flow_creator = AnalyticalFlowCreator(grid_size=grid_size,time_steps=timestep,domainBoundaryMin=domainBoundaryMin,domainBoundaryMax=domainBoundaryMax, parameters={'al_t': 1.0, 'scale': 8.0})
    flow_creator.setExperssioin(expression_u, expression_v)
    vectorField2d= flow_creator.create_flow_field()
    if scale!=1.0:
        vectorField2d.field*=scale
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
    
    flow_creator = AnalyticalFlowCreator(
        grid_size=grid_size,
        time_steps=timestep,
        domainBoundaryMin=domainBoundaryMin,
        domainBoundaryMax=domainBoundaryMax
    )
    flow_creator.setExperssioin(expression_u, expression_v)
    vectorField2d = flow_creator.create_flow_field()
    
    if scale != 1.0:
        vectorField2d.field *= scale
    return vectorField2d

def rotation_four_center(grid_size,timestep,domainBoundaryMin=(-2.0,-2.0,0.0),domainBoundaryMax=(2.0,2.0,2*np.pi),scale=1.0):
    """
    Create a constant rotation flow field.

    :param scale: Float, the scale of the rotation.
    :return: Two numpy arrays representing the x and y components of the flow field.
    """

    expression_u="exp(-y * y - x * x) * (al_t * y * exp(y * y + x * x) - 6.0 * scale * cos(al_t * t) * sin(al_t * t) * y * y * y + (12.0 * scale * (cos(al_t * t) * cos(al_t * t)) - 6.0 * scale) * x * y * y + (6.0 * scale * cos(al_t * t) * sin(al_t * t) * x * x + 6.0 * scale * cos(al_t * t) * sin(al_t * t)) * y + (3.0 * scale - 6.0 * scale * (cos(al_t * t) * cos(al_t * t))) * x)"
    expression_v="-exp(-y * y - x * x) * (al_t * x * exp(y * y + x * x) - 6.0 * scale * cos(al_t * t) * sin(al_t * t) * x * y * y + ((12.0 * scale * (cos(al_t * t) * cos(al_t * t)) - 6.0 * scale) * x * x - 6.0 * scale * (cos(al_t * t) * cos(al_t * t)) + 3.0 * scale) * y + 6.0 * scale * cos(al_t * t) * sin(al_t * t) * x * x * x - 6.0 * scale * cos(al_t * t) * sin(al_t * t) * x)"

    flow_creator = AnalyticalFlowCreator(grid_size=grid_size,time_steps=timestep,domainBoundaryMin=domainBoundaryMin,domainBoundaryMax=domainBoundaryMax, parameters={'al_t': 1.0, 'scale': 8.0})
    flow_creator.setExperssioin(expression_u, expression_v)
    vectorField2d= flow_creator.create_flow_field()
    if scale!=1.0:
        vectorField2d.field*=scale
    return vectorField2d

# def test_analytical_flow_creator():
#     flow_creator = AnalyticalFlowCreator( grid_size=(200, 200))    
#     flow_creator.setExperssioin('cos(y)', '-cos(x)')
#     vectorField2d= flow_creator.create_flow_field()
#     parameters = {'a': 0.5, 'b': 1.5}
#     flow_creator = AnalyticalFlowCreator( grid_size=(200, 200), parameters=parameters)
#     flow_creator.setExperssioin('a*sin(x)*cos(y)', '-b*cos(x)*sin(y)')
#     vectorField2d = flow_creator.create_flow_field()
#     new_parameters = {'a': 1.0, 'b': 2.0}
#     flow_creator.update_parameters(new_parameters)
#     vectorField2d= flow_creator.create_flow_field()
#     vectorField2d= rotation_four_center((32,32),32)









    