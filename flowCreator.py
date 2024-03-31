import numpy as np
import numexpr as ne

def bilinear_interpolate(x, y, v00, v10, v01, v11):
    """
    Perform bilinear interpolation for a point (x, y) given the values at the corners
    of the square surrounding the point. The square corners have values v00, v10, v01, v11
    corresponding to (x0, y0), (x1, y0), (x0, y1), and (x1, y1) respectively.
    """
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = x0 + 1, y0 + 1

    # Compute the interpolation
    fxy1 = (x1 - x) * v00 + (x - x0) * v10
    fxy2 = (x1 - x) * v01 + (x - x0) * v11
    fxy = (y1 - y) * fxy1 + (y - y0) * fxy2

    return fxy

def trilinear_interpolate(x, y, z, t, v000, v100, v010, v110, v001, v101, v011, v111):
    """
    Perform trilinear interpolation for a point (x, y, z) given the values at the corners
    of the cube surrounding the point.
    """
    x0, y0, z0 = np.floor(x).astype(int), np.floor(y).astype(int), np.floor(z).astype(int)
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1

    # Compute the interpolation
    fxyz1 = bilinear_interpolate(x, y, v000, v100, v010, v110)
    fxyz2 = bilinear_interpolate(x, y, v001, v101, v011, v111)
    fxyz = (z1 - z) * fxyz1 + (z - z0) * fxyz2

    return fxyz


import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorField2D(nn.Module):
    def __init__(self, width, height, time_steps):
        super(VectorField2D, self).__init__()
        
        self.parameters = nn.Parameter(torch.randn(time_steps, height, width, 2))

    def get_vector_at_grid(self, x, y, t):    
        return self.parameters[t, y, x]

    def get_vector_slice(self, t):
        
        return self.parameters[t]

    def get_bilinear_interpolated_vector_at(self, x, y, t):
        
        x0, y0 = torch.floor(x).long(), torch.floor(y).long()
        x1, y1 = x0 + 1, y0 + 1

        Q11 = self.get_vector_at_grid(x0, y0, t)
        Q12 = self.get_vector_at_grid(x0, y1, t)
        Q21 = self.get_vector_at_grid(x1, y0, t)
        Q22 = self.get_vector_at_grid(x1, y1, t)

        weights = torch.tensor([[x1-x, x-x0], [y1-y, y-y0]])
        return Q11 * weights[0,0] * weights[1,0] + Q21 * weights[0,1] * weights[1,0] + Q12 * weights[0,0] * weights[1,1] + Q22 * weights[0,1] * weights[1,1]

    def get_trilinear_interpolated_vector_at(self, x, y, t):        
        pass  

    def forward(self, x, y, t):
        
        return self.get_bilinear_interpolated_vector_at(x, y, t)


class VectorField2D:
    def __init__(self, name,min_boundary:np.array,max_boundary:np.array, grid_size:np.array, vector_data):
        self.name = name
        self.min_boundary =min_boundary
        self.max_boundary =max_boundary
        self.domain_size = max_boundary-min_boundary# Tuple, the size of the domain (width, height,timeMax-timeMin)
        self.grid_interval=(max_boundary-min_boundary)/(grid_size-1)
        self.grid_size = grid_size
        self.vector_data = vector_data  # Expected to be a list of two numpy arrays [x_component, y_component]
        
    
    def get_vector_at_grid(self, t, y, x):
        return np.array([self.vector_data[0][t, y, x], self.vector_data[1][t, y, x]])

    def get_bilinear_interpolated_vector_at(self, x, y, t:int):
        # Calculate the indices of the grid cell corners surrounding (x, y)
        CoordsX = (x-self.min_boundary[0])/(self.grid_interval[0])
        CoordsY = (y-self.min_boundary[1])/(self.grid_interval[1])
        x0, y0 = np.floor(CoordsX).astype(int), np.floor(CoordsY).astype(int)
        ratioX,ratioY = CoordsX - x0, CoordsY - y0         
        x1, y1 = x0 + 1, y0 + 1
        
        # Ensure indices are within the bounds of the vector field data
        x0, x1 = max(0, min(self.grid_size[0] - 1, x0)), max(0, min(self.grid_size[0] - 1, x1))
        y0, y1 = max(0, min(self.grid_size[1] - 1, y0)), max(0, min(self.grid_size[1] - 1, y1))

        # Retrieve vectors at the corners of the cell
        v00 = self.get_vector_at_grid(t, y0, x0)
        v10 = self.get_vector_at_grid(t, y0, x1)
        v01 = self.get_vector_at_grid(t, y1, x0)
        v11 = self.get_vector_at_grid(t, y1, x1)
        
        # Perform bilinear interpolation for each component
        fx1 = (1-ratioX) * v00[0] + (ratioX) * v10[0]
        fx2 = (1-ratioX) * v01[0] + (ratioX) * v11[0]
        
        interpolated_x = (y1 - y) * fx1 + (y - y0) * fx2
        fy1 = (x1 - x) * v00[1] + (x - x0) * v10[1]
        fy2 = (x1 - x) * v01[1] + (x - x0) * v11[1]
        interpolated_y = (y1 - y) * fy1 + (y - y0) * fy2
        
        return np.array([interpolated_x, interpolated_y])




class AnalyticalFlowCreator:
    def __init__(self, expression_x, expression_y, grid_size=(100, 100), parameters=None):
        """
        Initialize the analytical flow creator.

        :param expression_x: String, the mathematical expression for the x-component of the flow.
        :param expression_y: String, the mathematical expression for the y-component of the flow.
        :param grid_size: Tuple, the size of the grid on which to evaluate the flow.
        :param parameters: Dictionary, additional parameters to be used in the expression.
        """
        self.expression_x = expression_x
        self.expression_y = expression_y
        self.grid_size = grid_size
        self.parameters = parameters if parameters is not None else {}
        self.x, self.y = np.meshgrid(np.linspace(-2, 2, grid_size[0]), np.linspace(-2, 2, grid_size[1]))

    def create_flow_field(self):
        """
        Create the flow field based on the mathematical expressions.

        :return: Two numpy arrays representing the x and y components of the flow field.
        """
        local_dict = {'x': self.x, 'y': self.y}
        local_dict.update(self.parameters)  # Add additional parameters to the dictionary
        vx = ne.evaluate(self.expression_x, local_dict=local_dict)
        vy = ne.evaluate(self.expression_y, local_dict=local_dict)
        return vx, vy

    def update_parameters(self, new_parameters):
        """
        Update the parameters used in the mathematical expressions.

        :param new_parameters: Dictionary, the new parameters to be updated.
        """
        self.parameters.update(new_parameters)



def test_analytical_flow_creator():
    flow_creator = AnalyticalFlowCreator('cos(y)', '-cos(x)', grid_size=(200, 200))
    vx, vy = flow_creator.create_flow_field()
    parameters = {'a': 0.5, 'b': 1.5}
    flow_creator = AnalyticalFlowCreator('a*sin(x)*cos(y)', '-b*cos(x)*sin(y)', grid_size=(200, 200), parameters=parameters)
    vx, vy = flow_creator.create_flow_field()

    new_parameters = {'a': 1.0, 'b': 2.0}
    flow_creator.update_parameters(new_parameters)
    vx, vy = flow_creator.create_flow_field()

    flow_creator = AnalyticalFlowCreator('x / (x**2 + y**2)', 'y / (x**2 + y**2)', grid_size=(200, 200))
    vx, vy = flow_creator.create_flow_field()


if __name__ == '__main__':
    test_analytical_flow_creator()

    