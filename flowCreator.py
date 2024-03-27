import numpy as np
import numexpr as ne

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

    