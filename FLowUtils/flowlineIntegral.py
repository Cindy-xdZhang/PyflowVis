import numpy as np
from .VectorField2d import UnsteadyVectorField2D
from .VectorField3d import UnsteadyVectorField3D


def pathline_integration_one_direction_2D(
    vectorField: UnsteadyVectorField2D,
    start_pos:np.ndarray[np.float32,3],
    timeStart:float,
    timeEnd:float,
    stepSize:float=0.01,
    maxIterations:int=5000,
    NumericalMethod:str="RK4"
):
    """
    Integrate a pathline in one direction through an unsteady vector field.
    Args:
        vectorField: UnsteadyVectorField2D 
        start_pos: [x, y,z=0] initial position
        timeStart: float, start time
        timeEnd: float, end time
        stepSize: float, integration time step
        maxIterations: int, maximum number of steps
        NumericalMethod: str, "RK4" or "Euler"
    Returns:
        List of (position, time) tuples
    """
    pos_3d = np.array(start_pos, dtype=np.float32)
    t = timeStart
    path = [(pos_3d.copy(), t)]
    direction = 1 if timeEnd > timeStart else -1
    stepSize = abs(stepSize) * direction
    for i in range(maxIterations):
        if (direction > 0 and t >= timeEnd) or (direction < 0 and t <= timeEnd) or not vectorField.IsInside(pos_3d):
            break
        if NumericalMethod == "RK4":
            v1 = vectorField.get_vector(pos_3d[0], pos_3d[1], t)
            v2 = vectorField.get_vector(pos_3d[0] + 0.5 * stepSize * v1[0], pos_3d[1] + 0.5 * stepSize * v1[1], t + 0.5 * stepSize)
            v3 = vectorField.get_vector(pos_3d[0] + 0.5 * stepSize * v2[0], pos_3d[1] + 0.5 * stepSize * v2[1], t + 0.5 * stepSize)
            v4 = vectorField.get_vector(pos_3d[0] + stepSize * v3[0], pos_3d[1] + stepSize * v3[1], t + stepSize)
            delta = (stepSize / 6.0) * (v1 + 2 * v2 + 2 * v3 + v4)
        elif NumericalMethod == "RK5":
            k1 = vectorField.get_vector(pos_3d[0], pos_3d[1], t)
            p2 = pos_3d[:2] + stepSize * (1/4 * k1)
            k2 = vectorField.get_vector(p2[0], p2[1], t + 1/4 * stepSize)
            p3 = pos_3d[:2] + stepSize * (3/32 * k1 + 9/32 * k2)
            k3 = vectorField.get_vector(p3[0], p3[1], t + 3/8 * stepSize)
            p4 = pos_3d[:2] + stepSize * (1932/2197 * k1 - 7200/2197 * k2 + 7296/2197 * k3)
            k4 = vectorField.get_vector(p4[0], p4[1], t + 12/13 * stepSize)
            p5 = pos_3d[:2] + stepSize * (439/216 * k1 - 8 * k2 + 3680/513 * k3 - 845/4104 * k4)
            k5 = vectorField.get_vector(p5[0], p5[1], t + 1 * stepSize)
            p6 = pos_3d[:2] + stepSize * (-8/27 * k1 + 2 * k2 - 3544/2565 * k3 + 1859/4104 * k4 - 11/40 * k5)
            k6 = vectorField.get_vector(p6[0], p6[1], t + 1/2 * stepSize)
            delta = stepSize * (16/135 * k1 + 6656/12825 * k3 + 28561/56430 * k4 - 9/50 * k5 + 2/55 * k6)
        elif NumericalMethod == "Euler":
            v = vectorField.get_vector(pos_3d[0], pos_3d[1], t)
            delta = stepSize * v
        else:
            raise ValueError(f"Unknown NumericalMethod: {NumericalMethod}")
    
        delta_pos_3d = np.array([delta[0], delta[1], 0.0], dtype=np.float32)
        pos_3d = pos_3d + delta_pos_3d
        t = t + stepSize
        path.append((pos_3d.copy(), t))
    return path
    
def pathline_integration_one_direction_3D(
    vectorField: UnsteadyVectorField3D,
    start_pos:np.ndarray[np.float32,3],
    timeStart:float,
    timeEnd:float,
    stepSize:float=0.01,
    maxIterations:int=5000,
    NumericalMethod:str="RK4"):
    """
    Integrate a pathline in one direction through an unsteady vector field.
    Args:
        vectorField: UnsteadyVectorField3D 
        start_pos: [x, y,z=0] initial position
        timeStart: float, start time
        timeEnd: float, end time
        stepSize: float, integration time step
        maxIterations: int, maximum number of steps
        NumericalMethod: str, "RK4" or "Euler"
    Returns:
        List of (position, time) tuples
    """
    pos_3d = np.array(start_pos, dtype=np.float32)
    t = timeStart
    path = [(pos_3d.copy(), t)]
    direction = 1 if timeEnd > timeStart else -1
    stepSize = abs(stepSize) * direction
    for i in range(maxIterations):
        if (direction > 0 and t >= timeEnd) or (direction < 0 and t <= timeEnd) or not vectorField.IsInside(pos_3d):
            break
        if NumericalMethod == "RK4":
            v1 = vectorField.get_vector(pos_3d[0], pos_3d[1],pos_3d[2], t)
            v2 = vectorField.get_vector(pos_3d[0] + 0.5 * stepSize * v1[0], pos_3d[1] + 0.5 * stepSize * v1[1],pos_3d[2] + 0.5 * stepSize * v1[2], t + 0.5 * stepSize)
            v3 = vectorField.get_vector(pos_3d[0] + 0.5 * stepSize * v2[0], pos_3d[1] + 0.5 * stepSize * v2[1],pos_3d[2] + 0.5 * stepSize * v2[2], t + 0.5 * stepSize)
            v4 = vectorField.get_vector(pos_3d[0] + stepSize * v3[0], pos_3d[1] + stepSize * v3[1],pos_3d[2] + stepSize * v3[2], t + stepSize)
            delta = (stepSize / 6.0) * (v1 + 2 * v2 + 2 * v3 + v4)
        elif NumericalMethod == "RK5":
            k1 = vectorField.get_vector(pos_3d[0], pos_3d[1], pos_3d[2], t)
            p2 = pos_3d + stepSize * (1/4 * k1)
            k2 = vectorField.get_vector(p2[0], p2[1], p2[2], t + 1/4 * stepSize)
            p3 = pos_3d + stepSize * (3/32 * k1 + 9/32 * k2)
            k3 = vectorField.get_vector(p3[0], p3[1], p3[2], t + 3/8 * stepSize)
            p4 = pos_3d + stepSize * (1932/2197 * k1 - 7200/2197 * k2 + 7296/2197 * k3)
            k4 = vectorField.get_vector(p4[0], p4[1], p4[2], t + 12/13 * stepSize)
            p5 = pos_3d + stepSize * (439/216 * k1 - 8 * k2 + 3680/513 * k3 - 845/4104 * k4)
            k5 = vectorField.get_vector(p5[0], p5[1], p5[2], t + 1 * stepSize)
            p6 = pos_3d + stepSize * (-8/27 * k1 + 2 * k2 - 3544/2565 * k3 + 1859/4104 * k4 - 11/40 * k5)
            k6 = vectorField.get_vector(p6[0], p6[1], p6[2], t + 1/2 * stepSize)
            delta = stepSize * (16/135 * k1 + 6656/12825 * k3 + 28561/56430 * k4 - 9/50 * k5 + 2/55 * k6)
        elif NumericalMethod == "Euler":
            v = vectorField.get_vector(pos_3d[0], pos_3d[1],pos_3d[2], t)
            delta = stepSize * v
        else:
            raise ValueError(f"Unknown NumericalMethod: {NumericalMethod}")
    
        delta_pos_3d = np.array([delta[0], delta[1], delta[2]], dtype=np.float32)
        pos_3d = pos_3d + delta_pos_3d
        t = t + stepSize
        path.append((pos_3d.copy(), t))
    return path


def streamline_integration_one_direction_2D(
    vectorField: UnsteadyVectorField2D,
    start_pos,
    time,
    stepSize=0.01,
    maxIterations=5000,
    NumericalMethod="RK4",
    direction="forward"
):
    """
    Integrate a streamline in one direction through an unsteady vector field at a fixed time.
    Args:
        vectorField: UnsteadyVectorField2D instance
        start_pos: [x, y] initial position
        time: float, fixed time for streamline
        stepSize: float, integration step size
        maxIterations: int, maximum number of steps
        NumericalMethod: str, "RK4" or "Euler"
        direction: str, "forward" or "backward"
    Returns:
        List of (position, time) tuples
    """
    pos = np.array(start_pos, dtype=np.float32)
    path = [(pos.copy(), time)]
    step_direction = 1 if direction == "forward" else -1
    abs_step_size = abs(stepSize) * step_direction
    
    for i in range(maxIterations):
        if NumericalMethod == "RK4":
            v1 = vectorField.get_vector(pos[0], pos[1], time)
            v2 = vectorField.get_vector(pos[0] + 0.5 * abs_step_size * v1[0], pos[1] + 0.5 * abs_step_size * v1[1], time)
            v3 = vectorField.get_vector(pos[0] + 0.5 * abs_step_size * v2[0], pos[1] + 0.5 * abs_step_size * v2[1], time)
            v4 = vectorField.get_vector(pos[0] + abs_step_size * v3[0], pos[1] + abs_step_size * v3[1], time)
            delta = (abs_step_size / 6.0) * (v1 + 2 * v2 + 2 * v3 + v4)
        elif NumericalMethod == "RK5":
            t = time
            k1 = vectorField.get_vector(pos[0], pos[1], t)
            p2 = pos[:2] + abs_step_size * (1/4 * k1)
            k2 = vectorField.get_vector(p2[0], p2[1], t)
            p3 = pos[:2] + abs_step_size * (3/32 * k1 + 9/32 * k2)
            k3 = vectorField.get_vector(p3[0], p3[1], t)
            p4 = pos[:2] + abs_step_size * (1932/2197 * k1 - 7200/2197 * k2 + 7296/2197 * k3)
            k4 = vectorField.get_vector(p4[0], p4[1], t)
            p5 = pos[:2] + abs_step_size * (439/216 * k1 - 8 * k2 + 3680/513 * k3 - 845/4104 * k4)
            k5 = vectorField.get_vector(p5[0], p5[1], t)
            p6 = pos[:2] + abs_step_size * (-8/27 * k1 + 2 * k2 - 3544/2565 * k3 + 1859/4104 * k4 - 11/40 * k5)
            k6 = vectorField.get_vector(p6[0], p6[1], t)
            delta = abs_step_size * (16/135 * k1 + 6656/12825 * k3 + 28561/56430 * k4 - 9/50 * k5 + 2/55 * k6)
        elif NumericalMethod == "Euler":
            v = vectorField.get_vector(pos[0], pos[1], time)
            delta = abs_step_size * v
        else:
            raise ValueError(f"Unknown NumericalMethod: {NumericalMethod}")
        
        pos[0] = pos[0] + delta[0]
        pos[1] = pos[1] + delta[1]
        path.append((pos.copy(), time))
        
        # Check if we've gone out of bounds (optional safety check)
        if (pos[0] < vectorField.domainMinBoundary[0] or pos[0] > vectorField.domainMaxBoundary[0] or
            pos[1] < vectorField.domainMinBoundary[1] or pos[1] > vectorField.domainMaxBoundary[1]):
            break
    
    return path

def streamline_integration_one_direction_3D(
    vectorField: UnsteadyVectorField3D,
    start_pos,
    time,
    stepSize=0.01,
    maxIterations=5000,
    NumericalMethod="RK4",
    direction="forward"
):
    """
    Integrate a streamline in one direction through an unsteady 3D vector field at a fixed time.
    """
    pos_3d = np.array(start_pos, dtype=np.float32)
    path = [(pos_3d.copy(), time)]
    step_direction = 1 if direction == "forward" else -1
    abs_step_size = abs(stepSize) * step_direction

    for i in range(maxIterations):
        if not vectorField.IsInside(pos_3d):
            break

        if NumericalMethod == "RK4":
            v1 = vectorField.get_vector(pos_3d[0], pos_3d[1], pos_3d[2], time)
            v2_pos = pos_3d + 0.5 * abs_step_size * v1
            v2 = vectorField.get_vector(v2_pos[0], v2_pos[1], v2_pos[2], time)
            v3_pos = pos_3d + 0.5 * abs_step_size * v2
            v3 = vectorField.get_vector(v3_pos[0], v3_pos[1], v3_pos[2], time)
            v4_pos = pos_3d + abs_step_size * v3
            v4 = vectorField.get_vector(v4_pos[0], v4_pos[1], v4_pos[2], time)
            delta = (abs_step_size / 6.0) * (v1 + 2 * v2 + 2 * v3 + v4)
        elif NumericalMethod == "RK5":
            t = time
            k1 = vectorField.get_vector(pos_3d[0], pos_3d[1], pos_3d[2], t)
            p2 = pos_3d + abs_step_size * (1/4 * k1)
            k2 = vectorField.get_vector(p2[0], p2[1], p2[2], t)
            p3 = pos_3d + abs_step_size * (3/32 * k1 + 9/32 * k2)
            k3 = vectorField.get_vector(p3[0], p3[1], p3[2], t)
            p4 = pos_3d + abs_step_size * (1932/2197 * k1 - 7200/2197 * k2 + 7296/2197 * k3)
            k4 = vectorField.get_vector(p4[0], p4[1], p4[2], t)
            p5 = pos_3d + abs_step_size * (439/216 * k1 - 8 * k2 + 3680/513 * k3 - 845/4104 * k4)
            k5 = vectorField.get_vector(p5[0], p5[1], p5[2], t)
            p6 = pos_3d + abs_step_size * (-8/27 * k1 + 2 * k2 - 3544/2565 * k3 + 1859/4104 * k4 - 11/40 * k5)
            k6 = vectorField.get_vector(p6[0], p6[1], p6[2], t)
            delta = abs_step_size * (16/135 * k1 + 6656/12825 * k3 + 28561/56430 * k4 - 9/50 * k5 + 2/55 * k6)
        elif NumericalMethod == "Euler":
            v = vectorField.get_vector(pos_3d[0], pos_3d[1], pos_3d[2], time)
            delta = abs_step_size * v
        else:
            raise ValueError(f"Unknown NumericalMethod: {NumericalMethod}")

        pos_3d += delta
        path.append((pos_3d.copy(), time))

    return path


def compute_pathline_2D(args):
        vector_field, pos3d, t0, min_time, max_time, step_size, max_iteration, method = args
        forward = pathline_integration_one_direction_2D(vector_field, pos3d, t0, max_time, step_size, max_iteration, method)
        backward = pathline_integration_one_direction_2D(vector_field, pos3d, t0, min_time, step_size, max_iteration, method)
        backward = backward[::-1]
        full_path = backward + forward[1:]
        return full_path

def compute_pathline_3D(args):
    vector_field, pos3d, t0, min_time, max_time, step_size, max_iteration, method = args
    forward = pathline_integration_one_direction_3D(vector_field, pos3d, t0, max_time, step_size, max_iteration, method)
    backward = pathline_integration_one_direction_3D(vector_field, pos3d, t0, min_time, step_size, max_iteration, method)
    backward = backward[::-1]
    full_path = backward + forward[1:]
    return full_path

def compute_streamline_3D(args):
    vector_field, pos3d, time, step_size, max_iteration, method = args
    forward = streamline_integration_one_direction_3D(vector_field, pos3d, time, step_size, max_iteration, method, "forward")
    backward = streamline_integration_one_direction_3D(vector_field, pos3d, time, step_size, max_iteration, method, "backward")
    backward = backward[::-1]
    full_path = backward + forward[1:]
    return full_path

def compute_streamline_2D(args):
    vector_field, pos3d, time, step_size, max_iteration, method = args
    forward = streamline_integration_one_direction_2D(vector_field, pos3d, time, step_size, max_iteration, method, "forward")
    backward = streamline_integration_one_direction_2D(vector_field, pos3d, time, step_size, max_iteration, method, "backward")
    backward = backward[::-1]
    full_path = backward + forward[1:]
    return full_path
