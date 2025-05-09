



import numpy as np
from scipy.optimize import minimize
# Define the new input vector field v(x, y, t)
def v_field(x, y, t):
    vx = - y + np.sin(t)*(x**2 - y**2)
    vy = x + np.cos(t)*(2*x*y)
    return np.array([vx, vy])

# Compute the time derivative of v for the new field
def dv_dt(x, y, t):
    # v1 = -y + sin(t)*(x^2 - y^2)  --> d/dt: cos(t)*(x^2-y^2)
    # v2 = x + cos(t)*(2xy)         --> d/dt: -sin(t)*(2xy)
    return np.array([np.cos(t)*(x**2 - y**2), -np.sin(t)*(2*x*y)])


# Compute the gradient (Jacobian) of the new v
def grad_v(x, y, t):
    # For v1 = -y + sin(t)*(x^2 - y^2):
    # dvx/dx = 2x*sin(t)
    # dvx/dy = -1 - 2y*sin(t)
    #
    # For v2 = x + cos(t)*(2xy):
    # dvy/dx = 1 + 2y*cos(t)
    # dvy/dy = 2x*cos(t)
    dvx_dx = 2*x*np.sin(t)
    dvx_dy = -1 - 2*y*np.sin(t)
    dvy_dx = 1 + 2*y*np.cos(t)
    dvy_dy = 2*x*np.cos(t)
    return np.array([[dvx_dx, dvx_dy],
                     [dvy_dx, dvy_dy]])








# Define the observer field u(x, y, t) with unknown coefficients
def u_field(x, y, t, a, b, c, d, e, f, g):
    ux = -c * y + a + d * x + f * x**2
    uy = c * x + b + e * y + g * y**2
    return np.array([ux, uy])


def du_dt(x, y, t, da, db, dc, dd, de, df, dg):
    return np.array([da + dd * x + df * x**2, db + de * y + dg * y**2])


# Compute the gradient of u
def grad_u(x, y, t, c, d, e, f, g):
    dux_dx = d + 2 * f * x
    dux_dy = -c
    duy_dx = c
    duy_dy = e + 2 * g * y
    return np.array([[dux_dx, dux_dy], [duy_dx, duy_dy]])

# Compute the Lie derivative term
def lie_derivative(x, y, t, a, b, c, d, e, f, g, da, db, dc, dd, de, df, dg):
    v = v_field(x, y, t)
    u = u_field(x, y, t, a, b, c, d, e, f, g)
    dvdt = dv_dt(x, y, t)
    dudt = du_dt(x, y, t, da, db, dc, dd, de, df, dg)
    gradV = grad_v(x, y, t)
    gradU = grad_u(x, y, t, c, d, e, f, g)
    
    D = dvdt - dudt + gradV @ u - gradU @ v
    return np.sum(D**2)

# Compute the Killing energy term
def killing_energy(x, y, t, c, d, e, f, g):
    gradU = grad_u(x, y, t, c, d, e, f, g)
    symGradU = gradU + gradU.T
    return np.sum(symGradU**2)

# Define the loss function
def loss(params, x, y, t, k1):
    a, b, c, d, e, f, g, da, db, dc, dd, de, df, dg, theta = params
    return lie_derivative(x, y, t, a, b, c, d, e, f, g, da, db, dc, dd, de, df, dg) + k1 * killing_energy(x, y, t, c, d, e, f, g)

# Compute the new observed time derivative D' with the additional freedom theta
def observed_time_derivative_prime(x, y, t, a, b, c, d, e, f, g, da, db, dc, dd, de, df, dg, theta):
    v = v_field(x, y, t)
    u = u_field(x, y, t, a, b, c, d, e, f, g)
    dvdt = dv_dt(x, y, t)
    dudt = du_dt(x, y, t, da, db, dc, dd, de, df, dg)
    gradV = grad_v(x, y, t)
    gradU = grad_u(x, y, t, c, d, e, f, g)

    # Compute the rotation matrix q for theta
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    q = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

    # Compute Omega
    Omega = 0.5 * (q.T @ (gradU + gradU.T) @ q) + (gradU - gradU.T)

    # Compute D'
    D_prime = dvdt - dudt + gradV @ u - gradU @ v - Omega @ (v - u)
    
    return np.sum(D_prime**2)

# Update the loss function to include D'
def loss_prime(params, x, y, t, k1):
    a, b, c, d, e, f, g, da, db, dc, dd, de, df, dg, theta = params
    return observed_time_derivative_prime(x, y, t, a, b, c, d, e, f, g, da, db, dc, dd, de, df, dg, theta) + k1 * killing_energy(x, y, t, c, d, e, f, g)

# Define spatial grid
XGrid, YGrid, TGrid = 16, 16, 10  # Number of grid points
x_samples = np.linspace(-1, 1, XGrid)  
y_samples = np.linspace(-1, 1, YGrid)
X, Y = np.meshgrid(x_samples, y_samples)
positions = np.vstack([X.ravel(), Y.ravel()]).T  # Flattened grid points

# Define time grid
time_samples = np.linspace(0, np.pi / 2, TGrid)

# Optimization for all spatial and temporal samples
D_val_total, D_prime_total = 0, 0
killing_total,killing_total_prime=0,0
optimized_params_original = []
optimized_params_prime = []

for t in time_samples:
    # Flatten the grid for optimization
    def total_loss_prime(params):
        return np.sum([loss_prime(params, x, y, t, 0.1) for x, y in positions])
    
    def total_loss(params):
        return np.sum([loss(params, x, y, t, 0.1) for x, y in positions])

    # Initial guess for parameters
    initial_guess_original = np.zeros(15)  # Extra parameter for theta
    # Optimize parameters for all points at once
    result_original = minimize(total_loss, initial_guess_original, method='BFGS')
    optimized_params_original.append(result_original.x)
    # Extract optimized parameters
    a, b, c, d, e, f, g, da, db, dc, dd, de, df, dg, theta = result_original.x
    # Compute loss values for all points
    D_val = np.sum([lie_derivative(x, y, t, a, b, c, d, e, f, g, da, db, dc, dd, de, df, dg) for x, y in positions])
    D_val_total += D_val
    Kval = np.sum([killing_energy(x, y, t, c, d, e, f, g) for x, y in positions])
    killing_total+=Kval

    # Initial guess for parameters
    initial_guess_prime = np.zeros(15)  # Extra parameter for theta
    # Optimize parameters for all points at once
    result_prime = minimize(total_loss_prime, initial_guess_prime, method='BFGS')
    optimized_params_prime.append(result_prime.x)
    # Extract optimized parameters
    a, b, c, d, e, f, g, da, db, dc, dd, de, df, dg, theta = result_prime.x
    # Compute loss values for all points
    D_prime_val = np.sum([observed_time_derivative_prime(x, y, t, a, b, c, d, e, f, g, da, db, dc, dd, de, df, dg, theta) for x, y in positions])
    D_prime_total += D_prime_val
    Kval = np.sum([killing_energy(x, y, t, c, d, e, f, g) for x, y in positions])
    killing_total_prime+=Kval



# Return results
print("Total D:", D_val_total)
print("Total D':", D_prime_total)

print("Total K:", killing_total)
print("Total K':", killing_total_prime)

# print(optimized_params_prime,optimized_params_original)
# Total D: 170.2911493542086
# Total D': 74.85263834521331