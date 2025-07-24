import torch
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from FLowUtils.VectorField3d import UnsteadyVectorField3D

class LagrangianVortexCore:
    """
    Lagrangian function L(q, q_dot).
    """
    def __init__(self, v_field: UnsteadyVectorField3D, w_field: UnsteadyVectorField3D, params: dict):
        """
        Initialization.

        Args:
            v_field (UnsteadyVectorField3D): Main vector field v.
            w_field (UnsteadyVectorField3D): Observer/reference vector field w.
            params (dict): Dictionary containing weights lambda, mu, epsilon1.
        """
        self.v_field = v_field
        self.w_field = w_field
        self.lambda_ = params.get('lambda', 0.0)
        self.mu = params.get('mu', 0.0)
        self.epsilon1 = params.get('epsilon1', 0.0)

    def compute_L(self, q: torch.Tensor, q_dot: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the value of the Lagrangian function at point q, tangent q_dot, and time t.
        All inputs should be batched.

        Args:
            q (torch.Tensor): Position of shape (N, 3).
            q_dot (torch.Tensor): Tangent (unit vector) of shape (N, 3).
            t (torch.Tensor): Time of shape (N,).

        Returns:
            torch.Tensor: Lagrangian function value of shape (N,).
        """
        # Ensure q_dot requires grad, as we will differentiate with respect to it
        q_dot.requires_grad_(True)
        
        # Get vector field values v and w
        v = self.v_field.get_vectors_at_points(q, t)
        w = self.w_field.get_vectors_at_points(q, t)
        
        v_minus_w = v - w
        
        # --- Compute main component of L ---
        # According to Eq. 36: ||(u-w)(q,t) x q_dot||^2
        # Note: ||A x B||^2 == ||A||^2 * ||B||^2 - (A . B)^2
        # Since ||q_dot||=1, it simplifies to ||v-w||^2 - ((v-w).q_dot)^2
        # This is numerically more stable, but using cross directly is also fine.
        term_align = torch.linalg.norm(torch.cross(v_minus_w, q_dot, dim=1), dim=1)**2
        
        L = term_align
        
        # --- Add other regularization terms mentioned in the paper (Eq. 34) ---
        
        # A(q, q_dot) term, depends on observed acceleration 'a'
        if self.lambda_ > 0:
            # a(v,w) = dv/dt - dw/dt + (nabla_v)v - 2*(nabla_w)v + (nabla_w)w (Eq. 12)
            # To compute dv/dt, we need to use autograd again. This will slow down computation.
            # For simplicity, we assume dv/dt and dw/dt are zero (for steady or slowly varying fields).
            # A full implementation would require computing time derivatives.
            nabla_v = self.v_field.get_jacobian_at_points(q, t)
            nabla_w = self.w_field.get_jacobian_at_points(q, t)
            
            # (nabla_v)v -> bmm(nabla_v, v.unsqueeze(-1)).squeeze(-1)
            term_nabla_v_v = torch.bmm(nabla_v, v.unsqueeze(-1)).squeeze(-1)
            term_nabla_w_v = torch.bmm(nabla_w, v.unsqueeze(-1)).squeeze(-1)
            term_nabla_w_w = torch.bmm(nabla_w, w.unsqueeze(-1)).squeeze(-1)
            
            # Assume dv/dt = 0, dw/dt = 0
            a = term_nabla_v_v - 2 * term_nabla_w_v + term_nabla_w_w
            
            term_A = torch.linalg.norm(torch.cross(a, q_dot, dim=1), dim=1)**2
            L = L + self.lambda_ * term_A
            
        # D(q, q_dot) and R(q, q_dot) terms can be added here
        # ...
        
        return L
    
class VariationalIntegrator:
    """
    Integrate to generate vortex corelines using variational principle and RK4 method.
    """
    def __init__(self, lagrangian: LagrangianVortexCore, 
                 q0: torch.Tensor, q_dot0: torch.Tensor, t: float, 
                 ds: float = 0.01, device='cpu'):
        """
        Initialize the integrator.

        Args:
            lagrangian (LagrangianVortexCore): Lagrangian object defining L.
            q0 (torch.Tensor): Initial position (1, 3).
            q_dot0 (torch.Tensor): Initial tangent (1, 3).
            t (float): Physical time.
            ds (float): Integration step size along curve s.
            device (str): Compute device.
        """
        self.lagrangian = lagrangian
        self.q = q0.clone().to(device)
        self.q_dot = q_dot0.clone().to(device)
        self.t = torch.tensor([t], dtype=torch.float32, device=device)
        self.ds = ds
        self.device = torch.device(device)

    def _get_orthonormal_basis(self, v: torch.Tensor) -> torch.Tensor:
        """Create an orthonormal basis B = [b1, b2, b3] for unit vector v."""
        b1 = v / torch.linalg.norm(v)
        
        # Choose a reference vector not collinear with b1
        ref = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        if torch.abs(torch.dot(b1.flatten(), ref)) > 0.99:
            ref = torch.tensor([0.0, 1.0, 0.0], device=self.device)
            
        b3 = torch.cross(b1.flatten(), ref)
        b3 /= torch.linalg.norm(b3)
        
        b2 = torch.cross(b3, b1.flatten())
        
        return torch.stack([b1.flatten(), b2, b3], dim=1)

    def _compute_q_ddot(self, q: torch.Tensor, q_dot: torch.Tensor) -> torch.Tensor:
        """
        Solve Euler-Lagrange equation to get q_ddot. This is the most critical part.
        See paper Eq. 46, 47.
        """
        # --- 1. Compute required derivatives of L ---
        
        # Compute dL/dq and dL/dq_dot
        L_val = self.lagrangian.compute_L(q, q_dot, self.t)
        
        # create_graph=True allows us to compute higher-order derivatives (Hessian)
        dL_d_vars = torch.autograd.grad(L_val, [q, q_dot], create_graph=True)[0]
        dL_dq = dL_d_vars[0]       # (1, 3)
        
        # To compute mixed Hessian d/dq(dL/dq_dot), we need a new function
        def dL_dqdot_func(q_in):
             return torch.autograd.grad(self.lagrangian.compute_L(q_in, q_dot, self.t), q_dot, create_graph=True)[0]

        H_q_qdot = torch.autograd.functional.jacobian(dL_dqdot_func, q) # (1,3,1,3) -> (3,3)
        H_q_qdot = H_q_qdot.squeeze()
        
        # --- 2. Computation in local spherical coordinate system ---
        B = self._get_orthonormal_basis(q_dot)
        b2, b3 = B[:, 1], B[:, 2]

        # Define a function L_tilde(theta, phi) to compute 2x2 Hessian
        def L_tilde_func(spherical_coords: torch.Tensor):
            theta, phi = spherical_coords[0], spherical_coords[1]
            # Convert from spherical to local Cartesian coordinates
            # (cos(th)sin(ph), sin(th)sin(ph), cos(ph)) -> perturbation direction
            # At our reference point (th=0, ph=pi/2), perturbation is in b2, b3 directions
            # Here we directly define L_tilde(delta_b2, delta_b3)
            # perturbed_qdot = q_dot + delta_b2*b2 + delta_b3*b3
            # H_tilde is d^2L / d(delta)^2
            # This is equivalent to computing Hessian of L w.r.t. q_dot, then projecting onto the plane spanned by b2, b3
            return self.lagrangian.compute_L(q, q_dot, self.t)

        # Compute H_qdot_qdot (d^2L/dq_dot^2)
        H_qdot_qdot = torch.autograd.functional.hessian(lambda qd: self.lagrangian.compute_L(q, qd, self.t), q_dot)
        H_qdot_qdot = H_qdot_qdot.squeeze() # (3,3)

        # Project onto b2, b3 plane to get H_tilde (Eq. 46)
        H_tilde = torch.zeros(2, 2, device=self.device)
        H_tilde[0, 0] = torch.dot(b2, H_qdot_qdot @ b2)
        H_tilde[0, 1] = torch.dot(b2, H_qdot_qdot @ b3)
        H_tilde[1, 0] = torch.dot(b3, H_qdot_qdot @ b2)
        H_tilde[1, 1] = torch.dot(b3, H_qdot_qdot @ b3)

        # Compute right-hand side r_hat (Eq. 47)
        # r_hat = [b2, b3]^T * (dL/dq - H_q_qdot * q_dot)
        # There is a bug in the C++ code, which used dL/dq_dot, but the paper derivation seems to require q_dot.
        # The d/ds(dL/dq_dot) term is H_q_q * q_dot + H_q_qdot * q_ddot
        # Here we use q_dot
        rhs_vec = dL_dq.flatten() - (H_q_qdot @ q_dot.flatten())
        
        r_hat = torch.zeros(2, device=self.device)
        r_hat[0] = torch.dot(b2, rhs_vec)
        r_hat[1] = torch.dot(b3, rhs_vec)
        
        # --- 3. Solve linear system ---
        try:
            # Check condition number
            cond_num = torch.linalg.cond(H_tilde)
            if cond_num > 1e6:
                print(f"Warning: Hessian is ill-conditioned. Condition number: {cond_num:.2e}")
            
            q_ddot_spherical = torch.linalg.solve(H_tilde, r_hat)
        except torch.linalg.LinAlgError:
            print("Error: Hessian matrix is singular. Cannot solve for q_ddot. Returning zero.")
            return torch.zeros_like(q)
            
        # --- 4. Convert back to Cartesian coordinates ---
        # q_ddot = b2 * q_ddot_spherical[0] + b3 * q_ddot_spherical[1]
        q_ddot = B[:, 1:] @ q_ddot_spherical.unsqueeze(1)
        
        return q_ddot.T # (1, 3)

    def _rk4_step(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform one RK4 integration step."""
        ds = self.ds
        
        # k1
        q_dot_k1 = self.q_dot
        q_ddot_l1 = self._compute_q_ddot(self.q, self.q_dot)
        
        # k2
        q_k2 = self.q + 0.5 * ds * q_dot_k1
        q_dot_k2 = (self.q_dot + 0.5 * ds * q_ddot_l1).requires_grad_()
        q_ddot_l2 = self._compute_q_ddot(q_k2, q_dot_k2)

        # k3
        q_k3 = self.q + 0.5 * ds * q_dot_k2
        q_dot_k3 = (self.q_dot + 0.5 * ds * q_ddot_l2).requires_grad_()
        q_ddot_l3 = self._compute_q_ddot(q_k3, q_dot_k3)

        # k4
        q_k4 = self.q + ds * q_dot_k3
        q_dot_k4 = (self.q_dot + ds * q_ddot_l3).requires_grad_()
        q_ddot_l4 = self._compute_q_ddot(q_k4, q_dot_k4)

        # Update
        q_new = self.q + (ds / 6.0) * (q_dot_k1 + 2*q_dot_k2 + 2*q_dot_k3 + q_dot_k4)
        q_dot_new = self.q_dot + (ds / 6.0) * (q_ddot_l1 + 2*q_ddot_l2 + 2*q_ddot_l3 + q_ddot_l4)
        
        # Normalize q_dot to maintain unit length constraint
        q_dot_new = q_dot_new / torch.linalg.norm(q_dot_new)
        
        return q_new, q_dot_new.requires_grad_()

    def integrate(self, num_steps: int) -> np.ndarray:
        """
        Perform integration.

        Args:
            num_steps (int): Total number of integration steps.

        Returns:
            np.ndarray: Coreline point coordinates of shape (num_steps+1, 3).
        """
        coreline = [self.q.cpu().numpy().flatten()]
        
        for step in range(num_steps):
            try:
                q_new, q_dot_new = self._rk4_step()
                
                # Update state
                self.q = q_new
                self.q_dot = q_dot_new
                
                coreline.append(self.q.cpu().numpy().flatten())

                # Check for divergence
                if not torch.all(torch.isfinite(self.q)):
                    print(f"Integration diverged at step {step+1}. Stopping.")
                    break
                    
            except Exception as e:
                print(f"An error occurred at step {step+1}: {e}")
                import traceback
                traceback.print_exc()
                break

        return np.array(coreline)
    


def create_double_gyre_field(grid_res: int, time_res: int, device: str) -> Tuple[np.ndarray, list, list, list]:
    """Create a 2.5D double gyre field for demonstration."""
    t_max = 1.0
    domain_min = [-1.0, -1.0, -0.5]
    domain_max = [ 1.0,  1.0,  0.5]
    
    x = np.linspace(domain_min[0], domain_max[0], grid_res)
    y = np.linspace(domain_min[1], domain_max[1], grid_res)
    z = np.linspace(domain_min[2], domain_max[2], grid_res // 2)
    t = np.linspace(0, t_max, time_res)
    
    xx, yy, zz, tt = np.meshgrid(x, y, z, t, indexing='ij')

    # Double gyre parameters
    A = 0.1
    epsilon = 0.25
    omega = 2 * np.pi / t_max

    a = epsilon * np.sin(omega * tt)
    b = 1 - 2 * epsilon * np.sin(omega * tt)
    f = a * xx**2 + b * xx
    dfdx = 2 * a * xx + b
    
    vx = -np.pi * A * np.sin(np.pi * f) * np.cos(np.pi * yy)
    vy =  np.pi * A * np.cos(np.pi * f) * np.sin(np.pi * yy) * dfdx
    vz = np.zeros_like(vx) + 0.1 # Add a small z component

    # Shape: (grid, grid, grid/2, time, 3)
    field_data = np.stack([vx, vy, vz], axis=-1)
    # Convert to (T, D, H, W, 3)
    field_data = np.transpose(field_data, (3, 2, 1, 0, 4))
    
    return field_data, domain_min, domain_max, [0, t_max]

# ---- Main program ----
if __name__ == '__main__':
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")

    # 1. Create vector field data
    GRID_RESOLUTION = 32
    TIME_RESOLUTION = 16
    
    v_field_data, domain_min, domain_max, time_range = create_double_gyre_field(GRID_RESOLUTION, TIME_RESOLUTION, DEVICE)
    
    # In this example, we use a stationary reference frame (w=0)
    w_field_data = np.zeros_like(v_field_data)
    
    # 2. Instantiate vector field objects
    v_field = UnsteadyVectorField3D(v_field_data, domain_min, domain_max, time_range, device=DEVICE)
    w_field = UnsteadyVectorField3D(w_field_data, domain_min, domain_max, time_range, device=DEVICE)

    # 3. Set up Lagrangian function
    # In this simple example, we only use the main alignment term
    lagrangian_params = {'lambda': 0.0, 'mu': 0.0, 'epsilon1': 0.0}
    lagrangian = LagrangianVortexCore(v_field, w_field, lagrangian_params)

    # 4. Initialize integrator
    # Initial point and direction
    q_initial = torch.tensor([[0.5, 0.5, 0.0]], dtype=torch.float32, device=DEVICE)
    # Initial q_dot is usually aligned with (v-w) direction at this point
    v0 = v_field.get_vectors_at_points(q_initial, torch.tensor([0.0], device=DEVICE))
    q_dot_initial = (v0 / torch.linalg.norm(v0)).requires_grad_()
    
    
    integrator = VariationalIntegrator(
        lagrangian=lagrangian,
        q0=q_initial,
        q_dot0=q_dot_initial,
        t=0.0, # Physical time
        ds=0.05, # Curve step size
        device=DEVICE
    )

    # 5. Run integration
    print("Starting integration...")
    coreline_points = integrator.integrate(num_steps=200)
    print(f"Integration finished. Found {len(coreline_points)} points.")

    # 6. Visualize result
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(coreline_points[:, 0], coreline_points[:, 1], coreline_points[:, 2], '-o', markersize=2)
    
    # Plot starting point
    ax.scatter(q_initial[0,0].cpu(), q_initial[0,1].cpu(), q_initial[0,2].cpu(), color='red', s=100, label='Start Point')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Extracted Vortex Coreline using Variational Method')
    ax.legend()
    plt.show()