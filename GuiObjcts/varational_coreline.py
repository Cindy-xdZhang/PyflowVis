import torch
from torchdiffeq import odeint as odeint_torch
from FLowUtils.VectorField3d import UnsteadyVectorField3D
import torch
import torch.nn.functional as F
import torch
import numpy as np


class UnsteadyVectorField2D_Torch(torch.nn.Module):
    def __init__(self, data_tensor, domain_min, domain_max, t_min, t_max):
        super().__init__()
        self.register_buffer('data', data_tensor)
        self.domain_min = torch.tensor(domain_min, dtype=torch.float32)
        self.domain_max = torch.tensor(domain_max, dtype=torch.float32)
        self.t_min = t_min
        self.t_max = t_max
        self.time_steps, self.height, self.width, _ = data_tensor.shape
        self.register_buffer('data_permuted', self.data.permute(0, 3, 1, 2).contiguous())

    def get_vector(self, x, y, t):
        device = x.device
        self.domain_min = self.domain_min.to(device)
        self.domain_max = self.domain_max.to(device)
        
        x_norm = (x - self.domain_min[0]) / (self.domain_max[0] - self.domain_min[0])
        y_norm = (y - self.domain_min[1]) / (self.domain_max[1] - self.domain_min[1])
        t_norm = (t - self.t_min) / (self.t_max - self.t_min)
        
        grid_x = 2.0 * x_norm - 1.0
        grid_y = 2.0 * y_norm - 1.0
        
        coords = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(1).unsqueeze(1)
        
        t_idx_float = t_norm * (self.time_steps - 1)
        t_idx_floor = torch.floor(t_idx_float).long()
        t_idx_ceil = torch.ceil(t_idx_float).long()
        
        t_idx_floor = torch.clamp(t_idx_floor, 0, self.time_steps - 1)
        t_idx_ceil = torch.clamp(t_idx_ceil, 0, self.time_steps - 1)
        
        field_floor = self.data_permuted[t_idx_floor]
        field_ceil = self.data_permuted[t_idx_ceil]
        
        batch_size = coords.shape[0]
        field_floor_expanded = field_floor.unsqueeze(0).expand(batch_size, -1, -1, -1)
        field_ceil_expanded = field_ceil.unsqueeze(0).expand(batch_size, -1, -1, -1)

        vec_floor = F.grid_sample(field_floor_expanded, coords, align_corners=False, mode='bilinear').squeeze(-1).squeeze(-1)
        vec_ceil = F.grid_sample(field_ceil_expanded, coords, align_corners=False, mode='bilinear').squeeze(-1).squeeze(-1)
        
        if vec_floor.dim() == 1:
            vec_floor = vec_floor.unsqueeze(0)
            vec_ceil = vec_ceil.unsqueeze(0)

        weight = (t_idx_float - t_idx_floor.float()).reshape(-1, 1)
        vec = vec_floor * (1 - weight) + vec_ceil * weight
        return vec

    def IsInside(self, pos):
        return (
            (pos[..., 0] >= self.domain_min[0]) & (pos[..., 0] <= self.domain_max[0]) &
            (pos[..., 1] >= self.domain_min[1]) & (pos[..., 1] <= self.domain_max[1])
        )

class VectorFieldODE(torch.nn.Module):
    def __init__(self, vector_field):
        super().__init__()
        self.vector_field = vector_field

    def forward(self, t, pos):
        if self.vector_field.data.is_cuda and not pos.is_cuda:
            pos = pos.cuda()
        if pos.shape[-1] == 2:#(x,y)
            return self.vector_field.get_vector(pos[:, 0], pos[:, 1], t)
    

class DifferentiableVectorFieldInterpolator(torch.autograd.Function):
    """
    A custom autograd Function to wrap scipy's RegularGridInterpolator,
    making it differentiable with respect to the query points.
    This interpolator works with a 3D vector field defined on a regular grid.
    """

    @staticmethod
    def forward(ctx, query_points, grid_points, grid_values):
        """
        Forward pass: Interpolate vector values at query_points.
        
        Args:
            ctx: Context object to save information for backward pass.
            query_points (torch.Tensor): Tensor of shape (N, 3) with coordinates to query.
            grid_points (tuple of np.ndarray): Tuple of 3 arrays (grid_x, grid_y, grid_z).
            grid_values (np.ndarray): Array of shape (Nx, Ny, Nz, 3) holding the vector values.
        
        Returns:
            torch.Tensor: Interpolated vectors of shape (N, 3).
        """
        # Ensure inputs are on CPU and in numpy format for scipy
        query_points_np = query_points.detach().cpu().numpy()
        
        # Create the interpolator object. For efficiency, this should be created
        # once outside and passed in, but for clarity it's here.
        # In a real application, the interpolator would be an attribute of the main class.
        interpolator = RegularGridInterpolator(grid_points, grid_values, method='linear', bounds_error=False, fill_value=0.0)
        
        # Perform interpolation
        interpolated_values_np = interpolator(query_points_np)
        
        # Convert back to torch tensor
        interpolated_values = torch.from_numpy(interpolated_values_np).to(query_points.device, dtype=query_points.dtype)
        
        # Save necessary data for backward pass
        ctx.save_for_backward(query_points)
        ctx.interpolator = interpolator # Storing the interpolator object
        
        return interpolated_values

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Compute the gradient of the interpolated values w.r.t. query_points.
        
        The gradient of the output w.r.t the input query_points is the Jacobian of the
        interpolation function. For linear interpolation, this Jacobian is piecewise constant.
        We can approximate it using finite differences on the interpolator itself.
        
        Args:
            ctx: Context object with saved tensors.
            grad_output (torch.Tensor): Gradient of the loss w.r.t. the output of this function. Shape (N, 3).
            
        Returns:
            torch.Tensor: Gradient of the loss w.r.t. query_points. Shape (N, 3).
            (None, None): Gradients for grid_points and grid_values are not needed.
        """
        query_points, = ctx.saved_tensors
        interpolator = ctx.interpolator
        
        # Small epsilon for finite differences
        eps = 1e-6
        
        # Move to CPU for numpy operations
        query_points_np = query_points.detach().cpu().numpy()
        grad_output_np = grad_output.cpu().numpy()
        
        # Initialize Jacobian matrix for the batch
        # J_batch[i] will be the 3x3 Jacobian for the i-th query point
        num_points = query_points_np.shape
        J_batch = np.zeros((num_points, 3, 3)) # (N, output_dim, input_dim)
        
        # Compute Jacobian for each query point
        for i in range(3): # Iterate over input dimensions (x, y, z)
            # Perturb in the positive direction
            q_plus = query_points_np.copy()
            q_plus[:, i] += eps
            v_plus = interpolator(q_plus)
            
            # Perturb in the negative direction
            q_minus = query_points_np.copy()
            q_minus[:, i] -= eps
            v_minus = interpolator(q_minus)
            
            # Central difference to get the i-th column of the Jacobian
            J_column = (v_plus - v_minus) / (2 * eps)
            J_batch[:, :, i] = J_column
            
        # The gradient we need to return is grad_input = grad_output @ J
        # For a batch, this is computed element-wise:
        # grad_input[n, j] = sum_k(grad_output[n, k] * J_batch[n, k, j])
        grad_input_np = np.einsum('nk,nkj->nj', grad_output_np, J_batch)
        
        grad_input = torch.from_numpy(grad_input_np).to(query_points.device, dtype=query_points.dtype)
        
        # Gradients for grid_points and grid_values are None as they are not inputs to be differentiated against
        return grad_input, None, None





class VariationalCorelineExtractor:
    def __init__(self, vector_field:UnsteadyVectorField3D, config: dict):
        """
        Initializes the extractor.
        
        Args:
            vector_field: An object that provides access to the vector field data
                          and its differentiable interpolator.
            config: A dictionary with parameters like lambda, mu, epsilon, t0, t_T, etc.
        """
        self.v_field = vector_field
        self.config = config
        self.device = vector_field.device
        self.dtype = torch.float64 # Use float64 for numerical stability

        # Unpack config parameters
        self.lambda_ = self.config.get('lambda', 1.0)
        self.mu_ = self.config.get('mu', 1.0)
        self.epsilon_ = self.config.get('epsilon', 1.0)
        self.t0 = self.config.get('t0', 0.0)
        self.t_T = self.config.get('t_T', 1.0)
        self.time_steps = self.config.get('time_steps', 10)
        
        # For now, we assume w=0 (no frame change) for simplicity.
        # A full implementation would handle a differentiable field w.
        self.w_field = None 

    def _get_observed_velocity(self, q, t):
        # q is shape (N, 3), t is a scalar
        # In a batch setting, t could be (N, 1)
        # For now, assume batch over q, single t
        
        # Create a time tensor for each point in q
        t_tensor = torch.full((q.shape, 1), t, device=self.device, dtype=self.dtype)
        qt = torch.cat([q, t_tensor], dim=1) # Shape (N, 4) for a spacetime field
        
        v = self.v_field.interpolate(q) # Assuming v_field interpolator handles this
        
        if self.w_field:
            w = self.w_field.interpolate(q)
            return v - w
        return v

    def _get_observed_acceleration(self, q, t):
        # a(v,w) = D/Dt(v-w) + grad(v-w)(v-w)
        # This is a complex term. For now, let's use a placeholder.
        # A full implementation requires derivatives of the vector field.
        # PyTorch's autograd can compute grad(v-w) if the interpolator is differentiable.
        
        # Let's compute grad(v-w) w.r.t. q
        q_clone = q.clone().detach().requires_grad_(True)
        v_minus_w = self._get_observed_velocity(q_clone, t)
        
        grad_v_minus_w = torch.zeros(q.shape, 3, 3, device=self.device, dtype=self.dtype)
        for i in range(3):
            grad_v_minus_w[:, :, i] = torch.autograd.grad(v_minus_w[:, i].sum(), q_clone, create_graph=True)

        # a_geom = grad(v-w)(v-w)
        a_geom = torch.einsum('nij,nj->ni', grad_v_minus_w, v_minus_w)
        
        # D/Dt term is more complex, involving time derivatives of the field itself.
        # Let's assume it's zero for a steady field for now.
        # D_Dt = partial_t(v-w) + L_w(v-w)
        # For w=0, D_Dt = partial_t(v)
        # This requires the vector field to be defined in spacetime.
        # Let's assume this term is zero for this simplified example.
        D_Dt = torch.zeros_like(a_geom)
        
        return a_geom + D_Dt

    def _compute_lagrangian_integrand(self, q, q_dot, t):
        """Computes the value inside the time integral of the Lagrangian for a single time t."""
        # Ensure q_dot is normalized
        q_dot = q_dot / torch.linalg.norm(q_dot, dim=-1, keepdim=True)
        
        # Term 1: Velocity parallelism ||(v-w) x q_dot||^2
        v_obs = self._get_observed_velocity(q, t)
        term1 = torch.linalg.norm(torch.cross(v_obs, q_dot, dim=-1), dim=-1)**2
        
        # Term 2: Acceleration parallelism ||a(v,w) x q_dot||^2
        a_obs = self._get_observed_acceleration(q, t)
        term2 = torch.linalg.norm(torch.cross(a_obs, q_dot, dim=-1), dim=-1)**2
        
        # Term 3: D term (assumed zero for now)
        term3 = torch.zeros_like(term1)
        
        # Term 4: R term (requires grad(v-w)(q_dot))
        # This is another complex term needing the Jacobian.
        # Let's use a placeholder.
        term4 = torch.zeros_like(term1)
        
        return term1 + self.lambda_ * term2 + self.mu_ * term3 + self.epsilon_ * term4

    def _compute_total_lagrangian(self, q, q_dot):
        """
        Computes the full time-integrated Lagrangian.
        This version is simplified and does not include the flow map for now.
        It evaluates the integrand at different times t for a fixed q, q_dot.
        A full implementation needs the flow map.
        """
        t_span = torch.linspace(self.t0, self.t_T, self.time_steps, device=self.device, dtype=self.dtype)
        
        # Simple trapezoidal integration
        L_values = torch.stack([self._compute_lagrangian_integrand(q, q_dot, t) for t in t_span])
        L_total = torch.trapezoid(L_values, t_span, dim=0)
        
        return L_total

    def _get_el_equation_terms(self, q, q_dot):
        """
        The core function that computes all terms for the Euler-Lagrange equation
        and solves for q_ddot.
        """
        q.requires_grad_(True)
        q_dot.requires_grad_(True)

        # 1. Define a function for the Hessian calculation
        def L_func(q_dot_var):
            # The Lagrangian should be a scalar for hessian computation
            return self._compute_total_lagrangian(q, q_dot_var).sum()

        # 2. Compute the 3x3 Hessian H = d^2L / dq_dot^2
        H_3x3 = torch.autograd.functional.hessian(L_func, q_dot, create_graph=False)
        # H_3x3 shape will be (N, 3, N, 3) for batch size N. We handle N=1 for now.
        H_3x3 = H_3x3.squeeze() # From (1,3,1,3) to (3,3)

        # 3. Compute dL/dq
        L_total = self._compute_total_lagrangian(q, q_dot)
        dL_dq = torch.autograd.grad(L_total.sum(), q, create_graph=False)
        # 4. Compute the mixed derivative term: d/ds(dL/dq_dot) = (d^2L/dq_dotdq) @ q_dot
        # This is more involved. A simplification from the paper (Eq. 40) is used.
        # r = dL/dq - (d^2L/dq_dot dq) @ q_dot
        # The mixed term is hard to get directly. The paper's formulation avoids it.
        # Let's follow the paper's final form (Eq. 43, 46, 47)
        
        # 5. Solve the constrained 2x2 system
        q_dot_norm = q_dot / torch.linalg.norm(q_dot) # Ensure unit vector
        
        # Create an orthonormal basis B = [b1, b2, b3] where b1 = q_dot
        b1 = q_dot_norm.squeeze()
        # Find a vector not parallel to b1
        tmp = torch.tensor([1.0, 0, 0], device=self.device, dtype=self.dtype)
        if torch.allclose(torch.abs(torch.dot(tmp, b1)), torch.tensor(1.0)):
            tmp = torch.tensor([0, 1.0, 0], device=self.device, dtype=self.dtype)
        
        b2 = torch.cross(b1, tmp)
        b2 = b2 / torch.linalg.norm(b2)
        b3 = torch.cross(b1, b2)
        
        B = torch.stack([b1, b2, b3], dim=1) # Basis matrix

        # Project the Hessian and the RHS term dL/dq
        # H_hat = B_perp^T @ H_3x3 @ B_perp where B_perp = [b2, b3]
        B_perp = torch.stack([b2, b3], dim=1)
        H_2x2 = B_perp.T @ H_3x3 @ B_perp

        # RHS_bar = B_perp^T @ (dL/dq)
        # The full RHS from Eq. 43 is more complex, let's use the simplified one from Eq. 47
        r_bar = B_perp.T @ dL_dq.squeeze()

        # Solve the 2x2 system: H_hat * q_ddot_local = r_bar
        try:
            q_ddot_local = torch.linalg.solve(H_2x2, r_bar)
        except torch.linalg.LinAlgError:
            # If H_2x2 is singular, use pseudo-inverse as a fallback
            print("Warning: 2x2 Hessian is singular, using pseudo-inverse.")
            H_2x2_inv = torch.linalg.pinv(H_2x2)
            q_ddot_local = H_2x2_inv @ r_bar

        # Transform q_ddot back to global coordinates
        q_ddot = B_perp @ q_ddot_local
        
        # Detach from graph to use in standard numerical integrator
        return q_ddot.detach()

    def integrate_curve(self, q0, q_dot0, s_end, ds):
        """
        Integrates the curve using a simple forward Euler for demonstration.
        A better choice is RK4 or using torchdiffeq.
        """
        q = q0.clone().to(self.device, self.dtype)
        q_dot = q_dot0.clone().to(self.device, self.dtype)
        
        curve_q = [q.cpu().numpy()]
        curve_q_dot = [q_dot.cpu().numpy()]
        
        num_steps = int(s_end / ds)
        for s_idx in range(num_steps):
            q_dot = q_dot / torch.linalg.norm(q_dot) # Re-normalize at each step
            
            # Get acceleration
            q_ddot = self._get_el_equation_terms(q.unsqueeze(0), q_dot.unsqueeze(0))
            
            # Update state (Forward Euler)
            q_dot_new = q_dot + q_ddot * ds
            q_new = q + q_dot * ds # Use old q_dot for position update
            
            q, q_dot = q_new, q_dot_new
            
            curve_q.append(q.cpu().numpy())
            curve_q_dot.append(q_dot.cpu().numpy())
            
            if torch.isnan(q).any() or torch.isinf(q).any():
                print(f"Integration failed at step {s_idx}: NaN or Inf detected.")
                break

        return np.array(curve_q).squeeze(), np.array(curve_q_dot).squeeze()
