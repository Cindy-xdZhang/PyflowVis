import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint as odeint_torch
from FLowUtils.VectorField3d import UnsteadyVectorField3D
import numpy as np

class UnsteadyVectorField3D_Torch(nn.Module):
    """
    A pure PyTorch, differentiable interpolator for a 3D unsteady vector field.
    It uses a manual, twice-differentiable trilinear interpolation for space,
    and linear interpolation for time. This avoids using F.grid_sample, which
    does not have a second derivative implemented.
    """
    def __init__(self, data_tensor, domain_min, domain_max, t_min, t_max):
        super().__init__()
        # data_tensor shape: (T, D, H, W, 3)
        self.domain_min = torch.tensor(domain_min, dtype=torch.float64)
        self.domain_max = torch.tensor(domain_max, dtype=torch.float64)
        self.t_min = t_min
        self.t_max = t_max
        self.time_steps, self.depth, self.height, self.width, _ = data_tensor.shape
        
        # Permute for grid_sample which expects (N, C, D_in, H_in, W_in)
        # Our data becomes (T, 3, D, H, W)
        self.register_buffer('data_permuted', data_tensor.permute(0, 4, 1, 2, 3).contiguous())

    def _trilinear_interpolate(self, data_grids, grid_coords):
        """
        Performs trilinear interpolation for a batch of 3D grids.
        This function is twice-differentiable.
        Args:
            data_grids (torch.Tensor): Shape (B, C, D, H, W), batch of vector fields.
            grid_coords (torch.Tensor): Shape (B=N,3), query points in grid coordinates.
        Returns:
            torch.Tensor: Interpolated vectors of shape (B=N, C).
        """
        B, C, D, H, W = data_grids.shape
        grid_coords = grid_coords.view(B,3)

        device = data_grids.device
        dtype = data_grids.dtype
        
        b = torch.arange(B, device=device)

        x, y, z = grid_coords[:, 0], grid_coords[:, 1], grid_coords[:, 2]

        x0 = torch.floor(x).long()
        x1 = x0 + 1
        y0 = torch.floor(y).long()
        y1 = y0 + 1
        z0 = torch.floor(z).long()
        z1 = z0 + 1

        # Clamp coordinates to be within grid bounds
        x0 = torch.clamp(x0, 0, W - 1)
        x1 = torch.clamp(x1, 0, W - 1)
        y0 = torch.clamp(y0, 0, H - 1)
        y1 = torch.clamp(y1, 0, H - 1)
        z0 = torch.clamp(z0, 0, D - 1)
        z1 = torch.clamp(z1, 0, D - 1)

        # Pre-compute weights
        xd = (x - x0.to(dtype)).unsqueeze(1)
        yd = (y - y0.to(dtype)).unsqueeze(1)
        zd = (z - z0.to(dtype)).unsqueeze(1)

        # Fetch values at the 8 corners of the voxel
        c000 = data_grids[b, :, z0, y0, x0]
        c001 = data_grids[b, :, z0, y0, x1]
        c010 = data_grids[b, :, z0, y1, x0]
        c011 = data_grids[b, :, z0, y1, x1]
        c100 = data_grids[b, :, z1, y0, x0]
        c101 = data_grids[b, :, z1, y0, x1]
        c110 = data_grids[b, :, z1, y1, x0]
        c111 = data_grids[b, :, z1, y1, x1]
        
        # Interpolate along x-axis
        c00 = c000 * (1 - xd) + c001 * xd
        c01 = c010 * (1 - xd) + c011 * xd
        c10 = c100 * (1 - xd) + c101 * xd
        c11 = c110 * (1 - xd) + c111 * xd

        # Interpolate along y-axis
        c0 = c00 * (1 - yd) + c01 * yd
        c1 = c10 * (1 - yd) + c11 * yd

        # Interpolate along z-axis
        c = c0 * (1 - zd) + c1 * zd

        return c

    def interpolate(self, points, t):
        """
        Interpolates the vector field at given spatial points and a time t.
        
        Args:
            points (torch.Tensor): Shape (N, 3) of query points (x, y, z).
            t (torch.Tensor or float): The time of query. Can be a scalar or shape (B=1,T,3).
        
        Returns:
            torch.Tensor: Interpolated vectors of shape (N, 3).
        """
        device = points.device
        dtype = points.dtype
        self.domain_min = self.domain_min.to(device)
        self.domain_max = self.domain_max.to(device)
        
        # Map world coordinates to grid coordinates
        domain_size = self.domain_max - self.domain_min
        # Grid shape is (W, H, D) corresponding to (x, y, z)
        grid_shape_vec = torch.tensor([self.width - 1, self.height - 1, self.depth - 1], device=device, dtype=dtype)
        # Handle case where domain size is zero in some dimension
        domain_size[domain_size == 0] = 1
        grid_coords = (points - self.domain_min) / domain_size * grid_shape_vec
        
        N = points.shape[0]

        if self.time_steps > 1:
            # Normalize time to find adjacent time slices
            t_norm = (t - self.t_min) / (self.t_max - self.t_min)
            t_idx_float = t_norm * (self.time_steps - 1) 
            t_idx_floor = torch.floor(t_idx_float).long()
            t_idx_ceil = torch.ceil(t_idx_float).long()
            
            # Clamp indices to be within bounds
            t_idx_floor = torch.clamp(t_idx_floor, 0, self.time_steps - 1)
            t_idx_ceil = torch.clamp(t_idx_ceil, 0, self.time_steps - 1)
            
            # Get the vector fields at the two time slices
            field_floor = self.data_permuted[t_idx_floor]
            field_ceil = self.data_permuted[t_idx_ceil]

            # Handle case where t is a scalar
            if field_floor.dim() == 3:
                field_floor = field_floor.unsqueeze(0).expand(N, -1, -1, -1, -1)
            if field_ceil.dim() == 3:
                field_ceil = field_ceil.unsqueeze(0).expand(N, -1, -1, -1, -1)
            
            # Spatially interpolate at each time slice
            vec_floor = self._trilinear_interpolate(field_floor, grid_coords)
            vec_ceil = self._trilinear_interpolate(field_ceil, grid_coords)
            
            # Linearly interpolate in time
            weight = (t_idx_float - t_idx_floor.to(t_idx_float.dtype)).unsqueeze(-1)
            interp_vec = vec_floor * (1 - weight) + vec_ceil * weight
            return interp_vec

        else: # Steady field
            vec_data = self.data_permuted[0] # (C, D, H, W)
            #shape of grid_coords is (B,N, 3)
            vec = self._trilinear_interpolate(vec_data.unsqueeze(0), grid_coords)
            return vec
    
class FlowMapODE(nn.Module):
    """Defines the ODE system for computing the flow map and its pushforward."""
    def __init__(self, vector_field: UnsteadyVectorField3D_Torch):
        super().__init__()
        self.v_field = vector_field

    def forward(self, t, Y):
        """
        Computes dY/dt = [v, (nabla_v) @ v_dot].
        
        Args:
            t (float): Current time.
            Y (torch.Tensor): State vector of shape (N, 6), where Y = [q, q_dot].
        
        Returns:
            torch.Tensor: Derivative dY/dt of shape (N, 6).
        """
        q, q_dot = Y.split(3, dim=-1) # q and q_dot both have shape (N, 3)
        
        with torch.enable_grad():
            q.requires_grad_(True)
            # 1. Compute dq/dt = v(q, t)
            dq_dt = self.v_field.interpolate(q, t)
            
            # 2. Compute dq_dot/dt = (nabla_v)(q,t) @ q_dot
            v_at_q = self.v_field.interpolate(q, t)
            
            J_v = torch.autograd.functional.jacobian(
                lambda x: self.v_field.interpolate(x, t),
                q,
                create_graph=True
            ).squeeze() # This might need adjustment for batching

            # For N > 1, jacobian gives (N, 3, N, 3). We need (N, 3, 3)
            if J_v.dim() == 4:
                J_v = torch.diagonal(J_v, dim1=0, dim2=2).permute(2,0,1)

            dq_dot_dt = torch.einsum('nij,nj->ni', J_v, q_dot)
        
        return torch.cat([dq_dt.detach(), dq_dot_dt.detach()], dim=-1)
    
class VariationalCorelineExtractor:
    def __init__(self, vector_field: UnsteadyVectorField3D_Torch, config: dict):
        self.v_field = vector_field
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float64
        
        # Ensure model and data are on the same device and dtype
        self.v_field.to(device=self.device, dtype=self.dtype)

        self.weight_A = self.config.get('weight_A', 1.0)
        self.weight_D = self.config.get('weight_D', 1.0)
        self.weight_R = self.config.get('weight_R', 1.0)
        self.t0 = self.config.get('t0', 0.0)#start time of flowmap
        self.t_T = self.config.get('t_T', 1.0)#end time of flowmap
        #number of time steps to solve flowmap ODE, doesn't have to match the number of time steps of the vector field
        self.time_steps = self.config.get('time_steps', 1)
        
        self.ode_func = FlowMapODE(self.v_field)
        self.w_field = None # Assuming w=0 for simplicity

    def _get_observed_velocity(self, q, t):
        v = self.v_field.interpolate(q, t)
        return v # Assuming w=0

    def _compute_lagrangian_integrand(self, q, q_dot, t):
        """Computes the value inside the time integral of the Lagrangian."""
        v_obs = self._get_observed_velocity(q, t).view(-1, 1, 3)
        
        # Using the more stable cross-product formulation (Eq. 36)
        term1 = torch.linalg.norm(torch.cross(v_obs, q_dot, dim=-1), dim=-1)**2+ 0.001*torch.linalg.norm(q_dot-v_obs, dim=-1)**2
        
        # Placeholder for other terms (A, D, R) for clarity.
        # A full implementation would compute them here using autograd on v_field.
        # term2 = torch.zeros_like(term1)
        # term3 = torch.zeros_like(term1)
        # term4 = torch.zeros_like(term1)
        # return term1 + self.weight_A * term2 + self.weight_D * term3 + self.weight_R * term4
        return term1
    

    def _compute_total_lagrangian(self, q, q_dot):
        """
        Computes the full time-integrated Lagrangian, including the flow map.
        q and q_dot are the initial curve state at t0. Shape (N, 3).
        """
        N = q.shape[0]

        if self.time_steps > 1 and self.t0 < self.t_T:
            # Unsteady flowmap
            t_span = torch.linspace(self.t0, self.t_T, self.time_steps, device=self.device, dtype=self.dtype)
            
            # 1. Compute the flow map and pushforward by solving the ODE
            Y0 = torch.cat([q, q_dot], dim=-1)
            Y_t = odeint_torch(self.ode_func, Y0, t_span, method='dopri5')
            Y_t = Y_t.permute(1, 0, 2)  # (N, time_steps, 6)
            q_t, q_dot_t = Y_t.split(3, dim=-1)
            # 2. Compute the integrand at each advected state
            actual_time_steps = q_t.shape[1]
            # Flatten tensors for batched processing
            flat_q_t = q_t.reshape(-1, 3)
            flat_q_dot_t = q_dot_t.reshape(-1, 3)
            flat_t = t_span.unsqueeze(0).expand(N, -1).reshape(-1)

            # Compute integrand for all points and time steps at once
            integrand_values_flat = self._compute_lagrangian_integrand(
                flat_q_t, flat_q_dot_t, flat_t
            )
            integrand_values = integrand_values_flat.reshape(N, actual_time_steps)
            L_total = torch.trapezoid(integrand_values, t_span, dim=1)
            return L_total
        else:
            # Steady flowmap: treat as a single time step
            t_span = torch.tensor([self.t0], device=self.device, dtype=self.dtype)
            q_t = q.unsqueeze(1)      # (N, 1, 3)
            q_dot_t = q_dot.unsqueeze(1) # (N, 1, 3)
            L_total = self._compute_lagrangian_integrand(q_t, q_dot_t, t_span)
            return L_total

    def _solve_el_equation_for_q_ddot(self, q, q_dot):
        """
        Computes all terms for the Euler-Lagrange equation and solves for q_ddot.
        This version implements the robust 2x2 system based on spherical projection.
        """
        # Ensure input tensors are single items (batch size 1)
        if q.shape[0] > 1:
            # This function is not designed for batching, process one by one if needed.
            # Here we'll just process the first item as per the logic.
            q = q[0:1]
            q_dot = q_dot[0:1]

        q.requires_grad_(True)
        q_dot_squeezed = q_dot.squeeze(0) # Shape (3,)

        # 1. Get Orthonormal Frame B = [b1, b2, b3]
        b1 = q_dot_squeezed
        ref_vec = torch.tensor([1.0, 0.0, 0.0], device=self.device, dtype=self.dtype)
        if torch.abs(torch.dot(b1, ref_vec)) > 1.0 - 1e-7:
            ref_vec = torch.tensor([0.0, 1.0, 0.0], device=self.device, dtype=self.dtype)
        
        b3 = torch.cross(b1, ref_vec,dim=-1)
        b3 = F.normalize(b3, p=2, dim=-1)
        b2 = torch.cross(b3, b1,dim=-1)
        b2 = F.normalize(b2, p=2, dim=-1)
        B = torch.stack([b1, b2, b3], dim=1) # Basis vectors are columns

        # 2. Define Lagrangian in terms of spherical coordinates (theta, phi)
        #    This function will be used to compute the 2x2 Hessian.
        def lagrangian_of_q_theta_phi(q_dot_as_theta_phi):
            theta = q_dot_as_theta_phi[0]
            phi = q_dot_as_theta_phi[1]
            
            spherical_to_euclidean = torch.stack([
                torch.cos(theta) * torch.sin(phi),
                torch.sin(theta) * torch.sin(phi),
                torch.cos(phi)
            ])
            
            new_q_dot = B @ spherical_to_euclidean
            return self._compute_total_lagrangian(q, new_q_dot.unsqueeze(0)).sum()

        # 3. Compute Hessian H_L_thetaphi at q_dot
        #    q_dot corresponds to theta=0, phi=pi/2 in our new frame.
        hessian_eval_point = torch.tensor([0.0, np.pi/2], device=self.device, dtype=self.dtype)
        H_2x2 = torch.autograd.functional.hessian(lagrangian_of_q_theta_phi, hessian_eval_point,create_graph=True)

        # The C++ code negates off-diagonal elements. Let's replicate this.
        # Htelta(1,0)*=-1; Htelta(0,1)*=-1;
        H_2x2[0, 1] *= -1
        H_2x2[1, 0] *= -1

        
        # 4. Compute the right-hand side `r_hat`
        # We need dL/dq, dL/dq_dot, and H_L_q_qdot
        q.requires_grad_(True)
        q_dot.requires_grad_(True)

        # Define a wrapper for the Lagrangian that returns a scalar, suitable for hessian
        def lagrangian_sum(q_arg, q_dot_arg):
            return self._compute_total_lagrangian(q_arg, q_dot_arg).sum()

        L_total = lagrangian_sum(q, q_dot)
        dL_dq, = torch.autograd.grad(L_total, q, create_graph=True)

        # Compute the full Hessian tuple: ((H_qq, H_qq_dot), (H_q_dot_q, H_q_dot_q_dot))
        hessian_tuple = torch.autograd.functional.hessian(lagrangian_sum, (q, q_dot), create_graph=True)
        
        # We need the mixed derivative part H_L_q_qdot
        Hessian_L_q_qdot = hessian_tuple[0][1] # This is d^2L / dq dq_dot
        
        # The result has shape (1, 3, 1, 3), squeeze it to (3, 3)
        Hessian_L_q_qdot = Hessian_L_q_qdot.squeeze()
        
        r = dL_dq.squeeze(0) - Hessian_L_q_qdot.transpose(0,1) @ q_dot.squeeze(0)

        # Project r onto the b2, b3 plane
        B_perp_T = torch.stack([b2, b3], dim=0) # Transposed B_perp
        r_hat = B_perp_T @ r

        # 5. Solve the 2x2 system
        try:
            qdotdot_spherical = torch.linalg.solve(H_2x2, r_hat)
            #shape of qdotdot_spherical is (2)
        except torch.linalg.LinAlgError:
            print("Warning: 2x2 Hessian is singular, using pseudo-inverse.")
            qdotdot_spherical = torch.linalg.pinv(H_2x2) @ r_hat

        # 6. Convert local solution back to 3D space
        B_perp = torch.stack([b2, b3], dim=1) # b2 and b3 are columns
        q_ddot = B_perp @ qdotdot_spherical
        #shape of q_ddot is (3)

        return q_ddot.detach() # Return with batch dimension

    def _solve_el_equation_for_q_ddot_finite_diff(self, q, q_dot):
        """
        Computes all terms for the Euler-Lagrange equation and solves for q_ddot
        using finite differences to approximate derivatives, mimicking the C++ implementation.
        """
        # Ensure input tensors are single items (batch size 1)
        if q.shape[0] > 1:
            q = q[0:1]
            q_dot = q_dot[0:1]

        h = self.config.get('finite_diff_h', 5e-3)
        
        # Squeeze to (3,) for finite difference helpers
        q_vec = q.squeeze(0)
        q_dot_vec = q_dot.squeeze(0)

        # Helper to compute scalar Lagrangian for a given q and q_dot vector
        def compute_L_scalar(q_v, q_dot_v):
            q_t = q_v.unsqueeze(0)
            q_dot_t = q_dot_v.unsqueeze(0)
            # Ensure tensors passed to model are on the correct device and dtype
            q_t = q_t.to(device=self.device, dtype=self.dtype)
            q_dot_t = q_dot_t.to(device=self.device, dtype=self.dtype)
            return self._compute_total_lagrangian(q_t, q_dot_t).item()

        # 1. Get Orthonormal Frame B
        b1 = q_dot_vec
        ref_vec = torch.tensor([1.0, 0.0, 0.0], device=self.device, dtype=self.dtype)
        if torch.abs(torch.dot(b1, ref_vec)) > 1.0 - 1e-7:
            ref_vec = torch.tensor([0.0, 1.0, 0.0], device=self.device, dtype=self.dtype)
        
        b3 = torch.cross(b1, ref_vec, dim=-1)
        b3 = F.normalize(b3, p=2, dim=-1)
        b2 = torch.cross(b3, b1, dim=-1)
        b2 = F.normalize(b2, p=2, dim=-1)
        B = torch.stack([b1, b2, b3], dim=1)

        # 2. Compute derivatives using finite differences
        partial_L_q = self._compute_partial_L_partial_q_fd(q_vec, q_dot_vec, h, compute_L_scalar)
        # partial_L_qdot = self._compute_partial_L_partial_qdot_fd(q_vec, q_dot_vec, h, compute_L_scalar)
        hessian_L_qdot_q = self._compute_hessian_L_qdot_q_fd(q_vec, q_dot_vec, h, compute_L_scalar)

        # 3. Compute Hessian in spherical coordinates
        def lagrangian_of_q_theta_phi_scalar(q_dot_as_theta_phi_vec):
            theta, phi = q_dot_as_theta_phi_vec[0], q_dot_as_theta_phi_vec[1]
            spherical_to_euclidean = torch.stack([
                torch.cos(theta) * torch.sin(phi),
                torch.sin(theta) * torch.sin(phi),
                torch.cos(phi)
            ])
            new_q_dot = B @ spherical_to_euclidean
            return compute_L_scalar(q_vec, new_q_dot)
            
        hessian_eval_point = torch.tensor([0.0, np.pi/2], device=self.device, dtype=self.dtype)
        hessian_L_theta_fi = self._compute_hessian_L_qdot_qdot_spherical_fd(
            hessian_eval_point, h, lagrangian_of_q_theta_phi_scalar
        )

        Htelta = hessian_L_theta_fi.clone()
        Htelta[0, 1] *= -1
        Htelta[1, 0] *= -1

        # 4. Compute right-hand side `r_hat` and solve
        B_perp_T = torch.stack([b2, b3], dim=0)
        r_hat = B_perp_T @ (partial_L_q - hessian_L_qdot_q @ q_dot_vec)

        try:
            qdotdot_spherical = torch.linalg.solve(Htelta, r_hat)
        except torch.linalg.LinAlgError:
            print("Warning: Finite difference Hessian is singular, using pseudo-inverse.")
            qdotdot_spherical = torch.linalg.pinv(Htelta) @ r_hat
            
        # 5. Convert solution back to 3D space
        B_perp = torch.stack([b2, b3], dim=1)
        q_ddot = B_perp @ qdotdot_spherical
        
        return q_ddot.detach()

    def _compute_partial_L_partial_q_fd(self, q, q_dot, h, computeL):
        """4th-order finite difference for dL/dq."""
        n = 3
        one_over_12h = 1.0 / (12.0 * h)
        partialL_partialQ = torch.zeros(n, device=self.device, dtype=self.dtype)

        for i in range(n):
            q_p1 = q.clone(); q_p1[i] += h
            q_m1 = q.clone(); q_m1[i] -= h
            q_p2 = q.clone(); q_p2[i] += 2.0 * h
            q_m2 = q.clone(); q_m2[i] -= 2.0 * h

            L_p1 = computeL(q_p1, q_dot)
            L_m1 = computeL(q_m1, q_dot)
            L_p2 = computeL(q_p2, q_dot)
            L_m2 = computeL(q_m2, q_dot)
            
            partialL_partialQ[i] = (L_m2 - 8.0 * L_m1 + 8.0 * L_p1 - L_p2) * one_over_12h
        
        return partialL_partialQ

    def _compute_partial_L_partial_qdot_fd(self, q, q_dot, h, computeL):
        """4th-order finite difference for dL/dq_dot."""
        n = 3
        one_over_12h = 1.0 / (12.0 * h)
        partialL_partialQdot = torch.zeros(n, device=self.device, dtype=self.dtype)

        for i in range(n):
            qdot_p1 = q_dot.clone(); qdot_p1[i] += h
            qdot_m1 = q_dot.clone(); qdot_m1[i] -= h
            qdot_p2 = q_dot.clone(); qdot_p2[i] += 2.0 * h
            qdot_m2 = q_dot.clone(); qdot_m2[i] -= 2.0 * h

            L_p1 = computeL(q, qdot_p1)
            L_m1 = computeL(q, qdot_m1)
            L_p2 = computeL(q, qdot_p2)
            L_m2 = computeL(q, qdot_m2)
            
            partialL_partialQdot[i] = (L_m2 - 8.0 * L_m1 + 8.0 * L_p1 - L_p2) * one_over_12h
            
        return partialL_partialQdot

    def _compute_hessian_L_qdot_q_fd(self, q, q_dot, h, computeL):
        """4th-order finite difference for Hessian d^2L/dq_dot dq."""
        n = 3
        one_over_144hsq = 1.0 / (144.0 * h * h)
        hessian = torch.zeros((n, n), device=self.device, dtype=self.dtype)
        
        coeffs_k = torch.tensor([-1.0, 8.0, -8.0, 1.0], device=self.device, dtype=self.dtype)
        steps_k = torch.tensor([2.0 * h, h, -h, -2.0 * h], device=self.device, dtype=self.dtype)

        for i in range(n):
            for j in range(n):
                term_sum = 0.0
                for idx_i in range(4):
                    for idx_j in range(4):
                        q_eval = q.clone()
                        q_dot_eval = q_dot.clone()
                        q_dot_eval[i] += steps_k[idx_i]
                        q_eval[j] += steps_k[idx_j]
                        term_sum += coeffs_k[idx_i] * coeffs_k[idx_j] * computeL(q_eval, q_dot_eval)
                hessian[i, j] = term_sum * one_over_144hsq
        
        return hessian

    def _compute_hessian_L_qdot_qdot_spherical_fd(self, qdot_as_theta_fi, h, compute_L_spherical):
        """8th-order finite difference for Hessian in spherical coordinates."""
        n = 2
        hessian = torch.zeros((n, n), device=self.device, dtype=self.dtype)
        h_sq = h * h
        
        # Diagonal elements
        one_over_5040hsq = 1.0 / (5040.0 * h_sq)
        L_center = compute_L_spherical(qdot_as_theta_fi)

        for i in range(n):
            p = [qdot_as_theta_fi.clone() for _ in range(4)]
            m = [qdot_as_theta_fi.clone() for _ in range(4)]
            for k in range(4):
                p[k][i] += (k + 1.0) * h
                m[k][i] -= (k + 1.0) * h

            L_p = [compute_L_spherical(val) for val in p]
            L_m = [compute_L_spherical(val) for val in m]

            diag_val = (-9.0   * (L_p[3] + L_m[3])
                        + 128.0  * (L_p[2] + L_m[2])
                        - 1008.0 * (L_p[1] + L_m[1])
                        + 8064.0 * (L_p[0] + L_m[0])
                        - 14350.0 * L_center) * one_over_5040hsq
            hessian[i, i] = diag_val

        # Off-diagonal elements
        one_over_705600hsq = 1.0 / (705600.0 * h_sq)
        coeffs_k = torch.tensor([-5.0, 40.0, -180.0, 720.0, -720.0, 180.0, -40.0, 5.0], device=self.device, dtype=self.dtype)
        steps_k = torch.tensor([4.0*h, 3.0*h, 2.0*h, h, -h, -2.0*h, -3.0*h, -4.0*h], device=self.device, dtype=self.dtype)

        i, j = 0, 1
        term_sum = 0.0
        for idx_i in range(8):
            for idx_j in range(8):
                eval_point = qdot_as_theta_fi.clone()
                eval_point[i] += steps_k[idx_i]
                eval_point[j] += steps_k[idx_j]
                term_sum += coeffs_k[idx_i] * coeffs_k[idx_j] * compute_L_spherical(eval_point)
        
        off_diag_val = term_sum * one_over_705600hsq
        hessian[i, j] = off_diag_val
        hessian[j, i] = off_diag_val
        
        return hessian
        
    def _get_curve_derivatives(self, q, q_dot):
        """Computes the derivatives for curve integration: (dq/ds, dq_dot/ds)."""
        # dq/ds is the normalized tangent vector
        dq_ds = F.normalize(q_dot, p=2, dim=-1)
        # dq_dot/ds is the acceleration vector q_ddot
        # self._get_el_equation_terms expects batched input, but we process one by one
        q_ddot = self._solve_el_equation_for_q_ddot(q, dq_ds)

        # Correction to make q_ddot orthogonal to dq_ds (since dq_ds is on S2)
        # This is the Gram-Schmidt orthogonalization process.
        # q_ddot_corrected = q_ddot - proj_dq_ds(q_ddot)
        # proj_dq_ds(q_ddot) = (q_ddot . dq_ds / |dq_ds|^2) * dq_ds
        # Since dq_ds is normalized, |dq_ds|^2 = 1.
        dot_product = torch.sum(q_ddot * dq_ds, dim=-1, keepdim=True)
        q_ddot_final = q_ddot - dot_product * dq_ds

        return dq_ds, q_ddot_final

    def integrate_curve(self, q0, q_dot0, max_RK4_iter, ds):
        """Integrates the curve using the 4th-order Runge-Kutta (RK4) method."""
        q = q0.clone().to(self.device, self.dtype)
        q_dot = q_dot0.clone().to(self.device, self.dtype)
        
        curve_q = [q.cpu().numpy()]
        
        for s_idx in range(max_RK4_iter):
            # k1
            k1_dq_ds, k1_dq_dot_ds = self._get_curve_derivatives(q, q_dot)
            
            # k2
            k2_dq_ds, k2_dq_dot_ds = self._get_curve_derivatives(
                q + 0.5 * ds * k1_dq_ds, 
                q_dot + 0.5 * ds * k1_dq_dot_ds
            )

            # k3
            k3_dq_ds, k3_dq_dot_ds = self._get_curve_derivatives(
                q + 0.5 * ds * k2_dq_ds,
                q_dot + 0.5 * ds * k2_dq_dot_ds
            )

            # k4
            k4_dq_ds, k4_dq_dot_ds = self._get_curve_derivatives(
                q + ds * k3_dq_ds,
                q_dot + ds * k3_dq_dot_ds
            )

            # Update q and q_dot
            q = q + (ds / 6.0) * (k1_dq_ds + 2 * k2_dq_ds + 2 * k3_dq_ds + k4_dq_ds)
            q_dot = q_dot + (ds / 6.0) * (k1_dq_dot_ds + 2 * k2_dq_dot_ds + 2 * k3_dq_dot_ds + k4_dq_dot_ds)
            
            # It's good practice to re-normalize the tangent vector after each step for stability
            q_dot = F.normalize(q_dot, p=2, dim=-1)

            curve_q.append(q.clone().detach().cpu().numpy())
            
            if torch.isnan(q).any() or torch.isinf(q).any():
                print(f"Integration failed at step {s_idx}: NaN or Inf detected.")
                break

        return np.array(curve_q).squeeze()




def execute_varational_coreline_extractor(vector_field:UnsteadyVectorField3D, q0:torch.Tensor, q_dot0:torch.Tensor,config:dict):
  
    # generate UnsteadyVectorField3D_Torch
    vector_field_torch = UnsteadyVectorField3D_Torch(torch.from_numpy(vector_field.field), vector_field.domainMinBoundary, vector_field.domainMaxBoundary, vector_field.tmin, vector_field.tmax)
    extractor = VariationalCorelineExtractor(vector_field_torch, config)
    q0 = torch.tensor(q0, device=extractor.device, dtype=extractor.dtype)
    q_dot0 = torch.tensor(q_dot0, device=extractor.device, dtype=extractor.dtype)
    max_RK4_iter = config.get('max_RK4_iter',10)
    ds = config.get('ds',0.05)
    curve_q = extractor.integrate_curve(q0, q_dot0, max_RK4_iter, ds)  
    return curve_q