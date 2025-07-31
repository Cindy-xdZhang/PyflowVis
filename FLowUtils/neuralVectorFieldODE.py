from torchdiffeq import odeint
import torch
import torch.nn.functional as F

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
    




def integrate_pathline2D_torch(
    vector_field,
    start_pos: list,
    time_start: float,
    time_end: float,
    step_size: float = 0.01,    
    method: str = "dopri5",
    rtol: float = 1e-5,
    atol: float = 1e-6,
    device: str = "cpu"
):
    device = torch.device(device)
    y0 = torch.tensor(start_pos, dtype=torch.float32, device=device)
    
    direction = 1 if time_end >= time_start else -1
    t_eval = torch.arange(time_start, time_end + direction*step_size, direction*step_size, device=device)
    if direction == -1 and t_eval[-1] > time_end:
         t_eval = torch.cat([t_eval, torch.tensor([time_end], device=device)])
    elif direction == 1 and t_eval[-1] < time_end:
         t_eval = torch.cat([t_eval, torch.tensor([time_end], device=device)])


    ode_func = VectorFieldODE(vector_field).to(device)
    
    solution = odeint(ode_func, y0, t_eval, method=method, rtol=rtol, atol=atol)
    
    solution = solution.permute(1, 0, 2)
    paths = []
    for i in range(solution.shape[0]):
        path = []
        for j in range(solution.shape[1]):
            pos = solution[i, j].cpu().numpy()
            time = t_eval[j].item()
            path.append((pos, time))
        paths.append(path)
    return paths

