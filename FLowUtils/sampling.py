from .VectorField2d import *
from .VectorField3d import *
import torch
from torch.quasirandom import SobolEngine
from scipy.stats import qmc  # for Halton sequence
def samplePointsVelocity(Vectorfield2d:UnsteadyVectorField2D, sampleCount:int, sequence_type='sobol'):
    """
    Sample points using low-discrepancy sequences (Sobol or Halton)
    Args:
        Vectorfield2d: UnsteadyVectorField2D object
        sampleCount: Number of points to sample
        sequence_type: 'sobol' or 'halton'
    Returns:
        points: tensor shape [sampleCount,4] (#sampleCount points' 3d+time coordinate)
        velocities: tensor shape [sampleCount,2] 
    """
    T, H, W, _ = Vectorfield2d.field.shape
    
    # Generate LDS points in [0,1]^3 space
    if sequence_type == 'sobol':
        sampler = SobolEngine(dimension=3)
        samples = sampler.draw(sampleCount)  # shape: [sampleCount, 3]
    else:  # halton
        sampler = qmc.Halton(d=3)
        samples = torch.from_numpy(sampler.random(sampleCount).astype(np.float32))
    
    # Scale samples to indices
    t_idx = (samples[:, 0] * (T-1)).long()
    y_idx = (samples[:, 1] * (H-1)).long()
    x_idx = (samples[:, 2] * (W-1)).long()
    
    # Convert indices to normalized coordinates in domain
    x_coords = torch.linspace(Vectorfield2d.domainMinBoundary[0], 
                            Vectorfield2d.domainMaxBoundary[0], W)[x_idx]
    y_coords = torch.linspace(Vectorfield2d.domainMinBoundary[1], 
                            Vectorfield2d.domainMaxBoundary[1], H)[y_idx]
    t_coords = torch.linspace(Vectorfield2d.tmin, 
                            Vectorfield2d.tmax, T)[t_idx]
    
    # Get velocities at sampled points
    velocities = torch.from_numpy(Vectorfield2d.field[t_idx, y_idx, x_idx].astype(np.float32))
    
    # Stack coordinates and add z=0 coordinate
    z_coords = torch.zeros_like(x_coords)
    points = torch.stack([x_coords, y_coords, z_coords, t_coords], dim=1)
    
    return points, velocities

def samplingPointVelcityBySpatialTemporalDensity(Vectorfield2d: UnsteadyVectorField2D, spatial_density:float, temporal_density:float, sequence_type='sobol'):
    """
    Sample points using specified spatial and temporal densities
    
    Args:
        Vectorfield2d: UnsteadyVectorField2D object
        spatial_density: Number of points to sample per spatial frame is (spatial_density*H*W)
        temporal_density: Number of time steps to sample is temporal_density*T
        sequence_type: 'sobol' or 'halton'
    Returns:
        points: tensor shape [total_samples, 4] (spatial_density * temporal_density points' 3d+time coordinates)
        velocities: tensor shape [total_samples, 2] 
    """
    T, H, W, _ = Vectorfield2d.field.shape
    
    # Ensure temporal_density doesn't exceed available time steps
    temporal_Squence_count =int(temporal_density*T)
    
    # Generate temporal samples first
    if sequence_type == 'sobol':
        time_sampler = SobolEngine(dimension=1)
        t_samples = time_sampler.draw(temporal_Squence_count)
    else:  # halton
        time_sampler = qmc.Halton(d=1)
        t_samples = torch.from_numpy(time_sampler.random(temporal_Squence_count).astype(np.float32))
    
    # Convert to time indices
    t_idx = (t_samples.squeeze() * (T-1)).long()
    
    # For each time step, sample spatial points
    all_points = []
    all_velocities = []
    
    spatial_samples_per_slice=int(spatial_density*H*W)
    for t in t_idx:
        # Generate spatial samples
        if sequence_type == 'sobol':
            spatial_sampler = SobolEngine(dimension=2)
            spatial_samples = spatial_sampler.draw(spatial_samples_per_slice)
        else:  # halton
            spatial_sampler = qmc.Halton(d=2)
            spatial_samples = torch.from_numpy(spatial_sampler.random(spatial_samples_per_slice).astype(np.float32))
        
        # Scale spatial samples to indices
        y_idx = (spatial_samples[:, 0] * (H-1)).long()
        x_idx = (spatial_samples[:, 1] * (W-1)).long()
        
        # Convert indices to normalized coordinates
        x_coords = torch.linspace(Vectorfield2d.domainMinBoundary[0], 
                                Vectorfield2d.domainMaxBoundary[0], W)[x_idx]
        y_coords = torch.linspace(Vectorfield2d.domainMinBoundary[1], 
                                Vectorfield2d.domainMaxBoundary[1], H)[y_idx]
        t_coords = torch.full_like(x_coords, Vectorfield2d.tmin + (Vectorfield2d.tmax - Vectorfield2d.tmin) * t.float() / (T-1))
        
        # Get velocities for this time step
        velocities = torch.from_numpy(Vectorfield2d.field[t, y_idx, x_idx].astype(np.float32))
        
        # Stack coordinates with z=0
        z_coords = torch.zeros_like(x_coords)
        points = torch.stack([x_coords, y_coords, z_coords, t_coords], dim=1)
        
        all_points.append(points)
        all_velocities.append(velocities)
    
    # Combine all samples
    points = torch.cat(all_points, dim=0)
    velocities = torch.cat(all_velocities, dim=0)
    
    return points, velocities   

def velocityDerivativeSaliencse(Vectorfield2d: UnsteadyVectorField2D):
    """Calculate saliency field as sum of velocity gradient tensor norm and dv/dt norm.
    
    Args:
        Vectorfield2d: Input vector field
        
    Returns:
        Numpy array of shape (T,H,W) containing saliency values
    """
    # Get velocity gradient tensor norm
    grad_v = np.zeros((Vectorfield2d.time_steps, Vectorfield2d.Ydim, Vectorfield2d.Xdim))
    for t in range(Vectorfield2d.time_steps):
        # Calculate spatial derivatives using numpy gradient
        dv1_dx = np.gradient(Vectorfield2d.field[t,:,:,0], axis=1)
        dv1_dy = np.gradient(Vectorfield2d.field[t,:,:,0], axis=0) 
        dv2_dx = np.gradient(Vectorfield2d.field[t,:,:,1], axis=1)
        dv2_dy = np.gradient(Vectorfield2d.field[t,:,:,1], axis=0)
        
        # Frobenius norm of gradient tensor at each point
        grad_v[t] = np.sqrt(dv1_dx**2 + dv1_dy**2 + dv2_dx**2 + dv2_dy**2)
    
    # Get dv/dt norm using central differences
    dvdt = np.zeros((Vectorfield2d.time_steps, Vectorfield2d.Ydim, Vectorfield2d.Xdim,2))
    dvdt[1:-1] = (Vectorfield2d.field[2:] - Vectorfield2d.field[:-2]) / 2
    dvdt[0] = Vectorfield2d.field[1] - Vectorfield2d.field[0]  
    dvdt[-1] = Vectorfield2d.field[-1] - Vectorfield2d.field[-2]
    dvdt_norm = np.sqrt(np.sum(dvdt**2, axis=-1))
    
    # Return sum of both norms
    return grad_v + dvdt_norm
    
    

#lambdaFn operate on Vectorfield2d will return  a timedepdent scalar field, for example: velocity maginitude, use it as weighting for sampling points
def salienceSampling(Vectorfield2d: UnsteadyVectorField2D, sampleCount:int, lambdaFn):
    """Sample points from a vector field weighted by a saliency function.
    
    Args:
        Vectorfield2d: Input vector field to sample from
        sampleCount: Number of points to sample
        lambdaFn: Function that takes a vector field and returns a scalar field used as sampling weights
        
    Returns:
        Tuple of (points, velocities) where:
            points: Tensor of shape (N, 4) containing sampled (x,y,z,t) coordinates 
            velocities: Tensor of shape (N, 2) containing velocities at sampled points
    """
    # Get dimensions
    H, W, T = Vectorfield2d.Ydim, Vectorfield2d.Xdim, Vectorfield2d.time_steps
    # Calculate saliency weights
    weights = lambdaFn(Vectorfield2d)
    weights = torch.from_numpy(weights)
    
    # Flatten weights and normalize
    flat_weights = weights.reshape(-1)
    probs = flat_weights / flat_weights.sum()
    
    # torch.multinomial(probs, sampleCount, replacement=True)` performs weighted random sampling from a probability distribution. 
    indices = torch.multinomial(probs, sampleCount, replacement=True)
    
    # Convert flat indices back to t,y,x coordinates
    t_idx = indices // (H * W)
    remainder = indices % (H * W)
    y_idx = remainder // W
    x_idx = remainder % W
    
    # Convert indices to normalized coordinates
    x_coords = torch.linspace(Vectorfield2d.domainMinBoundary[0], 
                            Vectorfield2d.domainMaxBoundary[0], W)[x_idx]
    y_coords = torch.linspace(Vectorfield2d.domainMinBoundary[1], 
                            Vectorfield2d.domainMaxBoundary[1], H)[y_idx]
    t_coords = Vectorfield2d.tmin + (Vectorfield2d.tmax - Vectorfield2d.tmin) * t_idx.float() / (T-1)
    
    # Get velocities
    velocities = torch.from_numpy(Vectorfield2d.field[t_idx, y_idx, x_idx].astype(np.float32))
    
    # Stack coordinates with z=0
    z_coords = torch.zeros_like(x_coords)
    points = torch.stack([x_coords, y_coords, z_coords, t_coords], dim=1)
    
    return points, velocities
    
def hybridSampling(Vectorfield2d: UnsteadyVectorField2D, sampleCount:int, lambdaFn):
    p0,v0=salienceSampling( Vectorfield2d ,sampleCount//2,velocityDerivativeSaliencse )
    p,v=samplePointsVelocity(Vectorfield2d  ,sampleCount=sampleCount//2)
    # Merge points and velocities from both sampling methods
    merged_points = torch.cat([p0, p], dim=0)
    merged_velocities = torch.cat([v0, v], dim=0)
    
    return merged_points, merged_velocities





def samplePointsVelocity3d(Vectorfield3d: UnsteadyVectorField3D, sampleCount: int, sequence_type='sobol'):
    """
    Sample points using low-discrepancy sequences (Sobol or Halton)
    Args:
        Vectorfield3d: UnsteadyVectorField3D object
        sampleCount: Number of points to sample
        sequence_type: 'sobol' or 'halton'
    Returns:
        points: tensor shape [sampleCount, 4] (#sampleCount points' 3d+time coordinate)
        velocities: tensor shape [sampleCount, 3]  # Updated to 3 for 3D velocities
    """
    T, D,H, W, _ = Vectorfield3d.field.shape  # D is the depth (z dimension)
    
    # Generate LDS points in [0,1]^4 space (3D + time)
    if sequence_type == 'sobol':
        sampler = SobolEngine(dimension=4)  # Updated dimension to 4
        samples = sampler.draw(sampleCount)  # shape: [sampleCount, 4]
    else:  # halton
        sampler = qmc.Halton(d=4)  # Updated dimension to 4
        samples = torch.from_numpy(sampler.random(sampleCount).astype(np.float32))
    
    # Scale samples to indices
    t_idx = (samples[:, 0] * (T-1)).long()
    z_idx = (samples[:, 1] * (D-1)).long()  # Added z index
    y_idx = (samples[:, 2] * (H-1)).long()
    x_idx = (samples[:, 3] * (W-1)).long()
    
    # Convert indices to normalized coordinates in domain
    x_coords = torch.linspace(Vectorfield3d.domainMinBoundary[0], 
                            Vectorfield3d.domainMaxBoundary[0], W)[x_idx]
    y_coords = torch.linspace(Vectorfield3d.domainMinBoundary[1], 
                            Vectorfield3d.domainMaxBoundary[1], H)[y_idx]
    z_coords = torch.linspace(Vectorfield3d.domainMinBoundary[2], 
                            Vectorfield3d.domainMaxBoundary[2], D)[z_idx]  # Added z coordinates
    t_coords = torch.linspace(Vectorfield3d.tmin, 
                            Vectorfield3d.tmax, T)[t_idx]
    
    # Get velocities at sampled points
    velocities = torch.from_numpy(Vectorfield3d.field[t_idx, z_idx, y_idx, x_idx].astype(np.float32))  # Updated indexing for 3D
    
    # Stack coordinates
    points = torch.stack([x_coords, y_coords, z_coords, t_coords], dim=1)  # Updated to include z_coords
    
    return points, velocities




