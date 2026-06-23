import torch
import torch.nn as nn
from FLowUtils.AnalyticalFlowCreator import *
# from FLowUtils.LicRenderer import *
from FLowUtils.netCDFLoader import *
from FLowUtils.sampling import *
from torch.cuda.amp import autocast, GradScaler
from FLowUtils.heatStreamPlot import heatStreamPlot




class Sine(nn.Module):
    def __init__(self, w0: float = 1.0):
        """Sine activation function with w0 scaling support.

        Example:
            >>> w = torch.tensor([3.14, 1.57])
            >>> Sine(w0=1)(w)
            torch.Tensor([0, 1])

        :param w0: w0 in the activation step `act(x; w0) = sin(w0 * x)`.
            defaults to 1.0
        :type w0: float, optional
        """
        super(Sine, self).__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)
        return torch.sin(self.w0 * x)

    @staticmethod
    def _check_input(x):
        if not isinstance(x, torch.Tensor):
            raise TypeError(
                'input to forward() must be torch.xTensor')
            
class DemoPINN(nn.Module):
    def __init__(self, hidden_layers=[32, 64, 64, 32],  output_dim = 2,use_residual=True):
        """
        Initialize PINN for mapping (x,y,z,t) -> (u,w) with optional residual connections
        Args:
            hidden_layers: List of neurons in hidden layers
            use_residual: Whether to use residual connections between layers
        """
        super(DemoPINN, self).__init__()
        
        # Input dimension is 4 (x,y,z,t)
        input_dim = 4
        # Output dimension is 2 (u,w)

        
        self.use_residual = use_residual
        self.layers = nn.ModuleList()
        
        # Input projection layer
        self.input_proj = nn.Linear(input_dim, hidden_layers[0])
        self.input_act = nn.Tanh()
        
        # Hidden layers with residual connections
        for i in range(len(hidden_layers)-1):
            # Each residual block contains:
            # 1. Linear transformation
            # 2. Activation
            # 3. Optional projection if dimensions don't match
            layer_block = nn.ModuleDict({
                'linear': nn.Linear(hidden_layers[i], hidden_layers[i+1]),
                'activation': Sine(),
            })
            
            # Add projection layer if dimensions don't match and residual connections are enabled
            if use_residual and hidden_layers[i] != hidden_layers[i+1]:
                layer_block['projection'] = nn.Linear(hidden_layers[i], hidden_layers[i+1])
            
            self.layers.append(layer_block)
            
        # Output layer
        self.output = nn.Linear(hidden_layers[-1], output_dim)
        
    def forward(self, x):
        """
        Forward pass of the network with residual connections
        Args:
            x: Input tensor of shape (batch_size, 4) containing (x,y,z,t)
        Returns:
            Output tensor of shape (batch_size, 2) containing (u,w)
        """
        # Initial projection
        x = self.input_act(self.input_proj(x))
        
        # Process through residual blocks
        for layer in self.layers:
            identity = x
            
            # Forward through linear and activation
            out = layer['activation'](layer['linear'](x))
            
            # Add residual connection if enabled
            if self.use_residual:
                # Use projection if dimensions don't match
                if 'projection' in layer:
                    identity = layer['projection'](identity)
                out = out + identity
                
            x = out
            
        # Final output layer
        return self.output(x)

def compute_V_loss(model, points, values):
    points.requires_grad_(True)
     # regular term
    #shape of predictions (N,2)
    predictions = model(points)
    mse_loss = torch.mean((predictions - values)**2)
    return mse_loss
       

def compute_loss(modelU, dv_dt,nabla_v, points, values):
    """
    Compute total loss with physical constraints:
    1. MSE loss between predictions and true values
    2. Killing term: ∇u + (∇u)ᵀ = 0
    3. Observed time derivative constraint
    
    Args:
        model: PINN model
        points: Tensor of shape (N, 4) containing input coordinates (x,y,z,t)
        values: Tensor of shape (N, 2) containing true values (u,w)
    Returns:
        total_loss: Combined loss from all terms
    """
    points.requires_grad_(True)
     # regular term
    #shape of predictions (N,2)
    predictionsU = modelU(points)
  
    
    # Calculate velocity gradients for killing term
    grad_outputs = torch.ones_like(predictionsU[:, 0])  # For each sample point
    grad_component1_u = torch.autograd.grad(predictionsU[:, 0], points, grad_outputs=grad_outputs, create_graph=True)[0]
    grad_component2_u= torch.autograd.grad(predictionsU[:, 1], points, grad_outputs=grad_outputs, create_graph=True)[0]
    
    # Construct velocity gradient tensor (2D) for each point
    nabla_u = torch.stack([
        grad_component1_u[:, :2],  # Take only x,y components
        grad_component2_u[:, :2]
    ], dim=1)  # Shape: [batch_size, 2, 2]
    
    # Killing term: ∇u + (∇u)ᵀ = 0 for each point
    killing_term = nabla_u + nabla_u.transpose(1, 2)  # Shape: [batch_size, 2, 2]
    killing_loss = torch.mean(torch.sum(killing_term**2, dim=(1,2)))  # Mean over batch, sum over components


    mse_loss = torch.mean((predictionsU - values)**2)


    # # Calculate time derivatives
    # ∂v1/∂t
    du1_dt = torch.autograd.grad(predictionsU[:, 0], points, grad_outputs=grad_outputs, create_graph=True)[0][:, 3]
    # ∂v2/∂t
    du2_dt = torch.autograd.grad(predictionsU[:, 1], points, grad_outputs=grad_outputs, create_graph=True)[0][:, 3]
    du_dt = torch.stack([du1_dt, du2_dt], dim=1)  # shape: [N, 2]
    
    u = predictionsU  # u(x,t)
    v=values
    # # Calculate convective terms: ∇u·v and ∇v·u
    nabla_u_v = torch.bmm(nabla_u, v.unsqueeze(2)).squeeze()  # Matrix-vector product for each batch
    nabla_v_u = torch.bmm(nabla_v, u.unsqueeze(2)).squeeze()  # Matrix-vector product for each batch
    
    # # Observed time derivative constraint
    D = dv_dt-du_dt   + nabla_v_u  - nabla_u_v
    derivative_loss = torch.mean(D**2)
    
    # Combine losses with weights
    total_loss =  0.01 *mse_loss + killing_loss+derivative_loss
    
    return total_loss

class area2D:
    def __init__(self,minX,minY,maxX,maxY) -> None:
        self.minX=minX
        self.maxX=maxX
        self.minY=minY
        self.maxY=maxY
        
    def inside(self,point2d):
        return point2d[0] >= self.minX and point2d[0] <=self.maxX and point2d[1] >= self.minY and point2d[1] <= self.maxY 


def compute_loss_limited_region(modelU, dv_dt, nabla_v, points, values, area):    
    # Filter dv_dt and nabla_v tensors for points inside area
    mask = torch.tensor([area.inside(p[:2]) for p in points], device=points.device)
    new_points = points[mask]
    new_velocities = values[mask]
    new_dv_dt = dv_dt[mask]
    new_nabla_v = nabla_v[mask]
    
    # Compute loss only for points inside the area
    return compute_loss(modelU, new_dv_dt, new_nabla_v, new_points, new_velocities)
            
        
    




    

def GenerateVectorField2DFromCoordinateNetwork(Network, GridX, GridY, GridT, domainMinBoundary, domainMaxBoundary, tmin, tmax, device=None):
    """Generate a discrete vector field by evaluating a coordinate network on a grid.
    
    Args:
        Network: Neural network that takes (x,y,z,t) coordinates and outputs (u,w) velocities
        GridX: Number of points in x direction
        GridY: Number of points in y direction 
        GridT: Number of time steps
        device: torch.device to use for computation (defaults to Network's device)
        
    Returns:
        UnsteadyVectorField2D: Discrete vector field evaluated on the grid
    """
    if device is None:
        device = next(Network.parameters()).device
    # Create coordinate grids on device
    x = torch.linspace(domainMinBoundary[0], domainMaxBoundary[0], GridX, device=device)
    y = torch.linspace(domainMinBoundary[1], domainMaxBoundary[1], GridY, device=device)
    t = torch.linspace(tmin, tmax, GridT, device=device)
    
    # Create meshgrid for all coordinates
    t_grid, y_grid, x_grid = torch.meshgrid(t, y, x, indexing='ij')
    z_grid = torch.zeros_like(x_grid)
    
    # Reshape into (N, 4) tensor of points
    points = torch.stack([x_grid, y_grid, z_grid, t_grid], dim=-1)
    points = points.reshape(-1, 4)
    
    # Evaluate network in batches to avoid memory issues
    batch_size = 10000
    field = []
    
    Network = Network.to(device)
    with torch.no_grad():
        for i in range(0, points.shape[0], batch_size):
            batch_points = points[i:i + batch_size]
            batch_velocities = Network(batch_points)
            field.append(batch_velocities)
    
    # Combine batches and reshape to field dimensions
    field = torch.cat(field, dim=0)
    field = field.reshape(GridT, GridY, GridX, 2)
    
    # Create vector field object (move field back to CPU for storage)
    vectorField = UnsteadyVectorField2D(GridX, GridY, GridT, domainMinBoundary, domainMaxBoundary, tmin, tmax)
    vectorField.field = field.cpu().numpy()
    
    return vectorField


def GenerateVectorField3DFromCoordinateNetwork(Network, GridX, GridY, GridZ, GridT, domainMinBoundary, domainMaxBoundary, tmin, tmax, device=None):
    """Generate a discrete vector field by evaluating a coordinate network on a grid.
    
    Args:
        Network: Neural network that takes (x,y,z,t) coordinates and outputs (u,v,w) velocities
        GridX: Number of points in x direction
        GridY: Number of points in y direction 
        GridZ: Number of points in z direction 
        GridT: Number of time steps
        device: torch.device to use for computation (defaults to Network's device)
        
    Returns:
        UnsteadyVectorField3D: Discrete vector field evaluated on the grid
    """
    if device is None:
        device = next(Network.parameters()).device
    # Create coordinate grids on device
    x = torch.linspace(domainMinBoundary[0], domainMaxBoundary[0], GridX, device=device)
    y = torch.linspace(domainMinBoundary[1], domainMaxBoundary[1], GridY, device=device)
    z = torch.linspace(domainMinBoundary[2], domainMaxBoundary[2], GridZ, device=device)
    t = torch.linspace(tmin, tmax, GridT, device=device)
    
    # Create meshgrid for all coordinates
    t_grid, z_grid, y_grid, x_grid = torch.meshgrid(t, z, y, x, indexing='ij')
    
    # Reshape into (N, 4) tensor of points
    points = torch.stack([x_grid, y_grid, z_grid, t_grid], dim=-1)
    points = points.reshape(-1, 4)
    
    # Evaluate network in batches to avoid memory issues
    batch_size = 100000
    field = []
    
    Network = Network.to(device)
    with torch.no_grad():
        for i in range(0, points.shape[0], batch_size):
            batch_points = points[i:i + batch_size]
            batch_velocities = Network(batch_points)
            field.append(batch_velocities)
    
    # Combine batches and reshape to field dimensions
    field = torch.cat(field, dim=0)
    field = field.reshape(GridT, GridZ, GridY, GridX, 3)
    
    # Create vector field object (move field back to CPU for storage)
    vectorField = UnsteadyVectorField3D(GridX, GridY, GridZ, GridT, domainMinBoundary, domainMaxBoundary, tmin, tmax)
    vectorField.field = field.cpu().numpy()
    
    return vectorField

def cluster_dividing(points, velocities, K=4):
    """Cluster points based on combined spatial and velocity features using K-means.
    
    Args:
        points: Tensor of shape (N, 4) containing (x,y,z,t) coordinates
        velocities: Tensor of shape (N, 2) containing (u,w) velocities
        K: Number of clusters (default: 8)
    
    Returns:
        cluster_indices: Tensor of shape (N,) containing cluster assignments
        cluster_centers: Tensor of shape (K, 6) containing cluster centroids
    """
    points_cpu = points
    velocities_cpu = velocities
    
    # Combine spatial and velocity features
    # Only use x,y coordinates (skip z) and velocities for clustering
    features = np.concatenate([
        points_cpu[:, :],  # x,y coordinates
        points_cpu[:, 3:4],  # time
        velocities_cpu      # u,w velocities
    ], axis=1)
    
    # Normalize features to [0,1] range for better clustering
    features = (features - features.min(axis=0)) / (features.max(axis=0) - features.min(axis=0))
    
    # Use MiniBatchKMeans for faster clustering with large datasets
    from sklearn.cluster import MiniBatchKMeans
    kmeans = MiniBatchKMeans(n_clusters=K, 
                            batch_size=1024,
                            random_state=42)
    
    # Perform clustering
    cluster_indices = kmeans.fit_predict(features)
    
    # Convert back to torch tensors
    cluster_indices = torch.from_numpy(cluster_indices)
    cluster_centers = torch.from_numpy(kmeans.cluster_centers_)
    
    # Group points and velocities by cluster
    clustered_points = []
    clustered_velocities = []
    for i in range(K):
        mask = cluster_indices == i
        clustered_points.append(points[mask])
        clustered_velocities.append(velocities[mask])
    
    return clustered_points, clustered_velocities
    
    
def neural_partial_v_partial_t(model_vector_feild, GridX, GridY, GridT, domainMinBoundary, domainMaxBoundary, tmin, tmax) -> UnsteadyVectorField2D:
    """
    Construct a function that computes ∂v/∂t using the neural network model
    
    Args:
        model_vector_feild: Neural network model that maps (x,y,z,t) to velocity field v
        
    Returns:
        function that takes points tensor as input and returns ∂v/∂t at those points
    """
    
    def partial_v_partial_t(points):
        # Ensure points requires gradient for autograd
        device = next(model_vector_feild.parameters()).device
        points= points.to(device)
        points.requires_grad_(True)
        
        # Get velocity predictions from model
        v = model_vector_feild(points)
        
        # Setup for gradient computation
        grad_outputs = torch.ones_like(v[:, 0])
        
        # Compute ∂v1/∂t and ∂v2/∂t using autograd
        dv1_dt = torch.autograd.grad(v[:, 0], points, grad_outputs=grad_outputs, create_graph=True)[0][:, 3]
        dv2_dt = torch.autograd.grad(v[:, 1], points, grad_outputs=grad_outputs, create_graph=True)[0][:, 3]
        
        # Stack components into tensor [N, 2]
        dv_dt = torch.stack([dv1_dt, dv2_dt], dim=1)
        
        return dv_dt
        
        
    # Create grid points for evaluation
    x = torch.linspace(domainMinBoundary[0], domainMaxBoundary[0], GridX)
    y = torch.linspace(domainMinBoundary[1], domainMaxBoundary[1], GridY)
    t = torch.linspace(tmin, tmax, GridT)
    
    # Create meshgrid
    X, Y, T = torch.meshgrid(x, y, t, indexing='ij')
    
    # Reshape into points tensor [N, 4] where each point is (x,y,0,t)
    points = torch.stack([X.flatten(), Y.flatten(), 
                         torch.zeros_like(X.flatten()), T.flatten()], dim=1)
    
    # Compute partial_v_partial_t at all grid points
    dv_dt = partial_v_partial_t(points)
    
    # Reshape result back to field dimensions [T, Y, X, 2]
    dv_dt = dv_dt.reshape(GridX, GridY, GridT, 2)
    dv_dt = dv_dt.permute(2, 1, 0, 3).detach().cpu().numpy()  # Reorder to [T, Y, X, 2]
    
    # Create UnsteadyVectorField2D for output
    dv_dt_field = UnsteadyVectorField2D(
        GridX, GridY, GridT,
        domainMinBoundary=domainMinBoundary,
        domainMaxBoundary=domainMaxBoundary,
        tmin=tmin, tmax=tmax
    )
    dv_dt_field.field = dv_dt
    return dv_dt_field


def central_difference_partial_v_partial_t(vector_feild:UnsteadyVectorField2D) -> UnsteadyVectorField2D:
    """
    Compute partial derivative of velocity field with respect to time using central difference
    
    Args:
        vector_feild: Input unsteady vector field
    Returns:
        UnsteadyVectorField2D containing dv/dt computed using central difference
    """
    # Get dimensions
    T, H, W, _ = vector_feild.field.shape
    
    # Initialize output field with same spatial dimensions
    dv_dt = np.zeros((T, H, W, 2))
    dt=vector_feild.timeInterval
    # Use central difference for interior points
    dv_dt[1:-1] = (vector_feild.field[2:] - vector_feild.field[:-2]) / (2 * dt)
    
    # Forward difference for first point
    dv_dt[0] = (vector_feild.field[1] - vector_feild.field[0]) / dt
    
    # Backward difference for last point 
    dv_dt[-1] = (vector_feild.field[-1] - vector_feild.field[-2]) /dt
    
    # Create new vector field with same domain
    dv_dt_field = UnsteadyVectorField2D(
        vector_feild.Xdim,
        vector_feild.Ydim,
        vector_feild.time_steps,
        domainMinBoundary=vector_feild.domainMinBoundary,
        domainMaxBoundary=vector_feild.domainMaxBoundary,
        tmin=vector_feild.tmin,
        tmax=vector_feild.tmax
    )
    dv_dt_field.field=dv_dt
    return dv_dt_field
  
    
    

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    flow_asset_folder="C:\\Users\\zhanx0o\\OneDrive - KAUST\\WorkingInProcess\\FLowVisAssets\\flowNetCDFdata"
    cylider_netCDF=os.path.join(flow_asset_folder,"boussinesq.nc")
    vectorField2d =NetCDFLoader.load_vector_field2d(cylider_netCDF,600,800);
    GridX,GridY,GridT=vectorField2d.Xdim,vectorField2d.Ydim,vectorField2d.time_steps
    # scaler = GradScaler()
    # Generate some dummy data
    N = 64000
    points, velocities = samplePointsVelocity(vectorField2d,N)
    
    
    # Create model
    points, velocities=points.to(device), velocities.to(device)
    model_V = DemoPINN().to(device)

    
    # Create optimizer
    optimizerV = torch.optim.Adam(model_V.parameters(), lr=0.001)  

    
    # Training loop
    n_steps = 8000
    epsilo=0.0005
    for step in range(n_steps):
        optimizerV.zero_grad();
        lossV = compute_V_loss(model_V, points, velocities)
        lossV.backward()
        optimizerV.step()
        if (step + 1) % 200 == 0 or step==0:
            print(f'lossV- Epoch [{step+1}/{n_steps}], Loss: {lossV.item():.4f}')
        if lossV.item()<epsilo:
            break        
    
    
    dv_dt=None
    # with torch.no_grad:
    points.requires_grad_(True)
    predictionsV = model_V(points)
    grad_outputs_v1 = torch.ones_like(predictionsV[:, 0])  # For first component
    # ∂v1/∂t
    dv1_dt = torch.autograd.grad(predictionsV[:, 0], points, grad_outputs=grad_outputs_v1, create_graph=True)[0][:, 3]
    # ∂v2/∂t
    dv2_dt = torch.autograd.grad(predictionsV[:, 1], points, grad_outputs=grad_outputs_v1, create_graph=True)[0][:, 3]
    dv_dt = torch.stack([dv1_dt, dv2_dt], dim=1).detach()  # shape: [N, 2]
    
    grad_component1_v = torch.autograd.grad(predictionsV[:, 0], points, grad_outputs=grad_outputs_v1, create_graph=True)[0]
    grad_component2_v= torch.autograd.grad(predictionsV[:, 1], points, grad_outputs=grad_outputs_v1, create_graph=True)[0]
    # Construct velocity gradient tensor (2D) for each point
    nabla_v = torch.stack([
        grad_component1_v[:, :2],  # Take only x,y components
        grad_component2_v[:, :2]
    ], dim=1).detach()   # Shape: [batch_size, 2, 2]   
        
        
        
    #main optimiation for u field starting here
    n_steps = 100
    epsilo=0.0005
    # Keep track of previous loss for convergence check
    prev_loss = float('inf')
    patience = 5  # Number of epochs to wait before early stopping
    min_delta = 1e-5  # Minimum change in loss to be considered as improvement
    patience_counter = 0
    model_U = DemoPINN().to(device)
    optimizerU = torch.optim.Adam(model_U.parameters(), lr=0.002)  
    for step in range(n_steps):
        optimizerU.zero_grad()
        # with autocast():
        lossU = compute_loss(model_U, dv_dt, nabla_v, points, velocities)
        lossU.backward()
        optimizerU.step()
            # mix precision backward
        # scaler.scale(lossU).backward()
        # scaler.step(optimizerU)
        # scaler.update()
        
        current_loss = lossU.item()
        if (step + 1) % 200 == 0 or step == 0:
            print(f'lossU- Epoch [{step+1}/{n_steps}], Loss: {current_loss:.6f}')
        # Check for convergence
        if current_loss < epsilo:
            print("Reached target loss threshold. Stopping training.")
            break
        # Check if loss is not decreasing
        if prev_loss - current_loss < min_delta:
            patience_counter += 1
            if patience_counter >= patience:
                print("Loss has stopped decreasing. Stopping training.")
                break
        else:
            patience_counter = 0
            
        
        
    #visualize result field
    recRes=GenerateVectorField2DFromCoordinateNetwork(model_V,GridX,GridY,GridT,vectorField2d.domainMinBoundary,vectorField2d.domainMaxBoundary,vectorField2d.tmin,vectorField2d.tmax)
    name="test_pinn_V"
    # heatStreamPlot(vectorField2d,timeStepSkip=20,saveFolder="./test_pinn",saveName=f"sh__origin_v")
    # heatStreamPlot(recRes,timeStepSkip=20,saveFolder="./test_pinn",saveName=f"sh__{name}")
    recResU=GenerateVectorField2DFromCoordinateNetwork(model_U,GridX,GridY,GridT,vectorField2d.domainMinBoundary,vectorField2d.domainMaxBoundary,vectorField2d.tmin,vectorField2d.tmax)
    name="test_pinn_U"
    # heatStreamPlot(recResU,timeStepSkip=20,saveFolder="./test_pinn",saveName=f"sh__{name}")
    
    #compare dvdt
    dvdt_numerical=central_difference_partial_v_partial_t(vectorField2d)
    dvdt_estimate=neural_partial_v_partial_t(model_V ,GridX,GridY,GridT,vectorField2d.domainMinBoundary,vectorField2d.domainMaxBoundary,vectorField2d.tmin,vectorField2d.tmax)
    heatStreamPlot(dvdt_numerical,timeStepSkip=20,saveFolder="./test_pinn",saveName=f"sh__numerical_dvdt")
    heatStreamPlot(dvdt_estimate,timeStepSkip=20,saveFolder="./test_pinn",saveName=f"sh__estimate_dvdt")
    
    
    
    
    
def experiment_dv_dt_fitting_iteration():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    flow_asset_folder="C:\\Users\\xingdi\\OneDrive - KAUST\WorkingInProcess\\FLowVisAssets\\flowData2d"
    cylider_netCDF=os.path.join(flow_asset_folder,"boussinesq.nc")
    vectorField2d =NetCDFLoader.load_vector_field2d(cylider_netCDF,600,630);
    dvdt_numerical=central_difference_partial_v_partial_t(vectorField2d)
    heatStreamPlot(vectorField2d,timeStepSkip=80,saveFolder="./test_pinn",saveName=f"sh__numerical_v",colorBarmin=0.0,colorBarmax=1.0,redudant=True)
    heatStreamPlot(dvdt_numerical,timeStepSkip=80,saveFolder="./test_pinn",saveName=f"sh__numerical_dvdt",colorBarmin=0.0,colorBarmax=1.0,redudant=True)
    
    GridX,GridY,GridT=vectorField2d.Xdim,vectorField2d.Ydim,vectorField2d.time_steps
    N = 200000
    points, velocities = hybridSampling(vectorField2d,N,velocityDerivativeSaliencse)
    points, velocities=points.to(device), velocities.to(device)
    # scaler = GradScaler()
    # Generate some dummy data
    
    def genNeural_dvdt(save_check_points=None):
        # Create model
        model_V = DemoPINN().to(device)
        optimizerV = torch.optim.Adam(model_V.parameters(), lr=0.002)  
        if save_check_points is None:
            save_check_points=[i for i in range(500, 20001, 500)]
        # Training loop
        n_steps = max(save_check_points)
        print(f"fitting {n_steps} iterations")
        epsilo=0.00005
        for step in range(n_steps):
            optimizerV.zero_grad();
            lossV = compute_V_loss(model_V, points, velocities)
            lossV.backward()
            optimizerV.step()
            if (step + 1) % 200 == 0 or step==0:
                print(f'lossV- Epoch [{step+1}/{n_steps}], Loss: {lossV.item():.7f}')
            if lossV.item()<epsilo:
                break        
            if step in save_check_points:
                dvdt_estimate=neural_partial_v_partial_t(model_V ,GridX,GridY,GridT,vectorField2d.domainMinBoundary,vectorField2d.domainMaxBoundary,vectorField2d.tmin,vectorField2d.tmax)
                heatStreamPlot(dvdt_estimate,timeStepSkip=80,saveFolder="./test_pinn",saveName=f"sh__estimate_dvdt_{step}",colorBarmin=0.0,colorBarmax=1.0)
                recRes=GenerateVectorField2DFromCoordinateNetwork(model_V,GridX,GridY,GridT,vectorField2d.domainMinBoundary,vectorField2d.domainMaxBoundary,vectorField2d.tmin,vectorField2d.tmax)
                heatStreamPlot(recRes,timeStepSkip=80,saveFolder="./test_pinn",saveName=f"sh__estimate_v_{step}",colorBarmin=0.0,colorBarmax=1.0,redudant=False)
    
    
    genNeural_dvdt()
        

def visualize3d(self, timeSlice: int=0, save_path: str=None):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        # Ensure the field is initialized
        if self.field is None:
            raise ValueError("Field data is not initialized.")

        # Check if timeSlice is within the valid range
        if timeSlice < 0 or timeSlice >= self.field.shape[0]:
            raise ValueError("Invalid timeSlice. Must be within the range of the field data.")

        # Get the slice of the field for the specified time
        field_slice = self.field[timeSlice, :, :, :]

        # Create a grid for the vector field
        x = np.linspace(self.domainMinBoundary[0], self.domainMaxBoundary[0], self.Xdim)
        y = np.linspace(self.domainMinBoundary[1], self.domainMaxBoundary[1], self.Ydim)
        z = np.linspace(self.domainMinBoundary[2], self.domainMaxBoundary[2], self.Zdim)
        X, Y, Z = np.meshgrid(x, y, z, indexing='xy')  # Use 'xy' indexing

        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        field_slice = np.transpose(field_slice, (1, 2, 0,3))  # Change to [Ydim, Xdim, Zdim]
        
        # Sample points to reduce clutter
        stepZ = 30 # Adjust this value to control the density of arrows
        step = 20  # Adjust this value to control the density of arrows
        ax.quiver(X[::step, ::step, ::stepZ], Y[::step, ::step, ::stepZ], Z[::step, ::step, ::stepZ],
                   field_slice[..., 0][::step, ::step, ::stepZ], 
                   field_slice[..., 1][::step, ::step, ::stepZ], 
                   field_slice[..., 2][::step, ::step, ::stepZ], 
                   length=0.1, normalize=False)

        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        # Set equal aspect ratio for the axes
        ax.set_box_aspect([self.domainMaxBoundary[0] - self.domainMinBoundary[0],
                        self.domainMaxBoundary[1] - self.domainMinBoundary[1],
                        self.domainMaxBoundary[2] - self.domainMinBoundary[2]])

        # Set limits explicitly to ensure proper scaling
        ax.set_xlim(self.domainMinBoundary[0], self.domainMaxBoundary[0])
        ax.set_ylim(self.domainMinBoundary[1], self.domainMaxBoundary[1])
        ax.set_zlim(self.domainMinBoundary[2], self.domainMaxBoundary[2])
        plt.title(f'3D Vector Field at Time Slice {timeSlice}')

        plt.show()
        if save_path is not None:
            plt.savefig(save_path)
        
        
#give me function compute IVD
def compute_IVD(self,x,y,z):
    return (x**2+y**2+z**2)**0.5
def compute_IVD_derivative(self,x,y,z):
    return 2*x*y*z
    





def experiment_dv_dt_fitting_3d():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    flow_asset_folder="C:\\Users\\zhanx0o\\OneDrive - KAUST\\WorkingInProcess\\FLowVisAssets\\flowData3D"
    cylider_netCDF=os.path.join(flow_asset_folder,"halfcylinderRe160.nc")
    vectorField =NetCDFLoader.load_vector_field3d(cylider_netCDF,200,230);
    N=4000000
    n_steps=50000
    # visualize3d(vectorField,100,"./test_pinn/3d_test.png")
    points, velocities = samplePointsVelocity3d(vectorField,N)
    points, velocities=points.to(device), velocities.to(device)
    # Create model
    model_V = DemoPINN(output_dim = 3).to(device)
    optimizerV = torch.optim.Adam(model_V.parameters(), lr=0.002)  
    epsilo=0.00005
    for step in range(n_steps):
        optimizerV.zero_grad();
        lossV = compute_V_loss(model_V, points, velocities)
        lossV.backward()
        optimizerV.step()
        if (step + 1) % 200 == 0 or step==0:
            print(f'lossV- Epoch [{step+1}/{n_steps}], Loss: {lossV.item():.7f}')
        if lossV.item()<epsilo:
            break        
    GridX,GridY,GridZ,GridT=vectorField.Xdim,vectorField.Ydim,vectorField.Zdim,vectorField.time_steps
    recV=GenerateVectorField3DFromCoordinateNetwork(model_V,GridX//2,GridY//2,GridZ//2,GridT//2,vectorField.domainMinBoundary,vectorField.domainMaxBoundary,vectorField.tmin,vectorField.tmax)
    visualize3d(recV,50,"./test_pinn/3d_test_rec.png")
    visualize3d(vectorField,100,"./test_pinn/3d_test_raw.png")
    
    
    # # heatStreamPlot(dvdt_numerical,timeStepSkip=80,saveFolder="./test_pinn",saveName=f"sh__numerical_dvdt",colorBarmin=0.0,colorBarmax=1.0,redudant=True)
    # N = 200000

    # # scaler = GradScaler()
    # # Generate some dummy data
    # def genNeural_dvdt(save_check_points=None):
 
    #     if save_check_points is None:
    #         save_check_points=[i for i in range(500, 20001, 500)]
    #     # Training loop
    #     n_steps = max(save_check_points)
    #     print(f"fitting {n_steps} iterations")
 
            
    #         # if step in save_check_points:

         
    
    
    # genNeural_dvdt()
        








# Example usage
if __name__ == "__main__":
   experiment_dv_dt_fitting_iteration()
