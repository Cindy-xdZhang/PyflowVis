import torch
 
def PathlineTemporalSamplingLayer(in_pathline_src,temporal_sampling_ratio=0.5,random:bool=False):
    B, L_Full_length, K, C = in_pathline_src.shape
    L = int(0.5*L_Full_length)
    temporal_indices=None
    if random:        
        # Randomly downsample back to L steps 
        allL_indices=torch.randperm(L_Full_length)[:L]
        temporal_indices=torch.sort(allL_indices)[0]
    else:
        temporal_indices = torch.arange(0, L_Full_length, step=2)[:L]
    temporal_indices[0]=0
    temporal_indices[-1]=L_Full_length-1
    temporal_sampled_pathline = in_pathline_src[:,temporal_indices,:,:]
    return temporal_sampled_pathline,temporal_indices

def TemporalDownSampling(in_points_Cloud_xyz,sampledK):
    B,N, C =in_points_Cloud_xyz.shape
    temporalLength_full=N//sampledK
    L=max(temporalLength_full//2,2)
    if L>2:
        temporal_indices = torch.arange(0, temporalLength_full, step=2)[:L]
    else:
        temporal_indices= torch.arange(0, temporalLength_full, step=1)[:L]
    temporal_indices[0]=0
    temporal_indices[-1]=temporalLength_full-1
    # temporal_sampled_pathline = in_points_Cloud_xyz[:,temporal_indices,:,:]
    # temporal_sampled_feature = in_points_Cloud_feature[:,temporal_indices,:,:]
    return temporal_indices
        
def PathlineSpatialSamplingLayer(in_pathline_src,keepGroups,linesPerGroup,random:bool=False):
    B, L_Full_length, K, C = in_pathline_src.shape
    total_groups = K // linesPerGroup
    group_indices =None
    if random:
        # Randomly permute groups
        group_indices = torch.randperm(total_groups)[:keepGroups]
        mask = torch.zeros(K, dtype=torch.bool)
        for idx in group_indices:
            start = idx * linesPerGroup
            end = start + linesPerGroup
            mask[start:end] = True
        # Apply temporal and group sampling
        sampled_pathline = in_pathline_src[:, :, mask, :]
        # Random roll along the last dimension
        # roll_amount = torch.randint(0, int(K//8), (1,)).item()*4
        # sampled_pathline = torch.roll(sampled_pathline, shifts=roll_amount, dims=2)
        return sampled_pathline,mask
    else:
        group_indices = torch.arange(0, total_groups, step=max(1, total_groups // keepGroups))[:keepGroups]
        mask = torch.zeros(K, dtype=torch.bool)
        for idx in group_indices:
            start = idx * linesPerGroup
            end = start + linesPerGroup
            mask[start:end] = True
        sampled_pathline = in_pathline_src[:, :, mask, :]
        return sampled_pathline,mask
        
   
    
    
    
    


    
    
    
    
def PathlineSamplingLayer(pathline_src,keepGroups,linesPerGroup=4,temporal_sampling_range:list=[0.9,1.0]):
        B, L_Full_length, K, C = pathline_src.shape
        total_groups = K // linesPerGroup
        
        # L = torch.randint(int(temporal_sampling_range[0]*L_Full_length), int(temporal_sampling_range[1]*L_Full_length), (1,)).item()
        # Randomly downsample back to L steps 
        # allL_indices=torch.randperm(L_Full_length)[:L]
        # temporal_indices=torch.sort(allL_indices)[0]
        # temporal_indices[0]=0
        # temporal_sampled_pathline = pathline_src[:,temporal_indices,:,:]
        temporal_sampled_pathline=pathline_src
        temporal_indices=None
        # # Randomly  select first `keep_K` indices indices to keep
        # indices = torch.randperm(K)[:keepKlines]  
        # sampled_pathline=temporal_sampled_pathline[:, :, indices, :]
        
        # Randomly permute groups
        group_indices = torch.randperm(total_groups)[:keepGroups]
        # Create a mask for the selected groups
        mask = torch.zeros(K, dtype=torch.bool)
        for idx in group_indices:
            start = idx * linesPerGroup
            end = start + linesPerGroup
            mask[start:end] = True

        # Apply temporal and group sampling
        sampled_pathline = temporal_sampled_pathline[:, :, mask, :]
        
        return sampled_pathline,temporal_indices,mask    