import torch
import torch.nn as nn
import torch.nn.functional as F
from ..build import MODELS
from torch import nn, einsum
from .samplingLayers import *
def get_graph_feature(x, idx):
    batch_size, num_points, num_dims = x.size()
    k = idx.size(2)
    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    x = x.contiguous().view(batch_size * num_points, -1)[idx, :]
    x = x.view(batch_size, num_points, k, num_dims)
    return x


# PosE for Raw-point Embedding 
class PosE_Initial(nn.Module):
    def __init__(self, in_dim, out_dim, alpha=100, beta=1000):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta

    def forward(self, xyz):
        # xyz=xyz.permute(0,2,1)
        B,  N,_= xyz.shape    
        feat_dim = self.out_dim // (self.in_dim * 2)
        
        feat_range = torch.arange(feat_dim).float().cuda()     
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim)
        div_embed = torch.div(self.beta * xyz.unsqueeze(-1), dim_embed)

        sin_embed = torch.sin(div_embed)
        cos_embed = torch.cos(div_embed)
        position_embed = torch.stack([sin_embed, cos_embed], dim=4).flatten(3)
        # position_embed = position_embed.permute(0, 1, 3, 2).reshape(B, self.out_dim, N)
        position_embed = position_embed.reshape(B,N,self.out_dim)
        return position_embed


class TransitionDown(nn.Module):
    def __init__(self, in_dim, out_dim, n_sample, k=16):
        super().__init__()
        self.knn_k = k
        self.n_spatial_sample = n_sample
        self.in_dim=in_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim+3, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x,pos):
        # x: (B, N, C), pos: (B, N, 3)
        B, N, C = x.shape
        # temporal_idx sampling
        temporal_idx = TemporalDownSampling(pos, self.n_spatial_sample)
        new_pos = pos.contiguous().view(B, -1,self.n_spatial_sample,3)[:,temporal_idx,:,:].reshape(B,-1,3)
        new_x = x.contiguous().view(B, -1,self.n_spatial_sample,C)[:,temporal_idx,:,:].reshape(B,-1,C)
        
        B, N, C = new_x.shape
        # Find k nearest neighbors
        inner = -2 * torch.matmul(new_pos, new_pos.transpose(2, 1))
        xx = torch.sum(new_pos**2, dim=2, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        knn_idx = pairwise_distance.topk(k=self.knn_k, dim=-1)[1]  # (B, N, k)
        grouped_pos =get_graph_feature(new_pos, knn_idx)  # (B, n_sample, k, 3)
        grouped_x = get_graph_feature(new_x, knn_idx)  # (B, n_sample, k, C)
        # Local spatial encoding
        grouped_pos_enc = grouped_pos - new_pos.unsqueeze(2)
        # Concatenate features and positions
        grouped_features = torch.cat([grouped_x, grouped_pos_enc], dim=-1)
        # MLP
        grouped_features = self.mlp(grouped_features.view(B * N * self.knn_k, -1))
        grouped_features = grouped_features.view(B, N , self.knn_k, -1)
        
        # Local max pooling
        new_x = grouped_features.max(dim=2)[0]+grouped_features.mean(dim=2)[0]
        
        return new_x ,new_pos
    
    
# standard  point transformer operations
class PointTransformerLayer(nn.Module):
    def __init__(self, dim,k=16):
        super().__init__()
        self.k = k
        self.dim = dim

        self.pos_embedding = nn.Sequential(
            nn.Linear(3, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        self.attn_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        self.linear_q = nn.Linear(dim, dim, bias=False)
        self.linear_k = nn.Linear(dim, dim, bias=False)
        self.linear_v = nn.Linear(dim, dim, bias=False)
    
        self.linear_out = nn.Linear(dim, dim)

    def forward(self,  x,pos):
        # x: (B, N, C), pos: (B, N, 3)
        B, N, C = x.shape
        
        # Compute pairwise distances pi - pj
        pos_diff = pos.unsqueeze(2) - pos.unsqueeze(1)  # (B, N, N, 3)
        pos_enc = self.pos_embedding(pos_diff)  # (B, N, N, C)
   

        q = self.linear_q(x).unsqueeze(2)   # (B, N,1, C)
        k = self.linear_k(x).unsqueeze(2)   # (B, N,1, C)
        v = self.linear_v(x).unsqueeze(2)   # (B, N,1, C)
        
        energy = q - k + pos_enc  # (B, N, N, C)
        attn = self.attn_mlp(energy)
        attn = F.softmax(attn, dim=-2)  # (B, N, N, C)

        out = torch.sum(attn * v, dim=2)  # (B, N, C)
        out = F.relu(self.linear_out(out))
        return out
    
    

    def get_graph_feature(self, x, idx):
        batch_size, num_points, num_dims = x.size()
        k = idx.size(2)
        idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        x = x.contiguous().view(batch_size * num_points, -1)[idx, :]
        x = x.view(batch_size, num_points, k, num_dims)
        return x





@MODELS.register_module()
class PointTransformerV3(nn.Module):
    def __init__(self, in_channels, PathlineGroups,KpathlinePerGroup, num_classes=1, num_encoder_layers=3,dmodel=252,dropout=0.1,k=16,**kwargs):
        super().__init__()
        self.input_dim = in_channels
        self.dim = dmodel
        self.knn_k=k #knn neighbor size
        self.pathlinePerGroup=KpathlinePerGroup
        
        self.keep_Groups = int(0.5* PathlineGroups)
        if  self.keep_Groups % 2 != 0:
            self.keep_Groups= (self.keep_Groups //2) * 2         
        # self.keep_Groups = PathlineGroups    
        self.raw_pos_embedding = PosE_Initial(3, dmodel//2)#dmodel must be mutiple of inchannels.
        self.feature_embedding = nn.Linear(7, dmodel//2)
        self.pointTransformer_layers= nn.ModuleList([
            PointTransformerLayer(dmodel*(2**id),self.keep_Groups*self.pathlinePerGroup) for id in range(num_encoder_layers)
        ])
        self.transition_down_layers= nn.ModuleList([
            TransitionDown(dmodel*(2**id),dmodel*(2**(id+1)), self.keep_Groups*self.pathlinePerGroup) for id in range(num_encoder_layers)
        ])
        self.final_dim=dmodel*(2**(num_encoder_layers))
        
        self.feature_propagation = nn.Linear(self.final_dim, self.final_dim,)
        self.fc = nn.Linear(self.final_dim, num_classes)    
        self.output=nn.Sigmoid()

        
    def forward(self, data):
        _,pathline_src=data
        tmp_sampled_pathline,temporal_indices=PathlineTemporalSamplingLayer(pathline_src,random=True)
        sampled_pathline,pathline_mask=PathlineSpatialSamplingLayer(tmp_sampled_pathline,self.keep_Groups,self.pathlinePerGroup,random=True)
        
        B, L, sampleK, C =sampled_pathline.shape
        points=sampled_pathline.reshape(B,L*sampleK,C)   
        # points: (B, N, 3+C)
        pos,x = points[:, :, :3],points[:, :, 3:]
        pos_emb=self.raw_pos_embedding(pos)
        feature_emb=self.feature_embedding(x)
        x=torch.concat((pos_emb,feature_emb),dim=-1)

        for  idx,layer in enumerate(self.pointTransformer_layers):
            x = x + layer(x,pos)
            x,pos=self.transition_down_layers[idx](x,pos)
        
        
    
        x=x.reshape(B,-1,sampleK, self.final_dim)
        
        #x shape [B,K,Dimodel]
        # global  pool
        x = x.mean(dim=1)+x.max(dim=1)[0]
        
        # return  self.output(self.fc(x))
        full_features = self.propagate_features(pathline_src[:,0,:,:], sampled_pathline[:,0,:,:], x,pathline_mask)
     
        # Apply final layers to get output for all pathlines
        full_output = self.output(self.fc(full_features)).squeeze(-1)
        return full_output

    def propagate_features(self, full_pathline, sampled_pathline, sampled_features,mask):
        B, K, C = full_pathline.shape
        _, sampled_K, C = sampled_pathline.shape
        _, _,  FeatureDim =  sampled_features.shape
        
        
        sampled_start_pos = sampled_pathline.reshape(B, sampled_K, C)[:, :, :2]
        full_start_pos = full_pathline.reshape(B, K, C)[:, :, :2]

        # Calculate pairwise distances
        inner = -2 * torch.matmul(full_start_pos, sampled_start_pos.transpose(2, 1))
        xx = torch.sum(full_start_pos**2, dim=2, keepdim=True)
        yy = torch.sum(sampled_start_pos**2, dim=2, keepdim=True).transpose(2, 1)
        pairwise_distance = xx + inner + yy

        # Find k nearest neighbors, knn_idx shape=[B,N_full,k]
        _, knn_idx = pairwise_distance.topk(k=self.knn_k, dim=-1, largest=False)

         # Gather features of k-nearest neighbors
        knn_features = sampled_features.view(B, -1, FeatureDim).unsqueeze(1).expand(-1, K, -1, -1)
        knn_features = knn_features.gather(2, knn_idx.unsqueeze(-1).expand(-1, -1, -1, FeatureDim))#in.gather(dim=2,knn_idx):     out[i][j][k][m]=in[i][j] [knn_idx[i][j][k][m]] [m]
        
        # Calculate weights based on distance
        weights = F.softmax(-pairwise_distance.gather(2, knn_idx), dim=2)

        # Weighted sum of neighbor features
        propagated_features = torch.sum(weights.unsqueeze(-1) * knn_features, dim=2)
        # propagated_features=propagated_features.reshape(B,K,self.dim)
        
        # Combine sampled and propagated features
        full_features = torch.zeros(B, K, FeatureDim, device=sampled_features.device)
        full_features[:,mask, :] = sampled_features.view(B, -1, FeatureDim)
        full_features[:, ~mask, :] = self.feature_propagation(propagated_features[:, ~mask, :])

        # shape of full_features=(B,  K, dmodel)
        return full_features