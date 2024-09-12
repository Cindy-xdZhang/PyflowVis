import torch
import torch.nn as nn
import torch.nn.functional as F
from ..build import MODELS
from torch import nn, einsum
from einops import repeat

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels,dropout=0.0):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.dropout=nn.Dropout(dropout)

    def forward(self, x):
        return  self.dropout(self.conv(x)) 

@MODELS.register_module()
class TobiasVortexBoundaryUnet(nn.Module):
    def __init__(self, in_channels, out_channels, n=3, features=64, dropout= 0.005,**kwargs):
        super(TobiasVortexBoundaryUnet, self).__init__()
        self.n = n
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
       

        # Down part of U-Net
        in_features = in_channels
        for _ in range(n):
            self.downs.append(DoubleConv(in_features, features,dropout))
            in_features = features
            features *= 2

        # Bottom part of U-Net
        self.bottleneck = DoubleConv(features // 2, features,dropout)

        # Up part of U-Net
        for _ in range(n):
            self.ups.append(
                nn.ConvTranspose2d(features, features // 2, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(features, features // 2,dropout))
            features //= 2

        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)
 

    def forward(self, x):
        skip_connections = []

        # Downsampling
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # Reverse the list

        # Upsampling
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = nn.functional.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        #bs,c,w,h ->reshape to bs ,w,h,c            
        x=self.final_conv(x)
        x=  F.softmax(x.permute(0, 2, 3, 1),dim=-1)
        return x




# Transformer Encoder Layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(dim_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(dim_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim_model)
        
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        # Self-attention
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed-forward network
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

# Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src):
        for layer in self.layers:
            src = layer(src)
        return src

# Transformer Decoder Layer
class TransformerDecoderLayer(nn.Module):
    def __init__(self, dim_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(dim_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(dim_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(dim_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim_model)

        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.norm3 = nn.LayerNorm(dim_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory):
        # Self-attention (q,k,v)->softmax(qk/sqrt(dim_model))*V
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Multi-head attention with encoder output
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Feed-forward network
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

# Transformer Decoder
class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, tgt, memory):
        for layer in self.layers:
            tgt = layer(tgt, memory)
        return tgt
def max_value(t):
    return torch.finfo(t.dtype).max

class PointTransformerLayer(nn.Module):
    def __init__(
        self,
        dim,
        pos_mlp_hidden_dim = 64,
        attn_mlp_hidden_mult = 4,
        num_neighbors = 10
    ):
        super().__init__()
        self.num_neighbors = num_neighbors

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        self.pos_mlp = nn.Sequential(
            nn.Linear(3, pos_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(pos_mlp_hidden_dim, dim)
        )

        self.attn_mlp = nn.Sequential(
            nn.Linear(dim, dim * attn_mlp_hidden_mult),
            nn.ReLU(),
            nn.Linear(dim * attn_mlp_hidden_mult, dim),
        )

    def forward(self, x, pos):
        B,L,K,C= pos.shape
        #x shape will be [bs, seqlenth,C=3/feature(dim_modle)]
        pos=pos.reshape(B,L*K,3)

        # get queries, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        # B,L*K,Posistion3D(x,y,t) = pos.shape
        # calculate relative positional embeddings
        rel_pos = pos[:, :, None, :] - pos[:, None, :, :]
        rel_pos_emb = self.pos_mlp(rel_pos)

        # use subtraction of queries to keys. i suppose this is a better inductive bias for point clouds than dot product
        qk_rel = q[:, :, None, :] - k[:, None, :, :]


        # expand values
        v = repeat(v, 'b j d -> b i j d', i = L)

        # add relative positional embeddings to value
        v = v + rel_pos_emb

        # use attention mlp, making sure to add relative positional embedding first
        sim = self.attn_mlp(qk_rel + rel_pos_emb)

        # attention
        attn = sim.softmax(dim = -2)

        # aggregate
        agg = einsum('b i j d, b i j d -> b i d', attn, v)
        return agg
    
    
class PointTransformeEncoder(nn.Module):
    def __init__(self,  num_layers,dim):
        super(PointTransformeEncoder, self).__init__()
        
        self.layers = nn.ModuleList([PointTransformerLayer(dim) for _ in range(num_layers)])
        self.num_layers = num_layers
      
    def forward(self, feature,pos):
        for layer in self.layers:
            feature = layer(feature,pos)
        return feature



    
class ReferenceFrameCNN(nn.Module):

    def __init__(self,in_channels, DataSizeX,DataSizeY,TimeSteps,out_channels, hiddenSize=64, dropout=0.1, **kwargs):
        super(ReferenceFrameCNN, self).__init__()
        # the input tensor of Conv3d should be in the shape of[batch_size, chanel=2,W=16, H=16, depth(timsteps)]
        self.conv1_1 = nn.Conv3d(in_channels=in_channels, out_channels=hiddenSize, kernel_size=3, stride=[2,2,1],padding=1)
        self.bn1_1 = nn.BatchNorm3d(hiddenSize)
        
        # self.CNNlayerCountN=4-2 #n=log2(max(H,W))-2
        self.conv2_1 = nn.Conv3d(in_channels=hiddenSize, out_channels=hiddenSize*2, kernel_size=3, stride=[2,2,1],padding=1)
        self.bn2_1 = nn.BatchNorm3d(hiddenSize*2)
        self.dropout_cnv1 = nn.Dropout(dropout)
        self.dropout_cnv2 = nn.Dropout(dropout)


        self.flatten = nn.Flatten()
        DataSizeX = DataSizeX // 4
        DataSizeY = DataSizeY // 4
        # TimeSteps =( (TimeSteps+1)//2 +1)//2
        TimeSteps =TimeSteps
        # Fully connected layer
        self.outputDim=out_channels
        self.fc1 = nn.Linear(hiddenSize*2 * DataSizeX * DataSizeY , 3)

        

    def forward(self, x):
        x =self.dropout_cnv1  (  F.relu(self.bn1_1(self.conv1_1(x))))
        x = self.dropout_cnv2  ( F.relu(self.bn2_1(self.conv2_1(x))))
        B,C,DX,DY,Tstep=x.shape
        # x = self.flatten(x)
        x=x.reshape(B,C*DX*DY,Tstep).permute(0,2,1)
        #x.shape =B,T,3
        x = self.fc1(x)
        # Convert [theta, tx, ty] to 3x3 transformation matrices
        theta, tx, ty = x.unbind(-1)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        # Create the transformation matrices
        transform_matrices = torch.zeros(B, Tstep, 3, 3, device=x.device)
        transform_matrices[:, :, 0, 0] = cos_theta
        transform_matrices[:, :, 0, 1] = -sin_theta
        transform_matrices[:, :, 1, 0] = sin_theta
        transform_matrices[:, :, 1, 1] = cos_theta
        transform_matrices[:, :, 0, 2] = tx
        transform_matrices[:, :, 1, 2] = ty
        transform_matrices[:, :, 2, 2] = 1.0
        return transform_matrices
    
    
@MODELS.register_module()
class PathlineTransformerWithCNNVecField(nn.Module):
    def __init__(self, Kpathlines,in_channels, DataSizeX,DataSizeY,TimeSteps,nhead=8, num_encoder_layers=3,dmodel=256,dim_feedforward=512, dropout=0.1,**kwargs):
        super(PathlineTransformerWithCNNVecField, self).__init__()
        self.DataSizeY=DataSizeY
        self.DataSizeX=DataSizeX
        self.keep_K = int(0.6* Kpathlines)
        if self.keep_K % 8 != 0:
            self.keep_K = (self.keep_K // 8) * 8
            
        C=3
        self.dmodel=dmodel
        #the pathlin first feed into embedding layer then plus position embedding(time)
        
        encoder_layer = TransformerEncoderLayer(self.dmodel, nhead, dim_feedforward, dropout)
        self.encoder_pos =TransformerEncoder(encoder_layer,  num_encoder_layers)   
        # self.symetryFn=nn.MaxPool1d(kernel_size=3,stride=1,padding=1)
        decoder_layer = TransformerDecoderLayer( self.dmodel, nhead, dim_feedforward, dropout)
        self.decoder = TransformerDecoder(decoder_layer, num_encoder_layers)

        self.vector_field_feature_exct=ReferenceFrameCNN(in_channels,DataSizeX,DataSizeY,TimeSteps,self.dmodel,dropout=dropout)
        #final ouput layer
        #sequence length fix to 21.
        self.ln1= nn.Linear(self.dmodel, 2)
        self.ln2= nn.Linear((128+3)*self.keep_K, self.dmodel)
        self.pos_mlp = nn.Sequential(
            # nn.Conv2d(3, 32, kernel_size=3, padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            # nn.Conv2d(32, 64, kernel_size=3, padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            # nn.Conv2d(64, 128, kernel_size=3, padding=1),
            # nn.BatchNorm2d(128),
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.pos_aggregate= nn.Sequential(
            nn.Linear(32*self.keep_K,128),
            nn.ReLU(),
            )


    def forward(self, data):
        vector_field,pathline_src=data
        #vector field data, shape of fature_v is [B,d_model]
        # feature_v=self.vector_field_feature_exct(vector_field)
        # feature_v=feature_v.unsqueeze(0)
        
        sampled_pathline=PathlineSamplingLayer( pathline_src,self.keep_K )
        B, L, K, C =sampled_pathline.shape
        
        #rel pos 
        # pos=sampled_pathline.reshape(B,L*self.keep_K,3)
        # rel_pos = pos[:, :, None, :] - pos[:, None, :, :]
        # relpos_embeddding=self.pos_mlp(rel_pos)#b,L*k,l*K,dimodel
        rel_pos_embedding = torch.zeros((B, L, self.keep_K,128), device=sampled_pathline.device)
        for step in range(L):
            pos=sampled_pathline[:,step,:,:].reshape(B,self.keep_K,3)
            rel_pos = pos[:, :, None, :] - pos[:, None, :, :]
            temp=self.pos_mlp(rel_pos)#B,K,K,32
            agg=self.pos_aggregate(temp.reshape(B,self.keep_K,-1))
            rel_pos_embedding[:,step,:,:]=agg
            
        
        
        #pathline_src shape B,L,keepK,3
        encoder_input=torch.concat([rel_pos_embedding,sampled_pathline],dim=3)
        encoder_input=self.ln2(encoder_input.reshape(B,L,-1))
        encoder_input=encoder_input.permute(1,0,2)
        # position_embed shape [L,B,dmodel],pathline_time shape [L,B,1]
        # pathline_time=pathline_src[:, :, indices, 2].reshape(B,L,self.keep_K).permute(1,0,2)
        # position_embed+=pathline_time
 
        # Transpose input [B,L,D] to fit the transformer input format [Seq_len, Batch_size, d_model]
        memory = self.encoder_pos(encoder_input)

        #vector field shape is [1,B,d_model]->output shape [1,B,d_model]
     
        output = self.decoder(memory, memory)
        #output shape [B,d_model]
        # output=output.squeeze(0)
        output=output[0].squeeze(0).reshape(B,-1)
        
        #cls
        x=self.ln1(output).reshape(B,2)
        x=F.softmax(x,dim=-1)         
        return x

def PathlineSamplingLayer(pathline_src,keepGroups,linesPerGroup=4,temporal_sampling_range:list=[0.9,1.0]):
        B, L_Full_length, K, C = pathline_src.shape
        total_groups = K // linesPerGroup
        
        # L = torch.randint(int(temporal_sampling_range[0]*L_Full_length), int(temporal_sampling_range[1]*L_Full_length), (1,)).item()
        L=L_Full_length
        # Randomly downsample back to L steps 
        allL_indices=torch.randperm(L_Full_length)[:L]
        temporal_indices=torch.sort(allL_indices)[0]
        temporal_indices[0]=0
        temporal_sampled_pathline = pathline_src[:,temporal_indices,:,:]
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
    


class PointTransformerLayerv2(nn.Module):
    def __init__(self, dim, dropout=0.1,k=16):
        super(PointTransformerLayerv2, self).__init__()
        self.k = k
        self.dim = dim

        self.pos_mlp = nn.Sequential(
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
        self.dropout_1 = nn.Dropout(dropout)
    
        self.linear_out = nn.Linear(dim, dim)

    def forward(self, x, pos, knn_idx):
        # x: (B, N, C), pos: (B, N, 3)
        B, N, C = x.shape
        # Get the features and positions of k-neighbors
        knn_feat = self.get_graph_feature(x, knn_idx)  # (B, N, k, C)
        knn_pos = self.get_graph_feature(pos,knn_idx)  # (B, N, k, 3)
        self.dropout_1(knn_feat) 
     
        pos_enc = self.pos_mlp(knn_pos - pos.unsqueeze(2))  # (B, N, k, C)

        q = self.linear_q(x).unsqueeze(2)  # (B, N, 1, C)
        k = self.linear_k(knn_feat)  # (B, N, k, C)
        v = self.linear_v(knn_feat)  # (B, N, k, C)

        energy = q - k + pos_enc  # (B, N, k, C)
        attn = self.attn_mlp(energy)
        attn = F.softmax(attn, dim=-2)  # (B, N, k, C)

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
class PointTransformer(nn.Module):
    def __init__(self, in_channels, PathlineGroups,KpathlinePerGroup, num_classes=1, num_encoder_layers=3,dmodel=256,dropout=0.1,k=16,**kwargs):
        super(PointTransformer, self).__init__()
        self.input_dim = in_channels
        self.dim = dmodel
        self.knn_k=k #knn neighbor size
        self.pathlinePerGroup=KpathlinePerGroup
        
        # self.keep_Groups = int(0.8* PathlineGroups)
        # if  self.keep_Groups % 4 != 0:
        #     self.keep_Groups= (self.keep_Groups //4) * 4         
        self.keep_Groups = PathlineGroups    
        self.embedding = nn.Linear(in_channels, dmodel)
        
        self.transformer_layers = nn.ModuleList([
            PointTransformerLayerv2(dmodel,dropout) for _ in range(num_encoder_layers)
        ])
        self.feature_propagation = nn.Linear(dmodel, dmodel)
        self.norm = nn.LayerNorm(dmodel)
        self.fc = nn.Linear(9*dmodel,  num_classes)
        self.output=nn.Sigmoid()
        # self.vector_field_feature_exct=ReferenceFrameCNN(2,32,32,5, self.dim ,dropout=dropout)

        
    def forward(self, data):
        vector_field,pathline_src=data
        
        sampled_pathline,temporal_indices,pathline_mask=PathlineSamplingLayer(pathline_src, self.keep_Groups,self.pathlinePerGroup)
        B, L, K, C =sampled_pathline.shape
   
        points=sampled_pathline.reshape(B,L*K,C)      
        # points: (B, N, 3+C)
        pos = points[:, :, :3]
        # Find k nearest neighbors
        inner = -2 * torch.matmul(pos, pos.transpose(2, 1))
        xx = torch.sum(pos**2, dim=2, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        knn_idx = pairwise_distance.topk(k=self.knn_k, dim=-1)[1]  # (B, N, k)
        
        x = self.embedding(points)
        for layer in self.transformer_layers:
            x = x + layer(x, pos,knn_idx)

        x = self.norm(x)
        x = x.mean(dim=1)
        return  self.output(self.fc(x))
        # full_features = self.propagate_features(pathline_src, sampled_pathline, x, temporal_indices,pathline_mask)
        # Apply final layers to get output for all pathlines
        # full_output = self.output(self.fc(full_features)).squeeze(-1)
        # return full_output

    
    def propagate_features(self, full_pathline, sampled_pathline, sampled_features, temporal_indices,mask):
        B, L, K, C = full_pathline.shape
        _, L_sampled, sampled_K, _ = sampled_pathline.shape
        
        sampled_pos = sampled_pathline.reshape(B, L_sampled*sampled_K, C)[:, :, :3]
        full_pos = full_pathline.reshape(B, L*K, C)[:, :, :3]

        # Calculate pairwise distances
        inner = -2 * torch.matmul(full_pos, sampled_pos.transpose(2, 1))
        xx = torch.sum(full_pos**2, dim=2, keepdim=True)
        yy = torch.sum(sampled_pos**2, dim=2, keepdim=True).transpose(2, 1)
        pairwise_distance = xx + inner + yy

        # Find k nearest neighbors
        _, knn_idx = pairwise_distance.topk(k=self.knn_k, dim=-1, largest=False)

         # Gather features of k-nearest neighbors
        knn_features = sampled_features.view(B, -1, self.dim).unsqueeze(1).expand(-1, L*K, -1, -1)
        knn_features = knn_features.gather(2, knn_idx.unsqueeze(-1).expand(-1, -1, -1, self.dim))
        
        # Calculate weights based on distance
        weights = F.softmax(-pairwise_distance.gather(2, knn_idx), dim=2)

        # Weighted sum of neighbor features
        propagated_features = torch.sum(weights.unsqueeze(-1) * knn_features, dim=2)
        propagated_features=propagated_features.reshape(B,L,K,self.dim)
        
        # Combine sampled and propagated features
        full_features = torch.zeros(B, L,K, self.dim, device=sampled_features.device)
        full_features[:,:, mask, :] = sampled_features.view(B, L,-1, self.dim)
        full_features[:, :,~mask, :] = self.feature_propagation(propagated_features[:, :,~mask, :])

        # Reshape back to (B, L, K, dim)
        full_features = full_features.view(B, K,L* self.dim)
        # shape of full_features=(B, L, K* dmodel)
        return full_features









#predictTransformat shape =B,T,9(3x3 matrix)
# predictTransformat=self.vector_field_feature_exct(vector_field)
# bs=predictTransformat.shape[0]
# Reshape predictTransformat to (B, T, 3, 3)
# predictTransformat = predictTransformat.view(bs, -1, 3, 3)
    # Apply transformation to sampled_pathline
# transformed_pathline = torch.zeros_like(sampled_pathline)
# for t in range(L):
#     # Extract points at time step t
#     points = sampled_pathline[:, t, :, :3]  # (B, K, 3)
#     time_idx=int(t_indices[t])
#     # Apply transformation
#     transformed_points = torch.bmm(points, predictTransformat[:, time_idx, :, :].transpose(1, 2))
#     transformed_pathline[:, t, :, :3] = transformed_points
#     transformed_pathline[:, t, :, 3:] = sampled_pathline[:, t, :, 3:]