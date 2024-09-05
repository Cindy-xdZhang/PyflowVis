import torch
import torch.nn as nn
import torch.nn.functional as F
from ..build import MODELS

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
    
class ReferenceFrameCNN(nn.Module):

    def __init__(self,in_channels, DataSizeX,DataSizeY,TimeSteps,out_channels, hiddenSize=64, dropout=0.1, **kwargs):
        super(ReferenceFrameCNN, self).__init__()
        # the input tensor of Conv3d should be in the shape of[batch_size, chanel=2,W=16, H=16, depth(timsteps)]
        self.conv1_1 = nn.Conv3d(in_channels=in_channels, out_channels=hiddenSize, kernel_size=3, stride=2,padding=1)
        self.bn1_1 = nn.BatchNorm3d(hiddenSize)
        
        # self.CNNlayerCountN=4-2 #n=log2(max(H,W))-2
        self.conv2_1 = nn.Conv3d(in_channels=hiddenSize, out_channels=hiddenSize*2, kernel_size=3, stride=2,padding=1)
        self.bn2_1 = nn.BatchNorm3d(hiddenSize*2)
        self.dropout_cnv1 = nn.Dropout(dropout)
        self.dropout_cnv2 = nn.Dropout(dropout)


        self.flatten = nn.Flatten()
        DataSizeX = DataSizeX // 4
        DataSizeY = DataSizeY // 4
        TimeSteps =( (TimeSteps+1)//2 +1)//2
        # Fully connected layer
        self.outputDim=out_channels
        self.fc1 = nn.Linear(hiddenSize*2 * DataSizeX * DataSizeY*TimeSteps , 1024)
        self.bn_fc_1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, out_channels)
        self.bn_fc_2 = nn.BatchNorm1d(out_channels)
        self.dropout_fc = nn.Dropout(dropout)
        

    def forward(self, x):
        x =self.dropout_cnv1  (  F.relu(self.bn1_1(self.conv1_1(x))))
        x = self.dropout_cnv2  ( F.relu(self.bn2_1(self.conv2_1(x))))
        # x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = self.flatten(x)
        x = self.dropout_fc(F.relu(self.bn_fc_1(self.fc1(x))))
        x = F.relu(self.bn_fc_2(self.fc2(x)))
        return x
    
    
@MODELS.register_module()
class PathlineTransformerWithCNNVecField(nn.Module):
    def __init__(self, Kpathlines,in_channels, DataSizeX,DataSizeY,TimeSteps,nhead=8, num_encoder_layers=4,dmodel=512,dim_feedforward=1024, dropout=0.1,**kwargs):
        super(PathlineTransformerWithCNNVecField, self).__init__()
        self.DataSizeY=DataSizeY
        self.DataSizeX=DataSizeX
        self.keep_K = int(0.8 * Kpathlines)
        if self.keep_K % 8 != 0:
            self.keep_K = (self.keep_K // 8) * 8
            
        C=3
        self.dmodel=dmodel
        #the pathlin first feed into embedding layer then plus position embedding(time)
        self.emb=nn.Sequential(
            nn.Linear(self.keep_K*3, self.dmodel),
            nn.ReLU()
        )
      
        encoder_layer = TransformerEncoderLayer(self.dmodel, nhead, dim_feedforward, dropout)
        self.encoder_pos = TransformerEncoder(encoder_layer, num_encoder_layers)
        # self.symetryFn=nn.MaxPool1d(kernel_size=3,stride=1,padding=1)
        decoder_layer = TransformerDecoderLayer( self.dmodel, nhead, dim_feedforward, dropout)
        self.decoder = TransformerDecoder(decoder_layer, num_encoder_layers)

        self.vector_field_feature_exct=ReferenceFrameCNN(in_channels,DataSizeX,DataSizeY,TimeSteps,self.dmodel,dropout=dropout)
        #final ouput layer
        #sequence length fix to 21.
        self.ln1= nn.Linear(self.dmodel, DataSizeX*DataSizeY*2)


    def forward(self, data):
        vector_field,pathline_src=data
        #vector field data, shape of fature_v is [B,d_model]
        feature_v=self.vector_field_feature_exct(vector_field)
        feature_v=feature_v.unsqueeze(0)
        
        B, L, K, C = pathline_src.shape
        # Randomly  select first `keep_K` indices indices to keep
        indices = torch.randperm(K)[:self.keep_K]  
        #pathline_positions shape : [B,L,k*2]->[B,L,dmodel]
        pathline_positions= pathline_src[:, :, indices, :] .reshape(B,L,-1).permute(1,0,2)
        position_embed=self.emb(pathline_positions)
        # position_embed shape [L,B,dmodel],pathline_time shape [L,B,1]
        # pathline_time=pathline_src[:, :, indices, 2].reshape(B,L,self.keep_K).permute(1,0,2)
        # position_embed+=pathline_time
 
        # Transpose input [B,L,D] to fit the transformer input format [Seq_len, Batch_size, d_model]
        memory = self.encoder_pos(position_embed)

        #vector field shape is [1,B,d_model]->output shape [1,B,d_model]
        output = self.decoder(feature_v, memory)
        #output shape [B,d_model]
        output=output.squeeze(0)
        #cls
        x=self.ln1(output).reshape(B,self.DataSizeY,self.DataSizeX,2)
        x=F.softmax(x,dim=-1)         
        return x









# @MODELS.register_module()
# class PathlineVecFieldTransformer(nn.Module):
#     def __init__(self, Kpathlines,seqLength,in_channels, DataSizeX,DataSizeY,TimeSteps,nhead=8, num_encoder_layers=4, dim_feedforward=1024, dropout=0.1,**kwargs):
#         super(PathlineVecFieldTransformer, self).__init__()
#         self.DataSizeY=DataSizeY
#         self.DataSizeX=DataSizeX
#         self.keep_K = int(0.8 * Kpathlines)
#         if self.keep_K % 8 != 0:
#             self.keep_K = (self.keep_K // 8) * 8
#         C=3
#         self.dmodel=self.keep_K*C
#         self.emb=nn.Sequential(
#             nn.Linear(self.dmodel, self.dmodel),
#             nn.ReLU()
#         )
      
#         encoder_layer = TransformerEncoderLayer(self.dmodel, nhead, dim_feedforward, dropout)
#         self.encoder_pos = TransformerEncoder(encoder_layer, num_encoder_layers)
#         # self.symetryFn=nn.MaxPool1d(kernel_size=3,stride=1,padding=1)
#         decoder_layer = TransformerDecoderLayer( self.dmodel, nhead, dim_feedforward, dropout)
#         self.decoder = TransformerDecoder(decoder_layer, num_encoder_layers)

#         self.vector_field_feature_exct=ReferenceFrameCNN(in_channels,DataSizeX,DataSizeY,TimeSteps,self.dmodel,dropout=dropout)
#         #final ouput layer
#         #sequence length fix to 21.
#         self.ln1= nn.Linear(self.dmodel, DataSizeX*DataSizeY*2)


#     def forward(self, data):
#         vector_field,pathline_src=data
#         #vector field data, shape of fature_v is [B,d_model]
#         feature_v=self.vector_field_feature_exct(vector_field)
#         feature_v=feature_v.unsqueeze(0)
        
#         B, L, K, C = pathline_src.shape
#         # Randomly  select first `keep_K` indices indices to keep
#         indices = torch.randperm(K)[:self.keep_K]  
#         #pathline_src shape : [B,L,self.dmodel]
#         pathline_src = pathline_src[:, :, indices, :].reshape(B,L,self.dmodel)
#         # Transpose input [B,L,D] to fit the transformer input format [Seq_len, Batch_size, d_model]
#         pathline_src=pathline_src.permute(1,0,2)
#         memory = self.encoder_pos(pathline_src)

#         #vector field shape is [1,B,d_model]->output shape [1,B,d_model]
#         output = self.decoder(feature_v, memory)
#         #output shape [B,d_model]
#         output=output.squeeze(0)
#         #cls
#         x=self.ln1(output).reshape(B,self.DataSizeY,self.DataSizeX,2)
#         x=F.softmax(x,dim=-1)         
#         return x


# @MODELS.register_module()
# class SimpleTransformer(nn.Module):
#     def __init__(self, Kpathlines,seqLength,in_channels, DataSizeX,DataSizeY,TimeSteps,nhead=8, num_encoder_layers=4, dim_feedforward=2048, dropout=0.1,**kwargs):
#         super(SimpleTransformer, self).__init__()
#         self.kpathlines=Kpathlines
#         self.DataSizeX=DataSizeX
#         self.DataSizeY=DataSizeY
#         self.keep_K = int(0.8 * Kpathlines)
#         if self.keep_K % 8 != 0:
#             self.keep_K = (self.keep_K // 8) * 8
#         C=5
#         self.dmodel=self.keep_K*C
#         self.dmodel_pos=self.keep_K*3
#         self.dmodel_info=self.keep_K*2

#         encoder_layer = TransformerEncoderLayer(self.dmodel_pos, nhead, dim_feedforward, dropout)
#         encoder_layer2 = TransformerEncoderLayer(self.dmodel_info, nhead, dim_feedforward, dropout)
#         self.encoder_pos = TransformerEncoder(encoder_layer, num_encoder_layers)
#         self.encoder_info = TransformerEncoder(encoder_layer2, num_encoder_layers)
#         decoder_layer = TransformerDecoderLayer( self.dmodel, nhead, dim_feedforward, dropout)
#         self.decoder = TransformerDecoder(decoder_layer, num_encoder_layers)

#         self.vector_field_feature_exct=ReferenceFrameCNN(in_channels,DataSizeX,DataSizeY,TimeSteps,self.dmodel,dropout=dropout)
#         #final ouput layer
#         #sequence length fix to 21.
#         self.ln1= nn.Linear(self.dmodel, self.dmodel)
#         self.cls= nn.Linear(self.dmodel, DataSizeX*DataSizeY*2)


#     def forward(self, data):
#         vector_field,pathline_src=data
#         #vector field data, shape of fature_v is [B,d_model]
#         feature_v=self.vector_field_feature_exct(vector_field)
#         feature_v=feature_v.unsqueeze(0)
        
#         B, L, K, C = pathline_src.shape
#         # Randomly select indices to keep
#         indices = torch.randperm(K)[:self.keep_K]  # Generate random indices and select the first `keep_K` indices
#         pathline_src = pathline_src[:, :, indices, :]#B,L,K,C(posx,posy,ivd,distance)
#         pathline_positions=pathline_src[:, :, :, 0:3].reshape(B,L,self.keep_K*3).permute(1, 0, 2)
#         pathline_infors=pathline_src[:, :, :, 3:5].reshape(B,L,self.keep_K*2).permute(1, 0, 2)
#         # Transpose input [B,L,D] to fit the transformer input format [Seq_len, Batch_size, d_model_pos]
#         memory_pos = self.encoder_pos(pathline_positions)
#         memory_info = self.encoder_info(pathline_infors)
#         #memory shape is [L,B,d_model]
#         memory=torch.cat([memory_pos,memory_info],dim=-1)
        
        
#         #vector field shape is [1,B,d_model]->output shape [1,B,d_model]
#         output = self.decoder(feature_v, memory)
#         #output shape [B,d_model]
#         output=output.squeeze(0)
#         #cls
#         x=F.relu(self.ln1(output))
#         x=F.relu(self.cls( x))
#         x=x.resize(B,self.DataSizeY,self.DataSizeX,2)
#         x=F.softmax(x,dim=-1) 
#         return x
    
