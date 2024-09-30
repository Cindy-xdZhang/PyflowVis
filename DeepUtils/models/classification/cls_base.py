import torch
import torch.nn as nn
import logging
from typing import List
# from ..layers import create_linearblock
# from ...utils import get_missing_parameters_message, get_unexpected_parameters_message
# from ...utils import load_checkpoint
from ..build import MODELS, build_model_from_cfg,build_initialization_from_cfg
from ...loss import build_criterion_from_cfg


@MODELS.register_module()
class BaseCls(nn.Module):
    def __init__(self,
                 encoder_args=None,
                 cls_args=None,
                 criterion_args=None,
                 **kwargs):
        super().__init__()
        self.encoder = build_model_from_cfg(encoder_args)
        build_initialization_from_cfg(self.encoder,encoder_args)
        if isinstance(cls_args, dict) and cls_args: 
            in_channels = self.encoder.out_channels if hasattr(self.encoder, 'out_channels') else cls_args.get('in_channels', None)
            cls_args.in_channels = in_channels
            self.predictor = build_model_from_cfg(cls_args)
            build_initialization_from_cfg(self.predictor,cls_args)
        else:
            self.predictor = nn.Identity()
        self.criterion = build_criterion_from_cfg(criterion_args) if criterion_args is not None else None

    def forward(self, data) -> torch.Tensor:
        global_feat = self.encoder(data)
        # return self.predictor(global_feat)
        return global_feat

    def get_loss(self, pred, gt, inputs=None):
        return self.criterion(pred, gt)

    def get_logits_loss(self, data, gt):
        logits = self.forward(data)
        return logits, self.criterion(logits, gt)



# @MODELS.register_module()
# class ClsHead(nn.Module):
#     def __init__(self,
#                  num_classes: int,
#                  in_channels: int,
#                  mlps: List[int]=[256],
#                  norm_args: dict=None,
#                  act_args: dict={'act': 'relu'},
#                  dropout: float=0.5,
#                  global_feat: str=None,
#                  point_dim: int=2,
#                  **kwargs
#                  ):
#         """
#         A general classification head. supports global pooling and [CLS] token
#         Args:
#             num_classes (int): class num
#             in_channels (int): input channels size
#             mlps (List[int], optional): channel sizes for hidden layers. Defaults to [256].
#             norm_args (dict, optional): dict of configuration for normalization. Defaults to None.
#             act_args (_type_, optional): dict of configuration for activation. Defaults to {'act': 'relu'}.
#             dropout (float, optional): use dropout when larger than 0. Defaults to 0.5.
#             cls_feat (str, optional): preprocessing input features to obtain global feature.
#                                       $\eg$ cls_feat='max,avg' means use the concatenateion of maxpooled and avgpooled features.
#                                       Defaults to None, which means the input feautre is the global feature
#         Returns:
#             logits: (B, num_classes, N)
#         """
#         super().__init__()
#         if kwargs:
#             logging.warning(f"kwargs: {kwargs} are not used in {__class__.__name__}")
#         self.global_feat = global_feat.split(',') if global_feat is not None else None
#         self.point_dim = point_dim
#         in_channels = len(self.global_feat) * in_channels if global_feat is not None else in_channels
#         if mlps is not None:
#             mlps = [in_channels] + mlps + [num_classes]
#         else:
#             mlps = [in_channels, num_classes]

#         heads = []
#         for i in range(len(mlps) - 2):
#             heads.append(create_linearblock(mlps[i], mlps[i + 1],
#                                             norm_args=norm_args,
#                                             act_args=act_args))
#             if dropout:
#                 heads.append(nn.Dropout(dropout))
#         heads.append(create_linearblock(mlps[-2], mlps[-1], act_args=None))
#         self.head = nn.Sequential(*heads)


#     def forward(self, end_points):
#         if self.global_feat is not None:
#             global_feats = []
#             for preprocess in self.global_feat:
#                 if 'max' in preprocess:
#                     global_feats.append(torch.max(end_points, dim=self.point_dim, keepdim=False)[0])
#                 elif preprocess in ['avg', 'mean']:
#                     global_feats.append(torch.mean(end_points, dim=self.point_dim, keepdim=False))
#             end_points = torch.cat(global_feats, dim=1)
#         logits = self.head(end_points)
#         return logits
