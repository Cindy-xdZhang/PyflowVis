"""
Author: PointNeXt
"""
import copy
from typing import List
import torch
import torch.nn as nn
import logging
from ...utils import get_missing_parameters_message, get_unexpected_parameters_message
from ..build import MODELS, build_model_from_cfg
from ..layers import create_linearblock, create_convblock1d
from ...loss import build_criterion_from_cfg


@MODELS.register_module()
class BaseSeg(nn.Module):
    def __init__(self,
                 encoder_args=None,
                 decoder_args=None,
                 criterion_args=None,
                 cls_args=None,
                 **kwargs):
        super().__init__()
        self.encoder = build_model_from_cfg(encoder_args)
        if decoder_args is not None:
            decoder_args_merged_with_encoder = copy.deepcopy(encoder_args)
            decoder_args_merged_with_encoder.update(decoder_args)
            decoder_args_merged_with_encoder.encoder_channel_list = self.encoder.channel_list if hasattr(self.encoder,
                                                                                                         'channel_list') else None
            self.decoder = build_model_from_cfg(decoder_args_merged_with_encoder)
        else:
            self.decoder = None
        self.criterion = build_criterion_from_cfg(criterion_args) if criterion_args is not None else None
        # if cls_args is not None:
        #     if hasattr(self.decoder, 'out_channels'):
        #         in_channels = self.decoder.out_channels
        #     elif hasattr(self.encoder, 'out_channels'):
        #         in_channels = self.encoder.out_channels
        #     else:
        #         in_channels = cls_args.get('in_channels', None)
        #     cls_args.in_channels = in_channels
        #     self.head = build_model_from_cfg(cls_args)
        # else:
        #     self.head = None

    def forward(self, data):
        f = self.encoder.forward(data)
        # if self.decoder is not None:
        #     f = self.decoder(f).squeeze(-1)
        # if self.head is not None:
        #     f = self.head(f)
        return f
    
    def get_loss(self, pred, gt, inputs=None):
        return self.criterion(pred, gt)


