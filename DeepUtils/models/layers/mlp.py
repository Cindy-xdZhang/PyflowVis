""" MLP module w/ dropout and configurable activation layer

Hacked together by / Copyright 2020 Ross Wightman
"""
from torch import nn as nn

from . import create_norm, create_act


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self,
                 in_features, hidden_features=None, out_features=None,
                 act_args={'act': "gelu"}, norm_args=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = create_act(act_args)
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class ConvMlp(nn.Module):
    """ MLP using 1x1 convs that keeps spatial dims
    """

    def __init__(
            self, in_features, hidden_features=None, out_features=None,
            act_args={'act': "gelu"},
            norm_args=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=True)
        self.norm = create_norm(norm_args, hidden_features) or nn.Identity()
        self.act = create_act(act_args)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x
