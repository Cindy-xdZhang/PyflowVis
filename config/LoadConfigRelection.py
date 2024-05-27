import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

#EasyConfig con load multiple yaml file in hierarchical manner and upate the config values
#wip: using config load from yamle to buidl network
from openpoints.utils import EasyConfig
from openpoints.models import build_model_from_cfg
#config with relection for buidling network
def test_easyConfig_loading():
    
    # args, opts = parser.parse_known_args()
    opts={}
    cfg = EasyConfig()
    try:
        cfg.load('config/cfgs/default.yaml', recursive=True)
        cfg.update(opts)
        model = build_model_from_cfg(cfg.model)
        print(model)
    except Exception as e:
        print(e)
        print("Error loading config file")
        return
        





if __name__ == '__main__':
    test_easyConfig_loading()

