import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

#EasyConfig con load multiple yaml file in hierarchical manner and upate the config values
# #wip: using config load from yamle to buidl network
from DeepUtils.utils import EasyConfig
from DeepUtils.dataset import build_dataloader_from_cfg,getDatasetRootaMeta

#config with relection for buidling network
def test_easyConfig_loading():
    
    # args, opts = parser.parse_known_args()
    opts={}
    cfg = EasyConfig()
    try:
        cfg.load('config/cfgs/classification/vortexnet.yaml', recursive=True)
        # cfg.update(opts)
        # model = build_model_from_cfg(cfg.model)
        # print(model)
        # print(model.criterion)
        # # optimizer & scheduler
        # optimizer = build_optimizer_from_cfg(model, lr=cfg.lr, **cfg.optimizer)
        # print(optimizer)
        # build dataset
        rootInfo=getDatasetRootaMeta(cfg.dataset['data_dir'])
        if isinstance(cfg.datatransforms['kwargs'], dict):   
            cfg.datatransforms['kwargs'].update(rootInfo) 
        else: 
            cfg.datatransforms['kwargs']= rootInfo
        train_loader = build_dataloader_from_cfg(cfg.batch_size,
                                             cfg.dataset,
                                             cfg.dataloader,
                                             datatransforms_cfg=cfg.datatransforms,
                                             split='train'                                             
                                             )
        print(f"length of training dataset: {len(train_loader.dataset)}")
        val_loader = build_dataloader_from_cfg(cfg.batch_size,
                                             cfg.dataset,
                                             cfg.dataloader,
                                             datatransforms_cfg=cfg.datatransforms,
                                             split='val'                                             
                                             )
        print(f"length of training dataset: {len(train_loader.dataset)}")
        test_loader = build_dataloader_from_cfg(cfg.batch_size,
                                             cfg.dataset,
                                             cfg.dataloader,
                                             datatransforms_cfg=cfg.datatransforms,
                                             split='test'                                             
                                             )
        print(f"length of training dataset: {len(train_loader.dataset)}")
    except Exception as e:
        print(e)
        print("Error loading config file")
        return
        








if __name__ == '__main__':
    test_easyConfig_loading()

