from ..utils import registry
import torch.nn as nn
import torch.nn.init as init

MODELS = registry.Registry('models')
def kaiming_init(m, mode='fan_in', nonlinearity='relu'):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        init.kaiming_normal_(m.weight, mode=mode, nonlinearity=nonlinearity)
        if m.bias is not None:
            init.constant_(m.bias, 0)

def xavier_init(m, gain=1.0):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        init.xavier_normal_(m.weight, gain=gain)
        if m.bias is not None:
            init.constant_(m.bias, 0)

def gaussian_init(m, mean=0.0, std=0.02):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        init.normal_(m.weight, mean=mean, std=std)
        if m.bias is not None:
            init.constant_(m.bias, 0)

def orthogonal_init(m, gain=1.0):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        init.orthogonal_(m.weight, gain=gain)
        if m.bias is not None:
            init.constant_(m.bias, 0)

def build_model_from_cfg(cfg, **kwargs):
    """
    Build a model, defined by `NAME`.
    Args:
        cfg (eDICT): 
    Returns:
        Model: a constructed model specified by NAME.
    """
    return MODELS.build(cfg, **kwargs)


def build_initialization_from_cfg(model,init_cfg, **kwargs):
    init_method = init_cfg.get('method', 'xavier')  # Default to 'kaiming'
    init_args = init_cfg.get('kwargs', {})
    if init_method == 'kaiming':
        model.apply(lambda m: kaiming_init(m, **init_args))
    elif init_method == 'xavier':
        model.apply(lambda m: xavier_init(m, **init_args))
    elif init_method == 'gaussian':
        model.apply(lambda m: gaussian_init(m, **init_args))
    elif init_method == 'orthogonal':
        model.apply(lambda m: orthogonal_init(m, **init_args))
    else:
        raise ValueError(f"Unsupported initialization method: {init_method}")
 