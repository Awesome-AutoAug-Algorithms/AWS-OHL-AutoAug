from copy import deepcopy

import torch

from models.resnet import Res18CIFAR as Res18
from models.wresnet import WRN28_10 as WRN


def model_entry(model_cfg) -> torch.nn.Module:
    kw = deepcopy(model_cfg.kwargs) if model_cfg.kwargs is not None else {}
    bn_mom = kw.get('bn_mom', 0.5)
    
    def BN(*args, **kwargs):
        kwargs.update({'momentum': bn_mom})
        return torch.nn.BatchNorm2d(*args, **kwargs)
    
    kw.pop('bn_mom', None)
    kw['BN2d'] = BN
    return globals()[model_cfg.type](**kw)
