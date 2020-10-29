from typing import Sequence

import yaml
from easydict import EasyDict

from utils.data import get_num_classes


def parse_raw_config(path):
    with open(path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader) if hasattr(yaml, 'FullLoader') else yaml.load(f)
    cfg = EasyDict(cfg)
    return cfg


def update_model_and_criterion_cfgs(model_cfg, criterion_cfg, dataset_name):
    num_classes = get_num_classes(dataset_name=dataset_name)
    
    if 'kwargs' not in model_cfg or model_cfg.kwargs is None:
        model_cfg.kwargs = EasyDict()
    model_cfg.kwargs['num_classes'] = num_classes
    
    if 'kwargs' in criterion_cfg:
        criterion_cfg.kwargs['num_classes'] = num_classes


def update_op_sc_cfg(op, sc, iters_per_epoch):
    # scheduler epochs => iters
    kw = sc.kwargs
    if 'iters' not in kw:   # total number of iters in the training
        assert 'epochs' in kw, 'missing epoch/iter config in sc.kwargs'
        ep = round(sum(kw.epochs) if isinstance(kw.epochs, Sequence) else kw.epochs)
        kw['iters'] = iters_per_epoch * ep
        kw.pop('epochs')
    
    if sc.type == 'Step':           # multi-stage decay
        if 'step_iters' not in kw:  # step in each of `step_iters`
            if 'step_epochs' in kw:     # step in each of `step_epochs`
                kw['step_iters'] = [round(iters_per_epoch * ep) for ep in kw.step_epochs]
                kw.pop('step_epochs')
            elif 'step_ratios' in kw:
                kw['step_iters'] = [round(kw.iters * r) for r in kw.step_ratios]
                kw.pop('step_ratios')
            else:
                raise AttributeError(f'missing step epoch-iter config in sc.kwargs (type: {sc.type})')
    elif sc.type == 'StepDecay':    # exponential decay
        if 'step_size' not in kw:  # decay every `step_iters` iters
            if 'step_epochs' in kw:     # decay every `step_epochs` epochs
                kw['step_size'] = round(iters_per_epoch * kw.step_epochs)
                kw.pop('step_epochs')
            elif 'step_times' in kw:    # decay `step_times` times
                kw['step_size'] = round(kw.iters / kw.step_times)
                kw.pop('step_times')
            else:
                raise AttributeError(f'missing step epoch-iter config in sc.kwargs (type: {sc.type})')
    
    # warmup epochs/ratio/divisor => iters
    if 'warmup_iters' not in kw:
        if 'warmup_epochs' in kw:
            kw['warmup_iters'] = round(iters_per_epoch * kw.warmup_epochs)
            kw.pop('warmup_epochs')
        elif 'warmup_ratio' in kw:
            kw['warmup_iters'] = round(kw.iters * kw.warmup_ratio)
            kw.pop('warmup_ratio')
        elif 'warmup_divisor' in kw:
            kw['warmup_iters'] = round(kw.iters / kw.warmup_divisor)
            kw.pop('warmup_divisor')
        else:
            raise AttributeError('missing warmup epoch-iter config in sc.kwargs')
    
    # base_lr_divisor => base_lr
    if 'base_lr' not in kw:
        assert 'base_lr_divisor' in kw, 'missing warmup base lr config in sc.kwargs'
        kw['base_lr'] = kw.warmup_lr / kw.base_lr_divisor
        kw.pop('base_lr_divisor')
    
    # optimizer lr
    op.kwargs['lr'] = sc.kwargs.base_lr
