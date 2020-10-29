from .scheduler import LRScheduler
from .scheduler import StepLRScheduler as Step
from .scheduler import StepDecayLRScheduler as StepDecay
from .scheduler import CosineLRScheduler as Cosine
from .scheduler import ConstantLRScheduler as Constant


def scheduler_entry(optimizer, sc_cfg) -> LRScheduler:
    return globals()[sc_cfg.type](optimizer=optimizer, **sc_cfg.kwargs if sc_cfg.kwargs is not None else {})
