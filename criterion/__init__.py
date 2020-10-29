from torch.nn import Module
from torch.nn import CrossEntropyLoss as CE

from .labelsmooth import LabelSmoothCELoss as LSCE


def criterion_entry(criterion_cfg) -> Module:
    return globals()[criterion_cfg.type](**criterion_cfg.kwargs if criterion_cfg.get('kwargs', None) is not None else {})
