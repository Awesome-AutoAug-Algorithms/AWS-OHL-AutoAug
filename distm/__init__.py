from .base import BasicDistManager
from .local import LocalManager as local
from .torch import TorchDistManager as torch


def dist_entry(dist_cfg) -> BasicDistManager:
    return globals()[dist_cfg.type](**dist_cfg.kwargs if dist_cfg.kwargs is not None else {})
