from typing import Union, List

import torch

from distm import BasicDistManager


class DistLogger(object):
    def __init__(self, lg, verbose):
        self._lg, self._verbose = lg, verbose
    
    @staticmethod
    def do_nothing(*args, **kwargs):
        pass
    
    def __getattr__(self, attr: str):
        return getattr(self._lg, attr) if self._verbose else DistLogger.do_nothing


class DistModule(torch.nn.Module):
    def __init__(self, dist: BasicDistManager, model: torch.nn.Module, group_idx_or_handler=None):
        super(DistModule, self).__init__()
        self.dist = dist
        self.distributed = dist is not None and dist.world_size != 1
        self.group_idx_or_handler = group_idx_or_handler
        self.model = model
        
        if self.distributed:
            self.broadcast_params_from(src_rank=0)
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.model.state_dict(destination, prefix, keep_vars)
    
    def load_state_dict(self, state_dict, strict=False):
        return self.model.load_state_dict(state_dict, strict)
    
    def forward(self, *inputs, **kwargs):
        return self.model(*inputs, **kwargs)
    
    def sync_gradients(self, div_world_size=True):
        if self.distributed:
            for p in self.model.parameters():
                if p.grad is not None:
                    self.dist.allreduce(p.grad.data, group_idx_or_handler=self.group_idx_or_handler)
                    if div_world_size:
                        p.grad.data.div_(self.dist.world_size)
    
    def broadcast_params_from(self, src_rank):
        if self.distributed:
            for _, param in self.model.state_dict().items():
                self.dist.broadcast(param.data, src_rank, group_idx_or_handler=self.group_idx_or_handler)


def sync_vals(dist: BasicDistManager, val: float, fmt: Union[str, None] = '%.2f') -> Union[torch.Tensor, List]:
    ts = torch.zeros(dist.world_size)
    ts[dist.rank] = val
    dist.allreduce(ts)
    if fmt is None:
        return ts
    return [fmt % v for v in ts.numpy().tolist()]
