import torch
from typing import List


from .base import BasicDistManager


class LocalManager(BasicDistManager):
    WORLD_GROUP = 0
    
    def __init__(self):
        super(LocalManager, self).__init__(world_size=1, rank=0, device=0)
    
    def finalize(self) -> None:
        pass
    
    def get_world_group(self) -> int:
        return 0
    
    def new_group(self, ranks: List[int]) -> int:
        return 0
    
    def barrier(self, group_idx_or_handler: int = WORLD_GROUP) -> None:
        pass
    
    def allreduce(self, t: torch.Tensor, group_idx_or_handler: int = WORLD_GROUP) -> None:
        pass
    
    def broadcast(self, t: torch.Tensor, rank_in_the_group: int, group_idx_or_handler: int = WORLD_GROUP) -> None:
        pass
    
    def synchronize(self) -> None:
        pass
