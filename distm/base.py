from abc import abstractmethod, ABCMeta
from typing import List

import torch


class BasicDistManager(metaclass=ABCMeta):
    
    def __init__(self, world_size: int, rank: int, device: int):
        self._world_size, self._rank, self._device = world_size, rank, device
        self.barrier()
        self.synchronize()
    
    def is_master(self):
        return self.rank == 0
    
    @property
    def world_size(self):
        return self._world_size

    @property
    def rank(self):
        return self._rank

    # @property
    # def device(self):
    #     return self._device
    
    @abstractmethod
    def finalize(self) -> None:
        pass
    
    @abstractmethod
    def get_world_group(self):
        """
        :return: handler for torch.dist
        """
        pass
    
    @abstractmethod
    def new_group(self, ranks: List[int]):
        """
        :param ranks: ranks in this group
        :return: handler for torch.dist
        """
        pass
    
    @abstractmethod
    def barrier(self, group_idx_or_handler=None) -> None:
        """
        :param group_idx_or_handler: handler for torch.dist
        """
        pass
    
    @abstractmethod
    def allreduce(self, t: torch.Tensor, group_idx_or_handler=None) -> None:
        """
        :param t: tensor
        :param group_idx_or_handler: handler for torch.dist
        """
        pass
    
    @abstractmethod
    def broadcast(self, t: torch.Tensor, rank_in_the_group: int, group_idx_or_handler=None) -> None:
        """
        :param t: tensor
        :param rank_in_the_group: src rank
        :param group_idx_or_handler: handler for torch.dist
        """
        pass
    
    @abstractmethod
    def synchronize(self) -> None:
        pass
