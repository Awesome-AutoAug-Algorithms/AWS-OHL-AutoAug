import math
from bisect import bisect_right
from typing import List

import torch

from utils.misc import filter_params


class LRScheduler(object):
    def __init__(self, optimizer, last_iter=-1):
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_iter == -1:
            for group in optimizer.param_groups:
                # group e.g.: {
                #     'params': [Parameter containing: tensor([[0.1300]], requires_grad=True)],
                #     'lr': 0.1,
                #     'momentum': 0.9,
                #     'dampening': 0,
                #     'weight_decay': 0.001,
                #     'nesterov': True
                # }
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        # base_lrs e.g.: [0.1, 0.1, 0.1, 0.1, 0.1]
        self.last_iter = last_iter
    
    def reset(self):
        self.last_iter = -1
    
    def _get_new_lr(self):
        raise NotImplementedError
    
    def get_lr(self):
        return list(map(lambda group: group['lr'], self.optimizer.param_groups))
    
    def step(self, this_iter=None):
        if this_iter is None:
            this_iter = self.last_iter + 1
        self.last_iter = this_iter
        for param_group, lr in zip(self.optimizer.param_groups, self._get_new_lr()):
            param_group['lr'] = lr


class _WarmUpLRScheduler(LRScheduler):
    
    def __init__(self, optimizer, base_lr, warmup_lr, warmup_iters, last_iter=-1):
        self.base_lr = base_lr
        self.warmup_iters = warmup_iters
        if warmup_iters == 0:
            self.warmup_lr = base_lr
        else:
            self.warmup_lr = warmup_lr
        super(_WarmUpLRScheduler, self).__init__(optimizer, last_iter)
    
    def _get_warmup_lr(self):
        if self.warmup_iters > 0 and self.last_iter < self.warmup_iters:
            # first compute relative scale for self.base_lr, then multiply to base_lr
            scale = ((self.last_iter / self.warmup_iters) * (self.warmup_lr - self.base_lr) + self.base_lr) / self.base_lr
            # print('last_iter: {}, warmup_lr: {}, base_lr: {}, scale: {}'.format(self.last_iter, self.warmup_lr, self.base_lr, scale))
            return [scale * base_lr for base_lr in self.base_lrs]
        else:
            return None


class StepLRScheduler(_WarmUpLRScheduler):
    def __init__(self, optimizer, step_iters, lr_mults, base_lr, warmup_lr, warmup_iters, iters, last_iter=-1):
        super(StepLRScheduler, self).__init__(optimizer, base_lr, warmup_lr, warmup_iters, last_iter)
        
        assert len(step_iters) == len(lr_mults), "{} vs {}".format(step_iters, lr_mults)
        for x in step_iters:
            assert isinstance(x, int)
        if not list(step_iters) == sorted(step_iters):
            raise ValueError('step_iters should be a list of'
                             ' increasing integers. Got {}', step_iters)
        self.step_iters = step_iters
        self.lr_mults = [1.0]
        for x in lr_mults:
            self.lr_mults.append(self.lr_mults[-1]*x)
    
    def _get_new_lr(self):
        warmup_lr = self._get_warmup_lr()
        if warmup_lr is not None:
            return warmup_lr
        
        pos = bisect_right(self.step_iters, self.last_iter)
        scale = self.warmup_lr*self.lr_mults[pos] / self.base_lr
        return [base_lr*scale for base_lr in self.base_lrs]


class StepDecayLRScheduler(_WarmUpLRScheduler):
    def __init__(self, optimizer, step_size, decay, base_lr, warmup_lr, warmup_iters, iters, last_iter=-1):
        super(StepDecayLRScheduler, self).__init__(optimizer, base_lr, warmup_lr, warmup_iters, last_iter)
        
        self.step_size = step_size
        self.decay = decay
    
    def _get_new_lr(self):
        warmup_lr = self._get_warmup_lr()
        if warmup_lr is not None:
            return warmup_lr
        
        num = (self.last_iter - self.warmup_iters) // self.step_size
        scale = self.decay ** num * self.warmup_lr / self.base_lr
        return [base_lr*scale for base_lr in self.base_lrs]


class CosineLRScheduler(_WarmUpLRScheduler):
    def __init__(self, optimizer, iters, min_lr, base_lr, warmup_lr, warmup_iters, last_iter=-1):
        super(CosineLRScheduler, self).__init__(optimizer, base_lr, warmup_lr, warmup_iters, last_iter)
        self.iters = iters
        self.min_lr = min_lr
    
    def _get_new_lr(self):
        warmup_lr = self._get_warmup_lr()
        if warmup_lr is not None:
            return warmup_lr
        
        step_ratio = (self.last_iter-self.warmup_iters) / (self.iters-self.warmup_iters)
        target_lr = self.min_lr + (self.warmup_lr - self.min_lr)*(1 + math.cos(math.pi * step_ratio)) / 2
        scale = target_lr / self.base_lr
        return [scale*base_lr for base_lr in self.base_lrs]


class ConstantLRScheduler(_WarmUpLRScheduler):
    def __init__(self, optimizer, iters, base_lr, warmup_lr, warmup_iters, last_iter=-1):
        super(ConstantLRScheduler, self).__init__(optimizer, base_lr, warmup_lr, warmup_iters, last_iter)
        self.iters = iters
    
    def _get_new_lr(self):
        warmup_lr = self._get_warmup_lr()
        if warmup_lr is not None:
            return warmup_lr
        scale = self.warmup_lr / self.base_lr
        return [scale * base_lr for base_lr in self.base_lrs]


if __name__ == '__main__':
    lr = 0.5
    op = torch.optim.SGD(torch.nn.Linear(1, 2).parameters(), lr=lr)
    print(list(map(lambda group: group['lr'], op.param_groups)))
    N = 10
    sc = ConstantLRScheduler(op, iters=N, base_lr=lr, warmup_lr=1, warmup_iters=0, last_iter=-1)
    print("after construction lr =", sc.get_lr()[0])
    for i in range(N+1):
        sc.step()
        print("i =", i, ", after sc.step() lr =", sc.get_lr()[0])
    
    print()
    sc.reset()
    for i in range(N+1):
        sc.step(i)
        print("i =", i, ", after sc.step(i+1) lr =", sc.get_lr()[0])
