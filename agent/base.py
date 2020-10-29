import time
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from copy import deepcopy
from logging import Logger
from pprint import pprint
from typing import Tuple, Callable

import torch
from torch.optim.optimizer import Optimizer

from aug_op.ops import aug_ops_dict
from distm import BasicDistManager
from scheduler import LRScheduler, scheduler_entry
from utils.config import update_op_sc_cfg
from utils.misc import time_str


def get_descretized_operations():
    ops = OrderedDict()
    for op_class_name, op_class_type in sorted(aug_ops_dict.items()):   # lexicographic ordered items
        if op_class_type.RANGES is None:
            ops[f'{op_class_name}'] = op_class_type()
            continue
        for i in [3, 6, 9]:
            mag = op_class_type.RANGES[i]
            ops[f'{op_class_name}_mag{mag:.3g}'] = op_class_type(mag)
    return ops


class BasicAgent(metaclass=ABCMeta):
    
    @staticmethod
    def _sigmoid_prob(x: torch.Tensor):
        s = x.sigmoid()
        return s / s.sum(dim=-1, keepdim=True)
    
    @staticmethod
    def _softmax_prob(x: torch.Tensor):
        return x.softmax(dim=-1)
    
    def __init__(self, dist, lg, op_cfg, sc_cfg, grad_scale, initial_baseline_ratio, baseline_mom, func_name='sigmoid'):
        
        self.dist: BasicDistManager = dist
        self.num_trajs = self.dist.world_size
        self.lg: Logger = lg  # just for the code completion (actually is `DistLogger`)
        self.grad_scale = grad_scale
        self.running_baseline, self.initial_baseline = None, None
        self.initial_baseline_ratio = initial_baseline_ratio
        self.baseline_mom = baseline_mom
        self.advance_val = 0
        
        self.prob_func: Callable = {
            'sigmoid': BasicAgent._sigmoid_prob,
            'softmax': BasicAgent._softmax_prob,
        }[func_name]
        
        ops_dict = get_descretized_operations()
        self.num_ops = len(ops_dict)
        
        # aug_names (lexicographic order): Tuple[str], aug_ops: Tuple[Callable]
        self.aug_names, self.aug_ops = zip(*[(name, ops_dict[name]) for name in sorted(ops_dict.keys())])
        self.aug_names_mat = [deepcopy(self.aug_names) for _ in range(self.num_ops)]
        
        # noinspection PyTypeChecker
        # self.op_pairs: Tuple[Tuple[Callable, Callable]] = tuple(itertools.product(self.aug_ops, repeat=2))
        
        # history of trajectory
        # op_freq[i][j] indicates the frequency of the op pair <self.aug_ops[i], self.aug_ops[j]>
        self.op_freq = torch.zeros((self.num_ops, self.num_ops)).int()
        
        self.first_prob: torch.Tensor = None
        self.second_prob: torch.Tensor = None
        self.first_param = torch.zeros(self.num_ops, requires_grad=True)
        self.second_param = torch.zeros((self.num_ops, self.num_ops), requires_grad=True)
        self._update_params_and_probs(upd_fp=None, upd_sp=None)   # only update probs

        update_op_sc_cfg(op=op_cfg, sc=sc_cfg, iters_per_epoch=1)
        self.optimizer: Optimizer = getattr(torch.optim, op_cfg.type)(
            params=[self.first_param, self.second_param],
            **op_cfg.kwargs
        )
        self.scheduler: LRScheduler = scheduler_entry(self.optimizer, sc_cfg)
        # self.debug_print()
        self.lg.info(f'broadcast initial agent\'s params...')
        tt = time.time()
        self.broad_params_from(rank=0)
        self.lg.info(f'complete, time cost: {time.time() - tt:.2f}s')
        # self.debug_print()

    @torch.no_grad()
    def _update_params_and_probs(self, upd_fp=None, upd_sp=None):
        if upd_fp is not None:
            self.first_param.copy_(upd_fp)
        if upd_sp is not None:
            self.second_param.copy_(upd_sp)
        self.first_prob = self.prob_func(self.first_param)
        self.second_prob = self.prob_func(self.second_param)
    
    def __call__(self, img):
        op1_idx = torch.multinomial(self.first_prob, 1, True).item()
        op2_idx = torch.multinomial(self.second_prob[op1_idx], 1, True).item()
        op1, op2 = self.aug_ops[op1_idx], self.aug_ops[op2_idx]
        return op2(op1(img)), (op1_idx, op2_idx)

    def random_initialize(self):
        self._update_params_and_probs(
            upd_fp=torch.zeros_like(self.first_param.data),
            upd_sp=torch.zeros_like(self.second_param.data),
        )
    
    def record(self, op_indices: Tuple[Tuple[int, int]]):
        for i, j in op_indices:
            self.op_freq[i][j] += 1

    def _clear_his(self):
        self.op_freq.zero_()
    
    def step(self, reward: float):
        advance = self._step_prologue(reward)
        ret = self._step_process(advance)
        self._step_epilogue()
        return ret
    
    def set_baselines(self, initial_baseline, running_baseline):
        self.initial_baseline = initial_baseline
        self.running_baseline = running_baseline
        self.lg.info(f'==> autoaug initial baseline (global means): {self.initial_baseline:.3f}')
    
    def _step_prologue(self, reward):
        self.running_baseline = (
                self.baseline_mom * self.running_baseline
                +
                (1-self.baseline_mom) * reward
        )
        op_freq_sum = self.op_freq.float().sum().item()
        self.advance_val = (
            self.initial_baseline_ratio * (reward - self.initial_baseline)
            +
            (1-self.initial_baseline_ratio) * (reward - self.running_baseline)
        ) * self.grad_scale
        advance = self.op_freq.float() / op_freq_sum * self.advance_val
        self._update_params_and_probs(upd_fp=None, upd_sp=None)     # only update probs
        return advance
    
    @abstractmethod
    def _step_process(self, advance: torch.Tensor):
        pass
    
    def _step_epilogue(self):
        self._clear_his()
        self._update_params_and_probs(upd_fp=None, upd_sp=None)     # only update probs

    def state_dict(self):
        return dict(
            first_param=self.first_param,
            second_param=self.second_param,
            optimizer=self.optimizer.state_dict(),
            initial_baseline=self.initial_baseline,
            running_baseline=self.running_baseline,
        )
    
    def load_state(self, agent_state: dict):
        fp, sp = agent_state['first_param'], agent_state['second_param']
        self._update_params_and_probs(upd_fp=fp, upd_sp=sp)
        if 'initial_baseline' in agent_state:
            self.initial_baseline: float = agent_state['initial_baseline']
        if 'running_baseline' in agent_state:
            self.running_baseline: float = agent_state['running_baseline']
        if 'optimizer' in agent_state:
            self.optimizer.load_state_dict(agent_state['optimizer'])
    
    def broad_params_from(self, rank):
        self.dist.broadcast(self.first_param.data, rank)
        self.dist.broadcast(self.second_param.data, rank)
        self._update_params_and_probs(upd_fp=None, upd_sp=None)     # only update probs

    def get_prob_mat(self, req_grads=False) -> torch.Tensor:
        if req_grads:
            return (
                self.prob_func(self.first_param).unsqueeze(-1)
                *
                self.prob_func(self.second_param)
            )
        return (
                self.first_prob.unsqueeze(-1)
                *
                self.second_prob
        )

    def get_prob_dict(self):
        with torch.no_grad():
            first_p, second_p = self.prob_func(self.first_param), self.prob_func(self.second_param)
            p1 = first_p / first_p.sum()
            cond_p2 = second_p / second_p.sum(dim=1, keepdim=True)
            p2 = p1.unsqueeze(dim=1) * cond_p2
            p2s0, p2s1 = p2.sum(dim=0), p2.sum(dim=1)
        return {
            op_name: (p2s0[op_idx] + p2s1[op_idx] - p2[op_idx][op_idx]).item()
            for op_idx, op_name in enumerate(self.aug_names)
        }

    def get_prob_tensor(self):
        prob_dict = self.get_prob_dict()
        li = [prob_dict[op_name] for op_name in self.aug_names]
        return torch.tensor(li)

    def get_params_as_list(self):
        with torch.no_grad():
            li1, li2 = self.first_param.detach().numpy().tolist(), self.second_param.detach().numpy().tolist()
        return li1, li2
    
    def debug_print(self, prefix):
        # for rk in range(self.dist.world_size):
        #     self.dist.barrier()
        #     if rk == self.dist.rank:
        fp, sp = self.first_param, self.second_param
        self.lg.info(
            f'[rk{self.dist.rank:02d}][{prefix}]\n'
            f' fp.grad.abs.mean, .abs.max:\n'
            f'     {fp.grad.abs().mean().item():.4g},  {fp.grad.abs().max().item():.4g}\n'
            f' sp.grad.abs.mean, .abs.max:\n'
            f'     {sp.grad.abs().mean().item():.4g},  {sp.grad.abs().max().item():.4g}\n'
            f' fp[:4].data:\n'
            f'     {fp[:4].data},\n'
            f' fp[:4].grad:\n'
            f'     {fp.grad[:4]},\n'
            f' sp[3, :4].data:\n'
            f'     {sp[3, :4].data}\n'
            f' sp[3, :4].grad:\n'
            f'     {sp.grad[3, :4]}\n'
        )


if __name__ == '__main__':
    ops_dict = get_descretized_operations()
    aug_names, aug_ops = zip(*[(name, ops_dict[name]) for name in sorted(ops_dict.keys())])
    pprint(aug_names)
    pprint(len(aug_names))
    pprint(aug_ops)
