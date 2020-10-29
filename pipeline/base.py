import os
from logging import Logger
from pprint import pformat as pfmt
from typing import Optional, Tuple

import torch
from tensorboardX import SummaryWriter
from torch.optim.optimizer import Optimizer

from agent import agent_entry
from criterion import criterion_entry
from distm import BasicDistManager
from models import model_entry
from scheduler import scheduler_entry, LRScheduler
from utils.config import update_model_and_criterion_cfgs, update_op_sc_cfg
from utils.data import create_dataloaders
from utils.dist import DistLogger
from utils.file import create_logger
from utils.misc import filter_params


class BasicPipeline(object):
    
    def __init__(
            self, dist,
            dist_training,
            data,
            model, criterion, agent,
            job_name, exp_root, meta_tb_lg_root,
            model_grad_clip=None,
            test_freq=16,
    ):
        self.dist_training = dist_training
        self.dist: BasicDistManager = dist
        self.job_name: str = job_name
        self.exp_root: str = exp_root
        self.meta_tb_lg_root: str = meta_tb_lg_root
        
        self.ckpt_root: str = os.path.join(self.exp_root, 'ckpt')
        if self.dist.is_master() and not os.path.exists(self.ckpt_root):
            os.makedirs(self.ckpt_root)

        self.pre_ckpt_root: str = os.path.join(self.ckpt_root, 'pretrain')
        if self.dist.is_master() and not os.path.exists(self.pre_ckpt_root):
            os.makedirs(self.pre_ckpt_root)

        self.agents_ckpt_root: str = os.path.join(self.ckpt_root, 'agents')
        if self.dist.is_master() and not os.path.exists(self.agents_ckpt_root):
            os.makedirs(self.agents_ckpt_root)

        self.best_agent_ckpt_root: str = os.path.join(self.ckpt_root, 'best_agent')
        if self.dist.is_master() and not os.path.exists(self.best_agent_ckpt_root):
            os.makedirs(self.best_agent_ckpt_root)
        
        self.lg_root: str = os.path.join(self.exp_root, 'events')
        if self.dist.is_master() and not os.path.exists(self.lg_root):
            os.makedirs(self.lg_root)

        self.early_stop_root: str = os.path.join(self.exp_root, 'early_stop')
        if self.dist.is_master() and not os.path.exists(self.early_stop_root):
            os.makedirs(self.early_stop_root)
        
        self.dist.barrier()
        
        # create loggers
        lg, meta_tb_lg, g_tb_lg, l_tb_lg = self.create_loggers()
        # noinspection PyTypeChecker
        self.lg: Logger = lg  # just for the code completion (actually is `DistLogger`)
        # noinspection PyTypeChecker
        self.meta_tb_lg: SummaryWriter = meta_tb_lg  # just for the code completion (actually is `DistLogger`)
        # noinspection PyTypeChecker
        self.g_tb_lg: SummaryWriter = g_tb_lg  # just for the code completion (actually is `DistLogger`)
        # noinspection PyTypeChecker
        self.l_tb_lg: SummaryWriter = l_tb_lg  # just for the code completion (actually is `DistLogger`)
        
        # create the agent
        self.agent = agent_entry(dist=self.dist, lg=self.lg, agent_cfg=agent)
        
        # prepare data
        set_sizes, set_loaders = create_dataloaders(self.lg, 0, self.agent, data)
        self.full_train_sz, self.auged_full_train_sz, self.auged_sub_train_sz, self.val_sz, self.test_sz = set_sizes
        full_train_loader, auged_full_train_loader, auged_sub_train_loader, val_loader, test_loader = set_loaders

        self.full_train_ld, self.auged_full_train_ld, self.auged_sub_train_ld, self.val_ld, self.test_ld = (
            full_train_loader, auged_full_train_loader, auged_sub_train_loader, val_loader, test_loader
        )
        
        # self.full_train_iterator, self.auged_full_train_iterator, self.auged_sub_train_iterator, self.val_iterator, self.test_iterator = (
        #     iter(full_train_loader), iter(auged_full_train_loader), iter(auged_sub_train_loader), iter(val_loader), iter(test_loader)
        # )
        
        self.full_train_iters, self.auged_full_train_iters, self.auged_sub_train_iters, self.val_iters, self.test_iters = (
            len(full_train_loader), len(auged_full_train_loader), len(auged_sub_train_loader), len(val_loader), len(test_loader)
        )
        self.test_freq = test_freq
        
        self.lg.info(f'set sizes:'
                     f' full_train={self.full_train_sz},'
                     f' auged_full_train={self.auged_full_train_sz},'
                     f' auged_sub_train={self.auged_sub_train_sz},'
                     f' val={self.val_sz},'
                     f' test={self.test_sz}')
        
        self.lg.info(f'batch size={data.batch_size}, iters per epoch:'
                     f' full_train={self.full_train_iters},'
                     f' auged_full_train={self.auged_full_train_iters},'
                     f' auged_sub_train={self.auged_sub_train_iters},'
                     f' val={self.val_iters},'
                     f' test={self.test_iters}')
        
        # create model and criterion
        self.model_grad_clip: Optional[None, float] = model_grad_clip
        update_model_and_criterion_cfgs(model_cfg=model, criterion_cfg=criterion, dataset_name=data.type)
        self.model = model_entry(model_cfg=model)
        # init_params(self.model, output=self.lg.info)
        self.model = self.model.cuda()
        
        self.criterion = criterion_entry(criterion_cfg=criterion).cuda()
        
        final_common_kwargs = dict(
            data=data, model=model, criterion=criterion, agent=agent,
            job_name=job_name, exp_root=exp_root, meta_tb_lg_root=meta_tb_lg_root,
            model_grad_clip=model_grad_clip
        )
        self.lg.info(f'=> final common kwargs:\n{pfmt(final_common_kwargs)}')
    
    def create_loggers(self):
        logger = create_logger('G', os.path.join(self.exp_root, 'log.txt')) if self.dist.is_master() else None
        
        self.dist.barrier()
        
        meta_tensorboard_logger = SummaryWriter(self.meta_tb_lg_root) if self.dist.is_master() else None
        global_tensorboard_logger = SummaryWriter(self.lg_root) if self.dist.is_master() else None
        local_tensorboard_logger = SummaryWriter(os.path.join(self.lg_root, f'rank{self.dist.rank}'))
        
        return (
            DistLogger(logger, verbose=self.dist.is_master()),
            DistLogger(meta_tensorboard_logger, verbose=self.dist.is_master()),
            DistLogger(global_tensorboard_logger, verbose=self.dist.is_master()),
            DistLogger(local_tensorboard_logger, verbose=True)
        )
    
    @staticmethod
    def create_op_sc(model, op_cfg, sc_cfg, iters_per_epoch) -> (Optimizer, LRScheduler):
        update_op_sc_cfg(op=op_cfg, sc=sc_cfg, iters_per_epoch=iters_per_epoch)
        op = getattr(torch.optim, op_cfg.type)(params=filter_params(model=model)[0], **op_cfg.kwargs)
        sc = scheduler_entry(op, sc_cfg)
        return op, sc
    
    def finalize(self):
        # self.lg.info('[pipeline finalized]')
        self.meta_tb_lg.close()
        self.g_tb_lg.close()
        self.l_tb_lg.close()
