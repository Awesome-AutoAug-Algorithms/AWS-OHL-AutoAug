import json
import os
import time

import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer

from distm import BasicDistManager
from scheduler import LRScheduler
from utils.dist import sync_vals
from utils.misc import AverageMeter, accuracy, init_params
from .base import BasicPipeline


class AWSPipeline(BasicPipeline):
    
    def __init__(
            self, dist: BasicDistManager, common_kwargs,
            pretrained_ckpt_path, pretrain_ep, pretrain_op, pretrain_sc,
            finetuned_ckpt_path, finetune_lp, finetune_ep, rewarded_ep, finetune_op, finetune_sc,
            retrain_ep, retrain_op, retrain_sc,
            candidate_ratio=1,
            retrain_only_cutout=False,
            finetune_lsmooth=True,
            sync_mid=False,
    ):
        super(AWSPipeline, self).__init__(dist, **common_kwargs)
        if retrain_only_cutout:
            finetuned_state = None
        else:
            if finetuned_ckpt_path is None:
                if pretrained_ckpt_path is None:
                    pretrained_state = self.pretrain(pretrain_ep, pretrain_op, pretrain_sc, sync_mid, finetune_lsmooth)
                else:
                    pretrained_state = torch.load(pretrained_ckpt_path.replace('rk', f'rk{self.dist.rank}'), map_location='cpu')
                finetuned_state = self.loop_finetune(pretrained_state, finetune_lp, finetune_ep, rewarded_ep, finetune_op, finetune_sc, candidate_ratio, finetune_lsmooth)
                
            else:
                finetuned_state = torch.load(finetuned_ckpt_path, map_location='cpu')
        
        self.retrain(finetuned_state, retrain_ep, retrain_op, retrain_sc, retrain_only_cutout)
    
    def pretrain(self, max_ep, op_cfg, sc_cfg, sync_mid, lsmooth):
        init_params(model=self.model, output=self.lg.info)
        self.agent.random_initialize()
        pretrained_state = self._train_with_aug(
            max_iters=self.auged_sub_train_iters,
            loader=self.auged_sub_train_ld,
            sync_mid=sync_mid, lsmooth=lsmooth,
            max_ep=max_ep, op_cfg=op_cfg, sc_cfg=sc_cfg,
            save_mode='last', prefix='pre'
        )
        return pretrained_state
    
    def retrain(self, finetuned_state, max_ep, op_cfg, sc_cfg, retrain_only_cutout):
        init_params(model=self.model, output=self.lg.info)
        if not retrain_only_cutout:
            self.agent.load_state(finetuned_state['agent'])
            [self.g_tb_lg.add_scalars('final_probs', self.agent.get_prob_dict(), t) for t in [-10, 10]]
            self.g_tb_lg.add_histogram('final_probs_dist', self.agent.get_prob_tensor(), 0)
        
        self._train_with_aug(
            max_iters=self.full_train_iters if retrain_only_cutout else self.auged_full_train_iters,
            loader=self.full_train_ld if retrain_only_cutout else self.auged_full_train_ld,
            sync_mid=False, lsmooth=True,
            max_ep=max_ep, op_cfg=op_cfg, sc_cfg=sc_cfg,
            save_mode='best', prefix='re'
        )

    def test(self):
        return self._infer(self.test_iters, self.test_ld)
        
    def val(self):
        return self._infer(self.val_iters, self.val_ld)

    def _infer(self, max_iters, loader):
        # assert max_iters == len(loader)
        self.model.eval()
        with torch.no_grad():
            tot = 0
            sum_loss, sum_acc1, sum_acc5 = 0., 0., 0.
            for inp, tar in loader:
                inp, tar = inp.cuda(), tar.cuda()
                bs = tar.shape[0]
                logits = self.model(inp)
                loss, (acc1, acc5) = F.cross_entropy(logits.data, tar), accuracy(logits.data, tar)
                sum_loss += loss.item() * bs
                sum_acc1 += acc1 * bs
                sum_acc5 += acc5 * bs
                tot += bs

        if self.dist_training:
            pass    # todo: dist
    
        return sum_loss / tot, sum_acc1 / tot, sum_acc5 / tot

    def _train_with_aug(self, max_iters, loader, max_ep, op_cfg, sc_cfg, sync_mid, lsmooth, save_mode='best', prefix='pre'):
        # assert max_iters == len(loader)
        self.model.train()
    
        max_it = max_iters
        max_global_it = max_ep * max_it
        train_log_freq = max_it // 10
        test_freqs = [self.test_freq * 32, self.test_freq]
    
        speed = AverageMeter(max_it)
        tr_loss, tr_acc1, tr_acc5 = AverageMeter(train_log_freq), AverageMeter(train_log_freq), AverageMeter(train_log_freq)
    
        op, sc = self.create_op_sc(self.model, op_cfg, sc_cfg, iters_per_epoch=max_it)
        op: Optimizer
        sc: LRScheduler
        best_acc1 = 0
        start_train_t = time.time()
        crit = self.criterion if lsmooth else F.cross_entropy
        for ep in range(max_ep):
            ep_str = f'%{len(str(max_ep))}d' % (ep+1)
            is_late = int(ep >= 0.75 * max_ep)
            test_freq = test_freqs[is_late]
            if ep % 32 == 0:
                self.lg.info(f'==> at {self.exp_root}')
        
            last_t = time.time()
            for it, tup in enumerate(loader):
                if len(tup) == 3:
                    inp, tar, _ = tup
                else:
                    inp, tar = tup
                it_str = f'%{len(str(max_it))}d' % (it+1)
                global_it = ep * max_it + it
                data_t = time.time()
                
                if global_it == 1:
                    for i in range(self.dist.world_size):
                        if self.dist.rank == i:
                            print(f'rk[{i:2d}] dist test')
                        self.dist.barrier()
            
                inp, tar = inp.cuda(), tar.cuda()
                cuda_t = time.time()
            
                logits = self.model(inp)
                loss = crit(logits, tar)
                tr_loss.update(loss.item())
                op.zero_grad()
                loss.backward()
                if self.dist_training:
                    pass
                if self.model_grad_clip is not None:
                    total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.model_grad_clip)
                else:
                    total_norm = -233
                clipped_norm = torch.cat([p.grad.data.view(-1) for p in self.model.parameters()]).abs_().norm()
                
                sc.step()   # sc.step() before op.step()
                lr = sc.get_lr()[0]
                clipped_lr = lr * (clipped_norm / total_norm)
                
                op.step()
                acc1, acc5 = accuracy(logits, tar)
                op_t = time.time()
            
                total_r = tar.shape[0] / 128
                tr_acc1.update(val=acc1, num=total_r)
                tr_acc5.update(val=acc5, num=total_r)
            
                if global_it % test_freq == 0 or global_it == max_global_it - 1:
                    test_loss, test_acc1, test_acc5 = self.test()
                    test_t = time.time()
                    self.model.train()
                    is_best = test_acc1 >= best_acc1
                    best_acc1 = max(test_acc1, best_acc1)
                    
                    if self.dist.is_master() and it+1 == max_it:
                        remain_time, finish_time = speed.time_preds(max_global_it - global_it - 1)
                        self.lg.info(
                            f'ep[{ep_str}/{max_ep}], it[{it_str}/{max_it}]:'
                            f' tr-err1[{100-tr_acc1.last:5.2f}] ({100-tr_acc1.avg:5.2f}),'
                            f' tr-loss[{tr_loss.last:.4f}] ({tr_loss.avg:.4f}),'
                            f' te-err1[{100-test_acc1:5.2f}],'
                            f' te-loss[{test_loss:.4f}],\n'
                            f' data[{data_t-last_t:.3f}],'
                            f' cuda[{cuda_t-data_t:.3f}],'
                            f' bp[{op_t-cuda_t:.3f}],'
                            f' te[{test_t-op_t:.3f}]'
                            f' rem-t[{remain_time}] ({finish_time})'
                            f' lr[{lr:.4g}] ({clipped_lr:.4g})'
                        )
                
                    state = {
                        'model': self.model.state_dict(),
                        'op': op.state_dict(),
                        'last_iter': global_it,
                    }
                    
                    model_ckpt_path = os.path.join(self.ckpt_root, f'rk{self.dist.rank}_{prefix}_{save_mode}.pth.tar')
                    if save_mode == 'best' and is_best:
                        self.lg.info(f'==> saving best model ckpt (err{100-test_acc1:.3f}) at {os.path.abspath(model_ckpt_path)}...')
                        torch.save(state, model_ckpt_path)
                    elif save_mode == 'last':
                        torch.save(state, model_ckpt_path)
            
                speed.update(time.time() - last_t)
                last_t = time.time()
    
        if self.dist.world_size > 1:
            test_loss, test_acc1, test_acc5 = self.test()
            acc1_ts: torch.Tensor = sync_vals(self.dist, test_acc1, None)
            mid_rank = acc1_ts.argsort()[self.dist.world_size // 2].item()
            mid_ckpt_path = os.path.join(self.ckpt_root, f'midrk{mid_rank}_{prefix}_enderr{100-acc1_ts[mid_rank].item():.2f}.pth.tar')
            if self.dist.rank == mid_rank:
                torch.save({
                    'model': self.model.state_dict(),
                    'op': op.state_dict(),
                }, mid_ckpt_path)
            self.dist.barrier()
            
            if sync_mid:
                mid_ckpt = torch.load(mid_ckpt_path, map_location='cpu')
                self.model.load_state_dict(mid_ckpt['model'])
                op.load_state_dict(mid_ckpt['op'])

            best_errs: torch.Tensor = sync_vals(self.dist, 100-best_acc1, None)
            best_err: float = best_errs.mean().item()
            self.lg.info(
                f'==> {prefix}-training finished, mid rank={mid_rank},'
                f' total time cost: {(time.time()-start_train_t)/60:.2f} min,'
                f' test err @1: mean={best_err:.3f}'
            )
        else:
            best_err = 100-best_acc1
            self.lg.info(
                f'==> {prefix}-training finished,'
                f' total time cost: {(time.time()-start_train_t)/60:.2f} min,'
                f' test err @1: {100-best_acc1:.3f}'
            )

        [self.meta_tb_lg.add_scalar(f'{prefix}_best_err', best_err, t) for t in [0, max_ep]]
        [self.g_tb_lg.add_scalar(f'{prefix}_best_err', best_err, t) for t in [0, max_ep]]
        return {
            'model': self.model.state_dict(),
            'op': op.state_dict(),
            'last_iter': max_global_it
        }

    def loop_finetune(self, pretrained_state, max_lp, max_ep, rewarded_ep, op_cfg, sc_cfg, candidate_ratio, finetune_lsmooth):
        self.g_tb_lg.add_scalars('probs', self.agent.get_prob_dict(), -1)
        self.g_tb_lg.add_histogram('probs_dist', self.agent.get_prob_tensor(), -1)
        [self.g_tb_lg.add_scalar('ppo_step', self.agent.max_training_times, t) for t in [-1, max_lp, -1]]
        
        max_it = self.auged_sub_train_iters
        loader = self.auged_sub_train_ld
        # assert max_it == len(loader)
        agent_param_his = []

        best_rewards_mean = 0
        best_rewards_lp = 0
        best_agent_state = {}
        candidate_ep = max(round(max_ep * candidate_ratio), 1)
        loop_speed = AverageMeter(4)
        crit = self.criterion if finetune_lsmooth else F.cross_entropy
        for lp in range(max_lp):
            lp_str = f'%{len(str(max_lp))}d' % (lp+1)
            lp_start_t = time.time()
            self.model.load_state_dict(pretrained_state['model'])
            self.model.train()
            op, sc = self.create_op_sc(self.model, op_cfg, sc_cfg, max_it)
            op: torch.optim.optimizer.Optimizer
            op.load_state_dict(pretrained_state['op'])
        
            epoch_speed = AverageMeter(1)
            acc1s = []
            for ep in range(max_ep):
                ep_str = f'%{len(str(max_ep))}d' % (ep+1)
                ep_start_t = time.time()
                for it, (inp, tar, op_indices) in enumerate(loader):
                    global_it = ep * max_it + it
                    self.agent.record(op_indices)
                    inp, tar = inp.cuda(), tar.cuda()
                    loss = crit(self.model(inp), tar)
                    op.zero_grad()
                    loss.backward()
                    if self.model_grad_clip is not None:
                        total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.model_grad_clip)
                    else:
                        total_norm = -233
                    clipped_norm = torch.cat([p.grad.data.view(-1) for p in self.model.parameters()]).abs_().norm()

                    sc.step()   # sc.step() before op.step()
                    lr = sc.get_lr()[0]
                    clipped_lr = lr * (clipped_norm / total_norm)
                    op.step()
            
                val_loss, val_acc1, val_acc5 = self.val()
                self.model.train()
                acc1s.append(val_acc1)
                if lp % 20 == 0:
                    if ep == 0:
                        self.lg.info(f'==> at {self.exp_root}')
                    self.g_tb_lg.add_scalars('rk0_ft_v_loss', {f'loop_{lp}': val_loss}, ep)
                    self.g_tb_lg.add_scalars('rk0_ft_v_acc1', {f'loop_{lp}': val_acc1}, ep)
                    self.g_tb_lg.add_scalars('rk0_ft_v_acc5', {f'loop_{lp}': val_acc5}, ep)

                epoch_speed.avg = time.time() - ep_start_t
                remain_time, finish_time = epoch_speed.time_preds(max_ep - ep - 1)
                self.lg.info(
                    f'lp[{lp_str}/{max_lp}], ep[{ep_str}/{max_ep}]'
                    f' vacc1: {float(val_acc1):5.2f},'
                    f' verr1: {float(100.-val_acc1):5.2f},'
                    f' time cost: {time.time()-ep_start_t:.3f},'
                    f' op_freq.s: {self.agent.op_freq.sum()},'
                    f' rem-t: {remain_time} ({finish_time})'
                )
            
            acc1s = acc1s[-candidate_ep:]
            rewarded_acc1s = sorted(acc1s)[-rewarded_ep:]
            reward = sum(rewarded_acc1s) / len(rewarded_acc1s)
            rewards = sync_vals(self.dist, reward, fmt=None)
            rewards_mean = rewards.mean().item()
            if self.agent.initial_baseline is not None:
                d = f'{rewards_mean-self.agent.initial_baseline:.3f}'
            else:
                d = None
            
            if rewards_mean > best_rewards_mean:
                best_rewards_mean = rewards_mean
                best_rewards_lp = lp - 1
                best_agent_state = self.agent.state_dict()
                best_agent_state = {
                    'first_param': best_agent_state['first_param'].data.clone(),
                    'second_param': best_agent_state['second_param'].data.clone()
                }
                
                if self.dist.is_master() and d is not None:
                    for root, dirs, files in os.walk(self.best_agent_ckpt_root):
                        for f in files:
                            os.remove(os.path.join(root, f))
                    torch.save({
                        'lp': lp,
                        'agent': best_agent_state,
                    }, os.path.join(
                        self.best_agent_ckpt_root,
                        f'after_lp{best_rewards_lp}_d{d}.pth.tar')
                    )
            
            if lp == 0:
                self.agent.set_baselines(initial_baseline=rewards_mean, running_baseline=reward)
                [self.g_tb_lg.add_scalars('reward', {f'g_ini_bsln': rewards_mean}, t) for t in [0, max_lp // 2, max_lp - 1]]
                [self.l_tb_lg.add_scalars('reward', {f'rk{self.dist.rank}_ini_run_bsln': reward}, t) for t in [0, max_lp // 2, max_lp - 1]]

            ppo_step_times = self.agent.step(reward=reward)
            self.g_tb_lg.add_scalar('agent_lr', self.agent.scheduler.get_lr()[0], lp)

            loop_speed.update(time.time() - lp_start_t)
            remain_time, finish_time = loop_speed.time_preds(max_lp - lp - 1)
            self.lg.info(
                f'==> loop[{lp_str}/{max_lp}],'
                f' time cost: {(time.time()-lp_start_t) / 60:.2f} min,'
                f' rem-t[{remain_time}] ({finish_time}),'
                f' rew={rewards}'
            )
            if self.dist.is_master():
                agent_param_his.append(self.agent.get_params_as_list())
            
            self.g_tb_lg.add_scalar('ppo_step', ppo_step_times, lp)
            self.l_tb_lg.add_scalars('reward', {f'rk{self.dist.rank}_run_bsln': self.agent.running_baseline}, lp)
            self.l_tb_lg.add_scalars('advance', {f'rk{self.dist.rank}_adv': self.agent.advance_val}, lp)

            self.g_tb_lg.add_scalars('probs', self.agent.get_prob_dict(), lp)
            self.g_tb_lg.add_histogram('probs_dist', self.agent.get_prob_tensor(), lp)

            if self.dist.is_master():
                torch.save({
                    'lp': lp,
                    'agent': self.agent.state_dict(),
                }, os.path.join(self.agents_ckpt_root, f'lp{lp}_d{d}_rew_mean{rewards_mean:.2f}.pth.tar'))

            torch.cuda.empty_cache()

            if self.dist.is_master():
                f_name = os.path.join(self.ckpt_root, 'agent_param_his.json')
                self.lg.info(f'dump agent params into {f_name}')
                with open(f_name, 'w') as fp:
                    json.dump(agent_param_his, fp)
                # if lp == 0:
                #     self.lg.info(f'dumped list[0]: {agent_param_his[0]}')
            
            self.dist.barrier()
            if not os.path.exists(self.early_stop_root):
                break
            
        [self.meta_tb_lg.add_scalar('best_rew_mean', best_rewards_mean, t) for t in [0, best_rewards_lp, max_lp]]
        [self.g_tb_lg.add_scalar('best_rew_mean', best_rewards_mean, t) for t in [0, best_rewards_lp, max_lp]]
        return {'lp': best_rewards_lp, 'agent': best_agent_state}
