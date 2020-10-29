import torch

from .base import BasicAgent


class PPOAgent(BasicAgent):
    
    def __init__(
            self, dist, lg, op_cfg, sc_cfg, grad_scale=0.5,
            initial_baseline_ratio=0.3,
            baseline_mom=0.9,
            func_name='sigmoid',
            clip_epsilon=0.1,
            max_training_times=50,
            early_stopping_kl=0.01,
            entropy_bonus=0,
    ):
        grad_scale = float(grad_scale)
        super(PPOAgent, self).__init__(dist, lg, op_cfg, sc_cfg, grad_scale, initial_baseline_ratio, baseline_mom, func_name)
        
        self.clip_epsilon = clip_epsilon
        
        self.max_training_times = max_training_times
        self.early_stopping_kl = early_stopping_kl
        
        self.entropy_bonus = entropy_bonus
        self.with_entropy_bonus = entropy_bonus > 1e-6
        
        # kld_target: 0.01
        # initial kld_beta: 1
        # – If kld < kld_target / 1.5, kld_beta /= 2
        # – If kld > kld_target * 1.5, kld_beta *= 2
    
    def _step_process(self, advance: torch.Tensor):
        old_probs = self.get_prob_mat(req_grads=False)
        old_probs_log = old_probs.log()
        training_times = 0
        self.scheduler.step()
        for i in range(self.max_training_times):
            new_probs_log_req_grads = self.get_prob_mat(req_grads=True).log()
            ratio = (old_probs_log - new_probs_log_req_grads).exp()
            clipped_ratio = ratio.clamp(min=1 - self.clip_epsilon, max=1 + self.clip_epsilon)
            
            loss = torch.min(ratio * advance, clipped_ratio * advance).sum()
            if self.with_entropy_bonus:
                loss += self.entropy_bonus * (- torch.sum(new_probs_log_req_grads * new_probs_log_req_grads.log()))
            
            loss = -loss  # for gradient ASCENT
            self.optimizer.zero_grad()
            loss.backward()

            if i == 0:
                self.debug_print('step 0')
            
            for p in [self.first_param, self.second_param]:
                self.dist.allreduce(p.grad.data)
                p.grad.data.div_(self.num_trajs)
            self.optimizer.step()
            for p in [self.first_param, self.second_param]:
                self.dist.broadcast(p.data, rank_in_the_group=0)
            
            self._update_params_and_probs(upd_fp=None, upd_sp=None)  # only update probs
            
            training_times = i + 1
            approx_kl = (old_probs * (old_probs_log - new_probs_log_req_grads.detach())).sum()
            if approx_kl > self.early_stopping_kl:
                break

        self.debug_print('stepped')
            
        self.lg.info(f'ppo step process: updated {training_times}/{self.max_training_times} times')
        return training_times
