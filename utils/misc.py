import datetime
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F


def time_str():
    return datetime.datetime.now().strftime('[%m-%d %H:%M:%S]')


def ints_ceil(x: int, y: int) -> int:
    return (x + y - 1) // y  # or (x - 1) // y + 1


def init_params(model: torch.nn.Module, output=None):
    if output is not None:
        output('===================== param initialization =====================')
    tot_num_inited = 0
    for i, m in enumerate(model.modules()):
        clz = m.__class__.__name__
        is_conv = clz.find('Conv') != -1
        is_bn = clz.find('BatchNorm') != -1
        is_fc = clz.find('Linear') != -1
        
        cur_num_inited = []
        if is_conv:
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
            cur_num_inited.append(m.weight.numel())
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
                cur_num_inited.append(m.bias.numel())
        elif is_bn:
            if m.weight is not None:
                torch.nn.init.constant_(m.weight, 1)
                cur_num_inited.append(m.weight.numel())
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
                cur_num_inited.append(m.bias.numel())
        elif is_fc:
            # torch.nn.init.normal_(m.weight, std=0.001)
            torch.nn.init.normal_(m.weight, std=1/m.weight.size(-1))
            cur_num_inited.append(m.weight.numel())
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
                cur_num_inited.append(m.bias.numel())
        tot_num_inited += sum(cur_num_inited)
        
        if output is not None:
            builtin = any((is_conv, is_bn, is_fc))
            cur_num_inited = f' ({" + ".join([str(x) for x in cur_num_inited])})'
            output(f'clz{i:3d}: {"  => " if builtin else ""}{clz}{cur_num_inited if builtin else "*"}')
    
    if output is not None:
        output('----------------------------------------------------------------')
        output(f'tot_num_inited: {tot_num_inited} ({tot_num_inited / 1e6:.3f} M)')
        output('===================== param initialization =====================\n')
    return tot_num_inited


def clear_grads(*models):
    for m in models:
        m: torch.nn.Module
        for p in m.parameters():
            if p.grad is not None:
                p.grad.detach_().zero_()


def reduce_grads(*models):
    s = torch.tensor(0).float()
    for m in models:
        m: torch.nn.Module
        for p in m.parameters():
            if p.grad is not None:
                s.add_(p.grad.sum())
    return s


def detach_param(*models):
    for m in models:
        m: torch.nn.Module
        for p in m.parameters():
            p.detach_()


def attach_param(*models):
    for m in models:
        m: torch.nn.Module
        for p in m.parameters():
            p.requires_grad_()


def accuracy(output, target, tops=(1, 5)):
    max_k = max(tops)
    total = target.shape[0]
    
    _, preds = output.topk(max_k, 1, True, True)
    preds = preds.t()
    corrects = preds.eq(target.view(1, -1).expand_as(preds))
    
    accs = []
    for k in tops:
        corrects_k = corrects[:k].view(-1).float().sum(0, keepdim=True)
        accs.append(corrects_k.mul_(100.0 / total).item())
    return accs


class AverageMeter(object):
    def __init__(self, length=0):
        self.length = round(length)
        if self.length > 0:
            self.queuing = True
            self.val_history = []
            self.num_history = []
        self.val_sum = 0.0
        self.num_sum = 0.0
        self.last = 0.0
        self.avg = 0.0
    
    def reset(self):
        if self.length > 0:
            self.val_history.clear()
            self.num_history.clear()
        self.val_sum = 0.0
        self.num_sum = 0.0
        self.last = 0.0
        self.avg = 0.0
    
    def update(self, val, num=1):
        self.val_sum += val * num
        self.num_sum += num
        self.last = val
        if self.queuing:
            self.val_history.append(val)
            self.num_history.append(num)
            if len(self.val_history) > self.length:
                self.val_sum -= self.val_history[0] * self.num_history[0]
                self.num_sum -= self.num_history[0]
                del self.val_history[0]
                del self.num_history[0]
        self.avg = self.val_sum / self.num_sum
    
    def get_trimmed_mean(self):
        if len(self.val_history) >= 5:
            trimmed = max(int(self.length * 0.1), 1)
            return np.mean(sorted(self.val_history)[trimmed:-trimmed])
        else:
            return self.avg
    
    def time_preds(self, counts):
        remain_secs = counts * self.avg
        remain_time = datetime.timedelta(seconds=round(remain_secs))
        finish_time = time.strftime("%m-%d %H:%M:%S", time.localtime(time.time() + remain_secs))
        return remain_time, finish_time
    
    def state_dict(self):
        return vars(self)
    
    def load_state(self, state_dict):
        self.__dict__.update(state_dict)


class Cutout(object):
    def __init__(self, n_holes, length):
        """
        randomly mask out one or more patches from an image
        :param n_holes: the number of patches to cut out of each image
        :param length: the length (in pixels) of each square patch
        """
        self.n_holes = n_holes
        self.length = length
    
    def __call__(self, img):
        """
        :param img: (Tensor) tensor image of size (C, H, W)
        :return: (Tensor) image with n_holes of dimension length x length cut out of it
        """
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            
            mask[y1: y2, x1: x2] = 0.
        
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img.mul_(mask)
        
        return img


class SwishAutoFn(torch.autograd.Function):
    """ Memory Efficient Swish
    From: https://blog.ceshine.net/post/pytorch-memory-swish/
    """
    
    @staticmethod
    def forward(ctx, x):
        result = x.mul(torch.sigmoid(x))
        ctx.save_for_backward(x)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        sigmoid_x = torch.sigmoid(x)
        return grad_output * (sigmoid_x * (1 + x * (1 - sigmoid_x)))


def swish(x, inplace=False):
    return SwishAutoFn.apply(x)


def hswish(x, inplace=False):
    if inplace:
        return x.mul_(F.relu6(x.add(3.), inplace=True)) / 6.
    else:
        return x.mul(F.relu6(x.add(3.), inplace=True)) / 6.


def hsigmoid(x, inplace=False):
    if inplace:
        return F.relu6(x.add_(3.), inplace=True) / 6.
    else:
        return F.relu6(x.add(3.), inplace=True) / 6.


def get_af(af_name: str):
    af_name = af_name.strip().lower()
    af_dic = {
        'tanh': torch.tanh,
        'htanh': F.hardtanh,
        'relu': torch.relu,
        'relu6': F.relu6,
        'sigmoid': torch.sigmoid,
        'hsigmoid': hsigmoid,
        'swish': swish,
        'hswish': hswish,
    }
    if af_name not in af_dic.keys():
        raise NotImplementedError(f'unsupported activation: {af_name}')
    return af_dic[af_name]


def kld(x: torch.Tensor, y: torch.Tensor, inplace):
    """
    :param x: 1D tensor distribution (x.dim() == x.sum() == 1)
    :param y: 1D tensor distribution (y.dim() == y.sum() == 1)
    :param inplace: inplace or not
    :return: kld scalar
    """
    # return (x.mul_( (x/y).log_() )).sum() if inplace else (x * (x/y).log_()).sum()
    return F.kl_div(
        input=y.unsqueeze_(0).log_(),
        target=x.unsqueeze_(0),
        reduction='batchmean'
    ) if inplace else F.kl_div(
        input=y.unsqueeze(0).log(),  # log_() will change y (unsqueeze() do not clone)
        target=x.unsqueeze(0),
        reduction='batchmean'
    )


def ppo1_kld(old_policy_probs: torch.Tensor, new_policy_probs: torch.Tensor, inplace=False):
    """
    :param old_policy_probs: 1D tensor distribution (.dim() == .sum() == 1)
    :param new_policy_probs: 1D tensor distribution (.dim() == .sum() == 1)
    :return: kld scalar
    """
    return kld(old_policy_probs, new_policy_probs, inplace=inplace)


def filter_params(model: torch.nn.Module, special_decay_rules=None):
    if special_decay_rules is None:
        special_decay_rules = {
            'bn_b': {'weight_decay': 0.0},
            'bn_w': {'weight_decay': 0.0},
        }
    pgroup_normal = []
    pgroup = {'bn_w': [], 'bn_b': [], 'conv_b': [], 'linear_b': []}
    names = {'bn_w': [], 'bn_b': [], 'conv_b': [], 'linear_b': []}
    if 'conv_dw_w' in special_decay_rules:
        pgroup['conv_dw_w'] = []
        names['conv_dw_w'] = []
    if 'conv_dw_b' in special_decay_rules:
        pgroup['conv_dw_b'] = []
        names['conv_dw_b'] = []
    if 'conv_dense_w' in special_decay_rules:
        pgroup['conv_dense_w'] = []
        names['conv_dense_w'] = []
    if 'conv_dense_b' in special_decay_rules:
        pgroup['conv_dense_b'] = []
        names['conv_dense_b'] = []
    if 'linear_w' in special_decay_rules:
        pgroup['linear_w'] = []
        names['linear_w'] = []
    
    names_all = []
    type2num = defaultdict(lambda: 0)
    for name, m in model.named_modules():
        clz = m.__class__.__name__
        if clz.find('Conv') != -1:
            if m.bias is not None:
                if 'conv_dw_b' in pgroup and m.groups == m.in_channels:
                    pgroup['conv_dw_b'].append(m.bias)
                    names_all.append(name + '.bias')
                    names['conv_dw_b'].append(name + '.bias')
                    type2num[m.__class__.__name__ + '.bias(dw)'] += 1
                elif 'conv_dense_b' in pgroup and m.groups == 1:
                    pgroup['conv_dense_b'].append(m.bias)
                    names_all.append(name + '.bias')
                    names['conv_dense_b'].append(name + '.bias')
                    type2num[m.__class__.__name__ + '.bias(dense)'] += 1
                else:
                    pgroup['conv_b'].append(m.bias)
                    names_all.append(name + '.bias')
                    names['conv_b'].append(name + '.bias')
                    type2num[m.__class__.__name__ + '.bias'] += 1
            if 'conv_dw_w' in pgroup and m.groups == m.in_channels:
                pgroup['conv_dw_w'].append(m.weight)
                names_all.append(name + '.weight')
                names['conv_dw_w'].append(name + '.weight')
                type2num[m.__class__.__name__ + '.weight(dw)'] += 1
            elif 'conv_dense_w' in pgroup and m.groups == 1:
                pgroup['conv_dense_w'].append(m.weight)
                names_all.append(name + '.weight')
                names['conv_dense_w'].append(name + '.weight')
                type2num[m.__class__.__name__ + '.weight(dense)'] += 1
        
        elif clz.find('Linear') != -1:
            if m.bias is not None:
                pgroup['linear_b'].append(m.bias)
                names_all.append(name + '.bias')
                names['linear_b'].append(name + '.bias')
                type2num[m.__class__.__name__ + '.bias'] += 1
            if 'linear_w' in pgroup:
                pgroup['linear_w'].append(m.weight)
                names_all.append(name + '.weight')
                names['linear_w'].append(name + '.weight')
                type2num[m.__class__.__name__ + '.weight'] += 1
        
        elif clz.find('BatchNorm') != -1:
            if m.weight is not None:
                pgroup['bn_w'].append(m.weight)
                names_all.append(name + '.weight')
                names['bn_w'].append(name + '.weight')
                type2num[m.__class__.__name__ + '.weight'] += 1
            if m.bias is not None:
                pgroup['bn_b'].append(m.bias)
                names_all.append(name + '.bias')
                names['bn_b'].append(name + '.bias')
                type2num[m.__class__.__name__ + '.bias'] += 1
    
    for name, p in model.named_parameters():
        if name not in names_all:
            pgroup_normal.append(p)
    
    param_groups = [{'params': pgroup_normal}]
    for ptype in pgroup.keys():
        if ptype in special_decay_rules.keys():
            param_groups.append({'params': pgroup[ptype], **special_decay_rules[ptype]})
        else:
            param_groups.append({'params': pgroup[ptype]})
        
    return param_groups, type2num
