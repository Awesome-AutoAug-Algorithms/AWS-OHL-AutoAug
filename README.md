# Automatic Augmentation Zoo

## Introduction
This repository provides the official implementations of [OHL](http://openaccess.thecvf.com/content_ICCV_2019/papers/Lin_Online_Hyper-Parameter_Learning_for_Auto-Augmentation_Strategy_ICCV_2019_paper.pdf) and [AWS](https://arxiv.org/abs/2009.14737), and will also integrate some other popular auto-aug methods (like [Auto Augment](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cubuk_AutoAugment_Learning_Augmentation_Strategies_From_Data_CVPR_2019_paper.pdf), [Fast AutoAugment](http://papers.nips.cc/paper/8892-fast-autoaugment.pdf) and [Adversarial autoaugment](https://arxiv.org/pdf/1912.11188)) in pure PyTorch.
We use `torch.distributed` to conduct the distributed training. The model checkpoints will be upload to GoogleDrive or OneDrive soon.


<!-- ## Our Trained Model / Checkpoint -->

<!-- + OneDrive: [Link](https://1drv.ms/u/s!Am_mmG2-KsrnajesvSdfsq_cN48?e=aHVppN) -->


-------

## Dependencies

It would be recommended to conduct experiments under:

- python 3.6.3
- pytorch 1.1.0, torchvision 0.2.1


All the dependencies are listed in `requirements.txt`. You may use commands like `pip install -r requirements.txt` to install them.

-------

### Running

1. Create the directory for your experiment.
```shell
cd /path/to/this/repo
mkdir -p exp/aws_search1
```

2. Copy configurations into your workspace.
```shell
cp scripts/search.sh configs/aws.yaml exp/aws_search1
cd exp/aws_search1
```

3. Start searching
```shell
# sh ./search.sh <partition_name> <num_gpu>
sh ./search.sh Test 8
```

An instance of yaml:

```yaml
version: 0.1.0

dist:
    type: torch
    kwargs:
        node0_addr: auto
        node0_port: auto
        mp_start_method: fork   # fork or spawn; spawn would be too slow for Dalaloader

pipeline:
    type: aws
    common_kwargs:
        dist_training: &dist_training False
#        job_name:         [will be assigned in runtime]
#        exp_root:         [will be assigned in runtime]
#        meta_tb_lg_root:  [will be assigned in runtime]

        data:
            type: cifar100               # case-insensitive (will be converted to lower case in runtime)
#            dataset_root: /path/to/dataset/root   # default: ~/datasets/[type]
            train_set_size: 40000
            val_set_size: 10000
            batch_size: 256
            dist_training: *dist_training
            num_workers: 3
            cutout: True
            cutlen: 16

        model_grad_clip: 3.0
        model:
            type: WRN
            kwargs:
#                num_classes: [will be assigned in runtime]
                bn_mom: 0.5

        agent:
            type: ppo           # ppo or REINFORCE
            kwargs:
                initial_baseline_ratio: 0
                baseline_mom: 0.9
                clip_epsilon: 0.2
                max_training_times: 5
                early_stopping_kl: 0.002
                entropy_bonus: 0
                op_cfg:
                    type: Adam         # any type in torch.optim
                    kwargs:
#                        lr: [will be assigned in runtime] (=sc.kwargs.base_lr)
                        betas: !!python/tuple [0.5, 0.999]
                        weight_decay: 0
                sc_cfg:
                    type: Constant
                    kwargs:
                        base_lr_divisor: 8      # base_lr = warmup_lr / base_lr_divisor
                        warmup_lr: 0.1          # lr at the end of warming up
                        warmup_iters: 10      # warmup_epochs = epochs / warmup_divisor
                        iters: &finetune_lp 350
        
        criterion:
            type: LSCE
            kwargs:
                smooth_ratio: 0.05


    special_kwargs:
        pretrained_ckpt_path: ~ # /path/to/pretrained_ckpt.pth.tar
        pretrain_ep: &pretrain_ep 200
        pretrain_op: &sgd
            type: SGD       # any type in torch.optim
            kwargs:
#                lr: [will be assigned in runtime] (=sc.kwargs.base_lr)
                nesterov: True
                momentum: 0.9
                weight_decay: 0.0001
        pretrain_sc:
            type: Cosine
            kwargs:
                base_lr_divisor: 4      # base_lr = warmup_lr / base_lr_divisor
                warmup_lr: 0.2          # lr at the end of warming up
                warmup_divisor: 200     # warmup_epochs = epochs / warmup_divisor
                epochs: *pretrain_ep
                min_lr: &finetune_lr 0.001

        finetuned_ckpt_path: ~  # /path/to/finetuned_ckpt.pth.tar
        finetune_lp: *finetune_lp
        finetune_ep: &finetune_ep 10
        rewarded_ep: 2
        finetune_op: *sgd
        finetune_sc:
            type: Constant
            kwargs:
                base_lr: *finetune_lr
                warmup_lr: *finetune_lr
                warmup_iters: 0
                epochs: *finetune_ep

        retrain_ep: &retrain_ep 300
        retrain_op: *sgd
        retrain_sc:
            type: Cosine
            kwargs:
                base_lr_divisor: 4      # base_lr = warmup_lr / base_lr_divisor
                warmup_lr: 0.4          # lr at the end of warming up
                warmup_divisor: 200     # warmup_epochs = epochs / warmup_divisor
                epochs: *retrain_ep
                min_lr: 0

```

-------

## Citation

If you're going to to use this code in your research, please cite our papers ([OHL](https://arxiv.org/abs/1905.07373) and [AWS](https://arxiv.org/abs/2009.14737)).

```
@inproceedings{lin2019online,
  title={Online Hyper-parameter Learning for Auto-Augmentation Strategy},
  author={Lin, Chen and Guo, Minghao and Li, Chuming and Yuan, Xin and Wu, Wei and Yan, Junjie and Lin, Dahua and Ouyang, Wanli},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={6579--6588},
  year={2019}
}

@article{tian2020improving,
  title={Improving Auto-Augment via Augmentation-Wise Weight Sharing},
  author={Tian, Keyu and Lin, Chen and Sun, Ming and Zhou, Luping and Yan, Junjie and Ouyang, Wanli},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```

## Contact for Issues
- Keyu Tian, [tiankeyu@sensetime.com](tiankeyu@sensetime.com)
- Chen Lin, [chen.lin@eng.ox.ac.uk](chen.lin@eng.ox.ac.uk)


## References & Opensources

- **ResNet** : [paper1](https://arxiv.org/abs/1512.03385), [paper2](https://arxiv.org/abs/1603.05027), [code](https://github.com/osmr/imgclsmob/tree/master/pytorch/pytorchcv/models)
- **Wide-ResNet** : [paper](https://arxiv.org/pdf/1605.07146), [code](https://github.com/meliketoy/wide-resnet.pytorch)
- **Shake-Shake** : [paper](https://arxiv.org/pdf/1705.07485), [code](https://github.com/owruby/shake-shake_pytorch)
- **PyramidNet** : [paper](https://arxiv.org/abs/1610.02915), [code](https://github.com/dyhan0920/PyramidNet-PyTorch)
- **ShakeDrop Regularization** : [paper](https://arxiv.org/abs/1802.02375), [code](https://github.com/owruby/shake-drop_pytorch)
- **Cutout** : [paper](https://arxiv.org/pdf/1708.04552.pdf), [code](https://github.com/uoguelph-mlrg/Cutout)
- **Auto Augment** : [paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cubuk_AutoAugment_Learning_Augmentation_Strategies_From_Data_CVPR_2019_paper.pdf), [code](https://github.com/tensorflow/models/tree/master/research/autoaugment)
- **Fast AutoAugment** : [paper](https://arxiv.org/abs/1905.00397), [code](https://github.com/kakaobrain/fast-autoaugment)
- **Adversarial autoaugment** : [paper](https://arxiv.org/pdf/1912.11188)

