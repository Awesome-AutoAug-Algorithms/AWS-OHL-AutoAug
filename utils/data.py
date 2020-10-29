import os
import time
from typing import Tuple, List

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset, DataLoader, Dataset
from torch.utils.data.sampler import Sampler

from utils.misc import ints_ceil, Cutout


class _DatasetHelper(object):
    _DATASET_INFO = dict(
        imagenet={
            'img_cg': 3,
            'train_val_set_size': 1281168,
            'test_set_size': 50000,
            'img_size': 224,
            'num_classes': 1000,
            'class': None,  # todo: ImageNet
            'mean_std': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        },
        cifar10={
            'img_cg': 3,
            'train_val_set_size': 50000,
            'test_set_size': 10000,
            'img_size': 32,
            'num_classes': 10,
            'class': torchvision.datasets.CIFAR10,
            'mean_std': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        },
        cifar100={
            'img_cg': 3,
            'train_val_set_size': 50000,
            'test_set_size': 10000,
            'img_size': 32,
            'num_classes': 100,
            'class': torchvision.datasets.CIFAR100,
            'mean_std': ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        }
    )
    
    def __init__(self, name: str):
        """
        :param name: dataset name (lower case)
        """
        self.name = name
        if self.name not in _DatasetHelper._DATASET_INFO.keys():
            raise AttributeError(f'unknown dataset: {name}')
    
    def get_train_val_set_size(self):
        return _DatasetHelper._DATASET_INFO[self.name]['train_val_set_size']
    
    def get_num_classes(self):
        return _DatasetHelper._DATASET_INFO[self.name]['num_classes']
    
    def get_dataset_class(self):
        return _DatasetHelper._DATASET_INFO[self.name]['class']
    
    def get_rgb_mean_std(self):
        return _DatasetHelper._DATASET_INFO[self.name]['mean_std']


def get_train_val_set_size(dataset_name):
    return _DatasetHelper(dataset_name).get_train_val_set_size()


def get_dataset_settings(dataset_name):
    dataset_helper = _DatasetHelper(dataset_name)
    return dataset_helper.get_dataset_class(), dataset_helper.get_rgb_mean_std()


def get_num_classes(dataset_name):
    dataset_helper = _DatasetHelper(dataset_name)
    return dataset_helper.get_num_classes()


class InfiniteBatchSampler(Sampler):
    def __init__(self, dataset_len, batch_size, filling=False, shuffle=True, drop_last=False):
        self.dataset_len = dataset_len
        self.batch_size = batch_size
        self.iters_per_ep = dataset_len // batch_size if drop_last else ints_ceil(dataset_len, batch_size)
        self.max_p = self.iters_per_ep * batch_size
        self.filling = filling
        self.shuffle = shuffle
        self.epoch = 0
        self.indices = self.gener_indices()
    
    def gener_indices(self):
        indices = np.arange(self.dataset_len)
        if self.shuffle:
            np.random.shuffle(indices)
        tails = self.batch_size - (self.dataset_len % self.batch_size)
        
        if tails != self.batch_size and self.filling:
            tails = indices[:tails]
            np.random.shuffle(indices)
            indices = np.concatenate((indices, tails))
        
        # built-in list/tuple is faster than np.ndarray (when collating the data via a for-loop)
        # noinspection PyTypeChecker
        return tuple(indices.tolist())
    
    def __iter__(self):
        self.epoch = 0
        while True:
            self.epoch += 1
            p, q = 0, 0
            while p < self.max_p:
                q = p + self.batch_size
                yield self.indices[p:q]
                p = q
            if self.shuffle:
                self.indices = self.gener_indices()
    
    def __len__(self):
        return self.iters_per_ep


class DistInfiniteBatchSampler(InfiniteBatchSampler):
    def __init__(self, world_size, rank, dataset_len, tot_batch_size, filling=False, shuffle=True):
        assert tot_batch_size % world_size == 0
        self.world_size, self.rank = world_size, rank
        self.dataset_len = dataset_len
        self.tot_batch_size = tot_batch_size
        self.batch_size = tot_batch_size // world_size
        
        self.iters_per_ep = ints_ceil(dataset_len, tot_batch_size)
        self.filling = filling
        self.shuffle = shuffle
        self.epoch = 0
        self.indices = self.gener_indices()
    
    def gener_indices(self):
        global_max_p = self.iters_per_ep * self.tot_batch_size  # global_max_p % world_size == 0
        # print(f'global_max_p = iters_per_ep({self.iters_per_ep}) * tot_batch_size({self.tot_batch_size}) = {global_max_p}')
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            global_indices = torch.randperm(
                self.dataset_len, generator=g
            )
        else:
            global_indices = torch.arange(self.dataset_len)
        filling = global_max_p - global_indices.shape[0]
        if filling > 0 and self.filling:
            global_indices = torch.cat((global_indices, global_indices[:filling]))
        global_indices = tuple(global_indices.numpy().tolist())
        
        max_p = round(len(global_indices) / self.world_size)
        if self.rank == self.world_size - 1:
            local_indices = global_indices[max_p * (self.world_size - 1):]
        else:
            local_indices = global_indices[max_p * self.rank: max_p * (self.rank + 1)]
        
        self.max_p = len(local_indices)
        return local_indices


def collate_fn_for_autoaug(
        data_list: List[Tuple[
            Tuple[ torch.FloatTensor, Tuple[int, int] ], int
        ]]) -> Tuple[
            torch.FloatTensor, torch.LongTensor, Tuple[Tuple[int, int]]
        ]:

    img_ops_tuple, targets = zip(*data_list)
    images, op_indices = zip(*img_ops_tuple)
    op_indices: Tuple[Tuple[int, int]]
    
    images = torch.stack(images, dim=0)
    images: torch.FloatTensor
    targets = torch.LongTensor(targets)
    
    return images, targets, op_indices


def create_dataloaders(lg, seed_base, agent, data_cfg):
    data_cfg.type = data_cfg.type.strip().lower()
    clz, (MEAN, STD) = get_dataset_settings(data_cfg.type)
    clz: Dataset.__class__
    if 'dataset_root' not in data_cfg:
        data_cfg['dataset_root'] = os.path.abspath(os.path.join(os.path.expanduser('~'), 'datasets', data_cfg.type))
    
    # build transformers
    last_t = time.time()
    baseline_train_trans = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16),
        transforms.Normalize(MEAN, STD),
    ])
    
    to_tensor = transforms.ToTensor()
    cutout = Cutout(n_holes=1, length=16)
    normalize = transforms.Normalize(MEAN, STD)
    autoaug_train_trans = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        agent,
        lambda tup: (to_tensor(tup[0]), tup[1]),
        lambda tup: (cutout(tup[0]), tup[1]),
        lambda tup: (normalize(tup[0]), tup[1]),
    ])
    val_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    test_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    
    lg.info(f'=> after building transforms, time cost: {time.time() - last_t:.2f}s')
    
    # split data sets
    last_t = time.time()
    original_train_val_set = clz(root=data_cfg.dataset_root, train=True, download=False, transform=None)
    lg.info(f'=> after building original_train_val_set, time cost: {time.time() - last_t:.2f}s')
    
    last_t = time.time()
    targets_attr_name = 'targets' if hasattr(original_train_val_set, 'targets') else 'train_labels'
    reduced_size = data_cfg.train_set_size + data_cfg.val_set_size
    original_size = get_train_val_set_size(data_cfg.type)
    assert reduced_size <= original_size, f'too many images({reduced_size}) for the train_val_set of {data_cfg.type})'
    reduced = reduced_size < original_size
    
    def split(dataset, second_size) -> Tuple[np.ndarray, np.ndarray]:
        """
        split a given dataset into two subsets (preserving the percentage of samples for each class)
        :param dataset: the origin dataset
        :param second_size: the length of second_idx
        :return: two indices (np.ndarray) of the two subsets
                 len(second_idx) = second_size
                 len(first_idx) + len(second_idx) = len(dataset)
        """
        sss = StratifiedShuffleSplit(n_splits=1, test_size=second_size, random_state=seed_base)
        first_idx, second_idx = next(sss.split(
            X=list(range(len(dataset))),
            y=getattr(dataset, targets_attr_name)
        ))
        return first_idx, second_idx
    
    train_val_set = original_train_val_set
    if reduced:
        lg.info(f'use a reduced set ({reduced_size} of {original_size})')
        _, reduced_train_val_idx = split(original_train_val_set, reduced_size)
        reduced_train_val_set = Subset(original_train_val_set, reduced_train_val_idx)
        setattr(reduced_train_val_set, targets_attr_name, [getattr(original_train_val_set, targets_attr_name)[i] for i in reduced_train_val_idx])
        train_val_set = reduced_train_val_set
    
    train_idx, val_idx = split(train_val_set, data_cfg.val_set_size)
    lg.info(f'=> after splitting, time cost: {time.time() - last_t:.2f}s')
    
    # build datasets
    # data_cfg.dist_training
    last_t = time.time()
    auged_full_train_set = clz(
        root=data_cfg.dataset_root,
        train=True, download=False,
        transform=autoaug_train_trans
    )
    full_train_set = clz(
        root=data_cfg.dataset_root,
        train=True, download=False,
        transform=baseline_train_trans
    )
    auged_sub_train_set = Subset(
        dataset=clz(
            root=data_cfg.dataset_root,
            train=True, download=False,
            transform=autoaug_train_trans
        ), indices=np.array([train_val_set.indices[i] for i in train_idx]) if reduced else train_idx
    )
    val_set = Subset(
        dataset=clz(
            root=data_cfg.dataset_root,
            train=True, download=False,
            transform=val_trans
        ), indices=np.array([train_val_set.indices[i] for i in val_idx]) if reduced else val_idx
    )
    test_set = clz(
        root=data_cfg.dataset_root,
        train=False, download=False,
        transform=test_trans
    )
    set_sizes = len(full_train_set), len(auged_full_train_set), len(auged_sub_train_set), len(val_set), len(test_set)
    lg.info(f'=> after building sets, time cost: {time.time() - last_t:.2f}s, test_set[0][0].mean(): {test_set[0][0].mean():.4f} (expected: -0.2404)')  # -0.24041180312633514
    
    # build loaders
    from torch.utils.data._utils.collate import default_collate
    last_t = time.time()
    loaders = [DataLoader(
        dataset=dataset,
        num_workers=data_cfg.num_workers,
        pin_memory=True,
        collate_fn=cf,
    
        batch_size=bs,
        shuffle=shuffle,
        drop_last=False
    ) for dataset, cf, bs, shuffle in zip(
        (full_train_set, auged_full_train_set, auged_sub_train_set, val_set, test_set),
        (default_collate, collate_fn_for_autoaug, collate_fn_for_autoaug, default_collate, default_collate),
        (data_cfg.batch_size, data_cfg.batch_size, data_cfg.batch_size, data_cfg.batch_size*2, data_cfg.batch_size*2),
        (True, True, True, False, False)
    )]
    
    lg.info(f'=> after building loaders, time cost: {time.time() - last_t:.2f}s')
    
    return set_sizes, loaders


if __name__ == '__main__':
    d0 = DistInfiniteBatchSampler(world_size=2, rank=0, dataset_len=20, tot_batch_size=8, filling=False, shuffle=True)
    d1 = DistInfiniteBatchSampler(world_size=2, rank=1, dataset_len=20, tot_batch_size=8, filling=False, shuffle=True)
    
    # d0 = InfiniteBatchSampler(dataset_len=20, batch_size=8, filling=True, shuffle=True)
    # d1 = InfiniteBatchSampler(dataset_len=20, batch_size=8, filling=False, shuffle=True)
    
    itr0 = iter(d0)
    itr1 = iter(d1)
    for i in range(10):
        print(f'[iter{i}] d0={next(itr0)}, d1={next(itr1)}')
