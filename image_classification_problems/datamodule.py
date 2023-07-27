from enum import Enum
from typing import List, Optional, Union
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule
from torchvision.datasets import CIFAR10
import torchvision.transforms as T

def _maybe_get_subdataset_by_class(dataset, chosen_classes, new_label_names=None):
    '''
        Inplace
    '''
    if chosen_classes is None:
        return

    isin = np.isin(dataset.targets, chosen_classes)

    dataset.data = dataset.data[isin]
    dataset.targets = np.array(dataset.targets, dtype=np.int32)
    dataset.targets = dataset.targets[isin]
    if new_label_names is None:
        new_label_names = torch.arange(len(chosen_classes))

    assert len(new_label_names) == len(chosen_classes)
    for o, n in zip(chosen_classes, new_label_names):
        dataset.targets[dataset.targets == o] = n
    dataset.targets = dataset.targets.tolist()

def _maybe_random_flip(dataset, flip_seed,  percent):
    '''
    Note: Inplace modify
    '''
    if percent is None or percent == 0:
        return None
    number_change = int(len(dataset)*percent)
    seed = torch.Generator().manual_seed(flip_seed)
    perm_train_indx = torch.randperm(len(dataset), generator=seed)
    perm_train_indx = perm_train_indx[:number_change]

    dataset.targets = np.array(dataset.targets)
    dataset.targets[perm_train_indx] = 1 - dataset.targets[perm_train_indx]
    dataset.targets = dataset.targets.tolist()
    return perm_train_indx

def _maybe_random_change_label(dataset, rand_seed, percent, num_classes):
    '''
    Note: Inplace modify
    '''
    if percent is None or percent == 0:
        return None, None
    number_change = int(len(dataset)*percent)
    seed = torch.Generator().manual_seed(rand_seed)

    flipped_idx = torch.randperm(len(dataset), generator=seed)
    flipped_idx = flipped_idx[:number_change].clone()
    if num_classes <= 2:
        label_noise = torch.ones_like(flipped_idx)
    else:
        label_noise = torch.randint(low=1, high=num_classes, 
                                    size=flipped_idx.shape, generator=seed)

    targets = torch.tensor(dataset.targets, dtype=torch.int64)
    flipped_targets = targets[flipped_idx]
    orig_targets = flipped_targets.clone()
    # In-place modification
    targets[flipped_idx] = flipped_targets.add_(label_noise).fmod_(num_classes)

    dataset.targets = targets.tolist()
    return flipped_idx, orig_targets

def _random_subset(dataset, num_examples, seed):
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:num_examples]
    # ood_data_indices = np.arange(ood_num_examples)
    sub_data = Subset(dataset, indices)
    return sub_data

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

class DMStage(Enum):
    INIT = 'init'
    FIT = 'fit'
    TRACING = 'tracing'

class Cifar10DM(LightningDataModule):
    def __init__(self, 
                 data_root, 
                 *,
                 train_batch_size, 
                 chosen_classes: Optional[List[int]] = None, 
                 flip_percent: Optional[float] =None,
                 rand_seed=42, 
                 test_batch_size=None,
                 num_workers = 0,
                 tracing_test_size = None,
                 tracing_subset_seed = 43,
                 dataset_name = 'bin_cifar10') -> None:

        super().__init__()
        if test_batch_size is None:
            test_batch_size = train_batch_size
        self.save_hyperparameters(ignore=('root',))
        self.data_root = data_root
        self.stage = DMStage.INIT
        self.num_classes = 10 if chosen_classes is None else len(chosen_classes)
    
    def prepare_data(self) -> None:
        CIFAR10(self.data_root, train=True, download=True)
        CIFAR10(self.data_root, train=False, download=True)
        # TODO: Add download dataframe

    def setup(self, stage: DMStage) -> None:
        self.stage = stage
        train_transform = T.Compose([
            T.ToTensor(),
            T.RandomHorizontalFlip(),
            T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

        test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

        train_set = CIFAR10(self.data_root, train=True, download=False, transform=train_transform)
        
        _maybe_get_subdataset_by_class(train_set, self.hparams.chosen_classes)
        # Split train set
        # seed = torch.Generator().manual_seed(42)
        # # use 20% of training data for validation
        # train_set_size = int(len(train_set) * 0.8)
        # valid_set_size = len(train_set) - train_set_size
        # self.train_set, self.valid_set = random_split(train_set, [train_set_size, valid_set_size], generator=seed)
        # self.valid_set.dataset.transform = test_transform
        self.flipped_idx, self.orig_label = _maybe_random_change_label(train_set, 
                                                                       self.hparams.rand_seed, 
                                                                       self.hparams.flip_percent,
                                                                       self.num_classes)
        test_set = CIFAR10(self.data_root, train=False, download=False, transform=test_transform)
        _maybe_get_subdataset_by_class(test_set, self.hparams.chosen_classes)
        
        self.train_set = train_set
        if stage == "fit":
            self.valid_set = test_set
            self.test_set = test_set
        elif stage == "tracing":
            if self.hparams.tracing_test_size is not None:
                test_set = _random_subset(test_set, 
                                          self.hparams.tracing_test_size, 
                                          self.hparams.tracing_subset_seed)
            self.valid_set = test_set
            self.test_set = test_set

        self.stage = stage

    def train_dataloader(self, shuffle=True):
        return DataLoader(
            self.train_set,
            batch_size=self.hparams.train_batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.valid_set,
            batch_size=self.hparams.test_batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.hparams.test_batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers
        )


if __name__ == '__main__':
    from pyrootutils import setup_root
    root = setup_root(__file__, 
                     indicator=['.git'],
                     dotenv=True,
                     pythonpath=True,
                     cwd=False)
    import os
    dm = Cifar10DM(os.environ['PYTORCH_DATASET_ROOT'], 
                    chosen_classes=[4, 9], 
                    flip_percent=0.15, 
                    train_batch_size=64,
                    num_workers=4)
    dm.prepare_data()
    dm.setup('fit')
    print(next(iter((dm.train_dataloader())))[1])
    dm.val_dataloader()
    dm.test_dataloader()
    print(dm.hparams)
    print(dm.flipped_idx)

    # dm = Cifar10DM(os.environ['PYTORCH_DATASET_ROOT'], 
    #                 chosen_classes=[4, 9, 6], 
    #                 flip_percent=0.15, 
    #                 train_batch_size=64,
    #                 num_workers=4)
    # dm.prepare_data()
    # dm.setup('fit')
    # print(next(iter((dm.train_dataloader())))[1])
    # dm.val_dataloader()
    # dm.test_dataloader()
    # print(dm.hparams)
    