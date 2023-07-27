import pytorch_lightning as pl
import torch
from dataset import CoNLL2003
from transformers import BertTokenizerFast
from typing import Optional
from torch.utils.data import DataLoader

DataName2Class = {"conll2003": CoNLL2003}

class SequenceDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root,
        dataset_name,
        tokenizer,
        *,
        train_batch_size,
        flip_percent=None,
        flip_seed=2147483647,
        max_seq_length=128,
        val_batch_size=None,
        test_batch_size=None,
        num_workers=4,
    ):
        super().__init__()
        if val_batch_size is None:
            val_batch_size = train_batch_size
        if test_batch_size is None:
            test_batch_size = val_batch_size
        self.tokenizer = tokenizer
        self.data_root = data_root
        self.save_hyperparameters(ignore=["data_root", "tokenizer"])
        self.DatasetClass = DataName2Class[self.hparams.dataset_name]

    def setup(self, stage: Optional[str] = None):
        #TODO
        pass


    def train_dataloader(self, shuffle=True):
        return DataLoader(
            self.train_set,
            batch_size=self.hparams.train_batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_set,
            batch_size=self.hparams.val_batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.hparams.test_batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.num_workers > 0,
        )
