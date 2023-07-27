import pytorch_lightning as pl
import pandas as pd
from typing import Optional
from torch.utils.data import DataLoader, Subset
from dataset import SNLIDataset, IMDBDataset
import torch
import numpy as np
from transformers import AutoTokenizer


def _random_flip(data, percent, flip_seed, num_classes):
    # if percent is None or percent == 0:
    #     return
    num_flipped = int(len(data) * percent)
    # TODO: Split generator for indx and label noise?
    generator = torch.Generator().manual_seed(flip_seed)

    flipped_indx = torch.randperm(len(data), generator=generator)
    flipped_indx = flipped_indx[:num_flipped]

    orig_labels = data["label"][flipped_indx.numpy()].values
    if num_classes <= 2:
        label_noise = torch.ones_like(flipped_indx)
    else:
        label_noise = torch.randint(
            low=1, high=num_classes, size=flipped_indx.size(), generator=generator
        )
    flipped_targets = torch.tensor(orig_labels, dtype=torch.int64)
    flipped_targets = flipped_targets.add_(label_noise).fmod_(num_classes)

    flipped_targets = flipped_targets.numpy()
    flipped_indx = flipped_indx.numpy()
    data["isFlipped"] = 0
    data["originLabel"] = data.label
    data.loc[flipped_indx, "label"] = flipped_targets
    data.loc[flipped_indx, "isFlipped"] = 1

    # return flipped_indx, orig_labels


def _random_subset(dataset, num_examples_per_class, seed):
    # generator = torch.Generator().manual_seed(seed)
    df: pd.DataFrame = dataset.df
    new_df = pd.DataFrame()
    for c in range(dataset.num_classes):
        c_frac = df[df['label'] == c].sample(num_examples_per_class, random_state=seed)
        new_df = pd.concat((new_df, c_frac))
        seed += 997
    dataset.df = new_df
    # indices = torch.randperm(len(dataset), generator=generator)[:num_examples]
    # ood_data_indices = np.arange(ood_num_examples)
    # sub_data = Subset(dataset, indices.tolist())
    return dataset


DataName2Class = {"snli": SNLIDataset, "imdb": IMDBDataset}


class TextClassifierDataModule(pl.LightningDataModule):
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
        use_denoised_data=False
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

    @property
    def num_classes(self):
        return self.DatasetClass.num_classes

    def setup(self, stage: Optional[str] = None):
        data_root = self.data_root
        max_seq_length = self.hparams.max_seq_length
        flip_percent = self.hparams.flip_percent
        flip_seed = self.hparams.flip_seed
        DatasetClass = self.DatasetClass
        use_denoised_data = self.hparams.use_denoised_data
        dataset_gen = lambda split: DatasetClass(
            root=data_root,
            split=split,
            tokenizer=self.tokenizer,
            max_len=max_seq_length,
        )

        def get_train_set():
            train_set = dataset_gen("train")
            if flip_percent is not None:
                # Flip label of training data
                _random_flip(train_set.df, flip_percent, flip_seed, self.num_classes)
            return train_set

        if stage == "fit":
            self.train_set = get_train_set()
            self.valid_set = dataset_gen("val")
            self.test_set = dataset_gen("test")

        if stage == "tracing":
            self.train_set = get_train_set() # For KNN refference
            if "denoised_train" in self.DatasetClass.csv_name and use_denoised_data:
                self.trace_set = dataset_gen("denoised_train")
            else:
                self.trace_set = self.train_set
            self.valid_set = None
            self.ref_set = _random_subset(get_train_set(), 100, 43)

    def trace_dataloader(self):
        return DataLoader(
            self.trace_set,
            batch_size=self.hparams.train_batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            persistent_workers=True,
        )
    
    def ref_dataloader(self):
        return DataLoader(
            self.ref_set,
            batch_size=self.hparams.train_batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            persistent_workers=True,
        )

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

    @property
    def flipped_inds(self):
        where = (self.trace_set.df["isFlipped"] == 1).values.squeeze()
        inds = np.argwhere(where).squeeze()
        return torch.tensor(inds)

    def flip_stats(self):
        return (
            self.train_set.df.groupby(["originLabel", "label"], as_index=False)
            .size()
            .rename(columns={0: "count"})
        )


if __name__ == "__main__":
    from pyrootutils import setup_root

    root = setup_root(
        __file__, indicator=[".git"], dotenv=True, pythonpath=True, cwd=False
    )
    import os

    DATA_ROOT = os.environ["PYTORCH_DATASET_ROOT"]

    print("Load IMDb dataset")
    dm = TextClassifierDataModule(
        data_root=DATA_ROOT,
        dataset_name="imdb",
        tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased"),
        train_batch_size=4,
        flip_percent=0.0,
        num_workers=0,
    )
    print("Num classes:", dm.num_classes)
    dm.prepare_data()
    dm.setup("fit")
    # print(next(iter(dm.train_dataloader())))
    dm.val_dataloader()
    dm.test_dataloader()
    # print(dm.train_set.df.head(10))
    print(dm.train_set.df[dm.train_set.df["isFlipped"] == 1].head(10))
    print(dm.flip_stats())

    dm.setup("tracing")
    print(dm.ref_set.df)
    print(dm.ref_set.df.info())


    print("Load SNLI dataset")
    dm = TextClassifierDataModule(
        data_root=DATA_ROOT,
        dataset_name="snli",
        tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased"),
        train_batch_size=4,
        flip_percent=0.15,
        num_workers=0,
    )
    print("Num classes:", dm.num_classes)
    dm.prepare_data()
    dm.setup("fit")
    # print(next(iter(dm.train_dataloader())))
    dm.val_dataloader()
    dm.test_dataloader()
    # print(dm.train_set.df.head(10))
    print(dm.flip_stats())
