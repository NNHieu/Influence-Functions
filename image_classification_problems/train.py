import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import os
from pathlib import Path
import torch
from torch import nn
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import logging
import time
from pytorch_lightning.callbacks import ModelCheckpoint
from utils import get_pylogger, instantiate_loggers, close_loggers

log = get_pylogger(__name__)

class Model(LightningModule):
    def __init__(self, net: torch.nn.Module, learning_rate, num_classes) -> None:
        super().__init__()
        self.net = net
        self.criterion = nn.CrossEntropyLoss()
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        self.save_hyperparameters(ignore=['net'])
    
    def forward(self, x):
        return self.net(x)
    
    def _forward(self, batch):
        images, targets = batch
        logits = self.net(images)
        loss = self.criterion(logits, targets)
        preds = torch.argmax(logits, dim=1)
        return loss, preds
    
    def training_step(self, batch, batch_idx):
        loss, preds = self._forward(batch)
        targets = batch[1]
        self.train_accuracy.update(preds, targets)

        self.log('train/loss', loss, prog_bar=False)
        self.log('train/acc', self.train_accuracy, prog_bar=True, on_epoch=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, preds = self._forward(batch)
        targets = batch[1]
        self.val_accuracy.update(preds, targets)

        self.log('val/loss', loss.item(), prog_bar=True)
        self.log('val/acc', self.val_accuracy, prog_bar=True)
        self.log('val_acc', self.val_accuracy, prog_bar=False)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, preds = self._forward(batch)
        targets = batch[1]
        self.test_accuracy.update(preds, targets)

        self.log('test/loss', loss.item(), prog_bar=True)
        self.log('test/acc', self.test_accuracy, prog_bar=True)

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr=self.hparams.learning_rate)
        return optimizer


def train(cfg: DictConfig):
    t0 = time.time()
    log.info(cfg)
    out_dir = os.getcwd()
    log.info(f"out_dir={out_dir}")
    pl.seed_everything(cfg.seed)

    # Load datasets
    t1 = time.time()
    dm = hydra.utils.instantiate(cfg.datamodule, data_root=os.environ['PYTORCH_DATASET_ROOT'])
    dm.prepare_data()
    dm.setup('fit')
    log.info("Finish load datasets in {:.2f} sec".format(time.time() - t1))

    net = hydra.utils.instantiate(cfg.net, num_classes=dm.num_classes)
    lit_model = Model(net, cfg.learning_rate, num_classes=dm.num_classes)

    # Initialize trainer
    best_checkpoint_callback = ModelCheckpoint(
        save_top_k= cfg.trainer.max_epochs,  # avoid -1 value to work with wandb logger
        monitor="val_acc",
        mode="max",
        dirpath=Path(cfg.paths.output_dir) / "checkpoints",
        filename="{epoch:02d}_{val_acc:.4f}",
        save_weights_only=True,
    )
    last_checkpoint_callback = ModelCheckpoint(
        save_last=False,
        dirpath=Path(cfg.paths.output_dir) / "checkpoints",
        filename="last_{epoch:02d}_{val_acc:.4f}",
    )
    setattr(last_checkpoint_callback, "avail_to_wandb", False)

    log.info("Instantiating loggers...")
    logger = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=[last_checkpoint_callback, best_checkpoint_callback],
        logger=logger,
    )

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=lit_model, datamodule=dm, ckpt_path=cfg.get("ckpt_path"))
    
    log.info(f"Best ckpt: {best_checkpoint_callback.best_model_path}")
    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=lit_model, datamodule=dm, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    log.info(
        "Finish in {:.2f} sec. out_dir={}".format(
            time.time() - t0, cfg.paths.output_dir
        )
    )
    close_loggers()

@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(cfg):
    train(cfg)

if __name__ == '__main__':
    from pyrootutils import setup_root
    root = setup_root(__file__, 
                     indicator=['.git'],
                     dotenv=True,
                     pythonpath=True,
                     cwd=False)
    main()