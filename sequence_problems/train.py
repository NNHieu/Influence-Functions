import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import hydra
from omegaconf import DictConfig
import time
from modelmodule import TextClassifierModel
from pytorch_lightning.callbacks import ModelCheckpoint
from utils import get_pylogger, instantiate_loggers, close_loggers, log_hyperparameters
from pytorch_lightning.loggers import WandbLogger

log = get_pylogger(__name__)


def train(cfg: DictConfig):
    t0 = time.time()
    pl.seed_everything(cfg.seed)

    # Load datasets
    t1 = time.time()
    dm = hydra.utils.instantiate(cfg.datamodule)
    dm.prepare_data()
    dm.setup("fit")
    log.info("Finish load datasets in {:.2f} sec".format(time.time() - t1))

    # Load model
    net = hydra.utils.instantiate(cfg.net, num_classes=dm.num_classes)
    lit_model = TextClassifierModel(net, cfg.learning_rate, num_classes=dm.num_classes)

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

    object_dict = {
        "cfg": cfg,
        "datamodule": dm,
        "model": lit_model,
        # "callbacks": ,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=lit_model, datamodule=dm, ckpt_path=cfg.get("ckpt_path"))

    # We have problems with testing in ddp settings
    # if cfg.get("test"):
    #     log.info("Starting testing!")
    #     ckpt_path = best_checkpoint_callback.best_model_path
    #     if ckpt_path == "":
    #         log.warning("Best ckpt not found! Using current weights for testing...")
    #         ckpt_path = None
    #     trainer.test(model=lit_model, datamodule=dm, ckpt_path=ckpt_path)
    #     log.info(f"Best ckpt path: {ckpt_path}")

    log.info(
        "Finish in {:.2f} sec. out_dir={}".format(
            time.time() - t0, cfg.paths.output_dir
        )
    )
    close_loggers()


@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(cfg):
    train(cfg)


if __name__ == "__main__":
    from pyrootutils import setup_root

    root = setup_root(
        __file__, indicator=[".git"], dotenv=True, pythonpath=True, cwd=False
    )
    main()
