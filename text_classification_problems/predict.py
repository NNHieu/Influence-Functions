import argparse
from pathlib import Path
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import os
import pytorch_lightning as pl
import torch
from models.bert_classifier import BertClassifier
from datamodule import TextClassifierDataModule
import time
import logging
from transformers import AutoTokenizer

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def main(cfg):
    pl.seed_everything(cfg.seed)
    DATA_ROOT = Path(os.environ["PYTORCH_DATASET_ROOT"])
    device = cfg.device

    # Load checkpoint
    checkpoint = torch.load(cfg.best_ckpt_path)

    # Load module
    datamodule_hparams = checkpoint['datamodule_hyper_parameters']
    t1 = time.time()
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
    dm = TextClassifierDataModule(data_root=DATA_ROOT, tokenizer=tokenizer, **datamodule_hparams, use_denoised_data=False)
    dm.prepare_data()
    dm.setup("tracing")
    dm.train_set.df.to_csv(f'_train_noise{datamodule_hparams["flip_percent"]}%')
    logger.info("Finish load datasets in {:.2f} sec".format(time.time() - t1))

    # Create model
    model = BertClassifier(dm.num_classes, path_pretrained_or_name='bert-base-uncased', name='bert')
    model_weights = checkpoint["state_dict"]
    # update keys by dropping `net.`
    for key in list(model_weights):
        model_weights[key.replace("net.", "")] = model_weights.pop(key)
    logger.info(f"Model restored! Checkpoint path: {cfg.best_ckpt_path}")
    model.load_state_dict(model_weights)
    model.to(device)
    model.eval()

    test_loader = dm.test_dataloader()

    
    # /////////////// Predict ///////////////
    batch = next(iter(test_loader))

    with torch.no_grad():
        ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        y_hat = model(ids, attention_mask, token_type_ids)
        features = model.feature(ids, attention_mask, token_type_ids)
        print(f'Logit: {y_hat.cpu()}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--best_ckpt_path', type=str, required=True)
    cfg = parser.parse_args()

    main(cfg)
