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
import torch
import numpy as np
import copy
from datamodule import TextClassifierDataModule
from transformers import AutoTokenizer
import logging
import torch.nn.functional as F
from core.grads import RuntimeGradientExtractor
from core.aggregation import cal_neibor_matrices
from core.eval import eval_fn

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def loss_fn(pred_fn, batch):
    labels = batch["label"]
    preds = pred_fn(batch)
    return F.cross_entropy(preds, labels)

def load_datamodule_from_ckpt(ckpt, tokenizer, use_denoised_data):
    datamodule_hparams = ckpt["datamodule_hyper_parameters"] 
    datamodule_hparams["train_batch_size"] = 256
    datamodule_hparams["use_denoised_data"] = use_denoised_data
    dm = TextClassifierDataModule(
        data_root=os.environ["PYTORCH_DATASET_ROOT"],
        tokenizer=tokenizer,
        **datamodule_hparams
    )
    return dm

def eval_ckpt(tokenizer, lit_model, ckpt_path, use_denoised_data, is_self_ref, sel_sizes, colected_ks, use_cache, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    dm = load_datamodule_from_ckpt(ckpt, tokenizer, use_denoised_data)
    dm.prepare_data()
    dm.setup("tracing")

    lit_model.load_state_dict(ckpt["state_dict"])
    
    grad_extractor = RuntimeGradientExtractor(
        lit_model,
        split_params=lambda params: (params[:-2], params[-2:]),
        merge_params=lambda w1, w2: w1 + w2,
        loss_fn=loss_fn,
        input_sample=next(iter(dm.trace_dataloader())),
    )
    neibor_inds_path = Path(ckpt_path + f".neibor{'_desnoised' if use_denoised_data else ''}.npz")
    if neibor_inds_path.exists():
        print("Load neibor_inds from disk")
        neibor_inds = torch.tensor(np.load(neibor_inds_path)['arr_0'])
    else:
        print("Compute neibor_inds")
        neibor_inds = cal_neibor_matrices(
            lit_model,
            ref_loader=dm.train_dataloader(shuffle=False),
            trace_loader=dm.trace_dataloader(),
            device=device,
            k=1000,
            is_self_ref=True
        )
        print("Saving neibor_inds")
        np.savez_compressed(str(neibor_inds_path), neibor_inds.numpy())

    result_df, all_scores = eval_fn(
        dm,
        lit_model,
        grad_extractor,
        label_from_batch=lambda b: b["label"],
        neighbor_matrices=neibor_inds,
        sel_sizes=sel_sizes,
        colected_ks=colected_ks,
        is_self_ref=is_self_ref,
        use_cache=use_cache,
        cache_file=ckpt_path + ".cache"
    )
    return result_df, all_scores

def glop(cfg):
    start_dir = Path(cfg.start_glob)
    for influence_results in start_dir.rglob("*/tracing/*/result.pt"):
        print(influence_results)
        method = influence_results.parent.name
        train_dir = influence_results
        while train_dir.name != 'tracing': train_dir = train_dir.parent
        train_dir = train_dir.parent

        c_cfg = copy.deepcopy(cfg)
        c_cfg.tracing_method = method
        c_cfg.train_output_dir = train_dir

        aggregation(c_cfg)


def main(cfg):
    if cfg.start_glob is not None:
        glop(cfg)
    else:
        aggregation(cfg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=12345)
    # parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--num_top', default=None)
    parser.add_argument('--top', default=None)
    parser.add_argument('--start_glob', type=str, default=None)
    parser.add_argument('--train_output_dir', type=str, default=None)
    parser.add_argument('--tracing_method', type=str, default=None)

    cfg = parser.parse_args()
    main(cfg)
