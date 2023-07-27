from pathlib import Path
import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import os
import hydra
import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn.functional as F
from train import Model
from core.tracer import TracIn, GradientBasedTracer, IF
from core.grads import RuntimeGradientExtractor
import time
import logging
from tqdm import tqdm
from datamodule import DMStage

logger = logging.getLogger(__name__)

def cal_precision_top(scores, true_error_label, top=[0.05, 0.1, 0.15, 0.2]):
    num_top = [int(len(scores) * i) for i in top]
    ranking = scores.argsort()
    acc = []
    for t in num_top:
        candidate = ranking[:t]
        first = true_error_label[candidate].sum() 
        acc.append((first/t).item()*100)
        # print(f'{acc[-1]:.4f}', end=';\n')
    return acc

def process_results(results, ref_targets, flipped_idx, num_classes, verbose=True):
    true_error_labels = torch.isin(torch.arange(results.shape[1]), flipped_idx)
    
    # Sum-All Score
    sumall_score = results.sum(axis=0)
    all_prec = cal_precision_top(sumall_score, true_error_labels)

    # Sum-Class Score
    is_class = ref_targets[:, None] == torch.arange(num_classes)
    is_class = is_class.T
    logger.info(f'Number of element in each class: {is_class.sum(axis=1).tolist()}')
    num_per_class = is_class.sum(axis=1)
    print(f'Number of element in each class: {num_per_class}')

    class_score = [results[class_ind,:].mean(axis=0) for class_ind in is_class] 
    class_score = torch.vstack(class_score)
    print(class_score.shape)
    class_score = class_score.min(dim=0).values

    class_wise_prec = cal_precision_top(class_score, true_error_labels)
    if verbose:
        logger.info('Sum all evaluation')
        logger.info(all_prec)

        logger.info('Sum class-wise evaluation')
        logger.info(class_wise_prec)
    return all_prec, class_wise_prec

@hydra.main(config_path= Path(__file__).parent / 'configs', 
            config_name='evaluate_influence.yaml', 
            version_base="1.2")
def main(cfg):
    pl.seed_everything(cfg.seed)
    DATA_ROOT = Path(os.environ['PYTORCH_DATASET_ROOT'])

    # /////////////// Load data ///////////////
    # Load datasets
    t1 = time.time()
    dm = hydra.utils.instantiate(cfg.datamodule, data_root=DATA_ROOT)
    dm.prepare_data()
    logger.info("Finish load datasets in {:.2f} sec".format(time.time() - t1))
    dm.setup(DMStage.TRACING)
    test_loader = dm.test_dataloader()

    results = torch.tensor(torch.load(cfg.result_path))

    # /////////////// Evaluate ///////////////
    process_results(results, torch.concat([b[1] for b in test_loader], axis=0), dm.flipped_idx, dm.num_classes)
    

if __name__ == "__main__":
    main()