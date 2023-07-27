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
from convert_result import process_results

logger = logging.getLogger(__name__)

@hydra.main(config_path= Path(__file__).parent / 'configs', 
            config_name='tracing.yaml', 
            version_base="1.2")
def main(cfg):
    pl.seed_everything(cfg.seed)
    DATA_ROOT = Path(os.environ['PYTORCH_DATASET_ROOT'])
    device = 'cuda'
    train_output_dir = Path(cfg.train_output_dir)
    last_ckpt_path = next((train_output_dir/'checkpoints').glob('*.last.ckpt'))
    
     # /////////////// Load data ///////////////
    # Load datasets
    t1 = time.time()
    dm = hydra.utils.instantiate(cfg.datamodule, data_root=DATA_ROOT)
    dm.prepare_data()
    # dm.setup(DMStage.FIT)
    logger.info("Finish load datasets in {:.2f} sec".format(time.time() - t1))


    # Create model
    net = hydra.utils.instantiate(cfg.net, num_classes=dm.num_classes)
    lit_model = Model.load_from_checkpoint(last_ckpt_path, net=net, num_classes=dm.num_classes)
    lit_model.eval()
    lit_model.to(device)
    logger.info(f'Model restored! Checkpoint path: {last_ckpt_path}')
    
    # if cfg.run_test:
        # test_loader = dm.test_dataloader()
        # # /////////////// Evaluate model ///////////////
        # trainer = pl.Trainer(
        #     default_root_dir=cfg.paths.output_dir,
        #     gpus=1 if torch.cuda.is_available() else 0,
        #     num_sanity_val_steps=0,
        #     logger=False
        # )
        # trainer.test(lit_model, test_loader)

    dm.setup(DMStage.TRACING)
    train_loader, test_loader = dm.train_dataloader(shuffle=False), dm.test_dataloader()

    lit_model.eval()
    lit_model.to(device)

    def loss_fn(pred_fn, Z):
        X, Y = Z
        preds = pred_fn(X)
        return F.cross_entropy(preds, Y)

    with torch.no_grad():
    # /////////////// Detection Prelims ///////////////
        grad_extractor = RuntimeGradientExtractor(
                                lit_model, 
                                split_params=lambda params: (params[:-2], params[-2:]), 
                                merge_params=lambda w1, w2: w1 + w2,
                                loss_fn=loss_fn,
                                input_sample=next(iter(train_loader)))
        tracer = hydra.utils.instantiate(cfg.tracer, grad_extractor=grad_extractor)
        # detector = TracIn(grad_extractor=grad_extractor,
        #                   ckpt_paths=[cfg.ckpt_path])
        # detector = GradientBasedTracer(grad_extractor=grad_extractor)
        # detector = IF(grad_extractor=grad_extractor, recursion_depth=1)

    # /////////////// Tracing ///////////////
        results = tracer.trace_dataloader(train_loader, 
                                          test_loader,
                                          shuffled_train_loader=dm.train_dataloader(shuffle=True))
    
    results = results.cpu()
    # /////////////// Save result ///////////////
    logger.info('Saving result')
    torch.save(results.cpu().numpy(), Path(cfg.paths.output_dir) / 'result.pt')

    # /////////////// Evaluate ///////////////
    process_results(results, torch.concat([b[1] for b in test_loader], axis=0), dm.flipped_idx, dm.num_classes)

    # print(results.shape)
    # print(score[ranking])
    # print(ranking)
    


if __name__ == "__main__":
    main()