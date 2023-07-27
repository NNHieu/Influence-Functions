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
from ImgExp.train import Model
from core.tracer import TracIn, GradientBasedTracer, IF
from core.grads import RuntimeGradientExtractor
import time
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

@hydra.main(config_path= Path(__file__).parent / 'configs', 
            config_name='tracing.yaml', 
            version_base="1.2")
def main(cfg):
    pl.seed_everything(42)
    DATA_ROOT = Path(os.environ['PYTORCH_DATASET_ROOT'])
    device = 'cpu'
    torch.use_deterministic_algorithms(mode=True) 

    # Create model
    net = hydra.utils.instantiate(cfg.net)
    lit_model = Model.load_from_checkpoint(cfg.ckpt_path, net=net)
    lit_model.eval()
    lit_model.to(device)
    logger.info(f'Model restored! Checkpoint path: {cfg.ckpt_path}')
    
    # /////////////// Load data ///////////////
    # Load datasets
    t1 = time.time()
    dm = hydra.utils.instantiate(cfg.datamodule, data_root=DATA_ROOT, train_batch_size=1)
    dm.prepare_data()
    dm.setup('test')
    logger.info("Finish load datasets in {:.2f} sec".format(time.time() - t1))
    train_loader, test_loader = dm.train_dataloader(shuffle=False), dm.test_dataloader()
    

    with torch.no_grad():
        grad_extractor = RuntimeGradientExtractor(
                                lit_model, 
                                split_params=lambda params: (params[:-2], params[-2:]), 
                                merge_params=lambda w1, w2: w1 + w2,
                                loss_fn=F.cross_entropy)
    def loss_fn(X, y):
        return F.cross_entropy(lit_model(X), y)
    params = [p for p in lit_model.parameters() if p.requires_grad][-2:]

    for i, batch in enumerate(grad_extractor.get_iter(train_loader)):
        X, y, grad1 = batch
        X, y = X.to(device), y.to(device)
        # grad1 = grad_extractor.grad_per_x(X, y)
        grad1 = grad_extractor.unflatten(grad1[0])
        # print([p.shape for p in grad1])
        loss = loss_fn(X, y)
        grad2 = torch.autograd.grad(loss, params)
        # print([p.shape for p in grad2])
        
        print([torch.allclose(g1, g2) for g1, g2 in zip(grad1, grad2)])
        # print((grad1[0] - grad2[0]).abs_().flatten().sort())
        if i >= 10: break

if __name__ == "__main__":
    main()