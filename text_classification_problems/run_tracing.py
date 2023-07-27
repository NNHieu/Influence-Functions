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
import torch
import torch.nn.functional as F
from modelmodule import TextClassifierModel
from core.grads import RuntimeGradientExtractor
import time
import logging
import optree
from transformers.tokenization_utils_base import BatchEncoding

def register_BatchEncoding():
    def BatchEncoding_flatten(batch: BatchEncoding):  # -> (children, metadata, entries)
        reversed_keys = sorted(batch.keys(), reverse=True)
        return (
            [batch[key] for key in reversed_keys],  # children
            (reversed_keys, batch.encodings, 2),  # metadata
        )

    def BatchEncoding_unflatten(metadata, children):
        reversed_keys, encodings, n_sequences = metadata
        return BatchEncoding(
            data=dict(zip(reversed_keys, children)),
            encoding=encodings,
            n_sequences=n_sequences,
            tensor_type=None,
            prepend_batch_axis=False,
        )

    optree.register_pytree_node(
        BatchEncoding,
        flatten_func=BatchEncoding_flatten,
        unflatten_func=BatchEncoding_unflatten,
        namespace="tracing",
    )


logger = logging.getLogger(__name__)


@hydra.main(
    config_path=Path(__file__).parent / "configs",
    config_name="tracing.yaml",
    version_base="1.2",
)
def main(cfg):
    pl.seed_everything(cfg.seed)
    DATA_ROOT = Path(os.environ["PYTORCH_DATASET_ROOT"])
    device = "cuda"
    train_output_dir = Path(cfg.train_output_dir)
    checkpoint_dir = train_output_dir / "checkpoints"
    metrics = [(float(p.name[-11:-5]), p) for p in checkpoint_dir.glob("epoch*.ckpt")]
    metrics.sort(reverse=True)
    best_ckpt_path = metrics[0][1]
    logger.info(f"Checkpoint path: {best_ckpt_path}")
    checkpoint = torch.load(best_ckpt_path)

    # /////////////// Load data ///////////////
    # Load datasets
    datamodule_hparams = checkpoint['datamodule_hyper_parameters']
    t1 = time.time()
    dm = hydra.utils.instantiate(
        cfg.datamodule,
        flip_percent=datamodule_hparams['flip_percent'],
        flip_seed=datamodule_hparams['flip_seed'],
    )
    dm.prepare_data()
    dm.setup("tracing")
    logger.info("Finish load datasets in {:.2f} sec".format(time.time() - t1))

    # Create model
    net = hydra.utils.instantiate(cfg.net, num_classes=dm.num_classes)
    lit_model = TextClassifierModel.load_from_checkpoint(
        best_ckpt_path, net=net, num_classes=dm.num_classes
    )
    logger.info(f"Model restored! Checkpoint path: {best_ckpt_path}")

    # # /////////////// Evaluate model ///////////////
    # lit_model.eval()
    # lit_model.to(device)
    # test_loader = dm.test_dataloader()
    # trainer = pl.Trainer(
    #     default_root_dir=cfg.paths.output_dir,
    #     gpus=1 if torch.cuda.is_available() else 0,
    #     num_sanity_val_steps=0,
    #     logger=False
    # )
    # trainer.test(lit_model, test_loader)

    # dm.setup("tracing")
    train_loader, test_loader = dm.train_dataloader(shuffle=False), dm.test_dataloader()
    lit_model.eval()
    lit_model.to(device)
    def loss_fn(pred_fn, batch):
        labels = batch["label"]
        preds = pred_fn(batch)
        return F.cross_entropy(preds, labels)

    with torch.no_grad():
        # /////////////// Detection Prelims ///////////////
        grad_extractor = RuntimeGradientExtractor(
            lit_model,
            split_params=lambda params: (params[:-4], params[-4:]),
            merge_params=lambda w1, w2: w1 + w2,
            loss_fn=loss_fn,
            input_sample=next(iter(test_loader)),
        )
        tracer = hydra.utils.instantiate(cfg.tracer, grad_extractor=grad_extractor)
        # detector = TracIn(grad_extractor=grad_extractor,
        #                   ckpt_paths=[cfg.ckpt_path])
        # detector = GradientBasedTracer(grad_extractor=grad_extractor)
        # detector = IF(grad_extractor=grad_extractor, recursion_depth=1)

        # /////////////// Tracing ///////////////
        results = tracer.trace_dataloader(
            train_loader,
            test_loader,
            shuffled_train_loader=dm.train_dataloader(shuffle=True),
        )

    results = results.cpu()
    # /////////////// Save result ///////////////
    logger.info("Saving result")
    torch.save({"influence": results, "datamodule_hparams": datamodule_hparams}, Path(cfg.paths.output_dir) / "result.pt")

    # /////////////// Evaluate ///////////////
    # process_results(
    #     results,
    #     torch.concat([b['label'] for b in test_loader], axis=0),
    #     dm.flipped_idx,
    #     dm.num_classes,
    # )

    # print(results.shape)
    # print(score[ranking])
    # print(ranking)


if __name__ == "__main__":
    main()
