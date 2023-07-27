import argparse
from pathlib import Path
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import torch
import numpy as np
import copy
import logging
from tqdm import tqdm
from core.tree_utils import tree_to_device
from sklearn.neighbors import KNeighborsClassifier

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def cal_neibor_matrices(lit_model, ref_loader, trace_loader, device, k, is_self_ref=False):
    if is_self_ref:
        k = k + 1
    features = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(ref_loader):
            batch = tree_to_device(batch, device)
            features.append(lit_model.feature(batch).cpu())
            labels.append(batch['label'].cpu())
        features = torch.concat(features).numpy()
        labels = torch.concat(labels).numpy()
        neigh = KNeighborsClassifier(n_neighbors=k, metric='minkowski', algorithm='kd_tree', n_jobs=-1) # minkowski <=> euclidean  
        neigh.fit(features, labels)

        neibor_inds = []
        for batch in tqdm(trace_loader):
            batch = tree_to_device(batch, device)
            trace_feature = lit_model.feature(batch).cpu()
            neibor_inds.append(neigh.kneighbors(trace_feature, n_neighbors=k, return_distance=False))
        neibor_inds = np.concatenate(neibor_inds, axis=0)
        neibor_inds = torch.tensor(neibor_inds)
    if is_self_ref:
        neibor_inds = neibor_inds[:, 1:]
    return neibor_inds

def aggregation(results,  reduction: str, ref_targets=None, num_classes=None, verbose=True):
    if reduction == "none":
        return results
    elif reduction == "sum_all":
        return results.mean(axis=0)
    elif reduction == "sum_class":
        assert ref_targets is not None and num_classes is not None
        is_class = ref_targets[:, None] == torch.arange(num_classes)
        is_class = is_class.T
        if verbose:
            logger.info(f"Number of element in each class: {is_class.sum(axis=1).tolist()}")
        class_score = [results[class_ind, :].mean(axis=0) for class_ind in is_class]
        class_score = torch.vstack(class_score)
        class_score = class_score.min(dim=0).values
        return class_score