import time
from sklearn.metrics import roc_auc_score
import numpy as np
from core.tree_utils import tree_to_device
from tqdm import tqdm
import torch

def maha_distance(xs,cov_inv_in,mean_in,norm_type=None):
  diffs = xs - mean_in.reshape([1,-1])

  second_powers = np.matmul(diffs,cov_inv_in)*diffs

  if norm_type in [None,"L2"]:
    return np.sum(second_powers,axis=1)
  elif norm_type in ["L1"]:
    return np.sum(np.sqrt(np.abs(second_powers)),axis=1)
  elif norm_type in ["Linfty"]:
    return np.max(second_powers,axis=1)

def get_scores(
    indist_train_embeds_in,
    indist_train_labels_in,
    fc_weight,
    indist_test_embeds_in,
    indist_test_labels_in,
    subtract_mean = True,
    normalize_to_unity = True,
    subtract_train_distance = True,
    indist_classes = 100,
    norm_name = "L2",
    ):
  
  # storing the replication results
  maha_intermediate_dict = dict()
  
  description = ""
  
  all_train_mean = np.mean(indist_train_embeds_in,axis=0,keepdims=True)

  indist_train_embeds_in_touse = indist_train_embeds_in
  indist_test_embeds_in_touse = indist_test_embeds_in

  if subtract_mean:
    indist_train_embeds_in_touse -= all_train_mean
    indist_test_embeds_in_touse -= all_train_mean
    description = description+" subtract mean,"

  if normalize_to_unity:
    indist_train_embeds_in_touse = indist_train_embeds_in_touse / np.linalg.norm(indist_train_embeds_in_touse,axis=1,keepdims=True)
    indist_test_embeds_in_touse = indist_test_embeds_in_touse / np.linalg.norm(indist_test_embeds_in_touse,axis=1,keepdims=True)
    description = description+" unit norm,"

  #full train single fit
  mean = np.mean(indist_train_embeds_in_touse,axis=0)
  cov = np.cov((indist_train_embeds_in_touse-(mean.reshape([1,-1]))).T)

  eps = 1e-8
  cov_inv = np.linalg.inv(cov)

  #getting per class means and covariances
  class_means = []
  class_cov_invs = []
  class_covs = []
  fc_weight = fc_weight / np.linalg.norm(fc_weight, axis=1, keepdims=True)
  for c in range(indist_classes):

    mean_now_1 = np.mean(indist_train_embeds_in_touse[indist_train_labels_in == c],axis=0)
    mean_norm = np.mean(np.linalg.norm(indist_train_embeds_in_touse[indist_train_labels_in == c], axis=1), axis=0)
    print(mean_norm)
    mean_now = fc_weight[c] * mean_norm
    print(np.linalg.norm(mean_now - mean_now_1))

    cov_now = np.cov((indist_train_embeds_in_touse[indist_train_labels_in == c]-(mean_now.reshape([1,-1]))).T)
    class_covs.append(cov_now)
    # print(c)

    eps = 1e-8
    cov_inv_now = np.linalg.inv(cov_now)

    class_cov_invs.append(cov_inv_now)
    class_means.append(mean_now)

  #the average covariance for class specific
  class_cov_invs = [np.linalg.inv(np.mean(np.stack(class_covs,axis=0),axis=0))]*len(class_covs)

  maha_intermediate_dict["class_cov_invs"] = class_cov_invs
  maha_intermediate_dict["class_means"] = class_means
  maha_intermediate_dict["cov_inv"] = cov_inv
  maha_intermediate_dict["mean"] = mean

  in_totrain = maha_distance(indist_test_embeds_in_touse,cov_inv,mean,norm_name)

  in_totrainclasses = [maha_distance(indist_test_embeds_in_touse,class_cov_invs[c],class_means[c],norm_name) for c in range(indist_classes)]

  dist2clsmean = np.stack(in_totrainclasses,axis=0)
  # probs = np.exp(-0.5 * dist2clsmean)
  in_scores = dist2clsmean[indist_test_labels_in, np.arange(indist_test_labels_in.shape[0])]

  if subtract_train_distance:
    in_scores = in_scores - in_totrain


  scores = in_scores

  return scores, dist2clsmean, description, maha_intermediate_dict

def standalone_get_prelogits(feature_extractor, feature_to_logit, ds_in, max_sample_count=50000, device="cuda"):
  """Returns prelogits on the dataset"""
  count_so_far = 0
  prelogits_all = []
  logits_all = []
  labels_all = []

  for batch in tqdm(ds_in):
    batch = tree_to_device(batch, device)
    prelogits = feature_extractor(batch)
    prelogits_all.append(prelogits)
    
    if feature_to_logit is not None:
      logits = feature_to_logit(prelogits)
      logits_all.append(logits)
    
    labels_all.append(batch["label"])
    batch_size = prelogits.shape[0]
    count_so_far += batch_size
    if count_so_far >= max_sample_count:
      break #early break for subsets of data

  prelogits_all = torch.concat(prelogits_all,axis=0)[:max_sample_count]
  if feature_to_logit is not None:
    logits_all = torch.concat(logits_all,axis=0)[:max_sample_count]
  labels_all = torch.concat(labels_all,axis=0)[:max_sample_count]

  return  prelogits_all, logits_all, labels_all