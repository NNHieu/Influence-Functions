from pathlib import Path
import torch
from core.tree_utils import tree_to_device
from core.grads import RuntimeGradientExtractor
from core.aggregation import aggregation, cal_neibor_matrices
from core.tracer import IF

import pandas as pd
from tqdm import tqdm


def cal_precision_top(scores, true_error_label, num_top, descending=False):
    # scores, ranking = scores.sort(descending=descending)
    num_errors = true_error_label.sum()
    scores, ranking = torch.sort(scores, descending=descending)
    acc = []
    score_lower_vals = []
    label_ranking = true_error_label[ranking]
    for t in num_top:
        first = label_ranking[:t].sum()
        acc.append((first / t).item() * 100)
        score_lower_vals.append(scores[t - 1].item())
        # print(f'{acc[-1]:.4f}', end=';\n')
    where_errors = torch.argwhere(true_error_label == 1)
    where_noerrors = torch.argwhere(true_error_label != 1)
    right_traces = torch.zeros_like(true_error_label)
    right_traces[where_errors] = torch.isin(where_errors, ranking[:num_errors])
    right_traces[where_noerrors] = torch.isin(where_noerrors, ranking[num_errors:])
    return acc, label_ranking, right_traces, score_lower_vals


def eval_fn(
    datamodule,
    model,
    grad_extractor: RuntimeGradientExtractor,
    label_from_batch,
    neighbor_matrices,
    sel_sizes=range(50, 1310, 50),
    colected_ks=list(range(50, 1000, 50)),
    is_self_ref=False,
    use_cache=False,
    cache_file=None,
):
    # /////////////////// Preparation ////////////////////////
    num_classes = datamodule.num_classes
    trace_loader = datamodule.trace_dataloader()
    device = grad_extractor.device
    true_error_label = torch.isin(
        torch.arange(len(datamodule.trace_set)), datamodule.flipped_inds
    )
    columns = ["method"] + [f"top{s}" for s in sel_sizes]
    final_result_df = pd.DataFrame(columns=columns)
    all_scores = {}
    def cal_and_add(method_name, scores, descending=False):
        all_scores[method_name] = scores.cpu()
        results, _, _= cal_precision_top(
            scores, true_error_label, sel_sizes, descending=descending
        )
        row = [method_name] + list(results)
        final_result_df.loc[len(final_result_df.index)] = row

    ref_loader = datamodule.ref_dataloader()
    ref_targets = torch.cat([label_from_batch(b) for b in ref_loader])
    is_class = ref_targets[:, None] == torch.arange(datamodule.num_classes)
    is_class = is_class.T
    print(f"Number of element in each class: {is_class.sum(axis=1).tolist()}")
    del is_class
    shuffled_train_loader = datamodule.train_dataloader(shuffle=True)

    # /////////////////// Collecting ////////////////////////////
    # --------------------- Prepare Reference -------------------
    if_tracer = IF(grad_extractor, damp=0.01, scale=25, recursion_depth=10, r=2)
    refferent_grads = []
    # ref_stests = []
    pbar = tqdm(grad_extractor.get_iter(ref_loader), desc="Collect Ref Grads")
    for batch_z, grads in pbar:
        refferent_grads.append(grads)
        # ref_stest = if_tracer._calc_s_test_single(None, shuffled_train_loader, V=grads)
        # ref_stests.append(ref_stest)
    refferent_grads = torch.concat(refferent_grads, dim=0)
    refferent_grads_norms = torch.clamp(
        refferent_grads.norm(dim=1, keepdim=True), min=1e-8
    ).cpu()
    # ref_stests = torch.concat(ref_stests, dim=0)
    
    
    # --------------------- Loop Trace Set -------------------
    all_losses = []
    trace_grad_norms = []
    all_GD_influences = []
    trace_grads = []
    all_if_scores = []
    all_trace_labels = []
    with torch.no_grad():
        for batch in tqdm(trace_loader, desc="Tracing"):
            all_trace_labels.append(label_from_batch(batch).cpu())
            batch = tree_to_device(batch, device)
            grads, losses = grad_extractor.grad_per_input(batch, return_loss=True)
            grad_extractor.s_test
            GD_scores = refferent_grads @ grads.T
            # if_scores = ref_stests @ grads.T

            trace_grads.append(grads.cpu())
            # all_if_scores.append(if_scores.cpu())
            all_GD_influences.append(GD_scores.cpu())
            all_losses.append(losses.cpu())
            trace_grad_norms.append(grads.norm(dim=1).cpu())
    all_trace_labels = torch.cat(all_trace_labels).squeeze_()

    # /////////////////// Aggregation ////////////////////////////
    all_losses = torch.cat(all_losses).cpu()
    cal_and_add("LossValue", all_losses, descending=True)

    trace_grad_norms = torch.cat(trace_grad_norms).cpu()
    cal_and_add("LengthGD", trace_grad_norms, descending=True)
    trace_grad_norms = trace_grad_norms.clamp_(min=1e-8)

    # all_if_scores = torch.cat(all_if_scores, dim=1).cpu()
    # SA_IF_scores = aggregation(all_if_scores, reduction="sum_all")
    # cal_and_add("SA_IF_GD", SA_IF_scores, descending=False)
    # SC_IF_scores = aggregation(all_if_scores, "sum_class", ref_targets, num_classes, verbose=False)
    # cal_and_add("SC_IF_GD", SC_IF_scores, descending=False)

    all_GD_influences = torch.cat(all_GD_influences, dim=1).cpu()
    SA_GD_scores = aggregation(all_GD_influences, reduction="sum_all")
    cal_and_add("SA_GD", SA_GD_scores, descending=False)
    SC_GD_scores = aggregation(
        all_GD_influences, "sum_class", ref_targets, num_classes, verbose=False
    )
    cal_and_add("SC_GD", SC_GD_scores, descending=False)

    all_GN_influences = all_GD_influences / refferent_grads_norms
    SA_GN_scores = aggregation(all_GN_influences, reduction="sum_all")
    cal_and_add("SA_GN", SA_GN_scores, descending=False)
    SC_GN_scores = aggregation(
        all_GN_influences, "sum_class", ref_targets, num_classes, verbose=False
    )
    cal_and_add("SC_GN", SC_GN_scores, descending=False)

    all_GC_influences = all_GN_influences / trace_grad_norms[None, :]
    SA_GC_scores = aggregation(all_GC_influences, reduction="sum_all")
    cal_and_add("SA_GC", SA_GC_scores, descending=False)
    SC_GC_scores = aggregation(
        all_GC_influences, "sum_class", ref_targets, num_classes, verbose=False
    )
    cal_and_add("SC_GC", SC_GC_scores, descending=False)

    # /////////////////// Collecting KNN's Ref Set ////////////////////////////
    trace_grads = torch.cat(trace_grads, dim=0)
    if is_self_ref:
        train_grads = trace_grads
        train_grads_norms = trace_grad_norms
        all_labels = []
        with torch.no_grad():
            for batch in tqdm(
                datamodule.train_dataloader(shuffle=False), desc="Loop ref knn"
            ):
                all_labels.append(label_from_batch(batch).cpu())
        all_labels = torch.concat(all_labels).squeeze_()
    else:
        if use_cache and Path(cache_file).exists():
            print(f"Load cache {cache_file}")
            cache = torch.load(cache_file)
            all_labels = cache["all_labels"]
            train_grads = cache["train_grads"]
            train_grads_norms = cache["train_grads_norms"]
        else:
            all_labels = []
            train_grads = []
            train_grads_norms = []
            with torch.no_grad():
                for batch in tqdm(
                    datamodule.train_dataloader(shuffle=False), desc="Loop ref knn"
                ):
                    all_labels.append(label_from_batch(batch).cpu())
                    batch = tree_to_device(batch, device)
                    grads, losses = grad_extractor.grad_per_input(
                        batch, return_loss=True
                    )
                    train_grads.append(grads.cpu())
                    train_grads_norms.append(grads.norm(dim=1).cpu())
            all_labels = torch.concat(all_labels).squeeze_()
            train_grads = torch.cat(train_grads, dim=0)
            train_grads_norms = torch.cat(train_grads_norms).cpu().clamp_(min=1e-8)
            if cache_file is not None:
                print("Saving cache file " + str(cache_file))
                torch.save(
                    {
                        "all_labels": all_labels,
                        "train_grads": train_grads,
                        "train_grads_norms": train_grads_norms,
                    },
                    cache_file,
                )

    knn_scores = torch.zeros_like(all_trace_labels)
    refferent_grads = torch.zeros_like(trace_grads)
    refferent_grads_normed = torch.zeros_like(trace_grads)
    # k_orders = torch.randperm(neighbor_matrices.shape[1]).tolist()
    k_orders = range(neighbor_matrices.shape[1])
    for i, k_ind in tqdm(enumerate(k_orders), desc="Loop KNN GD"):
        neighbor_k_inds = neighbor_matrices[:, k_ind]

        neighbor_labels = all_labels[neighbor_k_inds]
        knn_scores += (neighbor_labels != all_trace_labels).int()

        neighnor_grads = train_grads[neighbor_k_inds]
        refferent_grads = refferent_grads + neighnor_grads

        neighnor_grads_normed = (
            neighnor_grads / train_grads_norms[neighbor_k_inds, None]
        )
        refferent_grads_normed = refferent_grads_normed + neighnor_grads_normed

        k = i + 1
        if k in colected_ks:
            cal_and_add(f"KNN_{k}", knn_scores / k, descending=True)

            KNN_GD_scores = ((refferent_grads / k) * trace_grads).sum(dim=1)
            cal_and_add(f"KNN_GD_{k}", KNN_GD_scores, descending=False)

            # refferent_grads_normed = refferent_grads / refferent_grads.norm(dim=1, keepdim=True).clamp_(min=1e-8)
            KNN_GN_scores = ((refferent_grads_normed / k) * trace_grads).sum(dim=1)
            cal_and_add(f"KNN_GN_{k}", KNN_GN_scores, descending=False)

            cal_and_add(
                f"KNN_GC_{k}", KNN_GN_scores / trace_grad_norms, descending=False
            )

    # print(final_result_df)
    return final_result_df, all_scores
