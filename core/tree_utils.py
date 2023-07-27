import torch
import functorch as ftorch
import optree

def tree_to_device(tree, device):
    return optree.tree_map(lambda x: x.to(device), tree, namespace="tracing")

def tree_flatten(tree):
    return optree.tree_flatten(tree, namespace='tracing')

def flatten_params(v):
    def f(v):
        leaves, _ = optree.tree_flatten(v)
        return torch.cat([x.view(-1) for x in leaves])

    out, pullback = ftorch.vjp(f, v)
    return out, lambda x: pullback(x)[0]