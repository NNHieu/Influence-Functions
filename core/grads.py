import torch
import functorch as ftorch
import optree

# from torch.utils.data import Dataset, DataLoader
from functools import partial


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


def hvp(J, w, v):
    return ftorch.jvp(ftorch.grad(J), primals=(w,), tangents=(v,))[1]


class GradientExtractor:
    def __init__(self):
        pass

    def load_state_dict(self, *, path=None, state_dict=None):
        raise NotImplementedError

    def grad_per_input(self, x, y):
        raise NotImplementedError

    def s_test(
        self, batch, ref_dloader, recursion_depth=5000, damp=0.01, scale=25.0
    ) -> torch.Tensor:
        raise NotImplemented

    def get_iter(self, dataloader):
        raise NotImplementedError

    @property
    def device(self):
        raise NotImplemented


class RuntimeGradientExtractor(GradientExtractor):
    def __init__(
        self, net: torch.nn.Module, split_params, merge_params, loss_fn, input_sample
    ) -> None:
        self.net = net
        self._split_params = split_params
        self._merge_params = merge_params
        self._loss_fn = loss_fn
        self._dump_input = optree.tree_map(
            lambda x: x.size(), input_sample, namespace="tracing"
        )
        print(self._dump_input)
        self._make_functional()

    def load_state_dict(self, *, path=None, state_dict=None):
        assert state_dict is not None
        self.net.load_state_dict(state_dict)
        self.net.to(self.device)
        self._make_functional()

    @property
    def flatten_params(self):
        return self._flatten_trace_params

    def _build_grad_fn(self, f_net, freeze_params, buffer, _dump_input):
        def loss_fn_wraper(flatten_params, z_flat, z_spec):
            z = optree.tree_unflatten(z_spec, z_flat)
            params = self._merge_params(freeze_params, self.unflatten(flatten_params))
            return self._loss_fn(partial(f_net, params, buffer), z)

        grad_loss = ftorch.grad_and_value(loss_fn_wraper)
        _dump_input_flat, _ = tree_flatten(_dump_input)
        print(_dump_input_flat)
        in_dims = (
            None,  # Params
            [ 0,] * len(_dump_input_flat),  # Flatten input
            None,  # Input's spec
        )
        batch_grad = ftorch.vmap(grad_loss, in_dims=in_dims)

        def flatten_grad_fn(w, z):
            z_flat, z_spec = tree_flatten(z)
            z_flat = [x.unsqueeze(1) for x in z_flat]
            return batch_grad(w, z_flat, z_spec)

        return flatten_grad_fn

    def _build_hvn_fn(self, f_net, freeze_params, buffer):
        def loss_fn_wraper(flatten_params, z):
            params = self._merge_params(freeze_params, self.unflatten(flatten_params))
            return self._loss_fn(partial(f_net, params, buffer), z)

        # TODO: This maybe not efficient
        hvp_fn = lambda V, Z: hvp(
            lambda w: loss_fn_wraper(w, Z),
            self._flatten_trace_params,
            V,
        )

        return ftorch.vmap(hvp_fn, in_dims=(0, None))

    def _make_functional(self):
        # Make network functional
        f_net, params, buffer = ftorch.make_functional_with_buffers(
            self.net, disable_autograd_tracking=True
        )
        freeze_params, trace_params = self._split_params(params)
        self._flatten_trace_params, self.unflatten = flatten_params(trace_params)

        self._grad_fn = self._build_grad_fn(f_net, freeze_params, buffer, self._dump_input)
        self.hvp_fn = self._build_hvn_fn(f_net, freeze_params, buffer)

    def grad_per_input(self, z, return_loss=False) -> torch.Tensor:
        if return_loss:
            return self._grad_fn(self._flatten_trace_params, z)
        return self._grad_fn(self._flatten_trace_params, z)[0]

    def s_test(self, ref_dloader, batch = None, V=None, recursion_depth=5000, damp=0.01, scale=25.0):
        device = self.device
        if V is None:
            assert batch is not None
            V = self.grad_per_input(batch)
        HVP_estimate = V.clone()

        for i in range(recursion_depth):
            # take just one random sample from training dataset
            # easiest way to just use the DataLoader once, break at the end of loop
            #########################
            # TODO: do x, t really have to be chosen RANDOMLY from the train set?
            #########################
            for batch_ref in ref_dloader:
                batch_ref = tree_to_device(batch_ref, device)
                # Recursively caclulate h_estimate
                HVP_estimate = (
                    V
                    + (1 - damp) * HVP_estimate
                    - self.hvp_fn(HVP_estimate, batch_ref) / scale
                )
                break
        return HVP_estimate

    def get_iter(self, dataloader):
        data_iter = iter(dataloader)
        device = self.device
        for batch_z in data_iter:
            batch_z = tree_to_device(batch_z, device)
            grads = self.grad_per_input(batch_z)
            yield batch_z, grads

    @property
    def device(self):
        return self._flatten_trace_params.device
