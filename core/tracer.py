from pathlib import Path
import torch
import torch.nn.functional as F
from tqdm import tqdm
from .grads import GradientExtractor, tree_to_device
from sklearn.neighbors import KNeighborsClassifier

class Tracer:
    def __init__(self) -> None:
        pass

    def setup(self, dataloader):
        raise NotImplemented

    def trace_batch(self, x, y):
        raise NotImplementedError

    def _trace_dataloader(self, train_loader, test_loader, **kwargs):
        raise NotADirectoryError

    def __str__(self) -> str:
        return "base_tracer"


class GradientBasedTracer(Tracer):
    def __init__(self, grad_extractor: GradientExtractor) -> None:
        super().__init__()
        self.grad_extractor = grad_extractor

    def setup(self, dataloader):
        self._refferent_grads = []
        pbar = tqdm(self.grad_extractor.get_iter(dataloader), desc="Seting up Tracer")
        for batch_z, grads in pbar:
            self._refferent_grads.append(grads)
        self._refferent_grads = torch.concat(self._refferent_grads, dim=0)

    @classmethod
    def _influence_from_grad(cls, grad_ref, grad_trace):
        out = grad_ref @ grad_trace.T
        # out = torch.norm(grad_2, dim=1).repeat(grad_1.shape[0],1)
        return out

    def trace_batch(self, *z):
        z_grad = self.grad_extractor.grad_per_input(*z)
        influence = self._influence_from_grad(self._refferent_grads, z_grad)
        return influence

    def _trace_dataloader(self, train_loader, test_loader, result_out=None, **kwargs):
        device = self.device
        out_shape = (len(test_loader.dataset), len(train_loader.dataset))

        results = result_out
        if results is None:
            print("Allocating a matrix of size:", out_shape)
            results = torch.zeros(
                len(test_loader.dataset), len(train_loader.dataset), dtype=float
            ).to(device)

        assert results.shape == (len(test_loader.dataset), len(train_loader.dataset))

        self.setup(test_loader)
        count = 0
        with torch.no_grad():
            for batch in tqdm(train_loader, desc="Tracing"):
                batch = tree_to_device(batch, device)
                influence = self.trace_batch(batch)
                results[:, count : count + influence.shape[1]] += influence
                count += influence.shape[1]
        return results

    def trace_dataloader(self, train_loader, test_loader, **kwargs):
        """
        We assume len(test_loader) < or = len(train_loader)
        """
        return self._trace_dataloader(train_loader, test_loader)

    @property
    def device(self):
        return self.grad_extractor.device

    def __str__(self) -> str:
        return "GD"

class TracIn(GradientBasedTracer):
    def __init__(self, grad_extractor: GradientExtractor, ckpt_paths) -> None:
        super().__init__(grad_extractor)
        self.ckpt_paths = Path(ckpt_paths).glob("*.ckpt")

    def trace_dataloader(self, train_loader, test_loader, **kwargs):
        results = torch.zeros(
            len(test_loader.dataset), len(train_loader.dataset), dtype=float
        ).to(self.device)
        for p in self.ckpt_paths:
            self.grad_extractor.load_state_dict(state_dict=torch.load(p)["state_dict"])
            super()._trace_dataloader(train_loader, test_loader, results)
        return results

    def __str__(self) -> str:
        return "TracIn"


class IF(GradientBasedTracer):
    def __init__(
        self,
        grad_extractor: GradientExtractor,
        damp=0.01,
        scale=25,
        recursion_depth=10,
        r=1,
        use_exact_hessian=False,
    ) -> None:
        super().__init__(grad_extractor)
        self.damp = damp
        self.scale = scale
        self.recursion_depth = recursion_depth
        self.r = r
        self.use_exact_hessian = use_exact_hessian

    def _calc_s_test_single(self, batch, train_loader, V=None):
        avg_s_test = 0
        # assert self.r == 1 or (self.r > 1 and train_loader.shuffle)
        for i in range(self.r):
            s_test = self.grad_extractor.s_test(
                ref_dloader=train_loader,
                batch=batch,
                V=V,
                damp=self.damp,
                scale=self.scale,
                recursion_depth=self.recursion_depth,
            )
            avg_s_test += s_test
        avg_s_test /= self.r
        return avg_s_test

    def _trace_dataloader(
        self, train_loader, test_loader, *, shuffled_train_loader=None
    ):
        if shuffled_train_loader is None and self.r > 1:
            raise ValueError("R > 1 is nonsense without shuffled train loader")
        device = self.device
        out_shape = (len(test_loader.dataset), len(train_loader.dataset))

        results = torch.zeros(size=out_shape, dtype=float).to(device)
        assert results.shape == out_shape

        if self.use_exact_hessian:
            raise NotImplemented
        else:
            g_test_all = []
            test_pbar = tqdm(
                test_loader,
                desc="Loop test loader",
                total=len(test_loader),
            )
            for batch in test_pbar:
                batch = tree_to_device(batch, device)
                g_test = self._calc_s_test_single(batch, shuffled_train_loader)
                g_test_all.append(g_test)
            g_test_all = torch.concat(g_test_all, dim=0)

            train_pbar = tqdm(self.grad_extractor.get_iter(train_loader),
                             desc="Loop train loader",
                             total=len(train_loader))
            count_col = 0
            for train_batch, grads_train in train_pbar:
                
                influence = self._influence_from_grad(g_test_all, grads_train)
                results[:, count_col : count_col + influence.shape[1]] += influence
                count_col += influence.shape[1]
        return results

    def trace_dataloader(
        self, train_loader, test_loader, *, shuffled_train_loader=None, **kwargs
    ):
        return self._trace_dataloader(
            train_loader, test_loader, shuffled_train_loader=shuffled_train_loader
        )

    def __str__(self) -> str:
        return (
            f"{IF}_scale={self.scale}_recursion_depth={self.recursion_depth}_r={self.r}"
        )

class GradientNormalize(GradientBasedTracer):
    def _influence_from_grad(self, grad_ref, grad_trace):
        # grad_ref, grad_trace = grad_trace, grad_ref
        grad_ref_norm = grad_ref / torch.clamp(grad_ref.norm(dim=1, keepdim=True), min=1e-8)
        return grad_ref_norm @ grad_trace.T 

class GradientCosin(GradientBasedTracer):
    def _influence_from_grad(self, grad_ref, grad_trace):
        grad_ref_norm = grad_ref / torch.clamp(grad_ref.norm(dim=1)[:, None], min=1e-8)
        grad_trace_norm = grad_trace / torch.clamp(grad_trace.norm(dim=1)[:, None], min=1e-8)
        return grad_ref_norm @ grad_trace_norm.T

class KNN(Tracer):
    def __init__(self, net: torch.nn.Module, neighbor_matrices: torch.Tensor, k:int) -> None:
        self.neighbor_matrices = neighbor_matrices[:, :k]
        self.net = net
        self.net.eval()
        self._device = next(self.net.parameters()).device

    def _trace_dataloader(self, train_loader, label_from_batch, test_loader=None, **kwargs):
        labels = []
        with torch.no_grad():
            for batch in train_loader:
                y = label_from_batch(batch)
                labels.append(y)
        labels = torch.concat(labels).squeeze_()
        
        neighbor_labels = labels[self.neighbor_matrices]
        numDiff = (neighbor_labels != labels[:, None]).sum(dim=1)
        numDiff = numDiff / self.neighbor_matrices.shape[1]
        return numDiff
    
    @property
    def device(self):
        return self._device

class KNNGD(GradientBasedTracer):
    def __init__(self, grad_extractor: GradientExtractor, feature_extractor=None, neighbor_matrices: torch.Tensor = None) -> None:
        super().__init__(grad_extractor)
        self.neighbor_matrices = neighbor_matrices
        self.feature_extractor = feature_extractor
        self.k = neighbor_matrices.shape[1]
    
    # def setup(self, dataloader):
    #     device = self.device
    #     features = []
    #     labels = []
    #     with torch.no_grad():
    #         for batch in dataloader:
    #             batch = tree_to_device(batch, device)
    #             feat, y = self.feature_extractor(batch)
    #             features.append(feat.cpu())
    #             labels.append(y.cpu())
    #     features = torch.concat(features).numpy()
    #     labels = torch.concat(labels).numpy()
    #     self.neigh = KNeighborsClassifier(n_neighbors=1000, metric='minkowski', algorithm='kd_tree', n_jobs=-1)
    #     self.neigh.fit(features, labels)
    def similarity_measure(self, grad_ref, grad_train):
        return (grad_ref * grad_train).sum(dim=1).unsqueeze(0)


    def _trace_dataloader(self, train_loader, test_loader, result_out=None, **kwargs):
        device = self.device
        self.setup(train_loader)
        train_grads = self._refferent_grads
        _refferent_grads = torch.zeros_like(train_grads)
        for i in range(self.neighbor_matrices.shape[1]):
            _refferent_grads += train_grads[self.neighbor_matrices[:, i]]
        results = self.similarity_measure(_refferent_grads, train_grads)
        return results

class KNNGN(KNNGD):
    def similarity_measure(self, grad_ref, grad_train):
        grad_ref_norm = grad_ref / torch.clamp(grad_ref.norm(dim=1, keepdim=True), min=1e-8)
        return super().similarity_measure(grad_ref_norm, grad_train)

class KNNGC(KNNGD):
    def similarity_measure(self, grad_ref, grad_trace):
        grad_ref_norm = grad_ref / torch.clamp(grad_ref.norm(dim=1, keepdim=True), min=1e-8)
        grad_trace_norm = grad_trace / torch.clamp(grad_trace.norm(dim=1)[:, None], min=1e-8)
        return super().similarity_measure(grad_ref_norm, grad_trace_norm)