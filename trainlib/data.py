"""A split is anything with `__len__` and `batch(idx)`. `TensorSplit` covers data that fits in
memory as a dict of tensors."""
import torch


class Split:
    def __len__(self):
        raise NotImplementedError

    def batch(self, idx: torch.Tensor):
        """idx: (B,) long tensor on the CPU -> a batch (any object) on the compute device."""
        raise NotImplementedError


class TensorSplit(Split):
    """Dict of tensors (first dim = sample) kept on `store_device`, gathered per batch onto `device`.
    Stored on the CPU for a CUDA device, the tensors are pinned so the copy is asynchronous."""

    def __init__(self, tensors: dict, device, store_device=None):
        self.device = torch.device(device)
        self.store_device = torch.device(store_device or device)
        pin = self.store_device.type == "cpu" and self.device.type == "cuda"
        self.tensors = {k: v.pin_memory() if pin else v.to(self.store_device) for k, v in tensors.items()}
        sizes = {v.shape[0] for v in self.tensors.values()}
        assert len(sizes) == 1, f"all tensors must share the sample dimension, got {sizes}"
        self.n = sizes.pop()

    def __len__(self):
        return self.n

    def batch(self, idx):
        idx = idx.to(self.store_device)
        return {k: v[idx].to(self.device, non_blocking=True) for k, v in self.tensors.items()}


def iter_batches(n, batch_size, shuffle=False, drop_last=False, generator=None):
    """Yields CPU index tensors for one pass over n samples."""
    perm = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    stop = n - n % batch_size if drop_last else n
    for i in range(0, stop, batch_size):
        yield perm[i:i + batch_size]
