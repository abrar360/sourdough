"""Splits: anything with a length and a `batch(idx)` method. `TensorSplit` covers the common case
of a dataset that fits in memory as a dict of tensors."""
import torch


class Split:
    """Interface. `index_device` is where the shuffled index tensor is drawn each epoch."""
    index_device = torch.device("cpu")

    def __len__(self):
        raise NotImplementedError

    def batch(self, idx: torch.Tensor):
        """idx: (B,) long tensor on `index_device` -> a batch (any object) on the compute device."""
        raise NotImplementedError


class TensorSplit(Split):
    """Whole split resident as a dict of tensors (first dim = sample) on `store_device`;
    batches are gathered and moved to `device`. Pinned host memory when storing on the CPU."""

    def __init__(self, tensors: dict, device, store_device=None):
        store_device = torch.device(store_device or device)
        self.device = torch.device(device)
        pin = store_device.type == "cpu" and self.device.type == "cuda"
        self.tensors = {k: (v.to(store_device) if not pin else v.pin_memory()) for k, v in tensors.items()}
        n = {v.shape[0] for v in self.tensors.values()}
        assert len(n) == 1, f"all tensors must share the sample dimension, got sizes {n}"
        self.n = n.pop()
        self.index_device = store_device

    def __len__(self):
        return self.n

    def batch(self, idx):
        return {k: v[idx].to(self.device, non_blocking=True) for k, v in self.tensors.items()}


def iter_batches(split: Split, batch_size, shuffle, drop_last, generator=None):
    """Yields index tensors on split.index_device for one pass over the split."""
    n = len(split)
    if shuffle:
        perm = torch.randperm(n, generator=generator).to(split.index_device)   # seeded on the CPU
    else:
        perm = torch.arange(n, device=split.index_device)
    stop = n - (n % batch_size) if drop_last else n
    for i in range(0, stop, batch_size):
        yield perm[i:i + batch_size]
