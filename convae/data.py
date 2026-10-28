"""Synthetic class maps, kept whole in memory and served as index-gathered batches."""
import numpy as np
import torch


def synthetic_maps(n, H, W, seed):
    """Blobby 5-class maps that drift over time, like a crude driving scene. -> (n, H, W) uint8"""
    yy, xx = np.mgrid[0:H, 0:W]
    out = np.zeros((n, H, W), np.uint8)
    for t in range(n):
        m = np.zeros((H, W), np.uint8)                                   # 0 = sky
        m[yy > H * 0.45 + 3 * np.sin((t + seed) / 7)] = 1                # road
        cx, cy = W * (0.3 + 0.4 * t / n), H * 0.55
        m[(xx - cx) ** 2 / 20 ** 2 + (yy - cy) ** 2 / 12 ** 2 < 1] = 2   # car
        m[(xx < W * 0.15) & (yy > H * 0.3)] = 3                          # sidewalk
        m[(xx > W * 0.8) & (yy < H * 0.5)] = 4                           # building
        out[t] = m
    return out


class Split:
    """A dict of tensors (first dim = sample) kept on `store_device`, gathered per batch onto `device`.
    Stored on the CPU for a CUDA device, the tensors are pinned so the copy is asynchronous."""

    def __init__(self, tensors, device, store_device=None):
        self.device = torch.device(device)
        self.store_device = torch.device(store_device or device)
        pin = self.store_device.type == "cpu" and self.device.type == "cuda"
        self.tensors = {k: v.pin_memory() if pin else v.to(self.store_device) for k, v in tensors.items()}
        self.n = next(iter(self.tensors.values())).shape[0]

    def __len__(self):
        return self.n

    def batch(self, idx):
        idx = idx.to(self.store_device)
        return {k: v[idx].to(self.device, non_blocking=True) for k, v in self.tensors.items()}


def build_splits(cfg, device):
    """-> {"train": Split, "val": Split}. Replace synthetic_maps with real loading (np.load, memmap, ...)."""
    store = "cpu" if cfg.data_on_cpu else device

    def split(n, seed):
        return Split({"gt": torch.from_numpy(synthetic_maps(n, cfg.H, cfg.W, seed))}, device, store)

    return {"train": split(cfg.n_train, seed=0), "val": split(cfg.n_val, seed=1000)}


def iter_batches(n, batch_size, shuffle=False, drop_last=False, generator=None):
    """Yields CPU index tensors for one pass over n samples."""
    perm = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    stop = n - n % batch_size if drop_last else n
    for i in range(0, stop, batch_size):
        yield perm[i:i + batch_size]
