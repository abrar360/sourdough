"""A complete, runnable example experiment: a small convolutional autoencoder that reconstructs
synthetic class maps through a low-dimensional bottleneck, scored by pixel accuracy.

It is deliberately tiny so the template runs on a laptop CPU in a minute. Copy this file, replace
`build_splits` (your data), `build_model` (your model), `train_step` / `eval_step` (your loss and
metrics) and `visualize` (your plots), then point `experiment:` in the config at the new class.
"""
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from trainlib.data import TensorSplit
from trainlib.experiment import Experiment


# ----------------------------------------------------------------------------- data

def synthetic_maps(n, H, W, num_classes, seed):
    """Blobby class maps that drift over time, like a crude driving scene. -> (n, H, W) uint8"""
    yy, xx = np.mgrid[0:H, 0:W]
    out = np.zeros((n, H, W), np.uint8)
    for t in range(n):
        m = np.zeros((H, W), np.uint8)                                       # 0 = sky
        m[yy > H * 0.45 + 3 * np.sin((t + seed) / 7)] = 1                    # road
        cx, cy = W * (0.3 + 0.4 * (t / n)), H * 0.55
        m[(xx - cx) ** 2 / 20 ** 2 + (yy - cy) ** 2 / 12 ** 2 < 1] = 2       # car
        m[(xx < W * 0.15) & (yy > H * 0.3)] = 3                              # sidewalk
        m[(xx > W * 0.8) & (yy < H * 0.5)] = 4 % num_classes                 # building
        out[t] = m
    return out


# ----------------------------------------------------------------------------- model

class ConvAE(nn.Module):
    """Stride-2 conv encoder to a `latent` channel bottleneck, mirrored decoder to class logits."""

    def __init__(self, num_classes, widths=(16, 32, 64), latent=4):
        super().__init__()
        enc, ch = [], num_classes
        for w in widths:
            enc += [nn.Conv2d(ch, w, 3, stride=2, padding=1), nn.GroupNorm(8, w), nn.SiLU()]
            ch = w
        self.encoder = nn.Sequential(*enc, nn.Conv2d(ch, latent, 1), nn.Tanh())
        dec, ch = [], latent
        for w in reversed(widths):
            dec += [nn.Upsample(scale_factor=2), nn.Conv2d(ch, w, 3, padding=1), nn.GroupNorm(8, w), nn.SiLU()]
            ch = w
        self.decoder = nn.Sequential(*dec, nn.Conv2d(ch, num_classes, 3, padding=1))
        self.downsample = 2 ** len(widths)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


# ----------------------------------------------------------------------------- experiment

class Experiment(Experiment):
    """params: num_classes, H, W, n_train, n_val, widths, latent"""

    def build_splits(self):
        p = self.p
        K, H, W = p.get("num_classes", 5), p.get("H", 96), p.get("W", 128)
        self.num_classes = K
        store = "cpu" if p.get("data_on_cpu", False) else self.device

        def split(n, seed):
            gt = torch.from_numpy(synthetic_maps(n, H, W, K, seed))
            return TensorSplit({"gt": gt}, self.device, store_device=store)

        # swap in real loading here (np.load / memmap / torch.load -> dict of tensors)
        return {"train": split(p.get("n_train", 256), seed=0),
                "val": split(p.get("n_val", 32), seed=1000)}

    def build_model(self):
        return ConvAE(self.num_classes, tuple(self.p.get("widths", (16, 32, 64))), self.p.get("latent", 4))

    def model_info(self, model):
        H, W = self.p.get("H", 96), self.p.get("W", 128)
        h, w = math.ceil(H / model.downsample), math.ceil(W / model.downsample)
        return {"enc_params": sum(x.numel() for x in model.encoder.parameters()),
                "dec_params": sum(x.numel() for x in model.decoder.parameters()),
                "latent_hw": [h, w], "latent_values": h * w * self.p.get("latent", 4)}

    def _forward(self, model, batch):
        gt = batch["gt"].long()
        x = F.one_hot(gt, self.num_classes).permute(0, 3, 1, 2).float()
        if self.cfg.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        logits, _ = model(x)
        return logits.float(), gt

    def train_step(self, model, batch, epoch):
        logits, gt = self._forward(model, batch)
        loss = F.cross_entropy(logits, gt)
        with torch.no_grad():
            acc = (logits.argmax(1) == gt).float().mean()
        return loss, {"acc": acc}

    def eval_step(self, model, batch):
        logits, gt = self._forward(model, batch)
        correct = logits.argmax(1) == gt
        self._perfect += correct.flatten(1).all(1).sum()          # custom accumulator (see eval_end)
        self._frames += gt.shape[0]
        return {"loss": F.cross_entropy(logits, gt), "acc": correct.float().mean()}

    def eval_start(self, name):
        self._perfect = torch.zeros((), device=self.device, dtype=torch.long)
        self._frames = 0

    def eval_end(self, name):
        return {"perfect_frame_frac": (self._perfect / max(self._frames, 1)).item()}

    def visualize(self, model, name, split):
        n = min(self.p.get("viz_samples", 2), len(split))
        idx = torch.arange(n, device=split.index_device)
        logits, gt = self._forward(model, split.batch(idx))
        pred = logits.argmax(1)
        gt, pred = gt.cpu().numpy(), pred.cpu().numpy()
        cmap = plt.get_cmap("tab20", self.num_classes)
        fig, axes = plt.subplots(n, 3, figsize=(12, 3.2 * n), squeeze=False)
        for r in range(n):
            acc = (pred[r] == gt[r]).mean()
            axes[r, 0].imshow(gt[r], cmap=cmap, vmin=0, vmax=self.num_classes - 1, interpolation="nearest")
            axes[r, 0].set_title(f"{name} gt #{r}")
            axes[r, 1].imshow(pred[r], cmap=cmap, vmin=0, vmax=self.num_classes - 1, interpolation="nearest")
            axes[r, 1].set_title(f"pred — acc {100 * acc:.2f}%")
            axes[r, 2].imshow(pred[r] != gt[r], cmap="gray", interpolation="nearest")
            axes[r, 2].set_title("disagreement")
            for ax in axes[r]:
                ax.axis("off")
        fig.tight_layout()
        return {"recon": fig}
