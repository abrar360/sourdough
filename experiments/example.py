"""A complete, runnable example: a small convolutional autoencoder that reconstructs synthetic
class maps through a low-dimensional bottleneck, scored by pixel accuracy. It runs on a laptop
CPU in a minute.

To start a new experiment, copy this file and replace `build_splits` (data), `build_model`,
`train_step` (loss + stats) and `visualize`, then point `experiment:` in the config at it.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from trainlib.data import TensorSplit
from trainlib.experiment import Experiment


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


class ConvAE(nn.Module):
    """Stride-2 conv encoder to a `latent`-channel bottleneck, mirrored decoder to class logits."""

    def __init__(self, num_classes, widths, latent):
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

    def forward(self, x):
        return self.decoder(self.encoder(x))


class Experiment(Experiment):
    NUM_CLASSES = 5

    def build_splits(self):
        p = self.p
        store = "cpu" if p.get("data_on_cpu") else self.device   # cpu when the data outgrows the GPU

        def split(n, seed):   # replace with real loading: any dict of tensors with a shared first dim
            return TensorSplit({"gt": torch.from_numpy(synthetic_maps(n, p["H"], p["W"], seed))},
                               self.device, store_device=store)

        return {"train": split(p["n_train"], seed=0), "val": split(p["n_val"], seed=1000)}

    def build_model(self):
        return ConvAE(self.NUM_CLASSES, self.p["widths"], self.p["latent"])

    def model_info(self, model):
        return {"enc_params": sum(x.numel() for x in model.encoder.parameters()),
                "dec_params": sum(x.numel() for x in model.decoder.parameters())}

    def _forward(self, model, batch):
        gt = batch["gt"].long()
        x = F.one_hot(gt, self.NUM_CLASSES).permute(0, 3, 1, 2).float()
        return model(x).float(), gt

    def train_step(self, model, batch, epoch):
        logits, gt = self._forward(model, batch)
        correct = logits.argmax(1) == gt
        return F.cross_entropy(logits, gt), {"acc": correct.float().mean(),
                                             "perfect_frame_frac": correct.flatten(1).all(1).float().mean()}

    def visualize(self, model, name, split):
        n = min(self.p["viz_samples"], len(split))
        logits, gt = self._forward(model, split.batch(torch.arange(n)))
        gt, pred = gt.cpu().numpy(), logits.argmax(1).cpu().numpy()
        cmap = plt.get_cmap("tab20", self.NUM_CLASSES)
        fig, axes = plt.subplots(n, 3, figsize=(12, 3.2 * n), squeeze=False)
        for r in range(n):
            axes[r, 0].imshow(gt[r], cmap=cmap, vmin=0, vmax=self.NUM_CLASSES - 1, interpolation="nearest")
            axes[r, 0].set_title(f"{name} gt #{r}")
            axes[r, 1].imshow(pred[r], cmap=cmap, vmin=0, vmax=self.NUM_CLASSES - 1, interpolation="nearest")
            axes[r, 1].set_title(f"pred, acc {100 * (pred[r] == gt[r]).mean():.2f}%")
            axes[r, 2].imshow(pred[r] != gt[r], cmap="gray", interpolation="nearest")
            axes[r, 2].set_title("disagreement")
            for ax in axes[r]:
                ax.axis("off")
        fig.tight_layout()
        return {"recon": fig}
