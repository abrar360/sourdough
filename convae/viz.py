import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from .model import NUM_CLASSES


@torch.no_grad()
def figure(model, name, split, n):
    gt = split.batch(torch.arange(min(n, len(split))))["gt"].long()
    pred = model(F.one_hot(gt, NUM_CLASSES).permute(0, 3, 1, 2).float()).argmax(1)
    gt, pred = gt.cpu().numpy(), pred.cpu().numpy()
    cmap = plt.get_cmap("tab20", NUM_CLASSES)
    fig, axes = plt.subplots(len(gt), 3, figsize=(12, 3.2 * len(gt)), squeeze=False)
    for r in range(len(gt)):
        axes[r, 0].imshow(gt[r], cmap=cmap, vmin=0, vmax=NUM_CLASSES - 1, interpolation="nearest")
        axes[r, 0].set_title(f"{name} gt #{r}")
        axes[r, 1].imshow(pred[r], cmap=cmap, vmin=0, vmax=NUM_CLASSES - 1, interpolation="nearest")
        axes[r, 1].set_title(f"pred, acc {100 * (pred[r] == gt[r]).mean():.2f}%")
        axes[r, 2].imshow(pred[r] != gt[r], cmap="gray", interpolation="nearest")
        axes[r, 2].set_title("disagreement")
        for ax in axes[r]:
            ax.axis("off")
    fig.tight_layout()
    return fig
