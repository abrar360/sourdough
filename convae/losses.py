import torch.nn.functional as F

from .model import NUM_CLASSES


def step(model, batch):
    """-> (loss, stats). stats are per-batch means as scalar tensors; train.py averages them
    over the epoch weighted by batch size."""
    gt = batch["gt"].long()
    x = F.one_hot(gt, NUM_CLASSES).permute(0, 3, 1, 2).float()
    logits = model(x).float()                     # loss in fp32 even under autocast
    correct = logits.argmax(1) == gt
    loss = F.cross_entropy(logits, gt)
    return loss, {"acc": correct.float().mean(),
                  "perfect_frame_frac": correct.flatten(1).all(1).float().mean()}
