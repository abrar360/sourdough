"""The experiment interface. Subclass it, override what you need, and point `experiment:` in the
config at your class. The harness (trainlib.train) owns the optimiser, schedule, AMP, compile,
logging, checkpoints and sweeps; the experiment owns data, model, loss, metrics and plots."""
import importlib

import torch
import torch.nn as nn

from .data import iter_batches


class Experiment:
    def __init__(self, cfg, device: torch.device):
        self.cfg = cfg
        self.p = cfg.params          # experiment options from the YAML
        self.device = device

    # ---- required ---------------------------------------------------------------------------

    def build_splits(self) -> dict:
        """-> {"train": Split, <eval name>: Split, ...}. Every split except "train" is evaluated
        every `eval_every` epochs and logged under "<name>/<key>"."""
        raise NotImplementedError

    def build_model(self) -> nn.Module:
        raise NotImplementedError

    def train_step(self, model, batch, epoch: int):
        """-> (loss, stats). `loss` is backpropagated. `stats` holds per-batch means as scalar
        tensors; the harness averages them over the epoch (weighted by batch size) and logs them
        as "train/<key>". Runs under autocast: cast model outputs to float before the loss."""
        raise NotImplementedError

    # ---- optional ---------------------------------------------------------------------------

    def eval_step(self, model, batch) -> dict:
        """-> per-batch means, same convention as train_step. Default: the training loss and stats."""
        loss, stats = self.train_step(model, batch, epoch=-1)
        return {"loss": loss, **stats}

    @torch.no_grad()
    def evaluate(self, model, name: str, split) -> dict:
        """One pass over an eval split -> {key: float}. Override for metrics that are not batch
        means (histograms, code usage, ...)."""
        sums, n = {}, 0
        for idx in iter_batches(len(split), self.cfg.batch_size):
            for k, v in self.eval_step(model, split.batch(idx)).items():
                sums[k] = sums.get(k, 0) + v.detach().float() * len(idx)
            n += len(idx)
        return {k: (v / n).item() for k, v in sums.items()}

    def model_info(self, model) -> dict:
        """Static facts about the model, logged to wandb config/summary and metrics.json."""
        return {"params": sum(p.numel() for p in model.parameters())}

    def visualize(self, model, name: str, split) -> dict:
        """-> {key: matplotlib Figure}, saved and logged at the end of training. Runs under no_grad."""
        return {}


def load_experiment(path: str):
    """'package.module:ClassName' -> class."""
    module, _, cls = path.partition(":")
    if not cls:
        raise ValueError(f"experiment must look like 'module.path:ClassName', got {path!r}")
    return getattr(importlib.import_module(module), cls)
