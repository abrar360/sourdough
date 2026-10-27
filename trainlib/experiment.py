"""The experiment interface. Subclass, override what you need, point `experiment:` in the config
at your class. Everything the harness does not know about (data format, model, loss, metrics,
plots) lives here; everything it does know about (optimiser, schedule, AMP, compile, logging,
checkpoints, sweeps) stays in trainlib.train."""
import torch
import torch.nn as nn


class Experiment:
    def __init__(self, cfg, device: torch.device):
        self.cfg = cfg
        self.p = cfg.params          # experiment-specific options from the YAML
        self.device = device

    # ---- required -------------------------------------------------------------------------

    def build_splits(self) -> dict:
        """-> {"train": Split, <eval name>: Split, ...}. "train" is required; every other split is
        evaluated every `eval_every` epochs and logged under "<name>/...". """
        raise NotImplementedError

    def build_model(self) -> nn.Module:
        raise NotImplementedError

    def train_step(self, model, batch, epoch: int):
        """-> (loss, stats). `loss` is backpropagated. `stats` is a dict of scalar tensors that are
        per-batch means; the harness averages them over the epoch weighted by batch size and logs
        them as "train/<key>". Runs inside autocast; cast model outputs to float for the loss."""
        raise NotImplementedError

    # ---- optional -------------------------------------------------------------------------

    def model_info(self, model) -> dict:
        """Static facts about the model (parameter counts, bitrate, ...) logged to wandb config/summary."""
        return {"params": sum(p.numel() for p in model.parameters())}

    def eval_step(self, model, batch) -> dict:
        """-> stats dict (per-batch means, same convention as train_step). Default: the train loss."""
        loss, stats = self.train_step(model, batch, epoch=-1)
        return {"loss": loss.detach(), **stats}

    def eval_start(self, split_name: str):
        """Called before an eval pass; reset any custom accumulators kept on `self`."""

    def eval_end(self, split_name: str) -> dict:
        """Called after an eval pass; return extra metrics computed from custom accumulators."""
        return {}

    def visualize(self, model, split_name: str, split) -> dict:
        """-> {key: matplotlib Figure}; saved as PNG and logged as images at the end of training."""
        return {}

    def batch_size_of(self, batch) -> int:
        """How many samples a batch holds (weights the running means)."""
        if isinstance(batch, dict):
            return next(iter(batch.values())).shape[0]
        if isinstance(batch, (tuple, list)):
            return batch[0].shape[0]
        return batch.shape[0]


def load_experiment(path: str):
    """'package.module:ClassName' -> class."""
    import importlib

    module, _, cls = path.partition(":")
    if not cls:
        raise ValueError(f"experiment must look like 'module.path:ClassName', got {path!r}")
    return getattr(importlib.import_module(module), cls)
