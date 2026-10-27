# sourdough

Starter code for PyTorch experiments: take a bit for each new project, and the starter lives on.

A small, experiment-agnostic training harness. The harness owns what is the same from one
experiment to the next; an `Experiment` subclass owns what is not.

| harness (`trainlib/`) | experiment (`experiments/<name>.py`) |
|---|---|
| config file, `key=value` overrides, sweeps | data and splits |
| AdamW + OneCycle, grad clipping, bf16, `torch.compile` | model |
| epoch means, wandb logging, run directory | loss, stats, eval metrics |
| `best.pt` / `final.pt` selection | figures |

## Quick start

```bash
pip install -r requirements.txt
python train.py --config configs/base.yaml                        # the example experiment
python train.py --config configs/base.yaml name=x lr=3e-4 params.latent=8
python sweep.py configs/sweeps/example.yaml --dry-run
python tests/smoke_test.py
```

Every run writes `runs/<name>/` with `config.yaml`, `history.json` (one row per epoch),
`metrics.json` (last row, model info, best epoch), `best.pt`, `final.pt` and
`viz_<split>_<key>.png`.

## Writing an experiment

Copy `experiments/example.py` and override what you need; `trainlib/experiment.py` is the
full interface:

```python
class Experiment(trainlib.experiment.Experiment):
    def build_splits(self) -> dict                   # {"train": Split, "val": Split, ...}
    def build_model(self) -> nn.Module
    def train_step(self, model, batch, epoch) -> (loss, {"acc": tensor, ...})
    # optional
    def eval_step(self, model, batch) -> {"loss": ..., "acc": ...}   # default: train_step
    def evaluate(self, model, name, split) -> dict                   # default: mean of eval_step
    def model_info(self, model) -> dict                              # logged once
    def visualize(self, model, name, split) -> {"key": Figure}
```

Point `experiment:` in the YAML at the class. Experiment options go under `params:` and are
available as `self.p`; nothing else in the config changes.

**Splits** are anything with `__len__` and `batch(idx)`. `TensorSplit` keeps a dict of tensors
on the GPU, or pinned on the CPU with `store_device="cpu"` when they do not fit, and gathers
batches by index. Augmentation belongs in `train_step`.

**Stats** returned by `train_step` and `eval_step` are per-batch means as scalar tensors. The
harness averages them over the epoch, weighted by batch size, with one host sync per epoch, and
logs them as `train/<key>` and `<split>/<key>`. Override `evaluate` for metrics that are not
batch means.

**Selection.** `select_metric` names any logged `<split>/<key>`; `select_mode` is `min` or `max`.
Keep one split for selection and another, never used for it, as the clean test number.

**Sweeps.** A sweep file names a base config, a wandb group, `common` overrides, and a list of
runs; each run's keys are overrides on top of that. Finished runs are skipped.

## License

MIT, see `LICENSE`.
