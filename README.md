# sourdough

Starter code for PyTorch experiments: take a bit for each new project, and the starter lives on.

A small, experiment-agnostic PyTorch training harness. The harness owns everything that is the
same from one experiment to the next; an `Experiment` subclass owns everything that is not.

| harness (`trainlib/`) | experiment (`experiments/<name>.py`) |
|---|---|
| config loading, `key=value` overrides, sweeps | data loading and splits |
| AdamW + OneCycle, grad clipping | model construction |
| bf16 autocast, `torch.compile`, channels-last, TF32 | loss and per-batch stats |
| running means, wandb logging, run directory | extra eval metrics needing custom state |
| `best.pt` / `final.pt` selection | figures |

## Quick start

```bash
pip install -r requirements.txt
python train.py --config configs/base.yaml                        # the example experiment
python train.py --config configs/base.yaml name=x lr=3e-4 params.latent=8
python sweep.py configs/sweeps/example.yaml --dry-run
python tests/smoke_test.py
```

Every run writes `runs/<name>/` with `config.yaml`, `model_info.json`, `history.json`
(one row per epoch), `metrics.json` (last row + model info + best epoch), `best.pt`, `final.pt`
and `viz_<split>_<key>.png`.

## Writing an experiment

Copy `experiments/example.py` and override the methods you need
(`trainlib/experiment.py` has the full interface with docstrings):

```python
class Experiment(trainlib.experiment.Experiment):
    def build_splits(self) -> dict:          # {"train": Split, "val": Split, ...}
    def build_model(self) -> nn.Module
    def train_step(self, model, batch, epoch) -> (loss, {"acc": tensor, ...})
    # optional
    def eval_step(self, model, batch) -> {"loss": ..., "acc": ...}   # default: train_step's loss
    def eval_start(self, name) / eval_end(self, name) -> dict        # custom accumulators
    def model_info(self, model) -> dict                                # logged to wandb config/summary
    def visualize(self, model, name, split) -> {"key": Figure}
```

Then set `experiment: experiments.<file>:Experiment` in the YAML. Experiment options go under
`params:` and are available as `self.p`; nothing else in the config needs to change.

**Splits.** Anything with `__len__` and `batch(idx)` (see `trainlib/data.py`). `TensorSplit`
keeps a dict of tensors resident on the GPU, or pinned on the CPU with `store_device="cpu"`
when it does not fit, and gathers batches by index. Augmentation belongs in `train_step`.

**Stats.** `train_step` and `eval_step` return per-batch means as scalar tensors. The harness
averages them over the epoch, weighted by batch size, and logs them as `train/<key>` and
`<split>/<key>` with no per-step host sync. Metrics that cannot be expressed as a batch mean
(histograms, code usage, per-frame flags) go through `eval_start` / `eval_end`.

**Selection.** `select_metric` names any logged `<split>/<key>`; `select_mode` is `min` or `max`.
Keep one split for selection and another, never used for it, as the clean test number.

**Sweeps.** A sweep file is a base config, a wandb group, `common` overrides, and a list of runs.
Nested `params` merge with the base rather than replacing it. Finished runs are skipped.

## Config reference

See `configs/base.yaml`; every top-level key is a field of `trainlib.config.Config`, and all of
them can be overridden on the command line with YAML-syntax values (`levels=[8,8]`,
`wandb_tags=[a,b]`, `params.flag=true`).

## License

MIT, see `LICENSE`.
