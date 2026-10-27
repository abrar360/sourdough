"""Run configuration: harness options + a free-form `params` dict for the experiment.

    python train.py --config configs/base.yaml name=x lr=3e-4 params.width=64 params.levels=[8,8]

Override values use YAML syntax; dotted keys address nested params.
"""
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Config:
    name: str = "run"
    out_dir: str = "runs"
    seed: int = 0
    device: str = "auto"             # "auto" | "cuda" | "mps" | "cpu"

    experiment: str = "experiments.example:Experiment"   # "module.path:ClassName"
    params: dict = field(default_factory=dict)            # anything the experiment wants

    epochs: int = 10
    batch_size: int = 16
    lr: float = 1e-3                 # AdamW peak lr under a OneCycle schedule
    weight_decay: float = 1e-4
    pct_start: float = 0.05          # OneCycle warmup fraction
    grad_clip: float = 1.0           # 0 disables
    amp: bool = True                 # bf16 autocast (CUDA only)
    compile: bool = True             # torch.compile (CUDA only)

    eval_every: int = 1              # epochs
    select_metric: str = "val/loss"  # logged "<split>/<key>" that picks best.pt
    select_mode: str = "min"         # "min" | "max"

    wandb_project: str = "experiments"
    wandb_entity: str = None
    wandb_mode: str = "disabled"     # "online" | "offline" | "disabled"
    wandb_group: str = None
    wandb_tags: list = field(default_factory=list)

    def __post_init__(self):
        for f in dataclasses.fields(self):      # YAML reads "1e-4" (no dot) as a string
            v = getattr(self, f.name)
            if v is not None and f.type in (int, float):
                setattr(self, f.name, f.type(v))

    def to_dict(self):
        return dataclasses.asdict(self)

    def dump(self, path):
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))


def _set_dotted(data: dict, key: str, value):
    *parents, leaf = key.split(".")
    for p in parents:
        data = data.setdefault(p, {})
    data[leaf] = value


def load_config(config_path=None, overrides=()):
    data = yaml.safe_load(Path(config_path).read_text()) or {} if config_path else {}
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"override must look like key=value, got {item!r}")
        _set_dotted(data, key, yaml.safe_load(value))
    unknown = set(data) - {f.name for f in dataclasses.fields(Config)}
    if unknown:
        raise ValueError(f"unknown config keys {sorted(unknown)}; experiment options go under params.*")
    return Config(**data)
