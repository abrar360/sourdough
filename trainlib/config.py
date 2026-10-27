"""Run configuration: core harness options + a free-form `params` dict for the experiment.

    python train.py --config configs/base.yaml name=x lr=3e-4 params.width=64 params.levels=[8,8]

Overrides use YAML syntax for values; dotted keys address nested params.
"""
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Config:
    # --- run bookkeeping ---
    name: str = "run"
    out_dir: str = "runs"
    seed: int = 0
    device: str = "auto"             # "auto" | "cuda" | "mps" | "cpu"

    # --- experiment ---
    experiment: str = "experiments.example:Experiment"   # "module.path:ClassName"
    params: dict = field(default_factory=dict)            # anything the experiment wants

    # --- optimisation (AdamW + OneCycle) ---
    epochs: int = 10
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    pct_start: float = 0.05          # OneCycle warmup fraction
    grad_clip: float = 1.0           # 0 disables
    amp: str = "auto"                # "auto" | "bf16" | "none"
    compile: str = "auto"            # "auto" | "true" | "false"
    compile_mode: str = "default"
    channels_last: bool = True

    # --- evaluation / checkpoints ---
    eval_every: int = 1              # epochs
    select_metric: str = "val/loss"  # which logged metric picks best.pt
    select_mode: str = "min"         # "min" | "max"
    log_every: int = 50              # steps, progress-bar refresh only

    # --- wandb ---
    wandb_project: str = "experiments"
    wandb_entity: str = None
    wandb_mode: str = "disabled"     # "online" | "offline" | "disabled"
    wandb_group: str = None
    wandb_tags: list = field(default_factory=list)

    def __post_init__(self):
        # YAML 1.1 reads "1e-4" (no dot) as a string; coerce scalar numerics
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if v is None:
                continue
            if f.type is float:
                setattr(self, f.name, float(v))
            elif f.type is int:
                setattr(self, f.name, int(v))
        self.wandb_tags = [str(t) for t in (self.wandb_tags or [])]
        self.params = dict(self.params or {})

    def to_dict(self):
        return dataclasses.asdict(self)

    def dump(self, path):
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))


def _parse_value(text: str):
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def _set_dotted(data: dict, key: str, value):
    parts = key.split(".")
    for p in parts[:-1]:
        data = data.setdefault(p, {})
        if not isinstance(data, dict):
            raise ValueError(f"cannot set {key!r}: {p!r} is not a mapping")
    data[parts[-1]] = value


def _deep_merge(base: dict, new: dict):
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(config_path=None, overrides=()):
    data = {}
    if config_path:
        data = yaml.safe_load(Path(config_path).read_text()) or {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must look like key=value, got {item!r}")
        key, value = item.split("=", 1)
        value = _parse_value(value)
        if key == "params" and isinstance(value, dict):
            _deep_merge(data.setdefault("params", {}), value)
        else:
            _set_dotted(data, key, value)
    unknown = set(data) - {f.name for f in dataclasses.fields(Config)}
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)} (experiment options go under params.*)")
    return Config(**data)


def describe(cfg: Config):
    return json.dumps(cfg.to_dict(), indent=2)
