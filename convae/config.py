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

    # data (synthetic class maps, see data.py)
    H: int = 96
    W: int = 128
    n_train: int = 256
    n_val: int = 32
    data_on_cpu: bool = False        # keep the dataset in pinned host memory instead of on the GPU

    # model
    widths: list = field(default_factory=lambda: [16, 32, 64])
    latent: int = 4

    # optimisation
    epochs: int = 10
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 100          # linear ramp from 0 to lr over this many optimizer steps
    decay: str = "cosine"            # "cosine" (to 0 after warmup) | "none" (flat)
    grad_clip: float = 1.0           # 0 disables
    amp: bool = True                 # bf16 autocast (CUDA only)
    compile: bool = True             # torch.compile (CUDA only)
    init_from: str = None            # checkpoint to continue from (weights + optimizer state)
    weights_only: bool = False       # init_from: load the weights but start the optimizer cold

    # evaluation / checkpoints
    eval_every: int = 1              # epochs
    select_metric: str = "val/acc"   # logged "<split>/<key>" that picks best.pt
    select_mode: str = "max"         # "min" | "max"
    viz_samples: int = 2

    # wandb
    wandb_project: str = "convae"
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


def load_config(config_path=None, overrides=()):
    data = yaml.safe_load(Path(config_path).read_text()) or {} if config_path else {}
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"override must look like key=value, got {item!r}")
        data[key] = yaml.safe_load(value)
    unknown = set(data) - {f.name for f in dataclasses.fields(Config)}
    if unknown:
        raise ValueError(f"unknown config keys {sorted(unknown)}")
    return Config(**data)
