"""The training loop. Experiment-agnostic: data, model, loss, metrics and plots come from the
Experiment; this file owns AdamW + OneCycle, AMP, torch.compile, channels-last, grad clipping,
running means, wandb logging, checkpoints and the run directory."""
import contextlib
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

from .config import Config, describe
from .data import iter_batches
from .experiment import load_experiment
from .logger import Logger


# ----------------------------------------------------------------------------- setup

def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_backend(device):
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def resolve_amp(cfg, device):
    if cfg.amp == "none":
        return None
    if cfg.amp == "bf16":
        return torch.bfloat16
    return torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else None


def resolve_compile(cfg, device):
    if cfg.compile in ("true", "false"):
        return cfg.compile == "true"
    return device.type == "cuda"


# ----------------------------------------------------------------------------- running means

class Meter:
    """Weighted running means of scalar tensors, kept on device; one host sync at the end."""

    def __init__(self):
        self.sums, self.n = {}, 0

    def add(self, stats: dict, n: int):
        for k, v in stats.items():
            v = v.detach().float() * n if torch.is_tensor(v) else float(v) * n
            self.sums[k] = self.sums[k] + v if k in self.sums else v
        self.n += n

    def result(self) -> dict:
        return {k: (v / max(self.n, 1)).item() if torch.is_tensor(v) else v / max(self.n, 1)
                for k, v in self.sums.items()}


@torch.no_grad()
def evaluate(exp, model, name, split, cfg, autocast):
    model.eval()
    exp.eval_start(name)
    meter = Meter()
    for idx in iter_batches(split, cfg.batch_size, shuffle=False, drop_last=False):
        batch = split.batch(idx)
        with autocast:
            stats = exp.eval_step(model, batch)
        meter.add(stats, exp.batch_size_of(batch))
    out = meter.result()
    out.update(exp.eval_end(name))
    model.train()
    return out


# ----------------------------------------------------------------------------- main loop

def train(cfg: Config):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = pick_device(cfg.device)
    configure_backend(device)
    amp_dtype = resolve_amp(cfg, device)
    use_compile = resolve_compile(cfg, device)
    autocast = torch.autocast(device.type, dtype=amp_dtype) if amp_dtype else contextlib.nullcontext()
    out_dir = Path(cfg.out_dir) / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.dump(out_dir / "config.yaml")
    print(f"device={device} amp={amp_dtype} compile={use_compile}")
    print(describe(cfg))

    # ---- experiment: data + model
    exp = load_experiment(cfg.experiment)(cfg, device)
    splits = exp.build_splits()
    train_split = splits["train"]
    eval_splits = {k: v for k, v in splits.items() if k != "train"}
    for k, s in splits.items():
        print(f"{k:12s} {len(s):8d} samples")

    model = exp.build_model().to(device)
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)
    info = dict(exp.model_info(model))
    info.update({f"{k}_samples": len(s) for k, s in splits.items()})
    print(json.dumps(info, indent=2))
    (out_dir / "model_info.json").write_text(json.dumps(info, indent=2))

    # ---- optimiser, schedule, compile
    steps_per_epoch = len(train_split) // cfg.batch_size
    assert steps_per_epoch > 0, "batch_size larger than the training set"
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                                  fused=(device.type == "cuda"))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr, total_steps=cfg.epochs * steps_per_epoch, pct_start=cfg.pct_start)

    run_model = torch.compile(model, mode=cfg.compile_mode) if use_compile else model
    logger = Logger(cfg, info)

    def save(path, epoch, extra=None):
        torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "info": info,
                    "epoch": epoch, **(extra or {})}, path)

    better = (lambda a, b: a > b) if cfg.select_mode == "max" else (lambda a, b: a < b)
    best, best_epoch = None, None
    history = []
    perm_gen = torch.Generator().manual_seed(cfg.seed)
    t_start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        meter = Meter()
        t0 = time.time()
        pbar = tqdm(total=steps_per_epoch, desc=f"epoch {epoch}/{cfg.epochs}", leave=False)
        for step, idx in enumerate(iter_batches(train_split, cfg.batch_size, shuffle=True,
                                                drop_last=True, generator=perm_gen)):
            batch = train_split.batch(idx)
            with autocast:
                loss, stats = exp.train_step(run_model, batch, epoch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            meter.add({"loss": loss, **stats}, exp.batch_size_of(batch))
            if step % cfg.log_every == 0:
                pbar.set_postfix(loss=f"{loss.item():.5f}")   # the only per-step host sync
            pbar.update(1)
        pbar.close()
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.time() - t0

        row = {"epoch": epoch, "lr": scheduler.get_last_lr()[0], "epoch_time": epoch_time,
               "samples_per_sec": steps_per_epoch * cfg.batch_size / epoch_time}
        row.update({f"train/{k}": v for k, v in meter.result().items()})
        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            for name, split in eval_splits.items():
                m = evaluate(exp, model, name, split, cfg, autocast)
                row.update({f"{name}/{k}": v for k, v in m.items()})
            if cfg.select_metric in row and (best is None or better(row[cfg.select_metric], best)):
                best, best_epoch = row[cfg.select_metric], epoch
                save(out_dir / "best.pt", epoch, {"selected_on": cfg.select_metric, "selected_value": best})
                print(f"  new best {cfg.select_metric} {best:.5f} @ epoch {epoch} -> best.pt")
            row["best_epoch"] = best_epoch
        history.append(row)
        logger.log(row)
        shown = [f"train/loss"] + [k for k in row if k.endswith("/loss") and not k.startswith("train/")]
        print(f"epoch {epoch:3d} | " + " | ".join(f"{k} {row[k]:.5f}" for k in shown if k in row)
              + f" | lr {row['lr']:.2e} | {epoch_time:.0f}s")

    # ---- final artefacts
    total_time = time.time() - t_start
    final = dict(history[-1])
    final.update(info)
    final.update(best_value=best, best_epoch=best_epoch, total_time=total_time)
    (out_dir / "metrics.json").write_text(json.dumps(final, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    save(out_dir / "final.pt", cfg.epochs)

    model.eval()
    with torch.no_grad(), autocast:
        for name, split in splits.items():
            for key, fig in exp.visualize(model, name, split).items():
                fig.savefig(out_dir / f"viz_{name}_{key}.png", dpi=80)
                logger.image(f"{name}/{key}", fig, cfg.epochs)
                plt.close(fig)

    logger.summary(final)
    logger.finish()
    print(f"done in {total_time / 60:.1f} min -> {out_dir}")
    return final
