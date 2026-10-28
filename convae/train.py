import contextlib
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

from .config import Config
from .data import build_splits, iter_batches
from .losses import step
from .model import build_model, model_info
from .viz import figure

try:
    import wandb
except ImportError:  # optional; wandb_mode=disabled works without it
    wandb = None

PBAR_EVERY = 50   # steps between progress-bar loss refreshes (each one is a host sync)


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def lr_lambda(cfg, total_steps):
    """Linear warmup over `warmup_steps`, then cosine to 0 or flat for the rest of the run."""
    warm = min(cfg.warmup_steps, total_steps)

    def f(s):
        if s < warm:
            return (s + 1) / warm
        if cfg.decay == "none":
            return 1.0
        return 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total_steps - warm, 1)))
    return f


class Meter:
    """Epoch means of per-batch stats, weighted by batch size; one host sync in result()."""

    def __init__(self):
        self.sums, self.n = {}, 0

    def add(self, stats, n):
        for k, v in stats.items():
            self.sums[k] = self.sums.get(k, 0) + v.detach().float() * n
        self.n += n

    def result(self):
        return {k: (v / self.n).item() for k, v in self.sums.items()}


@torch.no_grad()
def evaluate(model, split, cfg):
    model.eval()
    meter = Meter()
    for idx in iter_batches(len(split), cfg.batch_size):
        loss, stats = step(model, split.batch(idx))
        meter.add({"loss": loss, **stats}, len(idx))
    model.train()
    return meter.result()


def train(cfg: Config):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = pick_device(cfg.device)
    cuda = device.type == "cuda"
    if cuda:
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if cuda and cfg.amp else contextlib.nullcontext()
    out_dir = Path(cfg.out_dir) / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.dump(out_dir / "config.yaml")
    print(f"device={device} amp={cuda and cfg.amp} compile={cuda and cfg.compile} -> {out_dir}")

    # ---- data, model
    splits = build_splits(cfg, device)
    train_split = splits["train"]
    eval_splits = {k: v for k, v in splits.items() if k != "train"}
    model = build_model(cfg).to(device)
    info = {**model_info(model), **{f"{k}_samples": len(s) for k, s in splits.items()}}
    print(json.dumps(info, indent=2))

    # ---- optimiser, schedule, checkpoint init, compile
    steps_per_epoch = len(train_split) // cfg.batch_size
    assert steps_per_epoch > 0, "batch_size larger than the training set"
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, fused=cuda)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(cfg, cfg.epochs * steps_per_epoch))
    if cfg.init_from:
        ckpt = torch.load(cfg.init_from, map_location=device)
        model.load_state_dict(ckpt["model"])
        if not cfg.weights_only:
            optimizer.load_state_dict(ckpt["optimizer"])
        print(f"initialised from {cfg.init_from} (epoch {ckpt['epoch']}, optimizer {'cold' if cfg.weights_only else 'restored'})")
    run_model = torch.compile(model) if cuda and cfg.compile else model

    # ---- logging
    use_wandb = cfg.wandb_mode != "disabled"
    if use_wandb:
        assert wandb is not None, "wandb is not installed; pip install wandb or set wandb_mode=disabled"
        wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity, name=cfg.name, group=cfg.wandb_group,
                   tags=cfg.wandb_tags, mode=cfg.wandb_mode, config={**cfg.to_dict(), **info})
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")

    def save(path, epoch, **extra):
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "config": cfg.to_dict(), "info": info, "epoch": epoch, **extra}, path)

    better = (lambda a, b: a > b) if cfg.select_mode == "max" else (lambda a, b: a < b)
    best = best_epoch = None
    history = []
    perm_gen = torch.Generator().manual_seed(cfg.seed)
    t_start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        meter, t0 = Meter(), time.time()
        pbar = tqdm(total=steps_per_epoch, desc=f"epoch {epoch}/{cfg.epochs}", leave=False)
        for i, idx in enumerate(iter_batches(len(train_split), cfg.batch_size, shuffle=True,
                                             drop_last=True, generator=perm_gen)):
            with autocast:
                loss, stats = step(run_model, train_split.batch(idx))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            meter.add({"loss": loss, **stats}, len(idx))
            if i % PBAR_EVERY == 0:
                pbar.set_postfix(loss=f"{loss.item():.5f}")
            pbar.update(1)
        pbar.close()
        if cuda:
            torch.cuda.synchronize()
        epoch_time = time.time() - t0

        row = {"epoch": epoch, "lr": scheduler.get_last_lr()[0], "epoch_time": epoch_time,
               "samples_per_sec": meter.n / epoch_time}
        row.update({f"train/{k}": v for k, v in meter.result().items()})
        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            with autocast:
                for name, split in eval_splits.items():
                    row.update({f"{name}/{k}": v for k, v in evaluate(model, split, cfg).items()})
            if cfg.select_metric in row and (best is None or better(row[cfg.select_metric], best)):
                best, best_epoch = row[cfg.select_metric], epoch
                save(out_dir / "best.pt", epoch, selected_on=cfg.select_metric, selected_value=best)
            row["best_epoch"] = best_epoch
        history.append(row)
        if use_wandb:
            wandb.log(row)
        losses = [k for k in row if k.endswith("/loss")]
        print(f"epoch {epoch:3d} | " + " | ".join(f"{k} {row[k]:.5f}" for k in losses)
              + f" | lr {row['lr']:.2e} | {epoch_time:.0f}s" + ("  *" if best_epoch == epoch else ""))

    # ---- final artefacts
    final = {**history[-1], **info, "best_value": best, "best_epoch": best_epoch, "total_time": time.time() - t_start}
    (out_dir / "metrics.json").write_text(json.dumps(final, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    save(out_dir / "final.pt", cfg.epochs)
    model.eval()
    with autocast:
        for name, split in splits.items():
            fig = figure(model, name, split, cfg.viz_samples)
            fig.savefig(out_dir / f"viz_{name}.png", dpi=80)
            if use_wandb:
                wandb.log({f"{name}/viz": wandb.Image(fig), "epoch": cfg.epochs})
            plt.close(fig)
    if use_wandb:
        wandb.run.summary.update(final)
        wandb.finish()
    print(f"done in {final['total_time'] / 60:.1f} min -> {out_dir}")
    return final
