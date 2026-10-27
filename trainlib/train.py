"""The training loop. Experiment-agnostic: data, model, loss, metrics and plots come from the
Experiment; this file owns AdamW + OneCycle, AMP, torch.compile, grad clipping, running means,
wandb logging, checkpoints and the run directory."""
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

from .config import Config
from .data import iter_batches
from .experiment import load_experiment
from .logger import Logger

PBAR_EVERY = 50   # steps between progress-bar loss refreshes (each one is a host sync)


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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

    # ---- experiment: data + model
    exp = load_experiment(cfg.experiment)(cfg, device)
    splits = exp.build_splits()
    train_split = splits["train"]
    eval_splits = {k: v for k, v in splits.items() if k != "train"}
    model = exp.build_model().to(device)
    info = {**exp.model_info(model), **{f"{k}_samples": len(s) for k, s in splits.items()}}
    print(json.dumps(info, indent=2))

    # ---- optimiser, schedule, compile
    steps_per_epoch = len(train_split) // cfg.batch_size
    assert steps_per_epoch > 0, "batch_size larger than the training set"
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, fused=cuda)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr, total_steps=cfg.epochs * steps_per_epoch, pct_start=cfg.pct_start)
    run_model = torch.compile(model) if cuda and cfg.compile else model
    logger = Logger(cfg, info)

    def save(path, epoch, **extra):
        torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "info": info, "epoch": epoch, **extra}, path)

    better = (lambda a, b: a > b) if cfg.select_mode == "max" else (lambda a, b: a < b)
    best = best_epoch = None
    history = []
    perm_gen = torch.Generator().manual_seed(cfg.seed)
    t_start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        sums, n_seen, t0 = {}, 0, time.time()
        pbar = tqdm(total=steps_per_epoch, desc=f"epoch {epoch}/{cfg.epochs}", leave=False)
        for step, idx in enumerate(iter_batches(len(train_split), cfg.batch_size, shuffle=True,
                                                drop_last=True, generator=perm_gen)):
            with autocast:
                loss, stats = exp.train_step(run_model, train_split.batch(idx), epoch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()

            for k, v in {"loss": loss, **stats}.items():
                sums[k] = sums.get(k, 0) + v.detach().float() * len(idx)
            n_seen += len(idx)
            if step % PBAR_EVERY == 0:
                pbar.set_postfix(loss=f"{loss.item():.5f}")
            pbar.update(1)
        pbar.close()
        if cuda:
            torch.cuda.synchronize()
        epoch_time = time.time() - t0

        row = {"epoch": epoch, "lr": scheduler.get_last_lr()[0], "epoch_time": epoch_time,
               "samples_per_sec": n_seen / epoch_time}
        row.update({f"train/{k}": (v / n_seen).item() for k, v in sums.items()})
        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            model.eval()
            with autocast:
                for name, split in eval_splits.items():
                    row.update({f"{name}/{k}": v for k, v in exp.evaluate(model, name, split).items()})
            if cfg.select_metric in row and (best is None or better(row[cfg.select_metric], best)):
                best, best_epoch = row[cfg.select_metric], epoch
                save(out_dir / "best.pt", epoch, selected_on=cfg.select_metric, selected_value=best)
            row["best_epoch"] = best_epoch
        history.append(row)
        logger.log(row)
        losses = [k for k in row if k.endswith("/loss")]
        print(f"epoch {epoch:3d} | " + " | ".join(f"{k} {row[k]:.5f}" for k in losses)
              + f" | lr {row['lr']:.2e} | {epoch_time:.0f}s" + ("  *" if best_epoch == epoch else ""))

    # ---- final artefacts
    final = {**history[-1], **info, "best_value": best, "best_epoch": best_epoch, "total_time": time.time() - t_start}
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
    print(f"done in {final['total_time'] / 60:.1f} min -> {out_dir}")
    return final
