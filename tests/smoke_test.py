"""End-to-end check on CPU with no wandb (~1 minute).

    python tests/smoke_test.py
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "tests" / "_scratch"


def run(name, *overrides):
    cmd = [sys.executable, "train.py", "--config", "configs/base.yaml", f"name={name}",
           f"out_dir={SCRATCH / 'runs'}", "epochs=2", "batch_size=8", "compile=false", "device=cpu",
           "n_train=48", "n_val=16", *overrides]
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)
    d = SCRATCH / "runs" / name
    for f in ("config.yaml", "history.json", "metrics.json", "best.pt", "final.pt", "viz_val.png"):
        assert (d / f).exists(), f
    return json.loads((d / "metrics.json").read_text()), json.loads((d / "history.json").read_text())


if __name__ == "__main__":
    shutil.rmtree(SCRATCH, ignore_errors=True)

    m, _ = run("smoke_default")
    for k in ("train/loss", "train/acc", "val/loss", "val/acc", "val/perfect_frame_frac", "best_epoch", "enc_params"):
        assert k in m, k
    assert m["train_samples"] == 48 and m["val_samples"] == 16

    m, _ = run("smoke_overrides", "latent=2", "data_on_cpu=true", "select_metric=val/loss", "select_mode=min")
    assert m["best_epoch"] in (1, 2)

    # constant lr, then an anneal continued from best.pt with the optimizer restored, then a cold fine-tune
    _, h = run("smoke_const", "decay=none", "warmup_steps=12", "epochs=4")   # 6 steps/epoch
    assert abs(h[-1]["lr"] - 1e-3) < 1e-9 and h[0]["lr"] < h[-1]["lr"], [r["lr"] for r in h]
    _, h = run("smoke_anneal", f"init_from={SCRATCH / 'runs' / 'smoke_const' / 'best.pt'}", "warmup_steps=0")
    assert h[-1]["lr"] < 1e-5, h[-1]["lr"]
    run("smoke_finetune", f"init_from={SCRATCH / 'runs' / 'smoke_const' / 'best.pt'}", "weights_only=true")

    subprocess.run([sys.executable, "sweep.py", "configs/sweeps/example.yaml", "--dry-run", "--only", "latent8", "epochs=1"],
                   cwd=ROOT, check=True)
    shutil.rmtree(SCRATCH)
    print("\nsmoke test passed")
