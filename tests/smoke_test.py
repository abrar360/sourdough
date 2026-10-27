"""End-to-end check of the harness on the example experiment (CPU, no wandb, ~1 minute).

    python tests/smoke_test.py
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "tests" / "_scratch"


def run(name, *overrides):
    cmd = [sys.executable, "train.py", "--config", "configs/base.yaml", f"name={name}",
           f"out_dir={SCRATCH / 'runs'}", "wandb_mode=disabled", "epochs=2", "batch_size=8",
           "compile=false", "device=cpu", "params.n_train=48", "params.n_val=16", *overrides]
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)
    d = SCRATCH / "runs" / name
    for f in ("config.yaml", "model_info.json", "history.json", "metrics.json", "best.pt", "final.pt", "viz_val_recon.png"):
        assert (d / f).exists(), f
    return json.loads((d / "metrics.json").read_text())


if __name__ == "__main__":
    shutil.rmtree(SCRATCH, ignore_errors=True)

    m = run("smoke_default")
    for k in ("train/loss", "train/acc", "val/loss", "val/acc", "val/perfect_frame_frac", "best_epoch", "enc_params"):
        assert k in m, k
    assert m["train_samples"] == 48 and m["val_samples"] == 16

    # nested override, data on CPU, min-mode selection
    m = run("smoke_overrides", "params.latent=2", "params.data_on_cpu=true",
            "select_metric=val/loss", "select_mode=min")
    assert m["latent_values"] == 2 * 12 * 16, m["latent_values"]
    assert m["best_epoch"] in (1, 2)

    # sweep runner: dry run only
    subprocess.run([sys.executable, "sweep.py", "configs/sweeps/example.yaml", "--dry-run"], cwd=ROOT, check=True)
    print("\nsmoke test passed")
