#!/usr/bin/env python
"""Run a list of configured runs one after another.

    python sweep.py configs/sweeps/example.yaml                        # run everything
    python sweep.py configs/sweeps/example.yaml --dry-run              # print the commands
    python sweep.py configs/sweeps/example.yaml --only latent8 epochs=1   # subset, extra overrides

Sweep file:
    base: configs/base.yaml        # starting config
    group: example-latent          # wandb group (default: the sweep file stem)
    common: {epochs: 10}           # overrides applied to every run
    runs:
      - {name: latent8, latent: 8, lr: 2.0e-3}   # -> runs/latent8

Finished runs (metrics.json exists) are skipped unless --force.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", help="comma-separated run names")
    ap.add_argument("overrides", nargs="*", help="extra key=value overrides for every run")
    args = ap.parse_intermixed_args()

    spec = yaml.safe_load(Path(args.sweep).read_text())
    base = spec.get("base", "configs/base.yaml")
    common = spec.get("common", {})
    group = spec.get("group", Path(args.sweep).stem)
    only = args.only.split(",") if args.only else None
    results = {}
    for run in spec["runs"]:
        name = run["name"]
        if only and name not in only:
            continue
        if (Path(common.get("out_dir", "runs")) / name / "metrics.json").exists() and not args.force:
            results[name] = "skipped (finished)"
            continue
        overrides = {**common, **run, "wandb_group": group}
        cmd = [sys.executable, "train.py", "--config", base,
               *(f"{k}={json.dumps(v)}" for k, v in overrides.items()), *args.overrides]   # JSON is valid YAML
        print(">>> " + " ".join(cmd), flush=True)
        if not args.dry_run:
            rc = subprocess.call(cmd)
            results[name] = "ok" if rc == 0 else f"FAILED (rc={rc})"
    for name, status in results.items():
        print(f"{name:30s} {status}")


if __name__ == "__main__":
    main()
