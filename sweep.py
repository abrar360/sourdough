#!/usr/bin/env python
"""Run a list of configured runs one after another on this machine.

    python sweep.py configs/sweeps/example.yaml                 # run everything
    python sweep.py configs/sweeps/example.yaml --dry-run       # just print the commands
    python sweep.py configs/sweeps/example.yaml --only latent8 epochs=1   # subset + extra overrides

Sweep file format:
    base: configs/base.yaml        # starting config
    group: example-latent          # wandb group (defaults to the sweep file stem)
    common: {epochs: 10}           # overrides applied to every run
    runs:
      - name: latent8
        params: {latent: 8}        # nested params merge with the base config
        lr: 2.0e-3

Finished runs (runs/<name>/metrics.json exists) are skipped unless --force.
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
    ap.add_argument("--only", nargs="*", default=None, help="run only these names")
    ap.add_argument("extra", nargs="*", help="extra key=value overrides for every run")
    args = ap.parse_args()
    # `--only a b x=1` would swallow x=1 into --only; anything with '=' is an override
    if args.only:
        args.extra += [o for o in args.only if "=" in o]
        args.only = [o for o in args.only if "=" not in o]

    spec = yaml.safe_load(Path(args.sweep).read_text())
    base = spec.get("base", "configs/base.yaml")
    group = spec.get("group", Path(args.sweep).stem)
    common = spec.get("common", {})
    out_dir = Path(common.get("out_dir", "runs"))

    results = {}
    for run in spec["runs"]:
        run = dict(run)
        name = run.pop("name")
        if args.only and name not in args.only:
            continue
        if (out_dir / name / "metrics.json").exists() and not args.force:
            print(f"skip {name}: already finished")
            results[name] = "skipped"
            continue
        overrides = {**common, **run, "name": name, "wandb_group": group}
        cmd = [sys.executable, "train.py", "--config", base] + \
              [f"{k}={json.dumps(v)}" for k, v in overrides.items()] + list(args.extra)
        print("\n>>> " + " ".join(cmd), flush=True)
        if args.dry_run:
            continue
        rc = subprocess.call(cmd)
        results[name] = "ok" if rc == 0 else f"FAILED (rc={rc})"
    if not args.dry_run:
        print("\nsweep summary:")
        for k, v in results.items():
            print(f"  {k:30s} {v}")


if __name__ == "__main__":
    main()
