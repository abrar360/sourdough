#!/usr/bin/env python
"""Train one run.

    python train.py --config configs/base.yaml name=my_run lr=3e-4 params.width=64

Harness options are top-level Config fields; experiment options go under params.* (YAML values).
"""
import argparse

from trainlib.config import load_config
from trainlib.train import train


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("overrides", nargs="*", help="key=value overrides (dotted keys for params)")
    args = ap.parse_args()
    train(load_config(args.config, args.overrides))


if __name__ == "__main__":
    main()
