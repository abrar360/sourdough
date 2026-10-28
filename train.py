#!/usr/bin/env python
"""Train one run.

    python train.py --config configs/base.yaml name=my_run lr=3e-4 latent=8

Any Config field can be overridden with key=value (values are parsed as YAML).
"""
import argparse

from segmap.config import load_config
from segmap.train import train


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("overrides", nargs="*", help="key=value overrides")
    args = ap.parse_args()
    train(load_config(args.config, args.overrides))


if __name__ == "__main__":
    main()
