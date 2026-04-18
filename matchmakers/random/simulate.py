"""CLI for simulating a truly-random baseline matchmaker."""

import argparse
import logging
from pathlib import Path

import numpy as np

from sim.cli import add_common_args, run_and_write
from sim.common import load_model

from matchmaker import RandomMatchmaker

_own_dir = Path(__file__).resolve().parent

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main():
    parser = argparse.ArgumentParser(description="Simulate the random-baseline matchmaker")
    add_common_args(parser, _own_dir / "output")
    args = parser.parse_args()

    bots, gp, lookup = load_model(args.data_dir, args.model_dir)
    rng = np.random.default_rng(args.seed)
    matchmaker = RandomMatchmaker(rng)

    run_and_write(matchmaker, bots, gp, lookup, rng, args)


if __name__ == "__main__":
    main()
