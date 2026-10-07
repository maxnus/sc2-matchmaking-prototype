"""CLI for simulating the round-robin matchmaker (AI Arena's current system)."""

import argparse
import logging
from functools import partial
from pathlib import Path

import pandas as pd

from sim.cli import add_common_args, run_and_write

from matchmaker import RoundRobinMatchmaker

_own_dir = Path(__file__).resolve().parent

logging.basicConfig(level=logging.INFO, format="%(message)s")


def make_matchmaker(n_divisions: int, seed: int) -> RoundRobinMatchmaker:
    """Round-robin is deterministic; `seed` only fits the common factory signature."""
    return RoundRobinMatchmaker(n_divisions=n_divisions)


def ladder_divisions(data_dir: Path) -> int:
    """How many divisions the real ladder had, from the bots' divisions in
    bots.csv; 3 if the file has no division column."""
    bots = pd.read_csv(data_dir / "bots.csv")
    if "division" not in bots.columns:
        return 3
    return bots.loc[(bots["active"] == True) & (bots["division"] > 0), "division"].nunique()


def add_total_rounds(sim, summary):
    summary["metrics"]["total_rounds"] = sim.matchmaker._round


def main():
    parser = argparse.ArgumentParser(description="Simulate the round-robin matchmaker (AI Arena's current system)")
    add_common_args(parser, _own_dir / "output")
    parser.add_argument("--n-divisions", type=int,
                        help="Number of divisions (default: as many as the real ladder in bots.csv)")
    args = parser.parse_args()
    if args.n_divisions is None:
        args.n_divisions = ladder_divisions(args.data_dir)

    run_and_write(partial(make_matchmaker, args.n_divisions), args, extra_summary=add_total_rounds)


if __name__ == "__main__":
    main()
