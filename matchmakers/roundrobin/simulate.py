"""CLI for simulating the round-robin matchmaker (AI Arena's current system)."""

import argparse
import logging
from pathlib import Path

from sim.cli import add_common_args, run_and_write

from matchmaker import RoundRobinMatchmaker

_own_dir = Path(__file__).resolve().parent

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main():
    parser = argparse.ArgumentParser(description="Simulate the round-robin matchmaker (AI Arena's current system)")
    add_common_args(parser, _own_dir / "output")
    parser.add_argument("--n-divisions", type=int, default=3)
    args = parser.parse_args()

    matchmaker = RoundRobinMatchmaker(n_divisions=args.n_divisions)

    def add_total_rounds(sim, summary):
        summary["total_rounds"] = sim.matchmaker._round

    run_and_write(matchmaker, args, extra_summary=add_total_rounds)


if __name__ == "__main__":
    main()
