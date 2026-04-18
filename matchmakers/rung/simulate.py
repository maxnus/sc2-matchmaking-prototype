"""CLI for simulating the rung matchmaker."""

import argparse
import logging
from pathlib import Path

from sim.cli import add_common_args, run_and_write

from matchmaker import RungMatchmaker

_own_dir = Path(__file__).resolve().parent

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main():
    parser = argparse.ArgumentParser(description="Simulate the rung matchmaker")
    add_common_args(parser, _own_dir / "output")
    parser.add_argument("--rung-size", type=int, default=20)
    parser.add_argument("--rung-picks", type=int, default=8)
    parser.add_argument("--wildcard-picks", type=int, default=2)
    args = parser.parse_args()

    matchmaker = RungMatchmaker(
        rung_size=args.rung_size,
        rung_picks=args.rung_picks,
        wildcard_picks=args.wildcard_picks,
        seed=args.mm_seed,
    )

    def add_total_rounds(sim, summary):
        summary["total_rounds"] = sim.matchmaker._round

    run_and_write(matchmaker, args, extra_summary=add_total_rounds)


if __name__ == "__main__":
    main()
