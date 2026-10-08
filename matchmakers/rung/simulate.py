"""CLI for simulating the rung matchmaker."""

import argparse
import logging
from functools import partial
from pathlib import Path

from sim.cli import add_common_args, run_and_write

from matchmaker import RungMatchmaker

_own_dir = Path(__file__).resolve().parent

logging.basicConfig(level=logging.INFO, format="%(message)s")


def add_total_rounds(sim, summary):
    summary["metrics"]["total_rounds"] = sim.matchmaker._round


def main():
    parser = argparse.ArgumentParser(description="Simulate the rung matchmaker")
    add_common_args(parser, _own_dir / "output")
    parser.add_argument("--rung-size", type=int, default=20)
    parser.add_argument("--rung-picks", type=int, default=8)
    parser.add_argument("--wildcard-picks", type=int, default=2)
    parser.add_argument("--max-active-rounds", type=int, default=2,
                        help="Rounds that may run at once; the next starts when no match of the current ones can (default: 2, as on AI Arena)")
    args = parser.parse_args()

    make_matchmaker = partial(
        RungMatchmaker,
        rung_size=args.rung_size,
        rung_picks=args.rung_picks,
        wildcard_picks=args.wildcard_picks,
        max_active_rounds=args.max_active_rounds,
    )
    run_and_write(make_matchmaker, args, extra_summary=add_total_rounds)


if __name__ == "__main__":
    main()
