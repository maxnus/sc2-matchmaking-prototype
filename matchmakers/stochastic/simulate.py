"""CLI for simulating the stochastic greedy matchmaker."""

import argparse
import logging
from pathlib import Path

import pandas as pd

from sim.cli import add_common_args, run_and_write

from matchmaker import StochasticMatchmaker, ScoringParams

_own_dir = Path(__file__).resolve().parent

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def _add_score_components(sim, summary):
    choice_df = pd.DataFrame(sim.matchmaker.choice_log)
    components = {}
    for comp in ["s_skill", "s_fair", "s_var"]:
        vals = choice_df[comp]
        components[comp] = {
            "mean": round(float(vals.mean()), 4),
            "std": round(float(vals.std()), 4),
        }
    total_mean = sum(v["mean"] for v in components.values())
    for comp in components:
        components[comp]["pct"] = round(components[comp]["mean"] / total_mean * 100, 1) if total_mean else 0
    summary["score_components"] = components


def main():
    parser = argparse.ArgumentParser(description="Simulate the stochastic greedy matchmaking system")
    add_common_args(parser, _own_dir / "output")
    # Scoring weights
    parser.add_argument("--w-skill", type=float, default=0.15)
    parser.add_argument("--w-fair", type=float, default=1.0)
    parser.add_argument("--w-var", type=float, default=0.2)
    # Scoring parameters
    parser.add_argument("--tau", type=float, default=15.0)
    parser.add_argument("--lam", type=float, default=40.0)
    parser.add_argument("--temperature", type=float, default=0.01,
                        help="Softmax sampling temperature (0 = argmax)")
    args = parser.parse_args()

    params = ScoringParams(
        w_skill=args.w_skill, w_fair=args.w_fair, w_var=args.w_var,
        tau=args.tau, lam=args.lam, temperature=args.temperature,
    )
    log.info("Scoring params: %s", params)
    matchmaker = StochasticMatchmaker(params, seed=args.mm_seed)

    run_and_write(matchmaker, args, extra_summary=_add_score_components)


if __name__ == "__main__":
    main()
