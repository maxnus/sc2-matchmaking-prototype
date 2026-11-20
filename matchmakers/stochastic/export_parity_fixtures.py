"""Export snapshots of the stochastic matchmaker's scoring, for AI Arena's tests to check against.

Each snapshot holds random inputs (ranks, games in the fairness window, games since two bots last
met) and the weighted score components this prototype computes for every pair, under several
parameter settings. Parameters are written under AI Arena's field names.

    python matchmakers/stochastic/export_parity_fixtures.py <output.json>
"""

import argparse
import json

import numpy as np

from matchmaker import ScoringParams, f_fair, f_skill, f_var


def _snapshot(rng: np.random.Generator, n: int, params: ScoringParams) -> dict:
    ranks = rng.permutation(n)
    games = rng.integers(0, int(3 * params.window_games) + 1, size=n).astype(float)
    # Games since two bots last met; a bot with no games has infinity towards everyone, and
    # everyone has its total towards it, as in the prototype.
    since = rng.integers(0, 120, size=(n, n)).astype(float)
    no_games = rng.random(n) < 0.15
    since[no_games, :] = np.inf
    since[:, no_games] = rng.integers(1, 120, size=(n, no_games.sum()))
    since[no_games[:, None] & no_games[None, :]] = np.inf

    a, b = np.triu_indices(n, 1)
    skill = params.w_skill * f_skill(ranks[a], ranks[b], params.tau)
    fairness = params.w_fair * f_fair(games[a], games[b], games.sum() / len(games))
    variety = params.w_var * f_var(since[a, b], since[b, a], params.lam)
    return {
        "ranks": ranks.tolist(),
        "games": games.tolist(),
        "since": since.tolist(),
        "pairs": {"a": a.tolist(), "b": b.tolist()},
        "expected": {"skill": skill.tolist(), "fairness": fairness.tolist(), "variety": variety.tolist()},
        "best": int(np.argmax(skill + fairness + variety)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output")
    args = parser.parse_args()

    rng = np.random.default_rng(2026)
    settings = [
        ScoringParams(),
        ScoringParams(w_skill=0.3, w_fair=0.5, w_var=0.4, tau=5.0, lam=10.0, window_games=20.0),
    ]
    fixtures = []
    for params in settings:
        fixtures.append({
            "params": {
                "skill_weight": params.w_skill,
                "fairness_weight": params.w_fair,
                "variety_weight": params.w_var,
                "skill_tolerance_ranks": params.tau,
                "variety_memory_games": params.lam,
                "sampling_temperature": params.temperature,
                "fairness_window_games": params.window_games,
            },
            "snapshots": [_snapshot(rng, n, params) for n in (2, 3, 12, 40)],
        })
    with open(args.output, "w") as f:
        json.dump(fixtures, f)  # infinity is written as Infinity, which Python's json reads back


if __name__ == "__main__":
    main()
