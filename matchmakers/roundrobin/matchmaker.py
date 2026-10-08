"""Round-robin matchmaker (AI Arena's current system).

Bots are sorted by ELO into `n_divisions` equal bands. Every pair within a
band plays once per round. Each round re-assigns the divisions by the ELO at
its start. As on AI Arena, the next round starts as soon as no match of the
current one can start, while its last matches still run (see
`sim/rounds.py`).
"""

from itertools import combinations

import pandas as pd

from sim.rounds import RoundMatchmaker


def _assign_divisions(
    bot_ids: list[int], ratings: dict[int, float], n_divisions: int,
) -> list[list[int]]:
    sorted_bots = sorted(bot_ids, key=lambda b: ratings[b], reverse=True)
    n = len(sorted_bots)
    base_size = n // n_divisions
    remainder = n % n_divisions
    divisions = []
    start = 0
    for i in range(n_divisions):
        size = base_size + (1 if i < remainder else 0)
        divisions.append(sorted_bots[start:start + size])
        start += size
    return divisions


class RoundRobinMatchmaker(RoundMatchmaker):
    """Round-robin within divisions, re-assigned every round."""

    def __init__(self, n_divisions: int = 3, max_active_rounds: int = 2, seed=None):
        super().__init__(max_active_rounds, seed)
        self.n_divisions = n_divisions

    def _draw_round(self, bots: pd.DataFrame) -> list[tuple[int, int]]:
        ratings = dict(zip(bots["bot_id"], bots["elo"]))
        bot_ids = bots["bot_id"].tolist()
        divisions = _assign_divisions(bot_ids, ratings, self.n_divisions)
        return [
            (a, b)
            for div_bots in divisions
            for a, b in combinations(div_bots, 2)
        ]
