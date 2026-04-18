"""Round-robin matchmaker implementing the current AI Arena system.

Bots are sorted by ELO into `n_divisions` equal bands. Every pair within a
band plays once per round. After all of a round's matches have completed,
divisions are re-assigned by current ELO and the next round begins.

Strict round semantics: between rounds the matchmaker returns `None` while
the last round's matches drain (so no round-N+1 dispatch overlaps with
round-N completions). This mirrors the sequential for-loop in the original
`simulate_current.py` where ELO updates from round N are fully applied
before division re-assignment.
"""

from itertools import combinations
from typing import Optional

import pandas as pd


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


class RoundRobinMatchmaker:
    """Strict round-robin with periodic division re-assignment."""

    def __init__(self, n_divisions: int = 3):
        self.n_divisions = n_divisions
        # Remaining pairs to dispatch in the current round.
        self._queue: list[tuple[int, int]] = []
        self._round = 0

    def __call__(
        self,
        bots: pd.DataFrame,
        match_history: pd.DataFrame,
        available_bots: list[int],
    ) -> Optional[tuple[int, int]]:
        if not self._queue:
            # Round-done signal: nothing is in flight. Otherwise wait for
            # the last round's matches to complete before re-dividing.
            if bots["in_match"].any():
                return None
            self._start_next_round(bots)

        available_set = set(available_bots)
        for i, (a, b) in enumerate(self._queue):
            if a in available_set and b in available_set:
                del self._queue[i]
                return a, b
        return None

    def _start_next_round(self, bots: pd.DataFrame) -> None:
        self._round += 1
        ratings = dict(zip(bots["bot_id"], bots["elo"]))
        bot_ids = bots["bot_id"].tolist()
        divisions = _assign_divisions(bot_ids, ratings, self.n_divisions)
        self._queue = [
            (a, b)
            for div_bots in divisions
            for a, b in combinations(div_bots, 2)
        ]
