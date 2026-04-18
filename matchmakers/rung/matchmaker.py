"""Rung matchmaker: each bot picks 10 opponents per round.

Per round, each bot independently draws:
  - `rung_picks` opponents at random from its "rung" (the `rung_size`
    closest bots by current ELO)
  - `wildcard_picks` opponents at random from bots outside the rung

The round's schedule is the set of unique canonical pairs formed by
everyone's picks. Strict round semantics: between rounds the matchmaker
returns `None` until all in-flight matches have completed (so the next
round's division is based on fully-applied ELO updates).

Per-bot match count per round is at least `rung_picks + wildcard_picks`
(each bot's own picks) and typically ~1.5–2× that due to reciprocal picks
from other bots. This isn't strictly "exactly 10 matches per bot"; a
constrained-matching construction would be needed for that.
"""

from typing import Optional

import numpy as np
import pandas as pd


def _canonical(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


class RungMatchmaker:
    def __init__(
        self,
        rung_size: int = 20,
        rung_picks: int = 8,
        wildcard_picks: int = 2,
        seed=None,
    ):
        self.rung_size = rung_size
        self.rung_picks = rung_picks
        self.wildcard_picks = wildcard_picks
        self.rng = np.random.default_rng(seed)
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
            # the last round's matches to complete before re-picking.
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

        pairs: set[tuple[int, int]] = set()
        for b in bot_ids:
            others = [o for o in bot_ids if o != b]
            # Sort by ELO distance — closest first.
            others.sort(key=lambda o: abs(ratings[o] - ratings[b]))
            rung = others[:self.rung_size]
            wildcard_pool = others[self.rung_size:]

            for opp in self._sample(rung, self.rung_picks):
                pairs.add(_canonical(int(b), int(opp)))
            for opp in self._sample(wildcard_pool, self.wildcard_picks):
                pairs.add(_canonical(int(b), int(opp)))

        self._queue = list(pairs)
        self.rng.shuffle(self._queue)

    def _sample(self, pool: list[int], k: int) -> list[int]:
        if len(pool) <= k:
            return list(pool)
        return list(self.rng.choice(pool, size=k, replace=False))
