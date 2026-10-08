"""Rung matchmaker: each bot picks 10 opponents per round.

Per round, each bot independently draws:
  - `rung_picks` opponents at random from its "rung" (the `rung_size`
    closest bots by current ELO)
  - `wildcard_picks` opponents at random from bots outside the rung

The round's schedule is the set of unique canonical pairs formed by
everyone's picks, drawn from the ELO at the round's start. Rounds follow
AI Arena's: the next round starts as soon as no match of the current one can
start, while its last matches still run (see `sim/rounds.py`).

Per-bot match count per round is at least `rung_picks + wildcard_picks`
(each bot's own picks) and typically ~1.5–2× that due to reciprocal picks
from other bots. This isn't strictly "exactly 10 matches per bot"; a
constrained-matching construction would be needed for that.
"""

import pandas as pd

from sim.rounds import RoundMatchmaker, canonical


class RungMatchmaker(RoundMatchmaker):
    def __init__(
        self,
        rung_size: int = 20,
        rung_picks: int = 8,
        wildcard_picks: int = 2,
        max_active_rounds: int = 2,
        seed=None,
    ):
        super().__init__(max_active_rounds, seed)
        self.rung_size = rung_size
        self.rung_picks = rung_picks
        self.wildcard_picks = wildcard_picks

    def _draw_round(self, bots: pd.DataFrame) -> list[tuple[int, int]]:
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
                pairs.add(canonical(int(b), int(opp)))
            for opp in self._sample(wildcard_pool, self.wildcard_picks):
                pairs.add(canonical(int(b), int(opp)))

        return list(pairs)

    def _sample(self, pool: list[int], k: int) -> list[int]:
        if len(pool) <= k:
            return list(pool)
        return list(self.rng.choice(pool, size=k, replace=False))
