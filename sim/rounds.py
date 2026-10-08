"""Rounds as AI Arena runs them, shared by the round-based matchmakers.

A round is a list of pairs drawn up when the round starts. Pairs are handed
out oldest round first, the first pair whose bots are both free. Within a
round, pairs of two bots with bot data come first, then pairs with one, in
random order otherwise. When no unfinished round has a pair that can start
now, because its pairs have all started or wait for bots that are busy, the
next round starts, as long as fewer than `max_active_rounds` rounds are
unfinished. A round is finished once all of its matches have completed.

All of this follows AI Arena's 2026 Season 1. There, every round started
before the previous one had finished, and never more than two rounds ran at
once. Within a round, matches between two bots with bot
data started on average a fifth of the way through it, matches with one
such bot about halfway, and the rest near the end.
"""

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


def canonical(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


@dataclass(eq=False)
class _Round:
    number: int
    # Pairs not yet started, in the order they are handed out.
    pairs: list[tuple[int, int]]
    running: int = 0

    @property
    def finished(self) -> bool:
        return not self.pairs and self.running == 0


class RoundMatchmaker:
    """Base class; subclasses draw up each round's pairs in `_draw_round`."""

    def __init__(self, max_active_rounds: int = 2, seed=None):
        self.max_active_rounds = max_active_rounds
        self.rng = np.random.default_rng(seed)
        self._round = 0
        # Unfinished rounds, oldest first.
        self._rounds: list[_Round] = []
        # For each pair with running matches, the rounds they belong to, in
        # the order they started.
        self._running: dict[tuple[int, int], deque[_Round]] = defaultdict(deque)
        self._seen_mids: set[int] = set()

    def _draw_round(self, bots: pd.DataFrame) -> list[tuple[int, int]]:
        raise NotImplementedError

    def __call__(
        self,
        bots: pd.DataFrame,
        match_history: pd.DataFrame,
        available_bots: list[int],
    ) -> Optional[tuple[int, int]]:
        self._ingest_new_matches(match_history)
        available = set(available_bots)
        for rnd in self._rounds:
            pair = self._take(rnd, available)
            if pair is not None:
                return pair
        if len(self._rounds) >= self.max_active_rounds:
            return None
        rnd = self._start_round(bots)
        return self._take(rnd, available)

    def _start_round(self, bots: pd.DataFrame) -> _Round:
        self._round += 1
        data = dict(zip(bots["bot_id"], bots["bot_data_enabled"].astype(int)))
        pairs = self._draw_round(bots)
        self.rng.shuffle(pairs)
        pairs.sort(key=lambda pair: data[pair[0]] + data[pair[1]], reverse=True)
        rnd = _Round(self._round, pairs)
        self._rounds.append(rnd)
        return rnd

    def _take(self, rnd: _Round, available: set[int]) -> Optional[tuple[int, int]]:
        for i, (a, b) in enumerate(rnd.pairs):
            if a in available and b in available:
                del rnd.pairs[i]
                rnd.running += 1
                self._running[canonical(a, b)].append(rnd)
                return a, b
        return None

    def _ingest_new_matches(self, match_history: pd.DataFrame) -> None:
        """Count completions from the match_history rows not yet seen; new
        rows are appended at the end (see `sim/matchmaker.py`)."""
        match_ids = match_history["match_id"].to_numpy()
        new_start = len(match_ids)
        while new_start > 0 and int(match_ids[new_start - 1]) not in self._seen_mids:
            new_start -= 1
        new = match_history.iloc[new_start:]
        for mid, bot_a, bot_b in zip(new["match_id"], new["bot_a"], new["bot_b"]):
            self._seen_mids.add(int(mid))
            pair = canonical(int(bot_a), int(bot_b))
            # If two rounds have the same pair running, the earlier round is
            # credited first, which can close it one match early.
            rnd = self._running[pair].popleft()
            if not self._running[pair]:
                del self._running[pair]
            rnd.running -= 1
            if rnd.finished:
                self._rounds.remove(rnd)
