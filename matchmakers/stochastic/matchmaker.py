"""Stochastic matchmaker: weighted skill/fairness/variety score with softmax sampling."""

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ScoringParams:
    w_skill: float = 0.1
    w_fair: float = 1.0
    w_var: float = 0.2
    tau: float = 15.0  # rank-difference tolerance (in rank positions)
    lam: float = 40.0
    temperature: float = 0.01  # softmax sampling temperature (0 = argmax)


# The score's components. Each takes the values of many pairs at once as
# arrays, one element per pair.


def f_skill(rank_a: np.ndarray, rank_b: np.ndarray, tau: float) -> np.ndarray:
    return 1.0 / (1.0 + ((rank_a - rank_b) / tau) ** 2)


def f_fair(g_a: np.ndarray, g_b: np.ndarray, g_bar: float) -> np.ndarray:
    if g_bar == 0:
        return np.zeros(len(g_a))
    d_a = np.maximum(g_bar - g_a, 0) / g_bar
    d_b = np.maximum(g_bar - g_b, 0) / g_bar
    return np.sqrt((d_a**2 + d_b**2) / 2)


def f_var(a_ab: np.ndarray, b_ab: np.ndarray, lam: float) -> np.ndarray:
    return 1.0 - np.exp(-np.sqrt(a_ab * b_ab) / lam)


class StochasticMatchmaker:
    """Event-driven matchmaker with rematch prevention and softmax sampling.

    Maintains internal bookkeeping, indexed by each bot's position in the
    sim's `bots` table:
      - `_games_24h[i]`: completion times of bot i's matches in the last 24h
        of simulated time (the clock is the latest completion seen)
      - `_in_flight[i]`: matches dispatched but not yet completed
      - `_in_flight_pairs[i, j]`: the same, per pair (symmetric)
      - `_gso[i, j]`: games dispatched for bot i since it was last dispatched
        against bot j (0 = rematch, -1 = never met)
      - `_total_games[i]`: cumulative dispatched matches per bot

    The sim fills every free slot in one go, before any of those matches
    completes. Pair history and in-flight counts are therefore updated when a
    pair is returned, not when it completes; otherwise the same pair, or the
    same underplayed bot, would be picked for several slots at once. This
    relies on the sim dispatching every pair the matchmaker returns.

    Rematch prevention: any pair that is in flight, or where either bot's
    most recently dispatched opponent was the other, is excluded from
    scoring.

    Every candidate pair is scored at once with numpy; a Python loop over
    pairs took most of a simulation's time on a ladder of 170 bots.
    """

    def __init__(self, params: ScoringParams, seed=None):
        self.params = params
        self.rng = np.random.default_rng(seed)
        self._ids: Optional[np.ndarray] = None  # bot ids, in the sim's order
        self._index: dict[int, int] = {}
        self._seen_mids: set[int] = set()
        self._now = 0.0
        # Per-dispatch diagnostics. Index aligns with the sim's `match_id`
        # (matchmaker is called once per dispatch; we append on success).
        self.choice_log: list[dict[str, float]] = []

    def _ensure_bots(self, bots: pd.DataFrame) -> None:
        """Set up the bookkeeping on the first call; the sim's bots don't change."""
        ids = bots["bot_id"].to_numpy()
        if self._ids is not None:
            if not np.array_equal(ids, self._ids):
                raise ValueError("the bots changed between calls")
            return
        n = len(ids)
        self._ids = ids
        self._index = {int(b): i for i, b in enumerate(ids)}
        self._games_24h: list[deque[float]] = [deque() for _ in range(n)]
        self._in_flight = np.zeros(n, dtype=int)
        self._in_flight_pairs = np.zeros((n, n), dtype=int)
        self._gso = np.full((n, n), -1, dtype=int)
        self._total_games = np.zeros(n, dtype=int)

    def _ingest_new_matches(self, match_history: pd.DataFrame) -> None:
        """Record completions from any match_history rows not yet seen.

        The sim may pass a time-windowed DataFrame (shorter than the full
        history), so "new" rows can't be detected by length. Instead we
        track seen `match_id`s and scan back from the tail — new rows are
        always appended at the end, so the scan stops quickly.
        """
        match_ids = match_history["match_id"].to_numpy()
        new_start = len(match_ids)
        while new_start > 0 and int(match_ids[new_start - 1]) not in self._seen_mids:
            new_start -= 1
        if new_start == len(match_ids):
            return
        new = match_history.iloc[new_start:]
        for mid, bot_a, bot_b, end_time in zip(
            new["match_id"].to_numpy(), new["bot_a"].to_numpy(),
            new["bot_b"].to_numpy(), new["time_end"].to_numpy(),
        ):
            a = self._index[int(bot_a)]
            b = self._index[int(bot_b)]
            end_time = float(end_time)
            self._seen_mids.add(int(mid))

            self._games_24h[a].append(end_time)
            self._games_24h[b].append(end_time)
            self._now = max(self._now, end_time)

            self._in_flight[[a, b]] -= 1
            self._in_flight_pairs[a, b] -= 1
            self._in_flight_pairs[b, a] -= 1

    def _expire_old_games(self) -> None:
        """Drop completions older than 24h from every bot's window.

        Pruning against the global clock (not just when a bot plays again)
        keeps the count of a bot that hasn't played lately from going stale.
        """
        cutoff = self._now - 24 * 60
        for times in self._games_24h:
            while times and times[0] <= cutoff:
                times.popleft()

    def _games_in_window(self) -> np.ndarray:
        """Matches counted for fairness, per bot: completed in the last 24h, plus in flight."""
        completed = np.fromiter((len(t) for t in self._games_24h), dtype=int, count=len(self._games_24h))
        return completed + self._in_flight

    def _record_dispatch(self, a: int, b: int) -> None:
        self._in_flight[[a, b]] += 1
        self._in_flight_pairs[a, b] += 1
        self._in_flight_pairs[b, a] += 1
        for bot, opp in ((a, b), (b, a)):
            self._total_games[bot] += 1
            gso = self._gso[bot]
            gso[gso >= 0] += 1
            gso[opp] = 0

    def _games_since(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Games each bot in `a` played since it last faced its partner in `b`;
        all its games if they never met, or infinity if it has none."""
        since = self._gso[a, b].astype(float)
        never = since < 0
        total = self._total_games[a[never]].astype(float)
        total[total == 0] = np.inf
        since[never] = total
        return since

    def _score(
        self, a: np.ndarray, b: np.ndarray, ranks: np.ndarray, games: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Weighted score components of the pairs (a[k], b[k])."""
        p = self.params
        return {
            "s_skill": p.w_skill * f_skill(ranks[a], ranks[b], p.tau),
            "s_fair": p.w_fair * f_fair(games[a], games[b], games.sum() / len(games)),
            "s_var": p.w_var * f_var(self._games_since(a, b), self._games_since(b, a), p.lam),
        }

    def __call__(
        self,
        bots: pd.DataFrame,
        match_history: pd.DataFrame,
        available_bots: list[int],
    ) -> Optional[tuple[int, int]]:
        self._ensure_bots(bots)
        self._ingest_new_matches(match_history)
        self._expire_old_games()

        # Rank by rating, best first; ties by bot id.
        order = np.lexsort((self._ids, -bots["elo"].to_numpy()))
        ranks = np.empty(len(order), dtype=int)
        ranks[order] = np.arange(len(order))

        # Candidate pairs, in the order itertools.combinations would give them.
        available = np.array([self._index[int(b)] for b in available_bots])
        i, j = np.triu_indices(len(available), 1)
        a, b = available[i], available[j]
        allowed = (self._in_flight_pairs[a, b] == 0) & (self._gso[a, b] != 0) & (self._gso[b, a] != 0)
        a, b = a[allowed], b[allowed]
        if len(a) == 0:
            return None

        components = self._score(a, b, ranks, self._games_in_window())
        scores = components["s_skill"] + components["s_fair"] + components["s_var"]

        if self.params.temperature > 0:
            logits = (scores - scores.max()) / self.params.temperature
            probs = np.exp(logits)
            probs /= probs.sum()
            k = int(self.rng.choice(len(scores), p=probs))
        else:
            k = int(np.argmax(scores))

        self._record_dispatch(a[k], b[k])
        self.choice_log.append({
            "score": float(scores[k]),
            **{name: float(values[k]) for name, values in components.items()},
        })
        return int(self._ids[a[k]]), int(self._ids[b[k]])
