"""Stochastic matchmaker: weighted skill/fairness/variety score with softmax sampling."""

from collections import Counter, deque
from dataclasses import dataclass
from itertools import combinations
from math import exp, inf, sqrt
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class ScoringParams:
    w_skill: float = 0.15
    w_fair: float = 1.0
    w_var: float = 0.2
    tau: float = 15.0  # rank-difference tolerance (in rank positions)
    lam: float = 40.0
    temperature: float = 0.01  # softmax sampling temperature (0 = argmax)


def _canonical(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def f_skill(rank_a: int, rank_b: int, tau: float) -> float:
    return 1.0 / (1.0 + ((rank_a - rank_b) / tau) ** 2)


def f_fair(g_a: int, g_b: int, g_bar: float) -> float:
    if g_bar == 0:
        return 0.0
    d_a = max(g_bar - g_a, 0) / g_bar
    d_b = max(g_bar - g_b, 0) / g_bar
    return sqrt((d_a**2 + d_b**2) / 2)


def f_var(a_ab: float, b_ab: float, lam: float) -> float:
    return 1.0 - exp(-sqrt(a_ab * b_ab) / lam)


class StochasticMatchmaker:
    """Event-driven matchmaker with rematch prevention and softmax sampling.

    Maintains internal bookkeeping:
      - `_games_24h[bot]`: completion times of the bot's matches in the last
        24h of simulated time (the clock is the latest completion seen)
      - `_in_flight[bot]`: matches dispatched but not yet completed
      - `_in_flight_pairs[(lo, hi)]`: the same, per pair
      - `_gso[a][b]`: games dispatched for `a` since it was last dispatched
        against `b` (0 = rematch)
      - `_total_games[bot]`: cumulative dispatched matches per bot

    The sim fills every free slot in one go, before any of those matches
    completes. Pair history and in-flight counts are therefore updated when a
    pair is returned, not when it completes; otherwise the same pair, or the
    same underplayed bot, would be picked for several slots at once. This
    relies on the sim dispatching every pair the matchmaker returns.

    Rematch prevention: any pair that is in flight, or where either bot's
    most recently dispatched opponent was the other, is excluded from
    scoring.
    """

    def __init__(self, params: ScoringParams, seed=None):
        self.params = params
        self.rng = np.random.default_rng(seed)
        self._games_24h: dict[int, deque[float]] = {}
        self._in_flight: Counter[int] = Counter()
        self._in_flight_pairs: Counter[tuple[int, int]] = Counter()
        self._gso: dict[int, dict[int, int]] = {}
        self._total_games: dict[int, int] = {}
        self._seen_mids: set[int] = set()
        self._now = 0.0
        # Per-dispatch diagnostics. Index aligns with the sim's `match_id`
        # (matchmaker is called once per dispatch; we append on success).
        self.choice_log: list[dict[str, float]] = []

    def _ensure_bot(self, bot_id: int) -> None:
        if bot_id not in self._games_24h:
            self._games_24h[bot_id] = deque()
            self._gso[bot_id] = {}
            self._total_games[bot_id] = 0

    def _ingest_new_matches(self, match_history: pd.DataFrame) -> None:
        """Record completions from any match_history rows not yet seen.

        The sim may pass a time-windowed DataFrame (shorter than the full
        history), so "new" rows can't be detected by length. Instead we
        track seen `match_id`s and scan back from the tail — new rows are
        always appended at the end, so the scan stops quickly.
        """
        n = len(match_history)
        if n == 0:
            return
        new_start = n
        while new_start > 0:
            mid = int(match_history.iloc[new_start - 1]["match_id"])
            if mid in self._seen_mids:
                break
            new_start -= 1
        if new_start == n:
            return
        for row in match_history.iloc[new_start:].to_dict("records"):
            a = int(row["bot_a"])
            b = int(row["bot_b"])
            end_time = float(row["time_end"])
            self._seen_mids.add(int(row["match_id"]))
            self._ensure_bot(a)
            self._ensure_bot(b)

            self._games_24h[a].append(end_time)
            self._games_24h[b].append(end_time)
            self._now = max(self._now, end_time)

            self._in_flight[a] -= 1
            self._in_flight[b] -= 1
            self._in_flight_pairs[_canonical(a, b)] -= 1

    def _expire_old_games(self) -> None:
        """Drop completions older than 24h from every bot's window.

        Pruning against the global clock (not just when a bot plays again)
        keeps the count of a bot that hasn't played lately from going stale.
        """
        cutoff = self._now - 24 * 60
        for times in self._games_24h.values():
            while times and times[0] <= cutoff:
                times.popleft()

    def _games_in_window(self, bot_id: int) -> int:
        """Matches counted for fairness: completed in the last 24h, plus in flight."""
        return len(self._games_24h[bot_id]) + self._in_flight[bot_id]

    def _record_dispatch(self, bot_a: int, bot_b: int) -> None:
        self._in_flight[bot_a] += 1
        self._in_flight[bot_b] += 1
        self._in_flight_pairs[_canonical(bot_a, bot_b)] += 1
        for bot, opp in ((bot_a, bot_b), (bot_b, bot_a)):
            self._total_games[bot] += 1
            gso = self._gso[bot]
            for other in gso:
                gso[other] += 1
            gso[opp] = 0

    def _score(
        self, bot_a: int, bot_b: int, ranks: dict[int, int], g_bar: float,
    ) -> tuple[float, dict[str, float]]:
        skill = f_skill(ranks[bot_a], ranks[bot_b], self.params.tau)

        g_a = self._games_in_window(bot_a)
        g_b = self._games_in_window(bot_b)
        fair = f_fair(g_a, g_b, g_bar)

        gso_a = self._gso[bot_a]
        gso_b = self._gso[bot_b]
        a_ab = gso_a.get(bot_b, self._total_games[bot_a] or inf)
        b_ab = gso_b.get(bot_a, self._total_games[bot_b] or inf)
        var = f_var(a_ab, b_ab, self.params.lam)

        components = {
            "s_skill": self.params.w_skill * skill,
            "s_fair": self.params.w_fair * fair,
            "s_var": self.params.w_var * var,
        }
        total = sum(components.values())
        return total, components

    def __call__(
        self,
        bots: pd.DataFrame,
        match_history: pd.DataFrame,
        available_bots: list[int],
    ) -> Optional[tuple[int, int]]:
        all_bot_ids = bots["bot_id"].tolist()
        for b in all_bot_ids:
            self._ensure_bot(int(b))
        self._ingest_new_matches(match_history)
        self._expire_old_games()

        ratings = dict(zip(bots["bot_id"], bots["elo"]))
        sorted_bots = sorted(all_bot_ids, key=lambda b: (-ratings[b], b))
        ranks = {b: i for i, b in enumerate(sorted_bots)}

        g_bar = sum(self._games_in_window(b) for b in all_bot_ids) / len(all_bot_ids)

        pairs: list[tuple[int, int]] = []
        scores: list[float] = []
        components_list: list[dict[str, float]] = []
        for a, b in combinations(available_bots, 2):
            if (self._in_flight_pairs[_canonical(a, b)] > 0
                    or self._gso[a].get(b) == 0 or self._gso[b].get(a) == 0):
                continue
            s, components = self._score(a, b, ranks, g_bar)
            pairs.append((a, b))
            scores.append(s)
            components_list.append(components)

        if not pairs:
            return None

        if self.params.temperature > 0:
            scores_arr = np.asarray(scores)
            logits = (scores_arr - scores_arr.max()) / self.params.temperature
            probs = np.exp(logits)
            probs /= probs.sum()
            idx = int(self.rng.choice(len(pairs), p=probs))
        else:
            idx = int(np.argmax(scores))

        bot_a, bot_b = pairs[idx]
        self._record_dispatch(bot_a, bot_b)
        self.choice_log.append({"score": scores[idx], **components_list[idx]})
        return bot_a, bot_b
