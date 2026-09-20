"""Stochastic matchmaker: weighted skill/fairness/variety score with softmax sampling."""

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

    Maintains internal bookkeeping derived from `match_history`:
      - `_games_24h[bot]`: timestamps of completed matches in the last 24h
      - `_gso[a][b]`: games played by `a` since last facing `b` (0 = rematch)
      - `_total_games[bot]`: cumulative completed matches per bot

    Rematch prevention: any pair where either bot's most recent opponent was
    the other is excluded from scoring.
    """

    def __init__(self, params: ScoringParams, seed=None):
        self.params = params
        self.rng = np.random.default_rng(seed)
        self._games_24h: dict[int, list[float]] = {}
        self._gso: dict[int, dict[int, int]] = {}
        self._total_games: dict[int, int] = {}
        self._seen_mids: set[int] = set()
        # Per-dispatch diagnostics. Index aligns with the sim's `match_id`
        # (matchmaker is called once per dispatch; we append on success).
        self.choice_log: list[dict[str, float]] = []

    def _ensure_bot(self, bot_id: int) -> None:
        if bot_id not in self._games_24h:
            self._games_24h[bot_id] = []
            self._gso[bot_id] = {}
            self._total_games[bot_id] = 0

    def _ingest_new_matches(self, match_history: pd.DataFrame) -> None:
        """Update internal state from any match_history rows not yet seen.

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
            cutoff = end_time - 24 * 60
            self._games_24h[a] = [t for t in self._games_24h[a] if t > cutoff]
            self._games_24h[b] = [t for t in self._games_24h[b] if t > cutoff]

            self._total_games[a] += 1
            self._total_games[b] += 1
            for opp in self._gso[a]:
                self._gso[a][opp] += 1
            self._gso[a][b] = 0
            for opp in self._gso[b]:
                self._gso[b][opp] += 1
            self._gso[b][a] = 0

    def _score(
        self, bot_a: int, bot_b: int, ranks: dict[int, int], g_bar: float,
    ) -> tuple[float, dict[str, float]]:
        skill = f_skill(ranks[bot_a], ranks[bot_b], self.params.tau)

        g_a = len(self._games_24h[bot_a])
        g_b = len(self._games_24h[bot_b])
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

        ratings = dict(zip(bots["bot_id"], bots["elo"]))
        sorted_bots = sorted(all_bot_ids, key=lambda b: (-ratings[b], b))
        ranks = {b: i for i, b in enumerate(sorted_bots)}

        g_bar = sum(len(self._games_24h[b]) for b in all_bot_ids) / len(all_bot_ids)

        # Treat every bot as single-instance — exclude anyone currently
        # in a match. (The sim's `available_bots` only honors the physical
        # `bot_data_enabled` constraint; this matchmaker chooses to be
        # stricter so non-data bots are also serialized.)
        in_match = set(bots.loc[bots["in_match"], "bot_id"].astype(int))
        candidates = [b for b in available_bots if b not in in_match]

        pairs: list[tuple[int, int]] = []
        scores: list[float] = []
        components_list: list[dict[str, float]] = []
        for a, b in combinations(candidates, 2):
            if self._gso[a].get(b) == 0 or self._gso[b].get(a) == 0:
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
        self.choice_log.append({"score": scores[idx], **components_list[idx]})
        return bot_a, bot_b
