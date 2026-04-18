"""Event-driven ladder simulator, matchmaker-agnostic.

Consumes any object implementing the `Matchmaker` protocol (see
`sim/matchmaker.py`). The sim owns: simulation clock, server
concurrency, ELO state, match outcome sampling, and output logging.
The matchmaker owns: pair selection and any scoring bookkeeping.
"""

import heapq
import logging
from typing import NamedTuple

import numpy as np
import pandas as pd

from sim.common import simulate_match, update_elo
from sim.matchmaker import Matchmaker


class _ActiveMatch(NamedTuple):
    """Entry in the `active_matches` heapq.

    Tuple ordering (end_time, then unique mid) makes `heapq` pop the
    earliest-completing match; ties are always broken by mid.
    """
    end_time: float
    mid: int
    bot_a: int
    bot_b: int
    outcome_a: float
    duration: float
    category: str
    elo_diff: float

log = logging.getLogger(__name__)


_MATCH_HISTORY_COLUMNS = [
    "match_id", "time_start", "time_end", "bot_a", "bot_b", "outcome_a",
    "duration_minutes", "category", "elo_diff",
]


class LadderSim:
    """Event-driven ladder simulator with a pluggable matchmaker.

    Usage:
        sim = LadderSim(matchmaker, bots, gp, lookup, rng)
        sim.run(total_matches=20000)
        # Outputs: sim.match_history, sim.elo_snapshots, sim.ratings, sim.current_time

    Concurrency: each bot can play only one match at a time. `max_concurrent`
    caps the number of simultaneously running matches to model the AI
    Arena server slot constraint.

    `time_start` / `time_end` / `elo_diff` in `match_history` are stored at
    full precision so the matchmaker can derive state (e.g., 24h windows)
    without rounding drift. CLI callers round them when writing CSV.
    """

    def __init__(
        self,
        matchmaker: Matchmaker,
        bots: pd.DataFrame,
        global_params: dict,
        matchup_lookup: dict,
        seed=None,
    ):
        self.matchmaker = matchmaker
        self.bots = bots
        self.global_params = global_params
        self.matchup_lookup = matchup_lookup
        self.rng = np.random.default_rng(seed)

        self.bot_ids: list[int] = bots["bot_id"].tolist()
        self._bot_data_enabled: dict[int, bool] = dict(
            zip(bots["bot_id"], bots["bot_data_enabled"].astype(bool))
        )

        self.ratings: dict[int, float] = {b: 1600.0 for b in self.bot_ids}
        self.current_time = 0.0
        self.active_matches: list[_ActiveMatch] = []  # min-heap by end_time
        self.match_history: list[dict] = []
        self.elo_snapshots: list[dict] = []
        self._next_match_id = 0
        # Index of the first row of match_history still inside the current
        # time window. Matches at positions < this cursor have aged out and
        # are not shown to the matchmaker. Amortized O(1) per completion.
        self._window_cursor = 0

        self._record_snapshot(match_count=0, time_minutes=0.0)

    def run(
        self,
        total_matches: int,
        max_concurrent: int = 12,
        elo_snapshot_interval: int = 1000,
        history_window_minutes: float = 24 * 60,
    ) -> None:
        log.info("%d bots, %d server slots", len(self.bot_ids), max_concurrent)
        self._fill_slots(max_concurrent)
        while len(self.match_history) < total_matches and self.active_matches:
            self._advance(elo_snapshot_interval, history_window_minutes)
            self._fill_slots(max_concurrent)
        log.info(
            "Completed %d matches in %.0f simulated minutes (%.1f days)",
            len(self.match_history), self.current_time, self.current_time / (24 * 60),
        )

    def _fill_slots(self, max_concurrent: int) -> None:
        windowed = self.match_history[self._window_cursor:]
        if windowed:
            match_hist_df = pd.DataFrame(windowed)
        else:
            match_hist_df = pd.DataFrame(columns=_MATCH_HISTORY_COLUMNS)

        while len(self.active_matches) < max_concurrent:
            in_match = {bot for m in self.active_matches for bot in (m.bot_a, m.bot_b)}
            bots_snap = self.bots.copy()
            bots_snap["elo"] = bots_snap["bot_id"].map(self.ratings)
            bots_snap["in_match"] = bots_snap["bot_id"].isin(in_match)

            # `available_bots` excludes data-enabled bots currently in a
            # match (single-instance constraint). Non-data bots are always
            # included — they can run many parallel instances on AI Arena.
            available_bots = [
                b for b in self.bot_ids
                if not (self._bot_data_enabled[b] and b in in_match)
            ]
            if len(available_bots) < 2:
                break

            pair = self.matchmaker(bots_snap, match_hist_df, available_bots)
            if pair is None:
                break
            bot_a, bot_b = pair
            self._validate_pair(bot_a, bot_b, in_match)

            elo_diff = abs(self.ratings[bot_a] - self.ratings[bot_b])
            outcome_a, duration, category = simulate_match(
                bot_a, bot_b, self.matchup_lookup, self.global_params, self.rng,
            )
            end_time = self.current_time + duration
            mid = self._next_match_id
            self._next_match_id += 1
            heapq.heappush(self.active_matches, _ActiveMatch(
                end_time=end_time, mid=mid, bot_a=bot_a, bot_b=bot_b,
                outcome_a=outcome_a, duration=duration, category=category,
                elo_diff=elo_diff,
            ))

    def _validate_pair(
        self, bot_a: int, bot_b: int, in_match: set[int],
    ) -> None:
        """Raise if the matchmaker returned a physically invalid pair.

        The sim accepts multiple concurrent matches for a bot iff it has
        `bot_data_enabled == False` (non-data bots can run many parallel
        instances on AI Arena). Data-enabled bots must be single-instance;
        picking one that's already in flight is a hard error.
        """
        if bot_a == bot_b:
            raise ValueError(f"matchmaker returned self-match: {bot_a} vs {bot_a}")
        for bot in (bot_a, bot_b):
            if bot not in self._bot_data_enabled:
                raise ValueError(f"matchmaker returned unknown bot_id: {bot}")
            if self._bot_data_enabled[bot] and bot in in_match:
                raise ValueError(
                    f"matchmaker returned bot {bot} which is data-enabled "
                    f"and currently in an active match"
                )

    def _advance(self, snapshot_interval: int, history_window_minutes: float) -> None:
        m = heapq.heappop(self.active_matches)
        self.current_time = m.end_time
        update_elo(self.ratings, m.bot_a, m.bot_b, m.outcome_a)

        self.match_history.append({
            "match_id": m.mid,
            "time_start": m.end_time - m.duration,
            "time_end": m.end_time,
            "bot_a": m.bot_a,
            "bot_b": m.bot_b,
            "outcome_a": m.outcome_a,
            "duration_minutes": m.duration,
            "category": m.category,
            "elo_diff": m.elo_diff,
        })

        # Advance window cursor past any rows that have now aged out.
        cutoff = self.current_time - history_window_minutes
        while (self._window_cursor < len(self.match_history)
               and self.match_history[self._window_cursor]["time_end"] <= cutoff):
            self._window_cursor += 1

        n = len(self.match_history)
        if n % snapshot_interval == 0:
            self._record_snapshot(match_count=n, time_minutes=m.end_time)
        if n % 10000 == 0:
            log.info(
                "  %d matches, t=%.0f min (%.1f days)",
                n, m.end_time, m.end_time / (24 * 60),
            )

    def _record_snapshot(self, match_count: int, time_minutes: float) -> None:
        snapshot_num = len(self.elo_snapshots) // len(self.bot_ids)
        bot_names = dict(zip(self.bots["bot_id"], self.bots["name"]))
        for b in self.bot_ids:
            self.elo_snapshots.append({
                "snapshot": snapshot_num,
                "match_count": match_count,
                "time_minutes": round(time_minutes, 2),
                "bot_id": b,
                "bot_name": bot_names[b],
                "elo": round(self.ratings[b], 2),
            })
