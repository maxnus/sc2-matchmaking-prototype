"""Event-driven ladder simulator, matchmaker-agnostic.

Consumes any object implementing the `Matchmaker` protocol (see
`sim/matchmaker.py`). The sim owns: simulation clock, server
concurrency, ELO state, match outcome sampling, and output logging.
The matchmaker owns: pair selection and any scoring bookkeeping.
"""

import heapq
import logging
from collections import Counter
from typing import NamedTuple

import numpy as np
import pandas as pd

from sim.common import GlobalParams, simulate_match, update_elo
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
        sim = LadderSim(matchmaker, bots, gp, lookup, seed=42)
        sim.run(total_matches=20000)
        # Outputs: sim.match_history, sim.elo_snapshots, sim.ratings, sim.current_time

    Concurrency: `max_concurrent` caps the number of simultaneously running
    matches to model the AI Arena server slots. A bot with bot data plays
    one match at a time; a bot without plays at most `max_parallel`.

    `time_start` / `time_end` / `elo_diff` in `match_history` are stored at
    full precision so the matchmaker can derive state (e.g., 24h windows)
    without rounding drift. CLI callers round them when writing CSV.

    Ratings start at `initial_ratings` (bot_id → ELO) if given, otherwise
    every bot starts at 1600.
    """

    def __init__(
        self,
        matchmaker: Matchmaker,
        bots: pd.DataFrame,
        global_params: GlobalParams,
        matchup_lookup: dict,
        seed=None,
        initial_ratings: dict[int, float] | None = None,
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

        self.ratings: dict[int, float] = {
            b: float(initial_ratings[b]) if initial_ratings else 1600.0
            for b in self.bot_ids
        }
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
        max_parallel: int = 4,
        elo_snapshot_interval: int = 1000,
        history_window_minutes: float = 24 * 60,
    ) -> None:
        log.info("%d bots, %d server slots", len(self.bot_ids), max_concurrent)
        self._fill_slots(max_concurrent, max_parallel)
        while len(self.match_history) < total_matches and self.active_matches:
            self._advance(elo_snapshot_interval, history_window_minutes)
            self._fill_slots(max_concurrent, max_parallel)
        log.info(
            "Completed %d matches in %.0f simulated minutes (%.1f days)",
            len(self.match_history), self.current_time, self.current_time / (24 * 60),
        )

    def _parallel_limit(self, bot: int, max_parallel: int) -> int:
        return 1 if self._bot_data_enabled[bot] else max_parallel

    def _fill_slots(self, max_concurrent: int, max_parallel: int) -> None:
        windowed = self.match_history[self._window_cursor:]
        if windowed:
            match_hist_df = pd.DataFrame(windowed)
        else:
            match_hist_df = pd.DataFrame(columns=_MATCH_HISTORY_COLUMNS)

        while len(self.active_matches) < max_concurrent:
            playing = Counter(bot for m in self.active_matches for bot in (m.bot_a, m.bot_b))
            bots_snap = self.bots.copy()
            bots_snap["elo"] = bots_snap["bot_id"].map(self.ratings)
            bots_snap["in_match"] = bots_snap["bot_id"].isin(playing)

            # `available_bots` excludes bots already playing as many matches
            # as they may: one for data-enabled bots (single instance),
            # `max_parallel` for the others.
            available_bots = [
                b for b in self.bot_ids
                if playing[b] < self._parallel_limit(b, max_parallel)
            ]
            if len(available_bots) < 2:
                break

            pair = self.matchmaker(bots_snap, match_hist_df, available_bots)
            if pair is None:
                break
            bot_a, bot_b = pair
            self._validate_pair(bot_a, bot_b, playing, max_parallel)

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
        self, bot_a: int, bot_b: int, playing: Counter, max_parallel: int,
    ) -> None:
        """Raise if the matchmaker returned a physically invalid pair.

        A data-enabled bot must be single-instance, and a bot without bot
        data may play at most `max_parallel` matches at once; picking a bot
        that's already at its limit is a hard error.
        """
        if bot_a == bot_b:
            raise ValueError(f"matchmaker returned self-match: {bot_a} vs {bot_a}")
        for bot in (bot_a, bot_b):
            if bot not in self._bot_data_enabled:
                raise ValueError(f"matchmaker returned unknown bot_id: {bot}")
            if playing[bot] >= self._parallel_limit(bot, max_parallel):
                raise ValueError(
                    f"matchmaker returned bot {bot}, which is already playing "
                    f"{playing[bot]} matches, its limit"
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
