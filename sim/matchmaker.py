"""Matchmaker protocol for the ladder simulator."""

from typing import Optional, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class Matchmaker(Protocol):
    """Pluggable matchmaker protocol consumed by `sim.LadderSim`.

    Implementations may carry internal state across calls (e.g., incremental
    caches derived from `match_history`, or per-choice diagnostic logs for
    post-run analysis). Any such state is purely internal — the sim does
    not read it and `match_history` has a fixed sim-owned schema.
    """

    def __call__(
        self,
        bots: pd.DataFrame,
        match_history: pd.DataFrame,
        available_bots: list[int],
    ) -> Optional[tuple[int, int]]:
        """Pick the next pair of bots to play, or return `None` to skip.

        Arguments:

        `bots`: snapshot of bot metadata with one row per bot. Columns:

          - `bot_id`        int   — stable identifier
          - `name`          str   — display name
          - `race`          str   — SC2 race code: 'T', 'Z', 'P', or 'R'
          - `elo`           float — CURRENT ELO rating, updated by the sim
                                    after each match. Reflects the state at
                                    the moment of this call.
          - `active`        bool  — competition-active flag (always True for
                                    bots the sim considers — filtered in
                                    `load_model`)
          - `bot_data_enabled` bool — AI Arena setting. True means the bot
                                    requires single-instance execution
                                    (only one concurrent match allowed).
                                    False means the bot has no persistent
                                    state, so many parallel instances can
                                    run at once.
          - `in_match`     bool  — True iff this bot appears in at least one
                                    currently-dispatched (not yet completed)
                                    match.

        The DataFrame is a fresh snapshot each call. Do NOT retain references
        across calls; cache scalar values if needed.

        `match_history`: all completed matches in the last
        `history_window_minutes` (default 24h) of simulated time, in
        completion-time order (append order). Older matches are dropped from
        the front. Columns:

          - `match_id`         int    — dispatch-order id (NOT monotonic in
                                        this DataFrame; use as a set key for
                                        detecting already-ingested rows, not
                                        for ordering)
          - `time_start`       float  — simulated minutes when the match was
                                        dispatched. Matches the sim's
                                        internal clock exactly (the CSV
                                        writer rounds to 2 decimals at write
                                        time, but this column is the raw
                                        value so matchmaker time-window
                                        math agrees with the sim).
          - `time_end`         float  — simulated minutes when the match
                                        completed. Same precision note.
          - `bot_a`, `bot_b`   int    — participants
          - `outcome_a`        float  — 1.0 (a won), 0.5 (draw), 0.0 (a lost)
          - `duration_minutes` float  — match length, ≤ 60
          - `category`         str    — 'normal' | 'timelimit' | 'abnormal'
          - `elo_diff`         float  — |elo_a − elo_b| at dispatch time,
                                        also unrounded for the same reason

        Rows are in completion-time order, not dispatch (`match_id`) order —
        `heapq` pops by `end_time`, so a short match dispatched after a long
        one will appear first. To identify already-ingested rows, track seen
        `match_id`s in a set rather than a length or max-id cursor.

        `available_bots`: list of bot ids the sim considers eligible to
        start a new match right now. Data-enabled bots currently in a match
        (`in_match=True`) are excluded; non-data bots are always included
        (they can run many parallel instances on AI Arena). Matchmakers can
        pick any pair from this list without re-applying the single-instance
        rule. Guaranteed `len >= 2` when the matchmaker is called.

        Return:

        `(bot_a, bot_b)` — the pair to play next. Must be distinct, both must
        exist in `bots`, and neither may be a `bot_data_enabled=True` bot
        that's currently `in_match` (the sim raises `ValueError` otherwise
        — but if you pick from `available_bots` this is already guaranteed).
        Order is not significant to the sim.

        `None` — no acceptable pair under the matchmaker's own policy (e.g.
        every remaining candidate is a rematch). The sim breaks its fill
        loop and advances to the next completion event before trying again.
        """
        ...
