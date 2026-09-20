"""Rung matchmaker: Random Serial Dictatorship draft with rung & wildcard slots.

Each bot has 10 slots per round: `rung_picks` for rung matches + `wildcard_picks`
for wildcard matches. Bots are placed in a random draft order. Each bot in
turn fills its remaining slots by picking random opponents from:

  1. its rung pool (`rung_size` closest bots by current ELO) for rung slots,
  2. bots outside its rung pool for wildcard slots.

Each match consumes one slot on each side (the slot type from each bot's
perspective is determined by whether the other bot sits in that bot's own
rung). After a bot finishes drafting, it is removed from the draft pool —
later bots cannot pick it as an opponent.

If a bot's rung pool has fewer than its remaining rung-slot need (because
those rung members have already drafted and are removed, or their receiving
slot is full), the shortfall is filled with additional wildcard matches so
the bot still finishes with 10 scheduled matches.

Strict round semantics: between rounds the matchmaker returns `None` until
all in-flight matches have completed, so the next round's rung assignment
is based on fully-applied ELO updates.
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

        # Rung pool per bot: its `rung_size` closest-by-ELO neighbours.
        rung: dict[int, set[int]] = {}
        for b in bot_ids:
            others = sorted(
                (o for o in bot_ids if o != b),
                key=lambda o: abs(ratings[o] - ratings[b]),
            )
            rung[b] = set(others[:self.rung_size])

        # Per-bot slot counters (from that bot's own perspective):
        #   `rung_filled[b]` = matches where the other bot is in b's rung
        #   `wc_filled[b]`   = matches where the other bot is NOT in b's rung
        rung_filled = {b: 0 for b in bot_ids}
        wc_filled = {b: 0 for b in bot_ids}

        pairs: set[tuple[int, int]] = set()

        draft_order = list(bot_ids)
        self.rng.shuffle(draft_order)

        total_cap = self.rung_picks + self.wildcard_picks  # 10 slots per bot

        def can_accept(drafter: int, opp: int) -> bool:
            """Can `drafter` pick `opp` — not already paired, and opp still
            has at least one total slot remaining. We treat the 10-total
            cap as the strict invariant (so every bot ends with exactly
            10 matches when feasible) and the 8:2 rung/wildcard split as
            a preference — individual bots may end up at e.g. 7:3 when the
            strict split isn't realizable given their rung shape."""
            if opp == drafter:
                return False
            if _canonical(drafter, opp) in pairs:
                return False
            return rung_filled[opp] + wc_filled[opp] < total_cap

        def commit(drafter: int, opp: int, drafter_side_rung: bool) -> None:
            pairs.add(_canonical(drafter, opp))
            if drafter_side_rung:
                rung_filled[drafter] += 1
            else:
                wc_filled[drafter] += 1
            if drafter in rung[opp]:
                rung_filled[opp] += 1
            else:
                wc_filled[opp] += 1

        for B in draft_order:
            # Total slots B still needs to fill (the strict 10-total cap;
            # already-incoming picks during this round count toward it).
            total_remaining = total_cap - rung_filled[B] - wc_filled[B]
            if total_remaining <= 0:
                continue

            # Preferred split: try for 8 rung + 2 wildcard (adjusted by
            # already-filled slots), but never more than total_remaining.
            rung_target = min(
                total_remaining,
                max(0, self.rung_picks - rung_filled[B]),
            )

            rung_candidates = [C for C in rung[B] if can_accept(B, C)]
            num_rung = min(rung_target, len(rung_candidates))
            chosen_rung = [
                int(c) for c in (
                    self.rng.choice(rung_candidates, size=num_rung, replace=False)
                    if num_rung > 0 else []
                )
            ]

            # Anything left in B's total budget goes to wildcards — this
            # absorbs the rung shortfall and the wildcard target in one.
            wc_need = total_remaining - num_rung
            wc_candidates = [
                C for C in bot_ids
                if C != B and C not in rung[B]
                and can_accept(B, C) and C not in chosen_rung
            ]
            num_wc = min(wc_need, len(wc_candidates))
            chosen_wc = [
                int(c) for c in (
                    self.rng.choice(wc_candidates, size=num_wc, replace=False)
                    if num_wc > 0 else []
                )
            ]

            for C in chosen_rung:
                commit(B, C, drafter_side_rung=True)
            for C in chosen_wc:
                commit(B, C, drafter_side_rung=False)

        # Cleanup pass — bots whose quota wasn't filled in their RSD turn
        # (because their rung pool and wildcard pool were both exhausted at
        # the time) top up here by picking any opponent that still has
        # capacity. Keeps preferring rung partners when a slot of that
        # type is still desired.
        for B in draft_order:
            while rung_filled[B] + wc_filled[B] < total_cap:
                remaining = total_cap - rung_filled[B] - wc_filled[B]
                want_rung = rung_filled[B] < self.rung_picks
                # Try rung-pool partner first if still wanted.
                if want_rung:
                    rung_cands = [C for C in rung[B] if can_accept(B, C)]
                    if rung_cands:
                        C = int(self.rng.choice(rung_cands))
                        commit(B, C, drafter_side_rung=True)
                        continue
                # Otherwise any wildcard partner.
                wc_cands = [
                    C for C in bot_ids
                    if C != B and C not in rung[B] and can_accept(B, C)
                ]
                if wc_cands:
                    C = int(self.rng.choice(wc_cands))
                    commit(B, C, drafter_side_rung=False)
                    continue
                # If neither rung nor wildcard candidates exist, last-ditch:
                # any other bot with capacity, matched to whichever slot
                # type matches the rung relationship.
                any_cands = [C for C in bot_ids if C != B and can_accept(B, C)]
                if not any_cands:
                    break
                C = int(self.rng.choice(any_cands))
                commit(B, C, drafter_side_rung=(C in rung[B]))

        self._queue = list(pairs)
        self.rng.shuffle(self._queue)
