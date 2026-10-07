"""Random baseline matchmaker — picks two available bots uniformly at random."""

from typing import Optional

import numpy as np
import pandas as pd


class RandomMatchmaker:
    """Uniformly random pair selection. No state, no rematch avoidance.

    Intended as a baseline for comparing against smarter matchmakers. The
    sim's `available_bots` already honors the limits on parallel matches
    (one for data-enabled bots, `max_parallel` for the others), so we just
    pick two at random from it.
    """

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def __call__(
        self,
        bots: pd.DataFrame,
        match_history: pd.DataFrame,
        available_bots: list[int],
    ) -> Optional[tuple[int, int]]:
        idx = self.rng.choice(len(available_bots), size=2, replace=False)
        return int(available_bots[idx[0]]), int(available_bots[idx[1]])
