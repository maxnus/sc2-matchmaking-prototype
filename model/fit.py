"""Bayesian ladder outcome model: fit from raw matches, save/load.

The model is described, and its fit plotted, in the project report
(`analysis/report.md`).
"""

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from itertools import combinations
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sim.common import Category, GlobalParams, Outcome, preprocess_matches
from sim.paths import DATA_DIR

_own_dir = Path(__file__).resolve().parent

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# --- Pure-function helpers (stateless transforms) ---


def _load_raw(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    bots = pd.read_csv(data_dir / "bots.csv")
    matches = pd.read_csv(data_dir / "matches.csv")
    log.info("Loaded %d bots, %d matches", len(bots), len(matches))
    return bots, matches


def _compute_global_params(matches: pd.DataFrame, n_0: int) -> GlobalParams:
    total = len(matches)
    normal = matches[matches["category"] == Category.NORMAL]
    timelimit = matches[matches["category"] == Category.TIMELIMIT]
    abnormal = matches[matches["category"] == Category.ABNORMAL]

    p_normal = len(normal) / total
    p_timelimit = len(timelimit) / total
    p_abnormal = len(abnormal) / total
    # Each abnormal game is one bot crashing, and each game has two bots.
    crash_rate = len(abnormal) / (2 * total)

    d = (normal["lo_outcome"] == Outcome.DRAW).sum() / len(normal)

    dur_normal = normal["duration_minutes"].dropna()
    dur_normal = dur_normal[dur_normal > 0]
    log_dur = np.log(dur_normal.values)
    mu_0 = float(log_dur.mean())
    sigma = float(log_dur.std(ddof=0))

    # Split the spread into scatter within a pair and differences between
    # pairs' means. The between-pair variance is the prior on a pair's mean,
    # expressed as a pseudo-count of games in `n_0_duration`.
    log_dur_s = pd.Series(log_dur, index=dur_normal.index)
    by_pair = log_dur_s.groupby(
        [normal.loc[dur_normal.index, "bot_lo"], normal.loc[dur_normal.index, "bot_hi"]]
    )
    deviations = log_dur_s - by_pair.transform("mean")
    within_var = float((deviations**2).sum()) / (len(log_dur) - by_pair.ngroups)
    sigma_within = sqrt(within_var)
    n_0_duration = within_var / max(sigma**2 - within_var, 1e-9)

    dur_abnormal = abnormal["duration_minutes"].dropna()
    dur_abnormal = dur_abnormal[dur_abnormal > 0]
    if len(dur_abnormal) > 0:
        log_dur_abn = np.log(dur_abnormal.values)
        mu_abnormal = float(log_dur_abn.mean())
        sigma_abnormal = float(log_dur_abn.std(ddof=0))
    else:
        mu_abnormal = mu_0
        sigma_abnormal = sigma

    gp = GlobalParams(
        n_0=n_0,
        p_normal=round(p_normal, 6),
        p_timelimit=round(p_timelimit, 6),
        p_abnormal=round(p_abnormal, 6),
        crash_rate=round(crash_rate, 6),
        d=round(d, 6),
        mu_0=round(mu_0, 6),
        sigma=round(sigma, 6),
        sigma_within=round(sigma_within, 6),
        n_0_duration=round(n_0_duration, 6),
        mu_abnormal=round(mu_abnormal, 6),
        sigma_abnormal=round(sigma_abnormal, 6),
    )
    log.info("Global parameters:")
    for k, v in asdict(gp).items():
        log.info("  %s = %s", k, v)
    return gp


@dataclass
class MatchupRow:
    """One row of the fitted matchup_params CSV. Field order is the CSV
    column order.

    `x_bar` is a float when `n_dur > 0` and `""` otherwise — empty string
    is the CSV convention for "no duration data on this pair".
    """
    bot_lo: int
    bot_hi: int
    bot_lo_name: str
    bot_hi_name: str
    elo_lo: int
    elo_hi: int
    N_normal: int
    N_timelimit: int
    N_abnormal: int
    crash_rate_lo: float
    crash_rate_hi: float
    alpha_normal: float
    alpha_timelimit: float
    alpha_abnormal: float
    C_lo: int
    C_hi: int
    alpha_crash_lo: float
    alpha_crash_hi: float
    W: int
    L: int
    D: int
    n_dur: int
    x_bar: Any
    E_A: float
    alpha_win: float
    alpha_draw: float
    alpha_loss: float
    mu_duration: float
    sigma_duration: float


def _compute_crash_rates(
    matches: pd.DataFrame, bot_ids: list[int], gp: GlobalParams,
) -> dict[int, float]:
    """Posterior mean of each bot's crash rate per game.

    Crashing (or timing out) is mostly a property of the bot: a few bots
    cause most abnormal games. The prior is the ladder-wide rate with
    strength `n_0` games, updated with the bot's games and crashes.
    """
    games = pd.concat([matches["bot_lo"], matches["bot_hi"]]).value_counts()
    abnormal = matches[matches["category"] == Category.ABNORMAL]
    crashed = pd.concat([
        abnormal.loc[abnormal["lo_outcome"] == Outcome.LOSS, "bot_lo"],
        abnormal.loc[abnormal["lo_outcome"] == Outcome.WIN, "bot_hi"],
    ]).value_counts()
    return {
        b: (gp.n_0 * gp.crash_rate + crashed.get(b, 0)) / (gp.n_0 + games.get(b, 0))
        for b in bot_ids
    }


def _compute_matchup_params(
    matches: pd.DataFrame, bots: pd.DataFrame, gp: GlobalParams,
) -> pd.DataFrame:
    n_0 = gp.n_0
    d = gp.d
    mu_0 = gp.mu_0
    n_0_dur = gp.n_0_duration

    bot_info = bots.set_index("bot_id")[["name", "elo"]].to_dict("index")
    bot_ids = sorted(bots["bot_id"].values)
    crash_rates = _compute_crash_rates(matches, bot_ids, gp)
    # Of the games that don't end in a crash, the share that hits the time limit.
    p_timelimit_no_crash = gp.p_timelimit / (gp.p_normal + gp.p_timelimit)

    all_pairs = list(combinations(bot_ids, 2))
    pair_index = pd.MultiIndex.from_tuples(all_pairs, names=["bot_lo", "bot_hi"])

    cat_counts = (
        matches.groupby(["bot_lo", "bot_hi", "category"])
        .size()
        .unstack(fill_value=0)
        .reindex(pair_index, fill_value=0)
    )
    for col in Category:
        if col not in cat_counts.columns:
            cat_counts[col] = 0

    abnormal = matches[matches["category"] == Category.ABNORMAL]
    # In an abnormal game the bot that crashed loses: a loss for bot_lo
    # means bot_lo crashed.
    crash_counts = (
        abnormal.groupby(["bot_lo", "bot_hi", "lo_outcome"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[Outcome.LOSS, Outcome.WIN], fill_value=0)
        .reindex(pair_index, fill_value=0)
    )

    normal = matches[matches["category"] == Category.NORMAL]
    outcome_counts = (
        normal.groupby(["bot_lo", "bot_hi", "lo_outcome"])
        .size()
        .unstack(fill_value=0)
        .reindex(pair_index, fill_value=0)
    )
    for col in Outcome:
        if col not in outcome_counts.columns:
            outcome_counts[col] = 0

    dur_data = normal[["bot_lo", "bot_hi", "duration_minutes"]].dropna()
    dur_data = dur_data[dur_data["duration_minutes"] > 0].copy()
    dur_data["log_dur"] = np.log(dur_data["duration_minutes"])

    dur_agg = (
        dur_data.groupby(["bot_lo", "bot_hi"])["log_dur"]
        .agg(["count", "mean"])
        .rename(columns={"count": "n_dur", "mean": "x_bar"})
        .reindex(pair_index, fill_value=0)
    )

    rows = []
    for bot_lo, bot_hi in all_pairs:
        info_lo = bot_info.get(bot_lo, {"name": "?", "elo": 1500})
        info_hi = bot_info.get(bot_hi, {"name": "?", "elo": 1500})
        elo_lo = info_lo["elo"]
        elo_hi = info_hi["elo"]

        N_normal = int(cat_counts.loc[(bot_lo, bot_hi), Category.NORMAL])
        N_timelimit = int(cat_counts.loc[(bot_lo, bot_hi), Category.TIMELIMIT])
        N_abnormal = int(cat_counts.loc[(bot_lo, bot_hi), Category.ABNORMAL])

        # Prior: a game ends abnormally if either bot crashes; otherwise it
        # hits the time limit at the ladder-wide rate.
        c_lo, c_hi = crash_rates[bot_lo], crash_rates[bot_hi]
        p_crash = 1 - (1 - c_lo) * (1 - c_hi)
        alpha_normal = n_0 * (1 - p_crash) * (1 - p_timelimit_no_crash) + N_normal
        alpha_timelimit = n_0 * (1 - p_crash) * p_timelimit_no_crash + N_timelimit
        alpha_abnormal = n_0 * p_crash + N_abnormal

        # Who crashes in an abnormal game: the bots' crash rates as prior,
        # updated with the pair's own abnormal games.
        C_lo = int(crash_counts.loc[(bot_lo, bot_hi), Outcome.LOSS])
        C_hi = int(crash_counts.loc[(bot_lo, bot_hi), Outcome.WIN])
        share_lo = c_lo / (c_lo + c_hi)
        alpha_crash_lo = n_0 * share_lo + C_lo
        alpha_crash_hi = n_0 * (1 - share_lo) + C_hi

        W = int(outcome_counts.loc[(bot_lo, bot_hi), Outcome.WIN])
        L = int(outcome_counts.loc[(bot_lo, bot_hi), Outcome.LOSS])
        D = int(outcome_counts.loc[(bot_lo, bot_hi), Outcome.DRAW])

        E_A = 1.0 / (1.0 + 10.0 ** ((elo_hi - elo_lo) / 400.0))

        alpha_win = n_0 * (1 - d) * E_A + W
        alpha_draw = n_0 * d + D
        alpha_loss = n_0 * (1 - d) * (1 - E_A) + L

        n_dur = int(dur_agg.loc[(bot_lo, bot_hi), "n_dur"])
        x_bar = float(dur_agg.loc[(bot_lo, bot_hi), "x_bar"])
        mu_duration = (n_0_dur * mu_0 + n_dur * x_bar) / (n_0_dur + n_dur)
        # Spread of a single game's log-duration: the scatter within a pair
        # plus the remaining uncertainty about this pair's mean. With no data
        # it equals the global `sigma`.
        sigma_duration = gp.sigma_within * sqrt(1 + 1 / (n_0_dur + n_dur))

        rows.append(MatchupRow(
            bot_lo=bot_lo,
            bot_hi=bot_hi,
            bot_lo_name=info_lo["name"],
            bot_hi_name=info_hi["name"],
            elo_lo=elo_lo,
            elo_hi=elo_hi,
            N_normal=N_normal,
            N_timelimit=N_timelimit,
            N_abnormal=N_abnormal,
            crash_rate_lo=round(c_lo, 6),
            crash_rate_hi=round(c_hi, 6),
            alpha_normal=round(alpha_normal, 6),
            alpha_timelimit=round(alpha_timelimit, 6),
            alpha_abnormal=round(alpha_abnormal, 6),
            C_lo=C_lo,
            C_hi=C_hi,
            alpha_crash_lo=round(alpha_crash_lo, 6),
            alpha_crash_hi=round(alpha_crash_hi, 6),
            W=W,
            L=L,
            D=D,
            n_dur=n_dur,
            x_bar=round(x_bar, 6) if n_dur > 0 else "",
            E_A=round(E_A, 6),
            alpha_win=round(alpha_win, 6),
            alpha_draw=round(alpha_draw, 6),
            alpha_loss=round(alpha_loss, 6),
            mu_duration=round(mu_duration, 6),
            sigma_duration=round(sigma_duration, 6),
        ))

    result = pd.DataFrame([asdict(r) for r in rows])
    has_data = ((result["N_normal"] + result["N_timelimit"] + result["N_abnormal"]) > 0).sum()
    log.info("Matchup parameters: %d pairs (%d with data, %d pure prior)",
             len(result), has_data, len(result) - has_data)
    return result


# --- Model class ---


class LadderModel:
    """Bayesian ladder outcome model.

    Use `LadderModel.fit(bots, matches, n_0)` to fit from raw data, then
    `model.save(dir)` and `LadderModel.load(dir)` for persistence.

    Attributes populated by `fit()`:
      - `global_params`: dict of global parameters (draw rate, duration
        log-normal params, category rates, etc.)
      - `matchups`: DataFrame of per-pair Dirichlet + duration posteriors
    """

    def __init__(self, global_params: GlobalParams, matchups: pd.DataFrame):
        self.global_params = global_params
        self.matchups = matchups

    @classmethod
    def fit(
        cls, bots: pd.DataFrame, matches: pd.DataFrame, n_0: int = 5,
    ) -> "LadderModel":
        """Fit the full model from raw match data."""
        processed = preprocess_matches(matches)
        gp = _compute_global_params(processed, n_0)
        mp = _compute_matchup_params(processed, bots, gp)
        return cls(global_params=gp, matchups=mp)

    @classmethod
    def load(cls, model_dir: Path) -> "LadderModel":
        """Load a fitted model from disk (produced by `.save()`)."""
        with open(model_dir / "global_params.json") as f:
            gp = GlobalParams(**json.load(f))
        matchups = pd.read_csv(model_dir / "matchup_params.csv")
        return cls(global_params=gp, matchups=matchups)

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        gp_path = output_dir / "global_params.json"
        with open(gp_path, "w") as f:
            json.dump(asdict(self.global_params), f, indent=2)
        log.info("Wrote %s", gp_path)

        mp_path = output_dir / "matchup_params.csv"
        self.matchups.to_csv(mp_path, index=False)
        log.info("Wrote %s (%d matchups)", mp_path, len(self.matchups))


# --- CLI ---


def main():
    parser = argparse.ArgumentParser(description="Fit Bayesian ladder model parameters")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=_own_dir)
    parser.add_argument("--n0", type=int, default=5, help="Prior strength (default: 5)")
    args = parser.parse_args()

    bots, matches = _load_raw(args.data_dir)
    model = LadderModel.fit(bots, matches, n_0=args.n0)
    model.save(args.output_dir)


if __name__ == "__main__":
    main()
