"""Shared utilities for ladder simulation scripts."""

import json
import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

K = 16  # ELO adjustment per game


# --- Value enums for match outcomes / categories ---


class Category(StrEnum):
    """Match category. Values are used verbatim in CSV output."""
    NORMAL = "normal"
    TIMELIMIT = "timelimit"
    ABNORMAL = "abnormal"


class Outcome(StrEnum):
    """Match outcome from one bot's perspective (string, not score).

    Distinct from the numeric `outcome_a` column (1.0/0.5/0.0) used in
    match_history; `Outcome` is the categorical label in matchup_params.
    """
    WIN = "win"
    LOSS = "loss"
    DRAW = "draw"


# --- Model parameter schemas ---


@dataclass(frozen=True)
class GlobalParams:
    """Global Bayesian parameters for the ladder outcome model.

    Produced by `model.fit.LadderModel.fit(...)` and serialized to
    `global_params.json`. Consumed by the sim via `simulate_match`.
    """
    n_0: int                 # prior strength (in units of matches)
    p_normal: float          # marginal probability of a normal game
    p_timelimit: float       # probability of hitting the 60-min cap
    p_abnormal: float        # probability of crash / bot-level timeout
    d: float                 # draw rate within normal games
    mu_0: float              # log-duration mean for normal games
    sigma: float             # log-duration std for normal games
    mu_abnormal: float       # log-duration mean for abnormal games
    sigma_abnormal: float    # log-duration std for abnormal games


@dataclass(frozen=True)
class MatchupParams:
    """Per-pair posterior parameters used by `simulate_match`.

    The full matchups DataFrame (with names, ELOs, counts, etc.) is kept
    for analysis; this struct is the minimal subset needed to sample
    match outcomes.
    """
    alpha_normal: float
    alpha_timelimit: float
    alpha_abnormal: float
    alpha_win: float
    alpha_draw: float
    alpha_loss: float
    mu_duration: float
    sigma_duration: float


# --- Model loading ---


def load_model(
    data_dir: Path, model_dir: Path
) -> tuple[pd.DataFrame, GlobalParams, dict]:
    """Load bots, global params, and matchup params.

    Returns (active_bots_df, global_params, matchup_lookup).
    matchup_lookup is keyed by (bot_lo, bot_hi) tuples.
    """
    bots = pd.read_csv(data_dir / "bots.csv")
    bots = bots[bots["active"] == True].reset_index(drop=True)

    with open(model_dir / "global_params.json") as f:
        gp = GlobalParams(**json.load(f))

    matchups = pd.read_csv(model_dir / "matchup_params.csv")
    lookup: dict[tuple[int, int], MatchupParams] = {}
    for _, row in matchups.iterrows():
        key = (int(row["bot_lo"]), int(row["bot_hi"]))
        lookup[key] = MatchupParams(
            alpha_normal=float(row["alpha_normal"]),
            alpha_timelimit=float(row["alpha_timelimit"]),
            alpha_abnormal=float(row["alpha_abnormal"]),
            alpha_win=float(row["alpha_win"]),
            alpha_draw=float(row["alpha_draw"]),
            alpha_loss=float(row["alpha_loss"]),
            mu_duration=float(row["mu_duration"]),
            sigma_duration=float(row["sigma_duration"]),
        )

    log.info("Loaded %d active bots, %d matchup params", len(bots), len(lookup))
    return bots, gp, lookup


# --- Match simulation ---


def simulate_match(
    bot_a: int, bot_b: int, lookup: dict, gp: GlobalParams, rng: np.random.Generator
) -> tuple[float, float, str]:
    """Simulate a single match between two bots.

    Returns (outcome_a, duration_minutes, category).
    outcome_a: 1.0 (win), 0.5 (draw), 0.0 (loss) from bot_a's perspective.
    """
    bot_lo, bot_hi = min(bot_a, bot_b), max(bot_a, bot_b)
    params = lookup.get((bot_lo, bot_hi))

    categories = [Category.NORMAL, Category.TIMELIMIT, Category.ABNORMAL]
    if params is None:
        # No matchup data — use pure prior (shouldn't happen for active bots)
        category = rng.choice(categories, p=[gp.p_normal, gp.p_timelimit, gp.p_abnormal])
        if category == Category.NORMAL:
            outcome_lo = rng.choice([1.0, 0.5, 0.0], p=[0.5 - gp.d / 2, gp.d, 0.5 - gp.d / 2])
            duration = _sample_duration(gp.mu_0, gp.sigma, rng)
        elif category == Category.TIMELIMIT:
            outcome_lo = 0.5
            duration = 60.0
        else:
            outcome_lo = rng.choice([1.0, 0.0])
            duration = _sample_duration(gp.mu_abnormal, gp.sigma_abnormal, rng)
        return (outcome_lo if bot_a == bot_lo else 1.0 - outcome_lo, duration, category)

    # Sample category from Dirichlet posterior
    cat_probs = rng.dirichlet([
        params.alpha_normal,
        params.alpha_timelimit,
        params.alpha_abnormal,
    ])
    category = rng.choice(categories, p=cat_probs)

    if category == Category.NORMAL:
        # Sample outcome from Dirichlet posterior (bot_lo's perspective)
        outcome_probs = rng.dirichlet([
            params.alpha_win,
            params.alpha_draw,
            params.alpha_loss,
        ])
        outcome_idx = rng.choice([0, 1, 2], p=outcome_probs)
        outcome_lo = [1.0, 0.5, 0.0][outcome_idx]
        duration = _sample_duration(params.mu_duration, params.sigma_duration, rng)

    elif category == Category.TIMELIMIT:
        outcome_lo = 0.5
        duration = 60.0

    else:  # abnormal
        outcome_lo = rng.choice([1.0, 0.0])
        duration = _sample_duration(gp.mu_abnormal, gp.sigma_abnormal, rng)

    # Flip if bot_a is bot_hi
    outcome_a = outcome_lo if bot_a == bot_lo else 1.0 - outcome_lo
    return outcome_a, duration, category


def _sample_duration(mu: float, sigma: float, rng: np.random.Generator) -> float:
    """Sample from a log-normal truncated at 60 minutes."""
    for _ in range(100):
        d = float(np.exp(rng.normal(mu, sigma)))
        if d < 60.0:
            return round(d, 2)
    return 59.99  # fallback


# --- ELO ---


def update_elo(ratings: dict, bot_a: int, bot_b: int, outcome_a: float):
    """Update ELO ratings in-place."""
    r_a, r_b = ratings[bot_a], ratings[bot_b]
    e_a = 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))
    ratings[bot_a] = r_a + K * (outcome_a - e_a)
    ratings[bot_b] = r_b + K * ((1.0 - outcome_a) - (1.0 - e_a))


# --- Summary ---


def _last_n_matches_per_bot(
    df: pd.DataFrame, bot_ids: list[int], last_n: int
) -> pd.DataFrame:
    """Return the union of each bot's last N matches (by match index order)."""
    bot_matches: dict[int, list[int]] = {b: [] for b in bot_ids}
    for idx, row in df.iterrows():
        bot_matches[row["bot_a"]].append(idx)
        bot_matches[row["bot_b"]].append(idx)

    keep = set()
    for b in bot_ids:
        keep.update(bot_matches[b][-last_n:])

    return df.loc[sorted(keep)]


def compute_summary(
    match_history: list[dict],
    bot_ids: list[int],
    last_n_matches: int | None = None,
) -> dict:
    df = pd.DataFrame(match_history)

    if last_n_matches is not None:
        df_window = _last_n_matches_per_bot(df, bot_ids, last_n_matches)
    else:
        df_window = df

    # Matches per bot (within window)
    bot_a_counts = df_window["bot_a"].value_counts()
    bot_b_counts = df_window["bot_b"].value_counts()
    matches_per_bot = (bot_a_counts.add(bot_b_counts, fill_value=0)).reindex(bot_ids, fill_value=0)

    # Opponent variety (within window)
    opp_counts: dict[int, dict[int, int]] = {b: {} for b in bot_ids}
    for _, row in df_window.iterrows():
        a, b = row["bot_a"], row["bot_b"]
        opp_counts[a][b] = opp_counts[a].get(b, 0) + 1
        opp_counts[b][a] = opp_counts[b].get(a, 0) + 1
    unique_opp = pd.Series({b: len(v) for b, v in opp_counts.items()})
    max_repeat = pd.Series({b: max(v.values()) if v else 0 for b, v in opp_counts.items()})

    categories = df_window["category"].value_counts(normalize=True).to_dict()
    outcomes = df_window["outcome_a"].value_counts(normalize=True)
    outcome_rates = {
        "win": float(outcomes.get(1.0, 0)),
        "draw": float(outcomes.get(0.5, 0)),
        "loss": float(outcomes.get(0.0, 0)),
    }

    result: dict = {
        "total_matches": len(df),
    }
    if last_n_matches is not None:
        result["last_n_matches"] = last_n_matches
        result["window_matches"] = len(df_window)
    result.update({
        "matches_per_bot": {
            "mean": round(float(matches_per_bot.mean()), 1),
            "std": round(float(matches_per_bot.std()), 1),
            "min": int(matches_per_bot.min()),
            "max": int(matches_per_bot.max()),
        },
        "unique_opponents_per_bot": {
            "mean": round(float(unique_opp.mean()), 1),
            "std": round(float(unique_opp.std()), 1),
            "min": int(unique_opp.min()),
            "max": int(unique_opp.max()),
        },
        "max_repeat_opponent": {
            "mean": round(float(max_repeat.mean()), 1),
            "std": round(float(max_repeat.std()), 1),
            "min": int(max_repeat.min()),
            "max": int(max_repeat.max()),
        },
        "elo_diff": {
            "mean": round(float(df_window["elo_diff"].mean()), 1),
            "std": round(float(df_window["elo_diff"].std()), 1),
            "median": round(float(df_window["elo_diff"].median()), 1),
        },
        "duration_minutes": {
            "mean": round(float(df_window["duration_minutes"].mean()), 2),
            "std": round(float(df_window["duration_minutes"].std()), 2),
        },
        "category_rates": {k: round(v, 4) for k, v in categories.items()},
        "outcome_rates": outcome_rates,
    })
    return result


def compute_elo_convergence(
    elo_snapshots: list[dict], match_counts: dict[int, int] | None = None
) -> dict:
    """Compute ELO convergence stats from snapshots.

    elo_snapshots: list of dicts with at least 'elo' and a grouping key.
        - If 'match_count' is present (proposed system), use it directly.
        - If 'round' is present (current system), use match_counts to map.

    match_counts: mapping from round number to cumulative match count.
        Required when snapshots use 'round' instead of 'match_count'.

    Returns dict with 'elo_std_trajectory' and 'matches_to_90pct'.
    """
    df = pd.DataFrame(elo_snapshots)

    if "match_count" in df.columns:
        group_col = "match_count"
    elif "round" in df.columns:
        if match_counts is None:
            raise ValueError("match_counts required for round-based snapshots")
        df["match_count"] = df["round"].map(match_counts)
        group_col = "match_count"
    else:
        raise ValueError("Snapshots must have 'match_count' or 'round' column")

    trajectory = []
    for match_count, group in df.groupby(group_col):
        trajectory.append({
            "match_count": int(match_count),
            "elo_std": round(float(group["elo"].std()), 1),
        })
    trajectory.sort(key=lambda x: x["match_count"])

    # Find matches to reach 90% of final ELO spread
    if len(trajectory) < 2:
        return {"elo_std_trajectory": trajectory, "matches_to_90pct": None}

    final_std = trajectory[-1]["elo_std"]
    threshold = 0.9 * final_std
    matches_to_90 = None
    for point in trajectory:
        if point["elo_std"] >= threshold:
            matches_to_90 = point["match_count"]
            break

    # ELO stability: per-bot std across the last few snapshots
    # Use the second half of snapshots as the "settled" window
    snapshot_values = sorted(df[group_col].unique())
    n_settled = max(1, len(snapshot_values) // 2)
    settled_snapshots = snapshot_values[-n_settled:]
    settled = df[df[group_col].isin(settled_snapshots)]

    per_bot_std = settled.groupby("bot_id")["elo"].std().fillna(0)
    stability = {
        "mean": round(float(per_bot_std.mean()), 1),
        "std": round(float(per_bot_std.std()), 1),
        "max": round(float(per_bot_std.max()), 1),
        "n_snapshots": n_settled,
    }

    return {
        "elo_std_trajectory": trajectory,
        "matches_to_90pct": matches_to_90,
        "final_elo_std": final_std,
        "elo_stability": stability,
    }
