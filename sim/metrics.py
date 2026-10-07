"""Scoring simulation runs against the outcome model's ground truth.

The sim draws every match from the fitted matchup posteriors, so it knows
each pair's true expected score. Matchmakers are scored against that rather
than against the sim's own ELO: on a non-transitive ladder a small ELO gap is
only a weak proxy for a close game, and the ELO scale itself depends on who
the matchmaker lets play whom.
"""

from math import sqrt

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from sim.common import MatchupParams

# A match whose favourite has at least this expected score is close to a
# foregone conclusion.
LOPSIDED_THRESHOLD = 0.9

_ELO_SLOPE = np.log(10.0) / 400.0

# (metric key, label, format spec, which direction is better) for the
# metrics shown in the CLI log and the report's summary table: one per goal,
# plus both views of rating accuracy, which pull against each other.
HEADLINE_METRICS = [
    ("favourite_expected_score.mean", "Favourite's true expected score", ".3f", "lower"),
    ("rating_accuracy.spearman", "Rating accuracy: Spearman ρ vs true ratings", ".3f", "higher"),
    ("rating_accuracy.rmse", "Rating accuracy: RMSE vs true ratings (ELO)", ".1f", "lower"),
    ("matches_per_bot.min", "Fewest matches played by any bot", ".0f", "higher"),
    ("unique_opponents_per_bot.mean", "Distinct opponents per bot", ".1f", "higher"),
    ("server_utilisation", "Server time in use", ".1%", "higher"),
]


# --- Ground truth ---


def expected_score(params: MatchupParams | None) -> float:
    """Expected score of `bot_lo` in one simulated match of the pair.

    Mirrors `simulate_match`: drawing the probabilities from their
    Dirichlet or Beta posteriors and then one outcome is the same as drawing
    from the posterior means. Time-limit games are draws, and bot_lo wins an
    abnormal game when bot_hi is the one that crashed. A pair without
    parameters is simulated from the symmetric prior, which scores 0.5.
    """
    if params is None:
        return 0.5
    alpha_category = params.alpha_normal + params.alpha_timelimit + params.alpha_abnormal
    alpha_outcome = params.alpha_win + params.alpha_draw + params.alpha_loss
    p_normal = params.alpha_normal / alpha_category
    p_timelimit = params.alpha_timelimit / alpha_category
    p_abnormal = params.alpha_abnormal / alpha_category
    normal_score = (params.alpha_win + 0.5 * params.alpha_draw) / alpha_outcome
    abnormal_score = params.alpha_crash_hi / (params.alpha_crash_lo + params.alpha_crash_hi)
    return p_normal * normal_score + p_timelimit * 0.5 + p_abnormal * abnormal_score


class GroundTruth:
    """True expected scores and ELO-scale ratings for a set of bots.

    `ratings` are the ELO-scale ratings that best explain the expected-score
    matrix with every pair weighted equally (maximum likelihood), centred on
    0. The matrix is not transitive, so they are a projection: the best
    single number per bot, not an exact description of every match-up.
    """

    def __init__(self, bot_ids: list[int], lookup: dict):
        self.bot_ids = list(bot_ids)
        self._index = {b: i for i, b in enumerate(self.bot_ids)}
        n = len(self.bot_ids)
        self.matrix = np.full((n, n), 0.5)
        for i, a in enumerate(self.bot_ids):
            for j in range(i + 1, n):
                b = self.bot_ids[j]
                e_lo = expected_score(lookup.get((min(a, b), max(a, b))))
                e_a = e_lo if a < b else 1.0 - e_lo
                self.matrix[i, j] = e_a
                self.matrix[j, i] = 1.0 - e_a
        self.ratings = dict(zip(self.bot_ids, _fit_ratings(self.matrix)))

    def favourite_score(self, bot_a: np.ndarray, bot_b: np.ndarray) -> np.ndarray:
        """The stronger side's true expected score, per (bot_a, bot_b) pair."""
        i = np.array([self._index[b] for b in bot_a])
        j = np.array([self._index[b] for b in bot_b])
        e = self.matrix[i, j]
        return np.maximum(e, 1.0 - e)

    def rating_accuracy(self, ratings: dict[int, float]) -> dict[str, float]:
        """How well `ratings` match the true ratings.

        Returns the Spearman ρ, the RMSE in ELO points, and the ratio of the
        two spreads (standard deviations), which is above 1 when ratings are
        stretched relative to the truth. `ratings` are centred first, since
        the sim's keep whatever mean they started with.
        """
        sim = np.array([ratings[b] for b in self.bot_ids], dtype=float)
        sim -= sim.mean()
        true = np.array([self.ratings[b] for b in self.bot_ids])
        return {
            "spearman": float(stats.spearmanr(sim, true).statistic),
            "rmse": sqrt(float(np.mean((sim - true) ** 2))),
            "spread_ratio": float(sim.std() / true.std()),
        }


def _fit_ratings(matrix: np.ndarray) -> np.ndarray:
    """Maximum-likelihood ELO ratings for a matrix of expected scores."""

    def nll_and_grad(r: np.ndarray) -> tuple[float, np.ndarray]:
        p = 1.0 / (1.0 + np.exp(-_ELO_SLOPE * (r[:, None] - r[None, :])))
        nll = -np.sum(matrix * np.log(p) + (1.0 - matrix) * np.log(1.0 - p))
        grad = -2.0 * _ELO_SLOPE * np.sum(matrix - p, axis=1)
        return nll, grad

    result = minimize(nll_and_grad, np.zeros(len(matrix)), jac=True, method="L-BFGS-B")
    return result.x - result.x.mean()


# --- Per-run summary ---


def _describe(values: pd.Series) -> dict:
    return {
        "mean": round(float(values.mean()), 1),
        "std": round(float(values.std()), 1),
        "min": int(values.min()),
        "max": int(values.max()),
    }


def compute_summary(
    matches: pd.DataFrame,
    elo_snapshots: pd.DataFrame,
    truth: GroundTruth,
    burn_in: int,
    slots: int,
) -> dict:
    """Summary of one run, over the matches completed after the burn-in.

    `matches` is the sim's match history (completion order) and
    `elo_snapshots` its ELO snapshots, and `slots` the number of server
    slots. The rating-accuracy trajectory covers
    the whole run; its summary metrics average the snapshots taken after the
    burn-in, except the spread ratio, which is taken at the end of the run.
    """
    bot_ids = truth.bot_ids
    window = matches.iloc[burn_in:]
    start_time = float(matches["time_end"].iloc[burn_in - 1]) if burn_in > 0 else 0.0
    end_time = float(window["time_end"].iloc[-1])
    days = (end_time - start_time) / (24 * 60)
    # Server time spent on matches within the window; a match that started
    # during the burn-in counts from the window's start.
    busy = (window["time_end"] - window["time_start"].clip(lower=start_time)).sum()

    sides = pd.DataFrame({
        "bot": np.concatenate([window["bot_a"], window["bot_b"]]),
        "opp": np.concatenate([window["bot_b"], window["bot_a"]]),
    })
    per_opponent = sides.groupby(["bot", "opp"]).size()
    matches_per_bot = sides.groupby("bot").size().reindex(bot_ids, fill_value=0)
    unique_opp = per_opponent.groupby(level="bot").size().reindex(bot_ids, fill_value=0)
    max_repeat = per_opponent.groupby(level="bot").max().reindex(bot_ids, fill_value=0)

    favourite = truth.favourite_score(window["bot_a"].to_numpy(), window["bot_b"].to_numpy())

    trajectory = []
    for match_count, snap in elo_snapshots.groupby("match_count"):
        accuracy = truth.rating_accuracy(dict(zip(snap["bot_id"], snap["elo"])))
        trajectory.append({
            "match_count": int(match_count),
            "spearman": round(accuracy["spearman"], 4),
            "rmse": round(accuracy["rmse"], 1),
            "spread_ratio": round(accuracy["spread_ratio"], 4),
        })
    settled = [p for p in trajectory if p["match_count"] >= burn_in]

    metrics = {
        "favourite_expected_score": {
            "mean": round(float(favourite.mean()), 4),
            "median": round(float(np.median(favourite)), 4),
        },
        "lopsided_rate": round(float((favourite >= LOPSIDED_THRESHOLD).mean()), 4),
        "elo_diff": {
            "mean": round(float(window["elo_diff"].mean()), 1),
            "std": round(float(window["elo_diff"].std()), 1),
            "median": round(float(window["elo_diff"].median()), 1),
        },
        "rating_accuracy": {
            "spearman": round(float(np.mean([p["spearman"] for p in settled])), 4),
            "rmse": round(float(np.mean([p["rmse"] for p in settled])), 1),
            "spread_ratio_end": trajectory[-1]["spread_ratio"],
        },
        "matches_per_bot": _describe(matches_per_bot),
        "unique_opponents_per_bot": _describe(unique_opp),
        "max_repeat_opponent": _describe(max_repeat),
        "matches_per_day": round(len(window) / days, 1),
        "server_utilisation": round(float(busy / (slots * (end_time - start_time))), 4),
        "duration_minutes": {
            "mean": round(float(window["duration_minutes"].mean()), 2),
            "std": round(float(window["duration_minutes"].std()), 2),
        },
        "category_rates": {
            k: round(v, 4)
            for k, v in window["category"].value_counts(normalize=True).items()
        },
    }
    return {
        "total_matches": len(matches),
        "burn_in": burn_in,
        "window_matches": len(window),
        "window_days": round(days, 2),
        "metrics": metrics,
        "rating_accuracy_trajectory": trajectory,
    }


# --- Across runs ---


def _flatten(d: dict, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in d.items():
        if isinstance(value, dict):
            out.update(_flatten(value, f"{prefix}{key}."))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            out[f"{prefix}{key}"] = value
    return out


def mean_ci95(values) -> tuple[float, float | None]:
    """Mean and half-width of its 95% t-interval (`None` for one value)."""
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, None
    sem = values.std(ddof=1) / sqrt(len(values))
    return mean, float(stats.t.ppf(0.975, len(values) - 1) * sem)


def aggregate_metrics(runs: list[dict]) -> dict:
    """Mean and 95% confidence interval of every scalar metric across runs.

    `runs` are the `metrics` dicts of independent runs (different seeds).
    Keys are flattened with dots, e.g. `elo_diff.mean`.
    """
    flat = [_flatten(r) for r in runs]
    keys = list(dict.fromkeys(k for f in flat for k in f))
    out = {}
    for key in keys:
        values = [f[key] for f in flat if key in f]
        mean, ci = mean_ci95(values)
        out[key] = {
            "mean": round(mean, 4),
            "ci95": None if ci is None else round(ci, 4),
            "values": values,
        }
    return out


def format_metric(stat: dict, fmt: str) -> str:
    """`mean ± ci95` with the given format spec."""
    text = format(stat["mean"], fmt)
    if stat["ci95"] is not None:
        text += f" ± {format(stat['ci95'], fmt)}"
    return text
