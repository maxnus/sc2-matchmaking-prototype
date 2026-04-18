"""Bayesian ladder outcome model: fit from raw matches, save/load, plot."""

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from itertools import combinations
from math import sqrt
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from sim.common import Category, GlobalParams, Outcome
from sim.paths import DATA_DIR

_own_dir = Path(__file__).resolve().parent

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# Result types that never produced a real game
INVALID_RESULTS = {"InitializationError", "MatchCancelled", "Error"}

# Result types where bot1 wins / bot2 wins (for normal games)
BOT1_WINS = {"Player1Win"}
BOT1_LOSES = {"Player2Win"}

# Abnormal terminations (crash or bot-level timeout)
ABNORMAL_RESULTS = {"Player1Crash", "Player2Crash", "Player1TimeOut", "Player2TimeOut"}
# Within abnormal: bot1 wins if the OTHER bot crashed/timed out
ABNORMAL_BOT1_WINS = {"Player2Crash", "Player2TimeOut"}


# --- Pure-function helpers (stateless transforms) ---


def _load_raw(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    bots = pd.read_csv(data_dir / "bots.csv")
    matches = pd.read_csv(data_dir / "matches.csv")
    log.info("Loaded %d bots, %d matches", len(bots), len(matches))
    return bots, matches


def _preprocess_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Filter to valid games, classify categories, canonicalize bot ordering."""
    invalid = matches["result_type"].isin(INVALID_RESULTS)
    log.info("Dropping %d invalid results (%s)", invalid.sum(),
             ", ".join(INVALID_RESULTS))
    df = matches[~invalid].copy()

    df["category"] = Category.NORMAL
    df.loc[df["is_timeout"] == True, "category"] = Category.TIMELIMIT
    df.loc[df["result_type"].isin(ABNORMAL_RESULTS), "category"] = Category.ABNORMAL

    df["bot1_outcome"] = None
    normal = df["category"] == Category.NORMAL
    df.loc[normal & df["result_type"].isin(BOT1_WINS), "bot1_outcome"] = Outcome.WIN
    df.loc[normal & df["result_type"].isin(BOT1_LOSES), "bot1_outcome"] = Outcome.LOSS
    df.loc[normal & (df["result_type"] == "Tie"), "bot1_outcome"] = Outcome.DRAW

    df["bot_lo"] = df[["bot1_id", "bot2_id"]].min(axis=1).astype(int)
    df["bot_hi"] = df[["bot1_id", "bot2_id"]].max(axis=1).astype(int)

    flipped = df["bot1_id"] != df["bot_lo"]
    df["lo_outcome"] = df["bot1_outcome"]
    df.loc[flipped & (df["bot1_outcome"] == Outcome.WIN), "lo_outcome"] = Outcome.LOSS
    df.loc[flipped & (df["bot1_outcome"] == Outcome.LOSS), "lo_outcome"] = Outcome.WIN

    abnormal = df["category"] == Category.ABNORMAL
    bot1_wins_abnormal = df["result_type"].isin(ABNORMAL_BOT1_WINS)
    df.loc[abnormal & bot1_wins_abnormal & ~flipped, "lo_outcome"] = Outcome.WIN
    df.loc[abnormal & bot1_wins_abnormal & flipped, "lo_outcome"] = Outcome.LOSS
    df.loc[abnormal & ~bot1_wins_abnormal & ~flipped, "lo_outcome"] = Outcome.LOSS
    df.loc[abnormal & ~bot1_wins_abnormal & flipped, "lo_outcome"] = Outcome.WIN

    log.info("Valid games: %d (normal=%d, timelimit=%d, abnormal=%d)",
             len(df),
             (df["category"] == Category.NORMAL).sum(),
             (df["category"] == Category.TIMELIMIT).sum(),
             (df["category"] == Category.ABNORMAL).sum())
    return df


def _compute_global_params(matches: pd.DataFrame, n_0: int) -> GlobalParams:
    total = len(matches)
    normal = matches[matches["category"] == Category.NORMAL]
    timelimit = matches[matches["category"] == Category.TIMELIMIT]
    abnormal = matches[matches["category"] == Category.ABNORMAL]

    p_normal = len(normal) / total
    p_timelimit = len(timelimit) / total
    p_abnormal = len(abnormal) / total

    d = (normal["lo_outcome"] == Outcome.DRAW).sum() / len(normal)

    dur_normal = normal["duration_minutes"].dropna()
    dur_normal = dur_normal[dur_normal > 0]
    log_dur = np.log(dur_normal.values)
    mu_0 = float(log_dur.mean())
    sigma = float(log_dur.std(ddof=0))

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
        d=round(d, 6),
        mu_0=round(mu_0, 6),
        sigma=round(sigma, 6),
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
    alpha_normal: float
    alpha_timelimit: float
    alpha_abnormal: float
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


def _compute_matchup_params(
    matches: pd.DataFrame, bots: pd.DataFrame, gp: GlobalParams,
) -> pd.DataFrame:
    n_0 = gp.n_0
    d = gp.d
    mu_0 = gp.mu_0
    sigma = gp.sigma

    bot_info = bots.set_index("bot_id")[["name", "elo"]].to_dict("index")
    bot_ids = sorted(bots["bot_id"].values)

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

        alpha_normal = n_0 * gp.p_normal + N_normal
        alpha_timelimit = n_0 * gp.p_timelimit + N_timelimit
        alpha_abnormal = n_0 * gp.p_abnormal + N_abnormal

        W = int(outcome_counts.loc[(bot_lo, bot_hi), Outcome.WIN])
        L = int(outcome_counts.loc[(bot_lo, bot_hi), Outcome.LOSS])
        D = int(outcome_counts.loc[(bot_lo, bot_hi), Outcome.DRAW])

        E_A = 1.0 / (1.0 + 10.0 ** ((elo_hi - elo_lo) / 400.0))

        alpha_win = n_0 * (1 - d) * E_A + W
        alpha_draw = n_0 * d + D
        alpha_loss = n_0 * (1 - d) * (1 - E_A) + L

        n_dur = int(dur_agg.loc[(bot_lo, bot_hi), "n_dur"])
        x_bar = float(dur_agg.loc[(bot_lo, bot_hi), "x_bar"])
        mu_duration = (n_0 * mu_0 + n_dur * x_bar) / (n_0 + n_dur)
        sigma_duration = sigma / sqrt(n_0 + n_dur)

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
            alpha_normal=round(alpha_normal, 6),
            alpha_timelimit=round(alpha_timelimit, 6),
            alpha_abnormal=round(alpha_abnormal, 6),
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
      - `bots`, `raw_matches`, `processed_matches`: inputs retained for
        plotting. Not persisted — absent after `load()`.
    """

    def __init__(
        self,
        global_params: GlobalParams,
        matchups: pd.DataFrame,
        bots: Optional[pd.DataFrame] = None,
        raw_matches: Optional[pd.DataFrame] = None,
        processed_matches: Optional[pd.DataFrame] = None,
    ):
        self.global_params = global_params
        self.matchups = matchups
        self.bots = bots
        self.raw_matches = raw_matches
        self.processed_matches = processed_matches

    @classmethod
    def fit(
        cls, bots: pd.DataFrame, matches: pd.DataFrame, n_0: int = 5,
    ) -> "LadderModel":
        """Fit the full model from raw match data."""
        processed = _preprocess_matches(matches)
        gp = _compute_global_params(processed, n_0)
        mp = _compute_matchup_params(processed, bots, gp)
        return cls(
            global_params=gp,
            matchups=mp,
            bots=bots,
            raw_matches=matches,
            processed_matches=processed,
        )

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

    # --- Plots ---

    def plot_duration(self, output_dir: Path) -> None:
        if self.processed_matches is None:
            raise RuntimeError(
                "plot_duration requires processed_matches; only available "
                "after fit() (not load())."
            )
        _plot_duration(self.processed_matches, self.global_params, output_dir)

    def plot_calibration(self, output_dir: Path) -> None:
        _plot_calibration(self.matchups, output_dir)

    def plot_all(self, output_dir: Path) -> None:
        """Write every available plot into `output_dir`."""
        output_dir.mkdir(parents=True, exist_ok=True)
        if self.processed_matches is not None:
            self.plot_duration(output_dir)
        self.plot_calibration(output_dir)


# --- Plot implementations (private) ---


def _plot_duration(matches: pd.DataFrame, gp: GlobalParams, output_dir: Path) -> None:
    normal = matches[matches["category"] == Category.NORMAL]
    dur = normal["duration_minutes"].dropna()
    dur = dur[dur > 0]
    log_dur = np.log(dur.values)

    mu_0, sigma = gp.mu_0, gp.sigma

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Log-space", "Original space"])

    fig.add_trace(go.Histogram(
        x=log_dur, nbinsx=100, histnorm="probability density",
        name="Observed", opacity=0.7, showlegend=False,
    ), row=1, col=1)
    x_log = np.linspace(log_dur.min(), log_dur.max(), 200)
    fig.add_trace(go.Scatter(
        x=x_log, y=stats.norm.pdf(x_log, mu_0, sigma),
        mode="lines", name=rf"$\mathcal{{N}}(\mu_0={mu_0:.3f},\, \sigma={sigma:.3f})$",
        line=dict(width=2),
    ), row=1, col=1)

    fig.add_trace(go.Histogram(
        x=dur.values, nbinsx=100, histnorm="probability density",
        name="Observed", opacity=0.7, showlegend=False,
    ), row=1, col=2)
    x_dur = np.linspace(0.01, 60, 500)
    pdf = stats.lognorm.pdf(x_dur, s=sigma, scale=np.exp(mu_0))
    fig.add_trace(go.Scatter(
        x=x_dur, y=pdf, mode="lines", name="Log-normal fit",
        line=dict(width=2),
    ), row=1, col=2)

    fig.update_xaxes(title_text="log(duration in minutes)", row=1, col=1)
    fig.update_xaxes(title_text="Duration (minutes)", row=1, col=2)
    fig.update_yaxes(title_text="Density", row=1, col=1)
    fig.update_yaxes(title_text="Density", row=1, col=2)
    fig.update_layout(title="Match Duration Distribution (Normal Games)")

    path = output_dir / "duration.html"
    fig.write_html(str(path), include_mathjax="cdn")
    log.info("Wrote %s", path)


def _plot_calibration(matchups: pd.DataFrame, output_dir: Path) -> None:
    df = matchups.copy()
    df["total_normal"] = df["W"] + df["L"] + df["D"]
    df = df[df["total_normal"] >= 10]
    df["observed_winrate"] = df["W"] / df["total_normal"]
    df["loss_rate"] = df["L"] / df["total_normal"]
    df["posterior_winrate"] = df["alpha_win"] / (df["alpha_win"] + df["alpha_draw"] + df["alpha_loss"])
    df["posterior_lossrate"] = df["alpha_loss"] / (df["alpha_win"] + df["alpha_draw"] + df["alpha_loss"])
    n = len(df)

    # Each pair contributes two points: the canonical (bot_lo) perspective and
    # the mirror (bot_hi) perspective. Canonicalization by bot_id is not
    # rating-symmetric, so plotting only one direction makes the scatter and
    # any fit on it biased relative to the inherently symmetric calibration.
    text_lo = df["bot_lo_name"] + " vs " + df["bot_hi_name"]
    text_hi = df["bot_hi_name"] + " vs " + df["bot_lo_name"]
    marker_sizes = np.log1p(df["total_normal"]) * 3

    x_elo = np.concatenate([df["E_A"].values, 1 - df["E_A"].values])
    y_elo = np.concatenate([df["observed_winrate"].values, df["loss_rate"].values])
    x_post = np.concatenate([df["posterior_winrate"].values, df["posterior_lossrate"].values])
    y_post = y_elo
    text_all = np.concatenate([text_lo.values, text_hi.values])
    sizes_all = np.concatenate([marker_sizes.values, marker_sizes.values])
    marker = dict(size=sizes_all, opacity=0.5)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["ELO Prior", "Bayesian Posterior"])

    fig.add_trace(go.Scatter(
        x=x_elo, y=y_elo,
        mode="markers", marker=marker, text=text_all,
        hovertemplate="%{text}<br>Predicted: %{x:.2f}<br>Observed: %{y:.2f}<extra></extra>",
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash", color="gray"),
        name="Perfect calibration",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=x_post, y=y_post,
        mode="markers", marker=marker, text=text_all,
        hovertemplate="%{text}<br>Predicted: %{x:.2f}<br>Observed: %{y:.2f}<extra></extra>",
        showlegend=False,
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash", color="gray"),
        showlegend=False,
    ), row=1, col=2)

    fig.update_xaxes(title_text=r"$E_\text{A}$ (ELO predicted)", row=1, col=1)
    fig.update_xaxes(title_text=r"$\mathbb{E}[p_\text{win}]$ (posterior mean)", row=1, col=2)
    fig.update_yaxes(title_text=r"$\hat{p}_\text{win}$ (observed)", row=1, col=1)
    fig.update_yaxes(title_text=r"$\hat{p}_\text{win}$ (observed)", row=1, col=2)
    fig.update_layout(
        title=f"Win Rate Calibration (matchups with ≥10 normal games, n={n} pairs × 2 directions)",
    )

    path = output_dir / "win_rate_calibration.html"
    fig.write_html(str(path), include_mathjax="cdn")
    log.info("Wrote %s", path)


# --- CLI ---


def main():
    parser = argparse.ArgumentParser(description="Fit Bayesian ladder model parameters")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=_own_dir)
    parser.add_argument("--n0", type=int, default=5, help="Prior strength (default: 5)")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    bots, matches = _load_raw(args.data_dir)
    model = LadderModel.fit(bots, matches, n_0=args.n0)
    model.save(args.output_dir)

    if not args.no_plots:
        model.plot_all(args.output_dir / "plots")


if __name__ == "__main__":
    main()
