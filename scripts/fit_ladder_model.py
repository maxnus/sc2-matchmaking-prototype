"""Fit Bayesian ladder model parameters from historical match data."""

import argparse
import json
import logging
from itertools import combinations
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent

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


# --- Data loading ---


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    bots = pd.read_csv(data_dir / "bots.csv")
    matches = pd.read_csv(data_dir / "matches.csv")
    log.info("Loaded %d bots, %d matches", len(bots), len(matches))
    return bots, matches


# --- Preprocessing ---


def preprocess_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Filter to valid games, classify categories, canonicalize bot ordering."""
    # Drop non-games
    invalid = matches["result_type"].isin(INVALID_RESULTS)
    log.info("Dropping %d invalid results (%s)", invalid.sum(),
             ", ".join(INVALID_RESULTS))
    df = matches[~invalid].copy()

    # Classify game category
    df["category"] = "normal"
    df.loc[df["is_timeout"] == True, "category"] = "timelimit"
    df.loc[df["result_type"].isin(ABNORMAL_RESULTS), "category"] = "abnormal"

    # Classify outcome from bot1's perspective (only meaningful for normal games)
    df["bot1_outcome"] = None
    normal = df["category"] == "normal"
    df.loc[normal & df["result_type"].isin(BOT1_WINS), "bot1_outcome"] = "win"
    df.loc[normal & df["result_type"].isin(BOT1_LOSES), "bot1_outcome"] = "loss"
    df.loc[normal & (df["result_type"] == "Tie"), "bot1_outcome"] = "draw"

    # Canonicalize to (bot_lo, bot_hi) where bot_lo < bot_hi
    df["bot_lo"] = df[["bot1_id", "bot2_id"]].min(axis=1).astype(int)
    df["bot_hi"] = df[["bot1_id", "bot2_id"]].max(axis=1).astype(int)

    # Determine outcome from bot_lo's perspective
    flipped = df["bot1_id"] != df["bot_lo"]
    df["lo_outcome"] = df["bot1_outcome"]
    df.loc[flipped & (df["bot1_outcome"] == "win"), "lo_outcome"] = "loss"
    df.loc[flipped & (df["bot1_outcome"] == "loss"), "lo_outcome"] = "win"

    # For abnormal games: bot_lo wins if bot_hi crashed, and vice versa
    abnormal = df["category"] == "abnormal"
    bot1_wins_abnormal = df["result_type"].isin(ABNORMAL_BOT1_WINS)
    df.loc[abnormal & bot1_wins_abnormal & ~flipped, "lo_outcome"] = "win"
    df.loc[abnormal & bot1_wins_abnormal & flipped, "lo_outcome"] = "loss"
    df.loc[abnormal & ~bot1_wins_abnormal & ~flipped, "lo_outcome"] = "loss"
    df.loc[abnormal & ~bot1_wins_abnormal & flipped, "lo_outcome"] = "win"

    log.info("Valid games: %d (normal=%d, timelimit=%d, abnormal=%d)",
             len(df),
             (df["category"] == "normal").sum(),
             (df["category"] == "timelimit").sum(),
             (df["category"] == "abnormal").sum())
    return df


# --- Global parameters ---


def compute_global_params(matches: pd.DataFrame, n_0: int) -> dict:
    total = len(matches)
    normal = matches[matches["category"] == "normal"]
    timelimit = matches[matches["category"] == "timelimit"]
    abnormal = matches[matches["category"] == "abnormal"]

    p_normal = len(normal) / total
    p_timelimit = len(timelimit) / total
    p_abnormal = len(abnormal) / total

    # Draw rate among normal games
    d = (normal["lo_outcome"] == "draw").sum() / len(normal)

    # Duration model for normal games
    dur_normal = normal["duration_minutes"].dropna()
    dur_normal = dur_normal[dur_normal > 0]
    log_dur = np.log(dur_normal.values)
    mu_0 = float(log_dur.mean())
    sigma = float(log_dur.std(ddof=0))

    # Duration model for abnormal games
    dur_abnormal = abnormal["duration_minutes"].dropna()
    dur_abnormal = dur_abnormal[dur_abnormal > 0]
    if len(dur_abnormal) > 0:
        log_dur_abn = np.log(dur_abnormal.values)
        mu_abnormal = float(log_dur_abn.mean())
        sigma_abnormal = float(log_dur_abn.std(ddof=0))
    else:
        mu_abnormal = mu_0
        sigma_abnormal = sigma

    gp = {
        "n_0": n_0,
        "p_normal": round(p_normal, 6),
        "p_timelimit": round(p_timelimit, 6),
        "p_abnormal": round(p_abnormal, 6),
        "d": round(d, 6),
        "mu_0": round(mu_0, 6),
        "sigma": round(sigma, 6),
        "mu_abnormal": round(mu_abnormal, 6),
        "sigma_abnormal": round(sigma_abnormal, 6),
    }

    log.info("Global parameters:")
    for k, v in gp.items():
        log.info("  %s = %s", k, v)
    return gp


# --- Per-matchup parameters ---


def compute_matchup_params(
    matches: pd.DataFrame, bots: pd.DataFrame, gp: dict
) -> pd.DataFrame:
    n_0 = gp["n_0"]
    d = gp["d"]
    mu_0 = gp["mu_0"]
    sigma = gp["sigma"]

    # Build bot lookup
    bot_info = bots.set_index("bot_id")[["name", "elo"]].to_dict("index")
    bot_ids = sorted(bots["bot_id"].values)

    # All possible pairs
    all_pairs = list(combinations(bot_ids, 2))
    pair_index = pd.MultiIndex.from_tuples(all_pairs, names=["bot_lo", "bot_hi"])

    # --- Aggregate category counts ---
    cat_counts = (
        matches.groupby(["bot_lo", "bot_hi", "category"])
        .size()
        .unstack(fill_value=0)
        .reindex(pair_index, fill_value=0)
    )
    for col in ["normal", "timelimit", "abnormal"]:
        if col not in cat_counts.columns:
            cat_counts[col] = 0

    # --- Aggregate normal game outcomes ---
    normal = matches[matches["category"] == "normal"]
    outcome_counts = (
        normal.groupby(["bot_lo", "bot_hi", "lo_outcome"])
        .size()
        .unstack(fill_value=0)
        .reindex(pair_index, fill_value=0)
    )
    for col in ["win", "loss", "draw"]:
        if col not in outcome_counts.columns:
            outcome_counts[col] = 0

    # --- Aggregate normal game durations ---
    dur_data = normal[["bot_lo", "bot_hi", "duration_minutes"]].dropna()
    dur_data = dur_data[dur_data["duration_minutes"] > 0].copy()
    dur_data["log_dur"] = np.log(dur_data["duration_minutes"])

    dur_agg = (
        dur_data.groupby(["bot_lo", "bot_hi"])["log_dur"]
        .agg(["count", "mean"])
        .rename(columns={"count": "n_dur", "mean": "x_bar"})
        .reindex(pair_index, fill_value=0)
    )

    # --- Build result DataFrame ---
    rows = []
    for bot_lo, bot_hi in all_pairs:
        info_lo = bot_info.get(bot_lo, {"name": "?", "elo": 1500})
        info_hi = bot_info.get(bot_hi, {"name": "?", "elo": 1500})
        elo_lo = info_lo["elo"]
        elo_hi = info_hi["elo"]

        # Category counts
        N_normal = int(cat_counts.loc[(bot_lo, bot_hi), "normal"])
        N_timelimit = int(cat_counts.loc[(bot_lo, bot_hi), "timelimit"])
        N_abnormal = int(cat_counts.loc[(bot_lo, bot_hi), "abnormal"])

        # Category Dirichlet posterior
        alpha_normal = n_0 * gp["p_normal"] + N_normal
        alpha_timelimit = n_0 * gp["p_timelimit"] + N_timelimit
        alpha_abnormal = n_0 * gp["p_abnormal"] + N_abnormal

        # Normal outcome counts
        W = int(outcome_counts.loc[(bot_lo, bot_hi), "win"])
        L = int(outcome_counts.loc[(bot_lo, bot_hi), "loss"])
        D = int(outcome_counts.loc[(bot_lo, bot_hi), "draw"])

        # ELO expected score for bot_lo
        E_A = 1.0 / (1.0 + 10.0 ** ((elo_hi - elo_lo) / 400.0))

        # Outcome Dirichlet posterior
        alpha_win = n_0 * (1 - d) * E_A + W
        alpha_draw = n_0 * d + D
        alpha_loss = n_0 * (1 - d) * (1 - E_A) + L

        # Duration posterior
        n_dur = int(dur_agg.loc[(bot_lo, bot_hi), "n_dur"])
        x_bar = float(dur_agg.loc[(bot_lo, bot_hi), "x_bar"])
        mu_duration = (n_0 * mu_0 + n_dur * x_bar) / (n_0 + n_dur)
        sigma_duration = sigma / sqrt(n_0 + n_dur)

        rows.append({
            "bot_lo": bot_lo,
            "bot_hi": bot_hi,
            "bot_lo_name": info_lo["name"],
            "bot_hi_name": info_hi["name"],
            "elo_lo": elo_lo,
            "elo_hi": elo_hi,
            "N_normal": N_normal,
            "N_timelimit": N_timelimit,
            "N_abnormal": N_abnormal,
            "alpha_normal": round(alpha_normal, 6),
            "alpha_timelimit": round(alpha_timelimit, 6),
            "alpha_abnormal": round(alpha_abnormal, 6),
            "W": W,
            "L": L,
            "D": D,
            "n_dur": n_dur,
            "x_bar": round(x_bar, 6) if n_dur > 0 else "",
            "E_A": round(E_A, 6),
            "alpha_win": round(alpha_win, 6),
            "alpha_draw": round(alpha_draw, 6),
            "alpha_loss": round(alpha_loss, 6),
            "mu_duration": round(mu_duration, 6),
            "sigma_duration": round(sigma_duration, 6),
        })

    result = pd.DataFrame(rows)
    has_data = ((result["N_normal"] + result["N_timelimit"] + result["N_abnormal"]) > 0).sum()
    log.info("Matchup parameters: %d pairs (%d with data, %d pure prior)",
             len(result), has_data, len(result) - has_data)
    return result


# --- Plots ---


def plot_duration(matches: pd.DataFrame, gp: dict, output_dir: Path):
    from plotly.subplots import make_subplots

    normal = matches[matches["category"] == "normal"]
    dur = normal["duration_minutes"].dropna()
    dur = dur[dur > 0]
    log_dur = np.log(dur.values)

    mu_0, sigma = gp["mu_0"], gp["sigma"]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Log-space", "Original space"])

    # Left: log-duration + normal fit
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

    # Right: duration + log-normal fit
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



def plot_calibration(matchups: pd.DataFrame, output_dir: Path):
    from plotly.subplots import make_subplots

    df = matchups.copy()
    df["total_normal"] = df["W"] + df["L"] + df["D"]
    df = df[df["total_normal"] >= 10]
    df["observed_winrate"] = df["W"] / df["total_normal"]
    df["posterior_winrate"] = df["alpha_win"] / (df["alpha_win"] + df["alpha_draw"] + df["alpha_loss"])
    n = len(df)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["ELO Prior", "Bayesian Posterior"])

    marker = dict(size=np.log1p(df["total_normal"]) * 3, opacity=0.5)
    text = df["bot_lo_name"] + " vs " + df["bot_hi_name"]

    # Left: ELO prior
    fig.add_trace(go.Scatter(
        x=df["E_A"], y=df["observed_winrate"],
        mode="markers", marker=marker, text=text,
        hovertemplate="%{text}<br>Predicted: %{x:.2f}<br>Observed: %{y:.2f}<extra></extra>",
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dash", color="gray"),
        name="Perfect calibration",
    ), row=1, col=1)

    # Right: posterior
    fig.add_trace(go.Scatter(
        x=df["posterior_winrate"], y=df["observed_winrate"],
        mode="markers", marker=marker, text=text,
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
        title=f"Win Rate Calibration (matchups with ≥10 normal games, n={n})",
    )

    path = output_dir / "win_rate_calibration.html"
    fig.write_html(str(path), include_mathjax="cdn")
    log.info("Wrote %s", path)



# --- Main ---


def main():
    parser = argparse.ArgumentParser(description="Fit Bayesian ladder model parameters")
    parser.add_argument("--data-dir", type=Path, default=_repo_root / "ladder_data")
    parser.add_argument("--output-dir", type=Path, default=_repo_root / "ladder_model")
    parser.add_argument("--n0", type=int, default=5, help="Prior strength (default: 5)")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    bots, matches = load_data(args.data_dir)
    matches = preprocess_matches(matches)

    gp = compute_global_params(matches, args.n0)
    matchups = compute_matchup_params(matches, bots, gp)

    # Write outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gp_path = args.output_dir / "global_params.json"
    with open(gp_path, "w") as f:
        json.dump(gp, f, indent=2)
    log.info("Wrote %s", gp_path)

    mp_path = args.output_dir / "matchup_params.csv"
    matchups.to_csv(mp_path, index=False)
    log.info("Wrote %s (%d matchups)", mp_path, len(matchups))

    if not args.no_plots:
        plots_dir = args.output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_duration(matches, gp, plots_dir)
        plot_calibration(matchups, plots_dir)


if __name__ == "__main__":
    main()
