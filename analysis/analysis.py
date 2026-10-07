"""Build the project report: one HTML page with the data, the model, the
simulation, the matchmakers and the results.

Usage:
    python analysis/analysis.py                       # auto-discover matchmakers/*
    python analysis/analysis.py <dir1> <dir2> [...]   # explicit output dirs

The text lives in `report.md` next to this script: Markdown with TeX maths
(`$...$`, `$$...$$`) and two kinds of placeholders, filled in here:

    <!-- figure: name -->      a figure or table, on a line of its own
    <!-- value: key [fmt] -->  a number, formatted with the optional format spec

Each output dir holds one matchmaker's runs (`seed_*/`) and their aggregate
`summary.json`, as written by `sim.cli.run_and_write`; its name is taken
from the folder (e.g. `matchmakers/stochastic/output` → `stochastic`).
Writes `report.html`, with a contents sidebar built from the `##` and `###`
headings.
"""

import argparse
import html
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from markdown_it import MarkdownIt
from mdit_py_plugins.anchors import anchors_plugin
from mdit_py_plugins.dollarmath import dollarmath_plugin
from plotly.offline import get_plotlyjs_version
from plotly.subplots import make_subplots
from scipy import stats

from sim.common import Category, GlobalParams, Outcome, load_model, preprocess_matches
from sim.metrics import HEADLINE_METRICS, LOPSIDED_THRESHOLD, GroundTruth, format_metric, mean_ci95
from sim.paths import DATA_DIR, MODEL_DIR, REPO_ROOT

_own_dir = Path(__file__).resolve().parent

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# Plotly's default qualitative palette — enough for 8+ matchmakers.
PALETTE = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA",
    "#FFA15A", "#19D3F3", "#FF6692", "#B6E880",
]

# Pairs whose true ratings are closer than this count as near-equal.
NEAR_EQUAL_GAP = 50

KATEX = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist"


@dataclass
class Run:
    """One seed's outputs. `matches` holds only the post-burn-in window."""
    matches: pd.DataFrame
    summary: dict


@dataclass
class SimResult:
    name: str
    color: str
    summary: dict  # aggregate across runs
    runs: list[Run]

    @property
    def window_matches(self) -> pd.DataFrame:
        """Post-burn-in matches of all runs, concatenated."""
        return pd.concat([r.matches for r in self.runs], ignore_index=True)


def _infer_name(d: Path) -> str:
    """`matchmakers/stochastic/output` → `stochastic`; otherwise the dir name."""
    return d.parent.name if d.name == "output" else d.name


def load_simulation(sim_dir: Path, name: str, color: str) -> SimResult:
    with open(sim_dir / "summary.json") as f:
        summary = json.load(f)
    runs = []
    for run_name in summary["runs"]:
        run_dir = sim_dir / run_name
        with open(run_dir / "summary.json") as f:
            run_summary = json.load(f)
        matches = pd.read_csv(run_dir / "matches.csv.gz")
        window = matches.iloc[run_summary["burn_in"]:].reset_index(drop=True)
        runs.append(Run(matches=window, summary=run_summary))
    return SimResult(name=name, color=color, summary=summary, runs=runs)


def _pair_counts(matches: pd.DataFrame) -> dict[tuple[int, int], int]:
    lo = matches[["bot_a", "bot_b"]].min(axis=1)
    hi = matches[["bot_a", "bot_b"]].max(axis=1)
    return pd.DataFrame({"lo": lo, "hi": hi}).groupby(["lo", "hi"]).size().to_dict()


# --- Numbers quoted in the text ---


def _concurrency(matches: pd.DataFrame) -> pd.DataFrame:
    """Per bot, how many matches it had running (wall clock) from each
    start or end of one of its matches (`t`) until the next (`dt`)."""
    m = matches.dropna(subset=["match_started", "result_created"])
    start = pd.to_datetime(m["match_started"], format="ISO8601")
    end = pd.to_datetime(m["result_created"], format="ISO8601")
    events = pd.concat([
        pd.DataFrame({"bot": m[col], "t": t, "d": d})
        for col in ("bot1_id", "bot2_id")
        for t, d in ((start, 1), (end, -1))
    ])
    # At equal times, ends (-1) sort before starts (+1).
    events = events.sort_values(["bot", "t", "d"])
    events["running"] = events.groupby("bot")["d"].cumsum()
    events["dt"] = (events.groupby("bot")["t"].shift(-1) - events["t"]).dt.total_seconds()
    return events


def _drift_ratio(processed: pd.DataFrame, min_games: int = 10) -> float:
    """How much pairs' win rates change between the first and second half of
    the data, relative to chance.

    For pairs with at least `min_games` wins and losses in each half: the
    mean squared difference of the two win rates, divided by the variance
    binomial noise alone would give it. About 1 means no drift.
    """
    decided = processed[
        (processed["category"] == Category.NORMAL)
        & processed["lo_outcome"].isin([Outcome.WIN, Outcome.LOSS])
    ].copy()
    decided["lo_win"] = (decided["lo_outcome"] == Outcome.WIN).astype(int)
    started = pd.to_datetime(decided["match_started"], format="ISO8601")
    late = started >= started.min() + (started.max() - started.min()) / 2
    halves = [
        decided[mask].groupby(["bot_lo", "bot_hi"])["lo_win"].agg(["sum", "size"])
        for mask in (~late, late)
    ]
    both = halves[0].join(halves[1], lsuffix="_a", rsuffix="_b", how="inner")
    both = both[(both["size_a"] >= min_games) & (both["size_b"] >= min_games)]
    p_a, p_b = both["sum_a"] / both["size_a"], both["sum_b"] / both["size_b"]
    pooled = (both["sum_a"] + both["sum_b"]) / (both["size_a"] + both["size_b"])
    noise = pooled * (1 - pooled) * (1 / both["size_a"] + 1 / both["size_b"])
    keep = noise > 0
    return float(((p_a - p_b)[keep] ** 2 / noise[keep]).mean())


def _mean_concurrent(matches: pd.DataFrame) -> float:
    """Average number of matches running at once on the real servers."""
    m = matches.dropna(subset=["match_started", "result_created"])
    start = pd.to_datetime(m["match_started"], format="ISO8601")
    end = pd.to_datetime(m["result_created"], format="ISO8601")
    busy = (end - start).dt.total_seconds().sum()
    return float(busy / (end.max() - start.min()).total_seconds())


def data_values(
    raw: pd.DataFrame, processed: pd.DataFrame, bots: pd.DataFrame, max_parallel: int,
) -> dict:
    started = pd.to_datetime(raw["match_started"], format="ISO8601")
    active = bots[bots["active"] == True]
    data_enabled = dict(zip(bots["bot_id"], bots["bot_data_enabled"]))
    events = _concurrency(raw)
    single = events[events["bot"].map(data_enabled) == True].groupby("bot")["running"].max()
    busy = events[(events["bot"].map(data_enabled) == False) & (events["running"] > 0)]
    within_cap = busy.loc[busy["running"] <= max_parallel, "dt"].sum() / busy["dt"].sum()

    abnormal = processed[processed["category"] == Category.ABNORMAL]
    crashed = pd.concat([
        abnormal.loc[abnormal["lo_outcome"] == Outcome.LOSS, "bot_lo"],
        abnormal.loc[abnormal["lo_outcome"] == Outcome.WIN, "bot_hi"],
    ]).value_counts()
    games = pd.concat([processed["bot_lo"], processed["bot_hi"]]).value_counts()
    crash_rate = (crashed.reindex(games.index, fill_value=0) / games)
    crash_rate = crash_rate[crash_rate.index.isin(active["bot_id"])]

    normal = processed[processed["category"] == Category.NORMAL]
    normal = normal[normal["duration_minutes"] > 0]
    wall = (
        pd.to_datetime(normal["result_created"], format="ISO8601")
        - pd.to_datetime(normal["match_started"], format="ISO8601")
    ).dt.total_seconds() / 60
    n_days = (started.max().normalize() - started.min().normalize()).days + 1

    return {
        "data.n_matches": len(raw),
        "data.n_rounds": raw["round"].nunique(),
        "data.first_day": started.min().date().isoformat(),
        "data.last_day": started.max().date().isoformat(),
        "data.n_days": n_days,
        "data.matches_per_day": len(raw) / n_days,
        "data.n_bots": len(bots),
        "data.n_active": len(active),
        "data.n_active_data": int(active["bot_data_enabled"].sum()),
        "data.n_invalid": len(raw) - len(processed),
        **{
            f"data.n_{c}": int((processed["category"] == c).sum())
            for c in Category
        },
        "data.single_instance_share": float((single <= 1).mean()),
        "data.parallel_within_cap": float(within_cap),
        "data.crash_prone": int((crash_rate > 0.1).sum()),
        "data.median_crash_rate": float(crash_rate.median()),
        "data.max_crash_rate": float(crash_rate.max()),
        "data.wall_game_ratio": float((wall / normal["duration_minutes"]).median()),
        "data.drift_ratio": _drift_ratio(processed),
        "data.mean_concurrent": _mean_concurrent(raw),
    }


def model_values(gp: GlobalParams, matchups: pd.DataFrame, bots: pd.DataFrame) -> dict:
    active = set(bots.loc[bots["active"] == True, "bot_id"])
    pairs = matchups[matchups["bot_lo"].isin(active) & matchups["bot_hi"].isin(active)]
    games = pairs["N_normal"] + pairs["N_timelimit"] + pairs["N_abnormal"]

    # Pairs decided often enough to tell a one-sided match-up from a close one.
    decided = matchups[(matchups["W"] + matchups["L"]) >= 10]
    n = (decided["W"] + decided["L"]).to_numpy()
    win_rate = decided["W"].to_numpy() / n
    p = decided["E_A"].to_numpy()
    # Chance of a ≥90% or ≤10% win rate if every pair played exactly to its ELO expectation.
    p_lopsided = (
        stats.binom.sf(np.ceil(0.9 * n) - 1, n, p) + stats.binom.cdf(np.floor(0.1 * n), n, p)
    )

    return {
        **{f"model.{k}": v for k, v in asdict(gp).items()},
        "model.median_duration": float(np.exp(gp.mu_0)),
        "model.n_pairs": len(pairs),
        "model.n_pairs_10": int((games >= 10).sum()),
        "model.n_pairs_no_data": int((games == 0).sum()),
        "model.n_decided_10": len(decided),
        "model.lopsided_observed": float(((win_rate >= 0.9) | (win_rate <= 0.1)).mean()),
        "model.lopsided_elo": float(p_lopsided.mean()),
    }


def _near_equal_favourite(truth: GroundTruth) -> float:
    """Mean favourite's expected score over pairs within `NEAR_EQUAL_GAP` true rating."""
    r = np.array([truth.ratings[b] for b in truth.bot_ids])
    upper = np.triu_indices(len(r), 1)
    near = np.abs(r[:, None] - r[None, :])[upper] < NEAR_EQUAL_GAP
    favourite = np.maximum(truth.matrix, 1.0 - truth.matrix)[upper]
    return float(favourite[near].mean())


def truth_values(truth: GroundTruth, bots: pd.DataFrame) -> dict:
    true = np.array([truth.ratings[b] for b in truth.bot_ids])
    real = bots.set_index("bot_id").loc[truth.bot_ids, "elo"].to_numpy(dtype=float)
    return {
        "truth.rating_std": float(true.std()),
        "truth.near_equal_gap": NEAR_EQUAL_GAP,
        "truth.near_equal_favourite": _near_equal_favourite(truth),
        "truth.near_equal_elo": 1.0 / (1.0 + 10.0 ** (-NEAR_EQUAL_GAP / 400.0)),
        "truth.real_spread_ratio": float(real.std() / true.std()),
        "metrics.lopsided_threshold": LOPSIDED_THRESHOLD,
    }


def sim_values(sims: list[SimResult], n_bots: int) -> dict:
    config = sims[0].summary["config"]
    values = {f"config.{k}": v for k, v in config.items()}
    values["config.measured_matches"] = config["total_matches"] - config["burn_in"]
    values["config.matches_per_bot"] = 2 * config["total_matches"] / n_bots
    for sim in sims:
        values.update({f"{sim.name}.config.{k}": v for k, v in sim.summary["config"].items()})
        values.update({f"{sim.name}.{k}": v["mean"] for k, v in sim.summary["metrics"].items()})
    if "rung" in {s.name for s in sims}:
        values["rung.picks_per_round"] = (
            values["rung.config.rung_picks"] + values["rung.config.wildcard_picks"]
        )
    return values


# --- Tables and figures ---


def summary_table(sims: list[SimResult]) -> str:
    header = "".join(
        f"<th style='color:{s.color}'>{html.escape(s.name)}<br>"
        f"<span class='muted'>{len(s.runs)} runs</span></th>"
        for s in sims
    )
    rows = []
    for key, label, fmt, better in HEADLINE_METRICS:
        cells = "".join(
            f"<td>{format_metric(s.summary['metrics'][key], fmt)}</td>" for s in sims
        )
        rows.append(
            f"<tr><th>{html.escape(label)}<br><span class='muted'>{better} is better</span></th>"
            f"{cells}</tr>"
        )
    return (
        "<table class='summary'>\n"
        f"<thead><tr><th>Metric (mean ± 95% CI across runs)</th>{header}</tr></thead>\n"
        f"<tbody>{''.join(rows)}</tbody>\n</table>\n"
    )


def _style(fig: go.Figure, height: int, legend_below: bool = False) -> go.Figure:
    legend = (
        dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0)
        if legend_below else
        dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    fig.update_layout(
        template="plotly_white", height=height, legend=legend,
        margin=dict(l=60, r=20, t=40, b=50),
        font=dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", size=13),
    )
    return fig


def _density_trace(values: np.ndarray, bins: np.ndarray, name: str, color: str) -> go.Scatter:
    """Histogram as a step line, so overlaid distributions stay readable and
    the page doesn't embed every raw value."""
    density, edges = np.histogram(values, bins=bins, density=True)
    return go.Scatter(
        x=edges, y=np.append(density, density[-1]), mode="lines", line_shape="hv",
        name=name, line=dict(color=color, width=2),
    )


def _density_bars(values: np.ndarray, bins: np.ndarray, name: str) -> go.Bar:
    density, edges = np.histogram(values, bins=bins, density=True)
    return go.Bar(
        x=(edges[:-1] + edges[1:]) / 2, y=density, width=np.diff(edges),
        name=name, marker=dict(color="#9aa5b1"), opacity=0.7,
    )


def plot_calibration(matchups: pd.DataFrame) -> go.Figure:
    """Observed win rates against the ELO prior and the posterior mean, for
    pairs with at least 10 normal games, from both sides of each pair."""
    df = matchups.copy()
    df["total_normal"] = df["W"] + df["L"] + df["D"]
    df = df[df["total_normal"] >= 10]
    alpha = df["alpha_win"] + df["alpha_draw"] + df["alpha_loss"]

    # Each pair contributes two points, one per side: ordering pairs by bot id
    # is arbitrary, so one side alone would make the plot asymmetric.
    observed = np.concatenate([df["W"] / df["total_normal"], df["L"] / df["total_normal"]])
    elo = np.concatenate([df["E_A"], 1 - df["E_A"]])
    posterior = np.concatenate([df["alpha_win"] / alpha, df["alpha_loss"] / alpha])
    text = np.concatenate([
        df["bot_lo_name"] + " vs " + df["bot_hi_name"],
        df["bot_hi_name"] + " vs " + df["bot_lo_name"],
    ])
    sizes = np.tile(np.log1p(df["total_normal"].to_numpy()) * 3, 2)

    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.1,
        subplot_titles=["ELO prediction", "Posterior mean (in-sample)"],
    )
    for col, predicted in enumerate([elo, posterior], start=1):
        fig.add_trace(go.Scatter(
            x=predicted, y=observed, mode="markers", text=text,
            marker=dict(size=sizes, opacity=0.45, color="#636EFA"),
            hovertemplate="%{text}<br>Predicted: %{x:.2f}<br>Observed: %{y:.2f}<extra></extra>",
            showlegend=False,
        ), row=1, col=col)
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=col)
        fig.update_xaxes(title_text="Predicted win rate", range=[0, 1], row=1, col=col)
    fig.update_yaxes(title_text="Observed win rate", range=[0, 1], row=1, col=1)
    return _style(fig, 430)


def plot_durations(processed: pd.DataFrame, gp: GlobalParams, sim: SimResult) -> go.Figure:
    """Normal-game durations: the global log-normal fit, and the real against
    the simulated distribution."""
    normal = processed[processed["category"] == Category.NORMAL]
    dur = normal["duration_minutes"].dropna()
    dur = dur[dur > 0].to_numpy()
    sim_matches = sim.window_matches
    sim_dur = sim_matches.loc[sim_matches["category"] == "normal", "duration_minutes"].to_numpy()

    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.1,
        subplot_titles=["log(duration in minutes)", "Duration (minutes)"],
    )
    log_bins = np.linspace(np.log(dur).min(), np.log(60), 61)
    fig.add_trace(_density_bars(np.log(dur), log_bins, "Real games"), row=1, col=1)
    x = np.linspace(log_bins[0], log_bins[-1], 200)
    fig.add_trace(go.Scatter(
        x=x, y=stats.norm.pdf(x, gp.mu_0, gp.sigma), mode="lines",
        name="Global log-normal fit", line=dict(color="#EF553B", width=2),
    ), row=1, col=1)

    bins = np.linspace(0, 60, 61)
    fig.add_trace(_density_bars(dur, bins, "Real games"), row=1, col=2)
    fig.data[-1].showlegend = False
    fig.add_trace(_density_trace(sim_dur, bins, f"Simulated ({sim.name})", "#00CC96"), row=1, col=2)
    fig.update_yaxes(title_text="Density", row=1, col=1)
    fig.update_layout(bargap=0)
    return _style(fig, 400, legend_below=True)


def plot_favourite_score(sims: list[SimResult], truth: GroundTruth) -> go.Figure:
    fig = go.Figure()
    bins = np.linspace(0.5, 1.0, 26)
    for sim in sims:
        m = sim.window_matches
        fav = truth.favourite_score(m["bot_a"].to_numpy(), m["bot_b"].to_numpy())
        fig.add_trace(_density_trace(fav, bins, sim.name, sim.color))
    fig.add_vline(x=LOPSIDED_THRESHOLD, line=dict(color="gray", dash="dash", width=1))
    fig.update_layout(xaxis_title="Favourite's true expected score", yaxis_title="Density")
    return _style(fig, 420)


def plot_elo_diff(sims: list[SimResult]) -> go.Figure:
    fig = go.Figure()
    all_diffs = np.concatenate([s.window_matches["elo_diff"].to_numpy() for s in sims])
    bins = np.linspace(0, np.quantile(all_diffs, 0.995), 51)
    for sim in sims:
        fig.add_trace(_density_trace(sim.window_matches["elo_diff"].to_numpy(), bins, sim.name, sim.color))
    fig.update_layout(
        xaxis_title="Absolute ELO difference at dispatch (simulation's own ratings)",
        yaxis_title="Density",
    )
    return _style(fig, 420)


def plot_rating_accuracy(sims: list[SimResult], burn_in: int) -> go.Figure:
    """RMSE, Spearman ρ and spread ratio against the true ratings over each
    run, with the mean and 95% CI across runs."""
    panels = [
        ("rmse", "RMSE (ELO points)"),
        ("spearman", "Spearman ρ"),
        ("spread_ratio", "ELO spread ÷ true spread"),
    ]
    fig = make_subplots(
        rows=1, cols=len(panels), horizontal_spacing=0.09,
        subplot_titles=[title for _, title in panels],
    )
    for col, (key, _) in enumerate(panels, start=1):
        for sim in sims:
            traj = pd.DataFrame([
                p for r in sim.runs for p in r.summary["rating_accuracy_trajectory"]
            ])
            ci_by_count = traj.groupby("match_count")[key].apply(mean_ci95)
            x = ci_by_count.index.to_numpy()
            mean = np.array([m for m, _ in ci_by_count])
            ci = np.array([c or 0.0 for _, c in ci_by_count])
            fig.add_trace(go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([mean + ci, (mean - ci)[::-1]]),
                fill="toself", fillcolor=sim.color, opacity=0.2, line=dict(width=0),
                showlegend=False, hoverinfo="skip",
            ), row=1, col=col)
            fig.add_trace(go.Scatter(
                x=x, y=mean, mode="lines", name=sim.name,
                line=dict(color=sim.color, width=2), showlegend=(col == 1),
            ), row=1, col=col)
        fig.add_vline(x=burn_in, line=dict(color="gray", dash="dash", width=1), row=1, col=col)
    fig.update_xaxes(title_text="Matches completed")
    return _style(fig, 420, legend_below=True)


def plot_matches_per_bot(
    sims: list[SimResult], bot_ids: list[int], bot_names: dict[int, str],
) -> go.Figure:
    fig = go.Figure()
    for sim in sims:
        matches = sim.runs[0].matches
        a = matches["bot_a"].value_counts()
        b = matches["bot_b"].value_counts()
        counts = a.add(b, fill_value=0).reindex(bot_ids, fill_value=0)

        sides = pd.concat([
            matches[["bot_a", "duration_minutes"]].rename(columns={"bot_a": "bot"}),
            matches[["bot_b", "duration_minutes"]].rename(columns={"bot_b": "bot"}),
        ])
        avg_dur = sides.groupby("bot")["duration_minutes"].mean().reindex(bot_ids, fill_value=0)

        fig.add_trace(go.Scatter(
            x=counts.to_numpy(), y=avg_dur.to_numpy(),
            mode="markers", name=sim.name,
            marker=dict(color=sim.color, size=8, opacity=0.6),
            text=[bot_names[b_id] for b_id in bot_ids],
            hovertemplate="%{text}<br>Matches: %{x}<br>Avg duration: %{y:.1f} min<extra></extra>",
        ))
    fig.update_layout(xaxis_title="Matches played", yaxis_title="Avg game duration (minutes)")
    return _style(fig, 460)


def _schedules(matches: pd.DataFrame) -> pd.DataFrame:
    """Each bot's matches in the order it played them: one row per bot and
    match, with its opponent and `game`, the bot's 1st, 2nd, ... game."""
    sides = pd.concat([
        matches[["match_id", "time_start", "bot_a", "bot_b"]].rename(columns={"bot_a": "bot", "bot_b": "opp"}),
        matches[["match_id", "time_start", "bot_b", "bot_a"]].rename(columns={"bot_b": "bot", "bot_a": "opp"}),
    ]).sort_values(["bot", "time_start", "match_id"], ignore_index=True)
    sides["game"] = sides.groupby("bot").cumcount() + 1
    return sides


def plot_distinct_opponents(sims: list[SimResult]) -> go.Figure:
    """Distinct opponents a bot has met against the games it has played,
    median and interquartile band over bots and runs."""
    fig = go.Figure()
    for sim in sims:
        curves, games_per_bot = [], []
        for run in sim.runs:
            sides = _schedules(run.matches)
            sides["distinct"] = (~sides.duplicated(["bot", "opp"])).groupby(sides["bot"]).cumsum()
            curves.append(sides[["game", "distinct"]])
            games_per_bot.append(sides.groupby("bot")["game"].max())
        curves = pd.concat(curves)
        # Stop where fewer than 90% of the bots have played that many games,
        # so the median doesn't drift towards the busiest bots.
        last_game = int(pd.concat(games_per_bot).quantile(0.1))
        stats = curves[curves["game"] <= last_game].groupby("game")["distinct"].quantile([0.25, 0.5, 0.75]).unstack()
        x = stats.index.to_numpy()
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([stats[0.75].to_numpy(), stats[0.25].to_numpy()[::-1]]),
            fill="toself", fillcolor=sim.color, opacity=0.15, line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=x, y=stats[0.5].to_numpy(), mode="lines", name=sim.name,
            line=dict(color=sim.color, width=2),
            hovertemplate="%{x} games: %{y:.0f} distinct opponents<extra>" + sim.name + "</extra>",
        ))
    fig.update_layout(xaxis_title="Games played since the burn-in", yaxis_title="Distinct opponents met")
    return _style(fig, 440)


def plot_rematch_gaps(sims: list[SimResult]) -> go.Figure:
    """Cumulative share of repeat meetings by the number of games since the
    pair's previous meeting, counted in the bot's own games."""
    fig = go.Figure()
    for sim in sims:
        gaps = []
        for run in sim.runs:
            sides = _schedules(run.matches)
            gaps.append((sides["game"] - sides.groupby(["bot", "opp"])["game"].shift()).dropna().to_numpy())
        gaps = np.sort(np.concatenate(gaps))
        x = np.arange(1, int(gaps.max()) + 1)
        share = np.searchsorted(gaps, x, side="right") / len(gaps)
        fig.add_trace(go.Scatter(
            x=x, y=share, mode="lines", line_shape="hv", name=sim.name,
            line=dict(color=sim.color, width=2),
            hovertemplate="within %{x} games: %{y:.0%}<extra>" + sim.name + "</extra>",
        ))
    fig.update_layout(
        xaxis=dict(title="Games since the two bots last met", type="log",
                   tickvals=[1, 2, 5, 10, 20, 50, 100, 200, 500]),
        yaxis=dict(title="Share of repeat meetings", tickformat=".0%", range=[0, 1]),
    )
    return _style(fig, 440)


def plot_matchup_heatmap(
    sim: SimResult, bot_names: dict[int, str], true_ratings: dict[int, float],
) -> go.Figure:
    """Heatmap of one run's post-burn-in match counts, sorted by true rating."""
    sorted_bots = sorted(true_ratings, key=true_ratings.get, reverse=True)
    bot_idx = {b: i for i, b in enumerate(sorted_bots)}
    n = len(sorted_bots)

    grid = np.zeros((n, n), dtype=int)
    for (a, b), count in _pair_counts(sim.runs[0].matches).items():
        i, j = bot_idx[a], bot_idx[b]
        grid[i][j] += count
        grid[j][i] += count

    labels = [f"{bot_names[b]} ({true_ratings[b]:+.0f})" for b in sorted_bots]
    fig = go.Figure(data=go.Heatmap(
        z=grid, x=labels, y=labels,
        colorscale="YlOrRd", zmin=0,
        hovertemplate="%{y} vs %{x}<br>Matches: %{z}<extra></extra>",
        colorbar=dict(title="Matches"),
    ))
    fig.update_layout(
        xaxis=dict(tickangle=90, tickfont=dict(size=8)),
        yaxis=dict(tickfont=dict(size=8), autorange="reversed"),
    )
    fig = _style(fig, 820)
    fig.update_layout(margin=dict(l=140, r=20, t=20, b=140))
    return fig


def _figure_html(fig: go.Figure) -> str:
    div = pio.to_html(
        fig, full_html=False, include_plotlyjs=False,
        config={"responsive": True, "displaylogo": False},
    )
    return f"<figure class='plot'>{div}</figure>"


def heatmaps(sims: list[SimResult], bot_names: dict[int, str], truth: GroundTruth) -> str:
    """One collapsible heatmap per matchmaker."""
    return "\n".join(
        f"<details><summary>{html.escape(sim.name)}</summary>\n"
        f"{_figure_html(plot_matchup_heatmap(sim, bot_names, truth.ratings))}\n</details>"
        for sim in sims
    )


# --- Page assembly ---


_VALUE = re.compile(r"<!--\s*value:\s*([\w.\-]+)(?:\s+(\S+))?\s*-->")
_FIGURE = re.compile(r"<!--\s*figure:\s*([\w\-]+)\s*-->")


def _fill_values(text: str, values: dict) -> str:
    def replace(match: re.Match) -> str:
        key, fmt = match[1], match[2]
        if key not in values:
            raise KeyError(f"report.md uses unknown value {key!r}")
        return format(values[key], fmt) if fmt else str(values[key])
    return _VALUE.sub(replace, text)


def _fill_figures(body: str, blocks: dict[str, str]) -> str:
    used = set()

    def replace(match: re.Match) -> str:
        name = match[1]
        if name not in blocks:
            raise KeyError(f"report.md uses unknown figure {name!r}")
        used.add(name)
        return blocks[name]

    body = _FIGURE.sub(replace, body)
    for name in blocks.keys() - used:
        log.warning("Figure %r is not used in report.md", name)
    return body


def _render_math(content: str, options: dict) -> str:
    """Leave TeX for KaTeX's auto-render, in its delimiters."""
    if options["display_mode"]:
        return rf"\[{html.escape(content)}\]"
    return rf"\({html.escape(content)}\)"


def render_markdown(text: str) -> tuple[str, str, list[tuple[int, str, str]]]:
    """Render `text` to HTML; returns (title, body, contents).

    The title is the `#` heading; contents lists (level, id, text) of the
    `##` and `###` headings.
    """
    md = (
        MarkdownIt("commonmark", {"html": True})
        .enable("table")
        .use(dollarmath_plugin, renderer=_render_math)
        .use(anchors_plugin, min_level=2, max_level=3)
    )
    env: dict = {}
    tokens = md.parse(text, env)
    title, contents = "", []
    for i, token in enumerate(tokens):
        if token.type != "heading_open":
            continue
        heading = tokens[i + 1].content
        if token.tag == "h1":
            title = heading
        elif token.tag in ("h2", "h3"):
            contents.append((int(token.tag[1]), token.attrs["id"], heading))
    return title, md.renderer.render(tokens, md.options, env), contents


_PAGE_CSS = """
:root { --text: #1f2328; --muted: #59636e; --border: #d1d9e0; --side: #f6f8fa;
        --hover: #e7ebef; --accent: #2557c7; --accent-bg: #e6edfb; }
* { box-sizing: border-box; }
body { margin: 0; background: #fff; color: var(--text);
       font: 16px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
.layout { display: grid; grid-template-columns: 280px minmax(0, 1fr); }
nav.sidebar { position: sticky; top: 0; height: 100vh; overflow-y: auto;
              background: var(--side); border-right: 1px solid var(--border); padding: 24px 14px; }
nav .site-title { font-weight: 600; font-size: 15px; line-height: 1.35; margin: 0 8px 16px; }
nav summary { display: none; }
nav ul { list-style: none; margin: 0; padding: 0; }
nav a { display: block; padding: 3px 8px; border-radius: 6px; color: var(--muted);
        text-decoration: none; font-size: 14px; line-height: 1.4; }
nav li.level-2 { margin-top: 6px; }
nav li.level-2 > a { color: var(--text); font-weight: 500; }
nav li.level-3 > a { padding-left: 20px; font-size: 13.5px; }
nav a:hover { background: var(--hover); }
nav a.active { color: var(--accent); background: var(--accent-bg); }
main { min-width: 0; padding: 40px 48px 96px; }
article { max-width: 940px; margin: 0 auto; }
article h1 { font-size: 2.1rem; line-height: 1.2; margin: 0 0 20px; }
article h2 { font-size: 1.6rem; margin: 64px 0 12px; padding-bottom: 6px;
             border-bottom: 1px solid var(--border); scroll-margin-top: 16px; }
article h3 { font-size: 1.25rem; margin: 40px 0 8px; scroll-margin-top: 16px; }
article h4 { font-size: 1.05rem; margin: 28px 0 6px; }
article a { color: var(--accent); }
article code { font-size: 0.88em; background: var(--side); padding: 1px 5px; border-radius: 4px; }
article table { border-collapse: collapse; margin: 18px 0; font-size: 15px;
                display: block; overflow-x: auto; max-width: 100%; }
article th, article td { border-bottom: 1px solid var(--border); padding: 6px 12px;
                         text-align: left; vertical-align: top; }
article thead th { border-bottom: 2px solid #9aa5b1; }
.math.block { overflow-x: auto; overflow-y: hidden; padding: 2px 0; }
figure.plot { margin: 18px 0 8px; }
article details { border: 1px solid var(--border); border-radius: 8px; padding: 6px 14px; margin: 10px 0; }
article details > summary { cursor: pointer; font-weight: 500; padding: 4px 0; }
table.summary td { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
table.summary thead th { text-align: right; }
table.summary th:first-child { text-align: left; font-weight: normal; min-width: 240px; }
.muted { color: var(--muted); font-size: 0.85em; font-weight: normal; }
@media (max-width: 999px) {
  .layout { display: block; }
  nav.sidebar { position: sticky; top: 0; z-index: 10; height: auto; max-height: 100vh;
                border-right: none; border-bottom: 1px solid var(--border); padding: 8px 16px; }
  nav summary { display: block; cursor: pointer; font-weight: 600; padding: 4px 0; }
  nav .site-title { display: none; }
  main { padding: 20px 16px 64px; }
  article h1 { font-size: 1.7rem; }
  article h2, article h3 { scroll-margin-top: 64px; }
}
"""

_PAGE_JS = """
const toc = document.getElementById('toc');
const wide = matchMedia('(min-width: 1000px)');
const syncToc = () => { toc.open = wide.matches; };
syncToc();
wide.addEventListener('change', syncToc);
toc.addEventListener('click', e => {
  if (e.target.closest('a') && !wide.matches) toc.open = false;
});

// Highlight the section being read.
const links = new Map([...toc.querySelectorAll('a')].map(a => [a.hash.slice(1), a]));
const headings = [...document.querySelectorAll('article h2[id], article h3[id]')];
let ticking = false;
function highlight() {
  ticking = false;
  let current = headings[0];
  for (const h of headings) {
    if (h.getBoundingClientRect().top < 120) current = h; else break;
  }
  links.forEach(a => a.classList.toggle('active', a.hash.slice(1) === current.id));
}
addEventListener('scroll', () => { if (!ticking) { ticking = true; requestAnimationFrame(highlight); } },
                 { passive: true });
highlight();

// Plots in a closed <details> are drawn without a width; size them when opened.
document.querySelectorAll('article details').forEach(d => d.addEventListener('toggle', () => {
  if (d.open) d.querySelectorAll('.js-plotly-plot').forEach(p => Plotly.Plots.resize(p));
}));
"""


def write_page(title: str, body: str, contents: list[tuple[int, str, str]], path: Path) -> None:
    items = "\n".join(
        f"<li class='level-{level}'><a href='#{anchor}'>{html.escape(text)}</a></li>"
        for level, anchor, text in contents
    )
    math_delimiters = (
        "[{left: '\\\\[', right: '\\\\]', display: true},"
        " {left: '\\\\(', right: '\\\\)', display: false}]"
    )
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<link rel="stylesheet" href="{KATEX}/katex.min.css">
<script defer src="{KATEX}/katex.min.js"></script>
<script defer src="{KATEX}/contrib/auto-render.min.js"
  onload="renderMathInElement(document.querySelector('article'), {{delimiters: {math_delimiters}, throwOnError: false}})"></script>
<script src="https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"></script>
<style>{_PAGE_CSS}</style>
</head>
<body>
<div class="layout">
<nav class="sidebar">
<div class="site-title">{html.escape(title)}</div>
<details id="toc" open>
<summary>Contents</summary>
<ul>
{items}
</ul>
</details>
</nav>
<main>
<article>
{body}
</article>
</main>
</div>
<script>{_PAGE_JS}</script>
</body>
</html>
"""
    path.write_text(page, encoding="utf-8")
    log.info("Wrote %s", path)


# --- Main ---


def _discover_matchmaker_dirs() -> list[Path]:
    """Find `matchmakers/*/output` dirs that have an aggregate summary.json."""
    root = REPO_ROOT / "matchmakers"
    return sorted(
        d / "output" for d in root.iterdir()
        if d.is_dir() and (d / "output" / "summary.json").exists()
    )


def main():
    parser = argparse.ArgumentParser(description="Build the project report")
    parser.add_argument("dirs", type=Path, nargs="*",
                        help="Simulation output directories (default: every "
                             "`matchmakers/*/output` with a summary.json)")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--source", type=Path, default=_own_dir / "report.md")
    parser.add_argument("--output-dir", type=Path, default=_own_dir)
    args = parser.parse_args()

    dirs = args.dirs or _discover_matchmaker_dirs()
    if len(dirs) < 1:
        parser.error("no simulation output dirs found")
    names = [_infer_name(d) for d in dirs]

    log.info("Loading %d simulations: %s", len(dirs), ", ".join(names))
    sims = [
        load_simulation(d, name, PALETTE[i % len(PALETTE)])
        for i, (d, name) in enumerate(zip(dirs, names))
    ]

    bots, gp, lookup = load_model(args.data_dir, args.model_dir)
    bot_ids = bots["bot_id"].tolist()
    bot_names = dict(zip(bots["bot_id"], bots["name"]))
    truth = GroundTruth(bot_ids, lookup)
    matchups = pd.read_csv(args.model_dir / "matchup_params.csv")
    all_bots = pd.read_csv(args.data_dir / "bots.csv")
    raw = pd.read_csv(args.data_dir / "matches.csv")
    processed = preprocess_matches(raw)
    burn_in = sims[0].summary["config"]["burn_in"]

    values = {
        **data_values(raw, processed, all_bots, sims[0].summary["config"]["max_parallel"]),
        **model_values(gp, matchups, all_bots),
        **truth_values(truth, bots),
        **sim_values(sims, len(bot_ids)),
    }

    log.info("Generating report...")
    blocks = {
        "calibration": _figure_html(plot_calibration(matchups)),
        "durations": _figure_html(plot_durations(processed, gp, sims[0])),
        "summary-table": summary_table(sims),
        "favourite-score": _figure_html(plot_favourite_score(sims, truth)),
        "elo-diff": _figure_html(plot_elo_diff(sims)),
        "rating-accuracy": _figure_html(plot_rating_accuracy(sims, burn_in)),
        "matches-per-bot": _figure_html(plot_matches_per_bot(sims, bot_ids, bot_names)),
        "distinct-opponents": _figure_html(plot_distinct_opponents(sims)),
        "rematch-gaps": _figure_html(plot_rematch_gaps(sims)),
        "heatmaps": heatmaps(sims, bot_names, truth),
    }

    text = _fill_values(args.source.read_text(encoding="utf-8"), values)
    title, body, contents = render_markdown(text)
    body = _fill_figures(body, blocks)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_page(title, body, contents, args.output_dir / "report.html")


if __name__ == "__main__":
    main()
