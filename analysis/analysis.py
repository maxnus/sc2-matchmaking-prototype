"""Generate a single-page comparison report across matchmaker simulation runs.

Usage:
    python analysis/analysis.py                       # auto-discover matchmakers/*
    python analysis/analysis.py <dir1> <dir2> [...]   # explicit output dirs

Each output dir holds one matchmaker's runs (`seed_*/`) and their aggregate
`summary.json`, as written by `sim.cli.run_and_write`. Writes `report.html`
with a summary table and every plot stacked on one page. Labels come from the
folder name (e.g. `matchmakers/stochastic/output` → `stochastic`).
"""

import argparse
import html
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sim.common import load_model
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


# --- Summary table ---


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
        "<div class='table-wrap'><table class='summary'>\n"
        f"<thead><tr><th>Metric (mean ± 95% CI across runs)</th>{header}</tr></thead>\n"
        f"<tbody>{''.join(rows)}</tbody>\n</table></div>\n"
    )


# --- Plots (each returns a go.Figure) ---


def _density_trace(values: np.ndarray, bins: np.ndarray, name: str, color: str) -> go.Scatter:
    """Histogram as a step line, so overlaid distributions stay readable and
    the page doesn't embed every raw value."""
    density, edges = np.histogram(values, bins=bins, density=True)
    return go.Scatter(
        x=edges, y=np.append(density, density[-1]), mode="lines", line_shape="hv",
        name=name, line=dict(color=color, width=2),
    )


def plot_favourite_score(sims: list[SimResult], truth: GroundTruth) -> go.Figure:
    fig = go.Figure()
    bins = np.linspace(0.5, 1.0, 26)
    for sim in sims:
        m = sim.window_matches
        fav = truth.favourite_score(m["bot_a"].to_numpy(), m["bot_b"].to_numpy())
        fig.add_trace(_density_trace(fav, bins, sim.name, sim.color))
    fig.add_vline(x=LOPSIDED_THRESHOLD, line=dict(color="gray", dash="dash", width=1))
    fig.update_layout(
        xaxis_title="Favourite's true expected score",
        yaxis_title="Density",
        height=450,
    )
    return fig


def plot_elo_diff(sims: list[SimResult]) -> go.Figure:
    fig = go.Figure()
    all_diffs = np.concatenate([s.window_matches["elo_diff"].to_numpy() for s in sims])
    bins = np.linspace(0, np.quantile(all_diffs, 0.995), 51)
    for sim in sims:
        fig.add_trace(_density_trace(sim.window_matches["elo_diff"].to_numpy(), bins, sim.name, sim.color))
    fig.update_layout(
        xaxis_title="Absolute ELO difference at dispatch (sim's own ratings)",
        yaxis_title="Density",
        height=450,
    )
    return fig


def plot_rating_accuracy(sims: list[SimResult], burn_in: int) -> go.Figure:
    """RMSE and Spearman ρ against the true ratings over each run, with the
    mean and 95% CI across runs."""
    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.12,
        subplot_titles=["RMSE vs true ratings", "Spearman ρ vs true ratings"],
    )
    for col, key in enumerate(["rmse", "spearman"], start=1):
        for sim in sims:
            traj = pd.DataFrame([
                {"run": i, **p}
                for i, r in enumerate(sim.runs)
                for p in r.summary["rating_accuracy_trajectory"]
            ])
            stats = traj.groupby("match_count")[key].apply(lambda v: mean_ci95(v))
            x = stats.index.to_numpy()
            mean = np.array([m for m, _ in stats])
            ci = np.array([c or 0.0 for _, c in stats])
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
    fig.update_yaxes(title_text="ELO points", row=1, col=1)
    fig.update_yaxes(title_text="Rank correlation", row=1, col=2)
    fig.update_layout(height=450)
    return fig


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
    fig.update_layout(
        xaxis_title="Matches played",
        yaxis_title="Avg game duration (minutes)",
        height=500,
    )
    return fig


def plot_opponent_mean_vs_max(
    sims: list[SimResult], bot_ids: list[int], bot_names: dict[int, str],
) -> go.Figure:
    """Scatter of per-bot mean-matches-per-opponent vs max-matches-against-any-opponent."""
    fig = go.Figure()
    for sim in sims:
        bot_opp: dict[int, list[int]] = {b: [] for b in bot_ids}
        for (a, c), count in _pair_counts(sim.runs[0].matches).items():
            bot_opp[a].append(count)
            bot_opp[c].append(count)

        means, maxes, names = [], [], []
        for b in bot_ids:
            counts = bot_opp[b]
            if not counts:
                continue
            means.append(float(np.mean(counts)))
            maxes.append(max(counts))
            names.append(bot_names[b])

        fig.add_trace(go.Scatter(
            x=means, y=maxes, mode="markers", name=sim.name,
            marker=dict(color=sim.color, size=8, opacity=0.6),
            text=names,
            hovertemplate="%{text}<br>Mean: %{x:.2f}<br>Max: %{y}<extra></extra>",
        ))

    fig.update_layout(
        xaxis_title="Mean matches per opponent",
        yaxis_title="Max matches against a single opponent",
        height=500,
    )
    return fig


def plot_opponent_concentration(
    sims: list[SimResult], bot_ids: list[int],
) -> go.Figure:
    """For each bot in each run, sort its opponents by games played (most
    first) and compute cumulative match-share. Aggregate across bots and
    runs per matchmaker as median + interquartile band.

    A curve close to the diagonal = bots spread matches evenly. A curve
    bowed upwards = a few opponents dominate each bot's matches.
    """
    fig = go.Figure()
    x_grid = np.linspace(0.0, 1.0, 51)  # 0, 0.02, ..., 1.0

    for sim in sims:
        resampled = []
        for run in sim.runs:
            bot_opp: dict[int, dict[int, int]] = {b: {} for b in bot_ids}
            for (a, b), c in _pair_counts(run.matches).items():
                bot_opp[a][b] = c
                bot_opp[b][a] = c

            for b in bot_ids:
                counts = sorted(bot_opp[b].values(), reverse=True)
                if not counts:
                    continue
                total = sum(counts)
                n = len(counts)
                # Curve: (0, 0), (1/n, c1/total), (2/n, (c1+c2)/total), ..., (1, 1)
                xs = [0.0] + [(i + 1) / n for i in range(n)]
                ys = [0.0] + list(np.cumsum(counts) / total)
                resampled.append(np.interp(x_grid, xs, ys))

        arr = np.asarray(resampled)
        median = np.median(arr, axis=0)
        q25 = np.quantile(arr, 0.25, axis=0)
        q75 = np.quantile(arr, 0.75, axis=0)

        # IQR band (drawn first, below the median line)
        fig.add_trace(go.Scatter(
            x=list(x_grid) + list(x_grid[::-1]),
            y=list(q75) + list(q25[::-1]),
            fill="toself",
            fillcolor=sim.color, opacity=0.15,
            line=dict(width=0),
            name=f"{sim.name} IQR", showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=x_grid, y=median, mode="lines",
            name=sim.name, line=dict(color=sim.color, width=2),
        ))

    # Diagonal = perfect equality (each opponent faced equally often).
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Perfect equality",
        line=dict(color="gray", dash="dash", width=1), hoverinfo="skip",
    ))
    fig.update_layout(
        xaxis_title="Fraction of opponents (sorted by games played, most first)",
        yaxis_title="Fraction of bot's matches",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        height=500,
    )
    return fig


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
        height=850,
        xaxis=dict(tickangle=90, tickfont=dict(size=8)),
        yaxis=dict(tickfont=dict(size=8), autorange="reversed"),
    )
    return fig


NEAR_EQUAL_GAP = 50


def _near_equal_favourite(truth: GroundTruth) -> float:
    """Mean favourite's expected score over pairs within `NEAR_EQUAL_GAP` true rating."""
    r = np.array([truth.ratings[b] for b in truth.bot_ids])
    upper = np.triu_indices(len(r), 1)
    near = np.abs(r[:, None] - r[None, :])[upper] < NEAR_EQUAL_GAP
    favourite = np.maximum(truth.matrix, 1.0 - truth.matrix)[upper]
    return float(favourite[near].mean())


# --- Report assembly ---


_REPORT_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, sans-serif;
       max-width: 1200px; margin: 20px auto; padding: 0 20px; color: #222; }
h1 { border-bottom: 2px solid #333; padding-bottom: 10px; }
h2 { margin-top: 40px; color: #444;
     border-bottom: 1px solid #ccc; padding-bottom: 5px; }
p.meta { color: #444; font-size: 0.95em; line-height: 1.45; max-width: 950px; }
p.caption { color: #555; font-size: 0.95em; line-height: 1.45;
            margin: 8px 0 18px 0; max-width: 950px; }
.section { margin-bottom: 30px; }
table.summary { border-collapse: collapse; font-size: 0.92em; }
table.summary th, table.summary td { padding: 6px 12px; border-bottom: 1px solid #ddd;
                                     text-align: right; vertical-align: top; }
table.summary th:first-child { text-align: left; font-weight: normal; min-width: 240px; }
.table-wrap { overflow-x: auto; }
table.summary thead th { border-bottom: 2px solid #999; }
table.summary td { font-variant-numeric: tabular-nums; white-space: nowrap; }
.muted { color: #888; font-size: 0.85em; font-weight: normal; }
"""


def write_report(
    title: str, meta: str, sims: list[SimResult],
    sections: list[tuple[str, str, go.Figure | str]],
    output_path: Path,
) -> None:
    """Serialize all sections into a single HTML file.

    `sections` is a list of `(heading, caption, content)` tuples, where
    `content` is a figure or an HTML fragment.
    """
    sim_labels = ", ".join(f"<b style='color:{s.color}'>{s.name}</b>" for s in sims)
    parts = [
        "<!DOCTYPE html>\n<html>\n<head>\n",
        '<meta charset="utf-8">\n',
        f"<title>{title}</title>\n",
        f"<style>{_REPORT_CSS}</style>\n",
        "</head>\n<body>\n",
        f"<h1>{title}</h1>\n",
        f"<p class='meta'>Simulations compared: {sim_labels}</p>\n",
        f"<p class='meta'>{meta}</p>\n",
    ]
    plotly_js_included = False
    for heading, caption, content in sections:
        parts.append(f"<div class='section'>\n<h2>{heading}</h2>\n")
        if caption:
            parts.append(f"<p class='caption'>{caption}</p>\n")
        if isinstance(content, go.Figure):
            include_js = False if plotly_js_included else "cdn"
            plotly_js_included = True
            parts.append(content.to_html(full_html=False, include_plotlyjs=include_js))
        else:
            parts.append(content)
        parts.append("</div>\n")
    parts.append("</body>\n</html>\n")
    output_path.write_text("".join(parts), encoding="utf-8")
    log.info("Wrote %s", output_path)


def _methods_note(sims: list[SimResult]) -> str:
    config = sims[0].summary["config"]
    start = {
        "real": "the bots' current AI Arena ELOs",
        "flat": "1600 for every bot",
    }[config["initial_elo"]]
    return (
        f"Each matchmaker ran {len(sims[0].runs)} times with different seeds, "
        f"{config['total_matches']:,} matches per run on {config['max_concurrent']} server slots, "
        f"with ratings starting from {start}. The first {config['burn_in']:,} matches of every "
        "run are a burn-in and are left out of all metrics except the rating-accuracy "
        "trajectory. Table values are means across runs with 95% confidence intervals. "
        "<b>Truth</b> is the outcome model the sim draws from: every pair's true expected "
        "score, and the ELO-scale ratings that best fit those expected scores "
        "(<i>true ratings</i>, centred on 0). Unlike the sim's own ELO, it does not depend "
        "on who the matchmaker lets play whom."
    )


# --- Main ---


def _discover_matchmaker_dirs() -> list[Path]:
    """Find `matchmakers/*/output` dirs that have an aggregate summary.json."""
    root = REPO_ROOT / "matchmakers"
    return sorted(
        d / "output" for d in root.iterdir()
        if d.is_dir() and (d / "output" / "summary.json").exists()
    )


def main():
    parser = argparse.ArgumentParser(description="Compare matchmaker simulation runs")
    parser.add_argument("dirs", type=Path, nargs="*",
                        help="Simulation output directories (default: every "
                             "`matchmakers/*/output` with a summary.json)")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=_own_dir)
    args = parser.parse_args()

    dirs = args.dirs or _discover_matchmaker_dirs()
    if len(dirs) < 1:
        parser.error("no simulation output dirs found")
    names = [_infer_name(d) for d in dirs]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading %d simulations: %s", len(dirs), ", ".join(names))
    sims = [
        load_simulation(d, name, PALETTE[i % len(PALETTE)])
        for i, (d, name) in enumerate(zip(dirs, names))
    ]

    bots, _, lookup = load_model(args.data_dir, args.model_dir)
    bot_ids = bots["bot_id"].tolist()
    bot_names = dict(zip(bots["bot_id"], bots["name"]))
    truth = GroundTruth(bot_ids, lookup)
    burn_in = sims[0].summary["config"]["burn_in"]

    log.info("Generating report...")
    window_note = "after the burn-in"
    sections: list[tuple[str, str, go.Figure | str]] = [
        (
            "Summary",
            "<b>Favourite's true expected score</b>: for each match, the stronger side's "
            "expected score under the truth (0.5 = coin flip, 1 = certain win), averaged over "
            "matches. It measures how competitive the matches really are. "
            "<b>|ΔELO|</b> is the same question asked of the sim's own ratings, which is the "
            "quantity rating-based matchmakers optimise; compare the two to see how much of "
            "an ELO gain turns into closer games. "
            "<b>Rating accuracy</b> averages the post-burn-in ELO snapshots of each run. "
            "<b>Matches per bot</b>, <b>opponents</b> and <b>throughput</b> are counted over "
            "the post-burn-in window of each run.",
            summary_table(sims),
        ),
        (
            "How Close Are the Matches?",
            f"Distribution of the favourite's true expected score over all matches {window_note} "
            "(all runs pooled). Mass near 0.5 means competitive matches; the dashed line marks "
            f"the lopsided threshold ({LOPSIDED_THRESHOLD}). Even between bots whose true "
            f"ratings are within {NEAR_EQUAL_GAP} points of each other, the favourite's expected "
            f"score averages {_near_equal_favourite(truth):.2f}, because individual match-ups are "
            "often one-sided; no matchmaker that pairs by rating alone can get much below that.",
            plot_favourite_score(sims, truth),
        ),
        (
            "Skill Gap of Matched Pairs (sim's own ELO)",
            f"Distribution of |ELO<sub>a</sub> &minus; ELO<sub>b</sub>| at dispatch time, over all "
            f"matches {window_note} (all runs pooled). This is the matchmakers' own objective "
            "measured on their own rating scale; it overstates how much closer the matches "
            "get compared with the true expected scores above.",
            plot_elo_diff(sims),
        ),
        (
            "Rating Accuracy Over Time",
            "How well the sim's ELO matches the true ratings over each run: RMSE in ELO points "
            "(left) and rank correlation (right). Lines are means across runs, bands 95% "
            "confidence intervals, and the dashed line marks the end of the burn-in. "
            "ELO keeps fluctuating from game to game (K = 16), so the RMSE cannot reach zero. "
            "A curve that keeps rising instead of levelling off means the ratings drift away "
            "from the truth. With divisions, for example, bots only exchange ELO within their own "
            "division, and promotion picks the bots that are currently overrated (relegation the "
            "underrated ones), so the top division's ratings inflate and the bottom's deflate "
            "over time. Spearman ρ is insensitive to that kind of stretching; RMSE is not.",
            plot_rating_accuracy(sims, burn_in),
        ),
        (
            "Match Throughput per Bot",
            f"Each dot is one bot, in the first run's matches {window_note}. "
            "<b>X</b>: matches played. "
            "<b>Y</b>: average game duration. "
            "A narrow horizontal cluster within one matchmaker means matches are "
            "distributed evenly across bots — a wide spread indicates some bots "
            "are being dispatched more often than others. "
            "Differences in Y between matchmakers would hint at selection bias "
            "(e.g., a matchmaker preferentially picking a bot's short-game matchups).",
            plot_matches_per_bot(sims, bot_ids, bot_names),
        ),
        (
            "Opponent Concentration",
            f"For each bot, sort its opponents by games played (most first) and "
            f"track the cumulative share of the bot's matches ({window_note}). "
            "Each curve is the median across bots and runs within that matchmaker; the "
            "shaded band is the interquartile range. "
            "The <b>dashed diagonal</b> is perfect equality (bot plays every "
            "opponent equally often). "
            "Curves close to the diagonal = even spread across opponents; "
            "curves bowed up toward the top-left = a few opponents dominate "
            "that bot's schedule.",
            plot_opponent_concentration(sims, bot_ids),
        ),
        (
            "Mean vs Max Matches per Opponent",
            f"Each dot is one bot, in the first run's matches {window_note}. "
            "<b>X</b>: mean number of matches against each of its opponents. "
            "<b>Y</b>: games played against its most-frequent opponent. "
            "The ratio <code>Y / X</code> is the per-bot concentration at the "
            "top opponent — equivalent to the first-step value of the curve "
            "above, but resolved per bot so outliers (hover to see names) are "
            "identifiable.",
            plot_opponent_mean_vs_max(sims, bot_ids, bot_names),
        ),
    ]
    for sim in sims:
        sections.append((
            f"Matchup Frequency — {sim.name}",
            f"How often each pair of bots played in the first run's matches {window_note}. "
            "Rows and columns are sorted by true rating (strongest top/left; the number in "
            "brackets is the true rating relative to the ladder average). "
            "A dense diagonal band = skill-matched pairs; "
            "a uniform color = everyone plays everyone; "
            "bright spots off-diagonal = forced matchups across the skill gap.",
            plot_matchup_heatmap(sim, bot_names, truth.ratings),
        ))

    write_report("Matchmaker Comparison", _methods_note(sims), sims, sections,
                 args.output_dir / "report.html")


if __name__ == "__main__":
    main()
