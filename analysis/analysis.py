"""Generate a single-page comparison report across matchmaker simulation runs.

Usage:
    python analysis/analysis.py                       # auto-discover matchmakers/*
    python analysis/analysis.py <dir1> <dir2> [...]   # explicit output dirs

Writes `report.html` with every plot stacked on one page. Labels come from
the folder name (e.g. `matchmakers/stochastic/output` → `stochastic`).
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from sim.common import _last_n_matches_per_bot
from sim.paths import REPO_ROOT

_own_dir = Path(__file__).resolve().parent

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# Plotly's default qualitative palette — enough for 8+ matchmakers.
PALETTE = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA",
    "#FFA15A", "#19D3F3", "#FF6692", "#B6E880",
]


@dataclass
class SimResult:
    name: str
    matches: pd.DataFrame
    elo_history: pd.DataFrame
    summary: dict
    color: str


def _infer_name(d: Path) -> str:
    """`matchmakers/stochastic/output` → `stochastic`; otherwise the dir name."""
    return d.parent.name if d.name == "output" else d.name


def load_simulation(sim_dir: Path, name: str, color: str) -> SimResult:
    matches = pd.read_csv(sim_dir / "matches.csv")
    elo_history = pd.read_csv(sim_dir / "elo_history.csv")
    with open(sim_dir / "summary.json") as f:
        summary = json.load(f)
    return SimResult(name=name, matches=matches, elo_history=elo_history,
                     summary=summary, color=color)


# --- Plots (each returns a go.Figure) ---


def plot_elo_convergence(sims: list[SimResult]) -> go.Figure:
    fig = go.Figure()
    for sim in sims:
        traj = sim.summary["elo_convergence"]["elo_std_trajectory"]
        fig.add_trace(go.Scatter(
            x=[p["match_count"] for p in traj],
            y=[p["elo_std"] for p in traj],
            mode="lines+markers", name=sim.name,
            line=dict(color=sim.color, width=2), marker=dict(size=4),
        ))
    fig.update_layout(
        xaxis_title="Matches completed",
        yaxis_title="ELO standard deviation",
        height=450,
    )
    return fig


def plot_matches_per_bot(
    sims: list[SimResult], bot_ids: list[int], bot_names: dict[int, str],
) -> go.Figure:
    fig = go.Figure()
    for sim in sims:
        a = sim.matches["bot_a"].value_counts()
        b = sim.matches["bot_b"].value_counts()
        counts = a.add(b, fill_value=0).reindex(bot_ids, fill_value=0)

        dur = {b_id: [] for b_id in bot_ids}
        for _, row in sim.matches.iterrows():
            dur[row["bot_a"]].append(row["duration_minutes"])
            dur[row["bot_b"]].append(row["duration_minutes"])
        avg_dur = {b_id: (sum(v) / len(v) if v else 0) for b_id, v in dur.items()}

        fig.add_trace(go.Scatter(
            x=[counts[b_id] for b_id in bot_ids],
            y=[avg_dur[b_id] for b_id in bot_ids],
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


def plot_elo_diff(sims: list[SimResult]) -> go.Figure:
    fig = go.Figure()
    for sim in sims:
        fig.add_trace(go.Histogram(
            x=sim.matches["elo_diff"], name=sim.name,
            marker_color=sim.color, opacity=0.55, nbinsx=50,
            histnorm="probability density",
        ))
    fig.update_layout(
        barmode="overlay",
        xaxis_title="Absolute ELO difference",
        yaxis_title="Density",
        height=450,
    )
    return fig


def plot_opponent_mean_vs_max(
    sims: list[SimResult], bot_ids: list[int], bot_names: dict[int, str],
) -> go.Figure:
    """Scatter of per-bot mean-matches-per-opponent vs max-matches-against-any-opponent."""
    fig = go.Figure()
    for sim in sims:
        df = sim.matches
        lo = df[["bot_a", "bot_b"]].min(axis=1)
        hi = df[["bot_a", "bot_b"]].max(axis=1)
        pair_counts = (
            pd.DataFrame({"lo": lo, "hi": hi})
            .groupby(["lo", "hi"]).size().to_dict()
        )
        bot_opp: dict[int, list[int]] = {b: [] for b in bot_ids}
        for (a, c), count in pair_counts.items():
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
    """For each bot, sort its opponents by games played (most first) and
    compute cumulative match-share. Aggregate across bots per matchmaker as
    median + interquartile band.

    A curve close to the diagonal = bots spread matches evenly. A curve
    bowed upwards = a few opponents dominate each bot's matches.
    """
    fig = go.Figure()
    x_grid = np.linspace(0.0, 1.0, 51)  # 0, 0.02, ..., 1.0

    for sim in sims:
        df = sim.matches
        lo = df[["bot_a", "bot_b"]].min(axis=1)
        hi = df[["bot_a", "bot_b"]].max(axis=1)
        pair_counts = (
            pd.DataFrame({"lo": lo, "hi": hi})
            .groupby(["lo", "hi"]).size().to_dict()
        )
        bot_opp: dict[int, dict[int, int]] = {b: {} for b in bot_ids}
        for (a, b), c in pair_counts.items():
            bot_opp[a][b] = c
            bot_opp[b][a] = c

        resampled = []
        for b in bot_ids:
            counts = sorted(bot_opp[b].values(), reverse=True)
            if not counts:
                continue
            total = sum(counts)
            n = len(counts)
            # Curve: (0, 0), (1/n, c1/total), (2/n, (c1+c2)/total), ..., (1, 1)
            xs = [0.0] + [(i + 1) / n for i in range(n)]
            ys = [0.0]
            running = 0
            for c in counts:
                running += c
                ys.append(running / total)
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


def plot_elo_stability(sims: list[SimResult]) -> go.Figure:
    """RMS deviation from each bot's settled ELO over time."""
    fig = go.Figure()
    for sim in sims:
        df = sim.elo_history.copy()
        snapshots = sorted(df["match_count"].unique())
        settled_snaps = snapshots[len(snapshots) // 2:]
        settled_mean = df[df["match_count"].isin(settled_snaps)].groupby("bot_id")["elo"].mean()
        df = df.merge(settled_mean.rename("settled"), on="bot_id")
        df["dev_sq"] = (df["elo"] - df["settled"]) ** 2
        traj = df.groupby("match_count")["dev_sq"].mean().apply(np.sqrt).reset_index()
        fig.add_trace(go.Scatter(
            x=traj["match_count"], y=traj["dev_sq"],
            mode="lines+markers", name=sim.name,
            line=dict(color=sim.color, width=2), marker=dict(size=4),
        ))
    fig.update_layout(
        xaxis_title="Matches completed",
        yaxis_title="RMS deviation (ELO points)",
        height=500,
    )
    return fig


def _settled_elo(elo_df: pd.DataFrame, settled_start: int) -> dict[int, float]:
    """Mean ELO per bot over snapshots at/after `settled_start` matches."""
    settled = elo_df[elo_df["match_count"] >= settled_start]
    if settled.empty:
        settled = elo_df[elo_df["match_count"] == elo_df["match_count"].max()]
    return settled.groupby("bot_id")["elo"].mean().to_dict()


def plot_matchup_heatmap(
    sim: SimResult, full_matches: pd.DataFrame,
    bot_ids: list[int], bot_names: dict[int, str],
    final_elo: dict[int, float],
) -> go.Figure:
    """Heatmap of post-settlement match counts, rows/cols sorted by final ELO."""
    settled_start = len(full_matches) // 5  # drop first 20% as warmup
    settled_matches = full_matches.iloc[settled_start:]

    sorted_bots = sorted(bot_ids, key=lambda b: final_elo.get(b, 1600), reverse=True)
    bot_idx = {b: i for i, b in enumerate(sorted_bots)}
    n = len(sorted_bots)

    grid = np.zeros((n, n), dtype=int)
    for _, row in settled_matches.iterrows():
        a, b = int(row["bot_a"]), int(row["bot_b"])
        if a in bot_idx and b in bot_idx:
            i, j = bot_idx[a], bot_idx[b]
            grid[i][j] += 1
            grid[j][i] += 1

    labels = [f"{bot_names[b]} ({final_elo.get(b, 1600):.0f})" for b in sorted_bots]
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


# --- Report assembly ---


_REPORT_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, sans-serif;
       max-width: 1200px; margin: 20px auto; padding: 0 20px; color: #222; }
h1 { border-bottom: 2px solid #333; padding-bottom: 10px; }
h2 { margin-top: 40px; color: #444;
     border-bottom: 1px solid #ccc; padding-bottom: 5px; }
p.meta { color: #666; font-size: 0.9em; }
p.caption { color: #555; font-size: 0.95em; line-height: 1.45;
            margin: 8px 0 18px 0; max-width: 950px; }
.section { margin-bottom: 30px; }
"""


def write_report(
    title: str, sims: list[SimResult],
    sections: list[tuple[str, str, go.Figure]],
    output_path: Path,
) -> None:
    """Serialize all figures into a single HTML file.

    `sections` is a list of `(heading, caption, figure)` tuples.
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
    ]
    for i, (heading, caption, fig) in enumerate(sections):
        include_js = "cdn" if i == 0 else False
        parts.append(f"<div class='section'>\n<h2>{heading}</h2>\n")
        if caption:
            parts.append(f"<p class='caption'>{caption}</p>\n")
        parts.append(fig.to_html(full_html=False, include_plotlyjs=include_js))
        parts.append("</div>\n")
    parts.append("</body>\n</html>\n")
    output_path.write_text("".join(parts), encoding="utf-8")
    log.info("Wrote %s", output_path)


# --- Main ---


def _discover_matchmaker_dirs() -> list[Path]:
    """Find `matchmakers/*/output` dirs that have a matches.csv."""
    root = REPO_ROOT / "matchmakers"
    return sorted(
        d / "output" for d in root.iterdir()
        if d.is_dir() and (d / "output" / "matches.csv").exists()
    )


def main():
    parser = argparse.ArgumentParser(description="Compare matchmaker simulation runs")
    parser.add_argument("dirs", type=Path, nargs="*",
                        help="Simulation output directories (default: every "
                             "`matchmakers/*/output` with a matches.csv)")
    parser.add_argument("--output-dir", type=Path, default=_own_dir)
    parser.add_argument("--last-n-matches", type=int, default=500)
    args = parser.parse_args()

    dirs = args.dirs or _discover_matchmaker_dirs()
    if len(dirs) < 1:
        parser.error("no simulation output dirs found")
    names = [_infer_name(d) for d in dirs]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading %d simulations: %s", len(dirs), ", ".join(names))
    full_sims = [
        load_simulation(d, name, PALETTE[i % len(PALETTE)])
        for i, (d, name) in enumerate(zip(dirs, names))
    ]

    bot_ids = sorted(set(full_sims[0].matches["bot_a"]) | set(full_sims[0].matches["bot_b"]))
    bot_names = dict(zip(full_sims[0].elo_history["bot_id"],
                         full_sims[0].elo_history["bot_name"]))

    settled_start = len(full_sims[0].matches) // 5
    final_elo = _settled_elo(full_sims[0].elo_history, settled_start)

    windowed_sims = [
        SimResult(
            name=s.name,
            matches=_last_n_matches_per_bot(s.matches, bot_ids, args.last_n_matches),
            elo_history=s.elo_history,
            summary=s.summary,
            color=s.color,
        )
        for s in full_sims
    ]
    log.info("Window sizes: %s",
             ", ".join(f"{s.name}={len(s.matches)}" for s in windowed_sims))

    log.info("Generating report...")
    window_note = f"last {args.last_n_matches} matches per bot"
    sections: list[tuple[str, str, go.Figure]] = [
        (
            "Match Throughput per Bot",
            "Each dot is one bot over the full simulation. "
            "<b>X</b>: total matches played. "
            "<b>Y</b>: average game duration. "
            "A narrow horizontal cluster within one matchmaker means matches are "
            "distributed evenly across bots — a wide spread indicates some bots "
            "are being dispatched more often than others. "
            "Differences in Y between matchmakers would hint at selection bias "
            "(e.g., a matchmaker preferentially picking a bot's short-game matchups).",
            plot_matches_per_bot(full_sims, bot_ids, bot_names),
        ),
        (
            "Skill Gap of Matched Pairs",
            f"Distribution of |ELO<sub>a</sub> &minus; ELO<sub>b</sub>| at dispatch time "
            f"({window_note}). "
            "A skill-matched matchmaker concentrates mass near zero; a random or "
            "bucket-based matchmaker has a flatter, wider distribution. "
            "Computed over the late-simulation window only, so this is not diluted by "
            "the warm-up period when all ELOs start near 1600.",
            plot_elo_diff(windowed_sims),
        ),
        (
            "Opponent Concentration",
            f"For each bot, sort its opponents by games played (most first) and "
            f"track the cumulative share of the bot's matches ({window_note}). "
            "Each curve is the median across bots within that matchmaker; the "
            "shaded band is the interquartile range. "
            "The <b>dashed diagonal</b> is perfect equality (bot plays every "
            "opponent equally often). "
            "Curves close to the diagonal = even spread across opponents; "
            "curves bowed up toward the top-left = a few opponents dominate "
            "that bot's schedule.",
            plot_opponent_concentration(windowed_sims, bot_ids),
        ),
        (
            "Mean vs Max Matches per Opponent",
            f"Each dot is one bot ({window_note}). "
            "<b>X</b>: mean number of matches against each of its opponents. "
            "<b>Y</b>: games played against its most-frequent opponent. "
            "The ratio <code>Y / X</code> is the per-bot concentration at the "
            "top opponent — equivalent to the first-step value of the curve "
            "above, but resolved per bot so outliers (hover to see names) are "
            "identifiable.",
            plot_opponent_mean_vs_max(windowed_sims, bot_ids, bot_names),
        ),
    ]
    for sim in full_sims:
        sections.append((
            f"Matchup Frequency — {sim.name}",
            f"How often each pair of bots played in the post-settlement window "
            f"(matches {settled_start} onward — first 20% of the run dropped as warm-up). "
            "Rows and columns are sorted by final ELO (strongest top/left). "
            "A dense diagonal band = skill-matched pairs; "
            "a uniform color = everyone plays everyone; "
            "bright spots off-diagonal = forced matchups across the skill gap.",
            plot_matchup_heatmap(sim, sim.matches, bot_ids, bot_names, final_elo),
        ))
    sections += [
        (
            "ELO Ladder Spread Over Time",
            "Standard deviation of ELO across all bots at each snapshot. "
            "Starts at 0 (all bots seeded at 1600) and rises as strong/weak bots diverge; "
            "the plateau = the ladder's equilibrium spread. "
            "A higher plateau means the matchmaker produced clearer skill separation. "
            "Note: this is a <i>population-level</i> signal — it doesn't tell you whether "
            "individual bot rankings are stable (see the next plot for that).",
            plot_elo_convergence(full_sims),
        ),
        (
            "Per-Bot ELO Stabilization",
            "For each bot, its &ldquo;settled ELO&rdquo; is defined as its mean ELO over the "
            "second half of the snapshots. At every earlier snapshot, we compute the "
            "RMS distance from that settled value across all bots. "
            "The curve drops toward a noise floor as bots approach their long-run ratings. "
            "Read this to answer &ldquo;how many matches until ELOs stop moving meaningfully?&rdquo;",
            plot_elo_stability(full_sims),
        ),
    ]

    write_report("Matchmaker Comparison", full_sims, sections,
                 args.output_dir / "report.html")


if __name__ == "__main__":
    main()
