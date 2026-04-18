"""Shared CLI helpers for matchmaker simulation scripts.

Factors out the argument-parsing skeleton, the sim-run-plus-writes pipeline,
and the summary-logging boilerplate that every `matchmakers/*/simulate.py`
would otherwise duplicate.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from sim.common import compute_elo_convergence, compute_summary, load_model
from sim.ladder_sim import LadderSim
from sim.matchmaker import Matchmaker
from sim.paths import DATA_DIR, MODEL_DIR

log = logging.getLogger(__name__)


def add_common_args(parser: argparse.ArgumentParser, default_output_dir: Path) -> None:
    """Attach standard sim-running flags to `parser`."""
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--total-matches", type=int, default=20000)
    parser.add_argument("--max-concurrent", type=int, default=12)
    parser.add_argument("--sim-seed", type=int, default=42, help="Seed for the sim's RNG")
    parser.add_argument("--mm-seed", type=int, default=100, help="Seed for the matchmaker's RNG")
    parser.add_argument("--last-n-matches", type=int, default=500)
    parser.add_argument("--elo-snapshot-interval", type=int, default=1000)


def run_and_write(
    matchmaker: Matchmaker,
    args: argparse.Namespace,
    *,
    extra_summary: Optional[Callable[[LadderSim, dict], None]] = None,
) -> LadderSim:
    """Load the model, run the sim with `matchmaker`, write outputs.

    Writes `elo_history.csv`, `matches.csv`, `summary.json` into
    `args.output_dir`. If `extra_summary` is supplied, it's called with
    (sim, summary) before the JSON is serialized, so matchmakers can inject
    custom fields (e.g. score components, total rounds). The matchmaker
    itself is reachable as `sim.matchmaker`.
    """
    bots, gp, lookup = load_model(args.data_dir, args.model_dir)

    log.info(
        "Running %s for %d matches (%d slots, sim_seed=%d)...",
        type(matchmaker).__name__, args.total_matches,
        args.max_concurrent, args.sim_seed,
    )

    sim = LadderSim(matchmaker, bots, gp, lookup, seed=args.sim_seed)
    sim.run(
        total_matches=args.total_matches,
        max_concurrent=args.max_concurrent,
        elo_snapshot_interval=args.elo_snapshot_interval,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    elo_df = pd.DataFrame(sim.elo_snapshots)
    elo_path = args.output_dir / "elo_history.csv"
    elo_df.to_csv(elo_path, index=False)
    log.info("Wrote %s (%d rows)", elo_path, len(elo_df))

    match_df = pd.DataFrame(sim.match_history)
    for col in ("time_start", "time_end", "elo_diff"):
        if col in match_df.columns:
            match_df[col] = match_df[col].round(2)
    match_path = args.output_dir / "matches.csv"
    match_df.to_csv(match_path, index=False)
    log.info("Wrote %s (%d matches)", match_path, len(match_df))

    bot_ids = bots["bot_id"].tolist()
    summary = compute_summary(match_df.to_dict("records"), bot_ids, args.last_n_matches)
    summary["simulated_time_minutes"] = (
        round(sim.match_history[-1]["time_end"], 2) if sim.match_history else 0
    )
    summary["elo_convergence"] = compute_elo_convergence(sim.elo_snapshots)

    if extra_summary is not None:
        extra_summary(sim, summary)

    summary_path = args.output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Wrote %s", summary_path)

    _log_summary(summary, args.last_n_matches)
    return sim


def _log_summary(summary: dict, last_n_matches: int) -> None:
    window_label = f" (last {last_n_matches} per bot)" if last_n_matches else ""
    log.info("\nSummary%s:", window_label)
    log.info("  Total matches: %d", summary["total_matches"])
    if "total_rounds" in summary:
        log.info("  Total rounds: %d", summary["total_rounds"])
    if "window_matches" in summary:
        log.info("  Window matches: %d", summary["window_matches"])
    st = summary.get("simulated_time_minutes")
    if st:
        log.info("  Simulated time: %.0f min (%.1f days)", st, st / (24 * 60))
    for key, label in [
        ("matches_per_bot", "Matches per bot"),
        ("unique_opponents_per_bot", "Unique opponents"),
        ("max_repeat_opponent", "Max repeat opponent"),
    ]:
        v = summary[key]
        log.info("  %s: %.1f ± %.1f (min=%d, max=%d)",
                 label, v["mean"], v["std"], v["min"], v["max"])
    ed = summary["elo_diff"]
    log.info("  ELO diff: %.1f ± %.1f (median=%.1f)", ed["mean"], ed["std"], ed["median"])
    d = summary["duration_minutes"]
    log.info("  Duration: %.2f ± %.2f min", d["mean"], d["std"])
    log.info("  Categories: %s", summary["category_rates"])
    conv = summary.get("elo_convergence", {})
    if "final_elo_std" in conv:
        log.info("  ELO convergence: 90%% of final spread (%.1f) at %s matches",
                 conv["final_elo_std"], conv["matches_to_90pct"])
    if "elo_stability" in conv:
        stab = conv["elo_stability"]
        log.info("  ELO stability (per-bot std over last %d snapshots): %.1f ± %.1f (max=%.1f)",
                 stab["n_snapshots"], stab["mean"], stab["std"], stab["max"])
    if "score_components" in summary:
        log.info("  Score components (mean ± std, %% of total):")
        for comp, vals in summary["score_components"].items():
            log.info("    %s: %.4f ± %.4f (%5.1f%%)", comp, vals["mean"], vals["std"], vals["pct"])
