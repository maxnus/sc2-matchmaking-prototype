"""Shared CLI helpers for matchmaker simulation scripts.

Factors out the argument-parsing skeleton, the multi-seed sim-run-plus-writes
pipeline, and the summary-logging boilerplate that every
`matchmakers/*/simulate.py` would otherwise duplicate.
"""

import argparse
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from sim.common import load_model
from sim.ladder_sim import LadderSim
from sim.matchmaker import Matchmaker
from sim.metrics import (
    HEADLINE_METRICS, GroundTruth, aggregate_metrics, compute_summary, format_metric,
)
from sim.paths import DATA_DIR, MODEL_DIR

log = logging.getLogger(__name__)

# gzip without a timestamp, so rerunning with the same seeds rewrites identical files.
_GZIP = {"method": "gzip", "mtime": 0}

# Arguments that describe where files live or how the work is scheduled,
# rather than the experiment; left out of the recorded config.
_NOT_CONFIG = {"data_dir", "model_dir", "output_dir", "jobs"}


def add_common_args(parser: argparse.ArgumentParser, default_output_dir: Path) -> None:
    """Attach standard sim-running flags to `parser`."""
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--total-matches", type=int, default=50000,
                        help="Matches per run, including the burn-in")
    parser.add_argument("--burn-in", type=int, default=10000,
                        help="Leading matches of each run left out of the summary metrics")
    parser.add_argument("--initial-elo", choices=["real", "flat"], default="real",
                        help="Start from the AI Arena ELOs in bots.csv (real) "
                             "or with every bot at 1600 (flat)")
    parser.add_argument("--max-concurrent", type=int, default=12, help="Server slots")
    parser.add_argument("--max-parallel", type=int, default=4,
                        help="Most matches a bot without bot data plays at once")
    parser.add_argument("--seeds", type=int, default=8, help="Number of independent runs")
    parser.add_argument("--jobs", type=int, default=os.cpu_count(),
                        help="Runs executed in parallel")
    parser.add_argument("--sim-seed", type=int, default=42,
                        help="Seed for the sim's RNG; run i uses this + i")
    parser.add_argument("--mm-seed", type=int, default=100,
                        help="Seed for the matchmaker's RNG; run i uses this + i")
    parser.add_argument("--elo-snapshot-interval", type=int, default=1000)


def run_and_write(
    make_matchmaker: Callable[..., Matchmaker],
    args: argparse.Namespace,
    *,
    extra_summary: Optional[Callable[[LadderSim, dict], None]] = None,
) -> None:
    """Run `args.seeds` independent sims and write their outputs.

    `make_matchmaker(seed=...)` builds a fresh matchmaker for each run. Run
    `i` writes `matches.csv.gz`, `elo_history.csv.gz` and `summary.json` into
    `args.output_dir / f"seed_{i}"`, and `args.output_dir / "summary.json"`
    holds the mean and 95% confidence interval of every metric across runs.

    If `extra_summary` is supplied, it's called with (sim, summary) before a
    run's JSON is serialized, so matchmakers can add custom fields to
    `summary["metrics"]` (e.g. score components, total rounds). The
    matchmaker itself is reachable as `sim.matchmaker`.

    Runs execute in worker processes when `args.jobs > 1`, so
    `make_matchmaker` and `extra_summary` must be picklable: module-level
    functions, classes or `functools.partial`s of them, not lambdas or
    closures.
    """
    if not 0 <= args.burn_in < args.total_matches:
        raise SystemExit("--burn-in must be at least 0 and below --total-matches")

    run = partial(_run_seed, make_matchmaker, args, extra_summary)
    jobs = min(args.jobs, args.seeds)
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            summaries = list(pool.map(run, range(args.seeds)))
    else:
        summaries = [run(i) for i in range(args.seeds)]

    aggregate = {
        "config": {k: v for k, v in vars(args).items() if k not in _NOT_CONFIG},
        "runs": [f"seed_{i}" for i in range(args.seeds)],
        "metrics": aggregate_metrics([s["metrics"] for s in summaries]),
    }
    path = args.output_dir / "summary.json"
    with open(path, "w") as f:
        json.dump(aggregate, f, indent=2)
    log.info("Wrote %s", path)

    _log_aggregate(aggregate)


def _run_seed(
    make_matchmaker: Callable[..., Matchmaker],
    args: argparse.Namespace,
    extra_summary: Optional[Callable[[LadderSim, dict], None]],
    i: int,
) -> dict:
    """Run and write the `i`-th sim; returns its summary."""
    sim_seed, mm_seed = args.sim_seed + i, args.mm_seed + i
    bots, gp, lookup = load_model(args.data_dir, args.model_dir)
    initial = dict(zip(bots["bot_id"], bots["elo"])) if args.initial_elo == "real" else None
    matchmaker = make_matchmaker(seed=mm_seed)

    log.info(
        "Run %d: %s for %d matches (%d slots, sim_seed=%d, mm_seed=%d)...",
        i, type(matchmaker).__name__, args.total_matches,
        args.max_concurrent, sim_seed, mm_seed,
    )
    sim = LadderSim(matchmaker, bots, gp, lookup, seed=sim_seed, initial_ratings=initial)
    sim.run(
        total_matches=args.total_matches,
        max_concurrent=args.max_concurrent,
        max_parallel=args.max_parallel,
        elo_snapshot_interval=args.elo_snapshot_interval,
    )

    out_dir = args.output_dir / f"seed_{i}"
    out_dir.mkdir(parents=True, exist_ok=True)

    elo_df = pd.DataFrame(sim.elo_snapshots)
    elo_df.to_csv(out_dir / "elo_history.csv.gz", index=False, compression=_GZIP)

    match_df = pd.DataFrame(sim.match_history)
    summary = compute_summary(match_df, elo_df, GroundTruth(sim.bot_ids, lookup), args.burn_in)
    summary["sim_seed"] = sim_seed
    summary["mm_seed"] = mm_seed
    if extra_summary is not None:
        extra_summary(sim, summary)

    for col in ("time_start", "time_end", "elo_diff"):
        match_df[col] = match_df[col].round(2)
    match_df.to_csv(out_dir / "matches.csv.gz", index=False, compression=_GZIP)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Run %d: wrote %s", i, out_dir)
    return summary


def _log_aggregate(aggregate: dict) -> None:
    metrics = aggregate["metrics"]
    config = aggregate["config"]
    log.info(
        "\nSummary over %d runs (mean ± 95%% CI), matches %d–%d of each run:",
        len(aggregate["runs"]), config["burn_in"], config["total_matches"],
    )
    for key, label, fmt, _ in HEADLINE_METRICS:
        log.info("  %s: %s", label, format_metric(metrics[key], fmt))
    if "total_rounds" in metrics:
        log.info("  Total rounds: %s", format_metric(metrics["total_rounds"], ".1f"))
    components = [k for k in metrics if k.startswith("score_components.") and k.endswith(".pct")]
    if components:
        log.info("  Score components (% of total):")
        for key in components:
            log.info("    %s: %s", key.split(".")[1], format_metric(metrics[key], ".1f"))
