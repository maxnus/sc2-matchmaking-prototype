"""Simulate the current AI Arena round-robin matchmaking system."""

import argparse
import json
import logging
from itertools import combinations
from pathlib import Path

import pandas as pd

from common import (
    compute_elo_convergence,
    compute_summary,
    load_model,
    simulate_match,
    update_elo,
)

_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# --- Divisions ---


def assign_divisions(bot_ids: list[int], ratings: dict, n_divisions: int = 3) -> list[list[int]]:
    """Sort bots by rating descending and split into equal divisions."""
    sorted_bots = sorted(bot_ids, key=lambda b: ratings[b], reverse=True)
    n = len(sorted_bots)
    base_size = n // n_divisions
    remainder = n % n_divisions

    divisions = []
    start = 0
    for i in range(n_divisions):
        size = base_size + (1 if i < remainder else 0)
        divisions.append(sorted_bots[start : start + size])
        start += size
    return divisions


# --- Simulation ---


def simulate(
    bots: pd.DataFrame,
    lookup: dict,
    gp: dict,
    n_rounds: int,
    total_matches: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Run the round-robin simulation.

    Stops after total_matches have been played (may stop mid-round).
    Returns (elo_history, match_history).
    """
    rng = __import__("numpy").random.default_rng(seed)
    bot_ids = bots["bot_id"].tolist()
    bot_names = dict(zip(bots["bot_id"], bots["name"]))

    # Initialize
    ratings = {b: 1600.0 for b in bot_ids}
    divisions = assign_divisions(bot_ids, ratings)

    elo_history = []
    match_history = []
    matches_completed = 0

    # Record initial state
    for div_idx, div_bots in enumerate(divisions):
        for b in div_bots:
            elo_history.append({
                "round": 0, "bot_id": b, "bot_name": bot_names[b],
                "elo": ratings[b], "division": div_idx + 1,
            })

    for rnd in range(1, n_rounds + 1):
        round_matches = 0
        for div_idx, div_bots in enumerate(divisions):
            for bot_a, bot_b in combinations(div_bots, 2):
                if matches_completed >= total_matches:
                    break
                elo_diff = abs(ratings[bot_a] - ratings[bot_b])
                outcome_a, duration, category = simulate_match(
                    bot_a, bot_b, lookup, gp, rng
                )
                update_elo(ratings, bot_a, bot_b, outcome_a)
                match_history.append({
                    "round": rnd,
                    "bot_a": bot_a,
                    "bot_b": bot_b,
                    "outcome_a": outcome_a,
                    "duration_minutes": duration,
                    "category": category,
                    "division": div_idx + 1,
                    "elo_diff": round(elo_diff, 2),
                })
                round_matches += 1
                matches_completed += 1
            if matches_completed >= total_matches:
                break

        # Re-assign divisions
        divisions = assign_divisions(bot_ids, ratings)

        # Record ELO state
        for div_idx, div_bots in enumerate(divisions):
            for b in div_bots:
                elo_history.append({
                    "round": rnd, "bot_id": b, "bot_name": bot_names[b],
                    "elo": round(ratings[b], 2), "division": div_idx + 1,
                })

        log.info("Round %d: %d matches (total: %d)", rnd, round_matches, matches_completed)
        if matches_completed >= total_matches:
            break

    return elo_history, match_history


# --- Main ---


def main():
    parser = argparse.ArgumentParser(
        description="Simulate current AI Arena round-robin matchmaking"
    )
    parser.add_argument("--data-dir", type=Path, default=_repo_root / "ladder_data")
    parser.add_argument("--model-dir", type=Path, default=_repo_root / "ladder_model")
    parser.add_argument("--output-dir", type=Path, default=_repo_root / "simulation_current")
    parser.add_argument("--total-matches", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--last-n-matches", type=int, default=500,
        help="Compute summary stats over each bot's last N matches (default: 500)",
    )
    args = parser.parse_args()

    bots, gp, lookup = load_model(args.data_dir, args.model_dir)

    # Compute rounds needed from total-matches target (round up)
    bot_ids = bots["bot_id"].tolist()
    divisions = assign_divisions(bot_ids, {b: 1600.0 for b in bot_ids})
    matches_per_round = sum(len(d) * (len(d) - 1) // 2 for d in divisions)
    n_rounds = max(1, -(-args.total_matches // matches_per_round))  # ceil division
    log.info(
        "Simulating up to %d rounds (%d matches/round, capped at %d matches) with %d bots...",
        n_rounds, matches_per_round, args.total_matches, len(bots),
    )

    elo_history, match_history = simulate(
        bots, lookup, gp, n_rounds, args.total_matches, args.seed,
    )

    # Write outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)

    elo_df = pd.DataFrame(elo_history)
    elo_path = args.output_dir / "elo_history.csv"
    elo_df.to_csv(elo_path, index=False)
    log.info("Wrote %s (%d rows)", elo_path, len(elo_df))

    match_df = pd.DataFrame(match_history)
    match_path = args.output_dir / "matches.csv"
    match_df.to_csv(match_path, index=False)
    log.info("Wrote %s (%d matches)", match_path, len(match_df))

    bot_ids = bots["bot_id"].tolist()
    summary = compute_summary(match_history, bot_ids, args.last_n_matches)
    summary["total_rounds"] = n_rounds

    # ELO convergence (map round -> cumulative match count)
    round_match_counts = {0: 0}
    for rnd in range(1, n_rounds + 1):
        round_match_counts[rnd] = rnd * matches_per_round
    convergence = compute_elo_convergence(elo_history, match_counts=round_match_counts)
    summary["elo_convergence"] = convergence

    summary_path = args.output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Wrote %s", summary_path)

    # Print summary
    window_label = f" (last {args.last_n_matches} per bot)" if args.last_n_matches else ""
    log.info("\nSummary%s:", window_label)
    log.info("  Total matches: %d", summary["total_matches"])
    if "window_matches" in summary:
        log.info("  Window matches: %d", summary["window_matches"])
    log.info("  Matches per bot: %.1f ± %.1f (min=%d, max=%d)",
             summary["matches_per_bot"]["mean"], summary["matches_per_bot"]["std"],
             summary["matches_per_bot"]["min"], summary["matches_per_bot"]["max"])
    log.info("  Unique opponents: %.1f ± %.1f (min=%d, max=%d)",
             summary["unique_opponents_per_bot"]["mean"],
             summary["unique_opponents_per_bot"]["std"],
             summary["unique_opponents_per_bot"]["min"],
             summary["unique_opponents_per_bot"]["max"])
    log.info("  Max repeat opponent: %.1f ± %.1f (min=%d, max=%d)",
             summary["max_repeat_opponent"]["mean"],
             summary["max_repeat_opponent"]["std"],
             summary["max_repeat_opponent"]["min"],
             summary["max_repeat_opponent"]["max"])
    log.info("  ELO diff: %.1f ± %.1f (median=%.1f)",
             summary["elo_diff"]["mean"], summary["elo_diff"]["std"],
             summary["elo_diff"]["median"])
    log.info("  Duration: %.2f ± %.2f min",
             summary["duration_minutes"]["mean"], summary["duration_minutes"]["std"])
    log.info("  Categories: %s", summary["category_rates"])
    log.info("  ELO convergence: 90%% of final spread (%.1f) at %s matches",
             convergence["final_elo_std"],
             convergence["matches_to_90pct"])
    stab = convergence["elo_stability"]
    log.info("  ELO stability (per-bot std over last %d snapshots): %.1f ± %.1f (max=%.1f)",
             stab["n_snapshots"], stab["mean"], stab["std"], stab["max"])


if __name__ == "__main__":
    main()
