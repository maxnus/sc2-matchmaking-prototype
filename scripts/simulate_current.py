"""Simulate the current AI Arena round-robin matchmaking system."""

import argparse
import json
import logging
from itertools import combinations
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

K = 16  # ELO adjustment per game


# --- Model loading ---


def load_model(
    data_dir: Path, model_dir: Path
) -> tuple[pd.DataFrame, dict, dict]:
    """Load bots, global params, and matchup params.

    Returns (active_bots_df, global_params, matchup_lookup).
    matchup_lookup is keyed by (bot_lo, bot_hi) tuples.
    """
    bots = pd.read_csv(data_dir / "bots.csv")
    bots = bots[bots["active"] == True].reset_index(drop=True)

    with open(model_dir / "global_params.json") as f:
        gp = json.load(f)

    matchups = pd.read_csv(model_dir / "matchup_params.csv")
    lookup = {}
    for _, row in matchups.iterrows():
        key = (int(row["bot_lo"]), int(row["bot_hi"]))
        lookup[key] = row.to_dict()

    log.info("Loaded %d active bots, %d matchup params", len(bots), len(lookup))
    return bots, gp, lookup


# --- Match simulation ---


def simulate_match(
    bot_a: int, bot_b: int, lookup: dict, gp: dict, rng: np.random.Generator
) -> tuple[float, float, str]:
    """Simulate a single match between two bots.

    Returns (outcome_a, duration_minutes, category).
    outcome_a: 1.0 (win), 0.5 (draw), 0.0 (loss) from bot_a's perspective.
    """
    bot_lo, bot_hi = min(bot_a, bot_b), max(bot_a, bot_b)
    params = lookup.get((bot_lo, bot_hi))

    if params is None:
        # No matchup data — use pure prior (shouldn't happen for active bots)
        category = rng.choice(
            ["normal", "timelimit", "abnormal"],
            p=[gp["p_normal"], gp["p_timelimit"], gp["p_abnormal"]],
        )
        if category == "normal":
            outcome_lo = rng.choice([1.0, 0.5, 0.0], p=[0.5 - gp["d"] / 2, gp["d"], 0.5 - gp["d"] / 2])
            duration = _sample_duration(gp["mu_0"], gp["sigma"], rng)
        elif category == "timelimit":
            outcome_lo = 0.5
            duration = 60.0
        else:
            outcome_lo = rng.choice([1.0, 0.0])
            duration = _sample_duration(gp["mu_abnormal"], gp["sigma_abnormal"], rng)
        return (outcome_lo if bot_a == bot_lo else 1.0 - outcome_lo, duration, category)

    # Sample category from Dirichlet posterior
    cat_probs = rng.dirichlet([
        params["alpha_normal"],
        params["alpha_timelimit"],
        params["alpha_abnormal"],
    ])
    category = rng.choice(["normal", "timelimit", "abnormal"], p=cat_probs)

    if category == "normal":
        # Sample outcome from Dirichlet posterior (bot_lo's perspective)
        outcome_probs = rng.dirichlet([
            params["alpha_win"],
            params["alpha_draw"],
            params["alpha_loss"],
        ])
        outcome_idx = rng.choice([0, 1, 2], p=outcome_probs)
        outcome_lo = [1.0, 0.5, 0.0][outcome_idx]
        duration = _sample_duration(params["mu_duration"], params["sigma_duration"], rng)

    elif category == "timelimit":
        outcome_lo = 0.5
        duration = 60.0

    else:  # abnormal
        outcome_lo = rng.choice([1.0, 0.0])
        duration = _sample_duration(gp["mu_abnormal"], gp["sigma_abnormal"], rng)

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
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Run the round-robin simulation.

    Returns (elo_history, match_history).
    """
    rng = np.random.default_rng(seed)
    bot_ids = bots["bot_id"].tolist()
    bot_names = dict(zip(bots["bot_id"], bots["name"]))

    # Initialize
    ratings = {b: 1600.0 for b in bot_ids}
    divisions = assign_divisions(bot_ids, ratings)

    elo_history = []
    match_history = []

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
                })
                round_matches += 1

        # Re-assign divisions
        divisions = assign_divisions(bot_ids, ratings)

        # Record ELO state
        for div_idx, div_bots in enumerate(divisions):
            for b in div_bots:
                elo_history.append({
                    "round": rnd, "bot_id": b, "bot_name": bot_names[b],
                    "elo": round(ratings[b], 2), "division": div_idx + 1,
                })

        log.info("Round %d: %d matches", rnd, round_matches)

    return elo_history, match_history


# --- Summary ---


def _last_n_matches_per_bot(
    df: pd.DataFrame, bot_ids: list[int], last_n: int
) -> pd.DataFrame:
    """Return the union of each bot's last N matches (by match index order)."""
    # Build per-bot match indices (each match appears for both participants)
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
    n_rounds: int,
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

    # Unique opponents per bot (within window)
    opponents = {b: set() for b in bot_ids}
    for _, row in df_window.iterrows():
        opponents[row["bot_a"]].add(row["bot_b"])
        opponents[row["bot_b"]].add(row["bot_a"])
    unique_opp = pd.Series({b: len(v) for b, v in opponents.items()})

    categories = df_window["category"].value_counts(normalize=True).to_dict()
    outcomes = df_window["outcome_a"].value_counts(normalize=True)
    outcome_rates = {
        "win": float(outcomes.get(1.0, 0)),
        "draw": float(outcomes.get(0.5, 0)),
        "loss": float(outcomes.get(0.0, 0)),
    }

    result = {
        "total_matches": len(df),
        "total_rounds": n_rounds,
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
        "duration_minutes": {
            "mean": round(float(df_window["duration_minutes"].mean()), 2),
            "std": round(float(df_window["duration_minutes"].std()), 2),
        },
        "category_rates": {k: round(v, 4) for k, v in categories.items()},
        "outcome_rates": outcome_rates,
    })
    return result


# --- Main ---


def main():
    parser = argparse.ArgumentParser(
        description="Simulate current AI Arena round-robin matchmaking"
    )
    parser.add_argument("--data-dir", type=Path, default=_repo_root / "ladder_data")
    parser.add_argument("--model-dir", type=Path, default=_repo_root / "ladder_model")
    parser.add_argument("--output-dir", type=Path, default=_repo_root / "simulation_current")
    parser.add_argument("--rounds", type=int, default=63)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--last-n-matches", type=int, default=500,
        help="Compute summary stats over each bot's last N matches (default: 500)",
    )
    args = parser.parse_args()

    bots, gp, lookup = load_model(args.data_dir, args.model_dir)
    log.info("Simulating %d rounds with %d bots...", args.rounds, len(bots))

    elo_history, match_history = simulate(bots, lookup, gp, args.rounds, args.seed)

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
    summary = compute_summary(match_history, bot_ids, args.rounds, args.last_n_matches)
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
    log.info("  Duration: %.2f ± %.2f min",
             summary["duration_minutes"]["mean"], summary["duration_minutes"]["std"])
    log.info("  Categories: %s", summary["category_rates"])
    log.info("  Outcomes: %s", summary["outcome_rates"])


if __name__ == "__main__":
    main()
