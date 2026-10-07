"""Fetch bot ELO ratings and match history from the AI Arena API."""

import argparse
import asyncio
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from aiarena_api import TOKEN_ENV, AiArenaClient
from aiarena_api.schema import Match, Round

# --- .env loading ---

_own_dir = Path(__file__).resolve().parent
_repo_root = _own_dir.parent
_env_path = _repo_root / ".env"

try:
    from dotenv import load_dotenv

    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    if _env_path.exists():
        for line in _env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

logging.basicConfig(level=logging.INFO, format="%(message)s")
# httpx logs every request at INFO, which would bury the progress lines.
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger(__name__)


# --- Data fetching ---


async def fetch_bots(client: AiArenaClient, competition: int) -> pd.DataFrame:
    """Fetch all bots with ELO ratings for a competition."""
    log.info("Fetching competition participations...")
    participations = [p async for p in client.list_competition_participations(competition)]
    log.info("  %d participations", len(participations))

    log.info("Fetching bot details...")
    bots_raw = [bot async for bot in client.paginate("/bots/")]
    log.info("  %d bots", len(bots_raw))

    # Build bot lookup: id -> {name, race, bot_data_enabled}
    bot_info = {
        bot["id"]: {
            "name": bot["name"],
            "race": bot["plays_race"]["label"],
            "bot_data_enabled": bot.get("bot_data_enabled", False),
        }
        for bot in bots_raw
    }

    # Join participations with bot info
    rows = []
    for p in participations:
        bot_id = p["bot"]
        info = bot_info.get(bot_id)
        if not info:
            continue
        rows.append({
            "bot_id": bot_id,
            "name": info["name"],
            "race": info["race"],
            "elo": p["elo"],
            "active": p.get("active", False),
            "bot_data_enabled": info["bot_data_enabled"],
        })

    return pd.DataFrame(rows)


async def fetch_rounds(client: AiArenaClient, competition: int, since: datetime) -> list[Round]:
    """Fetch all rounds for a competition that started after the given date."""
    log.info("Fetching rounds for competition %d...", competition)
    rounds = [r async for r in client.list_rounds(competition)]
    recent = [r for r in rounds if r.get("started") and datetime.fromisoformat(r["started"]) >= since]
    log.info("  %d total rounds, %d in the last %d days",
             len(rounds), len(recent),
             (datetime.now(timezone.utc) - since).days)
    return recent


async def fetch_matches(client: AiArenaClient, days: int, competition: int) -> pd.DataFrame:
    """Fetch all matches from the last N days via rounds."""
    since = datetime.now(timezone.utc) - timedelta(days=days)

    rounds = await fetch_rounds(client, competition, since)
    if not rounds:
        log.warning("No rounds found in the last %d days", days)
        return pd.DataFrame()

    done = 0

    async def round_matches(rnd: Round) -> list[Match]:
        nonlocal done
        matches = [m async for m in client.list_matches_for_round(rnd["id"])]
        done += 1
        log.info("  Fetched %d matches of round %s (%d/%d)", len(matches), rnd["number"], done, len(rounds))
        return matches

    # Rounds are read concurrently, as many requests at a time as the client allows; gather keeps their order.
    per_round = await asyncio.gather(*(round_matches(rnd) for rnd in rounds))
    all_matches = [m for matches in per_round for m in matches]

    log.info("  %d total matches fetched", len(all_matches))

    return _process_matches(all_matches)


def _process_matches(matches: list[Match]) -> pd.DataFrame:
    """Process raw match dicts into a clean DataFrame."""
    rows = []
    skipped = 0

    for m in matches:
        result = m.get("result")
        if not result:
            skipped += 1
            continue

        game_steps = result.get("game_steps")
        duration_minutes = round(game_steps / 22.4 / 60, 2) if game_steps else None

        rows.append({
            "match_id": m["id"],
            "round": m.get("round"),
            "map": m.get("map"),
            "bot1_name": result.get("bot1_name"),
            "bot2_name": result.get("bot2_name"),
            "result_type": result.get("type"),
            "game_steps": game_steps,
            "duration_minutes": duration_minutes,
            "is_timeout": game_steps is not None and game_steps >= 80640,
            "match_created": m.get("created"),
            "match_started": m.get("started"),
            "result_created": result.get("created"),
        })

    if skipped:
        log.info("  Skipped %d matches without results", skipped)

    return pd.DataFrame(rows)


def add_bot_ids(matches_df: pd.DataFrame, bots_df: pd.DataFrame) -> pd.DataFrame:
    """Add bot IDs to matches by joining on bot name."""
    name_to_id = dict(zip(bots_df["name"], bots_df["bot_id"]))

    # Check for duplicate names
    if len(name_to_id) < len(bots_df):
        log.warning("Some bot names are duplicated — ID mapping may be ambiguous")

    matches_df["bot1_id"] = matches_df["bot1_name"].map(name_to_id)
    matches_df["bot2_id"] = matches_df["bot2_name"].map(name_to_id)

    missing = matches_df["bot1_id"].isna().sum() + matches_df["bot2_id"].isna().sum()
    if missing:
        log.warning("  %d bot name(s) could not be mapped to IDs", missing)

    return matches_df


# --- Main ---


def main():
    parser = argparse.ArgumentParser(description="Fetch AI Arena match history")
    parser.add_argument("--days", type=int, default=365, help="Number of days to fetch (default: 365)")
    parser.add_argument("--competition", type=int, default=36, help="Competition ID (default: 36)")
    parser.add_argument("--output-dir", type=Path, default=_own_dir, help="Output directory (default: data/)")
    args = parser.parse_args()

    if not os.environ.get(TOKEN_ENV):
        raise SystemExit(f"{TOKEN_ENV} not set. Put it in a .env file in {_repo_root}, or set it in the environment.")

    bots_df, matches_df = asyncio.run(_fetch(args.days, args.competition))

    if matches_df.empty:
        log.warning("No matches found. Exiting.")
        return

    matches_df = add_bot_ids(matches_df, bots_df)

    # Reorder columns
    matches_df = matches_df[
        ["match_id", "round", "map", "bot1_id", "bot1_name", "bot2_id", "bot2_name",
         "result_type", "game_steps", "duration_minutes", "is_timeout",
         "match_created", "match_started", "result_created"]
    ]

    # Write CSVs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bots_path = args.output_dir / "bots.csv"
    matches_path = args.output_dir / "matches.csv"

    bots_df.to_csv(bots_path, index=False)
    matches_df.to_csv(matches_path, index=False)

    log.info("Wrote %s (%d bots)", bots_path, len(bots_df))
    log.info("Wrote %s (%d matches)", matches_path, len(matches_df))

    # Summary stats
    timeouts = matches_df["is_timeout"].sum()
    log.info("  Timeouts: %d (%.1f%%)", timeouts, 100 * timeouts / len(matches_df) if len(matches_df) else 0)
    for rtype in ["Player1Win", "Player2Win", "Tie"]:
        count = (matches_df["result_type"] == rtype).sum()
        log.info("  %s: %d", rtype, count)


async def _fetch(days: int, competition: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The bots with their ELO, and the matches of the last `days`; the token comes from AIARENA_API_TOKEN."""
    async with AiArenaClient() as client:
        bots_df = await fetch_bots(client, competition)
        log.info("Bots with ELO: %d", len(bots_df))

        matches_df = await fetch_matches(client, days, competition)
        log.info("Valid matches: %d", len(matches_df))
    return bots_df, matches_df


if __name__ == "__main__":
    main()
