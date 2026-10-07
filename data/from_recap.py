"""Build bots.csv and matches.csv from an AI Arena Recap database.

AI Arena Recap (github.com/maxnus/ai-arena-recap) keeps every match of the
seasons it tracks in SQLite, so a whole season can be read without paging
the AI Arena API for hours. Writes the same files, with the same columns, as
`fetch.py`.

Usage:
    python data/from_recap.py path/to/recap.sqlite --competition 36 --since 2026-05-01
"""

import argparse
import logging
import sqlite3
from pathlib import Path

import pandas as pd

_own_dir = Path(__file__).resolve().parent

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# As in fetch.py: SC2 runs 22.4 game steps per second, and the time limit is
# 60 minutes of game time.
STEPS_PER_SECOND = 22.4
TIME_LIMIT_STEPS = 80640

MATCH_COLUMNS = [
    "match_id", "round", "map", "bot1_id", "bot1_name", "bot2_id", "bot2_name",
    "result_type", "game_steps", "duration_minutes", "is_timeout",
    "match_created", "match_started", "result_created",
]

_BOTS_SQL = """
SELECT cp.bot_id, b.name, b.plays_race AS race, cp.elo,
       cp.active, cp.division_num, b.bot_data_enabled
FROM competition_participation cp
JOIN bot b ON b.id = cp.bot_id
WHERE cp.competition_id = ?
"""

# "match" is a keyword in SQLite, hence the quotes.
_MATCHES_SQL = """
SELECT m.id AS match_id, m.round_id AS round, m.map_id AS map,
       p1.bot_id AS bot1_id, COALESCE(m.bot1_name, b1.name) AS bot1_name,
       p2.bot_id AS bot2_id, COALESCE(m.bot2_name, b2.name) AS bot2_name,
       m.result_type, m.result_game_steps AS game_steps,
       m.created AS match_created, m.started AS match_started, m.result_created
FROM "match" m
JOIN round r ON r.id = m.round_id
LEFT JOIN match_participation p1 ON p1.match_id = m.id AND p1.participant_number = 1
LEFT JOIN match_participation p2 ON p2.match_id = m.id AND p2.participant_number = 2
LEFT JOIN bot b1 ON b1.id = p1.bot_id
LEFT JOIN bot b2 ON b2.id = p2.bot_id
WHERE r.competition_id = ? AND m.result_created IS NOT NULL AND m.result_type IS NOT NULL
  AND (? IS NULL OR m.started >= ?)
ORDER BY m.id
"""


def read_bots(conn: sqlite3.Connection, competition: int) -> pd.DataFrame:
    """The competition's bots with their final ELO and division.

    aiarena marks every participation inactive when a competition closes, so
    for a closed one the ladder is the bots that ended in a division — the
    same rule the recap site uses (`web/season.py`).
    """
    status = conn.execute("SELECT status FROM competition WHERE id = ?", (competition,)).fetchone()
    if status is None:
        raise SystemExit(f"Competition {competition} is not in the database")
    closed = (status[0] or "").lower() == "closed"

    bots = pd.read_sql_query(_BOTS_SQL, conn, params=(competition,))
    if closed:
        bots["active"] = bots["division_num"].fillna(0) > 0
    else:
        bots["active"] = bots["active"] == 1
    bots["bot_data_enabled"] = bots["bot_data_enabled"] == 1
    log.info("Competition %d (%s): %d bots, %d on the ladder",
             competition, "closed" if closed else "open", len(bots), bots["active"].sum())

    missing_elo = bots["active"] & bots["elo"].isna()
    if missing_elo.any():
        raise SystemExit(f"{missing_elo.sum()} ladder bots have no ELO: {bots.loc[missing_elo, 'name'].tolist()}")
    bots["elo"] = bots["elo"].astype("Int64")
    bots["division"] = bots["division_num"].fillna(0).astype(int)
    return bots[["bot_id", "name", "race", "elo", "active", "bot_data_enabled", "division"]]


def _iso_utc(values: pd.Series) -> pd.Series:
    """SQLite's naive UTC datetimes, written like the API's (…Z)."""
    return pd.to_datetime(values, utc=True, format="ISO8601").dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def read_matches(conn: sqlite3.Connection, competition: int, since: str | None = None) -> pd.DataFrame:
    """The competition's finished matches, one row each; with `since` (a
    date), only those started on or after it."""
    matches = pd.read_sql_query(_MATCHES_SQL, conn, params=(competition, since, since))
    incomplete = matches["bot1_id"].isna() | matches["bot2_id"].isna()
    if incomplete.any():
        log.warning("Skipping %d matches without both participations", incomplete.sum())
        matches = matches[~incomplete].copy()

    for col in ("bot1_id", "bot2_id"):
        matches[col] = matches[col].astype(int)
    steps = matches["game_steps"]
    matches["game_steps"] = steps.astype("Int64")
    # Computed and rounded exactly as fetch.py does, with no duration when
    # the result has no game steps.
    matches["duration_minutes"] = (steps.where(steps > 0) / STEPS_PER_SECOND / 60).map(
        lambda minutes: round(float(minutes), 2), na_action="ignore",
    )
    matches["is_timeout"] = steps.fillna(0) >= TIME_LIMIT_STEPS
    for col in ("match_created", "match_started", "result_created"):
        matches[col] = _iso_utc(matches[col])
    return matches[MATCH_COLUMNS]


def main():
    parser = argparse.ArgumentParser(description="Build bots.csv and matches.csv from an AI Arena Recap database")
    parser.add_argument("db", type=Path, help="Recap SQLite database")
    parser.add_argument("--competition", type=int, default=36,
                        help="Competition ID (default: 36, 2026 Season 1)")
    parser.add_argument("--since", help="Only matches started on or after this date (YYYY-MM-DD)")
    parser.add_argument("--output-dir", type=Path, default=_own_dir, help="Output directory (default: data/)")
    args = parser.parse_args()

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    try:
        bots = read_bots(conn, args.competition)
        matches = read_matches(conn, args.competition, args.since)
    finally:
        conn.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    bots.to_csv(args.output_dir / "bots.csv", index=False)
    matches.to_csv(args.output_dir / "matches.csv", index=False)
    log.info("Wrote %s (%d bots)", args.output_dir / "bots.csv", len(bots))
    started = matches["match_started"].dropna()
    log.info("Wrote %s (%d matches, %s to %s)", args.output_dir / "matches.csv", len(matches),
             started.min()[:10], started.max()[:10])
    for rtype, count in matches["result_type"].value_counts().items():
        log.info("  %s: %d", rtype, count)


if __name__ == "__main__":
    main()
