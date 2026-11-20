# Plan: stochastic matchmaking on AI Arena

Bring this repository's stochastic matchmaker
([report](https://maxnus.github.io/sc2-matchmaking-prototype/analysis/report.html)) to the live ladder in
[aiarena-web](https://github.com/aiarena/aiarena-web).

Paths in phase 0 and in the rollout's comparison step are in this repository; all other paths are in
aiarena-web.

- **Goal.** Matches are picked one at a time, whenever an arena client asks, by scoring every pair of
  free bots on skill, fairness and opponent variety, and sampling a pair. No round-robin schedule, and
  divisions no longer decide who plays whom.
- **Divisions stay**, for display only: they show progress and accomplishment.
- **Round-robin stays**, selectable per competition, so any competition can be switched back from the
  admin.

## Decisions

| # | Decision | Why |
|---|---|---|
| 1 | **Rounds become periods.** In a stochastic competition a round is a day of play with no schedule: matches are attached to the open period as they are picked. | Every match keeps a round, so everything keyed on `match__round__competition` keeps working unchanged: ELO, stats, trends, head-to-head, the competition priority SQL, GraphQL, closing, final standings, finalising. Divisions keep updating at round boundaries. Rollback is clean because both matchmakers produce rounds. The rejected alternative, a `Match.competition` FK and no rounds, means a backfill on `core_match` over several deploys and new triggers for divisions, closing and standings. |
| 2 | **Periods are UTC calendar days**, not a parameter. | Today's rounds already take about a day: 1.01 rounds a day, median round ~27 h, ~1,450 matches a round against ~1,450 a day. So round pages stay the same size and the division cadence is unchanged for the same `rounds_per_cycle`. Days are anchored to midnight because a period only rolls over on the next match request; a 24 h timer would drift. |
| 3 | **Parameters per competition.** | τ, the weights and the window are tuned to a competition. A competition can be trialled alone. |
| 4 | **Every match counts towards fairness**: cancelled, errored and crashed ones too. | Otherwise a bot that fails to start keeps the biggest deficit and gets picked again and again. |
| 5 | **Fairness counts matches, over a window measured in matches**: the competition's last W·n/2 matches (W games per bot on average, running matches included), not the last 24 h. | Server time would starve bots with long games. Per-bot mean game length ranges from 10 min (p10) to 21 min (p90), max 40. W sets how much fairness weighs against skill: a bot k games behind scores ≈ k/(W√2), so at W = 10, about 1.5 games behind outweighs the whole skill term at `w_skill` = 0.1. A 24 h window ties W to throughput: ~17 in production against 13.7 in the sim, and less during outages. A window in matches keeps the tuned balance and carries over from the sim regardless of game time against wall time. **Default W = 10**: the results hardly change between 5 and 28 (phase 0), so it is a round number in that range, about 14 hours of play in production. |
| 6 | **τ fixed in rank positions** (default 15), not scaled with ladder size. | τ sets how many neighbours count as close. Variety, rematch blocks and busy bots need about the same number of neighbours whatever the ladder size. Small competitions (n ≲ 2τ) then play everyone, as round-robin does with one division. Closeness is tuned through `w_skill` against `w_fair` and W, not τ: in the sim, a quarter of stochastic matches are more than 40 ranks apart. |
| 7 | **New bots play immediately** in stochastic competitions, before they have a division. | The fairness term already favours them; waiting for the next division update would only delay them. |
| 8 | **Temperature T = 0.02.** | Against 0.01 it gives up a little closeness (favourite's expected score 0.728 → 0.741) for less stretched ratings (error 57 → 48 ELO), more varied opponents and more even match counts. At 0.05 matches get markedly less close (0.757, one in five lopsided) and the ratings order the bots less well, for only a little more variety and evenness. |
| 9 | **Stochastic from a competition's first bots on.** The rematch block applies only from three active bots up. | A competition opens with only a few bots. With two, the rule "neither bot's latest opponent may be the other" would stop them from ever playing again. From three up it can't stall: the latest opponents block at most n − 1 of the n(n − 1)/2 pairs, so once every bot is free some pair is allowed. |
| 10 | **Every bot plays one match at a time** in stochastic competitions, bots without bot data included. Replaces the parallel-match cap setting proposed earlier. | On a small ladder with spare servers, bots without bot data would otherwise take every spare server: on 8 bots, 163 matches each against 55 for bots with bot data, and on 16 bots each bot meets only 13 of its 15 possible opponents. One at a time keeps counts even, as round-robin does, at the cost of leaving servers idle, as round-robin does. On the full ladder it changes nothing but making match counts slightly more even (standard deviation 8.8 → 7.5): only about 22 of 173 bots are busy at any moment. The smallest ladders play more slowly: 4 bots can run only 2 matches at once (161 matches a day against round-robin's 249); from 8 bots up they're level. |
| 11 | **After switching to stochastic, leftover round-robin matches are played out**, not deleted. The stochastic matchmaker fills every slot they can't use. | It needs no new code, since unstarted matches in open rounds are already started first, and it loses no scheduled match. Switching back halfway simply continues the same rounds. The cost is up to about a day of mostly round-robin matches, which decision 12 keeps apart from stochastic ones. Deleting them would need a switch action that deletes under the lock and then completes the emptied rounds; an admin action for that can be added if an immediate clean start is ever wanted. |
| 12 | **A round records which matchmaker created it.** | So the UI can label periods as days, also in the history of a competition that was switched back, and the rollout can select exactly the matches stochastic picked, including during the transition. Neither the period flag (true only while a period is open) nor the competition's current matchmaker can tell this afterwards. Scheduling doesn't use it. |

## Concepts

- **Period**: a round of a stochastic competition. The first match request of a UTC day seals the
  previous period and opens a new one. Matches are created and started in one go and attached to the
  open period.
- **Sealed**: no more matches will be added to the round. Round-robin rounds are sealed when created,
  because their schedule is complete. Periods are sealed at rollover, when the competition starts
  closing, and when a switch back to round-robin opens a round.
- **Complete**: sealed, and every match has a result. A round is never completed while unsealed;
  otherwise a period would complete whenever its running matches happened to run out.
- **A period never holds an unstarted match.** Stochastic creates and starts each match in one
  savepoint, and rolls it back if the start fails. So "start unstarted matches in open rounds" only
  ever finds round-robin leftovers (decision 11). This belongs in the matchmaking package's
  CLAUDE.md once periods exist.
- **Availability is global; fairness and variety are per competition.** A bot playing in another
  competition, or in a requested match, is busy. But only this competition's matches count towards
  its fairness and variety here.

## Repositories

| Repository | Change |
|---|---|
| **aiarena-web** | All of the implementation: phases 1–5. |
| **sc2-ai-match-controller** | None. Each arena client asks for a match (`getNextMatch`) when its previous job ends. That is the simulator's "slot freed, ask the matchmaker" event, so the stochastic matchmaker plugs into the existing `ACCoordinator.next_match` path. `REISSUE_UNFINISHED_MATCHES` must stay on: the k8s controller asks for a match to create the job, and the job's match controller asks again and must get the same match back. |
| **sc2-matchmaking-prototype** | Re-tuning before go-live: phase 0. |

## Phase 0: re-tune in the prototype

Independent of phases 1–4; needed before the first competition switches.

1. **Fairness window in matches.** *Done.* The window is the last `W·n/2` dispatched matches; at
   W = 14 it reproduces the 24-hour window's results within their confidence intervals.
2. **Defaults.** *Done:* W = 10 and T = 0.02 (decisions 5 and 8), now the prototype's main setting,
   with W = 5, 20 and T = 0.01, 0.05 compared in the report.
3. **Parallel matches.** *Done:* every bot plays one match at a time (decision 10), now part of the
   prototype's matchmaker, with parallel matches compared in the report.
4. **Small ladders.** *Done:* the simulator plays a random subset of N bots (`--n-bots`), and the
   report compares stochastic with round-robin in one division on 2 to 32 bots, starting from flat
   ratings as a new competition does. This found the two-bot rematch stall (decision 9) and the
   uneven match counts that decision 10 fixes.
5. **Parity fixtures.** *Done:* `matchmakers/stochastic/export_parity_fixtures.py` writes random
   snapshots and the score components the prototype computes for them, under two parameter
   settings; aiarena-web's tests check its scoring core against them.

Output: the default parameter values for phase 2, and the fixtures for phase 4.

## Phase 1: split the matchmaking code (no behaviour change)

*Done:* [maxnus/aiarena-web#1](https://github.com/maxnus/aiarena-web/pull/1).

`aiarena/core/services/service_implementations/_matches.py` mixes round-robin scheduling, division
assignment, starting and prioritising round matches, and cancellations and timeouts. Split it before
adding a second matchmaker:

- **Divisions**: the division assignment, moved out of `_attempt_to_generate_new_round` unchanged
  (sort by ELO, placements, split and merge).
- **Opening a round**, shared by both matchmakers: create the `Round`, advance `rounds_this_cycle`,
  run division assignment at the start of a cycle, and set `participated_in_most_recent_round`.
- **Round-robin matchmaker**: schedule generation, `_attempt_to_start_a_ladder_match`, its
  prioritisation and `_last_ladder_match_starts`.
- **`Matches`** keeps requested matches, cancellations and timeouts.
- **Trusted-client rule.** Move the check in `MatchStarter.__requires_trusted_arenaclient` into a
  predicate that also works before a match exists (competition flag, bots' zip and data public
  flags), so the stochastic matchmaker can filter candidates with the same rule.
- **Local CLAUDE.md** for the matchmaking package: periods, sealed against complete, global
  availability against per-competition scoring, and the matchmaker boundary.

The existing tests are the regression suite and must pass unchanged: `test_round_robin_generation`,
`test_divisions`, `test_match_prioritization`, `test_last_ladder_match_starts`, `test_match_starter`.

## Phase 2: schema

*Done, with phases 3 and 4,* on the branch `ccr-2b67f687-pe4xp3-stochastic`, stacked on phase 1.
As built, the parameters have descriptive names (below), and they sit next to the division
settings rather than in a fieldset: the competition admin lists every field in model order, and a
fieldset would hide any field later added outside it. Left blank, a parameter takes its default.

All fields are additive, nullable with `# TODO(make-not-null)`, following
`aiarena/core/models/CLAUDE.md`. They can ship with phases 3–4, because migrations run first. Backfill
and NOT NULL follow in a later deploy.

**`Competition`**:

| Field | Default | Notes |
|---|---|---|
| `matchmaker` | `round_robin` | Choices `round_robin`, `stochastic`. NULL is read as `round_robin`. This is the rollback switch. |
| `skill_weight` | 0.1 | |
| `fairness_weight` | 1.0 | |
| `variety_weight` | 0.2 | |
| `skill_tolerance_ranks` | 15 | τ, in rank positions. |
| `variety_memory_games` | 40 | λ: games until a past match-up is forgotten. |
| `sampling_temperature` | 0.02 | T; 0 means always take the best pair. |
| `fairness_window_games` | 10 | W. |

Update the `rounds_per_cycle` help text: in stochastic competitions it only sets how often divisions
are refreshed, counted in days.

**`Round`**:

| Field | Default | Notes |
|---|---|---|
| `accepts_new_matches` | False | True only for an open period. False and NULL mean the round-robin behaviour, so NULLs written by old code during the rollout are harmless. |
| `matchmaker` | `round_robin` | Which matchmaker created the round. NULL means round-robin (all existing rows). |

## Phase 3: round lifecycle

As built, the seal calls became one rule: opening any round seals the competition's open period,
which covers both the daily rollover and a switch back to round-robin. On top of that, every
request of a competition that isn't stochastic seals an open period first. Otherwise, after a
switch back with `max_active_rounds` = 1, the open period would count as the one active round and
could never complete, and round-robin could never open a round.

- `update_round_if_completed` (`internal/rounds.py`) returns early while the round
  `accepts_new_matches`.
- A **seal** helper sets `accepts_new_matches = False` and then calls `update_round_if_completed`.
  Calling the second step matters: a period with no running matches would otherwise never be
  completed. It is called:
  - at period rollover;
  - when a competition starts closing (`Competition.start_closing`, via the admin action), so
    `try_to_close` fires after the last running match;
  - when the round-robin matchmaker opens a round, which seals an open period after a rollback.
- Paused and closing stochastic competitions start no new matches. They raise `CompetitionPaused` or
  `CompetitionClosing`, as round-robin does when it would need a new round. `max_active_rounds`
  applies to round-robin only.

Tests:
- A period whose matches have all finished stays incomplete until sealed.
- Sealing an idle period completes it, and closes a closing competition.

## Phase 4: the stochastic matchmaker

As built, "no allowed pair" raises the existing `NotEnoughAvailableBots`, which the coordinator
already catches. The variety history is also bounded to the last 28 days, only to keep its query
small; an active bot plays its 200 games of history in far less time.

### 4.1 Scoring core

Port from the prototype almost line for line: `f_skill`, `f_fair`, `f_var`, the weighted score and
softmax sampling (numpy is already a dependency).

- Arrays in, a pair index out. It knows nothing about Django.
- Its random generator can be injected, so tests can seed it.
- Tests compare it with the phase 0 parity fixtures.

### 4.2 Snapshot

The snapshot turns database state into the core's inputs. It is built on every request, inside the
transaction that already holds `select_for_update` on the competition's participations
(`competitions.check_has_matches_to_play_and_apply_locks`). Requests for the same competition are
therefore handled one at a time, and each sees the previous one's match as running. That is the
production form of the prototype's "bookkeeping at dispatch".

- **Participants**: active participations, with bot id, ELO, `bot_data_enabled` and the zip/data
  public flags. Ranks go by ELO descending, ties by bot id ascending.
- **Availability**, global across competitions and requested matches, from one query over these
  bots' started but unfinished matches:
  - A bot is busy while it is in any started, unfinished match, whether or not it has bot data
    (decision 10). That is stricter than `MatchStarter`'s rule, which only stops a bot whose bot
    data another match is updating, and which stays the final check at start time.
  - For an untrusted arena client, only bots whose zip and data are public are candidates. The
    coordinator already keeps untrusted clients out of competitions that require trusted
    infrastructure.
- **Fairness `g`**: per-bot counts over the competition's last `round(W·n/2)` matches by start time.
  These are matches with any result, including cancelled and errored ones, or still running. `n` is
  the number of active participants, and `ḡ` is the mean over all of them.
- **Variety and rematch block**: each candidate's last `ceil(5λ)` matches in the competition, by start
  time, give
  - the games each bot has played since it last met each other bot. If they haven't met within that
    history, the number of games in it is used, which matches the prototype for bots with fewer
    games; beyond 5λ the variety term is already ≈ 1;
  - each bot's last opponent;
  - the pairs currently playing.
- **Candidate pairs**: available × available, minus pairs currently playing and, when the
  competition has at least three active bots, pairs where either bot's last opponent was the other
  (decision 9).

A handful of queries in all. The fairness and variety data can come from one query over the
competitors' recent matches. Measure on production-sized data; add a cache only if needed.

### 4.3 Dispatch

1. If there is no open period, or it started before today's 00:00 UTC: seal it and open a new round
   (phase 1's shared step, which may update divisions).
2. Score and sample a pair. If no pair is allowed, raise an exception that the coordinator catches,
   like `NotEnoughAvailableBots`, so it moves on to the next competition. Returning `None` would stop
   `next_competition_match` from trying other competitions.
3. Create the match in the open period with a random map of the competition (raising `NoMaps` if
   there are none) and `require_trusted_arenaclient` from the competition. Start it with
   `MatchStarter.start` in a savepoint.
4. If the start fails (a race with another competition's dispatch), roll back the savepoint, drop the
   pair and sample again, up to a few times.

### 4.4 Wiring

`start_next_match_for_competition` becomes:
1. Start any unstarted match in an open round: the existing code. Between switches this finds
   nothing; after a switch to stochastic, it plays out the leftover round-robin matches.
2. Otherwise, hand over to the competition's matchmaker.

### 4.5 Tests

Tests use the arena client helper in `aiarena/api/arenaclient/testing_utils.py`; time is patched with
`unittest.mock`, since there is no freezegun.

- **Snapshot**:
  - a bot that is busy in another competition or in a requested match, with and without bot data;
  - the trust filter;
  - window counts including cancelled matches;
  - games since last met, the rematch block and pairs currently playing.
- **Dispatch**:
  - rollover at 00:00 UTC, and a division update at a cycle boundary;
  - a division-0 bot gets matched;
  - the coordinator moves on when only blocked pairs remain;
  - two dispatches in a row see each other's running match.
- **Switching**:
  - round-robin to stochastic plays out the leftovers;
  - stochastic to round-robin seals the period and opens a round from the current divisions;
  - matches still running from the other mode are scored normally;
  - closing closes after the last running match.

## Phase 5: display and documentation

- Expose `matchmaker` on `CompetitionType` and `RoundsType` in GraphQL. Run `uv run dev pre-commit`
  to regenerate the schema.
- SPA and Django competition pages: show the matchmaker. Label periods as days, e.g. *2026-12-03*
  instead of *Round 412*, using the round's `matchmaker`.
- `templates/rules.html` describes round-robin; add the stochastic rules: how pairs are scored,
  daily periods, and divisions for display.
- The match queue page shows only running matches for stochastic competitions; there is no queue.
  Say so on the page.

## Phase 6: rollout

1. **Deploy phases 1–5** with every competition on `round_robin`, and confirm nothing changes.
2. **Deploy the make-not-null follow-up.**
3. **Switch a competition** to `stochastic` in the admin.
4. **Expect ratings to adjust over about a week.** The sim's burn-in was 10,000 matches, about
   7 days at the current ~1,450 a day.
5. **Compare with the prototype's predictions**, using this repository's tooling: `data/fetch.py`
   pulls the real matches after the switch. Metrics that need no ground truth:
   - matches per bot (spread, least-played bot);
   - distinct opponents and rematch gaps;
   - rank gap and |ΔELO| of paired bots;
   - ELO spread over time.
6. **Watch the servers**: `getNextMatch` latency, and how often arena clients get no match.
7. **To roll back**, set `matchmaker` back to `round_robin`. The next request seals the period and
   opens a round-robin round from the current divisions. No data migration is needed.

## Unchanged

- **Results, ELO, statistics, trophies, finalising, and the competition priority order**: every match
  still belongs to a round.
- **Requested matches**: still dispatched first, and they don't count towards ladder fairness. They do
  make a bot with bot data busy.
- **sc2-ai-match-controller.**
