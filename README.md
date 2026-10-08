# StarCraft 2 Matchmaking Prototype

Prototype for matchmaking on AI Arena

## Report

The data, the ladder model, the simulation, the matchmakers and the results are described on one page,
rendered via GitHub Pages:
[maxnus.github.io/sc2-matchmaking-prototype/analysis/report.html](https://maxnus.github.io/sc2-matchmaking-prototype/analysis/report.html)

Its text lives in [`analysis/report.md`](analysis/report.md), Markdown with TeX maths, and
`analysis/analysis.py` turns it into `analysis/report.html`, filling in the figures and every number
quoted in the text from the data, the model and the simulation runs.

## Running

```bash
pip install -e .
python data/from_recap.py recap.sqlite --competition 36 --since 2026-05-01  # data/ from an AI Arena Recap database
python data/fetch.py                       # or recent matches from the AI Arena API (needs AIARENA_API_TOKEN, e.g. in .env)
python model/fit.py                        # refit the outcome model in model/
python matchmakers/stochastic/simulate.py  # likewise for random, roundrobin, rung
python analysis/analysis.py                # writes analysis/report.html from analysis/report.md
```

Each `simulate.py` runs `--seeds` independent simulations (default 8, in parallel) of
`--total-matches` matches (default 50,000), with ratings starting from the bots' AI Arena
ELOs (`--initial-elo flat` starts every bot at 1600 instead). The first `--burn-in` matches
(default 10,000) are left out of the metrics. Runs are scored against the outcome model's true
expected scores and ratings (`sim/metrics.py`), not only against the sim's own ELO, and the
summary reports each metric's mean with a 95% confidence interval across seeds.

To compare another parameter setting of a matchmaker, write its runs to an `output-<setting>` folder
next to `output/`; the page shows it as `<matchmaker>-<setting>`. For example, the stochastic
matchmaker with a higher softmax temperature:

```bash
python matchmakers/stochastic/simulate.py --temperature 0.02 --output-dir matchmakers/stochastic/output-temp-0.02
```
