# StarCraft 2 Matchmaking Prototype

Prototype for matchmaking on AI Arena

## Analysis report

The analysis report is rendered via GitHub Pages:
[maxnus.github.io/sc2-matchmaking-prototype/analysis/report.html](https://maxnus.github.io/sc2-matchmaking-prototype/analysis/report.html)

## Running

```bash
pip install -e .
python data/fetch.py                       # optional: refresh data/ (needs AIARENA_API_TOKEN, e.g. in .env)
python model/fit.py                        # refit the outcome model in model/
python matchmakers/stochastic/simulate.py  # likewise for random, roundrobin, rung
python analysis/analysis.py                # writes analysis/report.html
```

Each `simulate.py` runs `--seeds` independent simulations (default 8, in parallel) of
`--total-matches` matches (default 25,000), with ratings starting from the bots' AI Arena
ELOs (`--initial-elo flat` starts every bot at 1600 instead). The first `--burn-in` matches
(default 5,000) are left out of the metrics. Runs are scored against the outcome model's true
expected scores and ratings (`sim/metrics.py`), not only against the sim's own ELO, and the
summary reports each metric's mean with a 95% confidence interval across seeds.
