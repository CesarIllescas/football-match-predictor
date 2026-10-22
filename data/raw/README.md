# Raw data

Place raw, immutable input data here. Nothing in this folder is committed to git
(except these README files).

## Required

```
data/raw/kaggle_international_results/results.csv
```

## Strongly recommended (optional)

```
data/raw/kaggle_international_results/shootouts.csv
data/raw/kaggle_international_results/goalscorers.csv
data/raw/kaggle_international_results/former_names.csv
```

## Optional extra sources

```
data/raw/fifa_rankings.csv
data/raw/odds.csv
data/raw/player_squad_features.csv
data/raw/team_advanced_stats.csv
```

The pipeline runs as long as `results.csv` exists. Every other file simply enables
more features. If an optional file is missing, the build logs a warning and continues.

See the project `README.md` for download / manual setup instructions.
