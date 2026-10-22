# Processed data

Generated artifacts produced by `scripts/build_dataset.py`. Not committed to git.

```
match_features.parquet   # primary, typed, fast
match_features.csv       # human-inspectable mirror
```

Every column here is either:
- an identifier (date, teams, tournament, venue), or
- a strictly pre-match feature (known before kickoff), or
- a target / post-match column (clearly named, excluded from model features).

Regenerate with:

```bash
python scripts/build_dataset.py --config configs/base.yaml
```
