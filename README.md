# Football Match Predictor

A production-ready, modular system for predicting football/soccer match outcomes.
It is built **national-teams-first** (using the Kaggle international results dataset)
but designed so it can later support club football.

The single most important design principle:

> **Every feature must represent information that was available _before_ kickoff.**

Scores are used **only** to build targets and to update Elo *after* a match; they
are never exposed as model features. The leakage tests enforce this.

---

## 1. Project overview

The system can:

- load / download football datasets (Kaggle international results + optional extras),
- clean & normalize team names (incl. historical renames via `former_names.csv`),
- build strictly pre-match features:
  - custom **Elo** ratings (competition-weighted K-factor + margin-of-victory),
  - rolling **recent form** (windows 5/10/20, opponent-adjusted),
  - **scorer-derived** features from `goalscorers.csv`,
  - optional **FIFA rankings** (as-of merge), **odds**, **player/squad**, **advanced stats**,
- train multiple models:
  - **Elo baseline** (calibrated 1X2 from expected score),
  - **Poisson** goal model (+ optional **Dixon-Coles** low-score correction),
  - **tabular ML** (LightGBM → CatBoost → XGBoost → sklearn fallback),
  - **ensemble** (weighted average; market weight redistributed if odds disabled),
- predict 1X2, expected goals, Over/Under 1.5/2.5/3.5, BTTS, and correct-score probabilities,
- evaluate with **time-based** validation and **walk-forward** backtesting,
- serve predictions via a **CLI** and an optional **FastAPI** API.

---

## 2. Installation

```bash
python -m venv .venv
```

Linux/Mac:

```bash
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
.venv\Scripts\activate
```

Install:

```bash
pip install -r requirements.txt
```

Minimal core (no gradient-boosting / API) also works — the tabular model falls
back to scikit-learn's `HistGradientBoostingClassifier`:

```bash
pip install pandas numpy scikit-learn scipy pyyaml pydantic python-dotenv typer rich joblib pyarrow pytest
```

Copy the env template and fill in any keys you have (never commit real secrets):

```bash
cp .env.example .env
```

---

## 3. Dataset setup

### Kaggle file structure (required + optional)

```
data/raw/kaggle_international_results/
  results.csv        (required)
  shootouts.csv      (optional)
  goalscorers.csv    (optional)
  former_names.csv   (optional)
```

Optional extra sources:

```
data/raw/fifa_rankings.csv
data/raw/odds.csv
data/raw/player_squad_features.csv
data/raw/team_advanced_stats.csv
```

The pipeline runs as long as `results.csv` exists. Missing optional files produce
a warning and are skipped.

### Automatic download

```bash
python scripts/download_data.py --source all
python scripts/download_data.py --source kaggle_international_results
```

Requires `KAGGLE_USERNAME` and `KAGGLE_KEY` in `.env` (or a `kaggle.json`). If
credentials are missing, clear manual instructions are printed — nothing crashes.

### Manual dataset setup

Place the files here:

```
data/raw/kaggle_international_results/results.csv
data/raw/kaggle_international_results/shootouts.csv
data/raw/kaggle_international_results/goalscorers.csv
data/raw/kaggle_international_results/former_names.csv
```

---

## 4. Workflow

```bash
python scripts/download_data.py --source all
python scripts/build_dataset.py   --config configs/base.yaml
python scripts/train_models.py    --config configs/base.yaml
python scripts/evaluate_models.py --config configs/base.yaml
python scripts/backtest.py        --config configs/base.yaml --start-year 2019 --end-year 2024
```

### Build the dataset

`build_dataset.py` loads raw data, normalizes names, creates targets + all
pre-match features, and writes:

```
data/processed/match_features.parquet
data/processed/match_features.csv
```

### Train models

`train_models.py` uses a strict **time-based** split (see `configs/base.yaml` →
`validation`) and writes artifacts to `models/artifacts/`:

```
elo_config.json        elo_baseline.json
poisson_model.joblib   tabular_model.joblib
calibrator.joblib      ensemble_config.json
feature_columns.json   model_versions.json
```

### Evaluate / backtest

`evaluate_models.py` reports 1X2 metrics (accuracy, log loss, multiclass Brier,
macro-F1), binary-market metrics (Over 2.5, BTTS), expected-goals error, a
confusion matrix, and year/competition breakdowns, and saves calibration plots
when matplotlib is installed. `backtest.py` performs walk-forward validation.

---

## 5. Predictions

### CLI

```bash
python scripts/predict_match.py \
  --home-team "Japan" \
  --away-team "Tunisia" \
  --date "2026-06-21" \
  --tournament "FIFA World Cup" \
  --neutral true \
  --host-country "United States" \
  --prediction-mode early_prediction
```

Output (shape):

```json
{
  "home_team": "Japan",
  "away_team": "Tunisia",
  "date": "2026-06-21",
  "prediction_mode": "early_prediction",
  "predictions": {
    "home_win_prob": 0.52, "draw_prob": 0.27, "away_win_prob": 0.21,
    "lambda_home": 1.42, "lambda_away": 0.91,
    "over_15_prob": 0.68, "over_25_prob": 0.44, "over_35_prob": 0.22,
    "btts_prob": 0.46,
    "top_scorelines": [{"score": "1-0", "prob": 0.13}, {"score": "1-1", "prob": 0.12}]
  },
  "model_versions": {"elo": "v1", "poisson": "v1", "tabular": "v1", "ensemble": "v1"}
}
```

### Web UI + FastAPI

From the project root (with dependencies installed):

```bash
pip install fastapi uvicorn
uvicorn football_predictor.api.main:app --reload
```

Open **http://127.0.0.1:8000/** for the simple web form: pick teams, date, tournament,
neutral/host country, and prediction mode, then **Predict**. Use **Retrain (build + train)**
to rebuild `data/processed/` and refresh `models/artifacts/` (runs in the background;
the page polls status until finished).

API endpoints: `GET /health`, `GET /teams`, `POST /predict`, `POST /predict/batch`,
`GET /model-info`, `GET /retrain/status`, `POST /retrain`.

---

## 6. Prediction modes

- `early_prediction` — days/hours before kickoff. Uses Elo, FIFA rank, recent form,
  tournament, venue, known squad/injuries, and only odds available at prediction time.
  **Forbids** confirmed lineups, confirmed formation/keeper, and late odds.
- `lineup_known_prediction` — after official lineups: additionally uses confirmed
  starting XI / subs / keeper / formation and last-minute injuries/suspensions.

The mode is a CLI/API argument and gates which player features are emitted.

---

## 7. Data-leakage prevention

- Scores create **targets only**; `home_score`, `away_score`, all 1X2/over/BTTS
  targets, post-match Elo, and advancement columns live in
  `TARGET_AND_LEAKAGE_COLUMNS` and are dropped before training.
- Elo records **pre-match** ratings and updates **after**.
- Rolling and scorer features use `shift(1)` so the current match never feeds itself.
- FIFA rankings and odds use backward **as-of** merges; odds additionally respect a
  prediction-time cutoff so closing odds never leak into early predictions.
- Player snapshots are merged **as-of** (`snapshot_date <= match_date`); confirmed
  lineups are only used in `lineup_known_prediction`.
- `tests/test_no_leakage.py` enforces all of the above.

---

## 8. Model architecture

```
Elo baseline ─┐
Poisson ──────┤
Tabular ML ───┼──► weighted ensemble ──► calibrated 1X2 + goal markets
Market odds ──┘ (optional; weight redistributed if disabled)
```

- **Elo baseline** maps the home expected score to W/D/L via a fitted draw model.
- **Poisson** predicts `lambda_home`/`lambda_away`, builds the full scoreline matrix
  (optionally Dixon-Coles corrected) and derives every market.
- **Tabular ML** is a multiclass classifier with median imputation + one-hot
  categoricals; probabilities are isotonic-calibrated on the validation split.
- **Ensemble** is a renormalized weighted average; goal markets come from Poisson.

---

## 9. Optional player data

Player/squad data is fully optional and off by default (`features.use_player_features`).
Raw rows are aggregated into team-level pre-match features (market value, age,
availability, recent minutes/goals/xG, starting-XI strength in lineup mode). A
missing file simply disables these features without breaking anything.

---

## 10. Configuration

All behaviour is config-driven (`configs/base.yaml`, `configs/model.yaml`,
`configs/data_sources.yaml`). Toggle feature groups under `features:`, change the
time-split under `validation:`, tune Elo under `elo:`, and ensemble weights under
`ensemble:`.

---

## 11. Tests

```bash
pytest
```

Covers Elo, rolling features, scorer features, player features, the Poisson model,
the end-to-end prediction pipeline, and a dedicated no-leakage suite. Tests use
synthetic data and never require the real download.

---

## 12. Future improvements

Lineup-based & player-level models; injury/suspension API integration; club-form
aggregation for national-team players; travel/timezone/weather/altitude factors;
full tournament simulation; betting-value detection & odds comparison; corners /
cards / shots / xG models; and a live in-game prediction model.