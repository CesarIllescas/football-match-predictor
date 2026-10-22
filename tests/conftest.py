"""Shared synthetic-data fixtures for the test suite.

We generate deterministic, leakage-checkable data so tests never depend on the
real Kaggle download.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure src is importable even without an editable install.
_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


TEAMS = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"]


@pytest.fixture
def synthetic_results() -> pd.DataFrame:
    """A round-robin-ish schedule across several seasons with deterministic scores."""
    rng = np.random.default_rng(7)
    rows = []
    start = pd.Timestamp("2015-01-10")
    match_idx = 0
    # Latent strengths drive scores so models have signal.
    strength = {t: s for t, s in zip(TEAMS, np.linspace(1.6, 0.7, len(TEAMS)))}
    for week in range(120):
        date = start + pd.Timedelta(days=7 * week)
        # Two matches per week.
        teams = rng.permutation(TEAMS)
        for i in range(0, len(teams) - 1, 2):
            home, away = teams[i], teams[i + 1]
            hg = int(rng.poisson(strength[home] + 0.25))  # home edge
            ag = int(rng.poisson(strength[away]))
            tournament = "Friendly" if week % 3 else "FIFA World Cup qualification"
            rows.append({
                "date": date,
                "home_team": home,
                "away_team": away,
                "home_score": hg,
                "away_score": ag,
                "tournament": tournament,
                "city": "Town",
                "country": home,
                "neutral": False,
            })
            match_idx += 1
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_goalscorers(synthetic_results: pd.DataFrame) -> pd.DataFrame:
    """Goal events consistent with the result scores (one scorer per goal)."""
    rng = np.random.default_rng(11)
    rows = []
    scorers = [f"Player{i}" for i in range(20)]
    for r in synthetic_results.itertuples(index=False):
        for _ in range(int(r.home_score)):
            rows.append({
                "date": r.date, "home_team": r.home_team, "away_team": r.away_team,
                "team": r.home_team, "scorer": rng.choice(scorers),
                "own_goal": False, "penalty": bool(rng.random() < 0.15),
            })
        for _ in range(int(r.away_score)):
            rows.append({
                "date": r.date, "home_team": r.home_team, "away_team": r.away_team,
                "team": r.away_team, "scorer": rng.choice(scorers),
                "own_goal": False, "penalty": bool(rng.random() < 0.15),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_players(synthetic_results: pd.DataFrame) -> pd.DataFrame:
    """Squad snapshots dated a few days before each unique (team, date)."""
    rng = np.random.default_rng(13)
    rows = []
    unique = pd.concat([
        synthetic_results[["date", "home_team"]].rename(columns={"home_team": "team"}),
        synthetic_results[["date", "away_team"]].rename(columns={"away_team": "team"}),
    ]).drop_duplicates()
    for r in unique.itertuples(index=False):
        snap_date = r.date - pd.Timedelta(days=3)
        for p in range(15):
            rows.append({
                "date": snap_date,
                "team": r.team,
                "player": f"{r.team}_p{p}",
                "position": "MF",
                "club_country": rng.choice(["England", "Spain", "Other"]),
                "age": int(rng.integers(19, 34)),
                "market_value": float(rng.integers(1, 50)),
                "is_goalkeeper": p == 0,
                "is_defender": 1 <= p <= 4,
                "is_midfielder": 5 <= p <= 9,
                "is_forward": p >= 10,
                "is_likely_starter": p < 11,
                "is_confirmed_starter": p < 11,
                "is_injured": bool(rng.random() < 0.05),
                "is_suspended": False,
                "minutes_last_30_days": float(rng.integers(0, 400)),
                "minutes_last_90_days": float(rng.integers(0, 1000)),
                "goals_last_30_days": float(rng.integers(0, 5)),
                "assists_last_30_days": float(rng.integers(0, 5)),
                "xg_last_30_days": float(rng.random() * 3),
                "xa_last_30_days": float(rng.random() * 3),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def base_config():
    from football_predictor.utils.config import Config

    cfg = Config()
    cfg.features.use_fifa_rankings = False
    cfg.features.use_odds = False
    cfg.features.use_scorer_features = True
    cfg.features.use_player_features = False
    cfg.features.use_advanced_stats = False
    return cfg
