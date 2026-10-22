"""Tests for goalscorer-derived rolling features."""

from __future__ import annotations

import pandas as pd

from football_predictor.data.cleaning import build_base_table
from football_predictor.features.scorers import (
    add_scorer_rolling_features,
    build_team_scorer_history,
)


def _matches():
    rows = [
        ("2020-01-01", "A", "B", 2, 0),
        ("2020-02-01", "A", "C", 1, 0),
        ("2020-03-01", "A", "B", 0, 0),
    ]
    df = pd.DataFrame(rows, columns=["date", "home_team", "away_team", "home_score", "away_score"])
    df["tournament"] = "Friendly"
    df["country"] = df["home_team"]
    df["neutral"] = False
    return build_base_table(df)


def _scorers():
    rows = [
        # Match1: A scores 2 (one penalty), one scorer twice -> concentration high.
        ("2020-01-01", "A", "B", "A", "X", False, True),
        ("2020-01-01", "A", "B", "A", "X", False, False),
        # Match2: A scores 1 open play by Y, plus an own goal credited to A.
        ("2020-02-01", "A", "C", "A", "Y", False, False),
        ("2020-02-01", "A", "C", "A", None, True, False),
    ]
    return pd.DataFrame(rows, columns=["date", "home_team", "away_team", "team", "scorer", "own_goal", "penalty"])


def test_history_counts_penalties_and_own_goals():
    hist = build_team_scorer_history(_scorers())
    m1 = hist[(hist["team"] == "A") & (hist["date"] == pd.Timestamp("2020-01-01"))].iloc[0]
    assert m1["penalty_goals"] == 1
    assert m1["open_play_goals"] == 1
    assert m1["total_goals"] == 2
    assert m1["num_unique_scorers"] == 1
    assert m1["top_scorer_goals"] == 2

    m2 = hist[(hist["team"] == "A") & (hist["date"] == pd.Timestamp("2020-02-01"))].iloc[0]
    assert m2["own_goals"] == 1
    assert m2["open_play_goals"] == 1


def test_scorer_features_do_not_include_current_match():
    df = _matches()
    out = add_scorer_rolling_features(df, _scorers(), windows=[5]).sort_values("date").reset_index(drop=True)
    # First A match: no prior scorer history.
    assert pd.isna(out.iloc[0]["home_penalty_goals_last_5"]) or out.iloc[0]["home_penalty_goals_last_5"] == 0 or pd.isna(out.iloc[0]["home_total_goals_last_5"])


def test_goal_concentration_uses_only_prior_matches():
    df = _matches()
    out = add_scorer_rolling_features(df, _scorers(), windows=[5]).sort_values("date").reset_index(drop=True)
    # Before match2 (A vs C), only match1 counts: top scorer 2 / total 2 = 1.0.
    row2 = out.iloc[1]
    assert row2["home_goal_concentration_last_5"] == 1.0
    assert row2["home_penalty_goals_last_5"] == 1.0


def test_missing_goalscorers_returns_unchanged():
    df = _matches()
    out = add_scorer_rolling_features(df, None, windows=[5])
    assert len(out) == len(df)
    assert "home_penalty_goals_last_5" not in out.columns
