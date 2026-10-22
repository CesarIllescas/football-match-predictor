"""Tests for rolling recent-form features (the core no-leakage guarantee)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from football_predictor.data.cleaning import build_base_table
from football_predictor.features.rolling import add_rolling_features


def _toy_matches() -> pd.DataFrame:
    rows = [
        ("2020-01-01", "A", "B", 2, 0),
        ("2020-02-01", "A", "C", 1, 1),
        ("2020-03-01", "A", "B", 3, 1),
        ("2020-04-01", "A", "C", 0, 2),
    ]
    df = pd.DataFrame(rows, columns=["date", "home_team", "away_team", "home_score", "away_score"])
    df["tournament"] = "Friendly"
    df["country"] = df["home_team"]
    df["neutral"] = False
    return build_base_table(df)


def test_rolling_features_do_not_include_current_match():
    df = _toy_matches()
    out = add_rolling_features(df, windows=[5])
    a_home = out[out["home_team"] == "A"].sort_values("date")
    # First A match: no prior history -> rolling sums must be NaN (shifted).
    assert pd.isna(a_home.iloc[0]["home_points_last_5"])


def test_last_n_calculations_are_correct():
    df = _toy_matches()
    out = add_rolling_features(df, windows=[5]).sort_values("date").reset_index(drop=True)
    # Team A: match1 win(3pts), match2 draw(1pt), match3 win(3pts), match4 loss(0).
    # Before match3 (A vs B), A has played match1(W) + match2(D) => 4 points.
    row3 = out.iloc[2]
    assert row3["home_team"] == "A"
    assert row3["home_points_last_5"] == 4.0
    assert row3["home_goals_for_last_5"] == 3.0  # 2 + 1


def test_home_and_away_features_assigned_correctly():
    df = _toy_matches()
    out = add_rolling_features(df, windows=[5]).sort_values("date").reset_index(drop=True)
    # Match4: A (home) vs C (away). Before it, C played match2 (draw) -> 1 point.
    row4 = out.iloc[3]
    assert row4["away_team"] == "C"
    assert row4["away_points_last_5"] == 1.0


def test_difference_features_are_correct():
    df = _toy_matches()
    out = add_rolling_features(df, windows=[5]).sort_values("date").reset_index(drop=True)
    row3 = out.iloc[2]
    expected = row3["home_points_last_5"] - row3["away_points_last_5"]
    assert np.isclose(row3["diff_points_last_5"], expected, equal_nan=True) or (
        pd.isna(row3["diff_points_last_5"]) and (pd.isna(row3["home_points_last_5"]) or pd.isna(row3["away_points_last_5"]))
    )
