"""Leakage guards: no feature may encode post-kickoff information."""

from __future__ import annotations

import pandas as pd

from football_predictor.data import matching
from football_predictor.data.cleaning import build_base_table
from football_predictor.data.schemas import TARGET_AND_LEAKAGE_COLUMNS
from football_predictor.features.feature_builder import build_features
from football_predictor.features.odds import _prepare_odds
from football_predictor.features.players import EARLY_MODE, aggregate_player_features_asof
from football_predictor.features.rankings import add_fifa_ranking_features
from football_predictor.models.tabular_model import TabularResultModel


def test_target_columns_not_included_as_model_features(synthetic_results, base_config):
    raw = matching.normalize_all({"results": synthetic_results, "former_names": None,
                                  "shootouts": None, "goalscorers": None, "fifa_rankings": None,
                                  "odds": None, "player_squad_features": None, "advanced_stats": None})
    df, _ = build_features(raw, base_config)
    model = TabularResultModel(backend_preference=["sklearn"]).fit(df)
    for col in TARGET_AND_LEAKAGE_COLUMNS:
        assert col not in model.feature_columns, f"leakage column {col} used as feature"


def test_ranking_dates_are_always_before_match_dates():
    matches = pd.DataFrame([{
        "date": pd.Timestamp("2020-03-01"), "home_team": "A", "away_team": "B",
    }])
    rankings = pd.DataFrame([
        {"rank_date": "2020-01-01", "team": "A", "rank": 5, "points": 800},
        {"rank_date": "2020-06-01", "team": "A", "rank": 1, "points": 950},  # AFTER match
        {"rank_date": "2020-02-01", "team": "B", "rank": 9, "points": 700},
    ])
    out = add_fifa_ranking_features(matches, rankings)
    # Must use the 2020-01-01 ranking (rank 5), never the future 2020-06-01 one.
    assert out.iloc[0]["home_fifa_rank"] == 5
    assert out.iloc[0]["away_fifa_rank"] == 9


def test_odds_timestamps_are_always_before_kickoff():
    odds = pd.DataFrame([
        {"date": "2020-03-01", "home_team": "A", "away_team": "B",
         "home_odds": 2.0, "draw_odds": 3.3, "away_odds": 4.0,
         "timestamp": "2020-02-28 10:00:00"},
        {"date": "2020-03-01", "home_team": "A", "away_team": "B",
         "home_odds": 1.5, "draw_odds": 4.0, "away_odds": 6.0,
         "timestamp": "2020-03-01 20:00:00"},  # closing odds, after our cutoff
    ])
    cutoff = pd.Timestamp("2020-02-28 12:00:00")
    prepared = _prepare_odds(odds, prediction_timestamp=cutoff)
    assert len(prepared) == 1
    # The kept quote is the pre-cutoff one (implied home prob from 2.0 odds, ~0.43 after overround).
    assert prepared.iloc[0]["market_home_prob"] < 0.5


def test_rolling_feature_source_dates_are_before_match_date():
    rows = [
        ("2020-01-01", "A", "B", 2, 0),
        ("2020-02-01", "A", "B", 0, 3),
    ]
    df = pd.DataFrame(rows, columns=["date", "home_team", "away_team", "home_score", "away_score"])
    df["tournament"] = "Friendly"; df["country"] = "A"; df["neutral"] = False
    from football_predictor.features.rolling import add_rolling_features

    base = build_base_table(df)
    out = add_rolling_features(base, windows=[5]).sort_values("date").reset_index(drop=True)
    # The first match cannot know about the second; its rolling history is empty.
    assert pd.isna(out.iloc[0]["home_goals_for_last_5"])
    # The second match's home rolling reflects only match 1 (2 goals), not its own 0.
    assert out.iloc[1]["home_goals_for_last_5"] == 2.0


def test_player_feature_dates_before_match_date():
    matches = pd.DataFrame([{
        "date": pd.Timestamp("2020-03-01"), "home_team": "A", "away_team": "B",
        "home_goals": 1.0, "away_goals": 0.0,
    }])
    players = pd.DataFrame([
        # Valid snapshot before the match.
        {"date": "2020-02-25", "team": "A", "player": "a1", "market_value": 10.0,
         "is_injured": False, "is_suspended": False, "is_goalkeeper": True,
         "is_likely_starter": True, "is_confirmed_starter": True},
        # Future snapshot that must be ignored.
        {"date": "2020-04-01", "team": "A", "player": "a2", "market_value": 999.0,
         "is_injured": False, "is_suspended": False, "is_goalkeeper": False,
         "is_likely_starter": True, "is_confirmed_starter": True},
    ])
    players["date"] = pd.to_datetime(players["date"])
    out = aggregate_player_features_asof(matches, players, prediction_mode=EARLY_MODE)
    # Only the pre-match snapshot (value 10) should be aggregated; the 999 is future.
    assert out.iloc[0]["home_squad_total_market_value"] == 10.0
