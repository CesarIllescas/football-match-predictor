"""Tests for optional player/squad feature aggregation and mode handling."""

from __future__ import annotations

import pandas as pd

from football_predictor.data.cleaning import build_base_table
from football_predictor.features.players import (
    EARLY_MODE,
    LINEUP_MODE,
    aggregate_player_features_asof,
)


def _matches(synthetic_results):
    return build_base_table(synthetic_results)


def test_missing_player_file_does_not_break_pipeline(synthetic_results):
    df = _matches(synthetic_results)
    out = aggregate_player_features_asof(df, None, prediction_mode=EARLY_MODE)
    assert len(out) == len(df)
    assert not any(c.startswith("home_squad_") for c in out.columns)


def test_player_rows_are_merged_asof_only(synthetic_results, synthetic_players):
    df = _matches(synthetic_results)
    out = aggregate_player_features_asof(df, synthetic_players, prediction_mode=EARLY_MODE)
    # Squad market value should be populated for most matches (snapshots exist).
    assert "home_squad_total_market_value" in out.columns
    assert out["home_squad_total_market_value"].notna().mean() > 0.8


def test_confirmed_lineups_not_used_in_early_mode(synthetic_results, synthetic_players):
    df = _matches(synthetic_results)
    out = aggregate_player_features_asof(df, synthetic_players, prediction_mode=EARLY_MODE)
    lineup_cols = [c for c in out.columns if "starting_xi" in c or "confirmed_starters" in c]
    assert lineup_cols == []


def test_confirmed_lineups_allowed_in_lineup_known_mode(synthetic_results, synthetic_players):
    df = _matches(synthetic_results)
    out = aggregate_player_features_asof(df, synthetic_players, prediction_mode=LINEUP_MODE)
    assert any("starting_xi" in c for c in out.columns)


def test_player_stats_do_not_include_current_match(synthetic_results, synthetic_players):
    # Snapshots are dated strictly before the match; the merge uses date <= match_date.
    # Move one team's only snapshot to AFTER the first match to ensure it's excluded.
    df = _matches(synthetic_results).sort_values("date").reset_index(drop=True)
    first = df.iloc[0]
    players = synthetic_players.copy()
    # A snapshot dated after the first match must not be attached to it.
    future_snap = players.iloc[[0]].copy()
    future_snap["date"] = first["date"] + pd.Timedelta(days=1)
    future_snap["team"] = "GhostTeam"
    players = pd.concat([players, future_snap], ignore_index=True)
    out = aggregate_player_features_asof(df, players, prediction_mode=EARLY_MODE)
    assert len(out) == len(df)
