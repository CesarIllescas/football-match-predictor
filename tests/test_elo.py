"""Tests for the Elo rating system."""

from __future__ import annotations

from football_predictor.features.elo import EloRatingSystem


def test_ratings_initialize_correctly():
    elo = EloRatingSystem(initial_rating=1500)
    assert elo.get_rating("NewTeam") == 1500


def test_home_advantage_changes_expected_probability():
    elo = EloRatingSystem(initial_rating=1500, home_advantage=100)
    # Equal base ratings: with home advantage, home expected > 0.5.
    feats = elo.update_match("A", "B", home_score=None, away_score=None,
                             tournament="Friendly", neutral=False)
    assert feats["home_elo_expected"] > 0.5


def test_neutral_venue_removes_home_advantage():
    elo = EloRatingSystem(initial_rating=1500, home_advantage=100, neutral_home_advantage=0)
    feats = elo.update_match("A", "B", home_score=None, away_score=None,
                             tournament="Friendly", neutral=True)
    assert abs(feats["home_elo_expected"] - 0.5) < 1e-9


def test_ratings_update_after_home_win():
    elo = EloRatingSystem(initial_rating=1500, home_advantage=0, use_margin_multiplier=False)
    before = elo.get_rating("A")
    elo.update_match("A", "B", 3, 0, "Friendly", neutral=True)
    assert elo.get_rating("A") > before
    assert elo.get_rating("B") < before


def test_ratings_update_after_away_win():
    elo = EloRatingSystem(initial_rating=1500, home_advantage=0, use_margin_multiplier=False)
    elo.update_match("A", "B", 0, 2, "Friendly", neutral=True)
    assert elo.get_rating("B") > 1500
    assert elo.get_rating("A") < 1500


def test_ratings_after_draw_between_equals_unchanged():
    elo = EloRatingSystem(initial_rating=1500, home_advantage=0, use_margin_multiplier=False)
    elo.update_match("A", "B", 1, 1, "Friendly", neutral=True)
    # Equal ratings + neutral + draw -> expected 0.5, actual 0.5 -> no change.
    assert abs(elo.get_rating("A") - 1500) < 1e-9
    assert abs(elo.get_rating("B") - 1500) < 1e-9


def test_process_matches_returns_pre_match_features(synthetic_results):
    from football_predictor.data.cleaning import clean_results

    df = clean_results(synthetic_results)
    elo = EloRatingSystem()
    feats = elo.process_matches(df)
    for col in ["home_elo_pre", "away_elo_pre", "elo_diff", "home_elo_expected", "away_elo_expected"]:
        assert col in feats.columns
    # First-ever match for both teams should have pre-rating == initial.
    assert feats.iloc[0]["home_elo_pre"] == 1500
