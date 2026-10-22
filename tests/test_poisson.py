"""Tests for the Poisson goal model and derived markets."""

from __future__ import annotations

import numpy as np

from football_predictor.data import matching
from football_predictor.features.feature_builder import build_features
from football_predictor.models.poisson_model import PoissonGoalModel


def _features(synthetic_results, base_config):
    raw = {"results": synthetic_results, "shootouts": None, "goalscorers": None,
           "former_names": None, "fifa_rankings": None, "odds": None,
           "player_squad_features": None, "advanced_stats": None}
    raw = matching.normalize_all(raw)
    df, _ = build_features(raw, base_config)
    return df


def test_score_matrix_sums_to_one():
    model = PoissonGoalModel(max_goals=10)
    matrix = model.score_matrix(1.4, 1.1)
    assert np.isclose(matrix.sum(), 1.0, atol=1e-9)


def test_markets_are_valid_probabilities():
    model = PoissonGoalModel(max_goals=10)
    m = model.predict_markets(1.6, 0.9)
    assert np.isclose(m["home_win_prob"] + m["draw_prob"] + m["away_win_prob"], 1.0, atol=1e-6)
    for key in ["over_15_prob", "over_25_prob", "over_35_prob", "btts_prob"]:
        assert 0.0 <= m[key] <= 1.0
    assert m["over_15_prob"] >= m["over_25_prob"] >= m["over_35_prob"]


def test_fit_and_predict_lambdas_nonnegative(synthetic_results, base_config):
    df = _features(synthetic_results, base_config)
    model = PoissonGoalModel().fit(df)
    lh, la = model.predict_lambdas(df)
    assert (lh > 0).all() and (la > 0).all()


def test_top_scorelines_structure():
    model = PoissonGoalModel()
    m = model.predict_markets(1.5, 1.2, top_n=3)
    assert len(m["top_scorelines"]) == 3
    for s in m["top_scorelines"]:
        assert "-" in s["score"]
        assert 0.0 <= s["prob"] <= 1.0


def test_dixon_coles_changes_low_scores():
    base = PoissonGoalModel(use_dixon_coles=False).score_matrix(1.2, 1.1)
    dc = PoissonGoalModel(use_dixon_coles=True, rho=-0.1).score_matrix(1.2, 1.1)
    # The 0-0 / 1-1 cells should differ once the correction is applied.
    assert not np.isclose(base[0, 0], dc[0, 0])
