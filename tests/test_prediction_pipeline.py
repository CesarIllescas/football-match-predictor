"""End-to-end prediction pipeline tests on synthetic data."""

from __future__ import annotations

import numpy as np

from football_predictor.data import matching
from football_predictor.features.feature_builder import build_features
from football_predictor.prediction.predictor import (
    Predictor,
    predict_proba_df,
    train_pipeline,
)
from football_predictor.utils.dates import time_split


def _raw(synthetic_results):
    return matching.normalize_all({
        "results": synthetic_results, "former_names": None, "shootouts": None,
        "goalscorers": None, "fifa_rankings": None, "odds": None,
        "player_squad_features": None, "advanced_stats": None,
    })


def _train(synthetic_results, base_config):
    raw = _raw(synthetic_results)
    df, _ = build_features(raw, base_config)
    splits = time_split(df, "2018-12-31", "2019-01-01", "2019-12-31", "2020-01-01")
    artifacts = train_pipeline(splits["train"], splits["validation"], base_config)
    return artifacts, df, raw


def test_df_probabilities_sum_to_one(synthetic_results, base_config):
    artifacts, df, _ = _train(synthetic_results, base_config)
    proba = predict_proba_df(artifacts, df)
    assert proba.shape[1] == 3
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert (proba >= 0).all()


def test_expected_goals_non_negative(synthetic_results, base_config):
    artifacts, df, _ = _train(synthetic_results, base_config)
    lh, la = artifacts.poisson.predict_lambdas(df)
    assert (lh >= 0).all() and (la >= 0).all()


def test_single_prediction_is_valid(synthetic_results, base_config):
    artifacts, _df, raw = _train(synthetic_results, base_config)
    predictor = Predictor(artifacts)
    predictor._raw_norm = raw  # inject normalized raw to avoid disk loads

    result = predictor.predict_match(
        home_team="Alpha", away_team="Beta", date="2025-01-01",
        tournament="FIFA World Cup", neutral=True, prediction_mode="early_prediction",
    )
    p = result["predictions"]
    total = p["home_win_prob"] + p["draw_prob"] + p["away_win_prob"]
    assert abs(total - 1.0) < 0.02
    assert p["lambda_home"] >= 0 and p["lambda_away"] >= 0
    assert len(p["top_scorelines"]) == 3
    for s in p["top_scorelines"]:
        assert "-" in s["score"]
        assert 0.0 <= s["prob"] <= 1.0
    assert result["model_versions"]["ensemble"] == "v1"
