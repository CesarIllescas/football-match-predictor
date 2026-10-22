"""Build processed features and train model artifacts."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd

from ..data import loaders, matching
from ..evaluation.metrics import evaluate_1x2
from ..features.elo import EloRatingSystem
from ..features.feature_builder import build_features
from ..prediction.predictor import predict_proba_df, save_artifacts, train_pipeline
from ..utils.config import Config
from ..utils.dates import time_split
from ..utils.logging import get_logger

logger = get_logger(__name__)


def build_processed_dataset(cfg: Config) -> int:
    """Load raw data, build features, write parquet/csv. Returns row count."""
    raw = loaders.load_all_raw(cfg)
    raw = matching.normalize_all(raw)
    df, _ = build_features(raw, cfg)

    os.makedirs(cfg.data.processed_dir, exist_ok=True)
    parquet_path = cfg.data.processed_features_parquet
    csv_path = cfg.data.processed_features_csv

    try:
        df.to_parquet(parquet_path, index=False)
    except Exception as exc:
        logger.warning("Could not write parquet (%s); CSV will still be written.", exc)

    df.to_csv(csv_path, index=False)
    logger.info("Built processed dataset: %d rows.", len(df))
    return len(df)


def train_models_from_processed(cfg: Config) -> dict[str, Any]:
    """Train on processed parquet and save artifacts. Returns summary metrics."""
    target = cfg.model.target
    df = loaders.load_processed(cfg.data.processed_features_parquet)

    v = cfg.validation
    splits = time_split(
        df,
        v.train_end_date,
        v.validation_start_date,
        v.validation_end_date,
        v.test_start_date,
    )
    train, val, test = splits["train"], splits["validation"], splits["test"]

    artifacts = train_pipeline(train, val, cfg, target=target)

    elo_system = EloRatingSystem(
        initial_rating=cfg.elo.initial_rating,
        home_advantage=cfg.elo.home_advantage,
        neutral_home_advantage=cfg.elo.neutral_home_advantage,
        k_factors=None,
        use_margin_multiplier=cfg.elo.margin_of_victory_multiplier,
    )
    elo_system.process_matches(df)
    save_artifacts(artifacts, cfg.model.save_dir, elo_system=elo_system)

    metrics: dict[str, Any] = {
        "train_rows": len(train),
        "validation_rows": len(val),
        "test_rows": len(test),
        "tabular_backend": artifacts.tabular.backend,
        "splits": {},
    }
    for name, part in (("train", train), ("validation", val), ("test", test)):
        labeled = part[part[target].notna()]
        if len(labeled) == 0:
            continue
        proba = predict_proba_df(artifacts, labeled)
        metrics["splits"][name] = evaluate_1x2(labeled[target].to_numpy(), proba)

    return metrics


def run_full_retrain(cfg: Config) -> dict[str, Any]:
    """Build dataset then train. Returns combined summary."""
    rows = build_processed_dataset(cfg)
    train_summary = train_models_from_processed(cfg)
    return {"processed_rows": rows, **train_summary}
