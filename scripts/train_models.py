"""Train all models on the processed dataset using a strict time-based split.

    python scripts/train_models.py --config configs/base.yaml

Artifacts written to ``models/artifacts``:
    elo_config.json, elo_baseline.json, poisson_model.joblib,
    tabular_model.joblib, calibrator.joblib, ensemble_config.json,
    feature_columns.json, model_versions.json
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import typer

from football_predictor.data import loaders
from football_predictor.evaluation.metrics import evaluate_1x2
from football_predictor.features.elo import EloRatingSystem
from football_predictor.prediction.predictor import (
    predict_proba_df,
    save_artifacts,
    train_pipeline,
)
from football_predictor.utils.config import load_config
from football_predictor.utils.dates import time_split
from football_predictor.utils.logging import get_logger

logger = get_logger("train_models")
app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: str = typer.Option("configs/base.yaml", "--config"),
) -> None:
    cfg = load_config(config)
    target = cfg.model.target

    logger.info("Loading processed features ...")
    df = loaders.load_processed(cfg.data.processed_features_parquet)

    v = cfg.validation
    splits = time_split(df, v.train_end_date, v.validation_start_date,
                        v.validation_end_date, v.test_start_date)
    train, val, test = splits["train"], splits["validation"], splits["test"]
    logger.info("Split sizes: train=%d val=%d test=%d", len(train), len(val), len(test))

    logger.info("Training pipeline ...")
    artifacts = train_pipeline(train, val, cfg, target=target)

    # Rebuild full Elo state (ratings) from the processed data for persistence.
    elo_system = EloRatingSystem(
        initial_rating=cfg.elo.initial_rating,
        home_advantage=cfg.elo.home_advantage,
        neutral_home_advantage=cfg.elo.neutral_home_advantage,
        k_factors=None,
        use_margin_multiplier=cfg.elo.margin_of_victory_multiplier,
    )
    elo_system.process_matches(df)

    save_artifacts(artifacts, cfg.model.save_dir, elo_system=elo_system)

    # Evaluation on each split.
    for name, part in (("train", train), ("validation", val), ("test", test)):
        labeled = part[part[target].notna()]
        if len(labeled) == 0:
            continue
        proba = predict_proba_df(artifacts, labeled)
        metrics = evaluate_1x2(labeled[target].to_numpy(), proba)
        logger.info("[%s] %s", name, {k: round(v, 4) if isinstance(v, float) else v
                                       for k, v in metrics.items()})


if __name__ == "__main__":
    app()
