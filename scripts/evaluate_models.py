"""Evaluate trained models on the time-based test split.

    python scripts/evaluate_models.py --config configs/base.yaml

Reports 1X2 metrics, binary-market metrics (over 2.5, BTTS), expected-goals
error, confusion matrix, and year-by-year / competition breakdowns. Optionally
saves calibration plots when matplotlib is available.
"""

from __future__ import annotations

import json
import os

import _bootstrap  # noqa: F401
import numpy as np
import typer

from football_predictor.data import loaders
from football_predictor.evaluation.calibration_plots import plot_calibration
from football_predictor.evaluation.metrics import (
    confusion,
    evaluate_1x2,
    evaluate_binary,
    evaluate_expected_goals,
    performance_by_group,
)
from football_predictor.prediction.predictor import load_artifacts, predict_proba_df
from football_predictor.utils.config import load_config
from football_predictor.utils.dates import time_split, to_datetime
from football_predictor.utils.logging import get_logger

logger = get_logger("evaluate_models")
app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: str = typer.Option("configs/base.yaml", "--config"),
    plots_dir: str = typer.Option("models/artifacts/plots", "--plots-dir"),
) -> None:
    cfg = load_config(config)
    target = cfg.model.target

    df = loaders.load_processed(cfg.data.processed_features_parquet)
    artifacts = load_artifacts(cfg.model.save_dir, cfg)

    v = cfg.validation
    test = time_split(df, v.train_end_date, v.validation_start_date,
                      v.validation_end_date, v.test_start_date)["test"]
    test = test[test[target].notna()].reset_index(drop=True)
    if len(test) == 0:
        logger.warning("No labeled rows in the test period; nothing to evaluate.")
        return

    proba = predict_proba_df(artifacts, test)
    y = test[target].to_numpy(dtype=int)

    logger.info("=== 1X2 metrics (test) ===")
    logger.info(json.dumps(evaluate_1x2(y, proba), indent=2))

    logger.info("=== Confusion matrix [rows=true, cols=pred] order [away,draw,home] ===")
    logger.info("\n%s", confusion(y, proba))

    # Goal-based markets from the Poisson component.
    lh, la = artifacts.poisson.predict_lambdas(test)
    market_rows = [artifacts.poisson.predict_markets(float(lh[i]), float(la[i])) for i in range(len(test))]
    over25 = np.array([m["over_25_prob"] for m in market_rows])
    btts = np.array([m["btts_prob"] for m in market_rows])

    if "over_25" in test.columns and test["over_25"].notna().any():
        logger.info("=== Over 2.5 goals ===")
        logger.info(json.dumps(evaluate_binary(test["over_25"].to_numpy(), over25), indent=2))
    if "btts" in test.columns and test["btts"].notna().any():
        logger.info("=== BTTS ===")
        logger.info(json.dumps(evaluate_binary(test["btts"].to_numpy(), btts), indent=2))

    logger.info("=== Expected goals (home) ===")
    logger.info(json.dumps(evaluate_expected_goals(test["home_goals"].to_numpy(), lh), indent=2))
    logger.info("=== Expected goals (away) ===")
    logger.info(json.dumps(evaluate_expected_goals(test["away_goals"].to_numpy(), la), indent=2))

    # Year-by-year and competition breakdowns.
    test = test.copy()
    test["__year"] = to_datetime(test["date"]).dt.year
    logger.info("=== Year-by-year 1X2 ===")
    logger.info("\n%s", performance_by_group(test, y, proba, "__year").to_string(index=False))
    if "tournament_category" in test.columns:
        logger.info("=== Competition-category 1X2 ===")
        logger.info("\n%s", performance_by_group(test, y, proba, "tournament_category").to_string(index=False))

    # Favorite vs underdog.
    fav_home = proba[:, 2] >= proba[:, 0]
    logger.info("=== Favorite-side hit rate ===")
    pred = proba.argmax(axis=1)
    logger.info("Home favored share: %.3f", float(np.mean(fav_home)))
    logger.info("Overall accuracy: %.3f", float(np.mean(pred == y)))

    # Calibration plots (optional).
    os.makedirs(plots_dir, exist_ok=True)
    home_y = (y == 2).astype(int)
    if plot_calibration(home_y, proba[:, 2], os.path.join(plots_dir, "calibration_home_win.png"), "Home win calibration"):
        logger.info("Saved calibration plot to %s", plots_dir)
    else:
        logger.info("matplotlib not available; skipped calibration plots.")


if __name__ == "__main__":
    app()
