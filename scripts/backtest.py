"""Walk-forward backtest: train up to year Y-1, evaluate on year Y.

    python scripts/backtest.py --config configs/base.yaml --start-year 2019 --end-year 2024
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import typer

from football_predictor.data import loaders
from football_predictor.evaluation.backtesting import walk_forward_backtest
from football_predictor.prediction.predictor import predict_proba_df, train_pipeline
from football_predictor.utils.config import load_config
from football_predictor.utils.logging import get_logger

logger = get_logger("backtest")
app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: str = typer.Option("configs/base.yaml", "--config"),
    start_year: int = typer.Option(2019, "--start-year"),
    end_year: int = typer.Option(2024, "--end-year"),
) -> None:
    cfg = load_config(config)
    target = cfg.model.target
    df = loaders.load_processed(cfg.data.processed_features_parquet)

    def train_fn(train_df):
        # No separate validation inside a fold: calibrate on a tail slice of train.
        cut = int(len(train_df) * 0.85)
        tr = train_df.iloc[:cut]
        va = train_df.iloc[cut:]
        return train_pipeline(tr, va, cfg, target=target)

    def predict_fn(state, test_df):
        return predict_proba_df(state, test_df)

    results = walk_forward_backtest(df, start_year, end_year, train_fn, predict_fn, target=target)
    if len(results):
        logger.info("\n%s", results.to_string(index=False))
        logger.info("Mean log_loss=%.4f | mean accuracy=%.4f | mean Brier=%.4f",
                    results["log_loss"].mean(), results["accuracy"].mean(),
                    results["multiclass_brier_score"].mean())
    else:
        logger.warning("No backtest folds produced (check date range and data).")


if __name__ == "__main__":
    app()
