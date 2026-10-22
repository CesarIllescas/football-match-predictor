"""Walk-forward backtesting utilities.

We never use a random split. Instead we train up to year ``Y-1`` and evaluate on
year ``Y``, walking forward across the requested range. The actual training and
prediction logic is injected as callables so this module stays decoupled from any
specific model implementation.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from ..utils.dates import walk_forward_folds
from ..utils.logging import get_logger
from .metrics import evaluate_1x2

logger = get_logger(__name__)

# train_fn(train_df) -> model_state
# predict_fn(model_state, test_df) -> proba array (n, 3)
TrainFn = Callable[[pd.DataFrame], object]
PredictFn = Callable[[object, pd.DataFrame], np.ndarray]


def walk_forward_backtest(
    df: pd.DataFrame,
    start_year: int,
    end_year: int,
    train_fn: TrainFn,
    predict_fn: PredictFn,
    target: str = "result_1x2",
    date_col: str = "date",
) -> pd.DataFrame:
    """Run a walk-forward backtest and return per-fold 1X2 metrics."""
    folds = walk_forward_folds(df, start_year, end_year, date_col=date_col)
    results = []
    for fold in folds:
        year = fold["test_year"]
        train_df = fold["train"]
        test_df = fold["test"][fold["test"][target].notna()]
        if len(test_df) == 0:
            continue
        logger.info("Backtest fold: train<%d (%d rows) -> test %d (%d rows)",
                    year, len(train_df), year, len(test_df))
        state = train_fn(train_df)
        proba = predict_fn(state, test_df)
        metrics = evaluate_1x2(test_df[target].to_numpy(), proba)
        metrics["test_year"] = year
        metrics["train_rows"] = len(train_df)
        results.append(metrics)
    out = pd.DataFrame(results)
    if len(out):
        logger.info("Backtest complete: mean log_loss=%.4f, mean accuracy=%.4f",
                    out["log_loss"].mean(), out["accuracy"].mean())
    return out
