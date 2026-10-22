"""Market odds features.

Decimal odds are converted to overround-normalized implied probabilities. Odds
are strictly optional and gated behind ``features.use_odds``. When a prediction
timestamp is supplied, only odds observed at/before that timestamp are used so we
never leak closing odds into an "early" prediction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.dates import to_datetime
from ..utils.logging import get_logger

logger = get_logger(__name__)


def implied_probabilities(
    home_odds: float, draw_odds: float, away_odds: float
) -> dict[str, float]:
    """Convert decimal 1X2 odds to overround-normalized probabilities."""
    raw_h = 1.0 / home_odds
    raw_d = 1.0 / draw_odds
    raw_a = 1.0 / away_odds
    overround = raw_h + raw_d + raw_a
    return {
        "market_home_prob": raw_h / overround,
        "market_draw_prob": raw_d / overround,
        "market_away_prob": raw_a / overround,
        "market_overround": overround,
    }


def _prepare_odds(odds: pd.DataFrame, prediction_timestamp: pd.Timestamp | None) -> pd.DataFrame:
    o = odds.copy()
    o["date"] = to_datetime(o["date"])
    if "timestamp" in o.columns:
        o["timestamp"] = pd.to_datetime(o["timestamp"], errors="coerce")
    else:
        o["timestamp"] = o["date"]

    if prediction_timestamp is not None:
        before = len(o)
        o = o[o["timestamp"] <= prediction_timestamp]
        logger.info("Filtered odds to <= prediction timestamp: %d / %d kept.", len(o), before)

    # Keep the latest available quote per match (closest to allowed cutoff).
    o = o.sort_values("timestamp", kind="mergesort")
    o = o.drop_duplicates(subset=["date", "home_team", "away_team"], keep="last")

    probs = o.apply(
        lambda r: implied_probabilities(r["home_odds"], r["draw_odds"], r["away_odds"]),
        axis=1,
        result_type="expand",
    )
    out = pd.concat([o[["date", "home_team", "away_team"]].reset_index(drop=True),
                     probs.reset_index(drop=True)], axis=1)
    return out


def add_odds_features(
    matches: pd.DataFrame,
    odds: pd.DataFrame | None,
    prediction_timestamp: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Attach market implied-probability features.

    Returns ``matches`` unchanged (with a warning) when odds are missing.
    """
    out = matches.copy()
    if odds is None or len(odds) == 0:
        logger.warning("No odds data; skipping odds features.")
        return out

    prepared = _prepare_odds(odds, prediction_timestamp)
    out["date"] = to_datetime(out["date"])
    out = out.merge(prepared, on=["date", "home_team", "away_team"], how="left")
    logger.info("Added market odds features.")
    return out
