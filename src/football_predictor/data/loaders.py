"""Loaders for raw data files.

Every loader:
- returns ``None`` when an *optional* file is missing (logging a warning),
- raises ``FileNotFoundError`` only for the *required* results file,
- validates required columns via :mod:`schemas`,
- leaves heavy cleaning to :mod:`cleaning`.
"""

from __future__ import annotations

import os

import pandas as pd

from ..utils.config import Config
from ..utils.logging import get_logger
from . import schemas

logger = get_logger(__name__)


def _read_csv(path: str | os.PathLike[str]) -> pd.DataFrame:
    return pd.read_csv(path)


def _validate(df: pd.DataFrame, schema: schemas.Schema, path: str) -> None:
    missing = schema.validate(list(df.columns))
    if missing:
        raise ValueError(
            f"{schema.name} file {path!r} is missing required columns: {missing}"
        )


def load_results(path: str) -> pd.DataFrame:
    """Load the required ``results.csv``. Raises if absent."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required results file not found: {path}. "
            "Download it (scripts/download_data.py) or place it manually."
        )
    df = _read_csv(path)
    _validate(df, schemas.RESULTS, path)
    logger.info("Loaded results.csv: %d rows.", len(df))
    return df


def load_optional_csv(
    path: str,
    schema: schemas.Schema | None = None,
    label: str | None = None,
) -> pd.DataFrame | None:
    """Generic optional loader: returns None if the file is missing."""
    label = label or (schema.name if schema else os.path.basename(path))
    if not os.path.exists(path):
        logger.warning("Optional %s file not found at %s; continuing without it.", label, path)
        return None
    try:
        df = _read_csv(path)
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to read optional %s file %s: %s", label, path, exc)
        return None
    if schema is not None:
        missing = schema.validate(list(df.columns))
        if missing:
            logger.warning(
                "%s file %s missing columns %s; continuing with what is available.",
                label, path, missing,
            )
    logger.info("Loaded optional %s: %d rows.", label, len(df))
    return df


def load_shootouts(path: str) -> pd.DataFrame | None:
    return load_optional_csv(path, schemas.SHOOTOUTS, "shootouts")


def load_goalscorers(path: str) -> pd.DataFrame | None:
    return load_optional_csv(path, schemas.GOALSCORERS, "goalscorers")


def load_former_names(path: str) -> pd.DataFrame | None:
    return load_optional_csv(path, schemas.FORMER_NAMES, "former_names")


def load_fifa_rankings(path: str) -> pd.DataFrame | None:
    return load_optional_csv(path, schemas.FIFA_RANKINGS, "fifa_rankings")


def load_odds(path: str) -> pd.DataFrame | None:
    return load_optional_csv(path, schemas.ODDS, "odds")


def load_player_squad_features(path: str) -> pd.DataFrame | None:
    return load_optional_csv(path, schemas.PLAYER_SQUAD, "player_squad_features")


def load_advanced_stats(path: str) -> pd.DataFrame | None:
    return load_optional_csv(path, schemas.ADVANCED_STATS, "team_advanced_stats")


def load_all_raw(cfg: Config) -> dict[str, pd.DataFrame | None]:
    """Load every supported raw source according to config.

    Returns a dict keyed by source name. Missing optional sources are ``None``.
    """
    data = cfg.data
    return {
        "results": load_results(data.results_file),
        "shootouts": load_shootouts(data.shootouts_file),
        "goalscorers": load_goalscorers(data.goalscorers_file),
        "former_names": load_former_names(data.former_names_file),
        "fifa_rankings": load_fifa_rankings(data.fifa_rankings_file),
        "odds": load_odds(data.odds_file),
        "player_squad_features": load_player_squad_features(data.player_squad_features_file),
        "advanced_stats": load_advanced_stats(data.advanced_stats_file),
    }


def load_processed(path: str) -> pd.DataFrame:
    """Load the processed feature table (parquet preferred, csv fallback)."""
    if path.endswith(".parquet") and os.path.exists(path):
        return pd.read_parquet(path)
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["date"])
    # Try the sibling format.
    alt = path.replace(".parquet", ".csv") if path.endswith(".parquet") else path.replace(".csv", ".parquet")
    if os.path.exists(alt):
        return load_processed(alt)
    raise FileNotFoundError(
        f"Processed features not found at {path}. Run scripts/build_dataset.py first."
    )
