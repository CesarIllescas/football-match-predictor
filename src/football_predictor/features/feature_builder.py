"""Orchestrate the full pre-match feature pipeline.

Order matters:
1. base table (targets, competition, venue, shootouts)  -- from cleaning
2. Elo (provides opponent strength used by rolling)
3. rolling form
4. scorer features
5. FIFA rankings (as-of)
6. odds (as-of, gated)
7. player/squad (as-of, gated, mode-aware)
8. advanced stats (rolling)

Each optional stage is gated by config flags and degrades gracefully.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..data import cleaning
from ..utils.config import Config
from ..utils.logging import get_logger
from .advanced_stats import add_advanced_stats_features
from .elo import EloRatingSystem, add_elo_features
from .odds import add_odds_features
from .players import aggregate_player_features_asof
from .rankings import add_fifa_ranking_features
from .rolling import add_rolling_features
from .scorers import add_scorer_rolling_features

logger = get_logger(__name__)


@dataclass
class FeatureArtifacts:
    """Side outputs from feature building needed later (e.g. fitted Elo)."""
    elo_system: EloRatingSystem | None = None


def build_features(
    raw: dict[str, pd.DataFrame | None],
    cfg: Config,
    prediction_mode: str | None = None,
) -> tuple[pd.DataFrame, FeatureArtifacts]:
    """Build the full processed feature table from normalized raw sources.

    Parameters
    ----------
    raw:
        Normalized sources (see data.matching.normalize_all). ``results`` required.
    cfg:
        Loaded configuration.
    prediction_mode:
        Player-feature mode; defaults to ``cfg.prediction.default_mode``.
    """
    if raw.get("results") is None:
        raise ValueError("results is required to build features.")

    prediction_mode = prediction_mode or cfg.prediction.default_mode
    windows = cfg.features.rolling_windows
    artifacts = FeatureArtifacts()

    # 1) Base table.
    df = cleaning.build_base_table(raw["results"], raw.get("shootouts"))

    # 2) Elo.
    if cfg.features.use_custom_elo:
        df, elo_system = add_elo_features(df, cfg.elo.model_dump())
        artifacts.elo_system = elo_system

    # 3) Rolling form.
    df = add_rolling_features(df, windows)

    # 4) Scorer features.
    if cfg.features.use_scorer_features:
        df = add_scorer_rolling_features(df, raw.get("goalscorers"), windows)

    # 5) FIFA rankings.
    if cfg.features.use_fifa_rankings:
        df = add_fifa_ranking_features(df, raw.get("fifa_rankings"))

    # 6) Odds (gated).
    if cfg.features.use_odds:
        df = add_odds_features(df, raw.get("odds"))

    # 7) Player/squad (gated, mode-aware).
    if cfg.features.use_player_features:
        df = aggregate_player_features_asof(df, raw.get("player_squad_features"), prediction_mode)

    # 8) Advanced stats.
    if cfg.features.use_advanced_stats:
        df = add_advanced_stats_features(df, raw.get("advanced_stats"), windows)

    logger.info("Built feature table: %d rows x %d cols.", df.shape[0], df.shape[1])
    return df, artifacts
