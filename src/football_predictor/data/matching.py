"""Apply consistent team-name normalization across every dataset.

This is the single place where raw names from different sources are reconciled
to canonical names so that downstream joins (FIFA rankings, odds, scorers,
players) line up.
"""

from __future__ import annotations

import pandas as pd

from ..utils.logging import get_logger
from ..utils.names import NameNormalizer, load_former_names

logger = get_logger(__name__)

# Which team-name columns exist in each source.
TEAM_COLUMNS: dict[str, list[str]] = {
    "results": ["home_team", "away_team"],
    "shootouts": ["home_team", "away_team", "winner", "first_shooter"],
    "goalscorers": ["home_team", "away_team", "team"],
    "fifa_rankings": ["team"],
    "odds": ["home_team", "away_team"],
    "player_squad_features": ["team"],
    "advanced_stats": ["team"],
}


def build_normalizer(
    former_names: pd.DataFrame | None,
    extra_overrides: dict[str, str] | None = None,
) -> NameNormalizer:
    return NameNormalizer(former_names=former_names, extra_overrides=extra_overrides)


def normalize_all(
    raw: dict[str, pd.DataFrame | None],
    normalizer: NameNormalizer | None = None,
) -> dict[str, pd.DataFrame | None]:
    """Normalize team names across all loaded sources.

    Parameters
    ----------
    raw:
        Mapping like the one returned by :func:`loaders.load_all_raw`.
    normalizer:
        Optional pre-built normalizer. If not given, one is built from the
        ``former_names`` frame in ``raw``.
    """
    if normalizer is None:
        normalizer = build_normalizer(raw.get("former_names"))

    # Canonical universe of teams comes from results (the spine).
    known_teams: set[str] | None = None
    results = raw.get("results")
    if results is not None:
        h = normalizer.normalize_series(results["home_team"])
        a = normalizer.normalize_series(results["away_team"])
        known_teams = set(pd.concat([h, a]).dropna().unique())

    out: dict[str, pd.DataFrame | None] = {}
    for source, df in raw.items():
        if df is None or source == "former_names":
            out[source] = df
            continue
        cols = TEAM_COLUMNS.get(source, [])
        if cols:
            out[source] = normalizer.normalize_dataframe(df, cols, known_teams=known_teams)
        else:
            out[source] = df

    if normalizer.unmatched:
        logger.info("Total distinct unmatched team names observed: %d", len(normalizer.unmatched))
    return out


__all__ = ["build_normalizer", "normalize_all", "load_former_names", "TEAM_COLUMNS"]
