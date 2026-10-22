"""Team-name normalization.

International datasets disagree on team names (``USA`` vs ``United States``,
``Korea Republic`` vs ``South Korea``) and teams have been renamed over time
(``former_names.csv``). To join datasets correctly we map every raw name to a
single canonical form.

Resolution order for a raw name:
1. Manual overrides (:data:`TEAM_NAME_OVERRIDES`).
2. ``former_names.csv`` mapping (former -> current), if loaded.
3. Light cleanup (strip / collapse whitespace).

Unmatched / surprising names are logged once so data issues surface early.
"""

from __future__ import annotations

import os
from functools import lru_cache

import pandas as pd

from .logging import get_logger

logger = get_logger(__name__)


# Manual overrides applied first and unconditionally. Keys are matched after a
# basic cleanup (strip + collapse whitespace), case-sensitively by default but we
# also register a lowercase index for robustness.
TEAM_NAME_OVERRIDES: dict[str, str] = {
    "USA": "United States",
    "USMNT": "United States",
    "United States of America": "United States",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "Czech Republic": "Czechia",
    "Ivory Coast": "Côte d'Ivoire",
    "Congo DR": "DR Congo",
    "Congo Kinshasa": "DR Congo",
    "Cape Verde": "Cabo Verde",
    "Cape Verde Islands": "Cabo Verde",
    "Turkey": "Türkiye",
    "Swaziland": "Eswatini",
    "Macedonia": "North Macedonia",
    "FYR Macedonia": "North Macedonia",
}


def _basic_clean(name: str) -> str:
    """Strip and collapse internal whitespace; leave casing untouched."""
    return " ".join(str(name).strip().split())


class NameNormalizer:
    """Stateful normalizer that can consult ``former_names.csv``.

    Parameters
    ----------
    former_names:
        Optional dataframe with at least ``former`` and ``current`` columns
        (the Kaggle file uses exactly these names).
    extra_overrides:
        Additional manual override mappings merged on top of the defaults.
    """

    def __init__(
        self,
        former_names: pd.DataFrame | None = None,
        extra_overrides: dict[str, str] | None = None,
    ) -> None:
        self._overrides: dict[str, str] = dict(TEAM_NAME_OVERRIDES)
        if extra_overrides:
            self._overrides.update(extra_overrides)
        # Lowercase index for case-insensitive override matching.
        self._overrides_lower = {k.lower(): v for k, v in self._overrides.items()}

        self._former_map: dict[str, str] = {}
        if former_names is not None and len(former_names):
            self._load_former_names(former_names)

        self._unmatched_logged: set[str] = set()

    def _load_former_names(self, df: pd.DataFrame) -> None:
        cols = {c.lower(): c for c in df.columns}
        former_col = cols.get("former")
        current_col = cols.get("current")
        if former_col is None or current_col is None:
            logger.warning(
                "former_names file missing 'former'/'current' columns; got %s",
                list(df.columns),
            )
            return
        for _, row in df[[former_col, current_col]].dropna().iterrows():
            self._former_map[_basic_clean(row[former_col])] = _basic_clean(
                row[current_col]
            )
        logger.info("Loaded %d former-name mappings.", len(self._former_map))

    def normalize(self, name: str | float | None) -> str | None:
        """Normalize a single team name to its canonical form."""
        if name is None or (isinstance(name, float) and pd.isna(name)):
            return None
        cleaned = _basic_clean(name)
        if not cleaned:
            return None

        # 1) Manual overrides (case-insensitive).
        if cleaned in self._overrides:
            return self._overrides[cleaned]
        if cleaned.lower() in self._overrides_lower:
            return self._overrides_lower[cleaned.lower()]

        # 2) Former -> current mapping. Apply transitively through overrides too.
        if cleaned in self._former_map:
            mapped = self._former_map[cleaned]
            return self._overrides.get(mapped, mapped)

        # 3) Fallback: cleaned name as-is.
        return cleaned

    def normalize_series(self, series: pd.Series) -> pd.Series:
        return series.map(self.normalize)

    def normalize_dataframe(
        self,
        df: pd.DataFrame,
        columns: list[str],
        known_teams: set[str] | None = None,
    ) -> pd.DataFrame:
        """Normalize a set of team-name columns in-place (returns a copy).

        If ``known_teams`` is provided, any normalized value not in the set is
        logged once as potentially unmatched (helps catch join failures).
        """
        out = df.copy()
        for col in columns:
            if col not in out.columns:
                continue
            out[col] = self.normalize_series(out[col])
            if known_teams is not None:
                self._log_unmatched(out[col], known_teams)
        return out

    def _log_unmatched(self, series: pd.Series, known_teams: set[str]) -> None:
        for value in series.dropna().unique():
            if value not in known_teams and value not in self._unmatched_logged:
                self._unmatched_logged.add(value)
                logger.debug("Unmatched team name (no canonical entry): %r", value)

    @property
    def unmatched(self) -> set[str]:
        return set(self._unmatched_logged)


def load_former_names(path: str | os.PathLike[str]) -> pd.DataFrame | None:
    """Load ``former_names.csv`` if it exists, else return None."""
    if not os.path.exists(path):
        logger.warning("former_names file not found at %s; skipping.", path)
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to read former_names file %s: %s", path, exc)
        return None


@lru_cache(maxsize=1)
def default_normalizer() -> NameNormalizer:
    """A normalizer with only manual overrides (no former-names file)."""
    return NameNormalizer()
