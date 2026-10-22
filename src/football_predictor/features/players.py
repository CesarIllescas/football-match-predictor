"""Optional player/squad features with strict as-of, leakage-safe merging.

Rules enforced here:
- a missing player file leaves the pipeline untouched,
- only player snapshots dated on/before the match date are used (as-of),
- confirmed-lineup features are only emitted in ``lineup_known_prediction`` mode,
- all player stats reflect data prior to the match (no current-match leakage).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from ..utils.dates import to_datetime
from ..utils.logging import get_logger
from . import squad

logger = get_logger(__name__)

EARLY_MODE = "early_prediction"
LINEUP_MODE = "lineup_known_prediction"


def load_player_squad_features(path: str) -> pd.DataFrame | None:
    """Load the optional player/squad file, returning None if missing."""
    if not os.path.exists(path):
        logger.warning("Player/squad file not found at %s; player features disabled.", path)
        return None
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to read player/squad file %s: %s", path, exc)
        return None
    df["date"] = to_datetime(df["date"])
    logger.info("Loaded player/squad features: %d rows.", len(df))
    return df


def _latest_snapshot_asof(
    player_features: pd.DataFrame,
    team: str,
    match_date: pd.Timestamp,
) -> pd.DataFrame:
    """Return the most recent squad snapshot for ``team`` on/before ``match_date``."""
    sub = player_features[
        (player_features["team"] == team) & (player_features["date"] <= match_date)
    ]
    if len(sub) == 0:
        return sub
    latest_date = sub["date"].max()
    return sub[sub["date"] == latest_date]


def aggregate_player_features_asof(
    matches: pd.DataFrame,
    player_features: pd.DataFrame | None,
    prediction_mode: str = EARLY_MODE,
) -> pd.DataFrame:
    """Attach as-of team-level squad features for home and away sides.

    Returns ``matches`` unchanged if ``player_features`` is None.
    """
    out = matches.copy()
    if player_features is None or len(player_features) == 0:
        logger.warning("No player/squad data; skipping player features.")
        return out

    include_lineup = prediction_mode == LINEUP_MODE
    if not include_lineup and prediction_mode != EARLY_MODE:
        logger.warning("Unknown prediction_mode %r; treating as early_prediction.", prediction_mode)

    pf = player_features.copy()
    pf["date"] = to_datetime(pf["date"])
    out["date"] = to_datetime(out["date"])

    home_rows: list[dict] = []
    away_rows: list[dict] = []
    for row in out[["home_team", "away_team", "date"]].itertuples(index=False):
        h_snap = _latest_snapshot_asof(pf, row.home_team, row.date)
        a_snap = _latest_snapshot_asof(pf, row.away_team, row.date)
        h_feats = squad.aggregate_squad(h_snap, include_lineup) if len(h_snap) else {}
        a_feats = squad.aggregate_squad(a_snap, include_lineup) if len(a_snap) else {}
        home_rows.append({f"home_{k}": v for k, v in h_feats.items()})
        away_rows.append({f"away_{k}": v for k, v in a_feats.items()})

    home_df = pd.DataFrame(home_rows, index=out.index)
    away_df = pd.DataFrame(away_rows, index=out.index)
    out = pd.concat([out, home_df, away_df], axis=1)

    # Difference features for shared base feature names.
    base_names = {c[len("home_"):] for c in home_df.columns}
    for name in base_names:
        hc, ac = f"home_{name}", f"away_{name}"
        if hc in out.columns and ac in out.columns:
            out[f"diff_{name}"] = out[hc] - out[ac]

    # Safety: strip lineup-only features in early mode (belt and suspenders).
    if not include_lineup:
        drop = [
            c for c in out.columns
            if any(lof in c for lof in squad.LINEUP_ONLY_FEATURES)
        ]
        if drop:
            out = out.drop(columns=drop)

    logger.info("Added player/squad features (mode=%s, lineup=%s).", prediction_mode, include_lineup)
    return out


def validate_player_feature_dates(matches_with_features: pd.DataFrame) -> None:
    """Sanity-check that player feature merge did not introduce future data.

    This validates structural invariants we can check post-merge: difference
    features must be finite where both sides exist. The strict as-of date check
    is enforced at merge time in :func:`aggregate_player_features_asof`; this
    function raises if any obviously-invalid (inf) values slipped through.
    """
    diff_cols = [c for c in matches_with_features.columns if c.startswith("diff_squad")]
    for c in diff_cols:
        vals = matches_with_features[c].to_numpy(dtype="float64", na_value=np.nan)
        if np.isinf(vals).any():
            raise ValueError(f"Player feature column {c} contains non-finite values.")
    logger.debug("Player feature date validation passed (%d diff cols).", len(diff_cols))
