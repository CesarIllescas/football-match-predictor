"""Optional advanced team-stats rolling features (xG, shots, corners, cards).

Each metric is rolled per team with a shift so the current match is excluded.
Merged as home/away plus a couple of difference features. Fully optional.
"""

from __future__ import annotations

import pandas as pd

from ..utils.dates import to_datetime
from ..utils.logging import get_logger

logger = get_logger(__name__)

_METRICS = (
    "xg_for", "xg_against", "shots_for", "shots_against",
    "shots_on_target_for", "shots_on_target_against",
    "corners_for", "corners_against", "cards_for", "cards_against",
)


def _rolling_team_stats(stats: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    s = stats.copy()
    s["date"] = to_datetime(s["date"])
    present = [m for m in _METRICS if m in s.columns]
    for m in present:
        s[m] = pd.to_numeric(s[m], errors="coerce")
    s = s.sort_values(["team", "date"], kind="mergesort").reset_index(drop=True)
    g = s.groupby("team", sort=False)

    out = s[["date", "team"]].copy()
    for w in windows:
        for m in present:
            out[f"{m}_last_{w}"] = (
                g[m].apply(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
                .reset_index(level=0, drop=True)
            )
    return out, present


def add_advanced_stats_features(
    matches: pd.DataFrame,
    advanced_stats: pd.DataFrame | None,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Attach rolling advanced-stat features. Returns matches unchanged if missing."""
    windows = windows or [5]
    out = matches.copy()
    if advanced_stats is None or len(advanced_stats) == 0:
        logger.warning("No advanced team stats; skipping advanced features.")
        return out

    rolled, present = _rolling_team_stats(advanced_stats, windows)
    feature_cols = [c for c in rolled.columns if c not in ("date", "team")]

    rename_home = {c: f"home_{c}" for c in feature_cols}
    rename_away = {c: f"away_{c}" for c in feature_cols}
    home = rolled.rename(columns={"team": "home_team", **rename_home})
    away = rolled.rename(columns={"team": "away_team", **rename_away})

    out["date"] = to_datetime(out["date"])
    out = out.merge(home, on=["date", "home_team"], how="left")
    out = out.merge(away, on=["date", "away_team"], how="left")

    # A couple of useful difference features.
    for w in windows:
        hc, ac = f"home_xg_for_last_{w}", f"away_xg_for_last_{w}"
        if hc in out.columns and ac in out.columns:
            out[f"diff_xg_for_last_{w}"] = out[hc] - out[ac]

    logger.info("Added advanced-stats rolling features (metrics=%s, windows=%s).", present, windows)
    return out
