"""FIFA ranking features via backward as-of merge.

For each match we attach the most recent ranking published *strictly before or on*
the match date. A ranking published after the match is never used.
"""

from __future__ import annotations

import pandas as pd

from ..utils.dates import to_datetime
from ..utils.logging import get_logger

logger = get_logger(__name__)


def _prepare_rankings(rankings: pd.DataFrame) -> pd.DataFrame:
    r = rankings.copy()
    r["rank_date"] = to_datetime(r["rank_date"])
    if "points" not in r.columns:
        r["points"] = pd.NA
    r = r.dropna(subset=["rank_date", "team"])
    r = r.sort_values("rank_date", kind="mergesort").reset_index(drop=True)
    return r[["rank_date", "team", "rank", "points"]]


def _asof_for_side(
    matches: pd.DataFrame,
    rankings: pd.DataFrame,
    team_col: str,
    prefix: str,
) -> pd.DataFrame:
    """As-of merge ranking for one side (home/away)."""
    left = matches[["__row", "date", team_col]].rename(columns={team_col: "team"})
    left = left.sort_values("date", kind="mergesort")

    merged = pd.merge_asof(
        left,
        rankings,
        left_on="date",
        right_on="rank_date",
        by="team",
        direction="backward",
        allow_exact_matches=True,
    )
    merged = merged.rename(columns={"rank": f"{prefix}_fifa_rank", "points": f"{prefix}_fifa_points"})
    return merged[["__row", f"{prefix}_fifa_rank", f"{prefix}_fifa_points"]]


def add_fifa_ranking_features(
    matches: pd.DataFrame,
    rankings: pd.DataFrame | None,
) -> pd.DataFrame:
    """Attach home/away FIFA rank + points and difference features.

    Returns ``matches`` unchanged (with a warning) if rankings are missing.
    """
    out = matches.copy()
    if rankings is None or len(rankings) == 0:
        logger.warning("No FIFA rankings; skipping ranking features.")
        return out

    r = _prepare_rankings(rankings)
    out["date"] = to_datetime(out["date"])
    out["__row"] = range(len(out))

    home = _asof_for_side(out, r, "home_team", "home")
    away = _asof_for_side(out, r, "away_team", "away")

    out = out.merge(home, on="__row", how="left").merge(away, on="__row", how="left")
    out = out.drop(columns="__row")

    out["fifa_rank_diff"] = out["home_fifa_rank"] - out["away_fifa_rank"]
    out["fifa_points_diff"] = out["home_fifa_points"] - out["away_fifa_points"]
    logger.info("Added FIFA ranking features.")
    return out
