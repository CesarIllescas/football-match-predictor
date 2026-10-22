"""Rolling recent-form features.

We explode the match table into a long *team-match* view (one row per team per
match, from that team's perspective), sort each team chronologically, and compute
rolling statistics with a **shift of 1** so the current match never contributes to
its own features. Results are merged back as ``home_*`` / ``away_*`` columns plus
``diff_*`` features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

logger = get_logger(__name__)


def _team_match_long(matches: pd.DataFrame) -> pd.DataFrame:
    """Build a long per-team view with goals, points, results and opponent Elo.

    Requires the cleaned match table (with goals + optional Elo pre columns).
    """
    needed = ["date", "home_team", "away_team", "home_goals", "away_goals"]
    for c in needed:
        if c not in matches.columns:
            raise KeyError(f"rolling features need column {c!r}")

    has_elo = "home_elo_pre" in matches.columns and "away_elo_pre" in matches.columns

    base = matches.reset_index(drop=False).rename(columns={"index": "match_id"})

    home = pd.DataFrame({
        "match_id": base["match_id"],
        "date": base["date"],
        "team": base["home_team"],
        "opponent": base["away_team"],
        "goals_for": base["home_goals"],
        "goals_against": base["away_goals"],
        "is_home": 1,
    })
    away = pd.DataFrame({
        "match_id": base["match_id"],
        "date": base["date"],
        "team": base["away_team"],
        "opponent": base["home_team"],
        "goals_for": base["away_goals"],
        "goals_against": base["home_goals"],
        "is_home": 0,
    })
    if has_elo:
        home["own_elo"] = base["home_elo_pre"]
        home["opponent_elo"] = base["away_elo_pre"]
        away["own_elo"] = base["away_elo_pre"]
        away["opponent_elo"] = base["home_elo_pre"]

    long = pd.concat([home, away], ignore_index=True)

    gd = long["goals_for"] - long["goals_against"]
    long["goal_diff"] = gd
    long["win"] = (gd > 0).astype(float)
    long["draw"] = (gd == 0).astype(float)
    long["loss"] = (gd < 0).astype(float)
    # Points unknown where score missing.
    pts = np.where(gd > 0, 3.0, np.where(gd == 0, 1.0, 0.0))
    long["points"] = np.where(long["goals_for"].notna() & long["goals_against"].notna(), pts, np.nan)
    long["clean_sheet"] = (long["goals_against"] == 0).astype(float)
    long["failed_to_score"] = (long["goals_for"] == 0).astype(float)

    # Stable chronological order; mergesort keeps ties stable.
    long = long.sort_values(["team", "date", "match_id"], kind="mergesort").reset_index(drop=True)
    return long, has_elo


def _rolling_team_features(long: pd.DataFrame, windows: list[int], has_elo: bool) -> pd.DataFrame:
    """Compute shifted rolling aggregates per team for each window."""
    g = long.groupby("team", sort=False)

    out = long[["match_id", "team", "is_home"]].copy()

    # shift(1) -> exclude current match from its own rolling window.
    def roll_sum(col: str, w: int) -> pd.Series:
        return g[col].apply(lambda s: s.shift(1).rolling(w, min_periods=1).sum()).reset_index(level=0, drop=True)

    def roll_mean(col: str, w: int) -> pd.Series:
        return g[col].apply(lambda s: s.shift(1).rolling(w, min_periods=1).mean()).reset_index(level=0, drop=True)

    for w in windows:
        out[f"points_last_{w}"] = roll_sum("points", w)
        out[f"wins_last_{w}"] = roll_sum("win", w)
        out[f"draws_last_{w}"] = roll_sum("draw", w)
        out[f"losses_last_{w}"] = roll_sum("loss", w)
        out[f"goals_for_last_{w}"] = roll_sum("goals_for", w)
        out[f"goals_against_last_{w}"] = roll_sum("goals_against", w)
        out[f"goal_diff_last_{w}"] = roll_sum("goal_diff", w)
        out[f"clean_sheets_last_{w}"] = roll_sum("clean_sheet", w)
        out[f"failed_to_score_last_{w}"] = roll_sum("failed_to_score", w)
        out[f"avg_goals_for_last_{w}"] = roll_mean("goals_for", w)
        out[f"avg_goals_against_last_{w}"] = roll_mean("goals_against", w)
        out[f"avg_points_last_{w}"] = roll_mean("points", w)
        out[f"avg_goal_diff_last_{w}"] = roll_mean("goal_diff", w)
        if has_elo:
            out[f"avg_opponent_elo_last_{w}"] = roll_mean("opponent_elo", w)
            # Opponent-adjusted: points relative to opponent strength baseline.
            opp = out[f"avg_opponent_elo_last_{w}"]
            out[f"elo_adjusted_points_last_{w}"] = out[f"avg_points_last_{w}"] * (opp / 1500.0)
            out[f"elo_adjusted_goal_diff_last_{w}"] = out[f"avg_goal_diff_last_{w}"] * (opp / 1500.0)

    # Weighted recent form (linear recency weights over last 5).
    def weighted_last5(col: str) -> pd.Series:
        weights = np.array([1, 2, 3, 4, 5], dtype=float)

        def _w(s: pd.Series) -> pd.Series:
            shifted = s.shift(1)
            return shifted.rolling(5, min_periods=1).apply(
                lambda x: np.average(x, weights=weights[-len(x):]) if len(x) else np.nan,
                raw=True,
            )

        return g[col].apply(_w).reset_index(level=0, drop=True)

    out["weighted_points_last_5"] = weighted_last5("points")
    out["weighted_goal_diff_last_5"] = weighted_last5("goal_diff")

    return out


def add_rolling_features(
    matches: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Attach home/away rolling features and difference features to matches.

    Parameters
    ----------
    matches:
        Cleaned match table with goals (and ideally Elo pre columns for
        opponent-adjusted features).
    windows:
        Rolling windows, default ``[5, 10, 20]``.
    """
    windows = windows or [5, 10, 20]
    # Work on a positional 0..n-1 index so match_id aligns cleanly.
    out = matches.reset_index(drop=True).copy()
    long, has_elo = _team_match_long(out)
    feats = _rolling_team_features(long, windows, has_elo)

    feature_cols = [c for c in feats.columns if c not in ("match_id", "team", "is_home")]

    home_feats = feats[feats["is_home"] == 1].set_index("match_id")[feature_cols].add_prefix("home_")
    away_feats = feats[feats["is_home"] == 0].set_index("match_id")[feature_cols].add_prefix("away_")

    out = out.join(home_feats).join(away_feats)

    # Difference features (home - away) for every base feature, built in one block
    # to avoid DataFrame fragmentation from many sequential inserts.
    diffs = {
        f"diff_{col}": out[f"home_{col}"] - out[f"away_{col}"]
        for col in feature_cols
        if f"home_{col}" in out.columns and f"away_{col}" in out.columns
    }
    out = pd.concat([out, pd.DataFrame(diffs, index=out.index)], axis=1)

    logger.info("Added rolling features for windows %s (%d feature cols/team).",
                windows, len(feature_cols))
    return out
