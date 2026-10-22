"""Scorer-derived rolling features from ``goalscorers.csv``.

This file only records *who scored* (plus own-goal / penalty flags). It is not a
full player dataset. We aggregate it to per-team, per-match scorer statistics and
then roll them forward (shifted) so a match never sees its own goals.

Definitions
-----------
open_play_goals    : goals where ``penalty == False`` and ``own_goal == False``
goal_concentration : top_scorer_goals_last_N / total_team_goals_last_N
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.dates import to_datetime
from ..utils.logging import get_logger

logger = get_logger(__name__)


def load_goalscorers(path: str) -> pd.DataFrame:
    """Load goalscorers.csv and coerce flag columns to clean booleans."""
    df = pd.read_csv(path)
    return _coerce_goalscorers(df)


def _coerce_goalscorers(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = to_datetime(out["date"])
    for flag in ("own_goal", "penalty"):
        if flag in out.columns:
            out[flag] = _as_bool(out[flag])
        else:
            out[flag] = False
    if "scorer" not in out.columns:
        out["scorer"] = pd.NA
    return out


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return (
        series.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "t"])
    )


def build_team_scorer_history(goalscorers: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw goal events into one row per (team, match-date).

    A goal is credited to ``team`` (the scoring team). Own goals count *against*
    the conceding side in the source, but for "goals for" purposes we credit the
    team listed in the ``team`` column as recorded.

    Returns columns:
        date, team, match_home, match_away, total_goals, penalty_goals,
        open_play_goals, own_goals, num_unique_scorers, top_scorer_goals.
    """
    df = _coerce_goalscorers(goalscorers) if "date" in goalscorers.columns else goalscorers
    df = df.dropna(subset=["date", "team"])

    grp_keys = ["date", "home_team", "away_team", "team"]
    records = []
    for keys, sub in df.groupby(grp_keys, sort=False):
        date, home, away, team = keys
        total = len(sub)
        penalty_goals = int(sub["penalty"].sum())
        own_goals = int(sub["own_goal"].sum())
        open_play = int(((~sub["penalty"]) & (~sub["own_goal"])).sum())
        # Top scorer concentration uses named, non-own-goal scorers.
        named = sub.loc[~sub["own_goal"], "scorer"].dropna()
        unique_scorers = named.nunique()
        top_scorer_goals = int(named.value_counts().iloc[0]) if len(named) else 0
        records.append({
            "date": date,
            "team": team,
            "match_home": home,
            "match_away": away,
            "total_goals": total,
            "penalty_goals": penalty_goals,
            "open_play_goals": open_play,
            "own_goals": own_goals,
            "num_unique_scorers": unique_scorers,
            "top_scorer_goals": top_scorer_goals,
        })
    hist = pd.DataFrame.from_records(records)
    if len(hist):
        hist = hist.sort_values(["team", "date"], kind="mergesort").reset_index(drop=True)
    return hist


def _rolling_scorer_stats(hist: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Per-team shifted rolling scorer stats (one row per team-match)."""
    g = hist.groupby("team", sort=False)
    out = hist[["date", "team"]].copy()

    def roll_sum(col: str, w: int) -> pd.Series:
        return g[col].apply(lambda s: s.shift(1).rolling(w, min_periods=1).sum()).reset_index(level=0, drop=True)

    def roll_mean(col: str, w: int) -> pd.Series:
        return g[col].apply(lambda s: s.shift(1).rolling(w, min_periods=1).mean()).reset_index(level=0, drop=True)

    for w in windows:
        # Average unique scorers per match over the window.
        out[f"num_unique_scorers_last_{w}"] = roll_mean("num_unique_scorers", w)
        out[f"penalty_goals_last_{w}"] = roll_sum("penalty_goals", w)
        out[f"open_play_goals_last_{w}"] = roll_sum("open_play_goals", w)
        out[f"own_goals_for_last_{w}"] = roll_sum("own_goals", w)
        out[f"top_scorer_goals_last_{w}"] = roll_sum("top_scorer_goals", w)
        total_goals_w = roll_sum("total_goals", w)
        out[f"total_goals_last_{w}"] = total_goals_w
        with np.errstate(divide="ignore", invalid="ignore"):
            conc = out[f"top_scorer_goals_last_{w}"] / total_goals_w
        out[f"goal_concentration_last_{w}"] = conc.replace([np.inf, -np.inf], np.nan)

    return out


def add_scorer_rolling_features(
    matches: pd.DataFrame,
    goalscorers: pd.DataFrame | None,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Attach home/away scorer-derived rolling features.

    If ``goalscorers`` is None/empty, returns ``matches`` unchanged (with a warning).
    All features are computed strictly from matches before the current one.
    """
    windows = windows or [5, 10, 20]
    if goalscorers is None or len(goalscorers) == 0:
        logger.warning("No goalscorers data; skipping scorer features.")
        return matches.copy()

    hist = build_team_scorer_history(goalscorers)
    if len(hist) == 0:
        logger.warning("Goalscorers produced no history; skipping scorer features.")
        return matches.copy()

    stats = _rolling_scorer_stats(hist, windows)
    feature_cols = [c for c in stats.columns if c not in ("date", "team")]

    out = matches.copy()
    out["date"] = to_datetime(out["date"])

    # Merge on (date, team) for home and away separately, renaming only features.
    rename_home = {c: f"home_{c}" for c in feature_cols}
    rename_away = {c: f"away_{c}" for c in feature_cols}
    home = stats.rename(columns={"team": "home_team", **rename_home})
    away = stats.rename(columns={"team": "away_team", **rename_away})

    out = out.merge(home, on=["date", "home_team"], how="left")
    out = out.merge(away, on=["date", "away_team"], how="left")

    # Difference features.
    for col in feature_cols:
        hc, ac = f"home_{col}", f"away_{col}"
        if hc in out.columns and ac in out.columns:
            out[f"diff_{col}"] = out[hc] - out[ac]

    logger.info("Added scorer rolling features for windows %s.", windows)
    return out
