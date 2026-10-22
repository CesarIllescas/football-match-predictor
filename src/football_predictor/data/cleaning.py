"""Cleaning and base-table construction.

Responsibilities:
- parse dates, drop invalid rows, sort chronologically,
- create match-level targets (the only place scores become targets),
- enrich with shootout / knockout-advancement info,
- derive competition category + importance,
- derive venue features.

Crucially, ``home_score``/``away_score`` are used ONLY to build targets. They are
flagged via :data:`schemas.TARGET_AND_LEAKAGE_COLUMNS` and excluded from features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.dates import to_datetime
from ..utils.logging import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Competition category + importance
# --------------------------------------------------------------------------- #
COMPETITION_IMPORTANCE = {
    "friendly": 1,
    "nations_league": 2,
    "qualifier": 3,
    "continental": 4,
    "world_cup": 5,
}


def categorize_tournament(tournament: str | float | None) -> str:
    """Map a raw tournament string to a coarse category.

    Categories: world_cup, qualifier, continental, nations_league, friendly, other.
    """
    if tournament is None or (isinstance(tournament, float) and pd.isna(tournament)):
        return "friendly"
    t = str(tournament).strip().lower()

    if "qualif" in t:
        return "qualifier"
    if "nations league" in t:
        return "nations_league"
    if "fifa world cup" in t or t == "world cup":
        return "world_cup"

    continental_markers = (
        "uefa euro", "european championship", "copa am", "copa america",
        "afc asian cup", "african cup of nations", "africa cup of nations",
        "concacaf", "gold cup", "oceania nations", "afc", "uefa",
        "confederations cup", "copa",
    )
    if any(m in t for m in continental_markers):
        return "continental"
    if "friendly" in t:
        return "friendly"
    return "other"


def add_competition_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["tournament_category"] = out["tournament"].map(categorize_tournament)
    out["competition_importance"] = (
        out["tournament_category"].map(COMPETITION_IMPORTANCE).fillna(1).astype(int)
    )
    return out


# --------------------------------------------------------------------------- #
# Venue features
# --------------------------------------------------------------------------- #
def add_venue_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive venue features.

    ``host_country`` defaults to the match ``country`` (where it was played).
    On neutral grounds, neither side is treated as host unless they happen to be
    the host country.
    """
    out = df.copy()
    if "neutral" not in out.columns:
        out["neutral"] = False
    out["neutral"] = out["neutral"].fillna(False).astype(bool)
    out["is_neutral"] = out["neutral"].astype(int)

    host = out["country"] if "country" in out.columns else pd.Series([None] * len(out), index=out.index)
    out["host_country"] = host

    out["home_advantage_flag"] = (~out["neutral"]).astype(int)
    out["home_team_is_host_country"] = (
        (out["home_team"] == out["host_country"]).fillna(False).astype(int)
    )
    out["away_team_is_host_country"] = (
        (out["away_team"] == out["host_country"]).fillna(False).astype(int)
    )
    return out


# --------------------------------------------------------------------------- #
# Base cleaning + targets
# --------------------------------------------------------------------------- #
def clean_results(results: pd.DataFrame) -> pd.DataFrame:
    """Parse dates, drop invalid rows, sort, and reset index."""
    df = results.copy()
    df["date"] = to_datetime(df["date"])

    before = len(df)
    df = df.dropna(subset=["date", "home_team", "away_team"])
    # Scores may be missing for future fixtures; keep them but they yield NaN targets.
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df[df["home_team"] != df["away_team"]]
    df = df.sort_values("date", kind="mergesort").reset_index(drop=True)
    logger.info("clean_results: kept %d / %d rows.", len(df), before)
    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Create all match-level targets from the (post-match) scores.

    Target encoding for ``result_1x2``: 0 = away win, 1 = draw, 2 = home win.
    Rows without scores get NaN targets (useful for future fixtures).
    """
    out = df.copy()
    hs = out["home_score"]
    as_ = out["away_score"]
    has_score = hs.notna() & as_.notna()

    out["home_goals"] = hs
    out["away_goals"] = as_
    out["total_goals"] = hs + as_
    out["goal_diff"] = hs - as_

    out["home_win"] = np.where(has_score, (hs > as_).astype("float"), np.nan)
    out["draw"] = np.where(has_score, (hs == as_).astype("float"), np.nan)
    out["away_win"] = np.where(has_score, (hs < as_).astype("float"), np.nan)

    result = np.full(len(out), np.nan)
    result[(has_score & (hs > as_)).to_numpy()] = 2  # home win
    result[(has_score & (hs == as_)).to_numpy()] = 1  # draw
    result[(has_score & (hs < as_)).to_numpy()] = 0  # away win
    out["result_1x2"] = result

    total = out["total_goals"]
    out["over_15"] = np.where(has_score, (total > 1.5).astype("float"), np.nan)
    out["over_25"] = np.where(has_score, (total > 2.5).astype("float"), np.nan)
    out["over_35"] = np.where(has_score, (total > 3.5).astype("float"), np.nan)
    out["btts"] = np.where(has_score, ((hs > 0) & (as_ > 0)).astype("float"), np.nan)
    return out


# --------------------------------------------------------------------------- #
# Shootout + knockout advancement enrichment
# --------------------------------------------------------------------------- #
def add_shootout_features(
    matches: pd.DataFrame,
    shootouts: pd.DataFrame | None,
) -> pd.DataFrame:
    """Merge shootout info and build advancement targets.

    Shootout goals are NOT added to the match score: for 90/120-minute outcome
    prediction we keep the ``results.csv`` score. Advancement is a *separate*
    target used for knockout progression modelling.
    """
    out = matches.copy()

    # Defaults so the columns always exist.
    out["shootout_played"] = 0
    out["shootout_winner"] = pd.NA
    out["home_won_shootout"] = 0
    out["away_won_shootout"] = 0
    out["first_shooter_home"] = 0
    out["first_shooter_away"] = 0

    if shootouts is not None and len(shootouts):
        s = shootouts.copy()
        s["date"] = to_datetime(s["date"])
        key = ["date", "home_team", "away_team"]
        s = s.drop_duplicates(subset=key)
        merged = out.merge(
            s[key + [c for c in ["winner", "first_shooter"] if c in s.columns]],
            on=key,
            how="left",
            suffixes=("", "_so"),
        )
        played = merged["winner"].notna()
        merged["shootout_played"] = played.astype(int)
        merged["shootout_winner"] = merged["winner"]
        merged["home_won_shootout"] = (merged["winner"] == merged["home_team"]).fillna(False).astype(int)
        merged["away_won_shootout"] = (merged["winner"] == merged["away_team"]).fillna(False).astype(int)
        if "first_shooter" in merged.columns:
            merged["first_shooter_home"] = (merged["first_shooter"] == merged["home_team"]).fillna(False).astype(int)
            merged["first_shooter_away"] = (merged["first_shooter"] == merged["away_team"]).fillna(False).astype(int)
            merged = merged.drop(columns=["first_shooter"])
        merged = merged.drop(columns=["winner"])
        out = merged
    else:
        logger.warning("No shootouts data; advancement falls back to score only.")

    # Advancement target: who progressed in a knockout sense.
    hs, as_ = out["home_goals"], out["away_goals"]
    has_score = hs.notna() & as_.notna()
    home_adv = np.where(out["shootout_played"] == 1,
                        out["home_won_shootout"] == 1,
                        has_score & (hs > as_))
    away_adv = np.where(out["shootout_played"] == 1,
                        out["away_won_shootout"] == 1,
                        has_score & (as_ > hs))
    out["home_advanced"] = pd.Series(home_adv, index=out.index).astype("float")
    out["away_advanced"] = pd.Series(away_adv, index=out.index).astype("float")
    # 1 = home advanced, 0 = away advanced; NaN where unknown (true draw, no SO).
    advance = np.where(out["home_advanced"] == 1, 1.0,
                       np.where(out["away_advanced"] == 1, 0.0, np.nan))
    out["advance_result"] = advance
    return out


def build_base_table(
    results: pd.DataFrame,
    shootouts: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run the full base-table construction pipeline (no rolling/Elo yet)."""
    df = clean_results(results)
    df = add_targets(df)
    df = add_competition_features(df)
    df = add_venue_features(df)
    df = add_shootout_features(df, shootouts)
    return df
