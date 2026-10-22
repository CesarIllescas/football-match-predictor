"""Aggregate raw player rows into team-level squad features.

The first model version never trains on raw player rows. Instead we collapse a
squad snapshot (as of a given date) into a single team-level feature vector.
"""

from __future__ import annotations

import pandas as pd

from ..utils.logging import get_logger

logger = get_logger(__name__)

TOP5_LEAGUE_COUNTRIES = {"England", "Spain", "Germany", "Italy", "France"}

# Boolean-ish flag columns that may exist in the player file.
_FLAG_COLS = (
    "is_goalkeeper", "is_defender", "is_midfielder", "is_forward",
    "is_likely_starter", "is_confirmed_starter", "is_injured", "is_suspended",
)
# Numeric stat columns that may exist.
_NUM_COLS = (
    "age", "market_value", "minutes_last_30_days", "minutes_last_90_days",
    "starts_last_30_days", "goals_last_30_days", "assists_last_30_days",
    "xg_last_30_days", "xa_last_30_days",
    "yellow_cards_last_90_days", "red_cards_last_90_days",
)


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "t"])


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in _FLAG_COLS:
        out[c] = _as_bool(out[c]) if c in out.columns else False
    for c in _NUM_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce") if c in out.columns else pd.NA
    if "club_country" not in out.columns:
        out["club_country"] = pd.NA
    return out


def aggregate_squad(snapshot: pd.DataFrame, include_lineup: bool) -> dict[str, float]:
    """Collapse one team's squad snapshot into a feature dict.

    Parameters
    ----------
    snapshot:
        Player rows for a single team as of a single date.
    include_lineup:
        Whether confirmed-lineup-derived features (starting XI) are allowed.
        Only True in ``lineup_known_prediction`` mode.
    """
    s = _coerce(snapshot)
    feats: dict[str, float] = {}

    mv = s["market_value"]
    feats["squad_total_market_value"] = float(mv.sum(skipna=True))
    feats["squad_avg_market_value"] = float(mv.mean(skipna=True)) if mv.notna().any() else 0.0
    feats["squad_median_market_value"] = float(mv.median(skipna=True)) if mv.notna().any() else 0.0
    feats["squad_avg_age"] = float(s["age"].mean(skipna=True)) if s["age"].notna().any() else 0.0
    feats["squad_num_players_top5_leagues"] = float(
        s["club_country"].isin(TOP5_LEAGUE_COUNTRIES).sum()
    )
    feats["squad_num_likely_starters"] = float(s["is_likely_starter"].sum())
    feats["squad_num_injured"] = float(s["is_injured"].sum())
    feats["squad_num_suspended"] = float(s["is_suspended"].sum())

    for col, name in [
        ("minutes_last_30_days", "squad_minutes_last_30_days"),
        ("minutes_last_90_days", "squad_minutes_last_90_days"),
        ("goals_last_30_days", "squad_goals_last_30_days"),
        ("assists_last_30_days", "squad_assists_last_30_days"),
        ("xg_last_30_days", "squad_xg_last_30_days"),
        ("xa_last_30_days", "squad_xa_last_30_days"),
    ]:
        feats[name] = float(s[col].sum(skipna=True))

    feats["goalkeeper_available"] = float(
        (s["is_goalkeeper"] & ~s["is_injured"] & ~s["is_suspended"]).any()
    )
    # "captain" and "top scorer" availability are heuristic flags if present.
    feats["top_scorer_available"] = float(
        (~s["is_injured"] & ~s["is_suspended"]).any()
    )

    if include_lineup:
        starters = s[s["is_confirmed_starter"]]
        feats["squad_num_confirmed_starters"] = float(len(starters))
        feats["starting_xi_total_market_value"] = float(starters["market_value"].sum(skipna=True))
        feats["starting_xi_avg_age"] = (
            float(starters["age"].mean(skipna=True)) if starters["age"].notna().any() else 0.0
        )
        feats["starting_xi_minutes_last_30_days"] = float(starters["minutes_last_30_days"].sum(skipna=True))
        feats["starting_xi_goals_last_30_days"] = float(starters["goals_last_30_days"].sum(skipna=True))
        feats["starting_xi_xg_last_30_days"] = float(starters["xg_last_30_days"].sum(skipna=True))

    return feats


# Names of the lineup-only features (forbidden in early_prediction mode).
LINEUP_ONLY_FEATURES = (
    "squad_num_confirmed_starters",
    "starting_xi_total_market_value",
    "starting_xi_avg_age",
    "starting_xi_minutes_last_30_days",
    "starting_xi_goals_last_30_days",
    "starting_xi_xg_last_30_days",
)
