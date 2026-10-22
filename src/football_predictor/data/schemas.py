"""Expected column schemas for every supported raw data source.

These are intentionally lightweight: we describe required and optional columns
and provide a validation helper that raises only when *required* columns are
missing. Extra columns are always allowed and preserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Schema:
    name: str
    required: tuple[str, ...]
    optional: tuple[str, ...] = field(default_factory=tuple)

    @property
    def all_columns(self) -> tuple[str, ...]:
        return self.required + self.optional

    def validate(self, columns: list[str]) -> list[str]:
        """Return the list of missing required columns (empty == valid)."""
        cols = set(columns)
        return [c for c in self.required if c not in cols]


RESULTS = Schema(
    name="results",
    required=("date", "home_team", "away_team", "home_score", "away_score", "tournament"),
    optional=("city", "country", "neutral"),
)

SHOOTOUTS = Schema(
    name="shootouts",
    required=("date", "home_team", "away_team", "winner"),
    optional=("first_shooter",),
)

GOALSCORERS = Schema(
    name="goalscorers",
    required=("date", "home_team", "away_team", "team", "scorer"),
    optional=("minute", "own_goal", "penalty"),
)

FORMER_NAMES = Schema(
    name="former_names",
    required=("current", "former"),
    optional=("start_date", "end_date"),
)

FIFA_RANKINGS = Schema(
    name="fifa_rankings",
    required=("rank_date", "team", "rank"),
    optional=("points",),
)

ODDS = Schema(
    name="odds",
    required=("date", "home_team", "away_team", "home_odds", "draw_odds", "away_odds"),
    optional=("bookmaker", "timestamp"),
)

PLAYER_SQUAD = Schema(
    name="player_squad_features",
    required=("date", "team", "player"),
    optional=(
        "position", "club", "club_country", "age", "market_value",
        "is_goalkeeper", "is_defender", "is_midfielder", "is_forward",
        "is_likely_starter", "is_confirmed_starter", "is_injured", "is_suspended",
        "minutes_last_30_days", "minutes_last_90_days", "starts_last_30_days",
        "goals_last_30_days", "assists_last_30_days", "xg_last_30_days",
        "xa_last_30_days", "yellow_cards_last_90_days", "red_cards_last_90_days",
    ),
)

ADVANCED_STATS = Schema(
    name="team_advanced_stats",
    required=("date", "team"),
    optional=(
        "xg_for", "xg_against", "shots_for", "shots_against",
        "shots_on_target_for", "shots_on_target_against",
        "corners_for", "corners_against", "cards_for", "cards_against",
    ),
)


# Columns that are post-match outcomes or targets. NEVER usable as model
# features. Centralized here so the tabular model and leakage tests share one
# source of truth.
TARGET_AND_LEAKAGE_COLUMNS: tuple[str, ...] = (
    "home_score", "away_score",
    "home_goals", "away_goals",
    "total_goals", "goal_diff",
    "home_win", "draw", "away_win",
    "result_1x2",
    "over_15", "over_25", "over_35",
    "btts",
    "post_match_elo",
    "home_elo_post", "away_elo_post",
    "advance_result", "home_advanced", "away_advanced",
    "shootout_played", "shootout_winner",
    "home_won_shootout", "away_won_shootout",
)

# Identifier / non-feature columns that should also be dropped before training.
IDENTIFIER_COLUMNS: tuple[str, ...] = (
    "date", "home_team", "away_team", "city", "country", "host_country",
    "scorer", "winner", "first_shooter",
)
