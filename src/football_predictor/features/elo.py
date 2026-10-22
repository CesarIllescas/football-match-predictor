"""Custom Elo rating system for international football.

The Elo state is updated chronologically. For every match we record the *pre-match*
ratings (these are valid features) and only afterwards apply the update using the
final score (which is a target, not a feature).

K-factor scales with competition importance; an optional margin-of-victory
multiplier rewards bigger wins (capped to avoid runaway updates).
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import pandas as pd

from ..data.cleaning import categorize_tournament
from ..utils.logging import get_logger

logger = get_logger(__name__)


DEFAULT_K_FACTORS = {
    "friendly": 15.0,
    "nations_league": 20.0,
    "qualifier": 30.0,
    "continental": 40.0,
    "world_cup": 50.0,
    "other": 20.0,
}


class EloRatingSystem:
    """Chronological Elo engine.

    Parameters
    ----------
    initial_rating:
        Rating assigned to a team the first time it appears.
    home_advantage:
        Rating points added to the home side's effective rating on non-neutral
        grounds. Removed (or replaced by ``neutral_home_advantage``) at neutral venues.
    neutral_home_advantage:
        Home advantage applied at neutral venues (usually 0).
    k_factors:
        Mapping from tournament category to K. Defaults to :data:`DEFAULT_K_FACTORS`.
    use_margin_multiplier:
        If True, scale K by a log-based margin-of-victory factor.
    """

    def __init__(
        self,
        initial_rating: float = 1500,
        home_advantage: float = 60,
        neutral_home_advantage: float = 0,
        k_factors: dict | None = None,
        use_margin_multiplier: bool = True,
    ) -> None:
        self.initial_rating = float(initial_rating)
        self.home_advantage = float(home_advantage)
        self.neutral_home_advantage = float(neutral_home_advantage)
        self.k_factors = dict(DEFAULT_K_FACTORS)
        if k_factors:
            self.k_factors.update(k_factors)
        self.use_margin_multiplier = bool(use_margin_multiplier)
        self.ratings: dict[str, float] = defaultdict(lambda: self.initial_rating)

    # ------------------------------------------------------------------ #
    def get_rating(self, team: str) -> float:
        return self.ratings[team]

    @staticmethod
    def expected_score(rating_a: float, rating_b: float) -> float:
        """Expected score (win probability incl. half-draws) for A vs B."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def _k_for(self, tournament: str | None) -> float:
        category = categorize_tournament(tournament)
        return self.k_factors.get(category, self.k_factors.get("other", 20.0))

    def _margin_multiplier(self, goal_diff: int, elo_diff_effective: float) -> float:
        """Margin-of-victory multiplier (World Football Elo style, bounded)."""
        if not self.use_margin_multiplier:
            return 1.0
        gd = abs(int(goal_diff))
        if gd <= 1:
            return 1.0
        # Dampen for already-favored teams to avoid runaway ratings.
        return float(np.log(gd + 1.0) * (2.2 / (abs(elo_diff_effective) * 0.001 + 2.2)))

    def update_match(
        self,
        home_team: str,
        away_team: str,
        home_score: int,
        away_score: int,
        tournament: str | None,
        neutral: bool,
    ) -> dict:
        """Compute pre-match features and update ratings for one match.

        Returns a dict with pre-match Elo features. The update is applied as a
        side effect *after* the features are captured.
        """
        home_elo_pre = self.get_rating(home_team)
        away_elo_pre = self.get_rating(away_team)

        adv = self.neutral_home_advantage if neutral else self.home_advantage
        eff_home = home_elo_pre + adv
        elo_diff = home_elo_pre - away_elo_pre

        home_expected = self.expected_score(eff_home, away_elo_pre)
        away_expected = 1.0 - home_expected

        features = {
            "home_elo_pre": home_elo_pre,
            "away_elo_pre": away_elo_pre,
            "elo_diff": elo_diff,
            "home_elo_expected": home_expected,
            "away_elo_expected": away_expected,
        }

        # Apply update only when a real score is available.
        if pd.notna(home_score) and pd.notna(away_score):
            if home_score > away_score:
                home_actual = 1.0
            elif home_score == away_score:
                home_actual = 0.5
            else:
                home_actual = 0.0
            away_actual = 1.0 - home_actual

            k = self._k_for(tournament)
            mult = self._margin_multiplier(home_score - away_score, eff_home - away_elo_pre)
            delta = k * mult * (home_actual - home_expected)

            self.ratings[home_team] = home_elo_pre + delta
            self.ratings[away_team] = away_elo_pre - delta
            features["home_elo_post"] = self.ratings[home_team]
            features["away_elo_post"] = self.ratings[away_team]
        else:
            features["home_elo_post"] = home_elo_pre
            features["away_elo_post"] = away_elo_pre

        return features

    def process_matches(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """Process matches chronologically, returning per-match Elo features.

        The returned frame is aligned to ``matches_df`` order (sorted by date)
        and contains the pre-match Elo columns plus post-match Elo (post columns
        are leakage and excluded from model features by the schemas module).
        """
        df = matches_df.sort_values("date", kind="mergesort").reset_index()
        records = []
        for row in df.itertuples(index=False):
            neutral = bool(getattr(row, "neutral", False))
            feats = self.update_match(
                home_team=row.home_team,
                away_team=row.away_team,
                home_score=getattr(row, "home_score", np.nan),
                away_score=getattr(row, "away_score", np.nan),
                tournament=getattr(row, "tournament", None),
                neutral=neutral,
            )
            records.append(feats)

        feat_df = pd.DataFrame.from_records(records)
        feat_df["__orig_index"] = df["index"].to_numpy()
        feat_df = feat_df.set_index("__orig_index").sort_index()
        feat_df.index.name = None
        return feat_df

    # ------------------------------------------------------------------ #
    def to_config(self) -> dict:
        return {
            "initial_rating": self.initial_rating,
            "home_advantage": self.home_advantage,
            "neutral_home_advantage": self.neutral_home_advantage,
            "k_factors": self.k_factors,
            "use_margin_multiplier": self.use_margin_multiplier,
            "ratings": dict(self.ratings),
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_config(), fh, indent=2)
        logger.info("Saved Elo config + ratings to %s", path)

    @classmethod
    def load(cls, path: str) -> "EloRatingSystem":
        with open(path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        system = cls(
            initial_rating=cfg.get("initial_rating", 1500),
            home_advantage=cfg.get("home_advantage", 60),
            neutral_home_advantage=cfg.get("neutral_home_advantage", 0),
            k_factors=cfg.get("k_factors"),
            use_margin_multiplier=cfg.get("use_margin_multiplier", True),
        )
        for team, rating in cfg.get("ratings", {}).items():
            system.ratings[team] = float(rating)
        return system


def add_elo_features(
    matches: pd.DataFrame,
    elo_config: dict | None = None,
) -> tuple[pd.DataFrame, EloRatingSystem]:
    """Convenience wrapper: attach pre-match Elo features to a match table."""
    cfg = elo_config or {}
    system = EloRatingSystem(
        initial_rating=cfg.get("initial_rating", 1500),
        home_advantage=cfg.get("home_advantage", 60),
        neutral_home_advantage=cfg.get("neutral_home_advantage", 0),
        k_factors=cfg.get("k_factors"),
        use_margin_multiplier=cfg.get("margin_of_victory_multiplier", True),
    )
    feats = system.process_matches(matches)
    out = matches.copy()
    for col in feats.columns:
        out[col] = feats[col].to_numpy()
    return out, system
