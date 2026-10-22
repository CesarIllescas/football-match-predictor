"""Configuration loading and validation.

Configuration is stored in YAML (``configs/*.yaml``). We parse it into typed
pydantic models so the rest of the codebase gets attribute access, defaults and
validation for free. Unknown keys are preserved in ``Config.extra`` so optional
config files (e.g. ``model.yaml``) can be merged without losing information.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

try:  # python-dotenv is optional at import time
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None  # type: ignore


# --------------------------------------------------------------------------- #
# Typed config sections
# --------------------------------------------------------------------------- #
class ProjectConfig(BaseModel):
    name: str = "football-match-predictor"
    random_seed: int = 42


class DataConfig(BaseModel):
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"

    kaggle_dir: str = "data/raw/kaggle_international_results"
    results_file: str = "data/raw/kaggle_international_results/results.csv"
    shootouts_file: str = "data/raw/kaggle_international_results/shootouts.csv"
    goalscorers_file: str = "data/raw/kaggle_international_results/goalscorers.csv"
    former_names_file: str = "data/raw/kaggle_international_results/former_names.csv"

    fifa_rankings_file: str = "data/raw/fifa_rankings.csv"
    odds_file: str = "data/raw/odds.csv"
    player_squad_features_file: str = "data/raw/player_squad_features.csv"
    advanced_stats_file: str = "data/raw/team_advanced_stats.csv"

    processed_features_parquet: str = "data/processed/match_features.parquet"
    processed_features_csv: str = "data/processed/match_features.csv"


class FeaturesConfig(BaseModel):
    rolling_windows: list[int] = Field(default_factory=lambda: [5, 10, 20])
    use_custom_elo: bool = True
    use_fifa_rankings: bool = True
    use_odds: bool = False
    use_scorer_features: bool = True
    use_player_features: bool = False
    use_advanced_stats: bool = False


class PredictionConfig(BaseModel):
    default_mode: str = "early_prediction"


class EloConfig(BaseModel):
    initial_rating: float = 1500
    home_advantage: float = 60
    neutral_home_advantage: float = 0
    k_factor_friendly: float = 15
    k_factor_qualifier: float = 30
    k_factor_continental: float = 40
    k_factor_world_cup: float = 50
    margin_of_victory_multiplier: bool = True


class ValidationConfig(BaseModel):
    train_end_date: str = "2021-12-31"
    validation_start_date: str = "2022-01-01"
    validation_end_date: str = "2023-12-31"
    test_start_date: str = "2024-01-01"


class ModelConfig(BaseModel):
    target: str = "result_1x2"
    calibrate_probabilities: bool = True
    save_dir: str = "models/artifacts"


class EnsembleConfig(BaseModel):
    elo_weight: float = 0.25
    poisson_weight: float = 0.25
    tabular_weight: float = 0.35
    market_weight: float = 0.15


class Config(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    prediction: PredictionConfig = Field(default_factory=PredictionConfig)
    elo: EloConfig = Field(default_factory=EloConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)

    # Anything not modelled above (e.g. contents of model.yaml) lands here.
    extra: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #
_KNOWN_SECTIONS = {
    "project",
    "data",
    "features",
    "prediction",
    "elo",
    "validation",
    "model",
    "ensemble",
}


def _read_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} did not parse to a mapping.")
    return data


def load_config(
    path: str | os.PathLike[str] = "configs/base.yaml",
    extra_paths: list[str] | None = None,
    load_env: bool = True,
) -> Config:
    """Load and validate configuration.

    Parameters
    ----------
    path:
        Path to the base YAML config.
    extra_paths:
        Optional list of additional YAML files merged into ``Config.extra``
        (e.g. ``configs/model.yaml``). If not given, ``model.yaml`` next to the
        base config is auto-loaded when present.
    load_env:
        If True, load environment variables from ``.env`` (never committed).
    """
    if load_env and load_dotenv is not None:
        load_dotenv(override=False)

    raw = _read_yaml(path)

    known = {k: v for k, v in raw.items() if k in _KNOWN_SECTIONS}
    extra = {k: v for k, v in raw.items() if k not in _KNOWN_SECTIONS}

    # Auto-discover sibling model.yaml unless explicit extras were given.
    if extra_paths is None:
        sibling = Path(path).parent / "model.yaml"
        extra_paths = [str(sibling)] if sibling.exists() else []

    for ep in extra_paths:
        if Path(ep).exists():
            extra.update(_read_yaml(ep))

    cfg = Config(**known)
    cfg.extra = extra
    return cfg


def get_secret(name: str, default: str | None = None) -> str | None:
    """Read a secret from the environment (loaded from .env if available)."""
    return os.environ.get(name, default)
