"""End-to-end training pipeline + match predictor.

This module ties the components together:

- :func:`train_pipeline` fits the Elo baseline, Poisson model, tabular model,
  calibrator and configures the ensemble on time-split data.
- :class:`Predictor` loads those artifacts and produces a full prediction for an
  arbitrary fixture by rebuilding strictly pre-match features for that fixture
  (appending it to the historical data and reusing the exact training pipeline,
  which guarantees feature parity and leakage safety).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data import loaders, matching
from ..features.feature_builder import build_features
from ..models.baseline_elo import EloBaselineModel
from ..models.calibration import MulticlassProbabilityCalibrator
from ..models.ensemble import EnsembleModel
from ..models.poisson_model import PoissonGoalModel
from ..models.tabular_model import TabularResultModel
from ..utils.config import Config, load_config
from ..utils.dates import parse_date
from ..utils.logging import get_logger

logger = get_logger(__name__)

MODEL_VERSIONS = {"elo": "v1", "poisson": "v1", "tabular": "v1", "ensemble": "v1"}


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@dataclass
class TrainedArtifacts:
    cfg: Config
    elo_baseline: EloBaselineModel
    poisson: PoissonGoalModel
    tabular: TabularResultModel
    calibrator: MulticlassProbabilityCalibrator
    ensemble: EnsembleModel
    feature_columns: list[str] = field(default_factory=list)
    model_versions: dict = field(default_factory=lambda: dict(MODEL_VERSIONS))


def _model_cfg(cfg: Config) -> dict:
    return cfg.extra.get("tabular", {}), cfg.extra.get("poisson", {}), cfg.extra.get("calibration", {})


def train_pipeline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    cfg: Config,
    target: str = "result_1x2",
) -> TrainedArtifacts:
    """Fit every component on ``train_df`` and calibrate on ``val_df``."""
    tab_cfg, poi_cfg, cal_cfg = _model_cfg(cfg)

    # Elo baseline (1X2 from expected score).
    elo_baseline = EloBaselineModel()
    if "home_elo_expected" in train_df.columns:
        elo_baseline.fit(train_df["home_elo_expected"], train_df[target])

    # Poisson goal model.
    poisson = PoissonGoalModel(
        max_goals=poi_cfg.get("max_goals", 10),
        use_dixon_coles=poi_cfg.get("use_dixon_coles", True),
        rho=poi_cfg.get("dixon_coles_rho", -0.05),
    ).fit(train_df)

    # Tabular model.
    backend_pref = tab_cfg.get("backend_preference", ["lightgbm", "catboost", "xgboost", "sklearn"])
    tabular = TabularResultModel(backend_preference=backend_pref, params=tab_cfg).fit(train_df, target=target)

    # Calibration on validation (fallback to train if no val rows).
    method = cal_cfg.get("method", "isotonic") if cfg.model.calibrate_probabilities else "none"
    calibrator = MulticlassProbabilityCalibrator(method=method)
    cal_source = val_df if (val_df is not None and len(val_df)) else train_df
    cal_source = cal_source[cal_source[target].notna()]
    if len(cal_source):
        raw_proba = tabular.predict_proba(cal_source)
        calibrator.fit(raw_proba, cal_source[target].astype(int).to_numpy())

    ensemble = EnsembleModel.from_config(cfg.ensemble, use_odds=cfg.features.use_odds)

    return TrainedArtifacts(
        cfg=cfg,
        elo_baseline=elo_baseline,
        poisson=poisson,
        tabular=tabular,
        calibrator=calibrator,
        ensemble=ensemble,
        feature_columns=tabular.feature_columns,
    )


def _market_proba(df: pd.DataFrame) -> np.ndarray | None:
    cols = ["market_away_prob", "market_draw_prob", "market_home_prob"]
    if not all(c in df.columns for c in cols):
        return None
    sub = df[cols]
    if sub.isna().all().all():
        return None
    arr = sub.to_numpy(dtype=float)
    # Rows missing odds fall back to uniform; they get low effective weight anyway.
    nan_rows = np.isnan(arr).any(axis=1)
    arr[nan_rows] = [1 / 3, 1 / 3, 1 / 3]
    return arr


def predict_proba_df(artifacts: TrainedArtifacts, df: pd.DataFrame) -> np.ndarray:
    """Ensemble 1X2 probabilities (n, 3) = [away, draw, home] for a feature df."""
    components: dict[str, np.ndarray] = {}
    if "home_elo_expected" in df.columns:
        components["elo"] = artifacts.elo_baseline.predict_proba(df["home_elo_expected"])
    components["poisson"] = artifacts.poisson.predict_proba(df)
    tab_raw = artifacts.tabular.predict_proba(df)
    components["tabular"] = artifacts.calibrator.transform(tab_raw)
    if artifacts.cfg.features.use_odds:
        mk = _market_proba(df)
        if mk is not None:
            components["market"] = mk
    return artifacts.ensemble.combine_proba(components)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_artifacts(artifacts: TrainedArtifacts, save_dir: str, elo_system=None) -> None:
    os.makedirs(save_dir, exist_ok=True)
    artifacts.elo_baseline.save(os.path.join(save_dir, "elo_baseline.json"))
    if elo_system is not None:
        elo_system.save(os.path.join(save_dir, "elo_config.json"))
    artifacts.poisson.save(os.path.join(save_dir, "poisson_model.joblib"))
    artifacts.tabular.save(
        os.path.join(save_dir, "tabular_model.joblib"),
        feature_columns_path=os.path.join(save_dir, "feature_columns.json"),
    )
    artifacts.calibrator.save(os.path.join(save_dir, "calibrator.joblib"))
    artifacts.ensemble.save(os.path.join(save_dir, "ensemble_config.json"))
    with open(os.path.join(save_dir, "model_versions.json"), "w", encoding="utf-8") as fh:
        json.dump(artifacts.model_versions, fh, indent=2)
    logger.info("Saved all artifacts to %s", save_dir)


REQUIRED_ARTIFACT_FILES = (
    "elo_baseline.json",
    "poisson_model.joblib",
    "tabular_model.joblib",
    "calibrator.joblib",
    "ensemble_config.json",
)


class ArtifactsNotFoundError(FileNotFoundError):
    """Raised when model artifacts are missing (train step not run yet)."""


def _check_artifacts_dir(save_dir: str) -> None:
    missing = [f for f in REQUIRED_ARTIFACT_FILES if not os.path.isfile(os.path.join(save_dir, f))]
    if missing:
        raise ArtifactsNotFoundError(
            f"No trained models in {save_dir!r}. Missing: {', '.join(missing)}.\n"
            "Run these commands first:\n"
            "  python scripts/build_dataset.py --config configs/base.yaml\n"
            "  python scripts/train_models.py --config configs/base.yaml"
        )


def load_artifacts(save_dir: str, cfg: Config) -> TrainedArtifacts:
    _check_artifacts_dir(save_dir)
    elo_baseline = EloBaselineModel.load(os.path.join(save_dir, "elo_baseline.json"))
    poisson = PoissonGoalModel.load(os.path.join(save_dir, "poisson_model.joblib"))
    tabular = TabularResultModel.load(os.path.join(save_dir, "tabular_model.joblib"))
    calibrator = MulticlassProbabilityCalibrator.load(os.path.join(save_dir, "calibrator.joblib"))
    ensemble = EnsembleModel.load(os.path.join(save_dir, "ensemble_config.json"))
    versions = dict(MODEL_VERSIONS)
    vpath = os.path.join(save_dir, "model_versions.json")
    if os.path.exists(vpath):
        with open(vpath, "r", encoding="utf-8") as fh:
            versions = json.load(fh)
    return TrainedArtifacts(
        cfg=cfg,
        elo_baseline=elo_baseline,
        poisson=poisson,
        tabular=tabular,
        calibrator=calibrator,
        ensemble=ensemble,
        feature_columns=tabular.feature_columns,
        model_versions=versions,
    )


# --------------------------------------------------------------------------- #
# Predictor
# --------------------------------------------------------------------------- #
class Predictor:
    """Produce full predictions for arbitrary fixtures using trained artifacts."""

    def __init__(self, artifacts: TrainedArtifacts) -> None:
        self.artifacts = artifacts
        self.cfg = artifacts.cfg
        self._raw_norm: dict | None = None

    @classmethod
    def from_dir(cls, save_dir: str | None = None, cfg: Config | None = None) -> "Predictor":
        cfg = cfg or load_config()
        save_dir = save_dir or cfg.model.save_dir
        return cls(load_artifacts(save_dir, cfg))

    def _ensure_raw(self) -> dict:
        if self._raw_norm is None:
            raw = loaders.load_all_raw(self.cfg)
            self._raw_norm = matching.normalize_all(raw)
        return self._raw_norm

    def _normalize_team(self, team: str) -> str:
        norm = matching.build_normalizer(self._ensure_raw().get("former_names"))
        return norm.normalize(team) or team

    def _build_fixture_features(
        self,
        fixtures: pd.DataFrame,
        prediction_mode: str,
    ) -> pd.DataFrame:
        """Append future fixtures to history and rebuild features; return fixture rows."""
        raw = dict(self._ensure_raw())
        results = raw["results"].copy()
        results["__is_fixture"] = False
        fx = fixtures.copy()
        fx["__is_fixture"] = True
        combined = pd.concat([results, fx], ignore_index=True)
        raw["results"] = combined
        feats, _ = build_features(raw, self.cfg, prediction_mode=prediction_mode)
        if "__is_fixture" not in feats.columns:
            raise RuntimeError("Fixture marker lost during feature build.")
        fixture_rows = feats[feats["__is_fixture"].fillna(False).astype(bool)].copy()
        return fixture_rows.reset_index(drop=True)

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        date: str,
        tournament: str = "Friendly",
        neutral: bool = False,
        host_country: str | None = None,
        prediction_mode: str | None = None,
    ) -> dict:
        prediction_mode = prediction_mode or self.cfg.prediction.default_mode
        home_n = self._normalize_team(home_team)
        away_n = self._normalize_team(away_team)
        d = parse_date(date)

        fixture = pd.DataFrame([{
            "date": d,
            "home_team": home_n,
            "away_team": away_n,
            "home_score": np.nan,
            "away_score": np.nan,
            "tournament": tournament,
            "city": None,
            "country": host_country,
            "neutral": bool(neutral),
        }])

        feats = self._build_fixture_features(fixture, prediction_mode)
        if len(feats) == 0:
            raise RuntimeError("Failed to build features for the requested fixture.")
        row = feats.iloc[[-1]]

        return self._assemble_prediction(row, home_team, away_team, date, prediction_mode)

    def _assemble_prediction(
        self,
        row: pd.DataFrame,
        home_team: str,
        away_team: str,
        date: str,
        prediction_mode: str,
    ) -> dict:
        art = self.artifacts

        # Poisson markets (goals, over/under, BTTS, scorelines).
        lh, la = art.poisson.predict_lambdas(row)
        poisson_markets = art.poisson.predict_markets(float(lh[0]), float(la[0]))

        # Ensemble 1X2.
        ens = predict_proba_df(art, row)[0]  # [away, draw, home]

        predictions = {
            "home_win_prob": round(float(ens[2]), 4),
            "draw_prob": round(float(ens[1]), 4),
            "away_win_prob": round(float(ens[0]), 4),
            "lambda_home": round(poisson_markets["lambda_home"], 4),
            "lambda_away": round(poisson_markets["lambda_away"], 4),
            "over_15_prob": round(poisson_markets["over_15_prob"], 4),
            "over_25_prob": round(poisson_markets["over_25_prob"], 4),
            "over_35_prob": round(poisson_markets["over_35_prob"], 4),
            "btts_prob": round(poisson_markets["btts_prob"], 4),
            "top_scorelines": [
                {"score": s["score"], "prob": round(s["prob"], 4)}
                for s in poisson_markets["top_scorelines"][:3]
            ],
        }
        return {
            "home_team": home_team,
            "away_team": away_team,
            "date": date,
            "prediction_mode": prediction_mode,
            "predictions": predictions,
            "model_versions": art.model_versions,
        }


# --------------------------------------------------------------------------- #
# Console entry point (football-predict)
# --------------------------------------------------------------------------- #
def main() -> None:  # pragma: no cover - thin CLI wrapper
    import typer

    app = typer.Typer(add_completion=False)

    @app.command()
    def predict(
        home_team: str = typer.Option(..., "--home-team"),
        away_team: str = typer.Option(..., "--away-team"),
        date: str = typer.Option(..., "--date"),
        tournament: str = typer.Option("Friendly", "--tournament"),
        neutral: str = typer.Option("false", "--neutral", help="true/false"),
        host_country: str = typer.Option(None, "--host-country"),
        prediction_mode: str = typer.Option("early_prediction", "--prediction-mode"),
        config: str = typer.Option("configs/base.yaml", "--config"),
    ) -> None:
        cfg = load_config(config)
        predictor = Predictor.from_dir(cfg=cfg)
        neutral_bool = str(neutral).strip().lower() in ("true", "1", "yes", "t")
        result = predictor.predict_match(
            home_team, away_team, date, tournament, neutral_bool, host_country, prediction_mode
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))

    app()


if __name__ == "__main__":  # pragma: no cover
    main()
