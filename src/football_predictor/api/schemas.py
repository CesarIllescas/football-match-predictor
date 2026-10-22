"""Pydantic request/response models for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    home_team: str
    away_team: str
    date: str
    tournament: str = "Friendly"
    neutral: bool = False
    host_country: str | None = None
    prediction_mode: str = "early_prediction"


class Scoreline(BaseModel):
    score: str
    prob: float


class PredictResponse(BaseModel):
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    lambda_home: float
    lambda_away: float
    over_15_prob: float | None = None
    over_25_prob: float
    over_35_prob: float | None = None
    btts_prob: float
    top_scorelines: list[Scoreline] = Field(default_factory=list)


class BatchPredictRequest(BaseModel):
    matches: list[PredictRequest]


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]


class ModelInfoResponse(BaseModel):
    model_versions: dict
    feature_count: int
    tabular_backend: str
    use_odds: bool
    default_prediction_mode: str


class RetrainStatusResponse(BaseModel):
    running: bool
    phase: str
    message: str | None = None
    error: str | None = None
    last_completed_at: str | None = None
    last_result: dict | None = None
    artifacts_ready: bool


class TeamsResponse(BaseModel):
    teams: list[str]
