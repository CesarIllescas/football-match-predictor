"""FastAPI application exposing the football match predictor + simple web UI.

Run from the project root:

    uvicorn football_predictor.api.main:app --reload

Open http://127.0.0.1:8000/ for the prediction form.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..prediction.predictor import ArtifactsNotFoundError, Predictor
from ..utils.config import load_config
from ..utils.logging import get_logger
from .retrain_service import artifacts_ready, get_retrain_status, start_retrain
from .schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    RetrainStatusResponse,
    TeamsResponse,
)

logger = get_logger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Football Match Predictor", version="0.1.0")

_state: dict = {"predictor": None, "cfg": None, "teams": None}


def _invalidate_predictor() -> None:
    _state["predictor"] = None


def _get_cfg():
    if _state["cfg"] is None:
        _state["cfg"] = load_config()
    return _state["cfg"]


def _get_predictor():
    if _state["predictor"] is None:
        cfg = _get_cfg()
        try:
            _state["predictor"] = Predictor.from_dir(cfg=cfg)
        except ArtifactsNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _state["predictor"]


def _to_response(result: dict) -> PredictResponse:
    p = result["predictions"]
    return PredictResponse(**p)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/health")
def health() -> dict:
    cfg = _get_cfg()
    return {"status": "ok", "artifacts_ready": artifacts_ready(cfg)}


@app.get("/teams", response_model=TeamsResponse)
def list_teams() -> TeamsResponse:
    if _state["teams"] is None:
        cfg = _get_cfg()
        try:
            df = pd.read_csv(cfg.data.results_file, usecols=["home_team", "away_team"])
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=f"results.csv missing: {exc}") from exc
        teams = sorted(set(df["home_team"].dropna()) | set(df["away_team"].dropna()))
        _state["teams"] = [str(t) for t in teams]
    return TeamsResponse(teams=_state["teams"])


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    predictor = _get_predictor()
    art = predictor.artifacts
    return ModelInfoResponse(
        model_versions=art.model_versions,
        feature_count=len(art.feature_columns),
        tabular_backend=art.tabular.backend,
        use_odds=art.cfg.features.use_odds,
        default_prediction_mode=art.cfg.prediction.default_mode,
    )


@app.get("/retrain/status", response_model=RetrainStatusResponse)
def retrain_status() -> RetrainStatusResponse:
    return RetrainStatusResponse(**get_retrain_status(_get_cfg()))


@app.post("/retrain", response_model=RetrainStatusResponse)
def retrain_start() -> RetrainStatusResponse:
    started = start_retrain(on_complete=_invalidate_predictor)
    if not started:
        raise HTTPException(status_code=409, detail="Retrain already in progress.")
    return RetrainStatusResponse(**get_retrain_status(_get_cfg()))


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    status = get_retrain_status(_get_cfg())
    if status["running"]:
        raise HTTPException(status_code=409, detail="Retrain in progress; try again shortly.")
    predictor = _get_predictor()
    result = predictor.predict_match(
        home_team=req.home_team,
        away_team=req.away_team,
        date=req.date,
        tournament=req.tournament,
        neutral=req.neutral,
        host_country=req.host_country,
        prediction_mode=req.prediction_mode,
    )
    return _to_response(result)


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(req: BatchPredictRequest) -> BatchPredictResponse:
    status = get_retrain_status(_get_cfg())
    if status["running"]:
        raise HTTPException(status_code=409, detail="Retrain in progress; try again shortly.")
    predictor = _get_predictor()
    out = []
    for m in req.matches:
        result = predictor.predict_match(
            home_team=m.home_team,
            away_team=m.away_team,
            date=m.date,
            tournament=m.tournament,
            neutral=m.neutral,
            host_country=m.host_country,
            prediction_mode=m.prediction_mode,
        )
        out.append(_to_response(result))
    return BatchPredictResponse(predictions=out)
