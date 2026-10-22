"""Predict a single match and print JSON.

    python scripts/predict_match.py \
        --home-team "Japan" \
        --away-team "Tunisia" \
        --date "2026-06-21" \
        --tournament "FIFA World Cup" \
        --neutral true \
        --host-country "United States" \
        --prediction-mode early_prediction
"""

from __future__ import annotations

import json

import _bootstrap  # noqa: F401
import typer

from football_predictor.prediction.predictor import ArtifactsNotFoundError, Predictor
from football_predictor.utils.config import load_config
from football_predictor.utils.logging import get_logger

logger = get_logger("predict_match")
app = typer.Typer(add_completion=False)


@app.command()
def main(
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
    try:
        predictor = Predictor.from_dir(cfg=cfg)
    except ArtifactsNotFoundError as exc:
        logger.error("%s", exc)
        raise typer.Exit(code=1) from exc
    neutral_bool = str(neutral).strip().lower() in ("true", "1", "yes", "t")
    result = predictor.predict_match(
        home_team=home_team,
        away_team=away_team,
        date=date,
        tournament=tournament,
        neutral=neutral_bool,
        host_country=host_country,
        prediction_mode=prediction_mode,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    app()
