"""Build the processed pre-match feature dataset.

Loads raw sources, normalizes team names, builds targets + all pre-match
features, and writes ``match_features.parquet`` and ``.csv``.

    python scripts/build_dataset.py --config configs/base.yaml
"""

from __future__ import annotations

import os

import _bootstrap  # noqa: F401
import typer

from football_predictor.data import loaders, matching
from football_predictor.features.feature_builder import build_features
from football_predictor.utils.config import load_config
from football_predictor.utils.logging import get_logger

logger = get_logger("build_dataset")
app = typer.Typer(add_completion=False)


@app.command()
def main(
    config: str = typer.Option("configs/base.yaml", "--config"),
) -> None:
    cfg = load_config(config)

    logger.info("Loading raw data ...")
    raw = loaders.load_all_raw(cfg)

    logger.info("Normalizing team names across all sources ...")
    raw = matching.normalize_all(raw)

    logger.info("Building features ...")
    df, _artifacts = build_features(raw, cfg)

    os.makedirs(cfg.data.processed_dir, exist_ok=True)
    parquet_path = cfg.data.processed_features_parquet
    csv_path = cfg.data.processed_features_csv

    try:
        df.to_parquet(parquet_path, index=False)
        logger.info("Wrote %s", parquet_path)
    except Exception as exc:  # pyarrow may be missing
        logger.warning("Could not write parquet (%s); CSV will still be written.", exc)

    df.to_csv(csv_path, index=False)
    logger.info("Wrote %s", csv_path)
    logger.info("Final dataset: %d rows x %d columns.", df.shape[0], df.shape[1])


if __name__ == "__main__":
    app()
