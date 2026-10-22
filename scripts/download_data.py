"""Download (or print manual instructions for) supported data sources.

Examples
--------
    python scripts/download_data.py --source all
    python scripts/download_data.py --source kaggle_international_results
    python scripts/download_data.py --source fifa_rankings
    python scripts/download_data.py --source odds
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import typer

from football_predictor.data.downloaders import download_source
from football_predictor.utils.config import load_config
from football_predictor.utils.logging import get_logger

logger = get_logger("download_data")
app = typer.Typer(add_completion=False)


@app.command()
def main(
    source: str = typer.Option("all", "--source", help="all | kaggle_international_results | fifa_rankings | odds"),
    config: str = typer.Option("configs/base.yaml", "--config"),
) -> None:
    cfg = load_config(config)
    ok = download_source(source, cfg)
    if ok:
        logger.info("Download step for %r completed.", source)
    else:
        logger.warning("Download for %r not completed automatically; see instructions above.", source)


if __name__ == "__main__":
    app()
