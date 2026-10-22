"""Optional automatic downloaders.

The only first-class automatic source is the Kaggle international results dataset.
Everything else (FIFA rankings, odds, players) is manual or behind paid APIs, so
those functions only print clear manual instructions. Nothing here ever crashes
unclearly or hardcodes secrets.
"""

from __future__ import annotations

import os
import zipfile

from ..utils.config import Config, get_secret
from ..utils.logging import get_logger

logger = get_logger(__name__)

KAGGLE_DATASET = "martj42/international-football-results-from-1872-to-2017"
KAGGLE_FILES = ("results.csv", "shootouts.csv", "goalscorers.csv", "former_names.csv")


def _kaggle_manual_instructions(target_dir: str) -> None:
    logger.warning(
        "\n".join(
            [
                "Automatic Kaggle download not available.",
                "To set it up, either:",
                "  1) put KAGGLE_USERNAME and KAGGLE_KEY in your .env, OR",
                "  2) place a kaggle.json in ~/.kaggle/ (or %USERPROFILE%/.kaggle/).",
                "",
                "Or download manually:",
                f"  - Visit https://www.kaggle.com/datasets/{KAGGLE_DATASET}",
                "  - Download and extract these files into:",
                f"      {target_dir}",
                f"      {', '.join(KAGGLE_FILES)}",
            ]
        )
    )


def download_kaggle_international_results(cfg: Config) -> bool:
    """Download the Kaggle international results dataset if credentials exist.

    Returns True on success, False if it could not download (instructions are
    printed in that case). Never raises for the "no credentials" path.
    """
    target_dir = cfg.data.kaggle_dir
    os.makedirs(target_dir, exist_ok=True)

    username = get_secret("KAGGLE_USERNAME")
    key = get_secret("KAGGLE_KEY")
    if username and key:
        os.environ.setdefault("KAGGLE_USERNAME", username)
        os.environ.setdefault("KAGGLE_KEY", key)

    try:
        # Import is deferred: kaggle authenticates at import time.
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        logger.info("Authenticated with Kaggle. Downloading %s ...", KAGGLE_DATASET)
        api.dataset_download_files(KAGGLE_DATASET, path=target_dir, unzip=True, quiet=False)

        # Some kaggle versions leave a zip behind; extract if present.
        for fname in os.listdir(target_dir):
            if fname.endswith(".zip"):
                zpath = os.path.join(target_dir, fname)
                with zipfile.ZipFile(zpath) as zf:
                    zf.extractall(target_dir)
                os.remove(zpath)

        present = [f for f in KAGGLE_FILES if os.path.exists(os.path.join(target_dir, f))]
        logger.info("Kaggle download complete. Files present: %s", present)
        return "results.csv" in present
    except Exception as exc:
        logger.warning("Kaggle download failed: %s", exc)
        _kaggle_manual_instructions(target_dir)
        return False


def _manual_only(name: str, target: str, columns: list[str]) -> bool:
    logger.warning(
        "\n".join(
            [
                f"'{name}' has no supported automatic download (manual or paid API).",
                f"Provide it manually at: {target}",
                f"Expected columns: {columns}",
                "The pipeline runs fine without it.",
            ]
        )
    )
    return False


def download_fifa_rankings(cfg: Config) -> bool:
    return _manual_only(
        "fifa_rankings", cfg.data.fifa_rankings_file, ["rank_date", "team", "rank", "points"]
    )


def download_odds(cfg: Config) -> bool:
    return _manual_only(
        "odds",
        cfg.data.odds_file,
        ["date", "home_team", "away_team", "bookmaker", "home_odds", "draw_odds", "away_odds", "timestamp"],
    )


def download_source(source: str, cfg: Config) -> bool:
    """Dispatch a download by source name. ``all`` runs every known source."""
    source = source.strip().lower()
    if source in ("all",):
        ok = download_kaggle_international_results(cfg)
        download_fifa_rankings(cfg)
        download_odds(cfg)
        return ok
    if source == "kaggle_international_results":
        return download_kaggle_international_results(cfg)
    if source == "fifa_rankings":
        return download_fifa_rankings(cfg)
    if source == "odds":
        return download_odds(cfg)
    logger.error("Unknown source %r. Known: all, kaggle_international_results, fifa_rankings, odds", source)
    return False
