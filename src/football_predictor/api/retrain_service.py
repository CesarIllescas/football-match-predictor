"""Background retrain job with thread-safe status."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any

from ..pipeline.build_and_train import build_processed_dataset, train_models_from_processed
from ..prediction.predictor import REQUIRED_ARTIFACT_FILES
from ..utils.config import Config, load_config
from ..utils.logging import get_logger
import os

logger = get_logger(__name__)

_lock = threading.Lock()
_status: dict[str, Any] = {
    "running": False,
    "phase": "idle",
    "message": None,
    "error": None,
    "last_completed_at": None,
    "last_result": None,
}


def artifacts_ready(cfg: Config | None = None) -> bool:
    cfg = cfg or load_config()
    save_dir = cfg.model.save_dir
    return all(os.path.isfile(os.path.join(save_dir, f)) for f in REQUIRED_ARTIFACT_FILES)


def get_retrain_status(cfg: Config | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    with _lock:
        out = dict(_status)
    out["artifacts_ready"] = artifacts_ready(cfg)
    return out


def _set(**kwargs: Any) -> None:
    with _lock:
        _status.update(kwargs)


def start_retrain(config_path: str = "configs/base.yaml", on_complete=None) -> bool:
    """Start retrain in a background thread. Returns False if already running."""
    with _lock:
        if _status["running"]:
            return False

    def _worker() -> None:
        _set(running=True, phase="building", message="Building feature dataset…", error=None)
        try:
            cfg = load_config(config_path)
            _set(phase="building", message="Loading raw data and computing features…")
            # run_full_retrain does build + train; update phase mid-way via split calls
            from ..pipeline.build_and_train import build_processed_dataset, train_models_from_processed

            rows = build_processed_dataset(cfg)
            _set(phase="training", message=f"Training models ({rows:,} matches)…")
            summary = train_models_from_processed(cfg)
            result = {"processed_rows": rows, **summary}
            now = datetime.now(timezone.utc).isoformat()
            _set(
                running=False,
                phase="idle",
                message="Retrain finished successfully.",
                error=None,
                last_completed_at=now,
                last_result=result,
            )
            if on_complete:
                on_complete()
            logger.info("Retrain completed: %s", result)
        except Exception as exc:
            logger.exception("Retrain failed")
            _set(
                running=False,
                phase="idle",
                message=None,
                error=str(exc),
            )

    _set(running=True, phase="building", message="Starting…", error=None)
    threading.Thread(target=_worker, daemon=True).start()
    return True
