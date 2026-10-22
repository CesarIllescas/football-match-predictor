"""Evaluation metrics for 1X2, binary markets, and expected goals."""

from __future__ import annotations

import numpy as np
import pandas as pd


def multiclass_brier_score(y_true: np.ndarray, proba: np.ndarray, n_classes: int = 3) -> float:
    """Mean squared error between one-hot truth and predicted probabilities."""
    y_true = np.asarray(y_true, dtype=int)
    onehot = np.zeros((len(y_true), n_classes))
    onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def evaluate_1x2(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    """Accuracy, balanced accuracy, log loss, multiclass Brier, macro-F1."""
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        log_loss,
    )

    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    preds = proba.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
        "log_loss": float(log_loss(y_true, proba, labels=[0, 1, 2])),
        "multiclass_brier_score": multiclass_brier_score(y_true, proba),
        "macro_f1": float(f1_score(y_true, preds, average="macro")),
        "n": int(len(y_true)),
    }


def evaluate_binary(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    """ROC AUC, Brier, log loss, precision/recall/F1 at 0.5 threshold."""
    from sklearn.metrics import (
        brier_score_loss,
        f1_score,
        log_loss,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true, dtype=int)
    prob = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    preds = (prob >= 0.5).astype(int)
    out: dict[str, float] = {}
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, prob))
    except ValueError:
        out["roc_auc"] = float("nan")
    out["brier_score"] = float(brier_score_loss(y_true, prob))
    try:
        out["log_loss"] = float(log_loss(y_true, prob, labels=[0, 1]))
    except ValueError:
        out["log_loss"] = float("nan")
    out["precision"] = float(precision_score(y_true, preds, zero_division=0))
    out["recall"] = float(recall_score(y_true, preds, zero_division=0))
    out["f1"] = float(f1_score(y_true, preds, zero_division=0))
    out["n"] = int(len(y_true))
    return out


def evaluate_expected_goals(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """MAE, RMSE, and Poisson deviance for expected goals."""
    y_true = np.asarray(y_true, dtype=float)
    pred = np.clip(np.asarray(pred, dtype=float), 1e-9, None)
    mae = float(np.mean(np.abs(y_true - pred)))
    rmse = float(np.sqrt(np.mean((y_true - pred) ** 2)))
    out = {"MAE": mae, "RMSE": rmse, "n": int(len(y_true))}
    try:
        from sklearn.metrics import mean_poisson_deviance

        out["poisson_deviance"] = float(mean_poisson_deviance(y_true, pred))
    except Exception:
        out["poisson_deviance"] = float("nan")
    return out


def confusion(y_true: np.ndarray, proba: np.ndarray) -> np.ndarray:
    from sklearn.metrics import confusion_matrix

    preds = np.asarray(proba).argmax(axis=1)
    return confusion_matrix(np.asarray(y_true, dtype=int), preds, labels=[0, 1, 2])


def performance_by_group(
    df: pd.DataFrame,
    y_true: np.ndarray,
    proba: np.ndarray,
    group_col: str,
) -> pd.DataFrame:
    """Per-group 1X2 metrics (e.g. by year or competition)."""
    work = df.copy().reset_index(drop=True)
    work["__y"] = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    rows = []
    for key, idx in work.groupby(group_col).groups.items():
        positions = work.index.get_indexer(idx)
        m = evaluate_1x2(work.loc[idx, "__y"].to_numpy(), proba[positions])
        m[group_col] = key
        rows.append(m)
    return pd.DataFrame(rows).sort_values(group_col).reset_index(drop=True)
