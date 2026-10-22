"""Calibration curve computation and optional plotting.

Plotting requires matplotlib; if it is missing we still return the numeric
reliability data so calibration can be inspected programmatically.
"""

from __future__ import annotations

import numpy as np


def reliability_curve(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10):
    """Return (mean_predicted, fraction_positive, bin_counts) for a binary target."""
    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(prob, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)

    mean_pred = np.full(n_bins, np.nan)
    frac_pos = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = idx == b
        counts[b] = int(mask.sum())
        if counts[b] > 0:
            mean_pred[b] = prob[mask].mean()
            frac_pos[b] = y_true[mask].mean()
    return mean_pred, frac_pos, counts


def plot_calibration(y_true: np.ndarray, prob: np.ndarray, path: str, title: str = "Calibration", n_bins: int = 10) -> bool:
    """Save a reliability diagram to ``path``. Returns False if matplotlib missing."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    mean_pred, frac_pos, _ = reliability_curve(y_true, prob, n_bins)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    ax.plot(mean_pred, frac_pos, "o-", label="model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True
