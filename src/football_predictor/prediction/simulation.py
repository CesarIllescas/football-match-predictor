"""Monte-Carlo / analytic score simulation helpers.

The analytic path (exact scoreline matrix) lives in the Poisson model. This module
adds a Monte-Carlo sampler useful for tournament simulation and for sanity checks
against the analytic markets.
"""

from __future__ import annotations

import numpy as np


def simulate_scorelines(
    lambda_home: float,
    lambda_away: float,
    n_sims: int = 20000,
    seed: int = 42,
) -> dict:
    """Sample match scores from two independent Poisson processes.

    Returns 1X2 / over-under / BTTS estimates plus the most common scorelines.
    Useful as a cross-check of the analytic Poisson markets.
    """
    rng = np.random.default_rng(seed)
    h = rng.poisson(lambda_home, n_sims)
    a = rng.poisson(lambda_away, n_sims)
    total = h + a

    scorelines: dict[str, int] = {}
    for hh, aa in zip(h, a):
        key = f"{hh}-{aa}"
        scorelines[key] = scorelines.get(key, 0) + 1
    top = sorted(scorelines.items(), key=lambda kv: kv[1], reverse=True)[:5]

    return {
        "home_win_prob": float(np.mean(h > a)),
        "draw_prob": float(np.mean(h == a)),
        "away_win_prob": float(np.mean(h < a)),
        "over_15_prob": float(np.mean(total > 1.5)),
        "over_25_prob": float(np.mean(total > 2.5)),
        "over_35_prob": float(np.mean(total > 3.5)),
        "btts_prob": float(np.mean((h > 0) & (a > 0))),
        "top_scorelines": [{"score": s, "prob": c / n_sims} for s, c in top],
    }
