"""Date parsing and time-based split helpers."""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd


def parse_date(value: str | date | datetime | pd.Timestamp) -> pd.Timestamp:
    """Parse a single date-like value into a normalized (date-only) Timestamp."""
    ts = pd.Timestamp(value)
    return ts.normalize()


def to_datetime(series: pd.Series) -> pd.Series:
    """Vectorized, lenient parse of a date column to tz-naive Timestamps."""
    return pd.to_datetime(series, errors="coerce", utc=False)


def time_split(
    df: pd.DataFrame,
    train_end: str,
    val_start: str,
    val_end: str,
    test_start: str,
    date_col: str = "date",
) -> dict[str, pd.DataFrame]:
    """Split a dataframe into train/validation/test purely by date.

    No row is ever shuffled across boundaries, which is essential to avoid
    look-ahead leakage when evaluating temporal models.
    """
    d = to_datetime(df[date_col])
    train_end_ts = parse_date(train_end)
    val_start_ts = parse_date(val_start)
    val_end_ts = parse_date(val_end)
    test_start_ts = parse_date(test_start)

    train = df[d <= train_end_ts]
    val = df[(d >= val_start_ts) & (d <= val_end_ts)]
    test = df[d >= test_start_ts]

    return {"train": train, "validation": val, "test": test}


def walk_forward_folds(
    df: pd.DataFrame,
    start_year: int,
    end_year: int,
    date_col: str = "date",
) -> list[dict[str, object]]:
    """Generate walk-forward folds: train up to year Y-1, test on year Y.

    Returns a list of dicts with ``train``, ``test`` frames and ``test_year``.
    """
    d = to_datetime(df[date_col])
    folds: list[dict[str, object]] = []
    for year in range(start_year, end_year + 1):
        train = df[d < pd.Timestamp(year=year, month=1, day=1)]
        test = df[(d >= pd.Timestamp(year=year, month=1, day=1))
                  & (d < pd.Timestamp(year=year + 1, month=1, day=1))]
        if len(test) == 0 or len(train) == 0:
            continue
        folds.append({"test_year": year, "train": train, "test": test})
    return folds
