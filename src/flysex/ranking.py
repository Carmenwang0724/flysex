"""Rank flies within each scored frame using model scores."""

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("frame", "well", "fly", "tracker_row", "score")
INDEX_MINIMUMS = {"frame": 0, "well": 1, "fly": 1, "tracker_row": 0}


def _integer_values(values: pd.Series, minimum: int) -> np.ndarray:
    """Validate and normalize one integer index column."""
    if not pd.api.types.is_numeric_dtype(values.dtype) or pd.api.types.is_complex_dtype(values.dtype):
        raise ValueError(f"{values.name} must contain integer indexes.")
    if pd.api.types.is_bool_dtype(values.dtype):
        raise ValueError(f"{values.name} must contain integer indexes, not booleans.")

    if pd.api.types.is_integer_dtype(values.dtype):
        valid = values.notna() & values.ge(minimum) & values.le(np.iinfo(np.int64).max)
        if not valid.all():
            raise ValueError(f"{values.name} must contain finite integers >= {minimum}.")
        return values.to_numpy(dtype=np.int64)

    numbers = values.to_numpy(dtype=np.float64, na_value=np.nan)
    valid = np.isfinite(numbers) & (numbers >= minimum)
    valid &= numbers < float(2**63)
    valid &= numbers == np.floor(numbers)
    if not valid.all():
        raise ValueError(f"{values.name} must contain finite integers >= {minimum}.")
    return values.to_numpy(dtype=np.int64)


def _validate_calls(calls: pd.DataFrame) -> pd.DataFrame:
    """Validate score rows and return a normalized copy."""
    if not isinstance(calls, pd.DataFrame):
        raise ValueError("calls must be a pandas DataFrame.")
    if not calls.columns.is_unique:
        raise ValueError("Score table column names must be unique.")
    missing = set(REQUIRED_COLUMNS) - set(calls.columns)
    if missing:
        raise ValueError(f"Score table is missing columns: {', '.join(sorted(missing))}.")

    normalized = calls.copy()
    for column, minimum in INDEX_MINIMUMS.items():
        normalized[column] = _integer_values(values=calls[column], minimum=minimum)

    scores = calls["score"]
    if not pd.api.types.is_numeric_dtype(scores.dtype) or pd.api.types.is_complex_dtype(scores.dtype):
        raise ValueError("score must contain numeric values between 0 and 1.")
    if pd.api.types.is_bool_dtype(scores.dtype):
        raise ValueError("score must contain numeric values, not booleans.")
    values = scores.to_numpy(dtype=np.float64, na_value=np.nan)
    if not (np.isfinite(values) & (values >= 0) & (values <= 1)).all():
        raise ValueError("score must contain finite values between 0 and 1.")
    normalized["score"] = values

    if normalized.duplicated(subset=["frame", "well", "fly"]).any():
        raise ValueError("Each fly may have only one score per frame and well.")
    frame_rows = normalized.groupby(["well", "frame"])["tracker_row"].nunique()
    if (frame_rows != 1).any():
        raise ValueError("Each frame and well must map to exactly one tracker_row.")
    row_frames = normalized.groupby(["well", "tracker_row"])["frame"].nunique()
    if (row_frames != 1).any():
        raise ValueError("A tracker_row cannot map to different frames in one well.")
    return normalized


def rank_scores(calls: pd.DataFrame) -> pd.DataFrame:
    """Sort every scored fly within a frame and add its rank and next-score gap."""
    ranked = _validate_calls(calls=calls)
    ranked = ranked.sort_values(
        by=["well", "frame", "score", "fly"],
        ascending=[True, True, False, True],
        kind="stable",
    ).reset_index(drop=True)

    groups = ranked.groupby(["well", "frame"], sort=False)
    ranked["rank"] = groups.cumcount() + 1
    ranked["rank_gap"] = ranked["score"] - groups["score"].shift(periods=-1)
    return ranked
