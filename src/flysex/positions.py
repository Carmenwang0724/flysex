"""Tabulate the positions of flies the model calls male."""

from pathlib import Path

import numpy as np
import pandas as pd

from .ranking import rank_scores
from .session import find_calibration, find_tracker_directory, read_pixels_per_mm

# Define the score columns and the male position table layout
required_columns = (
    "frame",
    "well",
    "fly",
    "tracker_row",
    "score",
    "sex",
    "confident",
    "x",
    "y",
)
position_columns = [
    "frame",
    "well",
    "fly",
    "x_mm",
    "y_mm",
    "score",
    "confident",
    "rank",
    "x_px",
    "y_px",
    "tracker_row",
]
millimetre_decimals = 2
score_decimals = 4


def _validate_scales(pixels_per_mm: dict, wells: list[int]) -> None:
    """Require a positive physical scale for every scored well."""
    # Validate the input values
    missing = sorted(set(wells) - set(pixels_per_mm))
    if missing:
        raise ValueError(f"No pixels-per-millimetre value for wells {missing}.")
    for well in wells:
        value = pixels_per_mm[well]
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"Well {well} has an invalid pixels-per-millimetre value.")


def _validate_calls(calls: pd.DataFrame) -> None:
    """Require the score columns that describe each fly's sex call and position."""
    # Validate the input values
    missing = set(required_columns) - set(calls.columns)
    if missing:
        raise ValueError(
            f"Score table is missing columns: {', '.join(sorted(missing))}."
        )
    if not calls["sex"].isin(["M", "F"]).all():
        raise ValueError("sex must contain only M or F.")


def read_well_scales(session_directory: Path, wells: list[int]) -> dict[int, float]:
    """Read each well's pixels-per-millimetre value from its calibration file."""
    # Read the calibration beside each well's tracker folder
    scales = {}
    for well in wells:
        tracker_directory = find_tracker_directory(
            session_directory=session_directory,
            well_number=well,
        )
        calibration_path = find_calibration(tracker_directory=tracker_directory)
        scales[well] = read_pixels_per_mm(calibration_path=calibration_path)

    # Return the scale for each well
    return scales


def male_positions(
    calls: pd.DataFrame,
    pixels_per_mm: dict[int, float],
) -> pd.DataFrame:
    """List every fly scored as male with its position in millimetres and pixels."""
    # Validate and rank the scored flies
    _validate_calls(calls=calls)
    ranked = rank_scores(calls=calls)
    wells = sorted(int(well) for well in ranked["well"].unique())
    _validate_scales(
        pixels_per_mm=pixels_per_mm,
        wells=wells,
    )

    # Keep the flies the model calls male
    males = ranked[ranked["sex"] == "M"].copy()
    scales = np.asarray(
        a=[pixels_per_mm[int(well)] for well in males["well"]],
        dtype=np.float64,
    )

    # Convert tracker pixels to millimetres
    males["x_px"] = males["x"].to_numpy(dtype=np.float64)
    males["y_px"] = males["y"].to_numpy(dtype=np.float64)
    males["x_mm"] = np.round(
        a=males["x_px"].to_numpy(dtype=np.float64) / scales,
        decimals=millimetre_decimals,
    )
    males["y_mm"] = np.round(
        a=males["y_px"].to_numpy(dtype=np.float64) / scales,
        decimals=millimetre_decimals,
    )
    males["score"] = np.round(
        a=males["score"].to_numpy(dtype=np.float64),
        decimals=score_decimals,
    )

    # Return the male positions in frame order
    return males[position_columns].reset_index(drop=True)


def count_male_calls(calls: pd.DataFrame) -> list[dict]:
    """Count the scored frames in each well by how many flies were called male."""
    # Validate the input values
    _validate_calls(calls=calls)

    # Count the flies called male in each scored frame
    per_frame = (
        calls.assign(male=calls["sex"] == "M")
        .groupby(by=["well", "frame"])["male"]
        .sum()
    )

    # Summarize each well's frames by male count
    counts = []
    for well, frame_counts in per_frame.groupby(level="well"):
        counts.append(
            {
                "well": int(well),
                "scored_frames": int(len(frame_counts)),
                "frames_with_no_male": int((frame_counts == 0).sum()),
                "frames_with_one_male": int((frame_counts == 1).sum()),
                "frames_with_two_or_more": int((frame_counts >= 2).sum()),
            }
        )

    # Return one count record per well
    return counts


def write_male_positions(
    session_directory: Path,
    calls: pd.DataFrame,
    output_path: Path,
) -> dict:
    """Write the male position table for an existing score table."""
    # Refuse to replace an existing table
    output_path = Path(output_path)
    if output_path.exists():
        raise ValueError(f"Output file already exists: {output_path}")

    # Read the scale of each scored well
    _validate_calls(calls=calls)
    wells = sorted(int(well) for well in calls["well"].unique())
    scales = read_well_scales(
        session_directory=Path(session_directory),
        wells=wells,
    )

    # Build and save the male position table
    positions = male_positions(
        calls=calls,
        pixels_per_mm=scales,
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    positions.to_csv(
        path_or_buf=output_path,
        index=False,
    )

    # Return the counts behind the saved table
    return {
        "output_path": str(output_path),
        "male_rows": len(positions),
        "wells": count_male_calls(calls=calls),
        "pixels_per_mm": scales,
    }
