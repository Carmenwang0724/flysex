"""Export score-ranked FlyTracker rows with explicit coverage records."""

from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .ranking import rank_scores
from .session import find_calibration, find_tracker_directory, read_tracks

# Define the exported mapping columns
base_mapping_columns = ["frame", "well", "tracker_row", "rank", "fly", "score"]
export_readme = """FlyTracker rows ranked by model score

At rows marked ranked=True in rank_coverage.csv, fly1.csv contains the fly
with the highest model score, fly2.csv the next highest, and so on. Tied
scores are ordered by the original fly number. All flies are included,
including low-confidence predictions. No stocking record, male count, or
expert label is used to choose the order.

A rank is not an individual identity and is not necessarily a male. Ranks
are computed independently for each scored frame. They are never carried
forward or interpolated between stills. At rows marked ranked=False, each
file contains its original FlyTracker row and has no score-based rank.
Use rank_coverage.csv when analyzing these files.

rank_mapping.csv records the original fly number and score at every ranked
row. tracker_row is a zero-based CSV data-row position, excluding the header.
frame is the input image/video frame number. An empty frame in the coverage
table means that tracker row was not scored.

The mapping and coverage tables are in the export root. The original CSV
column order, row count, and unscored row values are preserved. Calibration
files are copied unchanged.
"""


def _validate_destination(session_directory: Path, output_directory: Path) -> None:
    """Reject overlapping paths and occupied export destinations."""
    # Validate the input values
    if not session_directory.is_dir():
        raise ValueError(f"Session directory does not exist: {session_directory}")
    overlaps = output_directory == session_directory
    overlaps |= session_directory in output_directory.parents
    overlaps |= output_directory in session_directory.parents
    if overlaps:
        raise ValueError("Session and export directories must not overlap.")
    if output_directory.exists():
        if not output_directory.is_dir() or any(output_directory.iterdir()):
            raise ValueError("Export destination must be absent or an empty directory.")


def _validate_tracks(tracks: list[pd.DataFrame]) -> None:
    """Check that tracker files have matching columns and row counts."""
    # Validate the input values
    if not tracks:
        raise ValueError("Tracker directory contains no fly tables.")
    expected_columns = tracks[0].columns
    expected_length = len(tracks[0])
    for track in tracks:
        if not track.columns.is_unique:
            raise ValueError("Tracker column names must be unique.")
        if not track.columns.equals(expected_columns):
            raise ValueError("Tracker files must have identical column order.")
        if len(track) != expected_length:
            raise ValueError("Tracker files must have identical row counts.")


def _validate_frame_groups(calls: pd.DataFrame, tracks: list[pd.DataFrame]) -> None:
    """Require one score per tracked fly at every exported frame."""
    # Require all tracker IDs in every scored frame
    expected_flies = set(
        range(
            1,
            len(tracks) + 1,
        )
    )
    for frame, group in calls.groupby(
        by="frame",
        sort=False,
    ):
        if set(group["fly"]) != expected_flies:
            raise ValueError(
                f"Frame {frame} must score every tracked fly exactly once."
            )
    if (calls["tracker_row"] >= len(tracks[0])).any():
        raise ValueError("A scored tracker_row exceeds the tracker table length.")


def _reorder_tracks(
    tracks: list[pd.DataFrame], calls: pd.DataFrame
) -> list[pd.DataFrame]:
    """Permute each scored tracker row while keeping unscored rows unchanged."""
    # Copy the original tables and permute each scored row
    reordered = [track.copy(deep=True) for track in tracks]
    for tracker_row, group in calls.groupby(
        by="tracker_row",
        sort=False,
    ):
        for record in group.itertuples(index=False):
            source = tracks[record.fly - 1]
            destination = reordered[record.rank - 1]
            for column_position in range(len(source.columns)):
                destination.iat[tracker_row, column_position] = source.iat[
                    tracker_row, column_position
                ]
    # Return the reordered tracker tables
    return reordered


def _coverage_table(
    well_number: int, row_count: int, calls: pd.DataFrame
) -> pd.DataFrame:
    """Record whether each tracker data row has a complete score ranking."""
    # Initialize coverage for every tracker row
    coverage = pd.DataFrame(
        data={
            "well": well_number,
            "tracker_row": np.arange(
                row_count,
                dtype=np.int64,
            ),
            "frame": pd.array(
                data=[pd.NA] * row_count,
                dtype="Int64",
            ),
            "ranked": False,
        }
    )
    frames = calls[["tracker_row", "frame"]].drop_duplicates()
    positions = frames["tracker_row"].to_numpy(dtype=np.int64)
    coverage.loc[positions, "frame"] = frames["frame"].to_numpy(dtype=np.int64)
    coverage.loc[positions, "ranked"] = True
    # Return the complete row coverage table
    return coverage


def _prepare_well(
    session_directory: Path, well_number: int, calls: pd.DataFrame
) -> dict:
    """Load and validate the complete export for one well."""
    # Load and validate this well before writing outputs
    tracker_directory = find_tracker_directory(
        session_directory=session_directory,
        well_number=well_number,
    )
    calibration_path = find_calibration(tracker_directory=tracker_directory)
    tracks = read_tracks(tracker_directory=tracker_directory)
    _validate_tracks(tracks=tracks)
    _validate_frame_groups(
        calls=calls,
        tracks=tracks,
    )
    if not calibration_path.is_file():
        raise ValueError(f"Calibration file does not exist: {calibration_path}")

    # Return the collected results
    return {
        "well": well_number,
        "tracker_name": tracker_directory.name,
        "calibration_path": calibration_path,
        "tracks": _reorder_tracks(
            tracks=tracks,
            calls=calls,
        ),
        "coverage": _coverage_table(
            well_number=well_number,
            row_count=len(tracks[0]),
            calls=calls,
        ),
    }


def _write_well(prepared: dict, output_directory: Path) -> None:
    """Write one well's tracker tables and calibration."""
    # Create the well folder and save its tracker tables
    tracker_output = (
        output_directory / f"well{prepared['well']}" / prepared["tracker_name"]
    )
    tracker_output.mkdir(parents=True)
    for fly_number, track in enumerate(
        prepared["tracks"],
        start=1,
    ):
        track.to_csv(
            path_or_buf=tracker_output / f"fly{fly_number}.csv",
            index=False,
        )
    shutil.copy2(
        src=prepared["calibration_path"],
        dst=tracker_output / "calibration.mat",
    )
    (tracker_output / "README.txt").write_text(
        data=export_readme,
        encoding="utf-8",
    )


def export_ranked_tracks(
    session_directory: Path,
    calls: pd.DataFrame,
    output_directory: Path,
) -> dict:
    """Export all scored wells after validating every frame and destination."""
    # Resolve and validate the source and destination paths
    session_directory = Path(session_directory).resolve()
    output_directory = Path(output_directory).resolve()
    _validate_destination(
        session_directory=session_directory,
        output_directory=output_directory,
    )
    ranked = rank_scores(calls=calls)
    if ranked.empty:
        raise ValueError("At least one complete scored frame is required for export.")

    # Validate and prepare every scored well
    prepared_wells = []
    for well_number, group in ranked.groupby(
        by="well",
        sort=True,
    ):
        prepared_wells.append(
            _prepare_well(
                session_directory=session_directory,
                well_number=int(well_number),
                calls=group,
            )
        )

    # Define the exported mapping columns
    mapping_columns = base_mapping_columns.copy()
    if "confident" in ranked.columns:
        mapping_columns.append("confident")
    coverage = pd.concat(
        objs=[well["coverage"] for well in prepared_wells],
        ignore_index=True,
    )
    mapping_path = output_directory / "rank_mapping.csv"
    coverage_path = output_directory / "rank_coverage.csv"

    # Write the tracker copies, mapping, and coverage files
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    for prepared in prepared_wells:
        _write_well(
            prepared=prepared,
            output_directory=output_directory,
        )
    ranked[mapping_columns].to_csv(
        path_or_buf=mapping_path,
        index=False,
    )
    coverage.to_csv(
        path_or_buf=coverage_path,
        index=False,
    )
    (output_directory / "README.txt").write_text(
        data=export_readme,
        encoding="utf-8",
    )
    # Return the collected results
    return {
        "output_directory": str(output_directory),
        "well_count": len(prepared_wells),
        "ranked_rows": int(coverage["ranked"].sum()),
        "total_rows": len(coverage),
        "mapping_path": str(mapping_path),
        "coverage_path": str(coverage_path),
    }
