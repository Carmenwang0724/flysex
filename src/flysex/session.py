"""Locate FlyTracker tables, calibration, and still images."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


def find_tracker_directory(session_directory, well_number):
    """Find the unique tracker folder for a requested well."""
    # Locate matching arena folders
    # Prepare the matches
    matches = sorted(
        path
        for path in Path(session_directory).rglob(
            pattern=f"*_arena_{well_number}-trackfeat.csv"
        )
        if path.is_dir()
    )

    # Require exactly one match
    if len(matches) != 1:
        raise ValueError(
            f"Expected one tracker folder for well {well_number}; found {len(matches)}."
        )

    # Return the matching tracker folder
    return matches[0]


def available_wells(session_directory):
    """List the well numbers with available tracker folders."""
    # Collect well numbers from tracker folder names
    # Prepare the wells
    wells = set()
    for path in Path(session_directory).rglob(pattern="*-trackfeat.csv"):
        match = re.search(
            pattern=r"_arena_(\d+)-trackfeat\.csv$",
            string=path.name,
        )
        if path.is_dir() and match:
            wells.add(int(match.group(1)))

    # Require at least one arena
    if not wells:
        raise ValueError(f"No FlyTracker folders found under {session_directory}.")

    # Return the well numbers in order
    return sorted(wells)


def find_calibration(tracker_directory):
    """Find a calibration file inside or beside a tracker folder."""
    # Search the tracker folder and its parent folders
    # Prepare the tracker directory
    tracker_directory = Path(tracker_directory)
    for directory in (
        tracker_directory,
        tracker_directory.parent,
        tracker_directory.parent.parent,
    ):
        candidate = directory / "calibration.mat"
        if candidate.is_file():
            # Return the candidate
            return candidate
    raise ValueError(f"No calibration.mat near {tracker_directory}.")


def read_pixels_per_mm(calibration_path):
    """Read and validate the physical pixel scale from calibration."""
    # Load the MATLAB calibration structure
    # Prepare the calibration
    calibration = loadmat(
        file_name=calibration_path,
        squeeze_me=True,
        struct_as_record=False,
    )["calib"]

    # Validate the physical pixel scale
    value = float(
        np.ravel(
            a=np.asarray(
                a=calibration.PPM,
                dtype=float,
            )
        )[0]
    )
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid pixels per millimetre in {calibration_path}.")

    # Return the pixels-per-millimetre value
    return value


def read_tracks(tracker_directory):
    """Read matching per-fly tracker tables in numeric ID order."""
    # Index the per-fly CSV paths
    # Prepare the numbered paths
    numbered_paths = []
    for path in Path(tracker_directory).glob(pattern="fly*.csv"):
        match = re.fullmatch(
            pattern=r"fly(\d+)\.csv",
            string=path.name,
        )
        if match:
            numbered_paths.append((int(match.group(1)), path))

    # Require consecutive fly IDs
    numbered_paths.sort()
    numbers = [number for number, _ in numbered_paths]
    if not numbers or numbers != list(
        range(
            1,
            len(numbers) + 1,
        )
    ):
        raise ValueError(
            f"Expected contiguous fly1.csv through flyN.csv in {tracker_directory}."
        )

    # Read the tables and validate their shape and coordinate columns
    tracks = [pd.read_csv(filepath_or_buffer=path) for _, path in numbered_paths]
    for track in tracks:
        if len(track) != len(tracks[0]) or list(track.columns) != list(
            tracks[0].columns
        ):
            raise ValueError(
                "FlyTracker tables must have the same row count and columns."
            )
        if not {"pos x", "pos y", "ori"}.issubset(track.columns):
            raise ValueError("FlyTracker tables need pos x, pos y, and ori columns.")

    # Return the validated tracker tables
    return tracks


def index_stills(session_directory, photos_directory=None):
    """Index the available still images by video frame number."""
    # Select the configured or standard image folders
    # Prepare the directories
    directories = (
        [Path(photos_directory)]
        if photos_directory
        else [
            Path(session_directory),
            Path(session_directory) / "photos",
            Path(session_directory) / "photos_full",
        ]
    )

    # Collect each frame path and reject duplicate frame numbers
    stills = {}
    for directory in directories:
        for path in directory.glob(pattern="frame_*.png"):
            match = re.match(
                pattern=r"frame_(\d+)_",
                string=path.name,
            )
            if not match:
                continue
            frame = int(match.group(1))
            if frame in stills:
                raise ValueError(
                    f"Duplicate frame {frame}; set photos_directory to one still folder."
                )
            stills[frame] = path

    # Require at least one matching image
    if not stills:
        raise ValueError("No frame_000123_HH-MM-SS.png stills found.")

    # Return the frame index in video order
    return dict(sorted(stills.items()))
