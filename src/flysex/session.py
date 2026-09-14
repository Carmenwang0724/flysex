"""Locate FlyTracker tables, calibration, and still images."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


def find_tracker_directory(session_directory, well_number):
    matches = sorted(
        path for path in Path(session_directory).rglob(f"*_arena_{well_number}-trackfeat.csv")
        if path.is_dir()
    )
    if len(matches) != 1:
        raise ValueError(f"Expected one tracker folder for well {well_number}; found {len(matches)}.")
    return matches[0]


def available_wells(session_directory):
    wells = set()
    for path in Path(session_directory).rglob("*-trackfeat.csv"):
        match = re.search(r"_arena_(\d+)-trackfeat\.csv$", path.name)
        if path.is_dir() and match:
            wells.add(int(match.group(1)))
    if not wells:
        raise ValueError(f"No FlyTracker folders found under {session_directory}.")
    return sorted(wells)


def find_calibration(tracker_directory):
    tracker_directory = Path(tracker_directory)
    for directory in (tracker_directory, tracker_directory.parent, tracker_directory.parent.parent):
        candidate = directory / "calibration.mat"
        if candidate.is_file():
            return candidate
    raise ValueError(f"No calibration.mat near {tracker_directory}.")


def read_pixels_per_mm(calibration_path):
    calibration = loadmat(calibration_path, squeeze_me=True, struct_as_record=False)["calib"]
    value = float(np.ravel(np.asarray(calibration.PPM, dtype=float))[0])
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid pixels per millimetre in {calibration_path}.")
    return value


def read_tracks(tracker_directory):
    numbered_paths = []
    for path in Path(tracker_directory).glob("fly*.csv"):
        match = re.fullmatch(r"fly(\d+)\.csv", path.name)
        if match:
            numbered_paths.append((int(match.group(1)), path))
    numbered_paths.sort()
    numbers = [number for number, _ in numbered_paths]
    if not numbers or numbers != list(range(1, len(numbers) + 1)):
        raise ValueError(f"Expected contiguous fly1.csv through flyN.csv in {tracker_directory}.")
    tracks = [pd.read_csv(path) for _, path in numbered_paths]
    for track in tracks:
        if len(track) != len(tracks[0]) or list(track.columns) != list(tracks[0].columns):
            raise ValueError("FlyTracker tables must have the same row count and columns.")
        if not {"pos x", "pos y", "ori"}.issubset(track.columns):
            raise ValueError("FlyTracker tables need pos x, pos y, and ori columns.")
    return tracks


def index_stills(session_directory, photos_directory=None):
    directories = [Path(photos_directory)] if photos_directory else [
        Path(session_directory), Path(session_directory) / "photos", Path(session_directory) / "photos_full"
    ]
    stills = {}
    for directory in directories:
        for path in directory.glob("frame_*.png"):
            match = re.match(r"frame_(\d+)_", path.name)
            if not match:
                continue
            frame = int(match.group(1))
            if frame in stills:
                raise ValueError(f"Duplicate frame {frame}; set photos_directory to one still folder.")
            stills[frame] = path
    if not stills:
        raise ValueError("No frame_000123_HH-MM-SS.png stills found.")
    return dict(sorted(stills.items()))
