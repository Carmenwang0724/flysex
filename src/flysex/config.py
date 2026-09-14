"""Read recording paths and geometry from JSON."""

import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunConfig:
    session_directory: Path
    model_directory: Path
    output_directory: Path
    wells: tuple[int, ...]
    frame_zero: int
    well_origins: dict[int, tuple[float, float]]
    photos_directory: Path | None = None
    max_frames: int | None = None
    device: str = "auto"
    seed: int = 0
    still_width: int = 2592
    still_height: int = 1944
    panorama_width: int = 1944


def load_config(path):
    """Load a run configuration relative to its JSON file."""
    path = Path(path).resolve()
    values = json.loads(path.read_text())
    if not isinstance(values, dict):
        raise ValueError("The configuration must be a JSON object.")
    allowed = set(RunConfig.__dataclass_fields__)
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")

    # Check the row origin before reading any images
    frame_zero = values.get("frame_zero")
    if type(frame_zero) is not int:
        raise ValueError("Set frame_zero to the video frame corresponding to tracker row index 0.")
    for key in ("max_frames", "seed", "still_width", "still_height", "panorama_width"):
        if key not in values or (key == "max_frames" and values[key] is None):
            continue
        minimum = 0 if key == "seed" else 1
        if type(values[key]) is not int or values[key] < minimum:
            raise ValueError(f"{key} must be an integer at least {minimum}.")

    # Resolve each input and output path
    for key in ("session_directory", "model_directory", "output_directory", "photos_directory"):
        raw = values.get(key)
        if raw is None and key == "photos_directory":
            continue
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"Set {key} to a nonempty path.")
        location = Path(raw).expanduser()
        values[key] = (path.parent / location).resolve()
    for key in ("session_directory", "model_directory"):
        if not values[key].is_dir():
            raise ValueError(f"Directory does not exist: {values[key]}")
    output = values["output_directory"]
    for key in ("session_directory", "model_directory", "photos_directory"):
        source = values.get(key)
        if source and (output == source or output in source.parents or source in output.parents):
            raise ValueError(f"output_directory must be separate from {key}.")

    # Read selected wells and their crop origins
    origins = values.get("well_origins")
    wells = values.get("wells", [])
    if not isinstance(origins, dict) or not origins:
        raise ValueError("Set well_origins from the recording's arena crop rectangles.")
    parsed_origins = {}
    for key, point in origins.items():
        if not str(key).isdigit() or int(key) < 1:
            raise ValueError(f"Invalid well number: {key}")
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError(f"Well {key} requires an [x, y] origin.")
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in point):
            raise ValueError(f"Well {key} has a nonfinite origin.")
        parsed_origins[int(key)] = tuple(float(value) for value in point)
    if not isinstance(wells, list) or any(type(well) is not int or well not in parsed_origins for well in wells):
        raise ValueError("wells must list integer keys present in well_origins.")
    if len(set(wells)) != len(wells):
        raise ValueError("wells must not contain duplicates.")
    if values.get("device", "auto") not in ("auto", "cpu", "mps", "cuda"):
        raise ValueError("device must be auto, cpu, mps, or cuda.")
    values["wells"] = tuple(wells)
    values["well_origins"] = parsed_origins
    return RunConfig(**values)
