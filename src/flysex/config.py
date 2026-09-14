"""Read recording paths and geometry from JSON."""

import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunConfig:
    """Store the paths, frame origin, and geometry for one run."""

    # Define the recording configuration fields
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


def validate_numeric_settings(values):
    """Check the frame origin, image dimensions, seed, and frame limit."""
    # Define the allowed integer ranges
    minimum_values = {
        "max_frames": 1,
        "seed": 0,
        "still_width": 1,
        "still_height": 1,
        "panorama_width": 1,
    }

    # Require an explicit tracker row origin
    if type(values.get("frame_zero")) is not int:
        raise ValueError(
            "Set frame_zero to the video frame corresponding to tracker row index 0."
        )

    # Validate each supplied integer setting
    for name, minimum in minimum_values.items():
        if name not in values or (name == "max_frames" and values[name] is None):
            continue
        if type(values[name]) is not int or values[name] < minimum:
            raise ValueError(f"{name} must be an integer at least {minimum}.")


def resolve_directories(values, config_directory):
    """Resolve input and output folders relative to the configuration file."""
    # Define the path fields
    path_names = (
        "session_directory",
        "model_directory",
        "output_directory",
        "photos_directory",
    )

    # Resolve each supplied path
    for name in path_names:
        path_value = values.get(name)
        if path_value is None and name == "photos_directory":
            continue
        if (
            not isinstance(
                path_value,
                str,
            )
            or not path_value.strip()
        ):
            raise ValueError(f"Set {name} to a nonempty path.")
        location = Path(path_value).expanduser()
        values[name] = (config_directory / location).resolve()

    # Verify that all supplied source folders exist
    for name in ("session_directory", "model_directory", "photos_directory"):
        source = values.get(name)
        if source is not None and not source.is_dir():
            raise ValueError(f"Directory does not exist: {source}")

    # Keep the output outside the source folders
    output = values["output_directory"]
    for name in ("session_directory", "model_directory", "photos_directory"):
        source = values.get(name)
        if source is None:
            continue
        if output == source or output in source.parents or source in output.parents:
            raise ValueError(f"output_directory must be separate from {name}.")


def parse_well_origins(origins):
    """Validate and convert the arena crop origins."""
    # Require at least one configured well
    if (
        not isinstance(
            origins,
            dict,
        )
        or not origins
    ):
        raise ValueError("Set well_origins from the recording's arena crop rectangles.")

    # Convert each well number and its coordinate pair
    parsed_origins = {}
    for name, point in origins.items():
        if not str(name).isdigit() or int(name) < 1:
            raise ValueError(f"Invalid well number: {name}")
        if (
            not isinstance(
                point,
                list,
            )
            or len(point) != 2
        ):
            raise ValueError(f"Well {name} requires an [x, y] origin.")
        if any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in point
        ):
            raise ValueError(f"Well {name} has a nonfinite origin.")
        well_number = int(name)
        if well_number in parsed_origins:
            raise ValueError(f"Duplicate well number: {well_number}")
        parsed_origins[well_number] = tuple(float(value) for value in point)

    # Return the numeric crop origins
    return parsed_origins


def validate_selection(values):
    """Validate the selected wells and compute device."""
    # Parse the arena coordinates
    origins = parse_well_origins(origins=values.get("well_origins"))
    wells = values.get(
        "wells",
        [],
    )

    # Check that each selected well has a crop origin
    if not isinstance(
        wells,
        list,
    ) or any(type(well) is not int or well not in origins for well in wells):
        raise ValueError("wells must list integer keys present in well_origins.")
    if len(set(wells)) != len(wells):
        raise ValueError("wells must not contain duplicates.")
    if values.get(
        "device",
        "auto",
    ) not in ("auto", "cpu", "mps", "cuda"):
        raise ValueError("device must be auto, cpu, mps, or cuda.")

    # Store the validated selection
    values["wells"] = tuple(wells)
    values["well_origins"] = origins


def load_config(path):
    """Load a run configuration relative to its JSON file."""
    # Read the configuration object
    path = Path(path).resolve()
    values = json.loads(s=path.read_text())
    if not isinstance(
        values,
        dict,
    ):
        raise ValueError("The configuration must be a JSON object.")

    # Reject unrecognized settings
    allowed_names = set(RunConfig.__dataclass_fields__)
    unknown_names = set(values) - allowed_names
    if unknown_names:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown_names)}")

    # Validate values and resolve local paths
    validate_numeric_settings(values=values)
    resolve_directories(
        values=values,
        config_directory=path.parent,
    )
    validate_selection(values=values)

    # Return the run configuration
    return RunConfig(**values)
