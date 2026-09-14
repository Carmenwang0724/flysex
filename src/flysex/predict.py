"""Score selected recording frames and save their score ranks."""

import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch

from . import __version__
from .imaging import crop_flies, read_still, save_overlay, save_review_sheet
from .model import load_models, predict_probabilities, select_device
from .ranking import rank_scores
from .session import (
    available_wells,
    find_calibration,
    find_tracker_directory,
    index_stills,
    read_pixels_per_mm,
    read_tracks,
)


def read_frame(config, tracks, frame_number, path):
    """Read the aligned tracker positions and still for one frame."""
    # Map the video frame to its tracker row
    tracker_row = frame_number - config.frame_zero
    if tracker_row < 0 or tracker_row >= len(tracks[0]):
        # Return the collected results
        return None, None, tracker_row, "outside tracker rows"

    # Read all tracked coordinates
    tracked = np.asarray(
        a=[
            [track[column].iloc[tracker_row] for column in ("pos x", "pos y", "ori")]
            for track in tracks
        ],
        dtype=np.float64,
    )
    if not np.isfinite(tracked).all():
        # Return the collected results
        return None, None, tracker_row, "nonfinite tracking coordinates"

    # Decode the still and report damaged images
    try:
        image = read_still(
            path=path,
            width=config.still_width,
            height=config.still_height,
        )
    except (OSError, ValueError) as error:
        # Return the collected results
        return None, None, tracker_row, str(error)

    # Return the aligned frame inputs
    return tracked, image, tracker_row, None


def make_score_rows(tracked, scores, frame_number, well_number, tracker_row):
    """Record each fly's score, position, and descriptive sex call."""
    # Define the descriptive score thresholds
    decision_threshold = 0.5
    confident_low = 0.1
    confident_high = 0.9

    # Collect one record per original tracker ID
    rows = []
    for index, (position_x, position_y, _) in enumerate(tracked):
        score = float(scores[index])
        rows.append(
            {
                "frame": frame_number,
                "well": well_number,
                "tracker_row": tracker_row,
                "fly": index + 1,
                "score": score,
                "sex": "M" if score > decision_threshold else "F",
                "confident": bool(score < confident_low or score > confident_high),
                "x": float(position_x),
                "y": float(position_y),
            }
        )

    # Return the per-fly score records
    return rows


def save_frame_preview(config, frame_record, crops, positions, scores):
    """Save an overlay and collect crops in score order for one frame."""
    # Read the frame and well identifiers
    frame_number, well_number, still_path = frame_record
    output_path = (
        config.output_directory / f"well{well_number}_frame{frame_number}_overlay.png"
    )

    # Caption the crops in descending score order
    examples = []
    order = np.argsort(
        a=-scores,
        kind="stable",
    )
    for rank, index in enumerate(
        order,
        start=1,
    ):
        caption = (
            f"frame {frame_number}  rank {rank}\n"
            f"fly {index + 1}  score {scores[index]:.3f}"
        )
        examples.append((crops[index], caption))

    # Write the position overlay
    save_overlay(
        still_path=still_path,
        positions=positions,
        scores=scores,
        output_path=output_path,
    )

    # Return the review crops and captions
    return examples


def score_well(config, well_number, networks, device, stills):
    """Score the usable stills in one well and record skipped frames."""
    # Define the preview and progress intervals
    preview_frames = 3
    progress_interval = 20

    # Read this well's tracker tables and physical scale
    tracker_directory = find_tracker_directory(
        session_directory=config.session_directory,
        well_number=well_number,
    )
    tracks = read_tracks(tracker_directory=tracker_directory)
    calibration_path = find_calibration(tracker_directory=tracker_directory)
    pixels_per_mm = read_pixels_per_mm(calibration_path=calibration_path)

    # Initialize the score and review records
    rows = []
    skipped = []
    examples = []
    completed = 0

    # Read and score each aligned frame
    for frame_number, path in stills.items():
        tracked, image, tracker_row, reason = read_frame(
            config=config,
            tracks=tracks,
            frame_number=frame_number,
            path=path,
        )
        if reason is not None:
            skipped.append(
                {"well": well_number, "frame": frame_number, "reason": reason}
            )
            continue

        # Crop and score the complete tracked group
        crops, positions = crop_flies(
            image=image,
            tracked=tracked,
            origin=config.well_origins[well_number],
            scale_factor=config.still_width / config.panorama_width,
            pixels_per_mm=pixels_per_mm,
        )
        scores = predict_probabilities(
            networks=networks,
            crops=crops,
            device=device,
        )
        rows.extend(
            make_score_rows(
                tracked=tracked,
                scores=scores,
                frame_number=frame_number,
                well_number=well_number,
                tracker_row=tracker_row,
            )
        )

        # Save previews for the first few scored frames
        if completed < preview_frames:
            examples.extend(
                save_frame_preview(
                    config=config,
                    frame_record=(frame_number, well_number, path),
                    crops=crops,
                    positions=positions,
                    scores=scores,
                )
            )

        # Report progress and apply the frame limit
        completed += 1
        if completed % progress_interval == 0:
            print(
                f"Well {well_number}: scored {completed} frames",
                flush=True,
            )
        if config.max_frames is not None and completed >= config.max_frames:
            break

    # Require at least one scored frame
    if not rows:
        raise ValueError(
            f"Well {well_number}: no frames could be scored; "
            "check frame_zero and tracking coordinates."
        )

    # Save the review sheet and coverage summary
    save_review_sheet(
        examples=examples,
        output_path=config.output_directory / f"well{well_number}_review.png",
    )
    summary = {
        "well": well_number,
        "scored_frames": completed,
        "total_tracker_rows": len(tracks[0]),
        "skipped_frames": len(skipped),
    }

    # Return this well's scores, skipped frames, and coverage
    return rows, skipped, summary


def save_run(config, rows, skipped, summaries, model_records, device):
    """Save the ranked scores, skipped frames, and run provenance."""
    # Rank and save every scored fly
    ranked = rank_scores(calls=pd.DataFrame(data=rows))
    ranked.to_csv(
        path_or_buf=config.output_directory / "ranked_scores.csv",
        index=False,
    )

    # Save the skipped frames with their reasons
    skipped_table = pd.DataFrame(
        data=skipped,
        columns=["well", "frame", "reason"],
    )
    skipped_table.to_csv(
        path_or_buf=config.output_directory / "skipped_frames.csv",
        index=False,
    )

    # Record the configuration and checkpoint provenance
    summary = {
        "version": __version__,
        "config": asdict(obj=config),
        "device": str(device),
        "models": model_records,
        "wells": summaries,
        "scored_flies": len(ranked),
        "confident_share": float(ranked["confident"].mean()),
        "ranking": "descending score; ties use original fly ID; no fixed male count",
        "tracker_row_formula": "frame - frame_zero",
    }
    summary_path = config.output_directory / "run_summary.json"
    summary_path.write_text(
        data=json.dumps(
            obj=summary,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    # Return the run summary
    return summary


def run_prediction(config):
    """Load the ensemble and score each selected well."""
    # Require an empty output directory
    if config.output_directory.exists():
        if not config.output_directory.is_dir() or any(
            config.output_directory.iterdir()
        ):
            raise ValueError(
                f"Choose an empty output_directory: {config.output_directory}"
            )

    # Select wells and locate the stills
    wells = config.wells or tuple(
        available_wells(session_directory=config.session_directory)
    )
    if any(well not in config.well_origins for well in wells):
        raise ValueError("Set well_origins for every selected well.")
    stills = index_stills(
        session_directory=config.session_directory,
        photos_directory=config.photos_directory,
    )

    # Initialize the device and checkpoint ensemble
    device = select_device(requested=config.device)
    torch.manual_seed(seed=config.seed)
    networks, model_records = load_models(
        model_directory=config.model_directory,
        device=device,
    )
    config.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    print(
        f"Device: {device}; loaded {len(networks)} models",
        flush=True,
    )

    # Collect results from each selected well
    rows = []
    skipped = []
    summaries = []
    for well in wells:
        well_rows, well_skipped, summary = score_well(
            config=config,
            well_number=well,
            networks=networks,
            device=device,
            stills=stills,
        )
        rows.extend(well_rows)
        skipped.extend(well_skipped)
        summaries.append(summary)
        print(
            f"Well {well}: scored {summary['scored_frames']} frames",
            flush=True,
        )

    # Save and return the complete run
    return save_run(
        config=config,
        rows=rows,
        skipped=skipped,
        summaries=summaries,
        model_records=model_records,
        device=device,
    )
