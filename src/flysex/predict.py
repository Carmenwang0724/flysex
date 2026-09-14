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
from .session import available_wells, find_calibration, find_tracker_directory, index_stills, read_pixels_per_mm, read_tracks


def score_well(config, well_number, networks, device, stills):
    tracker_directory = find_tracker_directory(session_directory=config.session_directory, well_number=well_number)
    tracks = read_tracks(tracker_directory=tracker_directory)
    pixels_per_mm = read_pixels_per_mm(calibration_path=find_calibration(tracker_directory=tracker_directory))
    rows = []
    skipped = []
    examples = []
    completed = 0
    output = config.output_directory
    for frame_number, path in stills.items():
        tracker_row = frame_number - config.frame_zero
        if tracker_row < 0 or tracker_row >= len(tracks[0]):
            skipped.append({"well": well_number, "frame": frame_number, "reason": "outside tracker rows"})
            continue
        tracked = np.asarray([
            [track[column].iloc[tracker_row] for column in ("pos x", "pos y", "ori")]
            for track in tracks
        ], dtype=np.float64)
        if not np.isfinite(tracked).all():
            skipped.append({"well": well_number, "frame": frame_number, "reason": "nonfinite tracking coordinates"})
            continue
        try:
            image = read_still(path=path, width=config.still_width, height=config.still_height)
        except (OSError, ValueError) as error:
            skipped.append({"well": well_number, "frame": frame_number, "reason": str(error)})
            continue
        crops, positions = crop_flies(
            image=image,
            tracked=tracked,
            origin=config.well_origins[well_number],
            scale_factor=config.still_width / config.panorama_width,
            pixels_per_mm=pixels_per_mm,
        )
        scores = predict_probabilities(networks=networks, crops=crops, device=device)
        for index, (position_x, position_y, _) in enumerate(tracked):
            score = float(scores[index])
            rows.append({
                "frame": frame_number, "well": well_number, "tracker_row": tracker_row,
                "fly": index + 1, "score": score, "sex": "M" if score > 0.5 else "F",
                "confident": bool(score < 0.1 or score > 0.9),
                "x": float(position_x), "y": float(position_y),
            })
        if completed < 3:
            order = np.argsort(-scores, kind="stable")
            for rank, index in enumerate(order, start=1):
                examples.append((crops[index], f"frame {frame_number}  rank {rank}\nfly {index + 1}  score {scores[index]:.3f}"))
            save_overlay(still_path=path, positions=positions, scores=scores, output_path=output / f"well{well_number}_frame{frame_number}_overlay.png")
        completed += 1
        if completed % 20 == 0:
            print(f"Well {well_number}: scored {completed} frames", flush=True)
        if config.max_frames is not None and completed >= config.max_frames:
            break
    if not rows:
        raise ValueError(f"Well {well_number}: no frames could be scored; check frame_zero and tracking coordinates.")
    save_review_sheet(examples=examples, output_path=output / f"well{well_number}_review.png")
    summary = {"well": well_number, "scored_frames": completed, "total_tracker_rows": len(tracks[0]), "skipped_frames": len(skipped)}
    return rows, skipped, summary


def run_prediction(config):
    if config.output_directory.exists():
        if not config.output_directory.is_dir() or any(config.output_directory.iterdir()):
            raise ValueError(f"Choose an empty output_directory: {config.output_directory}")
    wells = config.wells or tuple(available_wells(session_directory=config.session_directory))
    if any(well not in config.well_origins for well in wells):
        raise ValueError("Set well_origins for every selected well.")
    stills = index_stills(session_directory=config.session_directory, photos_directory=config.photos_directory)
    device = select_device(requested=config.device)
    torch.manual_seed(config.seed)
    networks, model_records = load_models(model_directory=config.model_directory, device=device)
    config.output_directory.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}; loaded {len(networks)} models", flush=True)
    rows = []
    skipped = []
    summaries = []
    for well in wells:
        well_rows, well_skipped, summary = score_well(config=config, well_number=well, networks=networks, device=device, stills=stills)
        rows.extend(well_rows)
        skipped.extend(well_skipped)
        summaries.append(summary)
        print(f"Well {well}: scored {summary['scored_frames']} frames", flush=True)
    ranked = rank_scores(calls=pd.DataFrame(rows))
    ranked.to_csv(config.output_directory / "ranked_scores.csv", index=False)
    pd.DataFrame(skipped, columns=["well", "frame", "reason"]).to_csv(config.output_directory / "skipped_frames.csv", index=False)
    summary = {
        "version": __version__, "config": asdict(config), "device": str(device), "models": model_records,
        "wells": summaries, "scored_flies": len(ranked), "confident_share": float(ranked["confident"].mean()),
        "ranking": "descending score; ties use original fly ID; no fixed male count",
        "tracker_row_formula": "frame - frame_zero",
    }
    (config.output_directory / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary
