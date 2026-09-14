"""Run prediction, score export, and video extraction."""

import argparse
from pathlib import Path


def make_parser():
    parser = argparse.ArgumentParser(prog="flysex", description="Rank tracked flies by model score at each sampled frame.")
    commands = parser.add_subparsers(dest="command", required=True)
    predict = commands.add_parser("predict", help="Score recording stills using a JSON configuration")
    predict.add_argument("--config", required=True, type=Path)
    export = commands.add_parser("export", help="Reorder scored tracker rows by descending score")
    export.add_argument("--session", required=True, type=Path)
    export.add_argument("--scores", required=True, type=Path)
    export.add_argument("--output", required=True, type=Path)
    extract = commands.add_parser("extract", help="Extract numbered PNG stills from a video")
    extract.add_argument("--video", required=True, type=Path)
    extract.add_argument("--output", required=True, type=Path)
    extract.add_argument("--stride", type=int, default=5)
    extract.add_argument("--maximum-frames", type=int)
    return parser


def main():
    parser = make_parser()
    arguments = parser.parse_args()
    try:
        if arguments.command == "predict":
            from .config import load_config
            from .predict import run_prediction

            summary = run_prediction(config=load_config(path=arguments.config))
            print(f"Saved scores for {summary['scored_flies']} flies")
        elif arguments.command == "export":
            import pandas as pd
            from .export import export_ranked_tracks

            result = export_ranked_tracks(
                session_directory=arguments.session,
                calls=pd.read_csv(arguments.scores),
                output_directory=arguments.output,
            )
            print(f"Exported ranked tracker tables to {arguments.output}")
        else:
            from .video import extract_stills

            extract_stills(
                video_path=arguments.video,
                output_directory=arguments.output,
                stride=arguments.stride,
                maximum_frames=arguments.maximum_frames,
            )
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.exit(status=2, message=f"flysex: {error}\n")
