"""Check prediction orchestration without running a neural network."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from flysex.config import RunConfig
from flysex.predict import run_prediction, score_well


class PredictionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.session = self.root / "session"
        self.session.mkdir()
        self.config = RunConfig(
            session_directory=self.session,
            model_directory=self.root / "model",
            output_directory=self.root / "output",
            wells=(2,),
            frame_zero=100,
            well_origins={2: (10, 20)},
            device="cpu",
        )
        self.tracks = [
            pd.DataFrame(
                {
                    "pos x": np.arange(30, dtype=np.float64) + fly_number * 100,
                    "pos y": np.arange(30, dtype=np.float64) + fly_number * 50,
                    "ori": np.full(30, fly_number / 10, dtype=np.float64),
                }
            )
            for fly_number in (1, 2)
        ]
        self.image = np.ones((10, 10), dtype=np.float32)
        self.crops = [np.ones((96, 96), dtype=np.float32)] * 2
        self.start_patch("find_tracker_directory", return_value=self.session / "tracker")
        self.start_patch("find_calibration", return_value=self.session / "calibration.mat")
        self.start_patch("read_pixels_per_mm", return_value=15.7)
        self.start_patch("read_tracks", return_value=self.tracks)
        self.read_image = self.start_patch("read_still", return_value=self.image)
        self.crop_images = self.start_patch(
            "crop_flies",
            return_value=(self.crops, [(10, 10), (20, 20)]),
        )
        self.predict = self.start_patch(
            "predict_probabilities",
            return_value=np.asarray([0.4, 0.6], dtype=np.float32),
        )
        self.start_patch("save_overlay")
        self.start_patch("save_review_sheet")

    def start_patch(self, name: str, **options):
        replacement = patch(f"flysex.predict.{name}", **options)
        mocked = replacement.start()
        self.addCleanup(replacement.stop)
        return mocked

    def stills(self, frames: list[int]) -> dict[int, Path]:
        return {
            frame: self.session / f"frame_{frame:06d}_00-00-00.png"
            for frame in frames
        }

    def score(self, frames: list[int], config: RunConfig | None = None):
        return score_well(
            config=config or self.config,
            well_number=2,
            networks=[],
            device="cpu",
            stills=self.stills(frames=frames),
        )

    def test_frame_origin_does_not_follow_first_retained_still(self):
        rows, skipped, summary = self.score(frames=[120, 123])
        self.assertEqual([row["tracker_row"] for row in rows], [20, 20, 23, 23])
        self.assertEqual([row["x"] for row in rows], [120, 220, 123, 223])
        self.assertEqual(summary["scored_frames"], 2)
        self.assertEqual(skipped, [])
        np.testing.assert_array_equal(
            self.crop_images.call_args_list[0].kwargs["tracked"][:, 0],
            [120, 220],
        )

    def test_corrupt_images_are_recorded_and_limit_counts_successful_frames(self):
        self.read_image.side_effect = [
            self.image,
            ValueError("contains an empty horizontal band"),
            OSError("image file is truncated"),
            self.image,
        ]
        limited = replace(self.config, max_frames=2)
        rows, skipped, summary = self.score(frames=[120, 121, 122, 123, 124], config=limited)
        self.assertEqual([row["frame"] for row in rows], [120, 120, 123, 123])
        self.assertEqual(summary["scored_frames"], 2)
        self.assertEqual(summary["skipped_frames"], 2)
        self.assertEqual([record["frame"] for record in skipped], [121, 122])
        self.assertIn("empty horizontal band", skipped[0]["reason"])
        self.assertIn("truncated", skipped[1]["reason"])
        self.assertEqual(self.read_image.call_count, 4)
        self.assertEqual(self.predict.call_count, 2)

    def test_nonfinite_coordinates_skip_the_complete_frame(self):
        self.tracks[0].loc[20, "pos y"] = np.nan
        self.tracks[1].loc[21, "ori"] = np.inf
        self.tracks[0].loc[22, "pos x"] = np.nan
        rows, skipped, summary = self.score(frames=[120, 121, 122, 123])
        self.assertEqual([row["frame"] for row in rows], [123, 123])
        self.assertEqual([record["frame"] for record in skipped], [120, 121, 122])
        self.assertEqual(summary["skipped_frames"], 3)
        self.assertEqual(self.read_image.call_count, 1)
        self.assertEqual(self.predict.call_count, 1)

    def test_out_of_range_frames_skip_before_reading_images(self):
        rows, skipped, summary = self.score(frames=[99, 120, 130])
        self.assertEqual([row["tracker_row"] for row in rows], [20, 20])
        self.assertEqual([record["frame"] for record in skipped], [99, 130])
        self.assertEqual(summary["skipped_frames"], 2)
        self.assertEqual(self.read_image.call_count, 1)

    def test_no_scorable_frames_fail_clearly(self):
        self.read_image.side_effect = ValueError("contains an empty horizontal band")
        with self.assertRaisesRegex(ValueError, "no frames could be scored"):
            self.score(frames=[120, 121])
        self.predict.assert_not_called()

    def test_run_saves_skipped_frames_and_matching_summary(self):
        self.start_patch("index_stills", return_value=self.stills(frames=[120, 121, 123]))
        self.start_patch("load_models", return_value=([], []))
        self.read_image.side_effect = [self.image, OSError("broken PNG"), self.image]
        summary = run_prediction(config=self.config)
        saved_summary = json.loads((self.config.output_directory / "run_summary.json").read_text())
        skipped = pd.read_csv(self.config.output_directory / "skipped_frames.csv")
        ranked = pd.read_csv(self.config.output_directory / "ranked_scores.csv")
        self.assertEqual(summary["wells"][0]["skipped_frames"], 1)
        self.assertEqual(saved_summary["wells"], summary["wells"])
        self.assertEqual(skipped["frame"].tolist(), [121])
        self.assertEqual(skipped["reason"].tolist(), ["broken PNG"])
        self.assertEqual(ranked["tracker_row"].tolist(), [20, 20, 23, 23])
        self.assertEqual(ranked["fly"].tolist(), [2, 1, 2, 1])
        self.assertEqual(summary["scored_flies"], len(ranked))


if __name__ == "__main__":
    unittest.main()
