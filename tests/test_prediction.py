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
        """Prepare the temporary inputs for each test."""
        # Prepare the temporary
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
                data={
                    "pos x": np.arange(
                        30,
                        dtype=np.float64,
                    )
                    + fly_number * 100,
                    "pos y": np.arange(
                        30,
                        dtype=np.float64,
                    )
                    + fly_number * 50,
                    "ori": np.full(
                        shape=30,
                        fill_value=fly_number / 10,
                        dtype=np.float64,
                    ),
                }
            )
            for fly_number in (1, 2)
        ]
        self.image = np.ones(
            shape=(10, 10),
            dtype=np.float32,
        )
        self.crops = [
            np.ones(
                shape=(96, 96),
                dtype=np.float32,
            )
        ] * 2
        self.start_patch(
            name="find_tracker_directory",
            return_value=self.session / "tracker",
        )
        self.start_patch(
            name="find_calibration",
            return_value=self.session / "calibration.mat",
        )
        self.start_patch(
            name="read_pixels_per_mm",
            return_value=15.7,
        )
        self.start_patch(
            name="read_tracks",
            return_value=self.tracks,
        )
        self.read_image = self.start_patch(
            name="read_still",
            return_value=self.image,
        )
        self.crop_images = self.start_patch(
            name="crop_flies",
            return_value=(self.crops, [(10, 10), (20, 20)]),
        )
        self.predict = self.start_patch(
            name="predict_probabilities",
            return_value=np.asarray(
                a=[0.4, 0.6],
                dtype=np.float32,
            ),
        )
        self.start_patch(name="save_overlay")
        self.start_patch(name="save_review_sheet")

    def start_patch(self, name: str, **options):
        """Replace one dependency for a prediction test."""
        # Prepare the replacement
        replacement = patch(
            f"flysex.predict.{name}",
            **options,
        )
        mocked = replacement.start()
        self.addCleanup(replacement.stop)
        # Return the mocked
        return mocked

    def stills(self, frames: list[int]) -> dict[int, Path]:
        """Build the frame-to-image lookup for a test."""
        # Return the result
        return {
            frame: self.session / f"frame_{frame:06d}_00-00-00.png" for frame in frames
        }

    def score(self, frames: list[int], config: RunConfig | None = None):
        """Run scoring with the prepared test inputs."""
        # Return the result
        return score_well(
            config=config or self.config,
            well_number=2,
            networks=[],
            device="cpu",
            stills=self.stills(frames=frames),
        )

    def test_frame_origin_does_not_follow_first_retained_still(self):
        """Check frame origin does not follow first retained still."""
        # Prepare the rows, skipped, summary
        rows, skipped, summary = self.score(frames=[120, 123])
        self.assertEqual(
            first=[row["tracker_row"] for row in rows],
            second=[20, 20, 23, 23],
        )
        self.assertEqual(
            first=[row["x"] for row in rows],
            second=[120, 220, 123, 223],
        )
        self.assertEqual(
            first=summary["scored_frames"],
            second=2,
        )
        self.assertEqual(
            first=skipped,
            second=[],
        )
        np.testing.assert_array_equal(
            actual=self.crop_images.call_args_list[0].kwargs["tracked"][:, 0],
            desired=[120, 220],
        )

    def test_corrupt_images_are_recorded_and_limit_counts_successful_frames(self):
        """Check corrupt images are recorded and limit counts successful frames."""
        # Prepare the read image.side effect
        self.read_image.side_effect = [
            self.image,
            ValueError("contains an empty horizontal band"),
            OSError("image file is truncated"),
            self.image,
        ]
        limited = replace(
            self.config,
            max_frames=2,
        )
        rows, skipped, summary = self.score(
            frames=[120, 121, 122, 123, 124],
            config=limited,
        )
        self.assertEqual(
            first=[row["frame"] for row in rows],
            second=[120, 120, 123, 123],
        )
        self.assertEqual(
            first=summary["scored_frames"],
            second=2,
        )
        self.assertEqual(
            first=summary["skipped_frames"],
            second=2,
        )
        self.assertEqual(
            first=[record["frame"] for record in skipped],
            second=[121, 122],
        )
        self.assertIn(
            member="empty horizontal band",
            container=skipped[0]["reason"],
        )
        self.assertIn(
            member="truncated",
            container=skipped[1]["reason"],
        )
        self.assertEqual(
            first=self.read_image.call_count,
            second=4,
        )
        self.assertEqual(
            first=self.predict.call_count,
            second=2,
        )

    def test_nonfinite_coordinates_skip_the_complete_frame(self):
        """Check nonfinite coordinates skip the complete frame."""
        # Store the computed values
        self.tracks[0].loc[20, "pos y"] = np.nan
        self.tracks[1].loc[21, "ori"] = np.inf
        self.tracks[0].loc[22, "pos x"] = np.nan
        rows, skipped, summary = self.score(frames=[120, 121, 122, 123])
        self.assertEqual(
            first=[row["frame"] for row in rows],
            second=[123, 123],
        )
        self.assertEqual(
            first=[record["frame"] for record in skipped],
            second=[120, 121, 122],
        )
        self.assertEqual(
            first=summary["skipped_frames"],
            second=3,
        )
        self.assertEqual(
            first=self.read_image.call_count,
            second=1,
        )
        self.assertEqual(
            first=self.predict.call_count,
            second=1,
        )

    def test_out_of_range_frames_skip_before_reading_images(self):
        """Check out of range frames skip before reading images."""
        # Prepare the rows, skipped, summary
        rows, skipped, summary = self.score(frames=[99, 120, 130])
        self.assertEqual(
            first=[row["tracker_row"] for row in rows],
            second=[20, 20],
        )
        self.assertEqual(
            first=[record["frame"] for record in skipped],
            second=[99, 130],
        )
        self.assertEqual(
            first=summary["skipped_frames"],
            second=2,
        )
        self.assertEqual(
            first=self.read_image.call_count,
            second=1,
        )

    def test_no_scorable_frames_fail_clearly(self):
        """Check no scorable frames fail clearly."""
        # Prepare the read image.side effect
        self.read_image.side_effect = ValueError("contains an empty horizontal band")
        with self.assertRaisesRegex(
            expected_exception=ValueError,
            expected_regex="no frames could be scored",
        ):
            self.score(frames=[120, 121])
        self.predict.assert_not_called()

    def test_run_saves_skipped_frames_and_matching_summary(self):
        """Check run saves skipped frames and matching summary."""
        # Process the next step
        self.start_patch(
            name="index_stills",
            return_value=self.stills(frames=[120, 121, 123]),
        )
        self.start_patch(
            name="load_models",
            return_value=([], []),
        )
        self.read_image.side_effect = [self.image, OSError("broken PNG"), self.image]
        summary = run_prediction(config=self.config)
        saved_summary = json.loads(
            s=(self.config.output_directory / "run_summary.json").read_text()
        )
        skipped = pd.read_csv(
            filepath_or_buffer=self.config.output_directory / "skipped_frames.csv"
        )
        ranked = pd.read_csv(
            filepath_or_buffer=self.config.output_directory / "ranked_scores.csv"
        )
        self.assertEqual(
            first=summary["wells"][0]["skipped_frames"],
            second=1,
        )
        self.assertEqual(
            first=saved_summary["wells"],
            second=summary["wells"],
        )
        self.assertEqual(
            first=skipped["frame"].tolist(),
            second=[121],
        )
        self.assertEqual(
            first=skipped["reason"].tolist(),
            second=["broken PNG"],
        )
        self.assertEqual(
            first=ranked["tracker_row"].tolist(),
            second=[20, 20, 23, 23],
        )
        self.assertEqual(
            first=ranked["fly"].tolist(),
            second=[2, 1, 2, 1],
        )
        self.assertEqual(
            first=summary["scored_flies"],
            second=len(ranked),
        )

    def test_run_saves_male_positions_in_millimetres(self):
        """Check the run writes male positions scaled by the calibration value."""
        # Run the prediction with one unreadable still
        self.start_patch(
            name="index_stills",
            return_value=self.stills(frames=[120, 121, 123]),
        )
        self.start_patch(
            name="load_models",
            return_value=([], []),
        )
        self.read_image.side_effect = [self.image, OSError("broken PNG"), self.image]
        summary = run_prediction(config=self.config)
        positions = pd.read_csv(
            filepath_or_buffer=self.config.output_directory / "male_positions.csv"
        )
        saved_summary = json.loads(
            s=(self.config.output_directory / "run_summary.json").read_text()
        )

        # Check the rows, pixel pass-through, and millimetre conversion
        self.assertEqual(
            first=positions["frame"].tolist(),
            second=[120, 123],
        )
        self.assertEqual(
            first=positions["fly"].tolist(),
            second=[2, 2],
        )
        self.assertEqual(
            first=positions["x_px"].tolist(),
            second=[220.0, 223.0],
        )
        np.testing.assert_allclose(
            actual=positions["x_mm"].to_numpy(),
            desired=[round(220 / 15.7, 2), round(223 / 15.7, 2)],
        )
        np.testing.assert_allclose(
            actual=positions["y_mm"].to_numpy(),
            desired=[round(120 / 15.7, 2), round(123 / 15.7, 2)],
        )

        # Check the saved counts and scale
        self.assertEqual(
            first=summary["male_rows"],
            second=2,
        )
        self.assertEqual(
            first=saved_summary["male_calls"],
            second=[
                {
                    "well": 2,
                    "scored_frames": 2,
                    "frames_with_no_male": 0,
                    "frames_with_one_male": 2,
                    "frames_with_two_or_more": 0,
                }
            ],
        )
        self.assertEqual(
            first=saved_summary["wells"][0]["pixels_per_mm"],
            second=15.7,
        )


# Run the test suite
if __name__ == "__main__":
    unittest.main()
