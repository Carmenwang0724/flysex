"""Check the male position table and its per-frame counts."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from flysex.positions import count_male_calls, male_positions, write_male_positions


def make_calls() -> pd.DataFrame:
    """Create scored frames in two wells with zero, one, and two flies called male."""
    # Return the result
    return pd.DataFrame(
        data={
            "frame": [100, 100, 100, 105, 105, 105, 100, 100, 100],
            "well": [2, 2, 2, 2, 2, 2, 6, 6, 6],
            "fly": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "tracker_row": [1, 1, 1, 6, 6, 6, 1, 1, 1],
            "score": [
                0.10,
                0.9964911341667175,
                0.02,
                0.30,
                0.40,
                0.20,
                0.95,
                0.70,
                0.05,
            ],
            "sex": ["F", "M", "F", "F", "F", "F", "M", "M", "F"],
            "confident": [False, True, True, False, False, False, True, False, True],
            "x": [10.0, 160.29, 30.0, 11.0, 21.0, 31.0, 79.0, 158.0, 237.0],
            "y": [40.0, 80.1435, 60.0, 41.0, 51.0, 61.0, 79.0, 237.0, 316.0],
        }
    )


class MalePositionTests(unittest.TestCase):
    def setUp(self):
        """Prepare the scores and the per-well physical scales."""
        # Prepare the scores and scales
        self.calls = make_calls()
        self.scales = {2: 16.0287, 6: 15.8}

    def test_lists_only_flies_called_male(self):
        """Check the table keeps exactly the flies whose sex call is male."""
        # Prepare the positions
        positions = male_positions(
            calls=self.calls,
            pixels_per_mm=self.scales,
        )
        self.assertEqual(
            first=list(zip(positions["well"], positions["frame"], positions["fly"])),
            second=[(2, 100, 2), (6, 100, 1), (6, 100, 2)],
        )
        self.assertEqual(
            first=positions.columns.tolist(),
            second=[
                "frame",
                "well",
                "fly",
                "x_mm",
                "y_mm",
                "score",
                "confident",
                "rank",
                "x_px",
                "y_px",
                "tracker_row",
            ],
        )

    def test_converts_pixels_with_each_wells_own_scale(self):
        """Check each well's pixel positions are divided by that well's scale."""
        # Prepare the positions
        positions = male_positions(
            calls=self.calls,
            pixels_per_mm=self.scales,
        )
        np.testing.assert_allclose(
            actual=positions["x_mm"].to_numpy(),
            desired=[
                round(160.29 / 16.0287, 2),
                round(79.0 / 15.8, 2),
                round(158.0 / 15.8, 2),
            ],
        )
        np.testing.assert_allclose(
            actual=positions["y_mm"].to_numpy(),
            desired=[
                round(80.1435 / 16.0287, 2),
                round(79.0 / 15.8, 2),
                round(237.0 / 15.8, 2),
            ],
        )

    def test_keeps_tracker_pixels_and_rounds_only_the_readable_columns(self):
        """Check pixel positions pass through unchanged while scores are rounded."""
        # Prepare the positions
        positions = male_positions(
            calls=self.calls,
            pixels_per_mm=self.scales,
        )
        self.assertEqual(
            first=positions["x_px"].tolist(),
            second=[160.29, 79.0, 158.0],
        )
        self.assertEqual(
            first=positions["y_px"].tolist(),
            second=[80.1435, 79.0, 237.0],
        )
        self.assertEqual(
            first=positions["score"].tolist(),
            second=[0.9965, 0.95, 0.7],
        )

    def test_ranks_match_the_unrounded_scores(self):
        """Check near-tied scores keep the rank order of the full-precision scores."""
        # Prepare near-tied male scores that round to the same value
        calls = self.calls.copy()
        calls.loc[calls["well"] == 6, "score"] = [0.700041, 0.700049, 0.05]
        positions = male_positions(
            calls=calls,
            pixels_per_mm=self.scales,
        )
        well_six = positions[positions["well"] == 6]
        self.assertEqual(
            first=well_six["fly"].tolist(),
            second=[2, 1],
        )
        self.assertEqual(
            first=well_six["rank"].tolist(),
            second=[1, 2],
        )
        self.assertEqual(
            first=well_six["score"].tolist(),
            second=[0.7, 0.7],
        )

    def test_leaves_the_score_table_unchanged(self):
        """Check building positions does not modify the full-precision score table."""
        # Prepare the original copy
        original = self.calls.copy(deep=True)
        male_positions(
            calls=self.calls,
            pixels_per_mm=self.scales,
        )
        pd.testing.assert_frame_equal(
            left=self.calls,
            right=original,
        )

    def test_frames_without_a_male_call_give_no_rows(self):
        """Check a frame where no fly is called male contributes no position rows."""
        # Prepare the positions
        positions = male_positions(
            calls=self.calls,
            pixels_per_mm=self.scales,
        )
        self.assertNotIn(
            member=105,
            container=positions["frame"].tolist(),
        )

    def test_empty_male_calls_give_an_empty_table_with_columns(self):
        """Check a recording with no male calls still writes the table header."""
        # Prepare all-female calls
        calls = self.calls.copy()
        calls["sex"] = "F"
        positions = male_positions(
            calls=calls,
            pixels_per_mm=self.scales,
        )
        self.assertTrue(expr=positions.empty)
        self.assertIn(
            member="x_mm",
            container=positions.columns.tolist(),
        )

    def test_rejects_a_well_without_a_scale(self):
        """Check a scored well missing from the scale lookup fails clearly."""
        # Prepare the incomplete scales
        with self.assertRaisesRegex(
            expected_exception=ValueError,
            expected_regex="No pixels-per-millimetre value for wells \\[6\\]",
        ):
            male_positions(
                calls=self.calls,
                pixels_per_mm={2: 16.0287},
            )

    def test_rejects_invalid_scales(self):
        """Check zero, negative, and nonfinite scales are rejected."""
        # Process each invalid scale
        for value in (0.0, -15.7, np.nan, np.inf):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    expected_exception=ValueError,
                    expected_regex="invalid pixels-per-millimetre",
                ):
                    male_positions(
                        calls=self.calls,
                        pixels_per_mm={2: 16.0287, 6: value},
                    )

    def test_rejects_missing_columns_and_unknown_sex_values(self):
        """Check score tables without positions or with unknown sex calls fail."""
        # Check the missing column
        with self.assertRaisesRegex(
            expected_exception=ValueError,
            expected_regex="missing columns: x",
        ):
            male_positions(
                calls=self.calls.drop(columns=["x"]),
                pixels_per_mm=self.scales,
            )

        # Check the unknown sex value
        calls = self.calls.copy()
        calls.loc[0, "sex"] = "male"
        with self.assertRaisesRegex(
            expected_exception=ValueError,
            expected_regex="sex must contain only M or F",
        ):
            male_positions(
                calls=calls,
                pixels_per_mm=self.scales,
            )


class MaleCountTests(unittest.TestCase):
    def test_counts_frames_by_number_of_male_calls(self):
        """Check each well reports its frames with zero, one, and several male calls."""
        # Prepare the counts
        counts = count_male_calls(calls=make_calls())
        self.assertEqual(
            first=counts,
            second=[
                {
                    "well": 2,
                    "scored_frames": 2,
                    "frames_with_no_male": 1,
                    "frames_with_one_male": 1,
                    "frames_with_two_or_more": 0,
                },
                {
                    "well": 6,
                    "scored_frames": 1,
                    "frames_with_no_male": 0,
                    "frames_with_one_male": 0,
                    "frames_with_two_or_more": 1,
                },
            ],
        )


class WritePositionTests(unittest.TestCase):
    def setUp(self):
        """Prepare a temporary output folder and patched calibration lookup."""
        # Prepare the temporary folder
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

        # Replace the calibration lookup with fixed scales
        replacement = patch(
            "flysex.positions.read_well_scales",
            return_value={2: 16.0287, 6: 15.8},
        )
        self.read_scales = replacement.start()
        self.addCleanup(replacement.stop)

    def test_writes_the_table_and_reports_its_counts(self):
        """Check the saved table matches the computed positions."""
        # Prepare the output
        output_path = self.root / "out" / "male_positions.csv"
        result = write_male_positions(
            session_directory=self.root,
            calls=make_calls(),
            output_path=output_path,
        )
        saved = pd.read_csv(filepath_or_buffer=output_path)
        self.assertEqual(
            first=result["male_rows"],
            second=3,
        )
        self.assertEqual(
            first=len(saved),
            second=3,
        )
        self.assertEqual(
            first=saved["x_px"].tolist(),
            second=[160.29, 79.0, 158.0],
        )
        self.assertEqual(
            first=self.read_scales.call_args.kwargs["wells"],
            second=[2, 6],
        )

    def test_refuses_to_replace_an_existing_file(self):
        """Check an existing output file is never overwritten."""
        # Prepare the existing file
        output_path = self.root / "male_positions.csv"
        output_path.write_text(
            data="keep\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            expected_exception=ValueError,
            expected_regex="already exists",
        ):
            write_male_positions(
                session_directory=self.root,
                calls=make_calls(),
                output_path=output_path,
            )
        self.assertEqual(
            first=output_path.read_text(encoding="utf-8"),
            second="keep\n",
        )


# Run the test suite
if __name__ == "__main__":
    unittest.main()
