"""Check per-frame ranking and coverage-aware tracker export."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from flysex.export import export_ranked_tracks
from flysex.ranking import rank_scores


def make_calls(well_number: int = 1) -> pd.DataFrame:
    """Create two complete scored frames with distinct tracker row positions."""
    # Return the result
    return pd.DataFrame(
        data={
            "frame": [100, 100, 100, 105, 105, 105],
            "well": [well_number] * 6,
            "fly": [1, 3, 2, 2, 1, 3],
            "tracker_row": [1, 1, 1, 4, 4, 4],
            "score": [0.1, 0.9, 0.9, 0.1, 0.2, 0.4],
            "confident": [True, True, False, True, False, False],
            "note": ["keep"] * 6,
        }
    )


class RankingTests(unittest.TestCase):
    def test_ranks_all_flies_with_deterministic_ties(self):
        """Check ranks all flies with deterministic ties."""
        # Prepare the ranked
        ranked = rank_scores(calls=make_calls())
        self.assertEqual(
            first=ranked["fly"].tolist(),
            second=[2, 3, 1, 3, 1, 2],
        )
        self.assertEqual(
            first=ranked["rank"].tolist(),
            second=[1, 2, 3, 1, 2, 3],
        )
        self.assertEqual(
            first=ranked["confident"].tolist(),
            second=[False, True, True, False, False, True],
        )
        np.testing.assert_allclose(
            actual=ranked["rank_gap"].to_numpy(),
            desired=[0.0, 0.8, np.nan, 0.2, 0.1, np.nan],
            equal_nan=True,
        )
        self.assertEqual(
            first=ranked["note"].tolist(),
            second=["keep"] * 6,
        )

    def test_keeps_input_unchanged_and_ranks_each_well(self):
        """Check keeps input unchanged and ranks each well."""
        # Prepare the calls
        calls = pd.concat(
            objs=[make_calls(well_number=2), make_calls()],
            ignore_index=True,
        )
        original = calls.copy(deep=True)
        ranked = rank_scores(calls=calls)
        pd.testing.assert_frame_equal(
            left=calls,
            right=original,
        )
        self.assertEqual(
            first=ranked.groupby(by=["well", "frame"])["rank"].max().tolist(),
            second=[3] * 4,
        )

    def test_rejects_invalid_scores(self):
        """Check rejects invalid scores."""
        # Process each score
        for score in (np.nan, np.inf, -np.inf, -0.01, 1.01, 0.5 + 1j, "invalid"):
            with self.subTest(score=score):
                calls = make_calls()
                values = calls["score"].tolist()
                values[0] = score
                calls["score"] = values
                with self.assertRaises(expected_exception=ValueError):
                    rank_scores(calls=calls)

    def test_rejects_invalid_indexes(self):
        """Check rejects invalid indexes."""
        # Prepare the invalid values
        invalid_values = {
            "frame": [-1, 0.5, np.nan, np.inf, True, 1 + 1j, "wrong", 2**63],
            "well": [0, -1, 1.5, np.nan],
            "fly": [0, -1, 1.5, np.nan],
            "tracker_row": [-1, 1.5, np.nan],
        }
        for column, values in invalid_values.items():
            for value in values:
                with self.subTest(
                    column=column,
                    value=value,
                ):
                    calls = make_calls()
                    invalid_indexes = calls[column].tolist()
                    invalid_indexes[0] = value
                    calls[column] = invalid_indexes
                    with self.assertRaises(expected_exception=ValueError):
                        rank_scores(calls=calls)

    def test_rejects_duplicate_flies_and_inconsistent_rows(self):
        """Check rejects duplicate flies and inconsistent rows."""
        # Prepare the calls
        calls = make_calls()
        duplicate = pd.concat(
            objs=[calls, calls.iloc[[0]]],
            ignore_index=True,
        )
        with self.assertRaisesRegex(
            expected_exception=ValueError,
            expected_regex="only one score",
        ):
            rank_scores(calls=duplicate)

        # Store the computed values
        calls.loc[0, "tracker_row"] = 2
        with self.assertRaisesRegex(
            expected_exception=ValueError,
            expected_regex="exactly one tracker_row",
        ):
            rank_scores(calls=calls)

        # Prepare the calls
        calls = make_calls()
        calls.loc[calls["frame"] == 105, "tracker_row"] = 1
        with self.assertRaisesRegex(
            expected_exception=ValueError,
            expected_regex="different frames",
        ):
            rank_scores(calls=calls)

    def test_rejects_missing_or_repeated_columns(self):
        """Check rejects missing or repeated columns."""
        # Create a temporary video recording
        with self.assertRaisesRegex(
            expected_exception=ValueError,
            expected_regex="missing columns",
        ):
            rank_scores(calls=make_calls().drop(columns="tracker_row"))
        calls = make_calls()
        calls.columns = ["frame", "well", "fly", "tracker_row", "score", "note", "note"]
        with self.assertRaisesRegex(
            expected_exception=ValueError,
            expected_regex="unique",
        ):
            rank_scores(calls=calls)

    def test_empty_numeric_table_preserves_schema(self):
        """Check empty numeric table preserves schema."""
        # Prepare the ranked
        ranked = rank_scores(calls=make_calls().iloc[:0])
        self.assertTrue(expr=ranked.empty)
        self.assertIn(
            member="rank",
            container=ranked.columns,
        )
        self.assertIn(
            member="rank_gap",
            container=ranked.columns,
        )


class ExportTests(unittest.TestCase):
    def setUp(self):
        """Prepare the temporary inputs for each test."""
        # Prepare the temporary
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.session = self.root / "session"
        self.output = self.root / "ranked"
        self.source_tables = self.make_well(well_number=1)

    def make_well(self, well_number: int) -> list[pd.DataFrame]:
        """Make well."""
        # Prepare the tracker
        tracker = (
            self.session
            / f"arena{well_number}"
            / f"trial_arena_{well_number}-trackfeat.csv"
        )
        tracker.mkdir(parents=True)
        tables = []
        for fly_number in range(
            1,
            4,
        ):
            table = pd.DataFrame(
                data={
                    "pos x": np.arange(
                        6,
                        dtype=np.float64,
                    )
                    + fly_number / 10,
                    "pos y": np.arange(
                        6,
                        dtype=np.float64,
                    )
                    * -1
                    + fly_number / 10,
                    "ori": np.full(
                        shape=6,
                        fill_value=fly_number / 3,
                        dtype=np.float64,
                    ),
                    "marker": np.arange(
                        6,
                        dtype=np.int64,
                    )
                    + fly_number * 100,
                    "state": [f"original_{fly_number}_{row}" for row in range(6)],
                }
            )
            table.to_csv(
                path_or_buf=tracker / f"fly{fly_number}.csv",
                index=False,
            )
            tables.append(
                pd.read_csv(filepath_or_buffer=tracker / f"fly{fly_number}.csv")
            )
        (tracker.parent / "calibration.mat").write_bytes(
            data=b"test calibration contents"
        )
        # Return the tables
        return tables

    def tracker_output(self, well_number: int = 1) -> Path:
        """Tracker output."""
        # Return the result
        return (
            self.output
            / f"well{well_number}"
            / f"trial_arena_{well_number}-trackfeat.csv"
        )

    def test_reorders_only_sampled_rows_and_preserves_source(self):
        """Check reorders only sampled rows and preserves source."""
        # Prepare the source contents
        source_contents = {
            path: path.read_bytes()
            for path in self.session.rglob(pattern="*")
            if path.is_file()
        }
        summary = export_ranked_tracks(
            session_directory=self.session,
            calls=make_calls(),
            output_directory=self.output,
        )
        self.assertEqual(
            first=summary["ranked_rows"],
            second=2,
        )
        self.assertEqual(
            first=summary["total_rows"],
            second=6,
        )
        self.assertEqual(
            first=summary["well_count"],
            second=1,
        )

        # Prepare the source order
        source_order = {1: [2, 3, 1], 4: [3, 1, 2]}
        for rank in range(
            1,
            4,
        ):
            actual = pd.read_csv(
                filepath_or_buffer=self.tracker_output() / f"fly{rank}.csv"
            )
            expected = self.source_tables[rank - 1].copy(deep=True)
            for tracker_row, flies in source_order.items():
                source = self.source_tables[flies[rank - 1] - 1]
                for column_position in range(len(expected.columns)):
                    expected.iat[tracker_row, column_position] = source.iat[
                        tracker_row, column_position
                    ]
            pd.testing.assert_frame_equal(
                left=actual,
                right=expected,
            )
            self.assertEqual(
                first=actual["marker"].dtype,
                second=np.dtype("int64"),
            )
            pd.testing.assert_frame_equal(
                left=actual.iloc[[0, 2, 3, 5]],
                right=self.source_tables[rank - 1].iloc[[0, 2, 3, 5]],
            )
        for path, contents in source_contents.items():
            self.assertEqual(
                first=path.read_bytes(),
                second=contents,
            )
        self.assertEqual(
            first=(self.tracker_output() / "calibration.mat").read_bytes(),
            second=b"test calibration contents",
        )

        # Prepare the mapping
        mapping = pd.read_csv(filepath_or_buffer=self.output / "rank_mapping.csv")
        self.assertEqual(
            first=mapping["fly"].tolist(),
            second=[2, 3, 1, 3, 1, 2],
        )
        self.assertIn(
            member="confident",
            container=mapping.columns,
        )
        coverage = pd.read_csv(filepath_or_buffer=self.output / "rank_coverage.csv")
        self.assertEqual(
            first=coverage["tracker_row"].tolist(),
            second=list(range(6)),
        )
        self.assertEqual(
            first=coverage["ranked"].tolist(),
            second=[False, True, False, False, True, False],
        )
        self.assertEqual(
            first=coverage.loc[coverage["ranked"], "frame"].tolist(),
            second=[100, 105],
        )
        self.assertTrue(expr=coverage.loc[~coverage["ranked"], "frame"].isna().all())
        readme = (self.output / "README.txt").read_text(encoding="utf-8")
        self.assertIn(
            member="not an individual identity",
            container=readme,
        )
        self.assertIn(
            member="never carried",
            container=readme,
        )

    def test_exports_multiple_wells_and_optional_confidence(self):
        """Check exports multiple wells and optional confidence."""
        # Process the next step
        self.make_well(well_number=2)
        calls = pd.concat(
            objs=[make_calls(well_number=2), make_calls()],
            ignore_index=True,
        )
        calls = calls.drop(columns="confident")
        summary = export_ranked_tracks(
            session_directory=self.session,
            calls=calls,
            output_directory=self.output,
        )
        self.assertEqual(
            first=summary["well_count"],
            second=2,
        )
        self.assertEqual(
            first=summary["ranked_rows"],
            second=4,
        )
        self.assertEqual(
            first=summary["total_rows"],
            second=12,
        )
        self.assertTrue(
            expr=(self.tracker_output(well_number=2) / "fly1.csv").is_file()
        )
        mapping = pd.read_csv(filepath_or_buffer=self.output / "rank_mapping.csv")
        self.assertNotIn(
            member="confident",
            container=mapping.columns,
        )

    def test_validates_all_wells_before_writing(self):
        """Check validates all wells before writing."""
        # Process the next step
        self.make_well(well_number=2)
        incomplete = make_calls(well_number=2).iloc[1:]
        calls = pd.concat(
            objs=[make_calls(), incomplete],
            ignore_index=True,
        )
        with self.assertRaisesRegex(
            expected_exception=ValueError,
            expected_regex="every tracked fly",
        ):
            export_ranked_tracks(
                session_directory=self.session,
                calls=calls,
                output_directory=self.output,
            )
        self.assertFalse(expr=self.output.exists())

    def test_rejects_out_of_range_rows_and_missing_flies(self):
        """Check rejects out of range rows and missing flies."""
        # Prepare the out of range
        out_of_range = make_calls()
        out_of_range["tracker_row"] += 100
        for calls in (out_of_range, make_calls().iloc[1:]):
            with self.subTest(calls=calls.to_dict(orient="list")):
                with self.assertRaises(expected_exception=ValueError):
                    export_ranked_tracks(
                        session_directory=self.session,
                        calls=calls,
                        output_directory=self.output,
                    )
                self.assertFalse(expr=self.output.exists())

    def test_rejects_overlapping_and_occupied_destinations(self):
        """Check rejects overlapping and occupied destinations."""
        # Process each destination
        for destination in (self.session, self.session / "export", self.root):
            with self.subTest(destination=destination):
                with self.assertRaisesRegex(
                    expected_exception=ValueError,
                    expected_regex="overlap",
                ):
                    export_ranked_tracks(
                        session_directory=self.session,
                        calls=make_calls(),
                        output_directory=destination,
                    )
        self.output.mkdir()
        existing = self.output / "existing.txt"
        existing.write_text(
            data="preserve this",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            expected_exception=ValueError,
            expected_regex="empty directory",
        ):
            export_ranked_tracks(
                session_directory=self.session,
                calls=make_calls(),
                output_directory=self.output,
            )
        self.assertEqual(
            first=existing.read_text(encoding="utf-8"),
            second="preserve this",
        )

    def test_accepts_empty_output_directory(self):
        """Check accepts empty output directory."""
        # Process the next step
        self.output.mkdir()
        export_ranked_tracks(
            session_directory=self.session,
            calls=make_calls(),
            output_directory=self.output,
        )
        self.assertTrue(expr=(self.output / "rank_mapping.csv").is_file())

    def test_missing_calibration_and_empty_calls_leave_no_output(self):
        """Check missing calibration and empty calls leave no output."""
        # Create a temporary video recording
        with self.assertRaisesRegex(
            expected_exception=ValueError,
            expected_regex="complete scored frame",
        ):
            export_ranked_tracks(
                session_directory=self.session,
                calls=make_calls().iloc[:0],
                output_directory=self.output,
            )
        for calibration in self.session.rglob(pattern="calibration.mat"):
            calibration.unlink()
        with self.assertRaises(expected_exception=ValueError):
            export_ranked_tracks(
                session_directory=self.session,
                calls=make_calls(),
                output_directory=self.output,
            )
        self.assertFalse(expr=self.output.exists())


# Run the test suite
if __name__ == "__main__":
    unittest.main()
