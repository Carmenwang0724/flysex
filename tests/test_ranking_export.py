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
    return pd.DataFrame(
        {
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
        ranked = rank_scores(calls=make_calls())
        self.assertEqual(ranked["fly"].tolist(), [2, 3, 1, 3, 1, 2])
        self.assertEqual(ranked["rank"].tolist(), [1, 2, 3, 1, 2, 3])
        self.assertEqual(ranked["confident"].tolist(), [False, True, True, False, False, True])
        np.testing.assert_allclose(
            actual=ranked["rank_gap"].to_numpy(),
            desired=[0.0, 0.8, np.nan, 0.2, 0.1, np.nan],
            equal_nan=True,
        )
        self.assertEqual(ranked["note"].tolist(), ["keep"] * 6)

    def test_keeps_input_unchanged_and_ranks_each_well(self):
        calls = pd.concat([make_calls(well_number=2), make_calls()], ignore_index=True)
        original = calls.copy(deep=True)
        ranked = rank_scores(calls=calls)
        pd.testing.assert_frame_equal(left=calls, right=original)
        self.assertEqual(ranked.groupby(["well", "frame"])["rank"].max().tolist(), [3] * 4)

    def test_rejects_invalid_scores(self):
        for score in (np.nan, np.inf, -np.inf, -0.01, 1.01, 0.5 + 1j, "invalid"):
            with self.subTest(score=score):
                calls = make_calls()
                values = calls["score"].tolist()
                values[0] = score
                calls["score"] = values
                with self.assertRaises(ValueError):
                    rank_scores(calls=calls)

    def test_rejects_invalid_indexes(self):
        invalid_values = {
            "frame": [-1, 0.5, np.nan, np.inf, True, 1 + 1j, "wrong", 2**63],
            "well": [0, -1, 1.5, np.nan],
            "fly": [0, -1, 1.5, np.nan],
            "tracker_row": [-1, 1.5, np.nan],
        }
        for column, values in invalid_values.items():
            for value in values:
                with self.subTest(column=column, value=value):
                    calls = make_calls()
                    invalid_indexes = calls[column].tolist()
                    invalid_indexes[0] = value
                    calls[column] = invalid_indexes
                    with self.assertRaises(ValueError):
                        rank_scores(calls=calls)

    def test_rejects_duplicate_flies_and_inconsistent_rows(self):
        calls = make_calls()
        duplicate = pd.concat([calls, calls.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "only one score"):
            rank_scores(calls=duplicate)

        calls.loc[0, "tracker_row"] = 2
        with self.assertRaisesRegex(ValueError, "exactly one tracker_row"):
            rank_scores(calls=calls)

        calls = make_calls()
        calls.loc[calls["frame"] == 105, "tracker_row"] = 1
        with self.assertRaisesRegex(ValueError, "different frames"):
            rank_scores(calls=calls)

    def test_rejects_missing_or_repeated_columns(self):
        with self.assertRaisesRegex(ValueError, "missing columns"):
            rank_scores(calls=make_calls().drop(columns="tracker_row"))
        calls = make_calls()
        calls.columns = ["frame", "well", "fly", "tracker_row", "score", "note", "note"]
        with self.assertRaisesRegex(ValueError, "unique"):
            rank_scores(calls=calls)

    def test_empty_numeric_table_preserves_schema(self):
        ranked = rank_scores(calls=make_calls().iloc[:0])
        self.assertTrue(ranked.empty)
        self.assertIn("rank", ranked.columns)
        self.assertIn("rank_gap", ranked.columns)


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.session = self.root / "session"
        self.output = self.root / "ranked"
        self.source_tables = self.make_well(well_number=1)

    def make_well(self, well_number: int) -> list[pd.DataFrame]:
        tracker = self.session / f"arena{well_number}" / f"trial_arena_{well_number}-trackfeat.csv"
        tracker.mkdir(parents=True)
        tables = []
        for fly_number in range(1, 4):
            table = pd.DataFrame(
                {
                    "pos x": np.arange(6, dtype=np.float64) + fly_number / 10,
                    "pos y": np.arange(6, dtype=np.float64) * -1 + fly_number / 10,
                    "ori": np.full(6, fly_number / 3, dtype=np.float64),
                    "marker": np.arange(6, dtype=np.int64) + fly_number * 100,
                    "state": [f"original_{fly_number}_{row}" for row in range(6)],
                }
            )
            table.to_csv(path_or_buf=tracker / f"fly{fly_number}.csv", index=False)
            tables.append(pd.read_csv(filepath_or_buffer=tracker / f"fly{fly_number}.csv"))
        (tracker.parent / "calibration.mat").write_bytes(data=b"test calibration contents")
        return tables

    def tracker_output(self, well_number: int = 1) -> Path:
        return self.output / f"well{well_number}" / f"trial_arena_{well_number}-trackfeat.csv"

    def test_reorders_only_sampled_rows_and_preserves_source(self):
        source_contents = {path: path.read_bytes() for path in self.session.rglob("*") if path.is_file()}
        summary = export_ranked_tracks(
            session_directory=self.session,
            calls=make_calls(),
            output_directory=self.output,
        )
        self.assertEqual(summary["ranked_rows"], 2)
        self.assertEqual(summary["total_rows"], 6)
        self.assertEqual(summary["well_count"], 1)

        source_order = {1: [2, 3, 1], 4: [3, 1, 2]}
        for rank in range(1, 4):
            actual = pd.read_csv(filepath_or_buffer=self.tracker_output() / f"fly{rank}.csv")
            expected = self.source_tables[rank - 1].copy(deep=True)
            for tracker_row, flies in source_order.items():
                source = self.source_tables[flies[rank - 1] - 1]
                for column_position in range(len(expected.columns)):
                    expected.iat[tracker_row, column_position] = source.iat[tracker_row, column_position]
            pd.testing.assert_frame_equal(left=actual, right=expected)
            self.assertEqual(actual["marker"].dtype, np.dtype("int64"))
            pd.testing.assert_frame_equal(
                left=actual.iloc[[0, 2, 3, 5]],
                right=self.source_tables[rank - 1].iloc[[0, 2, 3, 5]],
            )
        for path, contents in source_contents.items():
            self.assertEqual(path.read_bytes(), contents)
        self.assertEqual(
            (self.tracker_output() / "calibration.mat").read_bytes(),
            b"test calibration contents",
        )

        mapping = pd.read_csv(filepath_or_buffer=self.output / "rank_mapping.csv")
        self.assertEqual(mapping["fly"].tolist(), [2, 3, 1, 3, 1, 2])
        self.assertIn("confident", mapping.columns)
        coverage = pd.read_csv(filepath_or_buffer=self.output / "rank_coverage.csv")
        self.assertEqual(coverage["tracker_row"].tolist(), list(range(6)))
        self.assertEqual(coverage["ranked"].tolist(), [False, True, False, False, True, False])
        self.assertEqual(coverage.loc[coverage["ranked"], "frame"].tolist(), [100, 105])
        self.assertTrue(coverage.loc[~coverage["ranked"], "frame"].isna().all())
        readme = (self.output / "README.txt").read_text(encoding="utf-8")
        self.assertIn("not an individual identity", readme)
        self.assertIn("never carried", readme)

    def test_exports_multiple_wells_and_optional_confidence(self):
        self.make_well(well_number=2)
        calls = pd.concat([make_calls(well_number=2), make_calls()], ignore_index=True)
        calls = calls.drop(columns="confident")
        summary = export_ranked_tracks(
            session_directory=self.session,
            calls=calls,
            output_directory=self.output,
        )
        self.assertEqual(summary["well_count"], 2)
        self.assertEqual(summary["ranked_rows"], 4)
        self.assertEqual(summary["total_rows"], 12)
        self.assertTrue((self.tracker_output(well_number=2) / "fly1.csv").is_file())
        mapping = pd.read_csv(filepath_or_buffer=self.output / "rank_mapping.csv")
        self.assertNotIn("confident", mapping.columns)

    def test_validates_all_wells_before_writing(self):
        self.make_well(well_number=2)
        incomplete = make_calls(well_number=2).iloc[1:]
        calls = pd.concat([make_calls(), incomplete], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "every tracked fly"):
            export_ranked_tracks(
                session_directory=self.session,
                calls=calls,
                output_directory=self.output,
            )
        self.assertFalse(self.output.exists())

    def test_rejects_out_of_range_rows_and_missing_flies(self):
        out_of_range = make_calls()
        out_of_range["tracker_row"] += 100
        for calls in (out_of_range, make_calls().iloc[1:]):
            with self.subTest(calls=calls.to_dict(orient="list")):
                with self.assertRaises(ValueError):
                    export_ranked_tracks(
                        session_directory=self.session,
                        calls=calls,
                        output_directory=self.output,
                    )
                self.assertFalse(self.output.exists())

    def test_rejects_overlapping_and_occupied_destinations(self):
        for destination in (self.session, self.session / "export", self.root):
            with self.subTest(destination=destination):
                with self.assertRaisesRegex(ValueError, "overlap"):
                    export_ranked_tracks(
                        session_directory=self.session,
                        calls=make_calls(),
                        output_directory=destination,
                    )
        self.output.mkdir()
        existing = self.output / "existing.txt"
        existing.write_text(data="preserve this", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "empty directory"):
            export_ranked_tracks(
                session_directory=self.session,
                calls=make_calls(),
                output_directory=self.output,
            )
        self.assertEqual(existing.read_text(encoding="utf-8"), "preserve this")

    def test_accepts_empty_output_directory(self):
        self.output.mkdir()
        export_ranked_tracks(
            session_directory=self.session,
            calls=make_calls(),
            output_directory=self.output,
        )
        self.assertTrue((self.output / "rank_mapping.csv").is_file())

    def test_missing_calibration_and_empty_calls_leave_no_output(self):
        with self.assertRaisesRegex(ValueError, "complete scored frame"):
            export_ranked_tracks(
                session_directory=self.session,
                calls=make_calls().iloc[:0],
                output_directory=self.output,
            )
        for calibration in self.session.rglob("calibration.mat"):
            calibration.unlink()
        with self.assertRaises(ValueError):
            export_ranked_tracks(
                session_directory=self.session,
                calls=make_calls(),
                output_directory=self.output,
            )
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
