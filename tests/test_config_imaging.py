"""Check configuration alignment and crop geometry."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from flysex.config import load_config
from flysex.imaging import crop_along_body_axis, crop_flies, read_still
from flysex.session import index_stills


class ConfigTests(unittest.TestCase):
    def setUp(self):
        """Prepare the temporary inputs for each test."""
        # Prepare the temporary
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "data").mkdir()
        (self.root / "model").mkdir()
        self.path = self.root / "config.json"
        self.values = {
            "session_directory": "data",
            "model_directory": "model",
            "output_directory": "results",
            "wells": [2],
            "frame_zero": 100,
            "well_origins": {"2": [832, 163]},
        }

    def load(self):
        """Load the temporary test configuration."""
        # Save the output
        self.path.write_text(data=json.dumps(obj=self.values))
        # Return the result
        return load_config(path=self.path)

    def test_resolves_paths_relative_to_configuration(self):
        """Check resolves paths relative to configuration."""
        # Prepare the config
        config = self.load()
        self.assertEqual(
            first=config.session_directory,
            second=(self.root / "data").resolve(),
        )
        self.assertEqual(
            first=config.frame_zero,
            second=100,
        )
        self.assertEqual(
            first=120 - config.frame_zero,
            second=20,
        )

    def test_requires_explicit_frame_origin(self):
        """Check requires explicit frame origin."""
        # Process each value
        for value in (None, True, "100", 1.5):
            with self.subTest(value=value):
                self.values["frame_zero"] = value
                with self.assertRaises(expected_exception=ValueError):
                    self.load()

    def test_rejects_quota_or_annotation_fields(self):
        """Check rejects quota or annotation fields."""
        # Process each key
        for key in ("males_to_tag", "stocking_record", "labelled_dataset"):
            self.values[key] = 2
            with self.assertRaisesRegex(
                expected_exception=ValueError,
                expected_regex="Unknown",
            ):
                self.load()
            del self.values[key]

    def test_rejects_output_inside_source_data(self):
        """Check rejects output inside source data."""
        # Store the computed values
        self.values["output_directory"] = "data/results"
        with self.assertRaisesRegex(
            expected_exception=ValueError,
            expected_regex="separate",
        ):
            self.load()

    def test_rejects_invalid_geometry_and_limits(self):
        """Check rejects invalid geometry and limits."""
        # Process each key, value
        for key, value in (
            ("max_frames", 0),
            ("seed", -1),
            ("wells", [7]),
            ("well_origins", {"2": [float("nan"), 0]}),
        ):
            old = self.values.get(key)
            self.values[key] = value
            with self.assertRaises(expected_exception=ValueError):
                self.load()
            if old is None:
                del self.values[key]
            else:
                self.values[key] = old


class ImagingTests(unittest.TestCase):
    def test_negated_orientation_aligns_body_axis(self):
        """Check negated orientation aligns body axis."""
        # Prepare the horizontal, vertical
        horizontal, vertical = np.meshgrid(
            np.arange(220),
            np.arange(220),
        )
        angle = 0.55
        along = (horizontal - 110) * np.cos(angle) - (vertical - 110) * np.sin(angle)
        across = (horizontal - 110) * np.sin(angle) + (vertical - 110) * np.cos(angle)
        image = 1 - 0.6 * np.exp(-((along / 30) ** 2) - (across / 5) ** 2)
        crop = crop_along_body_axis(
            image=image,
            centre_x=110,
            centre_y=110,
            orientation=angle,
            step=1,
        )
        darkness = crop.max() - crop
        coordinates = np.arange(96) - 47.5
        horizontal_spread = (darkness.sum(axis=0) * coordinates**2).sum()
        vertical_spread = (darkness.sum(axis=1) * coordinates**2).sum()
        self.assertGreater(
            a=horizontal_spread,
            b=vertical_spread * 8,
        )

    def test_rejects_black_truncated_band(self):
        """Check rejects black truncated band."""
        # Create a temporary video recording
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "frame_000001_00-00-00.png"
            pixels = np.full(
                shape=(100, 100),
                fill_value=150,
                dtype=np.uint8,
            )
            pixels[50:] = 0
            Image.fromarray(obj=pixels).save(path)
            with self.assertRaisesRegex(
                expected_exception=ValueError,
                expected_regex="empty horizontal band",
            ):
                read_still(
                    path=path,
                    width=100,
                    height=100,
                )

    def test_accepts_compressed_complete_image_and_rejects_wrong_size(self):
        """Check accepts compressed complete image and rejects wrong size."""
        # Create a temporary video recording
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "frame_000001_00-00-00.png"
            Image.fromarray(
                obj=np.full(
                    shape=(100, 100),
                    fill_value=150,
                    dtype=np.uint8,
                )
            ).save(path)
            self.assertEqual(
                first=read_still(
                    path=path,
                    width=100,
                    height=100,
                ).shape,
                second=(100, 100),
            )
            with self.assertRaisesRegex(
                expected_exception=ValueError,
                expected_regex="expected",
            ):
                read_still(
                    path=path,
                    width=120,
                    height=100,
                )

    def test_subset_stills_keep_original_frame_numbers(self):
        """Check subset stills keep original frame numbers."""
        # Create a temporary video recording
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "frame_000120_00-00-04.png").touch()
            self.assertEqual(
                first=list(index_stills(session_directory=directory)),
                second=[120],
            )
            (directory / "photos").mkdir()
            (directory / "photos" / "frame_000120_00-00-04.png").touch()
            with self.assertRaisesRegex(
                expected_exception=ValueError,
                expected_regex="Duplicate",
            ):
                index_stills(session_directory=directory)


# Run the test suite
if __name__ == "__main__":
    unittest.main()
