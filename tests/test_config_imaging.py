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
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "data").mkdir()
        (self.root / "model").mkdir()
        self.path = self.root / "config.json"
        self.values = {
            "session_directory": "data", "model_directory": "model", "output_directory": "results",
            "wells": [2], "frame_zero": 100, "well_origins": {"2": [832, 163]},
        }

    def load(self):
        self.path.write_text(json.dumps(self.values))
        return load_config(self.path)

    def test_resolves_paths_relative_to_configuration(self):
        config = self.load()
        self.assertEqual(config.session_directory, (self.root / "data").resolve())
        self.assertEqual(config.frame_zero, 100)
        self.assertEqual(120 - config.frame_zero, 20)

    def test_requires_explicit_frame_origin(self):
        for value in (None, True, "100", 1.5):
            with self.subTest(value=value):
                self.values["frame_zero"] = value
                with self.assertRaises(ValueError):
                    self.load()

    def test_rejects_quota_or_annotation_fields(self):
        for key in ("males_to_tag", "stocking_record", "labelled_dataset"):
            self.values[key] = 2
            with self.assertRaisesRegex(ValueError, "Unknown"):
                self.load()
            del self.values[key]

    def test_rejects_output_inside_source_data(self):
        self.values["output_directory"] = "data/results"
        with self.assertRaisesRegex(ValueError, "separate"):
            self.load()

    def test_rejects_invalid_geometry_and_limits(self):
        for key, value in (("max_frames", 0), ("seed", -1), ("wells", [7]), ("well_origins", {"2": [float("nan"), 0]})):
            old = self.values.get(key)
            self.values[key] = value
            with self.assertRaises(ValueError):
                self.load()
            if old is None:
                del self.values[key]
            else:
                self.values[key] = old


class ImagingTests(unittest.TestCase):
    def test_negated_orientation_aligns_body_axis(self):
        horizontal, vertical = np.meshgrid(np.arange(220), np.arange(220))
        angle = 0.55
        along = (horizontal - 110) * np.cos(angle) - (vertical - 110) * np.sin(angle)
        across = (horizontal - 110) * np.sin(angle) + (vertical - 110) * np.cos(angle)
        image = 1 - 0.6 * np.exp(-(along / 30) ** 2 - (across / 5) ** 2)
        crop = crop_along_body_axis(image=image, centre_x=110, centre_y=110, orientation=angle, step=1)
        darkness = crop.max() - crop
        coordinates = np.arange(96) - 47.5
        horizontal_spread = (darkness.sum(axis=0) * coordinates ** 2).sum()
        vertical_spread = (darkness.sum(axis=1) * coordinates ** 2).sum()
        self.assertGreater(horizontal_spread, vertical_spread * 8)

    def test_rejects_black_truncated_band(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "frame_000001_00-00-00.png"
            pixels = np.full((100, 100), 150, dtype=np.uint8)
            pixels[50:] = 0
            Image.fromarray(pixels).save(path)
            with self.assertRaisesRegex(ValueError, "empty horizontal band"):
                read_still(path=path, width=100, height=100)

    def test_accepts_compressed_complete_image_and_rejects_wrong_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "frame_000001_00-00-00.png"
            Image.fromarray(np.full((100, 100), 150, dtype=np.uint8)).save(path)
            self.assertEqual(read_still(path=path, width=100, height=100).shape, (100, 100))
            with self.assertRaisesRegex(ValueError, "expected"):
                read_still(path=path, width=120, height=100)

    def test_subset_stills_keep_original_frame_numbers(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "frame_000120_00-00-04.png").touch()
            self.assertEqual(list(index_stills(directory)), [120])
            (directory / "photos").mkdir()
            (directory / "photos" / "frame_000120_00-00-04.png").touch()
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                index_stills(directory)


if __name__ == "__main__":
    unittest.main()
