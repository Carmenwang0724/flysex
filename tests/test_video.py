"""Check decoded video frame numbering and extraction limits."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from flysex.video import extract_stills


@unittest.skipUnless(
    importlib.util.find_spec("av"),
    "Install the video extra to test extraction",
)
class VideoTests(unittest.TestCase):
    def test_stride_keeps_original_decoded_frame_numbers(self):
        """Check stride keeps original decoded frame numbers."""
        import av

        # Create a temporary video recording
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "sample.mp4"
            with av.open(
                file=str(video),
                mode="w",
            ) as container:
                stream = container.add_stream(
                    "mpeg4",
                    rate=2,
                )
                stream.width = 64
                stream.height = 64
                stream.pix_fmt = "yuv420p"
                for index in range(8):
                    pixels = np.full(
                        shape=(64, 64, 3),
                        fill_value=30 + index * 20,
                        dtype=np.uint8,
                    )
                    frame = av.VideoFrame.from_ndarray(
                        pixels,
                        format="rgb24",
                    )
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            output = root / "photos"
            written = extract_stills(
                video_path=video,
                output_directory=output,
                stride=3,
                maximum_frames=7,
            )
            paths = sorted(output.glob(pattern="*.png"))
            self.assertEqual(
                first=written,
                second=3,
            )
            self.assertEqual(
                first=[int(path.name.split("_")[1]) for path in paths],
                second=[0, 3, 6],
            )
            for path, frame_number in zip(
                paths,
                (0, 3, 6),
            ):
                with Image.open(fp=path) as image:
                    self.assertAlmostEqual(
                        first=np.asarray(a=image).mean(),
                        second=30 + 20 * frame_number,
                        delta=4,
                    )
            with self.assertRaisesRegex(
                expected_exception=ValueError,
                expected_regex="empty",
            ):
                extract_stills(
                    video_path=video,
                    output_directory=output,
                )


# Run the test suite
if __name__ == "__main__":
    unittest.main()
