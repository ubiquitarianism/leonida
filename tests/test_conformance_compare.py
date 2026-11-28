"""Unit tests for tools/conformance/compare.py."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools" / "conformance"))

import compare  # noqa: E402


class TestConformanceCompare(unittest.TestCase):
    def test_raster_rgb_and_rgba_pixel_access(self) -> None:
        rgb = compare.Raster(
            width=2,
            height=1,
            colour_type=compare.PNG_COLOUR_TYPE_RGB,
            pixels=bytearray([10, 20, 30, 40, 50, 60]),
            source_path="rgb.png",
        )
        self.assertEqual(rgb.channels(), 3)
        self.assertEqual(rgb.colour_name(), "RGB")
        self.assertEqual(rgb.stored_pixel(1, 0), (40, 50, 60))
        self.assertEqual(rgb.pixel_rgba(1, 0), (40, 50, 60, 255))

        rgba = compare.Raster(
            width=1,
            height=1,
            colour_type=compare.PNG_COLOUR_TYPE_RGBA,
            pixels=bytearray([1, 2, 3, 128]),
            source_path="rgba.png",
        )
        self.assertEqual(rgba.channels(), 4)
        self.assertEqual(rgba.colour_name(), "RGBA")
        self.assertEqual(rgba.pixel_rgba(0, 0), (1, 2, 3, 128))

    def test_raster_bounds_and_colour_type_errors(self) -> None:
        rgb = compare.Raster(
            width=1,
            height=1,
            colour_type=compare.PNG_COLOUR_TYPE_RGB,
            pixels=bytearray([0, 0, 0]),
            source_path="tiny.png",
        )
        with self.assertRaises(compare.ToolError):
            rgb.stored_pixel(1, 0)
        with self.assertRaises(compare.ToolError):
            compare.channels_for_colour_type(99, "bad.png")

    def test_built_in_selftest_passes(self) -> None:
        self.assertEqual(compare.main(["selftest"]), compare.EXIT_PASSED)


if __name__ == "__main__":
    unittest.main()
