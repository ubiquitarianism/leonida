"""Additional unit tests for tools/file_mode.py (probe retry, exhaustion, fallback)."""

from __future__ import annotations

import stat
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

import file_mode  # noqa: E402


class TestFileModeExtra(unittest.TestCase):
    def test_new_file_mode_retries_on_file_exists_then_succeeds(self) -> None:
        with TemporaryDirectory() as tmp:
            real_open = file_mode.os.open
            calls = 0

            def flaky_open(path, flags, mode):
                nonlocal calls
                calls += 1
                if calls < 3:
                    raise FileExistsError("probe collision")
                return real_open(path, flags, mode)

            with mock.patch.object(file_mode.os, "open", side_effect=flaky_open):
                mode = file_mode.new_file_mode(tmp)
            self.assertEqual(calls, 3)
            self.assertEqual(mode & ~0o777, 0)

    def test_new_file_mode_raises_after_exhausting_retries(self) -> None:
        with TemporaryDirectory() as tmp:
            with mock.patch.object(file_mode.os, "open", side_effect=FileExistsError("always collides")):
                with self.assertRaises(FileExistsError):
                    file_mode.new_file_mode(tmp)

    def test_published_mode_falls_back_when_target_missing(self) -> None:
        with TemporaryDirectory() as tmp:
            missing = Path(tmp) / "new_output.json"
            mode = file_mode.published_mode(str(missing))
            self.assertEqual(stat.S_IMODE(mode), mode)


if __name__ == "__main__":
    unittest.main()
