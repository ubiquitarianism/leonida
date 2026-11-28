"""Unit tests for runtime/test/shaders/check.py."""

from __future__ import annotations

import os
import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "runtime" / "test" / "shaders"))

import check as sc  # noqa: E402


class TestShadersCheckBuildDirs(unittest.TestCase):
    def test_complete_and_resolve_build_dirs(self) -> None:
        with TemporaryDirectory() as tmp:
            build = Path(tmp)
            progs = ["a.exe", "b.exe"]
            self.assertEqual(sc.complete_build_dirs(str(build), progs), [])
            self.assertEqual(sc.resolve_build_dir(str(build), progs), str(build))

            rel = build / "Release"
            dbg = build / "Debug"
            rel.mkdir()
            dbg.mkdir()
            for p in progs:
                (rel / p).write_text("rel", encoding="utf-8")
                (dbg / p).write_text("dbg", encoding="utf-8")
            now = time.time()
            os.utime(rel / "a.exe", (now - 10, now - 10))
            os.utime(rel / "b.exe", (now - 10, now - 10))
            os.utime(dbg / "a.exe", (now, now))
            os.utime(dbg / "b.exe", (now, now))

            self.assertEqual(sc.complete_build_dirs(str(build), progs), [str(rel), str(dbg)])
            self.assertEqual(sc.resolve_build_dir(str(build), progs), str(dbg))

            (build / "a.exe").write_text("root", encoding="utf-8")
            (build / "b.exe").write_text("root", encoding="utf-8")
            self.assertEqual(sc.resolve_build_dir(str(build), progs), str(build))

    def test_not_found_message_names_missing_and_build_dir(self) -> None:
        msg = sc.not_found_message("/tmp/build", ["vk_compute_test"])
        self.assertIn("/tmp/build", msg)
        self.assertIn("vk_compute_test", msg)


if __name__ == "__main__":
    unittest.main()
