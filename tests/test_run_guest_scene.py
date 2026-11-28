"""Unit tests for scripts/run_guest_scene.py."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import run_guest_scene as rgs  # noqa: E402


class TestRunGuestSceneHelpers(unittest.TestCase):
    def test_fnv1a_32_and_64_vectors(self) -> None:
        self.assertEqual(rgs.fnv1a(b"", 32), 0x811C9DC5)
        self.assertEqual(rgs.fnv1a(b"foobar", 32), 0xBF9CF968)
        self.assertEqual(rgs.fnv1a(b"", 64), 0xCBF29CE484222325)
        self.assertEqual(rgs.fnv1a(b"foobar", 64), 0x85944171F73967E8)

    def test_check_raises_fail_on_false(self) -> None:
        rgs.check(True, "ok")
        with self.assertRaises(rgs.Fail) as ctx:
            rgs.check(False, "broken invariant")
        self.assertEqual(str(ctx.exception), "broken invariant")

    def test_tree_files_hashes_nested_files(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sub").mkdir()
            (root / "a.txt").write_bytes(b"hello")
            (root / "sub" / "b.bin").write_bytes(b"world")
            got = rgs.tree_files(root)
            self.assertEqual(
                got,
                {
                    "a.txt": rgs.sha256(b"hello"),
                    "sub/b.bin": rgs.sha256(b"world"),
                },
            )


if __name__ == "__main__":
    unittest.main()
