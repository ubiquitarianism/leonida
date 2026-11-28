"""Unit tests for runtime/test/shaders/build.py."""

from __future__ import annotations

import struct
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "runtime" / "test" / "shaders"))

import build as sb  # noqa: E402


class TestShadersBuild(unittest.TestCase):
    def test_asm_mattr_and_shader_source(self) -> None:
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "test.asm"
            p.write_text("; header\n; mattr: +wavefrontsize64\nv_mov_b32 v0, 0\n; mattr: ignored\n", encoding="utf-8")
            self.assertEqual(sb.asm_mattr(p), "+wavefrontsize64")

            p_none = Path(tmp) / "plain.asm"
            p_none.write_text("v_mov_b32 v0, 0\n", encoding="utf-8")
            self.assertIsNone(sb.asm_mattr(p_none))

        self.assertEqual(sb.shader_source("vs_clip_param4"), "vs_clip")
        self.assertEqual(sb.shader_source("ps_mullit"), "ps_mullit")
        self.assertEqual(sb.shader_source("cs_lds"), "cs_lds")

    def test_asm_fssm_and_cs_fssm_extra(self) -> None:
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "fssm.asm"
            p.write_text("; fssm-at: 0x10\n; fssm-data: 0x11223344 0x55667788\nv_mov_b32 v0, 0\n", encoding="utf-8")
            self.assertEqual(sb.asm_fssm_fields(p), (0x10, [0x11223344, 0x55667788]))
            blob = sb.asm_fssm(p)
            self.assertIsNotNone(blob)
            self.assertEqual(blob[:4], b"FSSM")
            self.assertEqual(len(blob), 0x10 + 8)
            self.assertEqual(struct.unpack_from("<2I", blob, 0x10), (0x11223344, 0x55667788))
            self.assertIsNone(sb.cs_fssm_extra(p))

            p_cs = Path(tmp) / "cs_fssm.asm"
            p_cs.write_text("; fssm-at: 0x148\n; fssm-data: 0xdeadbeef\nv_mov_b32 v0, 0\n", encoding="utf-8")
            extra = sb.cs_fssm_extra(p_cs)
            self.assertEqual(extra, (0x148, struct.pack("<I", 0xDEADBEEF)))


if __name__ == "__main__":
    unittest.main()
