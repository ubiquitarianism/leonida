"""Unit tests for tools/sandbox.py."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

import sandbox  # noqa: E402


class TestSandbox(unittest.TestCase):
    def test_read_ppm_and_check_frame_coverage_and_colours(self) -> None:
        with TemporaryDirectory() as tmp:
            sky_only = Path(tmp) / "sky.ppm"
            sky_only.write_bytes(b"P6\n10 10\n255\n" + sandbox.SKY * 100)
            w, h, px = sandbox.read_ppm(sky_only)
            self.assertEqual((w, h, len(px)), (10, 10, 300))
            ok, msg = sandbox.check_frame(sky_only, textured=False)
            self.assertFalse(ok)
            self.assertIn("0 of 100 pixels covered", msg)

            covered = Path(tmp) / "covered.ppm"
            pixels = bytearray(sandbox.SKY * 100)
            pixels[0:3] = b"\x00\x00\x00"
            pixels[3:6] = b"\xff\x00\x00"
            covered.write_bytes(b"P6\n10 10\n255\n" + bytes(pixels))
            ok_untex, _ = sandbox.check_frame(covered, textured=False)
            self.assertTrue(ok_untex)
            ok_tex, _ = sandbox.check_frame(covered, textured=True)
            self.assertFalse(ok_tex)

    def test_pick_pair_matches_expected_vs_and_ps(self) -> None:
        with TemporaryDirectory() as tmp:
            shaders = Path(tmp) / "shaders"
            spirv = shaders / "spirv"
            disasm = shaders / "dxbc" / "disasm"
            spirv.mkdir(parents=True)
            disasm.mkdir(parents=True)

            (spirv / "01_vs.spv").write_bytes(b"spv")
            (spirv / "02_ps.spv").write_bytes(b"spv")
            (disasm / "01_vs.s").write_text(
                "\n".join(
                    [
                        "s_load_dwordx4 s[0:3], s[8:9], 0",
                        "s_load_dwordx4 s[4:7], s[8:9], 0x40",
                        "s_load_dwordx4 s[12:15], s[8:9], 0x60",
                        "s_buffer_load_dwordx16 s[16:31], s[0:3], 0x160",
                        "s_buffer_load_dwordx4 s[32:35], s[0:3], 0x30",
                    ]
                ),
                encoding="utf-8",
            )
            (disasm / "02_ps.s").write_text(
                "s_load_dwordx8 s[0:7], s[8:9], 0x20\nimage_sample v[0:3], v[0:1], s[0:7], s[8:11]\nexp mrt0 v0, v1, v2, v3\n",
                encoding="utf-8",
            )
            index = {
                "01_vs": {
                    "stage": "vs",
                    "bindings": [],
                    "user_sgpr_reads": [{"kind": "ptr32_lo", "sgpr": 8}],
                    "input_signature": [
                        {"semantic": "POSITION", "semantic_index": 0, "mask": 7},
                        {"semantic": "NORMAL", "semantic_index": 0, "mask": 7},
                        {"semantic": "TEXCOORD", "semantic_index": 0, "mask": 3},
                    ],
                },
                "02_ps": {
                    "stage": "ps",
                    "bindings": [0, 1, 2],
                    "user_sgpr_reads": [{"kind": "ptr32_lo", "sgpr": 8}],
                    "input_signature": [
                        {"semantic": "SV_Position", "semantic_index": 0, "mask": 15},
                        {"semantic": "TEXCOORD", "semantic_index": 0, "mask": 3},
                        {"semantic": "COLOR", "semantic_index": 0, "mask": 15},
                    ],
                },
            }
            (spirv / "index.json").write_text(json.dumps(index), encoding="utf-8")

            pair = sandbox.pick_pair(str(shaders))
            self.assertEqual(pair, (str(spirv / "01_vs.spv"), str(spirv / "02_ps.spv")))


if __name__ == "__main__":
    unittest.main()
