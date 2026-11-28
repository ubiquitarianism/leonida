"""vs_layout_census: PAL hardware stage x code markers x VS entry VGPRs, on hand-built containers.

    python -I tests/test_vs_layout_census.py

Every input is built here (index.json, synthetic FSSM payloads in DXBC containers, hand-written .s
listings). No game data, no llvm-mc: the one unlisted payload is disassembled by an injected fake.
"""
import contextlib
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import vs_layout_census as V  # noqa: E402

TAG = 0x000000C000020000


def fssm(api_id, hardware_id, size=0x300):
    """A synthetic FSSM header with one PAL stage record."""
    data = bytearray(size)
    data[0:4] = b"FSSM"
    struct.pack_into("<H", data, 4, 0x101)
    struct.pack_into("<Q", data, 0xC0, 0x210)
    struct.pack_into("<Q", data, 0x100, TAG | (api_id << 8) | hardware_id)
    return bytes(data)


def xbch(blob):
    compressed = zlib.compress(blob)
    return b"XBCH" + struct.pack("<IIIII", 0, len(compressed), 0, len(blob), 0) + compressed


def dxbc(payloads):
    """A DXBC container whose DXIL chunk holds an XDAT directory of tag-0x0D payloads."""
    table_len = 12 + 12 * len(payloads)
    entries, body = [], bytearray()
    for payload in payloads:
        entries.append(struct.pack("<III", 0x0D, table_len + len(body), len(payload)))
        body.extend(payload)
    dxil = b"XDAT" + struct.pack("<II", len(payloads), 8) + b"".join(entries) + bytes(body)
    chunks = [(b"DXIL", dxil)]
    head = 32 + 4 * len(chunks)
    offsets, parts = [], bytearray()
    for fourcc, payload in chunks:
        offsets.append(head + len(parts))
        parts.extend(fourcc + struct.pack("<I", len(payload)) + payload)
    header = b"DXBC" + bytes(16) + struct.pack("<III", 1, head + len(parts), len(chunks))
    return header + b"".join(struct.pack("<I", off) for off in offsets) + bytes(parts)


# Hand-written listings: instructions start with a tab, like shader_disasm.py output.
VS_FETCH = ("\ts_swappc_b64 s[4:5], s[4:5]\n"
            "\tv_add_nc_u32_e32 v1, v0, v3\n"
            "\tv_fma_f32 v2, v4, v5, v6\n"
            "\texp pos0 v1, v1, v1, v1 done\n"
            "\texp param0 v2, v4, v5, v6\n"
            "\ts_endpgm\n")
VS_PLAIN = ("\tv_cvt_f32_u32_e32 v1, v0\n"
            "\texp pos0 v1, v1, v1, v1 done\n"
            "\texp param0 v1, v1, v1, v1\n"
            "\ts_endpgm\n")
ES_FETCH = ("\ts_swappc_b64 s[4:5], s[4:5]\n"
            "\tv_cvt_f32_u32_e32 v1, v0\n"
            "\texp pos0 v1, v1, v1, v1 done\n"
            "\texp param0 v1, v1, v1, v1\n"
            "\ts_endpgm\n")
MS_NGG = ("\tv_mbcnt_lo_u32_b32 v0, -1, 0\n"
          "\ts_sendmsg sendmsg(MSG_GS_ALLOC_REQ)\n"
          "\texp prim v1, off, off, off done\n"
          "\texp pos0 v1, v1, v1, v1 done\n"
          "\texp param0 v1, v1, v1, v1\n"
          "\ts_endpgm\n")
PS_PLAIN = ("\tv_mov_b32_e32 v1, 1.0\n"
            "\texp mrt0 v1, v1, v1, v1\n"
            "\ts_endpgm\n")


def listing(name, body):
    return f"; {name}  code 0x210..0x220 (header), gfx1013, 0 invalid words\n" + body


def build(root):
    dx = os.path.join(root, "dxbc")
    os.makedirs(os.path.join(dx, "disasm"))
    index = []
    for name, stage in (("a_vs", "vs"), ("b_vs", "vs"), ("c_ms", "ms"), ("d_ps", "ps"), ("e_empty", "vs")):
        index.append({"name": name, "file": name + ".dxbc", "stage": stage})
    with open(os.path.join(dx, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f)

    with open(os.path.join(dx, "a_vs.dxbc"), "wb") as f:
        f.write(dxbc([xbch(fssm(0, 4))]))
    with open(os.path.join(dx, "disasm", "a_vs.s"), "w", encoding="utf-8", newline="\n") as f:
        f.write(listing("a_vs", VS_FETCH))

    # Two payloads: the ES variant first (hardware id 2), the VS variant second (hardware id 4). The
    # listing is the last payload, as fssm_unpack writes it; the first is disassembled by the fake below.
    with open(os.path.join(dx, "b_vs.dxbc"), "wb") as f:
        f.write(dxbc([xbch(fssm(0, 2)), xbch(fssm(0, 4))]))
    with open(os.path.join(dx, "disasm", "b_vs.s"), "w", encoding="utf-8", newline="\n") as f:
        f.write(listing("b_vs", VS_PLAIN))

    with open(os.path.join(dx, "c_ms.dxbc"), "wb") as f:
        f.write(dxbc([xbch(fssm(6, 7))]))
    with open(os.path.join(dx, "disasm", "c_ms.s"), "w", encoding="utf-8", newline="\n") as f:
        f.write(listing("c_ms", MS_NGG))

    with open(os.path.join(dx, "d_ps.dxbc"), "wb") as f:
        f.write(dxbc([xbch(fssm(1, 5))]))
    with open(os.path.join(dx, "disasm", "d_ps.s"), "w", encoding="utf-8", newline="\n") as f:
        f.write(listing("d_ps", PS_PLAIN))

    # A container with no tag-0x0D payload contributes nothing.
    with open(os.path.join(dx, "e_empty.dxbc"), "wb") as f:
        f.write(dxbc([]))
    return dx


def fake_disasm(blob):
    """The ES variant's listing, chosen by its PAL hardware id (2); anything else is the VS text."""
    import pal_stages  # noqa: E402  (sys.path already has tools)
    hw = pal_stages.stage_descriptors(blob)[0]["hardware_id"]
    return listing("es", ES_FETCH) if hw == 2 else listing("vs", VS_PLAIN)


class TestSurvey(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dx = build(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_payloads_by_stage_and_hardware_id(self):
        r = V.survey(self.dx, disasm_fn=fake_disasm)
        self.assertEqual(r["containers"], 5)
        self.assertEqual(r["payloads"], 5)          # a_vs, b_vs x2, c_ms, d_ps; e_empty has none
        self.assertEqual(r["payloads_without_record"], 0)
        self.assertEqual(r["api_stage_mismatches"], 0)
        self.assertEqual(r["by_stage_hw"], {
            "ps|1|5": 1, "vs|0|2": 1, "vs|0|4": 2, "ms|6|7": 1})

    def test_markers_land_on_the_right_hardware_stage(self):
        r = V.survey(self.dx, disasm_fn=fake_disasm)
        hw = {int(k): v for k, v in r["markers_by_hw"].items()}
        self.assertEqual(hw[7], {"exp prim": 1, "s_sendmsg": 1, "s_sendmsg GS_ALLOC_REQ": 1,
                                 "exp pos0": 1, "exp param": 1, "v_mbcnt": 1})
        self.assertEqual(hw.get(5, {}), {})
        # hw 4: a_vs (fetch) + b_vs listed (plain) + b_vs... the ES variant is hw 2.
        self.assertEqual(hw[4]["fetch call"], 1)
        self.assertEqual(hw[4]["exp pos0"], 2)
        self.assertEqual(hw[2]["fetch call"], 1)
        self.assertNotIn("v_mbcnt", hw[4])
        self.assertNotIn("exp prim", hw[4])

    def test_vs_entry_vgprs_cover_both_payloads(self):
        r = V.survey(self.dx, disasm_fn=fake_disasm)
        self.assertEqual(r["vs_payloads"], 3)
        self.assertEqual(r["vs_by_hw"], {2: 1, 4: 2})
        self.assertEqual(r["vs_fetch"], 2)           # a_vs and the ES variant
        self.assertEqual(r["vs_reads_v0"], 3)        # every VS/ES payload reads v0
        self.assertEqual(r["disassembled_unlisted"], 1)

    def test_no_disassembler_leaves_unlisted_payloads_uncounted(self):
        r = V.survey(self.dx)                        # no disasm_fn: the ES payload has no text
        self.assertEqual(r["vs_payloads"], 3)
        self.assertEqual(r["vs_fetch"], 1)           # only a_vs's listing
        self.assertEqual(r["markers_by_hw"].get("2", {}), {})
        self.assertEqual(r["disassembled_unlisted"], 0)

    def test_cli_json(self):
        out = os.path.join(self._tmp.name, "out.json")
        result = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "scripts", "vs_layout_census.py"),
                                 self.dx, "--json", out],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        with open(out, encoding="utf-8") as f:
            document = json.load(f)
        self.assertEqual(document["payloads"], 5)
        self.assertEqual(document["vs_by_hw"], {"2": 1, "4": 2})
        # The CLI disassembles the unlisted ES payload with llvm-mc; a synthetic blob has no code, so the
        # marker count for hw 2 stays empty and the run still succeeds.
        self.assertIn("vs|0|2", document["by_stage_hw"])


class TestJsonGuard(unittest.TestCase):
    """--json is refused, before survey() reads the corpus, when it names an input or a directory."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dx = build(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def refuse(self, json_path):
        """Run main() with survey() mocked; assert one error line, exit 2 and survey() never called."""
        err = io.StringIO()
        with mock.patch.object(V, "survey") as survey, contextlib.redirect_stderr(err):
            code = V.main([self.dx, "--json", json_path])
        survey.assert_not_called()
        self.assertEqual(code, 2)
        lines = err.getvalue().splitlines()
        self.assertEqual(len(lines), 1, lines)
        self.assertTrue(lines[0].startswith("vs_layout_census: error: --json "), lines[0])
        return lines[0]

    @staticmethod
    def read(path):
        with open(path, "rb") as f:
            return f.read()

    def test_listed_listing_refuses_and_is_unchanged(self):
        listing_path = os.path.join(self.dx, "disasm", "a_vs.s")
        before = self.read(listing_path)
        self.assertIn("would overwrite the input", self.refuse(listing_path))
        self.assertEqual(self.read(listing_path), before)

    def test_indexed_container_refuses_and_is_unchanged(self):
        container = os.path.join(self.dx, "b_vs.dxbc")
        before = self.read(container)
        self.assertIn("would overwrite the input", self.refuse(container))
        self.assertEqual(self.read(container), before)

    def test_index_refuses(self):
        index_path = os.path.join(self.dx, "index.json")
        before = self.read(index_path)
        self.refuse(index_path)
        self.assertEqual(self.read(index_path), before)

    def test_directory_refuses_before_survey(self):
        self.assertIn("is a directory", self.refuse(os.path.join(self.dx, "disasm")))

    def test_directory_refuses_before_llvm_mc(self):
        # survey() not mocked: the CLI would run llvm-mc (default_disasm) on b_vs's unlisted ES payload.
        err = io.StringIO()
        with mock.patch.object(V, "default_disasm", side_effect=fake_disasm) as disasm, \
                contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            code = V.main([self.dx, "--json", self._tmp.name])
        disasm.assert_not_called()
        self.assertEqual(code, 2)

    def test_other_spelling_of_an_input_refuses(self):
        listing_path = os.path.join(self.dx, "disasm", "c_ms.s")
        before = self.read(listing_path)
        self.refuse(os.path.join(self.dx, "disasm", "..", "disasm", ".", "c_ms.s"))
        self.assertEqual(self.read(listing_path), before)

    def test_symlink_to_an_input_refuses(self):
        link = os.path.join(self._tmp.name, "link.json")
        container = os.path.join(self.dx, "a_vs.dxbc")
        try:
            os.symlink(container, link)
        except (OSError, NotImplementedError, AttributeError):
            self.skipTest("no symlinks here")
        before = self.read(container)
        self.refuse(link)
        self.assertEqual(self.read(container), before)

    def test_hard_link_to_an_input_refuses(self):
        link = os.path.join(self._tmp.name, "hard.json")
        listing_path = os.path.join(self.dx, "disasm", "d_ps.s")
        try:
            os.link(listing_path, link)
        except (OSError, NotImplementedError, AttributeError):
            self.skipTest("no hard links here")
        before = self.read(listing_path)
        self.refuse(link)
        self.assertEqual(self.read(listing_path), before)

    def test_other_case_refuses_on_a_case_folding_volume(self):
        upper = os.path.join(self.dx, "disasm", "A_VS.S")
        if not os.path.exists(upper):
            self.skipTest("this volume keeps letter case apart")
        before = self.read(os.path.join(self.dx, "disasm", "a_vs.s"))
        self.refuse(upper)
        self.assertEqual(self.read(os.path.join(self.dx, "disasm", "a_vs.s")), before)

    def test_cli_refuses_with_one_line(self):
        listing_path = os.path.join(self.dx, "disasm", "a_vs.s")
        before = self.read(listing_path)
        result = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "scripts", "vs_layout_census.py"),
                                 self.dx, "--json", listing_path],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr.strip().splitlines(),
                         [f"vs_layout_census: error: --json would overwrite the input {listing_path}"])
        self.assertEqual(self.read(listing_path), before)

    def test_separate_json_still_runs(self):
        out = os.path.join(self._tmp.name, "report.json")
        with mock.patch.object(V, "default_disasm", side_effect=fake_disasm), \
                contextlib.redirect_stdout(io.StringIO()):
            code = V.main([self.dx, "--json", out])
        self.assertEqual(code, 0)
        with open(out, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["payloads"], 5)


if __name__ == "__main__":
    unittest.main()
