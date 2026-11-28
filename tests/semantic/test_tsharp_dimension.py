"""The T# dimension contract (docs/design.md, "Images and samplers (T# dimension)").

On RDNA2 the hardware addresses an image through the T#'s own TYPE, and the bindless set gives each image its index
in ITS dimension's array. An instruction that reads a T# of another dimension therefore must not index its own
array with that index blindly: a 2D-array or cube T# read by a 2D instruction is its slice 0 (the runtime binds that
view at the same index in the 2D array), while a 3D T# read by a 2D instruction, or any other mismatch, reads that
dimension's fallback. A store, atomic or coherent read through such a T# goes to the storage fallback, and an image
with only a sampled view has no storage index at all.

The translator's side is: every entry of the descriptor table carries a kind word (one bit per array, sampled and
storage), the probe returns it next to the index, and every image access selects between the index and 0 on the bit
for the array its instruction reads. `--descriptor-table plain` keeps the old modules' bytes (no kind word read).

These tests translate small pixel and compute programs with the real translator (support) and check the emitted
SPIR-V: the bit each instruction dimension tests, the storage rule, the fallback operand, and the plain layout.
"""
import os
import re
import sys
import unittest

import support
import rdna2spirv

EPILOGUE = ["exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"]
# dim -> (address registers, sampled bit, storage bit)
DIMS = {"2D": ("v[4:5]", 0, 8), "2D_ARRAY": ("v[4:6]", 1, 9), "CUBE": ("v[4:6]", 1, 9),
        "3D": ("v[4:6]", 2, 10), "1D": ("v4", 3, 11)}
DESC = "s[0:7]"


def translate(lines, table=None):
    saved = rdna2spirv.DESCRIPTOR_TABLE
    if table is not None:
        rdna2spirv.DESCRIPTOR_TABLE = table
    try:
        return support.translate_lines("ps_tsharpdim", ["v_mov_b32 v4, 1", "v_mov_b32 v5, 2", "v_mov_b32 v6, 0"]
                                       + lines + EPILOGUE)
    finally:
        rdna2spirv.DESCRIPTOR_TABLE = saved


def uint_consts(text):
    out = {}
    for line in text.splitlines():
        m = re.match(r"\s*(%\w+) = OpConstant %uint (\d+)$", line)
        if m:
            out[m.group(1)] = int(m.group(2))
    return out


def one_bit_extracts(text):
    """The offsets of every OpBitFieldUExtract of one bit with a literal offset: the kind word's bit tests."""
    consts = uint_consts(text)
    offsets = []
    for line in text.splitlines():
        m = re.search(r"OpBitFieldUExtract %uint (.*)$", line)
        if not m:
            continue
        parts = m.group(1).split()
        if len(parts) != 3:
            continue
        off, width = consts.get(parts[1]), consts.get(parts[2])
        if width == 1 and off is not None:
            offsets.append(off)
    return offsets


def selects_with_zero(text):
    """(id, operands) of every OpSelect of a uint that picks a zero constant."""
    consts = uint_consts(text)
    out = []
    for line in text.splitlines():
        m = re.match(r"\s*(%\w+) = OpSelect %uint (.*)$", line)
        if not m:
            continue
        parts = m.group(2).split()
        if any(consts.get(p) == 0 for p in parts):
            out.append((m.group(1), parts))
    return out


class KindBits(unittest.TestCase):
    def test_dimension_tags(self):
        self.assertEqual(rdna2spirv.Gen.DIM_TAGS, {
            "dim:SQ_RSRC_IMG_2D": 0, "dim:SQ_RSRC_IMG_2D_ARRAY": 1, "dim:SQ_RSRC_IMG_CUBE": 1,
            "dim:SQ_RSRC_IMG_3D": 2, "dim:SQ_RSRC_IMG_1D": 3, "dim:SQ_RSRC_IMG_1D_ARRAY": 4})

    def test_each_sampled_dimension_reads_its_own_bit(self):
        for dim, (addr, sampled_bit, _) in DIMS.items():
            with self.subTest(dim=dim):
                spv = translate([f"image_load v[0:3], {addr}, {DESC} dmask:0xf dim:SQ_RSRC_IMG_{dim} unorm"])
                self.assertEqual(one_bit_extracts(spv), [sampled_bit])

    def test_each_storage_dimension_requires_its_storage_bit(self):
        for dim, (addr, _, storage_bit) in (DIMS | {"1D_ARRAY": ("v[4:5]", None, 12)}).items():
            with self.subTest(dim=dim):
                spv = translate([f"image_store v[0:3], {addr}, {DESC} dmask:0xf dim:SQ_RSRC_IMG_{dim} unorm"])
                self.assertEqual(one_bit_extracts(spv), [storage_bit])

    def test_atomics_use_the_storage_bit(self):
        for dim, bit in (("2D", 8), ("3D", 10)):
            with self.subTest(dim=dim):
                spv = translate([f"image_atomic_add v0, {DIMS[dim][0]}, {DESC} dmask:0x1 dim:SQ_RSRC_IMG_{dim} unorm"])
                self.assertEqual(one_bit_extracts(spv), [bit])

    def test_coherent_load_uses_the_storage_rule(self):
        spv = translate(["image_load v[0:3], v[4:5], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D unorm glc dlc"])
        self.assertEqual(one_bit_extracts(spv), [8])

    def test_gather_and_lod_reads_use_the_sampled_bit(self):
        spv = translate(["image_gather4_lz v[0:3], v[4:5], s[0:7], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_2D"])
        self.assertEqual(one_bit_extracts(spv), [0])
        spv = translate(["image_get_lod v[0:1], v[4:5], s[0:7], s[8:11] dmask:0x3 dim:SQ_RSRC_IMG_2D_ARRAY"])
        self.assertEqual(one_bit_extracts(spv), [1])

    def test_mismatch_reads_the_array_fallback(self):
        # The index is used only when the entry names the instruction's array; otherwise index 0 of that array,
        # whose element is the 1x1 fallback the runtime binds there. So the id the image array is indexed with is
        # an OpSelect (bit test, index, 0).
        for dim, (addr, bit, _) in DIMS.items():
            with self.subTest(dim=dim):
                spv = translate([f"image_load v[0:3], {addr}, {DESC} dmask:0xf dim:SQ_RSRC_IMG_{dim} unorm"])
                self.assertIn(bit, one_bit_extracts(spv))
                zero_selects = selects_with_zero(spv)
                self.assertTrue(zero_selects, spv)
                ids = {i for i, _ in zero_selects}
                indexed = [ln for ln in spv.splitlines()
                           if " = OpAccessChain %_ptr_UC_" in ln and " %images" in ln and ln.split()[-1] in ids]
                self.assertTrue(indexed, spv)


class PlainLayout(unittest.TestCase):
    def test_plain_reads_only_the_index(self):
        spv = translate(["image_load v[0:3], v[4:5], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D unorm"], table="plain")
        self.assertEqual(one_bit_extracts(spv), [])
        self.assertNotIn("dl_kind", spv)

    def test_tagged_probe_returns_index_and_kind(self):
        spv = translate(["image_load v[0:3], v[4:5], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D unorm"])
        self.assertIn("dl_kind", spv)
        self.assertIn("%desc_hit", spv)                  # a uvec2 (dedupe_types may rename it later)
        self.assertIn("OpCompositeConstruct %desc_hit", spv)

    def test_marker_is_recorded_per_module(self):
        for table, expected in (("tagged", "tagged"), ("plain", None)):
            with self.subTest(table=table):
                spv = translate(["image_load v[0:3], v[4:5], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D unorm"],
                                table=table)
                meta = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, spv)
                self.assertEqual(meta.get("descriptor_table"), expected)
