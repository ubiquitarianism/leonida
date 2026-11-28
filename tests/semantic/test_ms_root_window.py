"""A mesh shader reads its root arguments from the graphics window (D3D-448 follow-up), executed.

The console starts a mesh shader's root arguments at s16 and a vertex, pixel or geometry shader's at s8
(docs/d3d12_vulkan.md section 3.5). Both stages of a mesh pipeline read one push-constant block, which the runtime
fills once in root order with root dword k at push dword 8 + k. So the translator loads a mesh shader's s16..s31
from push dwords 8..23, its s8..s15 (below its base, not root data) as 0, and s0..s7 from push dwords 0..7 as for
every other stage; a VS, PS or GS keeps reading SGPR i from push dword i. index.json records the window.

The program (hand-written, no game data) exports, from each of its three vertex threads, a position made of the
user SGPRs it was started with: (s16, s17, s8, s1). It is run by spirv_mini_interp as a mesh shader (MeshEXT) and
with --mesh-compute, with push dword d holding 0x100 + d, so each word names the dword it came from.
"""
import struct
import unittest

import support
import rdna2spirv
import spirv_mini_interp as si

THREADS = 64
OUT = 0x10000                   # --mesh-compute: the workgroup's slot, at the output address
PUSH = [0x100 + d for d in range(36)]   # s0..s31, then the heap and code bases (unused here)

PROGRAM = """
v_mbcnt_lo_u32_b32 v0, -1, 0
v_mbcnt_hi_u32_b32 v0, -1, v0
s_mov_b32 m0, 0x1003
s_sendmsg sendmsg(MSG_GS_ALLOC_REQ)
s_barrier
v_mov_b32_e32 v1, s16
v_mov_b32_e32 v2, s17
v_mov_b32_e32 v3, s8
v_mov_b32_e32 v4, s1
v_cmpx_gt_u32_e32 3, v0
exp pos0 v1, v2, v3, v4 done
v_cmpx_eq_u32_e32 0, v0
v_mov_b32_e32 v9, 0x200400
exp prim v9, off, off, off done
s_endpgm
"""

# s16 -> push dword 8, s17 -> 9, s8 -> 0 (below the base), s1 -> push dword 1.
EXPECTED = (0x108, 0x109, 0, 0x101)


def _dxbc_psv0(threads, max_vertices, max_primitives):
    """A DXBC container with one PSV0 part: a triangle mesh shader (stage 13) and its limits."""
    r = bytearray(48)
    struct.pack_into("<HH", r, 12, max_vertices, max_primitives)
    r[24] = 13
    r[27] = 2
    struct.pack_into("<III", r, 36, threads, 1, 1)
    part = struct.pack("<I", len(r)) + bytes(r)
    hdr = 32 + 4
    return (b"DXBC" + bytes(16) + struct.pack("<III", 1, hdr + 8 + len(part), 1) + struct.pack("<I", hdr) +
            b"PSV0" + struct.pack("<I", len(part)) + part)


def _translate(mesh_compute):
    old = rdna2spirv.MESH_COMPUTE
    rdna2spirv.MESH_COMPUTE = mesh_compute
    try:
        lines = [ln.strip() for ln in PROGRAM.splitlines() if ln.strip()]
        spv = support.translate_lines("ms_root_window", lines, dxbc=_dxbc_psv0(THREADS, 64, 64))
        gen = rdna2spirv.translate.last_gen
        layout = gen.msc_layout if mesh_compute else None
        meta = rdna2spirv.shader_meta(gen, spv)
    finally:
        rdna2spirv.MESH_COMPUTE = old
    return spv, layout, meta


def _run(spv, push_constants):
    return si.run_workgroup(
        spv, THREADS, 64, push_constants=push_constants, memory={}, return_outputs=True,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%num_wg": (1, 1, 1), "%local_index": i, "%sg_lane": i % 64})


class MeshRootWindow(unittest.TestCase):
    def test_dword_mapping(self):
        for stage in ("vs", "ps", "gs", "cs"):
            with self.subTest(stage=stage):
                self.assertEqual([rdna2spirv.user_data_dword(stage, i) for i in range(32)], list(range(32)))
        ms = [rdna2spirv.user_data_dword("ms", i) for i in range(32)]
        self.assertEqual(ms[:8], list(range(8)))
        self.assertEqual(ms[8:16], [None] * 8)
        self.assertEqual(ms[16:], list(range(8, 24)))

    def test_mesh_shader_reads_the_graphics_window(self):
        spv, _, meta = _translate(False)
        self.assertIn("OpEntryPoint MeshEXT", spv)
        self.assertEqual(meta.get("root_window"), {"sgpr": 16, "dword": 8})
        _, outs = _run(spv, PUSH)
        for t in range(3):
            with self.subTest(vertex=t):
                self.assertEqual(outs[t]["%ms_out_pos"][4 * t:4 * t + 4], EXPECTED)

    def test_mesh_compute_reads_the_graphics_window(self):
        spv, lay, meta = _translate(True)
        self.assertIn("OpEntryPoint GLCompute", spv)
        self.assertEqual(meta.get("root_window"), {"sgpr": 16, "dword": 8})
        tail = [OUT, 0] if lay.get("address_dwords") else []
        memory, _ = _run(spv, PUSH + tail)
        for t in range(3):
            with self.subTest(vertex=t):
                v = OUT + lay["vertices"] + t * lay["vertex_stride"]
                self.assertEqual(tuple(memory.get(v + 4 * k) for k in range(4)), EXPECTED)

    def test_other_stages_record_no_window(self):
        spv = support.translate_lines("vs_root_window", ["v_mov_b32_e32 v1, s8", "s_endpgm"])
        meta = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, spv)
        self.assertNotIn("root_window", meta)


if __name__ == "__main__":
    unittest.main()
