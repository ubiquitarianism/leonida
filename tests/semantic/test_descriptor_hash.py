"""Shared descriptor vectors from plaaasma. Check full 32-bit hashes and probe the emitted lookup."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import support
import rdna2spirv as rd
import spirv_mini_interp as si

VECTORS = (
    ((0x00100000, 0, 0x90500fac, 0, 0, 0, 0, 0), 0x1cd4cd59, 345, 0x7ce036c5, 197),
    ((0x10100000, 0, 0x90500fac, 0, 0, 0, 0, 0), 0x0cd4cd59, 345, 0x646b8915, 277),
    ((0x20100000, 0, 0x90500fac, 0, 0, 0, 0, 0), 0x3cd4cd59, 345, 0xc2e843b6, 438),
    ((0x30100000, 0, 0x90500fac, 0, 0, 0, 0, 0), 0x2cd4cd59, 345, 0xe11420e3, 227),
    ((0x12345678, 0x9abcdef0, 0x0fedcba9, 0x87654321, 0x11111111, 0x22222222, 0x33333333, 0x44444444),
     0xbd4a977f, 383, 0x3ae592b1, 177),
    ((0,) * 8, 0x9be17165, 357, 0xbb10dc19, 25),
    ((0x00010001, 0x00400040, 0x10014fac, 0, *([0xffffffff] * 4)), 0x814225e4, 484, 0xd5398a35, 53),
    ((0xffffffff,) * 8, 0xb0e4285d, 93, 0x98a3f73a, 314),
)
OUTPUT = 0x8000


def table_and_expected(mode):
    table = [511] + [0xabababab] * (512 * 10)
    expected = []
    for i, (key, fnv, old_slot, mixed, new_slot) in enumerate(VECTORS):
        value, slot = (fnv, old_slot) if mode == "fnv1a" else (mixed, new_slot)
        assert value & 511 == slot
        # Start at the *published* home slot. Include collisions and a hit on the eighth probe.
        probe = 7 if i == 7 else 0
        while table[1 + 10 * ((slot + probe) & 511) + 8] != 0xabababab:
            probe += 1
        assert probe < 8
        start = 1 + 10 * ((slot + probe) & 511)
        table[start:start + 9] = [*key, 100 + i]
        table.extend(key)
        expected.extend((value, 100 + i))
    return table, expected


def readback_shader(mode):
    with patch.object(rd, "DESCRIPTOR_HASH", mode):
        g = rd.Gen("cs")
    g.emulate = True  # Single-invocation workgroups need no subgroup broadcast of the result.
    keys = []
    for i in range(8):
        off, ptr, key = g.tmp(), g.tmp(), g.tmp()
        g.emit(f"{off} = OpIAdd %uint %start {g.cu(i)}")
        g.emit(f"{ptr} = OpAccessChain %_ptr_SB_uint %desc_map {g.cu(0)} {off}")
        g.emit(f"{key} = OpLoad %uint {ptr}")
        keys.append(key)
    hashed = g.desc_hash(keys)
    index, _kind = g.desc_lookup(keys)      # tagged: the probe also returns the entry's kind, unused here
    g.emit("%hp = OpAccessChain %heap_ptr %pc %c0")
    g.emit("%heap = OpLoad %ulong %hp")
    for i, value in enumerate((hashed, index)):
        off, wide, addr, ptr = g.tmp(), g.tmp(), g.tmp(), g.tmp()
        g.emit(f"{off} = OpIAdd %uint %row {g.cu(OUTPUT + 4 * i)}")
        g.emit(f"{wide} = OpUConvert %ulong {off}")
        g.emit(f"{addr} = OpIAdd %ulong %heap {wide}")
        g.emit(f"{ptr} = OpConvertUToPtr %physical_word {addr}")
        g.emit(f"OpStore {ptr} {value} Aligned 4")
    g.cu(8)
    g.cu(1 + 512 * 10)
    asm = (HERE / "fixtures/descriptor_hash_lookup.asm").read_text(encoding="utf-8")
    constants = [f"{ident} = OpConstant %uint {value}" for value, ident in g.consts.items()]
    for marker, lines in (("CONSTANTS", constants), ("LOOKUP_AND_READBACK", g.body), ("LOOKUP_FUNCTION", g.desc_fn)):
        asm = asm.replace("; " + marker, "\n".join(lines))
    return "\n".join(line for line in asm.splitlines() if not line.startswith(";")) + "\n"


class DescriptorHashTest(unittest.TestCase):
    def test_entire_hash_and_probed_index(self):
        for mode in rd.DESCRIPTOR_HASHES:
            with self.subTest(mode=mode):
                asm = readback_shader(mode)
                table, expected = table_and_expected(mode)
                memory = {}
                for i in range(len(VECTORS)):
                    si.run_workgroup(asm, 1, 1, inputs_for=lambda _: {"%group": (i, 0, 0)},
                                     resources={"%desc_map": table}, push_constants=[0] * 36, memory=memory)
                self.assertEqual([memory[OUTPUT + 4 * i] for i in range(len(expected))], expected)

    def test_every_module_records_its_build_mode(self):
        for mode in rd.DESCRIPTOR_HASHES:
            with patch.object(rd, "DESCRIPTOR_HASH", mode):
                asm = support.translate_lines("ps_hash_marker", ["v_mov_b32 v0, 0",
                                                               "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"])
                meta = rd.shader_meta(rd.translate.last_gen, asm)
            self.assertEqual(meta.get("descriptor_hash"), None if mode == "fnv1a" else mode)


if __name__ == "__main__":
    unittest.main()
