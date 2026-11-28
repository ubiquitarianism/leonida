"""Check the pre-clamp counters in the runtime's embedded shader without a Vulkan device or SDK.

Regenerate the binary from runtime/src with:
    glslangValidator -V --target-env vulkan1.2 -Os --spirv-val -o indirect_clamp.spv indirect_clamp.comp
indirect_clamp_spv.h stores those bytes as little-endian uint32_t words, eight per line.
"""
from pathlib import Path
import re
import unittest


def mesh_counter_ops(header):
    # Follow the SPIR-V pointer for each counter atomic: load push.stats, then access Stats.data[word].
    # push.list_budget uses the same Stats type but is a different member; never count its atomics here.
    words = [int(w, 16) for w in re.findall(r"0x([0-9a-fA-F]+)u", header)]
    if len(words) < 5 or words[0] != 0x07230203:
        raise ValueError("invalid SPIR-V header")
    constants, loads, chains, push = {}, {}, {}, set()
    found = set()
    i = 5
    while i < len(words):
        length, op = words[i] >> 16, words[i] & 0xffff
        if not length or i + length > len(words):
            raise ValueError("invalid SPIR-V instruction")
        a = words[i + 1:i + length]
        i += length
        if op == 43:  # OpConstant
            constants[a[1]] = a[2]
        elif op == 59 and a[2] == 9:  # OpVariable, PushConstant
            push.add(a[1])
        elif op == 61:  # OpLoad
            loads[a[1]] = a[2]
        elif op == 65:  # OpAccessChain
            chains[a[1]] = a[2:]
        elif op in (234, 239):  # OpAtomicIAdd, OpAtomicUMax
            dst = chains.get(a[2], [])
            if len(dst) != 3 or constants.get(dst[1]) != 0:
                continue
            member = chains.get(loads.get(dst[0]), [])
            if len(member) == 2 and member[0] in push and constants.get(member[1]) == 4:
                word = constants[dst[2]]
                if word >= 8:
                    found.add((word, op))
    return found


class TestIndirectClampTelemetry(unittest.TestCase):
    def test_embedded_preclamp_mesh_counters(self):
        src = Path(__file__).resolve().parents[1] / "runtime" / "src"
        # Active mesh records + records requesting y/z > 1; max y and max z before clamping. Then the dispatch
        # records: wider than 65 535 groups in x, the widest x, cut at the device's limits, and all of them.
        self.assertEqual(mesh_counter_ops((src / "indirect_clamp_spv.h").read_text(encoding="ascii")),
                         {(8, 234), (9, 234), (10, 239), (11, 239), (12, 234), (13, 239), (14, 234), (15, 234)})


if __name__ == "__main__":
    unittest.main()
