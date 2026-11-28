"""A single-wave emulated program keeps its exchange barriers in the structured form (docs/design.md, "Wave64
emulation: single-wave barrier fallback").

`translate()` re-translates an emulated program as the round loop when an `OpControlBarrier` sits under a branch or
loop, because a barrier only part of the workgroup reaches hangs NVIDIA. A one-wave workgroup (`inline_all` plan)
does not need that: every guard around its barriers tests only `%pc`, which all 64 invocations hold alike. The
fixture puts a cross-half `v_readlane_b32` (an exchange behind a barrier) in a uniform loop and in a uniform branch.

    python -I tests/semantic/test_wave64_single_wave_barrier.py -v
"""
import os
import re
import sys
import unittest
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from test_loop_ballot_barrier import _translate_cs, rdna2spirv
from spirv_mini_interp import run_workgroup

FIXTURE = Path(HERE) / "fixtures" / "wave64_single_wave_barrier.asm"
_DEF = re.compile(r"^\s*(%[\w.]+) = (Op\w+)([^\n]*)$", re.M)
_HEAD = re.compile(r"^\s*(?:(%[\w.]+) = )?(OpLabel|OpSelectionMerge|OpLoopMerge|OpBranchConditional|OpControlBarrier)"
                   r"\b([^\n]*)$", re.M)


def module(lanes, structured=True):
    lines = [s for s in FIXTURE.read_text().splitlines() if s and not s.startswith(";")]
    lines.insert(0, "s_mov_b64 exec, -1")
    return _translate_cs("cs_single_wave_barrier", lines, threads=lanes, emulate=True, structured=structured)


def execute(text, lanes, s4):
    mem = run_workgroup(
        text, lanes, 32,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i % 32, "%local_index": i},
        push_constants=[0, 0, lanes * 4, 0x31016FAC, s4] + [0] * 27, memory={})
    return [mem[i * 4] for i in range(lanes)]


def barrier_guard_loads(text):
    """For every OpControlBarrier, the variables the conditions of its enclosing constructs load.

    A selection's condition is the OpBranchConditional after its OpSelectionMerge; a loop's are the conditional
    branches inside it that leave it or go back to its header."""
    defs = {name: (op, rest.split()) for name, op, rest in _DEF.findall(text)}

    def loads(value, seen):
        if value in seen or value not in defs:
            return set()
        seen.add(value)
        op, args = defs[value]
        if op == "OpLoad":
            return {args[1]}
        found = set()
        for arg in args:
            if arg.startswith("%"):
                found |= loads(arg, seen)
        return found

    events = [(m.start(), m.group(1), m.group(2), m.group(3).split()) for m in _HEAD.finditer(text)]
    label_at = {name: at for at, name, op, _ in events if op == "OpLabel"}
    constructs, header = [], None
    for i, (at, name, op, args) in enumerate(events):
        if op == "OpLabel":
            header = name
        elif op in ("OpSelectionMerge", "OpLoopMerge"):
            constructs.append((op, header, args[0], i))
    result = []
    for at, _, op, _ in events:
        if op != "OpControlBarrier":
            continue
        loaded = set()
        for kind, head, merge, index in constructs:
            if not label_at[head] <= at < label_at[merge]:
                continue
            if kind == "OpSelectionMerge":
                conds = [events[index + 1][3][0]]
            else:
                conds = [args[0] for b_at, _, b_op, args in events
                         if b_op == "OpBranchConditional" and label_at[head] <= b_at < label_at[merge]
                         and {merge, head} & set(args[1:])]
            for cond in conds:
                loaded |= loads(cond, set())
        result.append(loaded)
    return result


class SingleWaveBarrierTests(unittest.TestCase):
    def test_single_wave_stays_structured_with_pc_only_guards(self):
        text, gen = module(64)
        self.assertEqual(gen.control_flow, "emulated wave64 structured")
        self.assertEqual(gen.emulate_plan["mode"], "inline_all")
        self.assertNotIn("OpSwitch", text)
        # The barriers are under control flow (so the multi-wave fallback would fire), but only under `%pc`.
        self.assertTrue(rdna2spirv.barriers_under_control_flow(text))
        guards = barrier_guard_loads(text)
        self.assertTrue(any(guards), "no barrier inside the loop or the branch")
        for loaded in guards:
            self.assertLessEqual(loaded, {"%pc"})

    def test_structured_and_round_forms_agree(self):
        # lane 40 after each of the 3 iterations: 41 + 42 + 43; the branch adds lane 41's 41 + 3 + 100.
        structured, _ = module(64)
        rounds, gen = module(64, structured=False)
        self.assertEqual(gen.control_flow, "emulated wave64 rounds")
        for s4, want in ((0, 126), (1, 126 + 144)):
            with self.subTest(s4=s4):
                self.assertEqual(execute(structured, 64, s4), [want] * 64)
                self.assertEqual(execute(rounds, 64, s4), [want] * 64)

    def test_partial_single_wave_is_structured_too(self):
        text, gen = module(48)
        self.assertEqual(gen.control_flow, "emulated wave64 structured")
        self.assertEqual(gen.emulate_plan["mode"], "inline_all")
        self.assertEqual(execute(text, 48, 1), [126 + 144] * 48)


if __name__ == "__main__":
    unittest.main()
