"""Additional nested-loop and sibling-order tests for tools/cfg_structure.py."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import cfg_structure as cs


def test_nested_loop_with_early_exit_uses_cfg_run():
    succ = {0: [1], 1: [2, 4], 2: [3, 2], 3: [1, 4], 4: []}
    tree = cs.structure(succ)
    assert tree is not None

    b2_calls = 0

    def choose(b):
        nonlocal b2_calls
        if b == 0:
            return 1
        if b == 1:
            return 2 if b2_calls == 0 else 4
        if b == 2:
            b2_calls += 1
            return 2 if b2_calls == 1 else 3
        if b == 3:
            return 1
        if b == 4:
            return cs.END
        raise AssertionError(b)

    assert cs.run(tree, choose, 32) == [0, 1, 2, 2, 3, 1, 4]


def test_sibling_order_diamond_then_tail_executes_every_branch():
    # Block 0 branches to 1 or 2, both merge at 3, which exits at 4.
    # Reversing region children puts 3 before 1/2 and misses 3 on the second pass.
    succ = {0: [1, 2], 1: [3], 2: [3], 3: [4], 4: []}
    tree = cs.structure(succ)
    assert tree is not None

    for branch in (1, 2):
        def choose(b, pick=branch):
            if b == 0:
                return pick
            if b in (1, 2):
                return 3
            if b == 3:
                return 4
            if b == 4:
                return cs.END
            raise AssertionError(b)

        assert cs.run(tree, choose, 16) == [0, branch, 3, 4]


if __name__ == "__main__":
    test_nested_loop_with_early_exit_uses_cfg_run()
    test_sibling_order_diamond_then_tail_executes_every_branch()
