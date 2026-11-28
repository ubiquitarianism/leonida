"""No docs page repeats the same fenced block twice in a row.

Two copies of one code block, with nothing but blank lines between them, is what a merge of a
rebased copy leaves behind: the page shows the same snippet twice, and an edit to one copy
leaves the other stale next to it. That is how `docs/character_models.md` carried the same
`manifest.json` block twice (lines 312-320 and 322-330) until it was dropped.

The rule is deliberately narrow: the blocks must be identical, carry the same language tag, and be adjacent, so the
same command quoted again in a later section, two blocks that differ in one line, or one expression shown for two
languages are not reported.

    python -I tests/test_docs_adjacent_fenced_blocks.py

Synthetic Markdown only; no game data.
"""
import os
import re
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DOCS = os.path.join(ROOT, "docs")

FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


def fenced_blocks(lines):
    """[(opening fence index, closing fence index, info string, content lines)] for `lines`."""
    blocks, opened = [], None
    for index, line in enumerate(lines):
        match = FENCE.match(line)
        if opened is None:
            if match:
                marker = match.group(1)
                opened = (index, marker[0], len(marker), match.group(2).strip().lower(), [])
            continue
        # A closing fence is the same character, at least as long, and nothing after it.
        if match and match.group(1)[0] == opened[1] and len(match.group(1)) >= opened[2] \
                and not match.group(2).strip():
            blocks.append((opened[0], index, opened[3], opened[4]))
            opened = None
            continue
        opened[4].append(line)
    return blocks


def adjacent_repeats(text):
    """[(first block's line number, second block's line number)] for identical blocks back to back.

    Line numbers are 1-based and name the opening fence. Blocks with prose between them are not
    back to back, blocks with different info strings are different examples even when the text matches
    (a Python and a Ruby snippet can read the same), and empty blocks are ignored.
    """
    lines = text.splitlines()
    found = []
    blocks = fenced_blocks(lines)
    for (start_a, end_a, info_a, body_a), (start_b, _end_b, info_b, body_b) in zip(blocks, blocks[1:]):
        if body_a != body_b or info_a != info_b or not any(line.strip() for line in body_a):
            continue
        if all(not line.strip() for line in lines[end_a + 1:start_b]):
            found.append((start_a + 1, start_b + 1))
    return found


class AdjacentFencedBlocks(unittest.TestCase):
    def test_detector_finds_only_identical_adjacent_blocks(self):
        src = "\n".join([
            "# page",                                   # 1
            "",                                         # 2
            "```python",                                # 3
            "print('a')",                               # 4
            "```",                                      # 5
            "",                                         # 6
            "```python",                                # 7
            "print('a')",                               # 8
            "```",                                      # 9  -> 3 and 7 are a repeat
            "",                                         # 10
            "Some prose.",                              # 11
            "",                                         # 12
            "```python",                                # 13
            "print('a')",                               # 14
            "```",                                      # 15 same block, but prose above: allowed
            "",                                         # 16
            "```python",                                # 17
            "print('b')",                               # 18
            "```",                                      # 19
            "",                                         # 20
            "```python",                                # 21
            "print('b')",                               # 22
            "```",                                      # 23 -> 17 and 21 repeat
            "",                                         # 24
            "````markdown",                             # 25
            "```",                                      # 26 a shorter fence does not close it
            "print('c')",                               # 27
            "````",                                     # 28
            "",                                         # 29
            "~~~~",                                     # 30
            "print('d')",                               # 31
            "~~~~",                                     # 32
            "",                                         # 33
            "~~~python",                                # 34
            "print('e')",                               # 35
            "~~~",                                      # 36
            "",                                         # 37
            "~~~ruby",                                  # 38
            "print('e')",                               # 39
            "~~~",                                      # 40 same text, another language: allowed
        ])
        self.assertEqual(adjacent_repeats(src), [(3, 7), (17, 21)])

    def test_detector_ignores_one_line_differences_and_empty_blocks(self):
        src = "\n".join([
            "```sh",          # 1
            "run --one",      # 2
            "```",            # 3
            "```sh",          # 4
            "run --two",      # 5
            "```",            # 6
            "```",            # 7 empty
            "```",            # 8 empty
        ])
        self.assertEqual(adjacent_repeats(src), [])

    def test_no_page_repeats_a_block_back_to_back(self):
        if not os.path.isdir(DOCS):
            self.skipTest("no docs directory")
        pages = sorted(name for name in os.listdir(DOCS) if name.endswith(".md"))
        self.assertGreater(len(pages), 40, "the scan found almost no pages; is the layout different?")
        bad = []
        for name in pages:
            with open(os.path.join(DOCS, name), encoding="utf-8", errors="replace") as handle:
                for first, second in adjacent_repeats(handle.read()):
                    bad.append(f"docs/{name}: the block at line {first} is repeated at line {second}")
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
