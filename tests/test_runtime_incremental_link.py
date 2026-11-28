"""The runtime fixtures that can have no base relocations link with /INCREMENTAL:NO (no build run).

VS 2026 18.6's linker (MSVC 14.51) crashes with "LNK1000: Internal error during IncrCalcPtrs" on an incremental
relink of an image with no base relocations. CMake's Ninja and Makefile generators relink every target once, with only
its manifest resource changed, in each clean Debug or RelWithDebInfo build (default link flags /debug /INCREMENTAL).
A fixture linked with /FIXED has its relocations stripped, and one linked with /NODEFAULTLIB has no CRT and often no
absolute address to relocate, so each of them links with /INCREMENTAL:NO.
https://developercommunity.visualstudio.com/t/MSVC-v1451-Linker-Crash-in-IncrCalcPtrs/11090307

    python -I tests/test_runtime_incremental_link.py
"""
import os
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
CMAKE = Path(os.environ.get("LEONIDA_RUNTIME_CMAKE", str(ROOT / "runtime" / "CMakeLists.txt")))


def link_options(text):
    """Each target named in a target_link_options call, with every option the calls give it."""
    options = {}
    for target, args in re.findall(r"target_link_options\(\s*([A-Za-z0-9_]+)\s+([^)]*)\)", text):
        options.setdefault(target, []).extend(args.split())
    return options


class FixturesWithoutRelocationsLinkNonIncrementally(unittest.TestCase):
    def test_every_fixed_or_crt_free_target_links_with_incremental_no(self):
        options = link_options(CMAKE.read_text(encoding="utf-8"))
        targets = {t for t, opts in options.items() if "/FIXED" in opts or "/NODEFAULTLIB" in opts}
        # The scan has to see the fixtures that crashed the linker, or it checks nothing.
        self.assertLessEqual({"tiny_game_fixed", "dll_thread_calls", "gameinput_host"}, targets)
        missing = sorted(t for t in targets if "/INCREMENTAL:NO" not in options[t])
        self.assertEqual(missing, [], "linked with /FIXED or /NODEFAULTLIB but not /INCREMENTAL:NO")

    def test_the_scan_reads_options_across_lines(self):
        text = ("target_link_options(a PRIVATE /FIXED\n  /INCREMENTAL:NO)\n"
                "target_link_options(b PRIVATE /NODEFAULTLIB)\n")
        self.assertEqual(link_options(text), {"a": ["PRIVATE", "/FIXED", "/INCREMENTAL:NO"],
                                              "b": ["PRIVATE", "/NODEFAULTLIB"]})


if __name__ == "__main__":
    unittest.main()
