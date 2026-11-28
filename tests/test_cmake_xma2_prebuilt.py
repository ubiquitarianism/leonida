"""-DLEONIDA_XMA2_DECODER_LIB=<path> links a prebuilt Rust XMA2 decoder without running cargo.

On Linux cross-builds, a distribution rustc package normally lacks rust-std for x86_64-pc-windows-msvc, so configure
falls back to muted XMA voices unless the user supplies a prebuilt leonida_xma2_decoder.lib or installs that target.
Passing a missing path or directory must fail configure loudly rather than silently producing muted audio.

Runs runtime/cmake/xma2_decoder.cmake with ``cmake -P`` (no compiler or cargo needed); skipped without cmake.
"""
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODULE = ROOT / "runtime" / "cmake" / "xma2_decoder.cmake"
CMAKE_LISTS = ROOT / "runtime" / "CMakeLists.txt"
CMAKE = shutil.which("cmake")


def run_resolver(lib_input, base_dir):
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "driver.cmake"
        script.write_text(
            f'include("{MODULE.as_posix()}")\n'
            'leonida_resolve_prebuilt_xma2_decoder("${TEST_LIB}" "${TEST_BASE}" _out)\n'
            'message(STATUS "RESOLVED=${_out}")\n',
            encoding="utf-8",
            newline="\n",
        )
        args = [
            CMAKE,
            f"-DTEST_LIB={lib_input}",
            f"-DTEST_BASE={Path(base_dir).as_posix()}",
            "-P",
            str(script),
        ]
        return subprocess.run(args, capture_output=True, text=True, timeout=60)


@unittest.skipUnless(CMAKE, "cmake not on PATH")
class PrebuiltXma2ResolverTest(unittest.TestCase):
    def test_empty_value_leaves_output_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = run_resolver("", tmp)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("-- RESOLVED=\n", r.stdout)

    def test_existing_absolute_file_resolves(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib = Path(tmp) / "leonida_xma2_decoder.lib"
            lib.write_bytes(b"!<arch>\n")
            r = run_resolver(lib.as_posix(), tmp)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn(f"-- RESOLVED={lib.as_posix()}\n", r.stdout)

    def test_existing_relative_file_resolves_against_base_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            sub = Path(tmp) / "prebuilt"
            sub.mkdir()
            lib = sub / "leonida_xma2_decoder.lib"
            lib.write_bytes(b"!<arch>\n")
            r = run_resolver("prebuilt/leonida_xma2_decoder.lib", tmp)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn(f"-- RESOLVED={lib.as_posix()}\n", r.stdout)

    def test_existing_path_with_spaces_resolves(self):
        with tempfile.TemporaryDirectory() as tmp:
            sub = Path(tmp) / "xma prebuilt dir"
            sub.mkdir()
            lib = sub / "leonida xma2.lib"
            lib.write_bytes(b"!<arch>\n")
            r = run_resolver(lib.as_posix(), tmp)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn(f"-- RESOLVED={lib.as_posix()}\n", r.stdout)

    def test_missing_file_fails_configure(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "does_not_exist.lib"
            r = run_resolver(missing.as_posix(), tmp)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("LEONIDA_XMA2_DECODER_LIB", r.stderr)
            self.assertIn("does_not_exist.lib", r.stderr)

    def test_directory_path_fails_configure(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "not_a_file"
            folder.mkdir()
            r = run_resolver(folder.as_posix(), tmp)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("LEONIDA_XMA2_DECODER_LIB", r.stderr)


class CMakeListsXma2PrebuiltWiringTest(unittest.TestCase):
    def test_cmakelists_resolves_prebuilt_before_cargo_and_preserves_target_contract(self):
        text = CMAKE_LISTS.read_text(encoding="utf-8")
        self.assertIn('set(LEONIDA_XMA2_DECODER_LIB "" CACHE FILEPATH', text)
        self.assertIn('include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/xma2_decoder.cmake")', text)
        self.assertIn("leonida_resolve_prebuilt_xma2_decoder(", text)

        resolve_pos = text.index("leonida_resolve_prebuilt_xma2_decoder(")
        cargo_pos = text.index("find_program(LEONIDA_CARGO cargo)")
        imported_pos = text.index("add_library(leonida_xma2_decoder STATIC IMPORTED)")
        require_pos = text.index("if(LEONIDA_REQUIRE_XMA2_DECODER AND NOT TARGET leonida_xma2_decoder)")

        self.assertLess(resolve_pos, cargo_pos)
        self.assertLess(cargo_pos, imported_pos)
        self.assertLess(imported_pos, require_pos)

        block = text[resolve_pos:require_pos]
        self.assertRegex(
            block,
            re.compile(r"if\(_xma_lib_resolved\).*?else\(\).*?find_program\(LEONIDA_CARGO cargo\)", re.S),
        )
        self.assertIn('IMPORTED_LOCATION "${_xma_lib_resolved}"', block)
        self.assertIn("LEONIDA_HAVE_XMA2_DECODER=1", block)
        self.assertIn("synchronization", block)


if __name__ == "__main__":
    unittest.main()
