"""make_game.dotnet_sdk_ok() reads `dotnet --list-sdks` as text in the console's code page.

    python -I tests/test_make_game_dotnet_sdk.py

The child process is real: it writes its SDK list as bytes, including a byte that UTF-8 and cp1252 both refuse
(a Polish or Czech profile name in an SDK path). Only the executable is redirected to a python child, because
the test cannot install a dotnet; the decoding is the real subprocess.run(text=True) path. Before the fix, the
reader thread fails on the byte and the check raises TypeError instead of answering.
"""
import os
import shutil
import subprocess
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import make_game  # noqa: E402

REAL_RUN = subprocess.run


def child_for(stdout: bytes):
    """A subprocess.run replacement that starts a real python child printing `stdout` as raw bytes for
    `dotnet --list-sdks`. Every other argument, including text=True, is passed through unchanged."""
    script = f"import sys; sys.stdout.buffer.write({stdout!r})"

    def run(args, **kwargs):
        if list(args[:2]) == ["dotnet", "--list-sdks"]:
            args = [sys.executable, "-I", "-c", script]
        return REAL_RUN(args, **kwargs)
    return run


class DotnetSdkOkTest(unittest.TestCase):
    def check(self, stdout: bytes) -> bool:
        with mock.patch.object(make_game.shutil, "which", return_value="dotnet"), \
                mock.patch.object(make_game.subprocess, "run", side_effect=child_for(stdout)):
            return make_game.dotnet_sdk_ok()

    def test_an_sdk_8_or_newer_is_found(self):
        self.assertTrue(self.check(b"8.0.100 [C:\\Program Files\\dotnet\\sdk]\n"))

    def test_a_runtime_only_install_has_no_sdk(self):
        self.assertFalse(self.check(b"7.0.410 [C:\\Program Files\\dotnet\\sdk]\n"))

    def test_no_dotnet_on_path_has_no_sdk(self):
        with mock.patch.object(make_game.shutil, "which", return_value=None), \
                mock.patch.object(make_game.subprocess, "run", side_effect=AssertionError("must not run")):
            self.assertFalse(make_game.dotnet_sdk_ok())

    def test_an_undecodable_byte_in_an_sdk_path_does_not_stop_the_check(self):
        # 0x81 is undefined in cp1252 and is not valid UTF-8 on its own: the text decode fails on it.
        self.assertTrue(self.check(b"7.0.410 [C:\\dotnet\\sdk]\n8.0.100 [C:\\Users\\\x81ukasz\\dotnet\\sdk]\n"))

    def test_an_undecodable_byte_with_only_a_runtime_has_no_sdk(self):
        self.assertFalse(self.check(b"7.0.410 [C:\\Users\\\x81ukasz\\dotnet\\sdk]\n"))


if __name__ == "__main__":
    unittest.main()
