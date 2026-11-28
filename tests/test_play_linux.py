import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = Path("d3d12") / "target"
CROSS = TARGET / "x86_64-pc-windows-msvc" / "diag" / "leonida.exe"
SCRIPT = ROOT / "scripts" / "play-linux.sh"
DXVK = Path("files", "lib", "wine", "dxvk", "x86_64-windows")
PROTON_VKD3D = Path("files", "lib", "wine", "vkd3d-proton", "x86_64-windows")
NEW_STRUCTURIZER = b"MZ vkd3d-proton legacy_structurizer"
SYSTEM_TOOLS = (Path("/usr/share/steam/compatibilitytools.d"), Path("/usr/local/share/steam/compatibilitytools.d"))

STUB_WINE = """#!/bin/sh
if [ "$1" = wineboot ]; then
    mkdir -p "$WINEPREFIX/drive_c/windows/system32"
    exit 0
fi
for a in "$@"; do printf '%s\\n' "$a"; done > "$LEONIDA_STUB_LOG"
exit 0
"""

PROTON_WINE = """#!/bin/sh
printf '%s\\n' "$*" >> "$LEONIDA_TEST_LOG"
printf '%s\\n' "$WINEDLLOVERRIDES" > "$LEONIDA_TEST_LOG.overrides"
printf '%s\\n' "${WINEDLLPATH:-}" > "$LEONIDA_TEST_LOG.dllpath"
printf '%s\\n' "${LD_LIBRARY_PATH:-}" > "$LEONIDA_TEST_LOG.ldpath"
printf '%s\\n' "${VKD3D_SHADER_CACHE_PATH:-}" > "$LEONIDA_TEST_LOG.shadercache"
if [ "$1" = wineboot ]; then
    [ -n "${LEONIDA_TEST_WINEBOOT_FAILS:-}" ] && exit 3
    mkdir -p "$WINEPREFIX/drive_c/windows/system32"
    exit 0
fi
echo "runtime error output" >&2
echo "runtime output"
exit 0
"""

SYSTEM_WINE = """#!/bin/sh
printf '%s\\n' "$*" >> "$LEONIDA_TEST_LOG.system"
exit 0
"""


WINE_RUNS = """#!/bin/sh
if [ "$1" = --version ]; then
    echo "wine-99.0 (Staging)"
    exit 0
fi
printf '%s\\n' "$*" >> "$LEONIDA_TEST_LOG"
printf '%s\\n' "$WINEDLLOVERRIDES" > "$LEONIDA_TEST_LOG.overrides"
printf '%s\\n' "${WINEDLLPATH:-}" > "$LEONIDA_TEST_LOG.dllpath"
if [ "$1" = wineboot ]; then
    mkdir -p "$WINEPREFIX/drive_c/windows/system32"
fi
exit 0
"""

SYSTEM_DXVK = tuple(Path(d) for d in ("/usr/share/dxvk/x64", "/usr/lib/dxvk/x64", "/usr/lib/dxvk/wine64",
                                      "/usr/share/dxvk/x86_64-windows", "/usr/lib/dxvk/x86_64-windows"))


def system_proton_installed():
    return any(tools.is_dir() and any(tools.iterdir()) for tools in SYSTEM_TOOLS)


def system_dxvk_installed():
    return any((folder / "dxgi.dll").is_file() for folder in SYSTEM_DXVK)


@unittest.skipIf(os.name == "nt" or not os.path.exists("/bin/sh"), "POSIX shell script")
class PlayLinuxTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.repo = self.tmp / "repo"
        (self.repo / "scripts").mkdir(parents=True)
        shutil.copy(ROOT / "scripts" / "play-linux.sh", self.repo / "scripts" / "play-linux.sh")
        self.game = self.tmp / "game" / "game_scarlett_master.exe"
        self.game.parent.mkdir()
        self.game.write_bytes(b"MZ")
        self.proton = self.tmp / "proton"
        (self.proton / "files" / "bin").mkdir(parents=True)
        (self.proton / "files" / "bin" / "wine").write_text(STUB_WINE, encoding="utf-8", newline="\n")
        (self.proton / "files" / "bin" / "wine").chmod(0o755)
        (self.proton / DXVK).mkdir(parents=True)
        (self.proton / DXVK / "dxgi.dll").write_bytes(b"dxgi")
        self.log = self.tmp / "wine.log"

    def touch(self, relative):
        path = self.repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
        return path

    def vkd3d(self, folder):
        for dll in ("d3d12.dll", "d3d12core.dll"):
            self.touch(Path(folder) / dll)

    def play(self, **extra):
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(self.tmp / "home"),
               "LEONIDA_PROTON": str(self.proton), "LEONIDA_STUB_LOG": str(self.log), **extra}
        return subprocess.run(["/bin/sh", str(self.repo / "scripts" / "play-linux.sh"), str(self.game)],
                              env=env, capture_output=True, text=True, timeout=60)

    def started(self):
        return self.log.read_text(encoding="utf-8").splitlines()

    def test_no_runtime_names_the_build_script(self):
        result = self.play()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("build-d3d12-linux.sh", result.stderr)
        self.assertIn("LEONIDA_RUNTIME", result.stderr)
        self.assertFalse(self.log.exists())

    def test_checkout_layout(self):
        exe = self.touch(CROSS)
        self.vkd3d(TARGET / "vkd3d-proton")
        result = self.play()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.started()[:2], [str(exe), "run"])

    def test_windows_build_copied_into_the_checkout(self):
        exe = self.touch(TARGET / "diag" / "leonida.exe")
        self.vkd3d(TARGET / "vkd3d-proton")
        result = self.play()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.started()[0], str(exe))

    def test_package_layout_comes_first(self):
        self.touch(CROSS)
        exe = self.touch(Path("scripts") / "leonida.exe")
        self.vkd3d(Path("scripts") / "vkd3d-proton")
        result = self.play()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.started()[0], str(exe))

    def test_leonida_runtime_wins(self):
        self.touch(CROSS)
        self.vkd3d(TARGET / "vkd3d-proton")
        own = self.touch("elsewhere/leonida.exe")
        result = self.play(LEONIDA_RUNTIME=str(own))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.started()[0], str(own))
        missing = self.play(LEONIDA_RUNTIME=str(self.tmp / "nope.exe"))
        self.assertEqual(missing.returncode, 1)
        self.assertIn("LEONIDA_RUNTIME", missing.stderr)

    def proton_vkd3d(self, core, build=None, lib="lib"):
        folder = self.proton / "files" / lib / PROTON_VKD3D.relative_to(Path("files", "lib"))
        folder.mkdir(parents=True)
        (folder / "d3d12.dll").write_bytes(b"MZ proton d3d12")
        (folder / "d3d12core.dll").write_bytes(core)
        if build is not None:
            (folder.parent / "version").write_text(f" 0123abcd vkd3d-proton ({build})\n", encoding="utf-8")
        return folder

    def installed(self, dll):
        system32 = self.tmp / "home" / ".local" / "share" / "leonida" / "prefix" / "drive_c" / "windows" / "system32"
        return (system32 / dll).read_bytes()

    def test_a_proton_vkd3d_with_the_new_structurizer_replaces_the_bundled_one(self):
        self.touch(CROSS)
        self.vkd3d(TARGET / "vkd3d-proton")
        folder = self.proton_vkd3d(NEW_STRUCTURIZER, "vkd3d-1.1-9999-g0123abcd")
        result = self.play()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("the VKD3D-Proton of proton (vkd3d-1.1-9999-g0123abcd)", result.stdout)
        self.assertNotIn("structurizer", result.stderr)
        for dll in ("d3d12.dll", "d3d12core.dll"):
            self.assertEqual(self.installed(dll), (folder / dll).read_bytes())

    def test_a_proton_vkd3d_under_lib64_replaces_the_bundled_one(self):
        self.touch(CROSS)
        self.vkd3d(TARGET / "vkd3d-proton")
        folder = self.proton_vkd3d(NEW_STRUCTURIZER, "vkd3d-1.1-9999-g0123abcd", lib="lib64")
        result = self.play()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("the VKD3D-Proton of proton (vkd3d-1.1-9999-g0123abcd)", result.stdout)
        for dll in ("d3d12.dll", "d3d12core.dll"):
            self.assertEqual(self.installed(dll), (folder / dll).read_bytes())

    def test_an_old_layout_proton_vkd3d_under_lib64_replaces_the_bundled_one(self):
        self.touch(CROSS)
        self.vkd3d(TARGET / "vkd3d-proton")
        folder = self.proton / "files" / "lib64" / "wine" / "vkd3d-proton"
        folder.mkdir(parents=True)
        (folder / "d3d12.dll").write_bytes(b"MZ proton d3d12")
        (folder / "d3d12core.dll").write_bytes(NEW_STRUCTURIZER)
        (folder / "version").write_text(" 0123abcd vkd3d-proton (vkd3d-1.1-9999-g0123abcd)\n", encoding="utf-8")
        result = self.play()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("the VKD3D-Proton of proton (vkd3d-1.1-9999-g0123abcd)", result.stdout)
        for dll in ("d3d12.dll", "d3d12core.dll"):
            self.assertEqual(self.installed(dll), (folder / dll).read_bytes())

    def test_a_proton_vkd3d_stands_in_for_a_missing_bundled_one(self):
        self.touch(CROSS)
        folder = self.proton_vkd3d(NEW_STRUCTURIZER)
        result = self.play()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("the VKD3D-Proton of proton", result.stdout)
        self.assertEqual(self.installed("d3d12core.dll"), (folder / "d3d12core.dll").read_bytes())

    def test_a_bundled_vkd3d_with_the_new_structurizer_is_kept(self):
        self.touch(CROSS)
        bundled = self.repo / TARGET / "vkd3d-proton"
        self.vkd3d(TARGET / "vkd3d-proton")
        (bundled / "d3d12core.dll").write_bytes(NEW_STRUCTURIZER + b" bundled")
        self.proton_vkd3d(NEW_STRUCTURIZER)
        result = self.play()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("the bundled VKD3D-Proton", result.stdout)
        self.assertEqual(self.installed("d3d12core.dll"), (bundled / "d3d12core.dll").read_bytes())

    @unittest.skipIf(system_proton_installed(), "a Proton is installed system-wide on this machine")
    def test_without_a_newer_vkd3d_the_start_says_what_to_install(self):
        self.touch(CROSS)
        self.vkd3d(TARGET / "vkd3d-proton")
        self.proton_vkd3d(b"MZ vkd3d-proton 3.0")
        result = self.play()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("new structurizer", result.stderr)
        self.assertIn("LEONIDA_VKD3D", result.stderr)
        self.assertIn("the bundled VKD3D-Proton", result.stdout)
        self.assertEqual(self.installed("d3d12core.dll"), b"x")

    @unittest.skipIf(system_proton_installed(), "a Proton is installed system-wide on this machine")
    def test_another_proton_lends_its_vkd3d_when_none_was_chosen(self):
        self.touch(CROSS)
        self.vkd3d(TARGET / "vkd3d-proton")
        tools = self.tmp / "home" / ".local" / "share" / "Steam" / "compatibilitytools.d"
        tools.mkdir(parents=True)
        older = tools / "GE-Proton98-1"
        shutil.copytree(self.proton, tools / "GE-Proton99-4")
        shutil.copytree(self.proton, older)
        (older / PROTON_VKD3D).mkdir(parents=True)
        for dll in ("d3d12.dll", "d3d12core.dll"):
            (older / PROTON_VKD3D / dll).write_bytes(NEW_STRUCTURIZER + dll.encode())
        result = self.play(LEONIDA_PROTON="")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("under GE-Proton99-4", result.stdout)
        self.assertIn("the VKD3D-Proton of GE-Proton98-1", result.stdout)
        self.assertEqual(self.installed("d3d12core.dll"), NEW_STRUCTURIZER + b"d3d12core.dll")

    @unittest.skipIf(system_proton_installed(), "a Proton is installed system-wide on this machine")
    def test_another_proton_lends_its_vkd3d_to_a_chosen_one(self):
        self.touch(CROSS)
        self.vkd3d(TARGET / "vkd3d-proton")
        other = self.tmp / "home" / ".local" / "share" / "Steam" / "compatibilitytools.d" / "GE-Proton99-4"
        shutil.copytree(self.proton, other)
        (other / PROTON_VKD3D).mkdir(parents=True)
        for dll in ("d3d12.dll", "d3d12core.dll"):
            (other / PROTON_VKD3D / dll).write_bytes(NEW_STRUCTURIZER + dll.encode())
        result = self.play()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("structurizer", result.stderr)
        self.assertIn("under proton", result.stdout)
        self.assertIn("the VKD3D-Proton of GE-Proton99-4", result.stdout)
        self.assertEqual(self.installed("d3d12core.dll"), NEW_STRUCTURIZER + b"d3d12core.dll")

    def test_leonida_vkd3d_wins_over_a_proton_vkd3d(self):
        self.touch(CROSS)
        own = self.repo / "own-vkd3d"
        self.vkd3d("own-vkd3d")
        self.proton_vkd3d(NEW_STRUCTURIZER)
        result = self.play(LEONIDA_VKD3D=str(own))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("structurizer", result.stderr)
        self.assertIn(f"the VKD3D-Proton in {own} (LEONIDA_VKD3D)", result.stdout)
        self.assertEqual(self.installed("d3d12core.dll"), (own / "d3d12core.dll").read_bytes())

    @unittest.skipIf(system_proton_installed(), "a Proton is installed system-wide on this machine")
    def test_missing_vkd3d_names_the_fetch(self):
        self.touch(CROSS)
        result = self.play()
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("fetch_vkd3d_proton.py", result.stderr)
        self.assertIn("build-d3d12-linux.sh", result.stderr)


@unittest.skipIf(os.name == "nt" or not os.path.exists("/bin/sh"), "POSIX shell script")
class PlayLinuxProtonTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.home = self.tmp / "home"
        self.steam = self.home / ".local" / "share" / "Steam"
        self.steam.mkdir(parents=True)
        self.log = self.tmp / "log"
        bin_dir = self.tmp / "bin"
        bin_dir.mkdir()
        for name in ("wine", "wine64", "wineboot", "wineserver"):
            stub = bin_dir / name
            stub.write_text(SYSTEM_WINE, encoding="utf-8", newline="\n")
            stub.chmod(0o755)
        self.path = os.pathsep.join([str(bin_dir), os.environ.get("PATH", "/usr/bin:/bin")])
        self.package = self.tmp / "package"
        (self.package / "vkd3d-proton").mkdir(parents=True)
        for dll in ("d3d12.dll", "d3d12core.dll"):
            (self.package / "vkd3d-proton" / dll).write_bytes(b"vkd3d " + dll.encode())
        (self.package / "leonida.exe").write_bytes(b"MZ")
        self.game = self.tmp / "game" / "game_scarlett_master.exe"
        self.game.parent.mkdir()
        self.game.write_bytes(b"MZ")

    def proton(self, folder, version=None, dxvk=True, wine=True, lib=True, lib_name="lib"):
        folder = Path(folder)
        (folder / "files" / "bin").mkdir(parents=True)
        if lib:
            (folder / "files" / lib_name / "wine").mkdir(parents=True, exist_ok=True)
        if wine:
            stub = folder / "files" / "bin" / "wine"
            stub.write_text(PROTON_WINE, encoding="utf-8", newline="\n")
            stub.chmod(0o755)
            server = folder / "files" / "bin" / "wineserver"
            server.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8", newline="\n")
            server.chmod(0o755)
        if dxvk:
            dxgi = folder / "files" / lib_name / DXVK.relative_to(Path("files", "lib"))
            dxgi.mkdir(parents=True)
            (dxgi / "dxgi.dll").write_bytes(b"dxvk dxgi " + folder.name.encode())
        if version is not None:
            (folder / "version").write_text(version + "\n", encoding="utf-8")
        return folder

    def run_script(self, *args, **env):
        environment = {"PATH": self.path, "HOME": str(self.home), "LEONIDA_TEST_LOG": str(self.log),
                       "LEONIDA_RUNTIME": str(self.package / "leonida.exe"),
                       "LEONIDA_VKD3D": str(self.package / "vkd3d-proton")}
        environment.update(env)
        return subprocess.run(["/bin/sh", str(SCRIPT), *args], env=environment, capture_output=True, text=True,
                              timeout=60)

    def found(self, **env):
        p = self.run_script("--find-proton", **env)
        self.assertEqual(p.returncode, 0, p.stderr)
        name, folder = p.stdout.strip().split("\t")
        return name, Path(folder)

    def calls(self):
        return self.log.read_text(encoding="utf-8").splitlines() if self.log.exists() else []

    def assert_system_wine_unused(self):
        self.assertFalse(Path(str(self.log) + ".system").exists(), "the system's Wine was started")

    def test_the_newest_complete_proton_is_picked(self):
        tools = self.steam / "compatibilitytools.d"
        self.proton(tools / "GE-Proton98-30", "1700000000 GE-Proton98-30")
        newest = self.proton(tools / "GE-Proton99-4", "1700000000 GE-Proton99-4")
        self.proton(tools / "GE-Proton99-12-partial", "1700000000 GE-Proton99-12", dxvk=False, lib=False)
        self.proton(tools / "GE-Proton99-13-nowine", "1700000000 GE-Proton99-13", wine=False)
        (tools / "not-a-proton").mkdir()
        self.proton(self.steam / "steamapps" / "common" / "Proton - Experimental",
                    "1700000000 experimental-99.0-20261120")
        self.proton(self.steam / "steamapps" / "common" / "Proton 99.0", "1700000000 proton-99.0-3")
        self.assertEqual(self.found(), ("GE-Proton99-4", newest.resolve()))

    @unittest.skipIf(system_proton_installed(), "a Proton is installed system-wide on this machine")
    def test_list_protons_names_every_usable_build_newest_first_once(self):
        tools = self.steam / "compatibilitytools.d"
        older = self.proton(tools / "GE-Proton98-30", "1700000000 GE-Proton98-30")
        newest = self.proton(tools / "GE-Proton99-4", "1700000000 GE-Proton99-4")
        self.proton(tools / "Proton-Wineland Latest", "1700000000 wineland-99.0-20261127", dxvk=False, lib=False)
        valve = self.proton(self.steam / "steamapps" / "common" / "Proton 99.0", "1700000000 proton-99.0-3")
        (self.home / ".steam").mkdir()
        (self.home / ".steam" / "root").symlink_to(self.steam)
        p = self.run_script("--list-protons", LEONIDA_PROTON=str(older))
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual([line.split("\t") for line in p.stdout.splitlines()],
                         [["GE-Proton99-4", str(newest.resolve())], ["proton-99.0-3", str(valve.resolve())],
                          ["GE-Proton98-30", str(older.resolve())]])

    @unittest.skipIf(system_proton_installed(), "a Proton is installed system-wide on this machine")
    def test_list_protons_without_any_build_is_empty(self):
        p = self.run_script("--list-protons")
        self.assertEqual((p.returncode, p.stdout), (0, ""))

    def test_valve_proton_is_ranked_by_its_version(self):
        common = self.steam / "steamapps" / "common"
        self.proton(common / "Proton 99.0", "1700000000 proton-99.0-3")
        experimental = self.proton(common / "Proton - Experimental", "1700000000 experimental-99.0-20261120")
        self.proton(common / "Proton EasyAntiCheat Runtime", dxvk=False, wine=False)
        self.assertEqual(self.found(), ("experimental-99.0-20261120", experimental.resolve()))

    def test_without_a_version_file_the_folder_name_is_the_version(self):
        tools = self.steam / "compatibilitytools.d"
        self.proton(tools / "GE-Proton98-1")
        newest = self.proton(tools / "GE-Proton99-2")
        self.assertEqual(self.found(), ("GE-Proton99-2", newest.resolve()))

    def test_other_steam_folders_libraries_and_extra_paths_are_searched(self):
        library = self.tmp / "games" / "SteamLibrary"
        steam_root = self.home / ".steam" / "root"
        (steam_root / "steamapps").mkdir(parents=True)
        (steam_root / "steamapps" / "libraryfolders.vdf").write_text(
            '"libraryfolders"\n{\n\t"1"\n\t{\n\t\t"path"\t\t"' + str(library) + '"\n\t}\n}\n', encoding="utf-8")
        in_library = self.proton(library / "steamapps" / "common" / "Proton 99.0", "1700000000 proton-99.0-7")
        self.assertEqual(self.found(), ("proton-99.0-7", in_library.resolve()))
        extra = self.tmp / "extra"
        flatpak = self.home / ".var" / "app" / "com.valvesoftware.Steam" / "data" / "Steam"
        in_flatpak = self.proton(flatpak / "compatibilitytools.d" / "GE-Proton99-8", "1 GE-Proton99-8")
        self.assertEqual(self.found(), ("GE-Proton99-8", in_flatpak.resolve()))
        in_extra = self.proton(extra / "GE-Proton99-9", "1 GE-Proton99-9")
        self.assertEqual(self.found(STEAM_EXTRA_COMPAT_TOOLS_PATHS=f"/nowhere:{extra}"),
                         ("GE-Proton99-9", in_extra.resolve()))

    def test_leonida_proton_wins_and_takes_the_folder_or_its_wine(self):
        tools = self.steam / "compatibilitytools.d"
        self.proton(tools / "GE-Proton99-9", "1 GE-Proton99-9")
        chosen = self.proton(self.tmp / "my proton", "1 GE-Proton10-25")
        for value in (str(chosen), str(chosen) + "/", str(chosen / "files" / "bin" / "wine")):
            with self.subTest(value=value):
                self.assertEqual(self.found(LEONIDA_PROTON=value), ("GE-Proton10-25", chosen.resolve()))

    def test_a_leonida_proton_that_is_no_proton_is_refused(self):
        self.proton(self.steam / "compatibilitytools.d" / "GE-Proton99-9", "1 GE-Proton99-9")
        partial = self.proton(self.tmp / "partial proton", dxvk=False, lib=False)
        for value in (str(partial), str(self.tmp / "missing")):
            with self.subTest(value=value):
                p = self.run_script(str(self.game), LEONIDA_PROTON=value)
                self.assertEqual(p.returncode, 1)
                self.assertIn("is not a Proton build", p.stderr)
                self.assertEqual(self.calls(), [])
        self.assert_system_wine_unused()

    def test_a_proton_without_dxvk_runs_with_the_dxgi_of_another(self):
        tools = self.steam / "compatibilitytools.d"
        lender = self.proton(tools / "GE-Proton98-30", "1 GE-Proton98-30")
        self.proton(tools / "DW-Proton Latest", "1 dwproton-99.0-17", dxvk=False)
        p = self.run_script(str(self.game))
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("under dwproton-99.0-17", p.stdout)
        self.assertIn("with DXVK's dxgi of GE-Proton98-30", p.stdout)
        system32 = self.home / ".local" / "share" / "leonida" / "prefix" / "drive_c" / "windows" / "system32"
        self.assertEqual((system32 / "dxgi.dll").read_bytes(), (lender / DXVK / "dxgi.dll").read_bytes())
        self.assert_system_wine_unused()

    def test_the_game_runs_under_the_proton_wine_with_dxvk_dxgi(self):
        proton = self.proton(self.steam / "compatibilitytools.d" / "GE-Proton99-4", "1 GE-Proton99-4")
        p = self.run_script(str(self.game), "--series-x", WINEDLLOVERRIDES="mscoree=")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("series-x under GE-Proton99-4", p.stdout)
        self.assertEqual(self.calls(), ["wineboot -u", f"{self.package / 'leonida.exe'} run {self.game} "
                                        "--execute --console series-x"])
        self.assertEqual(Path(str(self.log) + ".overrides").read_text(encoding="utf-8").strip(),
                         "d3d12,d3d12core,dxgi=n;mscoree=")
        system32 = self.home / ".local" / "share" / "leonida" / "prefix" / "drive_c" / "windows" / "system32"
        self.assertEqual((system32 / "dxgi.dll").read_bytes(), (proton / DXVK / "dxgi.dll").read_bytes())
        for dll in ("d3d12.dll", "d3d12core.dll"):
            self.assertEqual((system32 / dll).read_bytes(), (self.package / "vkd3d-proton" / dll).read_bytes())
        self.assert_system_wine_unused()

    def test_the_proton_wine_finds_its_own_libraries(self):
        proton = self.proton(self.steam / "compatibilitytools.d" / "GE-Proton99-4", "1 GE-Proton99-4")
        lib = proton / "files" / "lib"
        for sub in ("vkd3d/x86_64-windows", "wine/x86_64-windows", "x86_64-linux-gnu", "i386-linux-gnu"):
            (lib / sub).mkdir(parents=True, exist_ok=True)
        p = self.run_script(str(self.game), WINEDLLPATH="/extra/dlls", LD_LIBRARY_PATH="/extra/lib")
        self.assertEqual(p.returncode, 0, p.stderr)
        real = lib.resolve()
        self.assertEqual(Path(str(self.log) + ".dllpath").read_text(encoding="utf-8").strip(),
                         f"{real / 'vkd3d'}:{real / 'wine'}:/extra/dlls")
        self.assertEqual(Path(str(self.log) + ".ldpath").read_text(encoding="utf-8").strip(),
                         f"{real / 'x86_64-linux-gnu'}:{real / 'i386-linux-gnu'}:/extra/lib")

    def test_a_proton_with_only_lib64_is_listed_found_and_runs_with_its_own_libraries(self):
        proton = self.proton(self.steam / "compatibilitytools.d" / "Proton-lib64", "1 GE-Proton99-4",
                             lib_name="lib64")
        lib64 = proton / "files" / "lib64"
        (lib64 / "vkd3d" / "x86_64-windows").mkdir(parents=True)
        listed = self.run_script("--list-protons")
        self.assertEqual(listed.returncode, 0, listed.stderr)
        self.assertIn(str(proton.resolve()), listed.stdout)
        self.assertEqual(self.found(), ("GE-Proton99-4", proton.resolve()))
        p = self.run_script(str(self.game), WINEDLLPATH="/extra/dlls", LD_LIBRARY_PATH="/extra/lib")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("with DXVK's dxgi", p.stdout)
        real = lib64.resolve()
        self.assertEqual(Path(str(self.log) + ".dllpath").read_text(encoding="utf-8").strip(),
                         f"{real / 'vkd3d'}:{real / 'wine'}:/extra/dlls")
        self.assertEqual(Path(str(self.log) + ".ldpath").read_text(encoding="utf-8").strip(), f"{real}:/extra/lib")
        system32 = self.home / ".local" / "share" / "leonida" / "prefix" / "drive_c" / "windows" / "system32"
        self.assertEqual((system32 / "dxgi.dll").read_bytes(),
                         (lib64 / "wine" / "dxvk" / "x86_64-windows" / "dxgi.dll").read_bytes())
        self.assert_system_wine_unused()

    def test_with_lib_and_lib64_the_lib64_libraries_come_first(self):
        proton = self.proton(self.steam / "compatibilitytools.d" / "GE-Proton99-4", "1 GE-Proton99-4")
        for lib in ("lib", "lib64"):
            for sub in ("wine", "vkd3d"):
                (proton / "files" / lib / sub).mkdir(parents=True, exist_ok=True)
        p = self.run_script(str(self.game))
        self.assertEqual(p.returncode, 0, p.stderr)
        real = (proton / "files").resolve()
        self.assertEqual(Path(str(self.log) + ".dllpath").read_text(encoding="utf-8").strip(),
                         f"{real / 'lib64' / 'vkd3d'}:{real / 'lib64' / 'wine'}:{real / 'lib' / 'vkd3d'}:"
                         f"{real / 'lib' / 'wine'}")

    def test_an_old_layout_proton_dxvk_under_lib64_is_used(self):
        proton = self.proton(self.steam / "compatibilitytools.d" / "Proton-5", "1 proton-5.13-6", dxvk=False,
                             lib_name="lib64")
        dxvk = proton / "files" / "lib64" / "wine" / "dxvk"
        dxvk.mkdir(parents=True)
        (dxvk / "dxgi.dll").write_bytes(b"dxvk dxgi old layout")
        p = self.run_script(str(self.game), LEONIDA_PROTON=str(proton))
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("with DXVK's dxgi", p.stdout)
        system32 = self.home / ".local" / "share" / "leonida" / "prefix" / "drive_c" / "windows" / "system32"
        self.assertEqual((system32 / "dxgi.dll").read_bytes(), b"dxvk dxgi old layout")
        self.assert_system_wine_unused()

    def test_the_vkd3d_shader_cache_goes_to_the_leonida_cache(self):
        self.proton(self.steam / "compatibilitytools.d" / "GE-Proton99-4", "1 GE-Proton99-4")
        cache = self.tmp / "cache"
        self.assertEqual(self.run_script(str(self.game), LEONIDA_CACHE_DIR=str(cache)).returncode, 0)
        recorded = Path(str(self.log) + ".shadercache")
        self.assertEqual(recorded.read_text(encoding="utf-8").strip(), str(cache / "vkd3d-proton"))
        self.assertTrue((cache / "vkd3d-proton").is_dir())
        own = self.tmp / "own shader cache"
        self.assertEqual(self.run_script(str(self.game), LEONIDA_CACHE_DIR=str(cache),
                                         VKD3D_SHADER_CACHE_PATH=str(own)).returncode, 0)
        self.assertEqual(recorded.read_text(encoding="utf-8").strip(), str(own))

    def test_log_sends_the_runtime_error_output_to_a_file_and_keeps_five(self):
        self.proton(self.steam / "compatibilitytools.d" / "GE-Proton99-4", "1 GE-Proton99-4")
        logs = self.home / ".local" / "share" / "leonida" / "logs"
        logs.mkdir(parents=True)
        for index in range(6):
            old = logs / f"play-0101-00000{index}.log"
            old.write_text("old", encoding="utf-8")
            os.utime(old, (1000 + index, 1000 + index))
        (logs / "other.log").write_text("kept", encoding="utf-8")
        p = self.run_script(str(self.game), "--log", "--series-x")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertNotIn("runtime error output", p.stderr)
        self.assertIn("runtime output", p.stdout)
        line = [l for l in p.stdout.splitlines() if l.startswith("log: ")]
        self.assertEqual(len(line), 1, p.stdout)
        written = Path(line[0][len("log: "):])
        self.assertEqual(written.parent, logs)
        self.assertRegex(written.name, r"^play-\d{4}-\d{6}\.log$")
        self.assertEqual(written.read_text(encoding="utf-8").strip(), "runtime error output")
        self.assertIn("--console series-x", self.calls()[-1])
        kept = sorted(path.name for path in logs.glob("play-*.log"))
        self.assertEqual(len(kept), 5)
        self.assertIn(written.name, kept)
        self.assertNotIn("play-0101-000000.log", kept)
        self.assertNotIn("play-0101-000001.log", kept)
        self.assertTrue((logs / "other.log").exists())

    def test_without_log_the_error_output_stays_on_the_console(self):
        self.proton(self.steam / "compatibilitytools.d" / "GE-Proton99-4", "1 GE-Proton99-4")
        p = self.run_script(str(self.game))
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("runtime error output", p.stderr)
        self.assertNotIn("log: ", p.stdout)
        self.assertFalse((self.home / ".local" / "share" / "leonida" / "logs").exists())

    def test_an_unknown_option_is_refused(self):
        self.proton(self.steam / "compatibilitytools.d" / "GE-Proton99-4", "1 GE-Proton99-4")
        p = self.run_script(str(self.game), "--nope")
        self.assertEqual(p.returncode, 2)
        self.assertIn("--log", p.stderr)
        self.assertEqual(self.calls(), [])

    def test_the_prefix_follows_the_proton_in_use(self):
        tools = self.steam / "compatibilitytools.d"
        prefix = self.tmp / "prefix"
        (prefix / "drive_c" / "windows" / "system32").mkdir(parents=True)
        (prefix / "drive_c" / "windows" / "system32" / "dxgi.dll").write_bytes(b"wine's own dxgi")
        first = self.proton(tools / "GE-Proton99-4", "1 GE-Proton99-4")
        self.assertEqual(self.run_script(str(self.game), LEONIDA_WINEPREFIX=str(prefix)).returncode, 0)
        self.assertEqual(self.calls()[0], "wineboot -u", "a prefix another Wine made is updated first")
        self.assertEqual((prefix / "drive_c" / "windows" / "system32" / "dxgi.dll").read_bytes(),
                         (first / DXVK / "dxgi.dll").read_bytes())
        self.log.unlink()
        self.assertEqual(self.run_script(str(self.game), LEONIDA_WINEPREFIX=str(prefix)).returncode, 0)
        self.assertNotIn("wineboot -u", self.calls(), "the same Proton reuses the prefix as it is")
        self.log.unlink()
        second = self.proton(tools / "GE-Proton99-5", "1 GE-Proton99-5")
        p = self.run_script(str(self.game), LEONIDA_WINEPREFIX=str(prefix))
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("updating its Wine prefix", p.stdout)
        self.assertEqual(self.calls()[0], "wineboot -u")
        self.assertEqual((prefix / "drive_c" / "windows" / "system32" / "dxgi.dll").read_bytes(),
                         (second / DXVK / "dxgi.dll").read_bytes())

    def test_a_prefix_without_the_proton_vkd3d_libraries_is_updated(self):
        proton = self.proton(self.steam / "compatibilitytools.d" / "GE-Proton99-4", "1 GE-Proton99-4")
        vkd3d = proton / "files" / "lib" / "vkd3d" / "x86_64-windows"
        vkd3d.mkdir(parents=True)
        (vkd3d / "libvkd3d-1.dll").write_bytes(b"vkd3d")
        prefix = self.tmp / "prefix"
        system32 = prefix / "drive_c" / "windows" / "system32"
        system32.mkdir(parents=True)
        (prefix / ".leonida-proton").write_text(f"GE-Proton99-4 {proton.resolve()}\n", encoding="utf-8")
        p = self.run_script(str(self.game), LEONIDA_WINEPREFIX=str(prefix))
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("updating its Wine prefix", p.stdout)
        self.assertEqual(self.calls()[0], "wineboot -u")
        self.log.unlink()
        (system32 / "libvkd3d-1.dll").write_bytes(b"vkd3d")
        self.assertEqual(self.run_script(str(self.game), LEONIDA_WINEPREFIX=str(prefix)).returncode, 0)
        self.assertNotIn("wineboot -u", self.calls())

    def test_a_prefix_proton_cannot_set_up_stops_the_start(self):
        self.proton(self.steam / "compatibilitytools.d" / "GE-Proton99-4", "1 GE-Proton99-4")
        p = self.run_script(str(self.game), LEONIDA_TEST_WINEBOOT_FAILS="1")
        self.assertEqual(p.returncode, 1)
        self.assertIn("could not set up the prefix", p.stderr)
        self.assertEqual(self.calls(), ["wineboot -u"])

    def test_a_chosen_proton_wins_over_leonida_wine(self):
        proton = self.proton(self.steam / "compatibilitytools.d" / "GE-Proton99-4", "1 GE-Proton99-4")
        self.assertEqual(self.found(LEONIDA_PROTON=str(proton), LEONIDA_WINE="wine"),
                         ("GE-Proton99-4", proton.resolve()))
        self.assert_system_wine_unused()

    def test_the_script_is_executable_and_lf(self):
        self.assertTrue(os.access(SCRIPT, os.X_OK))
        self.assertNotIn(b"\r", SCRIPT.read_bytes())
        self.assertTrue(SCRIPT.read_bytes().startswith(b"#!/bin/sh\n"))
        if shutil.which("dash"):
            self.assertEqual(subprocess.run(["dash", "-n", str(SCRIPT)]).returncode, 0)


@unittest.skipIf(os.name == "nt" or not os.path.exists("/bin/sh"), "POSIX shell script")
class PlayLinuxSystemWineTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.home = self.tmp / "home"
        self.home.mkdir()
        self.repo = self.tmp / "repo"
        (self.repo / "scripts").mkdir(parents=True)
        shutil.copy(SCRIPT, self.repo / "scripts" / "play-linux.sh")
        self.log = self.tmp / "log"
        self.bin = self.tmp / "bin"
        self.bin.mkdir()
        self.wine = self.bin / "wine"
        self.wine.write_text(WINE_RUNS, encoding="utf-8", newline="\n")
        self.wine.chmod(0o755)
        server = self.bin / "wineserver"
        server.write_text('#!/bin/sh\nprintf \'%s\\n\' "wineserver $*" >> "$LEONIDA_TEST_LOG.server"\n',
                          encoding="utf-8", newline="\n")
        server.chmod(0o755)
        self.package = self.tmp / "package"
        (self.package / "vkd3d-proton").mkdir(parents=True)
        for dll in ("d3d12.dll", "d3d12core.dll"):
            (self.package / "vkd3d-proton" / dll).write_bytes(NEW_STRUCTURIZER + dll.encode())
        (self.package / "leonida.exe").write_bytes(b"MZ")
        self.game = self.tmp / "game" / "game_scarlett_master.exe"
        self.game.parent.mkdir()
        self.game.write_bytes(b"MZ")
        self.system32 = self.home / ".local" / "share" / "leonida" / "prefix" / "drive_c" / "windows" / "system32"

    def bundled_dxvk(self, folder=TARGET / "dxvk"):
        folder = self.repo / folder
        folder.mkdir(parents=True)
        (folder / "dxgi.dll").write_bytes(b"bundled dxvk dxgi " + str(folder).encode())
        (folder / "SOURCE.txt").write_text("DXVK 99.1, unmodified 64-bit dxgi.dll from\n", encoding="utf-8")
        return folder

    def run_script(self, *args, **env):
        environment = {"PATH": os.pathsep.join([str(self.bin), "/usr/bin", "/bin"]), "HOME": str(self.home),
                       "LEONIDA_TEST_LOG": str(self.log), "LEONIDA_RUNTIME": str(self.package / "leonida.exe"),
                       "LEONIDA_VKD3D": str(self.package / "vkd3d-proton")}
        environment.update(env)
        return subprocess.run(["/bin/sh", str(self.repo / "scripts" / "play-linux.sh"), *args], env=environment,
                              capture_output=True, text=True, timeout=60)

    def calls(self):
        return self.log.read_text(encoding="utf-8").splitlines() if self.log.exists() else []

    @unittest.skipIf(system_proton_installed(), "a Proton is installed system-wide on this machine")
    def test_without_proton_the_system_wine_runs_with_the_bundled_dxvk(self):
        bundled = self.bundled_dxvk()
        p = self.run_script(str(self.game), WINEDLLOVERRIDES="mscoree=")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn(f"series-s under Wine 99.0 (Staging) ({self.wine}) with the bundled DXVK's dxgi "
                      "(DXVK 99.1) and", p.stdout)
        self.assertEqual(self.calls(), ["wineboot -u", f"{self.package / 'leonida.exe'} run {self.game} "
                                        "--execute --console series-s"])
        self.assertEqual(Path(str(self.log) + ".overrides").read_text(encoding="utf-8").strip(),
                         "d3d12,d3d12core,dxgi=n;mscoree=")
        self.assertEqual(Path(str(self.log) + ".dllpath").read_text(encoding="utf-8").strip(), "")
        self.assertEqual(Path(str(self.log) + ".server").read_text(encoding="utf-8").strip(), "wineserver -w")
        self.assertEqual((self.system32 / "dxgi.dll").read_bytes(), (bundled / "dxgi.dll").read_bytes())
        for dll in ("d3d12.dll", "d3d12core.dll"):
            self.assertEqual((self.system32 / dll).read_bytes(), NEW_STRUCTURIZER + dll.encode())
        self.assertEqual((self.system32.parents[2] / ".leonida-proton").read_text(encoding="utf-8").strip(),
                         f"Wine 99.0 (Staging) {self.wine}")
        self.log.unlink()
        self.assertEqual(self.run_script(str(self.game)).returncode, 0)
        self.assertNotIn("wineboot -u", self.calls(), "the same Wine reuses the prefix as it is")

    @unittest.skipIf(system_proton_installed(), "a Proton is installed system-wide on this machine")
    def test_the_dxvk_beside_the_script_comes_first(self):
        self.bundled_dxvk()
        package = self.bundled_dxvk(Path("scripts") / "dxvk")
        p = self.run_script(str(self.game))
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual((self.system32 / "dxgi.dll").read_bytes(), (package / "dxgi.dll").read_bytes())

    def test_leonida_wine_picks_the_wine_and_find_proton_names_it(self):
        self.bundled_dxvk()
        other = self.tmp / "other wine" / "wine-custom"
        other.parent.mkdir()
        shutil.copy(self.wine, other)
        for value, wine in (("wine", self.wine), (str(other), other)):
            with self.subTest(value=value):
                p = self.run_script("--find-proton", LEONIDA_WINE=value)
                self.assertEqual(p.returncode, 0, p.stderr)
                self.assertEqual(p.stdout.strip().split("\t"), ["Wine 99.0 (Staging)", str(wine)])

    def test_a_leonida_wine_that_is_no_wine_is_refused(self):
        self.bundled_dxvk()
        for value in (str(self.tmp / "missing"), "no-such-wine", str(self.tmp)):
            with self.subTest(value=value):
                p = self.run_script(str(self.game), LEONIDA_WINE=value)
                self.assertEqual(p.returncode, 1)
                self.assertIn("is not a Wine program", p.stderr)
                self.assertEqual(self.calls(), [])

    def test_leonida_dxvk_wins_and_must_have_a_dxgi(self):
        self.bundled_dxvk()
        own = self.tmp / "own dxvk"
        own.mkdir()
        p = self.run_script(str(self.game), LEONIDA_WINE="wine", LEONIDA_DXVK=str(own))
        self.assertEqual(p.returncode, 1)
        self.assertIn("has no dxgi.dll", p.stderr)
        self.assertEqual(self.calls(), [])
        (own / "dxgi.dll").write_bytes(b"own dxgi")
        p = self.run_script(str(self.game), LEONIDA_WINE="wine", LEONIDA_DXVK=str(own) + "/")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn(f"DXVK's dxgi from {own} (LEONIDA_DXVK)", p.stdout)
        self.assertEqual((self.system32 / "dxgi.dll").read_bytes(), b"own dxgi")

    @unittest.skipIf(system_dxvk_installed(), "DXVK is installed system-wide on this machine")
    def test_an_installed_proton_lends_its_dxvk_to_the_system_wine(self):
        proton = self.home / ".local" / "share" / "Steam" / "compatibilitytools.d" / "GE-Proton99-4"
        (proton / "files" / "bin").mkdir(parents=True)
        (proton / "files" / "bin" / "wine").write_text("#!/bin/sh\ntouch \"$LEONIDA_TEST_LOG.proton\"\n",
                                                        encoding="utf-8", newline="\n")
        (proton / "files" / "bin" / "wine").chmod(0o755)
        (proton / DXVK).mkdir(parents=True)
        (proton / DXVK / "dxgi.dll").write_bytes(b"proton dxvk dxgi")
        (proton / "version").write_text("1 GE-Proton99-4\n", encoding="utf-8")
        p = self.run_script(str(self.game), LEONIDA_WINE="wine")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("under Wine 99.0 (Staging)", p.stdout)
        self.assertIn("with DXVK's dxgi of GE-Proton99-4", p.stdout)
        self.assertEqual((self.system32 / "dxgi.dll").read_bytes(), b"proton dxvk dxgi")
        self.assertFalse(Path(str(self.log) + ".proton").exists(), "the Proton's Wine was started")

    @unittest.skipIf(system_dxvk_installed() or system_proton_installed(),
                     "DXVK or a Proton is installed system-wide on this machine")
    def test_without_any_dxvk_the_start_says_what_to_fetch(self):
        for args in (("--find-proton",), (str(self.game),)):
            with self.subTest(args=args):
                p = self.run_script(*args)
                self.assertEqual(p.returncode, 1)
                self.assertIn("swap chain", p.stderr)
                self.assertIn("fetch_dxvk.py", p.stderr)
                self.assertIn("LEONIDA_DXVK", p.stderr)
                self.assertEqual(self.calls(), [])

    @unittest.skipIf(system_proton_installed() or shutil.which("wine", path="/usr/bin:/bin"),
                     "a Proton or Wine is installed system-wide on this machine")
    def test_without_proton_and_wine_the_start_says_what_to_install(self):
        p = self.run_script(str(self.game), PATH="/usr/bin:/bin")
        self.assertEqual(p.returncode, 1)
        self.assertIn("no Proton and no Wine found", p.stderr)
        self.assertIn("LEONIDA_PROTON", p.stderr)
        self.assertIn("LEONIDA_WINE", p.stderr)

    def test_the_prefix_is_updated_when_the_wine_changes(self):
        self.bundled_dxvk()
        prefix = self.tmp / "prefix"
        (prefix / "drive_c" / "windows" / "system32").mkdir(parents=True)
        (prefix / ".leonida-proton").write_text("GE-Proton99-4 /somewhere/GE-Proton99-4\n", encoding="utf-8")
        p = self.run_script(str(self.game), LEONIDA_WINE="wine", LEONIDA_WINEPREFIX=str(prefix))
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("updating its Wine prefix", p.stdout)
        self.assertEqual(self.calls()[0], "wineboot -u")


if __name__ == "__main__":
    unittest.main()
