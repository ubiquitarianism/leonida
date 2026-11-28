"""tools/leonida_launcher.py: settings, the command line and the mod list (no display, no game data).

    python -I tests/test_leonida_launcher.py -v
"""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SYSTEM_TOOLS = (Path("/usr/share/steam/compatibilitytools.d"), Path("/usr/local/share/steam/compatibilitytools.d"))
sys.path.insert(0, str(ROOT / "tools"))
import leonida_launcher as launcher  # noqa: E402


def system_proton_installed():
    return any(tools.is_dir() and any(tools.iterdir()) for tools in SYSTEM_TOOLS)


def settings(**changes):
    value = dict(launcher.DEFAULTS, game_dir="D:/Games/GTA")
    value.update(changes)
    return value


class SettingsTests(unittest.TestCase):
    def test_round_trip_and_line_endings(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sub", "launcher.json")
            launcher.save_settings(settings(dlss=False, resolution="1080", data_dir="E:/d"), path)
            with open(path, "rb") as f:
                raw = f.read()
            self.assertNotIn(b"\r", raw)
            self.assertTrue(raw.endswith(b"}\n"))
            loaded = launcher.load_settings(path)
            self.assertFalse(loaded["dlss"])
            self.assertEqual((loaded["resolution"], loaded["data_dir"]), ("1080", "E:/d"))

    def test_missing_or_broken_file_gives_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(launcher.load_settings(os.path.join(tmp, "none.json")), launcher.DEFAULTS)
            bad = os.path.join(tmp, "bad.json")
            Path(bad).write_text("{not json", encoding="utf-8")
            self.assertEqual(launcher.load_settings(bad), launcher.DEFAULTS)

    def test_wrong_types_and_unknown_keys_are_dropped(self):
        got = launcher.normalise({"dlss": "yes", "resolution": "999", "game_dir": 5, "extra": 1, "log": True})
        self.assertEqual(got["dlss"], True)
        self.assertEqual(got["resolution"], "")
        self.assertEqual(got["game_dir"], "")
        self.assertTrue(got["log"])
        self.assertNotIn("extra", got)
        self.assertEqual(launcher.normalise([1]), launcher.DEFAULTS)

    def test_linux_keeps_logs_in_the_leonida_data_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            saved = (launcher.WINDOWS, os.environ.get("XDG_DATA_HOME"))
            launcher.WINDOWS = False
            os.environ["XDG_DATA_HOME"] = tmp
            try:
                self.assertEqual(launcher.logs_dir(), os.path.join(tmp, "leonida", "logs"))
                self.assertEqual(launcher.captures_dir(), os.path.join(tmp, "leonida", "captures"))
            finally:
                launcher.WINDOWS = saved[0]
                if saved[1] is None:
                    os.environ.pop("XDG_DATA_HOME", None)
                else:
                    os.environ["XDG_DATA_HOME"] = saved[1]

    def test_game_folder_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(launcher.game_exe_present(tmp))
            Path(tmp, launcher.EXPECTED_EXE).write_bytes(b"")
            self.assertTrue(launcher.game_exe_present(tmp))
            self.assertFalse(launcher.game_exe_present(""))


class CommandTests(unittest.TestCase):
    def test_default_windows_command(self):
        command = launcher.play_command(settings(), repo="R", windows=True)
        self.assertEqual(command[:6], ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                                       os.path.join("R", "scripts", "play-d3d12.ps1")])
        self.assertEqual(command[6:], ["-Game", "D:/Games/GTA"])

    def test_every_option_maps_to_its_switch(self):
        command = launcher.play_command(
            settings(data_dir="E:/d", dlss=False, ray_tracing=True, smooth_motion=True, precompile=False,
                     series_x=True, quality=True, resolution="1440", capture=True, log=True),
            repo="R", windows=True)
        self.assertEqual(command[6:], ["-Game", "D:/Games/GTA", "-Data", "E:/d", "-NoDlss", "-RayTracing",
                                       "-SmoothMotion", "-NoPrecompile", "-SeriesX", "-Quality",
                                       "-Resolution", "1440", "-Capture", "-Log"])

    def test_the_newer_switches(self):
        got = launcher.switch_args(settings(series_s=True, series_x=True, dlss_mode="balanced", frame_gen=3,
                                            resolution="1080", aspect="21:9", present_scale="stretch",
                                            adapter="5070"))
        # Series S wins over Series X, and -Resolution is left out with a DLSS mode, as the script asks
        self.assertEqual(got, ["-SeriesS", "-Dlss", "balanced", "-FrameGen", "3", "-Aspect", "21:9",
                               "-PresentScale", "stretch", "-Adapter", "5070"])
        self.assertEqual(launcher.switch_args(settings(dlss=False, dlss_mode="quality", resolution="1080")),
                         ["-NoDlss", "-Resolution", "1080"])
        self.assertEqual(launcher.switch_args(settings(present_scale="stretch")), [])   # only with -Aspect

    def test_the_newer_settings_are_checked(self):
        got = launcher.normalise({"frame_gen": False, "dlss_mode": "max", "aspect": "21:9; calc", "adapter": 'a"b',
                                  "present_scale": "fill", "series_s": True, "series_x": True})
        self.assertEqual((got["frame_gen"], got["dlss_mode"], got["aspect"], got["adapter"], got["present_scale"]),
                         (0, "", "", "", ""))
        self.assertEqual((got["series_s"], got["series_x"]), (True, False))
        kept = launcher.normalise({"frame_gen": 4, "dlss_mode": "ultra", "aspect": " auto ", "adapter": "#1"})
        self.assertEqual((kept["frame_gen"], kept["dlss_mode"], kept["aspect"], kept["adapter"]),
                         (4, "ultra", "auto", "#1"))

    def test_switches_exist_in_the_script(self):
        text = (ROOT / "scripts" / "play-d3d12.ps1").read_text(encoding="utf-8")
        every = settings(dlss=False, ray_tracing=True, smooth_motion=True, precompile=False, series_x=True,
                         quality=True, resolution="720", capture=True, log=True, aspect="32:9",
                         present_scale="fit", adapter="#1", frame_gen=2)
        newer = settings(series_s=True, dlss_mode="quality")
        for arg in launcher.switch_args(every) + launcher.switch_args(newer):
            if arg.startswith("-") and arg[1:].isalpha():
                self.assertIn("$" + arg[1:], text, arg)
        self.assertIn("$Game", text)
        self.assertIn("$Data", text)

    def test_linux_command(self):
        command = launcher.play_command(settings(game_dir="/g", series_x=True, dlss=False), repo="/r",
                                        windows=False)
        self.assertEqual(command, ["sh", os.path.join("/r", "scripts", "play-linux.sh"),
                                   os.path.join("/g", launcher.EXPECTED_EXE), "--series-x"])
        command = launcher.play_command(settings(game_dir="/g", log=True), repo="/r", windows=False)
        self.assertEqual(command[2:], [os.path.join("/g", launcher.EXPECTED_EXE), "--log"])

    def test_the_proton_folder_goes_to_play_linux_sh(self):
        base = {"PATH": "/bin", "LEONIDA_PROTON": "/from/the/shell"}
        self.assertEqual(launcher.play_environment(settings(proton_dir=" /p "), base)["LEONIDA_PROTON"], "/p")
        self.assertEqual(launcher.play_environment(settings(), base), base)
        self.assertNotIn("LEONIDA_PROTON", launcher.play_environment(settings(), {"PATH": "/bin"}))
        self.assertEqual(launcher.proton_command(repo="/r"),
                         ["sh", os.path.join("/r", "scripts", "play-linux.sh"), "--find-proton"])
        self.assertIn('"--find-proton"', (ROOT / "scripts" / "play-linux.sh").read_text(encoding="utf-8"))

    def test_system_wine_goes_to_play_linux_sh_as_leonida_wine(self):
        base = {"PATH": "/bin", "LEONIDA_PROTON": "/from/the/shell"}
        env = launcher.play_environment(settings(proton_dir=launcher.SYSTEM_WINE), base)
        self.assertEqual(env["LEONIDA_WINE"], "wine")
        self.assertNotIn("LEONIDA_PROTON", env)
        self.assertEqual(base["LEONIDA_PROTON"], "/from/the/shell")
        self.assertIn("LEONIDA_WINE", (ROOT / "scripts" / "play-linux.sh").read_text(encoding="utf-8"))

    @unittest.skipIf(os.name == "nt" or not os.path.exists("/bin/sh"), "POSIX shell script")
    def test_find_proton_names_the_system_wine_when_it_is_chosen(self):
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = Path(tmp, "bin")
            bin_dir.mkdir()
            wine = bin_dir / "wine"
            wine.write_text('#!/bin/sh\n[ "$1" = --version ] && echo wine-99.0\nexit 0\n', encoding="utf-8")
            wine.chmod(0o755)
            dxvk = Path(tmp, "dxvk")
            dxvk.mkdir()
            (dxvk / "dxgi.dll").write_bytes(b"x")
            base = {"PATH": os.pathsep.join([str(bin_dir), "/usr/bin", "/bin"]), "HOME": tmp,
                    "LEONIDA_DXVK": str(dxvk)}
            self.assertEqual(launcher.find_proton(settings(proton_dir=launcher.SYSTEM_WINE), repo=str(ROOT),
                                                  base=base), (True, "Wine 99.0", str(wine)))
            if not system_proton_installed():
                self.assertEqual(launcher.list_protons(repo=str(ROOT), base=dict(base, LEONIDA_WINE="nowhere")), [])

    @unittest.skipIf(os.name == "nt" or not os.path.exists("/bin/sh"), "POSIX shell script")
    def test_find_proton_reports_what_play_linux_sh_uses(self):
        with tempfile.TemporaryDirectory() as tmp:
            proton = Path(tmp, "GE-Proton99-4")
            (proton / "files" / "bin").mkdir(parents=True)
            (proton / "files" / "bin" / "wine").write_text("#!/bin/sh\n", encoding="utf-8")
            (proton / "files" / "bin" / "wine").chmod(0o755)
            dxvk = proton / "files" / "lib" / "wine" / "dxvk" / "x86_64-windows"
            dxvk.mkdir(parents=True)
            (dxvk / "dxgi.dll").write_bytes(b"x")
            base = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": tmp}
            found, name, folder = launcher.find_proton(settings(proton_dir=str(proton)), repo=str(ROOT), base=base)
            self.assertEqual((found, name, folder), (True, "GE-Proton99-4", str(proton.resolve())))
            found, message, folder = launcher.find_proton(settings(proton_dir=os.path.join(tmp, "missing")),
                                                          repo=str(ROOT), base=base)
            self.assertFalse(found)
            self.assertIn("is not a Proton build", message)
            self.assertEqual(folder, "")

    @unittest.skipIf(os.name == "nt" or not os.path.exists("/bin/sh"), "POSIX shell script")
    @unittest.skipIf(system_proton_installed(), "a Proton is installed system-wide on this machine")
    def test_list_protons_reports_every_build_play_linux_sh_finds(self):
        with tempfile.TemporaryDirectory() as tmp:
            tools = Path(tmp, ".local", "share", "Steam", "compatibilitytools.d")
            partial = tools / "Proton-Wineland Latest"
            (partial / "files" / "bin").mkdir(parents=True)
            (partial / "files" / "bin" / "wine").write_text("#!/bin/sh\n", encoding="utf-8")
            (partial / "files" / "bin" / "wine").chmod(0o755)
            (partial / "version").write_text("1 wineland-99.0\n", encoding="utf-8")
            for name in ("GE-Proton98-1", "GE-Proton99-2"):
                proton = tools / name
                (proton / "files" / "lib" / "wine").mkdir(parents=True)
                (proton / "files" / "bin").mkdir(parents=True)
                (proton / "files" / "bin" / "wine").write_text("#!/bin/sh\n", encoding="utf-8")
                (proton / "files" / "bin" / "wine").chmod(0o755)
                dxvk = proton / "files" / "lib" / "wine" / "dxvk" / "x86_64-windows"
                dxvk.mkdir(parents=True)
                (dxvk / "dxgi.dll").write_bytes(b"x")
            base = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": tmp, "LEONIDA_PROTON": "/elsewhere"}
            self.assertEqual(launcher.list_protons(repo=str(ROOT), base=base),
                             [("GE-Proton99-2", str((tools / "GE-Proton99-2").resolve())),
                              ("GE-Proton98-1", str((tools / "GE-Proton98-1").resolve()))])
            self.assertEqual(launcher.list_protons(repo=os.path.join(tmp, "nowhere"), base=base), [])

    def test_the_newest_proton_is_the_default_entry_of_the_list(self):
        found = [("GE-Proton99-2", "/p/99"), ("GE-Proton98-1", "/p/98")]
        choices = launcher.proton_choices(found, "/usr/bin/wine")
        self.assertEqual(choices, [("GE-Proton99-2  (/p/99)", "", "/p/99"),
                                   ("GE-Proton98-1  (/p/98)", "/p/98", "/p/98"),
                                   ("System Wine  (/usr/bin/wine)", launcher.SYSTEM_WINE, launcher.SYSTEM_WINE)])
        self.assertEqual(launcher.proton_label(choices, ""), "GE-Proton99-2  (/p/99)")
        self.assertEqual(launcher.proton_label(choices, "/p/99"), "GE-Proton99-2  (/p/99)")
        self.assertEqual(launcher.proton_label(choices, "/p/98"), "GE-Proton98-1  (/p/98)")
        self.assertEqual(launcher.proton_label(choices, launcher.SYSTEM_WINE), "System Wine  (/usr/bin/wine)")
        self.assertEqual(launcher.proton_label(choices, "/my/proton"), "/my/proton")
        self.assertFalse(any(label.startswith("Newest") for label, _value, _target in choices))

    def test_without_a_proton_the_system_wine_is_the_default_entry(self):
        choices = launcher.proton_choices([], "/usr/bin/wine")
        self.assertEqual(choices, [("System Wine  (/usr/bin/wine)", "", launcher.SYSTEM_WINE)])
        self.assertEqual(launcher.proton_label(choices, ""), "System Wine  (/usr/bin/wine)")
        self.assertEqual(launcher.proton_label(choices, launcher.SYSTEM_WINE), "System Wine  (/usr/bin/wine)")
        nothing = launcher.proton_choices([], None)
        self.assertEqual(nothing, [("No Proton or Wine found", "", "")])
        self.assertEqual(launcher.proton_label(nothing, launcher.SYSTEM_WINE), "System Wine  (not found)")

    def test_no_game_folder_is_refused(self):
        with self.assertRaises(ValueError):
            launcher.play_command(settings(game_dir="  "), windows=True)

    def test_tools(self):
        self.assertEqual(launcher.tool_command("get_dlss", repo="R", windows=True)[:2], ["cmd", "/c"])
        self.assertTrue(launcher.tool_command("build", repo="R", windows=True)[-1].endswith("build-nvidia.ps1"))
        self.assertEqual(launcher.tool_command("build", repo="/r", windows=False),
                         ["sh", os.path.join("/r", "scripts", "build-d3d12-linux.sh")])
        self.assertEqual(launcher.tool_command("build", repo="/r", windows=False, accept_sdk_licence=True)[:2],
                         ["env", "XWIN_ACCEPT_LICENSE=1"])
        self.assertTrue((ROOT / "scripts" / "build-d3d12-linux.sh").is_file())
        with self.assertRaises(ValueError):
            launcher.tool_command("get_dlss", windows=False)
        with self.assertRaises(ValueError):
            launcher.tool_command("nope", windows=True)

    def test_state_follows_the_scripts_messages(self):
        state = "Starting"
        for line, want in (("Series S, 720p. F1 in the game: settings.\n", "Starting"),
                           ("Precompiling shaders (minutes) ...\n", "Preparing shaders"),
                           ("  1000 shaders\n", "Preparing shaders"),
                           ("log: x.log\n", "Running")):
            state = launcher.state_for_line(state, line)
            self.assertEqual(state, want, line)
        self.assertEqual(launcher.state_for_line("Starting", "Building the layer (a few minutes) ...\n"),
                         "Building the runtime")


class ModsAndPicturesTests(unittest.TestCase):
    def test_list_and_toggle_mods(self):
        with tempfile.TemporaryDirectory() as tmp:
            mod = Path(tmp, "alpha")
            mod.mkdir()
            manifest = mod / "mod.json"
            manifest.write_text(json.dumps({"name": "Alpha", "archetype_swaps": [{"from": "a_b", "to": "c_d"}]}),
                                encoding="utf-8")
            Path(tmp, "broken").mkdir()
            Path(tmp, "broken", "mod.json").write_text("{", encoding="utf-8")
            Path(tmp, "notamod").mkdir()
            rows, errors = launcher.list_mods(tmp)
            self.assertEqual([(r["folder"], r["name"], r["enabled"], r["swaps"]) for r in rows],
                             [("alpha", "Alpha", True, 1)])
            self.assertEqual(len(errors), 1)
            launcher.set_mod_enabled(rows[0]["path"], False)
            self.assertFalse(json.loads(manifest.read_text(encoding="utf-8"))["enabled"])
            self.assertNotIn(b"\r", manifest.read_bytes())
            self.assertFalse(launcher.list_mods(tmp)[0][0]["enabled"])
            self.assertEqual(launcher.list_mods(os.path.join(tmp, "missing")), ([], []))

    def test_newest_picture(self):
        with tempfile.TemporaryDirectory() as tmp:
            old, new = Path(tmp, "a", "old.png"), Path(tmp, "b", "new.png")
            old.parent.mkdir()
            new.parent.mkdir()
            old.write_bytes(b"x")
            new.write_bytes(b"x")
            Path(tmp, "note.txt").write_bytes(b"x")
            os.utime(old, (1000, 1000))
            os.utime(new, (2000, 2000))
            self.assertEqual(launcher.newest_picture([tmp]), str(new))
            self.assertIsNone(launcher.newest_picture([os.path.join(tmp, "none")]))


if __name__ == "__main__":
    unittest.main()
