"""Package-gated first-light test: draw the real model a first_light.py run assembled.

tools/first_light.py writes a directory with vs.spv, ps.spv, mesh.bin and manifest.json from the user's own
package. This test runs vk_harness on that directory and checks the pixels against the manifest: a coverage
window, a corner the mesh does not cover (the harness clear value) and the UV the model's last triangle writes at
the centre pixel. The oracle is also exercised on synthetic images, without a GPU or a package, so the
all-opaque and swapped-UV negatives stay red.

    LEONIDA_FIRST_LIGHT_DIR=~/leonida-work/first_light/out python -I tests/test_first_light.py
    LEONIDA_FIRST_LIGHT_DIR=~/leonida-work/first_light/camera python -I tests/test_first_light.py   # --camera

The image is written to the directory as image_first_light.raw/.ppm; no game data is in the repository. The camera
tests check --camera's matrices, its constants (read back the way CAMERA_VS loads them) and the perspective-correct
UV prediction on the CPU.
"""

import contextlib
from contextlib import redirect_stdout
import io
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import first_light as fl
from spirv_stubs import build_ps_stub, build_vs_stub

CLEAR = fl.CLEAR
check_image = fl.check_image
write_ppm = fl.write_ppm


def harness() -> str | None:
    exe = ".exe" if os.name == "nt" else ""
    for path in (
        os.environ.get("LEONIDA_VK_HARNESS"),
        os.path.join(ROOT, "runtime", "build", "linux", f"vk_harness{exe}"),
        os.path.join(ROOT, "runtime", "build", "windows", f"vk_harness{exe}"),
        os.path.join(ROOT, "runtime", "build", f"vk_harness{exe}"),
    ):
        if path and os.path.isfile(path):
            return path
    return None


class MeshOutputTests(unittest.TestCase):
    def test_fit_keeps_relative_depth_for_an_offset_mesh(self) -> None:
        import tempfile

        vertices = [(-2.0, -1.0, 900.0, 0.0, 0.0),
                    (2.0, -1.0, 901.0, 1.0, 0.0),
                    (0.0, 1.0, 902.0, 0.5, 1.0)]
        fitted = fl.fit(vertices, 128, 128, True, False)
        with tempfile.TemporaryDirectory() as directory:
            mesh = os.path.join(directory, "mesh.bin")
            fl.write_mesh(mesh, fitted, [0, 1, 2])
            with open(mesh, "rb") as handle:
                encoded = handle.read()

        records = encoded[fl.MESH_HEADER.size:fl.MESH_HEADER.size + 3 * 20]
        depth = b"".join(records[offset + 8:offset + 12] for offset in (0, 20, 40))
        self.assertEqual(depth, struct.pack("<3f", 0.05, 0.5, 0.95))
        self.assertEqual(encoded[-12:], struct.pack("<3I", 0, 1, 2))


class ProbeDepthTests(unittest.TestCase):
    @staticmethod
    def triangle(depths, uv):
        return [(x, y, z, *uv) for (x, y), z in zip(((-1, -1), (1, -1), (-1, 1)), depths)]

    def test_only_visible_fragments_replace_an_earlier_triangle(self):
        first, last = (0.125, 0.75), (0.75, 0.125)
        back = self.triangle((0.25,) * 3, first)
        for depths, left, right in (
            ((-1, -1, -1), first, first),
            ((2, 2, 2), first, first),
            ((-1, 1, -1), first, last),
            ((2, 0, 2), first, last),
            ((0, 0, 0), last, last),
            ((1, 1, 1), last, last),
            ((1 + 2**-25,) * 3, last, last),  # Stored as exactly 1 in mesh.bin.
        ):
            with self.subTest(depths=depths):
                vertices = back + self.triangle(depths, last)
                self.assertEqual(fl.uv_at(vertices, range(6), 0, 0, 8, 8), left)
                self.assertEqual(fl.uv_at(vertices, range(6), 5, 0, 8, 8), right)

    def test_clipped_region_can_be_clear_and_supplies_no_uv_probes(self):
        vertices = self.triangle((-1, 1, -1), (0.75, 0.125))
        self.assertEqual(fl.clear_pixel(vertices, range(3), 8, 8), [0, 0])
        probes = fl.uv_probes(vertices, range(3), 8, 8)
        self.assertTrue(probes)
        self.assertTrue(all(p["pixel"][0] >= 4 for p in probes), probes)


class FirstLightTests(unittest.TestCase):
    def setUp(self):
        self.out = os.environ.get("LEONIDA_FIRST_LIGHT_DIR")
        if not self.out:
            self.skipTest(
                "LEONIDA_FIRST_LIGHT_DIR is not set (run tools/first_light.py first)"
            )
        self.out = os.path.abspath(os.path.expanduser(self.out))
        self.harness = harness()
        if not self.harness:
            self.skipTest(
                "vk_harness is not built (runtime/build/linux or runtime/build/windows, target "
                "vk_harness); set LEONIDA_VK_HARNESS"
            )
        with open(os.path.join(self.out, "manifest.json"), encoding="utf-8") as f:
            self.manifest = json.load(f)
        if self.manifest.get("layer_texture"):
            # A --texture ramp folder feeds d3d12_first_light_textured (LEONIDA_FIRST_LIGHT_TEX_DIR): its PS samples a
            # T# that only the D3D12 layer binds, so vk_harness has nothing to draw it with (@zzEmmazz, #2906).
            self.skipTest("a --texture ramp folder is for d3d12_first_light_textured, not vk_harness")

    def test_draw_matches_the_manifest(self):
        raw = os.path.join(self.out, "image_first_light.raw")
        if os.path.exists(raw):
            os.unlink(
                raw
            )  # a fresh run: a harness that writes nothing cannot pass on a stale file
        env = fl.harness_env(self.manifest, self.out)  # a --camera folder also passes its constants
        subprocess.run(
            [
                self.harness,
                os.path.join(self.out, "vs.spv"),
                os.path.join(self.out, "ps.spv"),
                raw,
                str(self.manifest["width"]),
                str(self.manifest["height"]),
            ],
            check=True,
            env=env,
        )
        self.assertTrue(os.path.isfile(raw), "the harness did not write a fresh image")
        with open(raw, "rb") as f:
            data = f.read()
        stats = check_image(data, self.manifest)
        write_ppm(
            os.path.join(self.out, "image_first_light.ppm"),
            stats["pixels"],
            self.manifest["width"],
            self.manifest["height"],
        )
        self.assertGreater(stats["covered"], 0)

    def test_check_cli_matches_the_oracle(self):
        """The tool's --check must accept and reject the same images the test does."""
        subprocess.run(
            [
                sys.executable,
                "-I",
                os.path.join(ROOT, "tools", "first_light.py"),
                "--check",
                "--out",
                self.out,
                "--harness",
                self.harness,
            ],
            check=True,
        )

    def test_manifest_names_the_assets(self):
        self.assertGreater(self.manifest["vertex_count"], 0)
        self.assertGreater(self.manifest["triangle_count"], 0)
        for key in ("vs", "ps"):
            self.assertTrue(
                self.manifest[key].get("fssm_sha256"), f"{key} has no FSSM hash"
            )
            self.assertTrue(
                os.path.isfile(os.path.join(self.out, self.manifest["files"][key]))
            )


class DrawOutputTests(unittest.TestCase):
    def test_failed_recheck_clears_previous_preview_and_result(self) -> None:
        manifest = {
            "width": 2,
            "height": 1,
            "min_coverage": 0.5,
            "max_coverage": 0.99,
            "files": {"mesh": "mesh.bin"},
        }
        first = struct.pack("<8f", 1, 0, 0, 1, 0, 0, 0, 0)
        blank = bytes(32)
        retry = struct.pack("<8f", 0, 1, 0, 1, 0, 0, 0, 0)
        frames = iter((first, blank, retry))

        def write_readback(cmd: list[str], *, check: bool, env: dict[str, str]) -> None:
            Path(cmd[3]).write_bytes(next(frames))

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            manifest_path = out / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8", newline="\n")
            ppm = out / "image.ppm"
            argv = ["--out", tmp, "--check"]
            with (
                patch.object(fl, "find_harness", return_value="stub.exe"),
                patch.object(fl.subprocess, "run", side_effect=write_readback),
                redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(fl.main(argv), 0)
                self.assertEqual(ppm.read_bytes(), b"P6\n2 1\n255\n\xff\0\0\0\0\0")
                saved = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.assertEqual(saved["result"]["ratio"], 0.5)

                with self.assertRaisesRegex(SystemExit, "coverage 0.0000 outside"):
                    fl.main(argv)
                self.assertEqual((out / "image.raw").read_bytes(), blank)
                self.assertFalse(ppm.exists())
                self.assertEqual(json.loads(manifest_path.read_text(encoding="utf-8")), manifest)

                self.assertEqual(fl.main(argv), 0)
                self.assertEqual((out / "image.raw").read_bytes(), retry)
                self.assertEqual(ppm.read_bytes(), b"P6\n2 1\n255\n\0\xff\0\0\0\0")
                saved = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.assertEqual(saved["result"]["ratio"], 0.5)


class PpmTests(unittest.TestCase):
    def test_nonfinite_readback_channels_write_complete_ppm(self) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory

        pixels = (float("nan"), float("inf"), float("-inf"), 1.0,
                  -0.25, 0.5, 1.25, 1.0)
        with TemporaryDirectory() as temp:
            path = Path(temp) / "preview.ppm"
            write_ppm(str(path), pixels, 2, 1)
            self.assertEqual(path.read_bytes(), b"P6\n2 1\n255\n" + bytes((0, 255, 0, 0, 128, 255)))


class ShaderPreparationTests(unittest.TestCase):
    def test_same_display_name_keeps_both_stage_files(self) -> None:
        modules = {"vs": build_vs_stub([0]), "ps": build_ps_stub([0])}
        entries = [
            {"name": "shared", "file": "shared.dxbc", "stage": "vs"},
            {"name": "shared", "file": "shared~1.dxbc", "stage": "ps"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            dxbc = work / "shaders" / "dxbc"
            dxbc.mkdir(parents=True)
            (dxbc / "index.json").write_text(json.dumps(entries), encoding="utf-8", newline="\n")
            for entry in entries:
                (dxbc / entry["file"]).write_bytes(entry["stage"].encode("ascii"))

            def write_translation(argv: list[str]) -> None:
                if Path(argv[0]).name != "rdna2spirv.py":
                    return
                staged = work / "first_light" / "dxbc"
                picked = json.loads((staged / "index.json").read_text(encoding="utf-8"))
                self.assertEqual(picked, entries)
                spirv = Path(argv[argv.index("--out") + 1])
                spirv.mkdir()
                index = {}
                for entry in picked:
                    stage = entry["stage"]
                    self.assertEqual((staged / entry["file"]).read_bytes(), stage.encode("ascii"))
                    stem = Path(entry["file"]).stem
                    (spirv / f"{stem}.spv").write_bytes(modules[stage])
                    index[stem] = {"stage": stage, "fssm_sha256": stage + "-fixture"}
                (spirv / "index.json").write_text(json.dumps(index), encoding="utf-8", newline="\n")

            with patch.object(fl, "run", side_effect=write_translation), redirect_stdout(io.StringIO()):
                self.assertEqual(fl.main(["--game", tmp, "--work", tmp, "--no-fit",
                                          "--vs", "shared", "--ps", "shared"]), 0)
            out = work / "first_light" / "out"
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            runtime_index = json.loads((out / "index.json").read_text(encoding="utf-8"))
            for stage, module in modules.items():
                self.assertEqual((out / manifest["files"][stage]).read_bytes(), module)
                self.assertEqual((out / f"{stage}.dxbc").read_bytes(), stage.encode("ascii"))
                self.assertEqual(runtime_index[stage]["fssm_sha256"], stage + "-fixture")
                self.assertEqual(manifest[stage]["fssm_sha256"], stage + "-fixture")


LINK_KINDS = ("symlink", "junction") if os.name == "nt" else ("symlink",)


def link_folder(kind: str, link: Path, target: Path) -> bool:
    """Make a directory symlink or a Windows junction; False where this machine does not allow it."""
    if kind == "junction":
        made = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True)
        return made.returncode == 0
    try:
        os.symlink(target, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        return False
    return True


class RunFolderTests(unittest.TestCase):
    """first_light/dxbc and first_light/spirv hold one run's pair, whatever an earlier run left there.

    Fully synthetic: the containers are a few hand-written bytes, and the unpacker, the disassembler and the
    translator are replaced by stand-ins that, like the real tools, take every file in the folder they are given.
    """

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory(prefix="first-light-run-folders-")
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.work = self.tmp / "work"
        self.staged = self.work / "first_light" / "dxbc"
        self.spirv = self.work / "first_light" / "spirv"
        carved = self.work / "shaders" / "dxbc"
        carved.mkdir(parents=True)
        entries = [
            {"name": "vertex", "file": "vertex.dxbc", "stage": "vs"},
            {"name": "first", "file": "first.dxbc", "stage": "ps"},
            {"name": "second", "file": "second.dxbc", "stage": "ps"},
        ]
        (carved / "index.json").write_text(json.dumps(entries), encoding="utf-8", newline="\n")
        for entry in entries:
            (carved / entry["file"]).write_bytes(entry["name"].encode("ascii"))
        self.handed = []  # per translator run: the listings it was given

    def tools(self, argv: list[str]) -> None:
        tool = Path(argv[0]).name
        src = Path(argv[1])
        if tool == "fssm_unpack.py":
            (src / "fssm").mkdir(exist_ok=True)
            for container in sorted(src.glob("*.dxbc")):
                (src / "fssm" / f"{container.stem}.fssm").write_bytes(container.read_bytes())
            return
        out = Path(argv[argv.index("--out") + 1])
        out.mkdir(exist_ok=True)
        if tool == "shader_disasm.py":
            for blob in sorted(src.glob("*.fssm")):
                (out / f"{blob.stem}.s").write_bytes(blob.read_bytes())
            return
        self.assertEqual(tool, "rdna2spirv.py")
        stages = {Path(e["file"]).stem: e["stage"]
                  for e in json.loads((src.parent / "index.json").read_text(encoding="utf-8"))}
        index_path = out / "index.json"  # the translator merges a small run into an index it finds
        index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else {}
        listings = sorted(src.glob("*.s"))
        for listing in listings:
            (out / f"{listing.stem}.spv").write_bytes(b"spv " + listing.read_bytes())
            index[listing.stem] = {"stage": stages.get(listing.stem, "?"), "fssm_sha256": listing.stem}
        index_path.write_text(json.dumps(index), encoding="utf-8", newline="\n")
        self.handed.append([listing.stem for listing in listings])

    def prepare(self, ps: str, *extra: str) -> None:
        with patch.object(fl, "run", side_effect=self.tools), redirect_stdout(io.StringIO()):
            self.assertEqual(fl.main(["--game", str(self.tmp), "--work", str(self.work), "--no-fit",
                                      "--vs", "vertex", "--ps", ps, *extra]), 0)

    def files(self, folder: Path) -> list[str]:
        return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())

    def test_a_second_run_holds_only_its_own_pair(self) -> None:
        other_out = self.work / "first_light" / "out_first"
        self.prepare("first", "--out", str(other_out))
        self.assertEqual(self.handed, [["first", "vertex"]])
        (self.staged / "leftover.dxbc").write_bytes(b"leftover")  # a container no run selected
        (self.spirv / "leftover.spvasm").write_bytes(b"leftover")
        carved_before = {name: (self.work / "shaders" / "dxbc" / name).read_bytes()
                         for name in self.files(self.work / "shaders" / "dxbc")}
        other_out_before = {name: (other_out / name).read_bytes() for name in self.files(other_out)}

        self.prepare("second")

        self.assertEqual(self.handed[1], ["second", "vertex"])
        self.assertEqual(self.files(self.staged),
                         ["disasm/second.s", "disasm/vertex.s", "fssm/second.fssm", "fssm/vertex.fssm",
                          "index.json", "second.dxbc", "vertex.dxbc"])
        self.assertEqual(self.files(self.spirv), ["index.json", "second.spv", "vertex.spv"])
        translated = json.loads((self.spirv / "index.json").read_text(encoding="utf-8"))
        self.assertEqual({stem: meta["stage"] for stem, meta in translated.items()},
                         {"second": "ps", "vertex": "vs"})
        out = self.work / "first_light" / "out"
        self.assertEqual((out / "ps.dxbc").read_bytes(), b"second")
        self.assertEqual((out / "ps.spv").read_bytes(), b"spv second")
        # Only the two run folders are emptied: the carve and the other run's --out are as they were.
        self.assertEqual({name: (self.work / "shaders" / "dxbc" / name).read_bytes()
                          for name in self.files(self.work / "shaders" / "dxbc")}, carved_before)
        self.assertEqual({name: (other_out / name).read_bytes() for name in self.files(other_out)},
                         other_out_before)
        self.assertEqual((other_out / "ps.dxbc").read_bytes(), b"first")

    def linked(self, kind: str, where: str) -> tuple[Path, Path]:
        """A work folder whose first_light/<where> is a link of this kind to a folder outside it."""
        work, outside = self.tmp / kind / "work", self.tmp / kind / "outside"
        (outside / "dxbc").mkdir(parents=True)
        for keep in (outside / "keep.dxbc", outside / "dxbc" / "keep.dxbc"):
            keep.write_bytes(b"keep")
        link = work / "first_light" / where
        link.parent.mkdir(parents=True)
        if not link_folder(kind, link, outside):
            self.skipTest(f"cannot create a {kind} here")
        return work, outside

    def test_a_link_inside_a_run_folder_is_removed_not_followed(self) -> None:
        for kind in LINK_KINDS:
            with self.subTest(kind=kind):
                work, outside = self.linked(kind, os.path.join("dxbc", "link"))
                staged = work / "first_light" / "dxbc"
                self.assertEqual(fl.reset_run_folders(str(work)), str(staged))
                self.assertEqual(os.listdir(staged), [])
                self.assertEqual(self.files(outside), ["dxbc/keep.dxbc", "keep.dxbc"])

    def test_a_run_folder_that_is_a_link_is_refused(self) -> None:
        for kind in LINK_KINDS:
            with self.subTest(kind=kind):
                work, outside = self.linked(kind, "spirv")
                staged = work / "first_light" / "dxbc"
                staged.mkdir()
                (staged / "earlier.dxbc").write_bytes(b"earlier")
                with self.assertRaises(SystemExit) as refused:
                    fl.reset_run_folders(str(work))
                self.assertIn(str(work / "first_light" / "spirv"), str(refused.exception))
                self.assertNotIn("\n", str(refused.exception))
                self.assertEqual(self.files(outside), ["dxbc/keep.dxbc", "keep.dxbc"])
                self.assertEqual(os.listdir(staged), ["earlier.dxbc"])  # refused before anything is removed

    def test_a_first_light_folder_that_is_a_link_is_refused(self) -> None:
        for kind in LINK_KINDS:
            with self.subTest(kind=kind):
                work, outside = self.linked(kind, "")
                with self.assertRaises(SystemExit) as refused:
                    fl.reset_run_folders(str(work))
                self.assertIn(str(work / "first_light"), str(refused.exception))
                self.assertNotIn("\n", str(refused.exception))
                self.assertEqual(self.files(outside), ["dxbc/keep.dxbc", "keep.dxbc"])

    def test_a_linked_first_light_with_no_run_folders_is_refused_too(self) -> None:
        # Nothing stale lies behind the link, so only the check of first_light itself can refuse it; without that
        # check the new dxbc was created in the folder the link points to, outside the work folder.
        for kind in LINK_KINDS:
            with self.subTest(kind=kind):
                work, outside = self.tmp / kind / "work", self.tmp / kind / "outside"
                outside.mkdir(parents=True)
                work.mkdir()
                if not link_folder(kind, work / "first_light", outside):
                    self.skipTest(f"cannot create a {kind} here")
                with self.assertRaises(SystemExit) as refused:
                    fl.reset_run_folders(str(work))
                self.assertIn(str(work / "first_light"), str(refused.exception))
                self.assertNotIn("\n", str(refused.exception))
                self.assertEqual(os.listdir(outside), [])

    def test_a_file_that_cannot_be_removed_stops_the_run(self) -> None:
        self.prepare("first")
        held = self.staged / "first.dxbc"
        denied = PermissionError(13, "Permission denied", str(held))
        with patch.object(fl.shutil, "rmtree", side_effect=denied), self.assertRaises(SystemExit) as stopped:
            self.prepare("second")
        self.assertIn(str(held), str(stopped.exception))
        self.assertIn("Permission denied", str(stopped.exception))
        self.assertNotIn("\n", str(stopped.exception))
        self.assertEqual(self.handed, [["first", "vertex"]])  # nothing was translated from the stale folder


class OracleTests(unittest.TestCase):
    """The readback oracle on synthetic images: the controls run without a GPU or a game package."""

    def setUp(self):
        self.manifest = {
            "width": 4,
            "height": 4,
            "min_coverage": 0.2,
            "max_coverage": 0.5,
            "centre_pixel": [2, 2],
            "centre_uv": [0.25, 0.75],
            "uv_probes": [{"pixel": [2, 2], "uv": [0.25, 0.75]}],
            "clear_pixel": [0, 0],
            "clear": CLEAR,
        }

    def image(self, centre=(0.25, 0.75), fill=None) -> bytes:
        pixels = [[CLEAR] * 4 for _ in range(16)]
        if fill is not None:
            pixels = [[fill, fill, fill, 1.0] for _ in range(16)]
        else:
            for x, y in ((1, 1), (2, 1), (1, 2), (2, 2)):
                pixels[y * 4 + x] = [centre[0], centre[1], 0.0, 1.0]
        return struct.pack("<64f", *(v for pixel in pixels for v in pixel))

    def test_accepts_the_expected_image(self):
        check_image(self.image(), self.manifest)

    def test_rejects_an_all_opaque_image(self):
        with self.assertRaises(AssertionError):
            check_image(self.image(fill=0.5), self.manifest)

    def test_rejects_a_swapped_uv(self):
        with self.assertRaises(AssertionError):
            check_image(self.image(centre=(0.75, 0.25)), self.manifest)

    def test_rejects_a_short_image(self):
        with self.assertRaises(AssertionError):
            check_image(b"", self.manifest)

    def test_rejects_nonfinite_channels_outside_the_probes(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                raw = bytearray(self.image())
                # The red channel of covered pixel (1, 1) is outside the clear and UV probes.
                struct.pack_into("<f", raw, (self.manifest["width"] + 1) * 16, value)
                with self.assertRaisesRegex(AssertionError, "non-finite channel"):
                    check_image(raw, self.manifest)


class ProbeCoordinateTests(unittest.TestCase):
    rgba = (0.25, 0.75, 0.0, 1.0)
    image = struct.pack("<16f", *([CLEAR] * 8 + list(rgba) * 2))

    def setUp(self):
        self.manifest = {"width": 2, "height": 2, "min_coverage": 0.5, "max_coverage": 0.5}

    def test_rejects_invalid_coordinates_in_every_probe_path(self):
        for point in ([-1, 1], [-1, 0], [2, 0], [0, 2], [0, -1], {}, [0], [True, 1], [0, 1.0]):
            probes = (
                ("clear_pixel", {"clear_pixel": point}),
                ("centre_pixel", {"centre_pixel": point, "centre_uv": self.rgba[:2]}),
                ("centre_pixel", {"centre_pixel": point, "centre_rgba": self.rgba}),
                ("uv_probes", {"uv_probes": [{"pixel": point, "uv": self.rgba[:2]}]}),
                ("rgba_probes", {"rgba_probes": [{"pixel": point, "rgba": self.rgba}]}),
            )
            for field, probe in probes:
                with self.subTest(point=point, probe=probe):
                    with self.assertRaisesRegex(AssertionError, f"invalid {field}.*coordinates"):
                        check_image(self.image, self.manifest | probe)

    def test_accepts_edges_and_optional_probes(self):
        for probe in (
            {},
            {"clear_pixel": None, "centre_pixel": None, "centre_uv": self.rgba[:2]},
            {"centre_uv": self.rgba[:2]},
            {
                "clear_pixel": [1, 0],
                "centre_pixel": [0, 1],
                "centre_uv": self.rgba[:2],
                "uv_probes": [{"pixel": [1, 1], "uv": self.rgba[:2]}],
            },
            {
                "centre_pixel": [1, 1],
                "centre_rgba": self.rgba,
                "centre_uv": [0.0, 0.0],
                "rgba_probes": [{"pixel": [0, 1], "rgba": self.rgba}],
                "uv_probes": [{"pixel": [1, 1], "uv": [0.0, 0.0]}],
            },
        ):
            with self.subTest(probe=probe):
                self.assertEqual(check_image(self.image, self.manifest | probe)["covered"], 2)

    def test_empty_centre_pixel_uses_the_image_centre(self):
        for key, expected in (("centre_uv", self.rgba[:2]), ("centre_rgba", self.rgba)):
            with self.subTest(key=key):
                manifest = self.manifest | {"centre_pixel": [], key: expected}
                self.assertEqual(check_image(self.image, manifest)["covered"], 2)

    def test_rgba_probes_still_reject_invalid_uv_coordinates(self):
        manifest = self.manifest | {
            "rgba_probes": [{"pixel": [0, 1], "rgba": self.rgba}],
            "uv_probes": [{"pixel": [99, -5]}],
        }
        with self.assertRaisesRegex(AssertionError, r"invalid uv_probes pixel coordinates \[99, -5\]"):
            check_image(self.image, manifest)

    def test_null_uv_probes_are_optional_with_rgba(self):
        manifest = self.manifest | {
            "rgba_probes": [{"pixel": [0, 1], "rgba": self.rgba}],
            "uv_probes": None,
        }
        self.assertEqual(check_image(self.image, manifest)["covered"], 2)
        with self.assertRaises(TypeError):
            check_image(self.image, manifest | {"uv_probes": False})

    def test_check_reports_invalid_coordinates(self):
        def draw(command, **kwargs):
            Path(command[3]).write_bytes(self.image)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            manifest = self.manifest | {
                "files": {"mesh": "mesh.bin"},
                "centre_uv": self.rgba[:2],
                "centre_pixel": [2, 0],
            }
            (out / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8", newline="\n")
            with (patch.object(fl, "find_harness", return_value="vk_harness"),
                  patch.object(fl.subprocess, "run", side_effect=draw),
                  self.assertRaisesRegex(SystemExit, r"invalid centre_pixel coordinates \[2, 0\] for 2x2 image")):
                fl.main(["--check", "--out", tmp])
            self.assertFalse((out / "image.ppm").exists())


class ReadObjTests(unittest.TestCase):
    """Wavefront OBJ parsing in tools/first_light.py without a GPU or game package."""

    def _write_obj(self, text: str) -> str:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = os.path.join(tmp.name, "mesh.obj")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        return path

    def test_requires_finite_vertex_positions(self):
        for coordinates in ("nan 0 0", "0 inf 0", "0 0 -inf", "0 1e999 0"):
            with self.subTest(coordinates=coordinates):
                path = self._write_obj(f"v {coordinates}\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
                with self.assertRaises(SystemExit) as caught:
                    fl.read_obj(path)
                self.assertIn(path, str(caught.exception))
                self.assertIn("non-finite vertex coordinates", str(caught.exception))

    def test_requires_finite_texture_coordinates(self):
        def triangle(u, v):
            return self._write_obj(
                "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
                f"vt {u} {v}\n"
                "f 1/1 2/1 3/1\n"
            )

        vertices, indices = fl.read_obj(triangle("-0.25", "1.5"))
        self.assertEqual(indices, [0, 1, 2])
        self.assertEqual([vertex[3:] for vertex in vertices], [(-0.25, 1.5)] * 3)
        for u, v in (("nan", "0.75"), ("0.25", "inf"), ("-inf", "0.75"), ("0.25", "1e999")):
            with self.subTest(u=u, v=v):
                with self.assertRaisesRegex(SystemExit, "non-finite texture coordinates"):
                    fl.read_obj(triangle(u, v))

        large_finite = triangle("1e38", "-1e38")
        mesh = Path(large_finite).with_suffix(".bin")
        fl.write_mesh(str(mesh), *fl.read_obj(large_finite))
        previous = mesh.read_bytes()
        self.assertEqual(len(previous), 80)
        for u, v in (("1e39", "0.75"), ("0.25", "-1e39")):
            with self.subTest(u=u, v=v):
                with self.assertRaisesRegex(SystemExit, "texture coordinates exceed the binary32 range"):
                    fl.write_mesh(str(mesh), *fl.read_obj(triangle(u, v)))
                self.assertEqual(mesh.read_bytes(), previous)

    def test_negative_relative_indices_inline_comments_and_continuations(self):
        positive = self._write_obj(
            "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
            "vt 0.1 0.2\nvt 0.3 0.4\nvt 0.5 0.6\n"
            "f 1/1 2/2 3/3\n"
        )
        relative = self._write_obj(
            "v 0 0 0 # origin\n"
            "v 1 0 0\n"
            "v 0 1 0\n"
            "vt 0.1 0.2 # uv0\n"
            "vt 0.3 0.4\n"
            "vt 0.5 0.6\n"
            "f -3/-3 \\\n"
            "  -2/-2 -1/-1 # trailing comment\n"
        )
        self.assertEqual(fl.read_obj(relative), fl.read_obj(positive))

    def test_rejects_zero_and_out_of_range_vertex_indices(self):
        for bad in (
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 0 1 2\n",
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 4\n",
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nf -4 -2 -1\n",
            "v 0 0\nf 1 1 1\n",
        ):
            with self.subTest(bad=bad), self.assertRaises(SystemExit):
                fl.read_obj(self._write_obj(bad))


class TexturedOracleTests(unittest.TestCase):
    """Synthetic tests for the textured first-light path (TEST-524): no GPU or game files."""

    def setUp(self):
        self.manifest = {
            "width": 4,
            "height": 4,
            "min_coverage": 0.2,
            "max_coverage": 0.5,
            "centre_pixel": [2, 2],
            "centre_rgba": [0.2, 0.4, 0.6, 1.0],
            "rgba_probes": [{"pixel": [1, 1], "uv": [0.25, 0.75], "rgba": [0.8, 0.2, 0.1, 1.0]}],
            "rgba_tolerance": 0.02,
            "clear_pixel": [0, 0],
            "clear": CLEAR,
            "files": {"mesh": "mesh.bin", "texture": "texture.rgba"},
            "ps": {"subgroup_size": 32},
            "texture": {
                "width": 2,
                "height": 2,
                "ptr_sgpr": 9,
                "tsharp_va": 0x100,
                "ssharp": [0x92, 0xC00000, 0x14500000, 0],
            },
        }

    def image(self, centre=(0.2, 0.4, 0.6, 1.0), probe=(0.8, 0.2, 0.1, 1.0)) -> bytes:
        pixels = [[CLEAR] * 4 for _ in range(16)]
        for x, y in ((1, 1), (2, 1), (1, 2), (2, 2)):
            pixels[y * 4 + x] = list(centre)
        pixels[1 * 4 + 1] = list(probe)
        return struct.pack("<64f", *(v for pixel in pixels for v in pixel))

    @staticmethod
    def _make_raw(width: int, height: int, overrides: dict[tuple[int, int], tuple[float, float, float, float]]) -> bytes:
        pixels = [[CLEAR] * 4 for _ in range(width * height)]
        for (x, y), rgba in overrides.items():
            pixels[y * width + x] = list(rgba)
        return struct.pack(f"<{width * height * 4}f", *(v for pixel in pixels for v in pixel))

    def test_accepts_expected_textured_image(self):
        check_image(self.image(), self.manifest)

    def test_rejects_magenta_fallback_texture(self):
        with self.assertRaises(AssertionError):
            check_image(self.image(centre=(1.0, 0.0, 1.0, 1.0)), self.manifest)

    def test_rejects_wrong_textured_probe(self):
        with self.assertRaises(AssertionError):
            check_image(self.image(probe=(0.1, 0.2, 0.8, 1.0)), self.manifest)

    def test_rejects_expected_rgba_with_wrong_channel_count(self):
        for field, colour, next_red in (
            ("centre_rgba", self.manifest["centre_rgba"], CLEAR),
            ("rgba_probes", self.manifest["rgba_probes"][0]["rgba"], 0.2),
        ):
            for values in ([], colour[:3], [*colour, next_red]):
                with self.subTest(field=field, values=values):
                    manifest = self.manifest.copy()
                    if field == "centre_rgba":
                        manifest[field] = values
                    else:
                        manifest[field] = [{**self.manifest[field][0], "rgba": values}]
                    with self.assertRaisesRegex(AssertionError, f"{field}.*exactly four"):
                        check_image(self.image(), manifest)

    def test_compares_alpha_for_four_channel_expectations(self):
        for probe, readback in (
            ("centre", self.image(centre=(0.2, 0.4, 0.6, 0.75))),
            ("offset", self.image(probe=(0.8, 0.2, 0.1, 0.75))),
        ):
            with self.subTest(probe=probe):
                with self.assertRaises(AssertionError):
                    check_image(readback, self.manifest)

    def test_sample_rgba_bilinear(self):
        rgba = bytes(
            [
                255, 0, 0, 255,    0, 255, 0, 255,
                0, 0, 255, 255,    255, 255, 0, 255,
            ]
        )
        # Centre (0.5, 0.5) averages all 4 texels equally -> (0.5, 0.5, 0.25, 1.0).
        mid = fl.sample_rgba_bilinear(2, 2, rgba, 0.5, 0.5)
        for got, want in zip(mid, (0.5, 0.5, 0.25, 1.0)):
            self.assertAlmostEqual(got, want, places=4)
        # Clamped corner (0.0, 0.0) hits texel (0, 0) -> red.
        tl = fl.sample_rgba_bilinear(2, 2, rgba, -0.5, -0.5)
        for got, want in zip(tl, (1.0, 0.0, 0.0, 1.0)):
            self.assertAlmostEqual(got, want, places=4)

    def test_harness_env_sets_texture_vars(self):
        env = fl.harness_env(self.manifest, "/tmp/fl_out", base_env={})
        self.assertEqual(env["VKH_TSHARP_VA"], "0x100")
        self.assertEqual(env["VKH_SSHARP"], "0x92,0xc00000,0x14500000,0x0")
        sgprs = env["VKH_SGPRS"].split(",")
        self.assertEqual(len(sgprs), 10)
        self.assertEqual(sgprs[9], "0x100")
        self.assertEqual(env["VKH_SUBGROUP_SIZE"], "32")
        self.assertTrue(env["VKH_TEX_RGBA"].endswith("texture.rgba"))

    def test_harness_rejects_wrapped_tsharp_va_overflow(self):
        h = harness()
        if not h:
            self.skipTest("vk_harness is not built")
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            rgba_path = os.path.join(td, "tex.rgba")
            fl.write_texture_rgba(rgba_path, 1, 1, bytes([255, 0, 0, 255]))
            env = dict(
                os.environ,
                VKH_TEX_RGBA=rgba_path,
                VKH_TSHARP_VA="0xFFFFFFE0",
            )
            proc = subprocess.run(
                [h, "unused_vs.spv", "unused_ps.spv", os.path.join(td, "out.raw"), "4", "4"],
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 2)
            self.assertIn("VKH_TSHARP_VA", proc.stderr)

    def test_decode_synthetic_dds_and_write_rgba(self):
        import tempfile

        # Minimal valid 2x2 uncompressed R8G8B8A8 DDS (128-byte header + 16 bytes pixels).
        dds = bytearray(128)
        dds[0:4] = b"DDS "
        struct.pack_into("<I", dds, 4, 124)       # dwSize
        struct.pack_into("<I", dds, 8, 0x1007)    # DDSD_CAPS|HEIGHT|WIDTH|PIXELFORMAT
        struct.pack_into("<I", dds, 12, 2)        # height
        struct.pack_into("<I", dds, 16, 2)        # width
        struct.pack_into("<I", dds, 76, 32)       # ddspf.dwSize
        struct.pack_into("<I", dds, 80, 0x41)     # DDPF_RGB | DDPF_ALPHAPIXELS
        struct.pack_into("<I", dds, 88, 32)       # RGBBitCount
        struct.pack_into("<IIII", dds, 92, 0x000000FF, 0x0000FF00, 0x00FF0000, 0xFF000000)
        struct.pack_into("<I", dds, 108, 0x1000)  # DDSCAPS_TEXTURE
        pixels = bytes([10, 20, 30, 255, 40, 50, 60, 255, 70, 80, 90, 255, 100, 110, 120, 255])
        dds.extend(pixels)
        with tempfile.TemporaryDirectory() as td:
            dds_path = os.path.join(td, "tex.dds")
            rgba_path = os.path.join(td, "tex.rgba")
            with open(dds_path, "wb") as f:
                f.write(dds)
            w, h, rgba, sha = fl.decode_dds_to_rgba(dds_path)
            self.assertEqual((w, h), (2, 2))
            self.assertEqual(rgba, pixels)
            self.assertEqual(len(sha), 64)
            fl.write_texture_rgba(rgba_path, w, h, rgba)
            with open(rgba_path, "rb") as f:
                blob = f.read()
            self.assertEqual(blob[:8], struct.pack("<II", 2, 2))
            self.assertEqual(blob[8:], pixels)

    def test_off_centre_default_uv_triangle_gets_probes_and_rejects_magenta(self):
        # Off-centre triangle in the top-left quadrant with default (0, 0) UVs: centre_uv is None
        # and uv_probes() is empty (|u - v| == 0). rgba_probes_for_mesh() must still pick covered
        # pixels so a magenta fallback draw is rejected when red is expected.
        verts = [
            (-0.9, -0.9, 0.5, 0.0, 0.0),
            (-0.2, -0.9, 0.5, 0.0, 0.0),
            (-0.9, -0.2, 0.5, 0.0, 0.0),
        ]
        indices = [0, 1, 2]
        self.assertIsNone(fl.uv_at(verts, indices, 16, 16, 32, 32))
        self.assertEqual(fl.uv_probes(verts, indices, 32, 32), [])
        red_tex = bytes([255, 0, 0, 255])
        probes = fl.rgba_probes_for_mesh(verts, indices, 32, 32, 1, 1, red_tex, uv_probe_list=[])
        self.assertTrue(len(probes) >= 1)
        manifest = {
            "width": 32,
            "height": 32,
            "min_coverage": 0.01,
            "max_coverage": 0.99,
            "clear_pixel": [31, 31],
            "clear": -1.0,
            "texture": {"width": 1, "height": 1},
            "rgba_tolerance": 0.02,
            "rgba_probes": probes,
        }
        # Build synthetic image: covered pixels in the triangle set to red vs magenta fallback.
        covered_pixels = {
            (px, py)
            for py in range(32)
            for px in range(32)
            if fl.uv_at(verts, indices, px, py, 32, 32) is not None
        }
        self.assertTrue(len(covered_pixels) > 0)
        red_img = self._make_raw(
            32, 32,
            {pos: (1.0, 0.0, 0.0, 1.0) for pos in covered_pixels},
        )
        res = fl.check_image(red_img, manifest)
        self.assertEqual(res["covered"], len(covered_pixels))
        magenta_img = self._make_raw(
            32, 32,
            {pos: (1.0, 0.0, 1.0, 1.0) for pos in covered_pixels},
        )
        with self.assertRaises(AssertionError):
            fl.check_image(magenta_img, manifest)
        # Empty-probe textured manifest and fallback-indistinguishable expected colour are both rejected.
        empty_manifest = dict(manifest, rgba_probes=[])
        with self.assertRaises(AssertionError):
            fl.check_image(magenta_img, empty_manifest)
        magenta_manifest = dict(
            manifest,
            rgba_probes=[{"pixel": probes[0]["pixel"], "uv": [0.0, 0.0], "rgba": [1.0, 0.0, 1.0, 1.0]}],
        )
        with self.assertRaises(AssertionError):
            fl.check_image(magenta_img, magenta_manifest)

    def test_nan_channels_rejected(self):
        nan = float("nan")
        # NaN in centre_rgba must fail (abs(NaN) > tol is False without an explicit finiteness check).
        with self.assertRaises(AssertionError):
            fl.check_image(self.image(centre=(nan, 0.4, 0.6, 1.0)), self.manifest)
        # NaN in rgba_probes must fail.
        with self.assertRaises(AssertionError):
            fl.check_image(self.image(probe=(0.8, nan, 0.1, 1.0)), self.manifest)
        # NaN in clear_pixel must fail.
        raw_clear_nan = self._make_raw(
            4, 4,
            {
                (0, 0): (nan, -1.0, -1.0, -1.0),
                (1, 1): (0.8, 0.2, 0.1, 1.0),
                (2, 1): (0.2, 0.4, 0.6, 1.0),
                (1, 2): (0.2, 0.4, 0.6, 1.0),
                (2, 2): (0.2, 0.4, 0.6, 1.0),
            },
        )
        with self.assertRaises(AssertionError):
            fl.check_image(raw_clear_nan, self.manifest)

    def test_transparent_unorm_fragments_counted_as_covered(self):
        # Valid transparent-red UNORM fragments (alpha == 0.0) must be counted as covered by
        # distinguishing them from the -1.0 clear sentinel rather than requiring alpha > 0.5.
        manifest = {
            "width": 4,
            "height": 4,
            "min_coverage": 0.2,
            "max_coverage": 0.9,
            "centre_pixel": [2, 2],
            "centre_rgba": [1.0, 0.0, 0.0, 0.0],
            "clear_pixel": [0, 0],
            "clear": -1.0,
            "texture": {"width": 1, "height": 1},
            "rgba_tolerance": 0.02,
            "rgba_probes": [{"pixel": [1, 1], "uv": [0.0, 0.0], "rgba": [1.0, 0.0, 0.0, 0.0]}],
        }
        overrides = {(x, y): (1.0, 0.0, 0.0, 0.0) for x in (1, 2) for y in (1, 2)}
        raw = self._make_raw(4, 4, overrides)
        res = fl.check_image(raw, manifest)
        self.assertEqual(res["covered"], 4)
        self.assertAlmostEqual(res["ratio"], 0.25)


def vs_on_cpu(sgprs: str, heap: str, x: float, y: float, z: float):
    """CAMERA_VS's arithmetic, reading its constants the way its scalar loads do: the table address from s9, a V#
    at table + 0x20 (world rows at 0x00..0x30) and one at table + 0x60 (view-projection rows at 0x160..0x19f)."""
    s = [int(w, 0) for w in sgprs.split(",")]
    words = {}
    for item in heap.split(","):
        addr, word = item.split("=")
        words[int(addr, 0)] = int(word, 0)

    def dword(addr):
        return words.get(addr, 0)

    def f(addr):
        return struct.unpack("<f", struct.pack("<I", dword(addr)))[0]

    def load(vsharp_addr, offset, count):
        base, size = dword(vsharp_addr), dword(vsharp_addr + 8)
        assert offset + 4 * count <= size, "read past num_records"
        return [f(base + offset + 4 * i) for i in range(count)]

    table = s[fl.CAMERA_TABLE_SGPR]
    w = [load(table + fl.CAMERA_WORLD_VSHARP, 16 * r, 4) for r in range(4)]
    vp = load(table + fl.CAMERA_VIEW_VSHARP, fl.CAMERA_VIEWPROJ, 16)
    world = [w[0][c] * x + w[1][c] * y + w[2][c] * z + w[3][c] for c in range(3)]
    return [vp[c] * world[0] + vp[4 + c] * world[1] + vp[8 + c] * world[2] + vp[12 + c] for c in range(4)]


class CameraTests(unittest.TestCase):
    """--camera on the CPU: no GPU or package. The cube stands in for a model."""

    CUBE = [(x, y, z, 0.5 + x / 4, 0.5 - y / 4) for x in (-2.0, 2.0) for y in (3.0, 7.0) for z in (0.0, 4.0)]

    def test_the_model_is_in_view_and_upright(self):
        world, viewproj = fl.camera_matrices(self.CUBE, 160, 120, 30.0, 25.0, 50.0)
        projected = fl.project(self.CUBE, world, viewproj)
        for x, y, z, _, _, w in projected:
            self.assertGreater(w, 0.0)
            self.assertTrue(0.0 < z < 1.0 and -1.0 < x < 1.0 and -1.0 < y < 1.0)
        centre = fl.project([(0.0, 5.0, 2.0, 0.0, 0.0)], world, viewproj)[0]
        self.assertAlmostEqual(centre[0], 0.0, places=5)  # the camera looks at the bounding-box centre
        self.assertAlmostEqual(centre[1], 0.0, places=5)
        top = fl.project([(0.0, 5.0, 4.0, 0.0, 0.0)], world, viewproj)[0]
        self.assertLess(top[1], centre[1])  # +Z (the game's up) is up the image, and the image's y grows down

    def test_constants_are_where_the_vs_reads_them(self):
        world, viewproj = fl.camera_matrices(self.CUBE, 128, 128, -40.0, 10.0, 60.0)
        sgprs, heap = fl.camera_constants(world, viewproj)
        for x, y, z, *_ in self.CUBE:
            got = vs_on_cpu(sgprs, heap, x, y, z)
            wx, wy, wz, _ = fl.row_times((x, y, z, 1.0), world)
            want = fl.row_times((wx, wy, wz, 1.0), viewproj)
            for g, e in zip(got, want):
                self.assertAlmostEqual(g, e, places=5)

    def test_uv_is_perspective_correct(self):
        # One triangle with clip w 1, 1 and 3: at the screen-space centroid the UV leans towards the near vertices.
        tri = [(-1.0, -1.0, 0.5, 0.0, 0.0, 1.0), (1.0, -1.0, 0.5, 1.0, 0.0, 1.0), (0.0, 1.0, 0.5, 0.0, 1.0, 3.0)]
        affine = [v[:5] for v in tri]
        # pixel (32, 21) of 64 x 64 is at NDC (0.015625, -0.328125), about the screen centroid
        u_aff, v_aff = fl.uv_at(affine, [0, 1, 2], 32, 21, 64, 64)
        u, v = fl.uv_at(tri, [0, 1, 2], 32, 21, 64, 64)
        b = fl.barycentric(*fl.pixel_ndc(32, 21, 64, 64), -1.0, -1.0, 1.0, -1.0, 0.0, 1.0)
        q = (b[0] / 1.0, b[1] / 1.0, b[2] / 3.0)
        self.assertAlmostEqual(u, q[1] / sum(q), places=9)
        self.assertAlmostEqual(v, q[2] / sum(q), places=9)
        self.assertAlmostEqual(v_aff, b[2], places=9)
        self.assertLess(v, v_aff - 0.1)

    def test_a_clipped_vertex_is_an_error(self):
        world, viewproj = fl.camera_matrices(self.CUBE, 128, 128, 0.0, 0.0, 50.0)
        behind = [(0.0, -1000.0, 2.0, 0.0, 0.0)]  # far behind the camera, which looks along +Y from -Y
        with self.assertRaises(SystemExit):
            fl.project(behind, world, viewproj)

    def test_image_size_is_at_least_one_pixel(self):
        # camera_matrices divides by the width. 0 used to raise ZeroDivisionError once a model was loaded.
        err = io.StringIO()
        for argv in (["--width", "0", "--list"], ["--height", "0", "--list"], ["--width", "-3", "--height", "4", "--list"]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(err):
                with self.assertRaises(SystemExit) as caught:
                    fl.main(argv)
                self.assertEqual(caught.exception.code, 2)
        self.assertIn("--width and --height must be at least 1", err.getvalue())

    def test_camera_flags_are_checked(self):
        for argv in (
            ["--camera", "--vs", fl.DEFAULT_VS],  # another VS reads other constants
            ["--camera", "--no-fit"],
            ["--camera", "--pitch", "90"],  # straight down: the up axis is the view axis
            ["--camera", "--dds", "albedo.dds"],  # the texture's T# would take s9 and heap VA 0x100 too
        ):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as cm:
                    fl.main(argv)
                self.assertEqual(cm.exception.code, 2)

    def test_harness_env_passes_only_the_manifests_constants(self):
        saved = {k: os.environ.get(k) for k in ("VKH_SGPRS", "VKH_HEAP")}
        os.environ.update(VKH_SGPRS="1,2,3", VKH_HEAP="0x0=0x1")
        try:
            files = {"mesh": "mesh.bin"}
            plain = fl.harness_env({"camera": None, "files": files}, "out")
            self.assertNotIn("VKH_SGPRS", plain)
            self.assertNotIn("VKH_HEAP", plain)
            camera = fl.harness_env({"camera": {"sgprs": "0,9", "heap": "0x100=0x1"}, "files": files}, "out")
            self.assertEqual((camera["VKH_SGPRS"], camera["VKH_HEAP"]), ("0,9", "0x100=0x1"))
            self.assertEqual(camera["VKH_DRAW_MESH"], os.path.join("out", "mesh.bin"))
            self.assertNotIn("VKH_TEX_RGBA", camera)
            # The run hint builds the same environment from nothing.
            hint = fl.harness_env({"camera": {"sgprs": "0,9", "heap": "0x100=0x1"}, "files": files}, "out", base_env={})
            self.assertEqual(set(hint), {"VKH_DRAW_MESH", "VKH_VERTEX_LAYOUT", "VKH_SGPRS", "VKH_HEAP"})
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_write_json_atomic_and_replace_failure_preserves_existing(self):
        import first_light
        import tempfile
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "manifest.json")
            first_light._write_json_atomic(out, {"ok": True})
            self.assertEqual(os.listdir(tmp), ["manifest.json"])
            with open(out, "rb") as f:
                raw = f.read()
            self.assertTrue(raw.endswith(b"\n"))
            self.assertNotIn(b"\r\n", raw)

            with open(out, "w", encoding="utf-8", newline="\n") as f:
                f.write('{"keep": true}\n')
            with mock.patch.object(first_light, "replace_retry", side_effect=OSError("replace denied")):
                with self.assertRaises(OSError):
                    first_light._write_json_atomic(out, {"ok": False})
            self.assertEqual(os.listdir(tmp), ["manifest.json"])
            with open(out, encoding="utf-8") as f:
                self.assertEqual(f.read(), '{"keep": true}\n')


class LayerTexturedOracleTests(unittest.TestCase):
    """The D3D12-layer textured first light's CPU reference (#2906), without a GPU or a package."""

    def test_ramp_inputs_leave_the_harness_texture_keys_alone(self):
        # @zzEmmazz (#2906): the ramp's manifest keys once shared "texture" and files["texture"] with the --dds
        # harness path, so harness_env read the string "ramp" as the --dds dict and the tool exited with a
        # TypeError. The ramp's keys are layer_*; harness_env must neither fail nor bind a harness texture.
        with tempfile.TemporaryDirectory() as tmp:
            verts = [(-0.5, -0.5, 0.5, 0.0, 0.0), (0.5, -0.5, 0.5, 1.0, 0.0), (-0.5, 0.5, 0.5, 0.0, 1.0)]
            manifest = {"files": {"mesh": "mesh.bin"}}
            manifest.update(fl.write_textured_inputs(tmp, verts, [0, 1, 2], 8, 8))
            manifest["files"].update(layer_texture="texture.bin", expected="expected.raw")
            self.assertNotIn("texture", manifest)
            self.assertNotIn("texture", manifest["files"])
            env = fl.harness_env(manifest, tmp, base_env={})
            self.assertNotIn("VKH_TEX_RGBA", env)
            self.assertTrue(os.path.isfile(os.path.join(tmp, "texture.bin")))

    def test_tsharp_load_offset_reads_the_first_8_dword_load(self):
        # The T# slot comes from the PS's own s_load_dwordx8; a missing listing or a null offset is slot 0.
        with tempfile.TemporaryDirectory() as tmp:
            listing = os.path.join(tmp, "ps.s")
            with open(listing, "w", encoding="utf-8") as f:
                f.write("\ts_load_dwordx4 s[8:11], s[2:3], 0x40\n\ts_load_dwordx8 s[0:7], s[2:3], 0xa0 ; T#\n"
                        "\ts_load_dwordx8 s[12:19], s[2:3], 0x20\n")
            self.assertEqual(fl.tsharp_load_offset(listing), 0xA0)
            with open(listing, "w", encoding="utf-8") as f:
                f.write("\ts_load_dwordx8 s[4:11], s[8:9], null\n")
            self.assertEqual(fl.tsharp_load_offset(listing), 0)
            self.assertEqual(fl.tsharp_load_offset(os.path.join(tmp, "missing.s")), 0)

    @staticmethod
    def _dds(path, width, height, fmt, payload):
        head = bytearray(148)
        head[0:4] = b"DDS "
        struct.pack_into("<I", head, 4, 124)
        struct.pack_into("<II", head, 12, height, width)
        head[84:88] = b"DX10"
        struct.pack_into("<I", head, 128, fmt)
        with open(path, "wb") as f:
            f.write(bytes(head) + payload)

    def test_dds_channel_rules(self):
        # A8 samples as (0, 0, 0, a); R8 as (r, 0, 0, 1); an sRGB texel is linearised before filtering; a BC1 block
        # decodes through ttpf_mips; an RGBA8 alpha is not predicted (NaN, so the layer test skips it).
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "t.dds")
            self._dds(p, 2, 1, 65, bytes([0, 255]))
            w, h, fmt, payload = fl.read_dds_mip0(p)
            self.assertEqual((w, h, fmt), (2, 1, 65))
            self.assertEqual(fl.dds_texels(w, h, fmt, payload), [(0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)])
            self._dds(p, 1, 1, 61, bytes([255]))
            self.assertEqual(fl.dds_texels(*fl.read_dds_mip0(p)), [(1.0, 0.0, 0.0, 1.0)])
            self._dds(p, 1, 1, 91, bytes([0, 0, 188, 255]))               # B8G8R8A8_SRGB: R = 188
            (r, g, b, a), = fl.dds_texels(*fl.read_dds_mip0(p))
            self.assertAlmostEqual(r, fl.srgb_to_linear(188 / 255), places=6)
            self.assertTrue(math.isnan(a))
            block = struct.pack("<HHI", 0xF800, 0x001F, 0)                 # BC1: every texel index 0 = pure red
            self._dds(p, 4, 4, 71, block)
            texels = fl.dds_texels(*fl.read_dds_mip0(p))
            self.assertEqual(len(texels), 16)
            self.assertEqual(texels[5][:3], (1.0, 0.0, 0.0))

    def test_dds_rejects_unsupported_or_short(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "t.dds")
            self._dds(p, 4, 4, 2, bytes(256))                             # R32G32B32A32_FLOAT: not a game format
            with self.assertRaises(SystemExit):
                fl.read_dds_mip0(p)
            self._dds(p, 8, 8, 71, bytes(8))                              # BC1 8x8 needs 32 bytes
            with self.assertRaises(SystemExit):
                fl.read_dds_mip0(p)

    def test_dds_tolerance_keeps_mapping_errors_out(self):
        # Every per-format tolerance stays far below a mapping error (a missed sRGB decode measured 0.31).
        for fmt in fl.DDS_FORMATS:
            self.assertLess(fl.dds_tolerance(fmt), 0.05, fmt)

    def test_ramp_texels_hold_their_own_centre(self):
        texels, data = fl.ramp_texture(4)
        self.assertEqual(struct.unpack_from("<III", data), (4, 4, fl.DXGI_FORMAT_R16G16B16A16_FLOAT))
        self.assertEqual(len(data), 12 + 4 * 4 * 8)
        self.assertEqual(texels[1 * 4 + 2], (2.5 / 4, 1.5 / 4, 0.0, 1.0))
        self.assertEqual(struct.unpack_from("<4e", data, 12 + (1 * 4 + 2) * 8), (0.625, 0.375, 0.0, 1.0))

    def test_bilinear_ramp_gives_back_the_uv_and_clamps(self):
        texels, _ = fl.ramp_texture(64)
        for u, v in ((0.3, 0.7), (0.51, 0.49), (0.123, 0.9)):
            r, g, b, a = fl.sample_bilinear_clamp(texels, 64, 64, u, v)
            self.assertAlmostEqual(r, u, places=6)
            self.assertAlmostEqual(g, v, places=6)
            self.assertEqual((b, a), (0.0, 1.0))
        r, g, _, _ = fl.sample_bilinear_clamp(texels, 64, 64, 0.001, 0.999)
        self.assertAlmostEqual(r, 0.5 / 64, places=7)       # clamp to the first texel centre
        self.assertAlmostEqual(g, 63.5 / 64, places=7)      # and to the last

    def test_point_sampling_fails_the_tolerance(self):
        # What the draw gave before literal S#s were registered (#2887): the nearest texel centre, up to half a
        # texel (1/128) from the UV, which the manifest's 0.002 tolerance rejects.
        texels, _ = fl.ramp_texture(64)
        u = 10.5 / 64 + 0.49 / 64
        nearest = (round(u * 64 - 0.5) + 0.5) / 64
        self.assertGreater(abs(nearest - fl.sample_bilinear_clamp(texels, 64, 64, u, 0.5)[0]), 0.002)

    def test_uv_map_matches_uv_at(self):
        # Two overlapping triangles: the later one wins where both cover, as uv_at says.
        vertices = [(-1.0, -1.0, 0.5, 0.0, 0.0), (1.0, -1.0, 0.5, 1.0, 0.0), (-1.0, 1.0, 0.5, 0.0, 1.0),
                    (-0.5, -0.5, 0.5, 0.9, 0.9), (0.9, -0.5, 0.5, 0.8, 0.9), (-0.5, 0.9, 0.5, 0.9, 0.8)]
        indices = [0, 1, 2, 3, 4, 5]
        got, ambiguous = fl.uv_map(vertices, indices, 16, 12)
        for py in range(12):
            for px in range(16):
                want = fl.uv_at(vertices, indices, px, py, 16, 12)
                # uv_map measures in pixels and uv_at in NDC, so the same UV can differ in the last bits.
                if want is None:
                    self.assertIsNone(got[py * 16 + px], (px, py))
                else:
                    for g, w in zip(got[py * 16 + px], want):
                        self.assertAlmostEqual(g, w, places=12, msg=(px, py))
        self.assertTrue(any(g is None for g in got) and any(g is not None for g in got))
        self.assertEqual(ambiguous, set())   # overlapping interiors, not a seam: draw order decides

    def test_uv_seam_pixels_are_not_guessed(self):
        # @whattheflop's fixture (#2906): a quad over framebuffer (4,4)-(28,28) on 32x32, split on its diagonal, the
        # two halves with constant but different UVs. Which triangle owns a diagonal sample is the rasteriser's edge
        # rule, so swapping the draw order must change nothing outside the reported seam.
        def ndc(v):
            return v / 32 * 2 - 1
        p = [(ndc(4), ndc(4)), (ndc(28), ndc(4)), (ndc(28), ndc(28)), (ndc(4), ndc(28))]
        vertices = [(p[0][0], p[0][1], 0.5, 0.25, 0.75), (p[1][0], p[1][1], 0.5, 0.25, 0.75),
                    (p[2][0], p[2][1], 0.5, 0.25, 0.75),
                    (p[0][0], p[0][1], 0.5, 0.75, 0.25), (p[2][0], p[2][1], 0.5, 0.75, 0.25),
                    (p[3][0], p[3][1], 0.5, 0.75, 0.25)]
        forward, seam_f = fl.uv_map(vertices, [0, 1, 2, 3, 4, 5], 32, 32)
        backward, seam_b = fl.uv_map(vertices, [3, 4, 5, 0, 1, 2], 32, 32)
        self.assertEqual(seam_f, seam_b)
        self.assertEqual(len(seam_f), 24)    # the diagonal's sample points
        self.assertEqual(sum(uv is not None for uv in forward), 576)
        for k in range(32 * 32):
            if k not in seam_f:
                self.assertEqual(forward[k], backward[k], k)

    def test_a_sample_just_off_an_edge_is_an_edge_sample(self):
        # @throny's finding (#4251): pixel centres 1e-5 off a triangle edge were given to one triangle by the exact
        # float test, while the GPU, which snaps vertices to 1/256 pixel, gave them to the one behind. Two layers
        # with different UVs whose upper triangle's edge passes 0.001 pixel from a column of pixel centres: those
        # samples are reported, not guessed. 0.02 pixel away is past EDGE_PX, so the upper triangle owns it.
        def tri(pts, uv):
            return [((x / 64) * 2 - 1, (y / 64) * 2 - 1, 0.5, uv[0], uv[1]) for x, y in pts]
        for off, column_ambiguous in ((0.001, True), (-0.001, True), (0.02, False)):
            with self.subTest(off=off):
                below = tri([(2, 2), (60, 2), (2, 60)], (0.25, 0.75))
                above = tri([(20.5 + off, 10), (40, 10), (20.5 + off, 40)], (0.75, 0.25))
                _, seam = fl.uv_map(below + above, [0, 1, 2, 3, 4, 5], 64, 64)
                self.assertEqual(bool(seam) and {k % 64 for k in seam} == {20}, column_ambiguous)

    def test_band_widens_only_where_the_texture_changes_at_pixel_scale(self):
        # The band is how much the reference moves when the sample moves by BAND_PX: zero over a flat texture, and
        # under strong minification (here 16 texels per pixel across a hard edge) wide enough for the GPU's snapped
        # position.
        flat = [(0.5, 0.5, 0.5, 1.0)] * 64
        edge = [((1.0 if x >= 4 else 0.0), 0.0, 0.0, 1.0) for y in range(8) for x in range(8)]
        uv = (4.0 / 8, 0.5)
        grads = [(16.0 / 8, 0.0, 0.0, 16.0 / 8)]
        _, band, _ = fl.reference_pixels(flat, 8, 8, [uv], set(), grads)
        self.assertEqual(struct.unpack("<4f", band), (0.0, 0.0, 0.0, 0.0))
        _, band, _ = fl.reference_pixels(edge, 8, 8, [uv], set(), grads)
        self.assertGreater(struct.unpack("<4f", band)[0], 0.2)
        # At one texel per pixel the same edge moves the sample by only BAND_PX of a texel.
        _, band, _ = fl.reference_pixels(edge, 8, 8, [uv], set(), [(1.0 / 8, 0.0, 0.0, 1.0 / 8)])
        self.assertLessEqual(struct.unpack("<4f", band)[0], fl.BAND_PX + 1e-6)

    def test_float_formats_get_a_relative_tolerance(self):
        relative = {f for f, spec in fl.DDS_FORMATS.items() if spec[3] is None}
        self.assertEqual(relative, {10, 54, 95, 96})

    def test_an_interior_hides_earlier_edges(self):
        # @whattheflop's second fixture (#2906): an inner triangle whose left edge (x = 24.5, on pixel centres) lies
        # inside a larger outer one, the two with different UVs. Drawn inner-then-outer, the outer interior owns
        # every sample on the inner's edge: nothing is ambiguous and the image equals a solid outer reference.
        # Drawn outer-then-inner, the inner is last and those samples are on its edge, so the edge rule decides
        # whether they show the inner or the outer UV: exactly those 48 samples are ambiguous.
        def tri(pts, uv):
            return [((x / 128) * 2 - 1, (y / 128) * 2 - 1, 0.5, uv[0], uv[1]) for x, y in pts]
        outer = tri([(8, 8), (104, 8), (8, 103)], (0.25, 0.75))
        inner = tri([(24.5, 20), (56, 20), (24.5, 68)], (0.75, 0.25))
        vertices = outer + inner
        uvs_oi, seam_oi = fl.uv_map(vertices, [0, 1, 2, 3, 4, 5], 128, 128)
        uvs_io, seam_io = fl.uv_map(vertices, [3, 4, 5, 0, 1, 2], 128, 128)
        self.assertEqual(seam_io, set())
        self.assertEqual(len(seam_oi), 48)
        self.assertEqual({k % 128 for k in seam_oi}, {24})   # pixel column 24: centre x = 24.5
        solid, _ = fl.uv_map(outer, [0, 1, 2], 128, 128)
        self.assertEqual(uvs_io, solid)   # outer drawn last covers the inner one everywhere

    def test_a_shared_edge_over_an_older_interior_is_not_ambiguous(self):
        # A front quad (two triangles sharing a diagonal, continuous UVs) drawn over a back triangle with another
        # UV: the diagonal's samples belong to one of the two front triangles, both of which give the same UV.
        def tri(pts, uvs):
            return [((x / 64) * 2 - 1, (y / 64) * 2 - 1, 0.5, u, v) for (x, y), (u, v) in zip(pts, uvs)]
        back = tri([(2, 2), (62, 2), (2, 62)], [(0.9, 0.9)] * 3)
        quad = [(8.5, 8.5), (40.5, 8.5), (40.5, 40.5), (8.5, 40.5)]
        quv = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        front = tri([quad[0], quad[1], quad[2]], [quv[0], quv[1], quv[2]]) + \
            tri([quad[0], quad[2], quad[3]], [quv[0], quv[2], quv[3]])
        _, seam = fl.uv_map(back + front, list(range(9)), 64, 64)
        self.assertEqual(seam & {32 * 64 + 32, 20 * 64 + 20}, set())   # two diagonal samples
        # The quad's outer edges over the back triangle are lone edges with another UV behind: ambiguous.
        self.assertIn(20 * 64 + 8, seam)

    def test_overlapping_edges_on_the_same_side_are_ambiguous(self):
        # @whattheflop's third fixture (#2906): two front triangles with equal UVs whose right edges overlap and
        # whose interiors both lie to the left, over an older triangle with another UV. They don't share an edge
        # (different endpoints, same side), so a top-left rule can exclude both: those samples stay ambiguous.
        def tri(pts, uv):
            return [((x / 128) * 2 - 1, (y / 128) * 2 - 1, 0.5, uv[0], uv[1]) for x, y in pts]
        outer = tri([(8, 8), (104, 8), (8, 103)], (0.25, 0.75))
        f1 = tri([(16, 20), (32.5, 20), (32.5, 52)], (0.75, 0.25))
        f2 = tri([(12, 16), (32.5, 16), (32.5, 56)], (0.75, 0.25))
        for order in ([0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2, 6, 7, 8, 3, 4, 5]):
            _, seam = fl.uv_map(outer + f1 + f2, order, 128, 128)
            column = {k for k in seam if k % 128 == 32}
            self.assertGreaterEqual(len(column), 32, order)

if __name__ == "__main__":
    sys.exit(unittest.main())
