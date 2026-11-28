"""Two synthetic compute records exercise parallel translation, cache reuse and interrupted publication."""
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def authored_pair(tmp):
    root = Path(tmp)
    src, out = root / "input", root / "output"
    (src / "disasm").mkdir(parents=True)
    (src / "fspx").mkdir()
    rows = []
    for i in (0, 1):
        name = f"cs_test_{i}"
        rows.append({"file": name + ".dxbc", "stage": "cs"})
        code = struct.pack("<II", 0xbf800000 | i, 0xbf810000)
        (src / "fspx" / (name + ".code")).write_bytes(code)
        (src / "fspx" / (name + ".state")).write_bytes(struct.pack("<7I", 0, 0, 0, 1, 1, 0, 0))
        (src / "fspx" / (name + ".json")).write_text(
            json.dumps({"stage": "cs", "threads": [1, 1, 1], "rsrc2": 0, "wave32": True}),
            encoding="utf-8", newline="\n")
        (src / (name + ".dxbc")).write_bytes(b"DXBC" + bytes(28))
        listing = f"; {name}  code 0x0..0x8 (fspx), gfx1013, 0 invalid words\n"
        for asm, word in ((f"s_nop {i}", code[:4]), ("s_endpgm", code[4:])):
            listing += f"\t{asm} ; encoding: [" + ",".join(hex(b) for b in word) + "]\n"
        (src / "disasm" / (name + ".s")).write_text(listing, encoding="utf-8", newline="\n")
    (src / "index.json").write_text(json.dumps(rows), encoding="utf-8", newline="\n")
    return src, out


@unittest.skipUnless(sys.platform.startswith("linux") and shutil.which("spirv-val") and shutil.which("spirv-as"),
                     "requires Linux and SPIRV-Tools")
class ComputeBatch(unittest.TestCase):
    def test_resume_and_publish(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, out = authored_pair(tmp)
            cmd = [sys.executable, "-I", str(ROOT / "scripts/translate_pipeline_compute.py"),
                   str(src), "--out", str(out), "--wave64-emulate"]
            run = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr + run.stdout)
            index = json.loads((out / "index.json").read_text(encoding="utf-8"))
            self.assertEqual(len(index), 2)
            files = sorted(out.glob("*.spv"))
            self.assertEqual(len(files), 2)
            for p in files:
                self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), p.stem.rsplit("_", 1)[1])
                os.utime(p, ns=(1000000000, 1000000000))
            (out / "index.json").unlink()  # interruption after checkpointing both shaders
            subprocess.run(cmd, check=True, capture_output=True)
            self.assertEqual(index, json.loads((out / "index.json").read_text(encoding="utf-8")))
            self.assertTrue(all(p.stat().st_mtime_ns == 1000000000 for p in files))
            files[0].write_bytes(b"damaged")
            subprocess.run(cmd, check=True, capture_output=True)
            self.assertEqual(hashlib.sha256(files[0].read_bytes()).hexdigest(), files[0].stem.rsplit("_", 1)[1])
            self.assertEqual(files[1].stat().st_mtime_ns, 1000000000)
            # A newly rejected input must leave the current index, even when the older file remains readable.
            (src / "disasm/cs_test_0.s").write_text("; synthetic empty listing\n", encoding="utf-8", newline="\n")
            partial = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(partial.returncode, 1, partial.stderr)
            self.assertEqual(len(json.loads((out / "index.json").read_text(encoding="utf-8"))), 1)
            failures = json.loads((out / "failures.json").read_text(encoding="utf-8"))
            rejected = next(m for n, m in index.items() if n.startswith("cs_test_0_"))
            self.assertIn("no instructions", failures[rejected["fspx_code_sha256"]][rejected["fspx_state_sha256"]])


@unittest.skipUnless((sys.platform.startswith("linux") or os.name == "nt") and
                     shutil.which("spirv-as") and shutil.which("spirv-val"), "requires SPIRV-Tools")
class WorkerParity(unittest.TestCase):
    def compare(self, extra, expected):
        with tempfile.TemporaryDirectory() as tmp:
            src, out = authored_pair(tmp)
            results = []
            for jobs in (1, 2):
                dest = out.with_name("out" + str(jobs))
                cmd = [sys.executable, "-I", "-B", str(ROOT / "scripts/translate_pipeline_compute.py"),
                       str(src), "--out", str(dest), "--jobs", str(jobs), *extra]
                def run():
                    p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                    self.assertEqual(p.returncode, 0, p.stderr+p.stdout)
                run()
                index = (dest/"index.json").read_bytes()
                self.assertEqual(len(json.loads(index)), 2)
                audit = json.loads((dest/".compute-cache/worker-receipt.json").read_bytes())
                self.assertEqual(audit["recipe"]["options"], expected)
                self.assertEqual(audit["index_sha256"], hashlib.sha256(index).hexdigest())
                shards = audit["shards"]
                self.assertEqual(len(shards), jobs)
                for i, part in enumerate(shards):
                    self.assertEqual((part["shard"], part["options"], part["failed"]), (i, expected, 0))
                    self.assertEqual(part["accepted"], 2 if jobs==1 else 1)
                cache = [json.loads(p.read_bytes()) for p in (dest/".compute-cache").glob("*.json")
                         if p.name!="worker-receipt.json"]
                self.assertEqual([r["options"] for r in cache], [expected]*2)
                files = sorted(dest.glob("*.spv"))
                self.assertEqual(len(files), 2)
                for path in files:
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), path.stem.rsplit("_", 1)[1])
                    os.utime(path, ns=(1000000000, 1000000000))
                run()  # no rewrite on verified cache hit
                self.assertEqual((dest/"index.json").read_bytes(), index)
                self.assertTrue(all(p.stat().st_mtime_ns == 1000000000 for p in files))
                if jobs == 2:
                    broken_code = json.loads(index)[files[0].stem]["fspx_code_sha256"]
                    damaged = next(p for p in (dest/".compute-cache").glob("*.json")
                                   if p.name!="worker-receipt.json" and
                                   json.loads(p.read_bytes())["meta"]["fspx_code_sha256"]==broken_code)
                    damaged.write_text("[]", encoding="utf-8")  # valid JSON, wrong receipt structure
                    files[0].write_bytes(b"damaged cache")  # only corrupt a disposable output, never input
                    run()
                    self.assertEqual(hashlib.sha256(files[0].read_bytes()).hexdigest(), files[0].stem.rsplit("_",1)[1])
                    self.assertEqual(json.loads(damaged.read_bytes())["options"], expected)
                    self.assertEqual(files[1].stat().st_mtime_ns, 1000000000)
                results.append((index, audit["recipe"], [(f.name, f.read_bytes()) for f in files]))
            self.assertEqual(results[0], results[1])

    def test_positive_default_parity(self):
        self.compare([], dict(wave64_emulate=False, compact_spirv=False, loop_limit=65536, descriptor_hash="fnv1a"))

    def test_positive_explicit_parity(self):
        import inspect
        sys.path.insert(0, str(ROOT/"scripts"))
        import translate_pipeline_compute as batch
        capable = "compact" in inspect.signature(batch.translator.assemble_spirv).parameters
        compact = capable and bool(shutil.which("spirv-opt"))
        if not capable:
            with self.assertRaisesRegex(ValueError, "does not implement --compact-spirv"):
                batch.recipe(batch.Options(True, 513, "fnv1a_fmix32", True))
        self.compare(["--wave64-emulate", "--loop-limit", "513", "--descriptor-hash", "fnv1a_fmix32"] +
                     (["--compact-spirv"] if compact else []),
                     dict(wave64_emulate=True, compact_spirv=compact, loop_limit=513, descriptor_hash="fnv1a_fmix32"))

    @unittest.skipUnless(os.name == "nt", "Windows job and exclusive sharing contract")
    def test_windows_lock_and_interrupt(self):
        sys.path.insert(0, str(ROOT/"scripts"))
        import compute_workers_win as win
        import translate_pipeline_compute as batch
        with tempfile.TemporaryDirectory() as tmp:
            src, out = authored_pair(tmp)
            cache = out/".compute-cache"
            cache.mkdir(parents=True)
            previous = b"{}"
            (out/"index.json").write_bytes(previous)
            request = cache/"positive.request"
            request.write_text(json.dumps(dict(src=str(src), out=str(out), names=["cs_test_0", "cs_test_1"],
                settings=batch.recipe(batch.Options(True, 513, "fnv1a_fmix32", False)),
                report=str(cache/"fixture-part"), shard=0)), encoding="utf-8")
            with win.output_lock(cache/"lock") as lock:
                with self.assertRaises(OSError):
                    with win.output_lock(cache/"lock"):
                        self.fail("overlapping publisher admitted")
                with self.assertRaises(KeyboardInterrupt):
                    with win.start(request, 768, lock) as child:
                        self.assertIsNone(child.poll())
                        raise KeyboardInterrupt  # only this bounded, authored translation
                self.assertIsNotNone(child.poll())
            with win.output_lock(cache/"lock"):
                self.assertEqual((out/"index.json").read_bytes(), previous)
            denied = subprocess.run([sys.executable, "-I", "-B", str(ROOT/"scripts/compute_workers_win.py"),
                                     "--shard"], input=b"", capture_output=True, timeout=10)
            self.assertNotEqual(denied.returncode, 0)
            self.assertIn(b"no complete Windows compute gate", denied.stderr)


if __name__ == "__main__":
    unittest.main()
