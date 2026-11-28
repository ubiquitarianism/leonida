"""Build an interrupted or new pipeline compute pack from shader_library.py output (Linux or Windows).

Up to two translation processes each run at most one SPIRV-Tools process.
Validated modules and their receipts survive interruption. The runtime index is published last.
"""
import argparse
import contextlib
from dataclasses import dataclass, asdict
import hashlib
import inspect
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT), str(ROOT / "scripts")]
from make_game import file_sha256 as digest, translation_sources_digest
import rdna2spirv as translator


@dataclass(frozen=True)
class Options:
    wave64_emulate: bool
    loop_limit: int
    descriptor_hash: str
    compact_spirv: bool

    def __post_init__(self):
        if (type(self.wave64_emulate) is not bool or type(self.compact_spirv) is not bool or
                type(self.loop_limit) is not int or not 0 <= self.loop_limit <= 0xffffffff or
                self.descriptor_hash not in translator.DESCRIPTOR_HASHES):
            raise ValueError("invalid compute options")

    def apply(self):
        translator.WAVE64_EMULATE = self.wave64_emulate
        translator.LOOP_LIMIT = self.loop_limit
        translator.DESCRIPTOR_HASH = self.descriptor_hash


def observed(compact):
    return asdict(Options(translator.WAVE64_EMULATE, translator.LOOP_LIMIT, translator.DESCRIPTOR_HASH, compact))


def recipe(options):
    if options.compact_spirv and "compact" not in inspect.signature(translator.assemble_spirv).parameters:
        raise ValueError("this translator does not implement --compact-spirv")
    binaries = {}
    for tool in ["spirv-as", "spirv-val"] + (["spirv-opt"] if options.compact_spirv else []):
        binary = shutil.which(tool)
        if not binary:
            raise ValueError(tool + " is required")
        binaries[tool] = digest(Path(binary))
    return {"sources": translation_sources_digest(), "builder": digest(Path(__file__)),
            "windows_adapter": digest(Path(__file__).with_name("compute_workers_win.py")),
            "tools": binaries, "options": asdict(options)}


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_one(src, out, name, settings, compact=False):
    if observed(compact) != settings["options"]:
        raise ValueError("translation options differ from cache recipe")
    listing = src / "disasm" / (name + ".s")
    side = src / "fspx" / name
    code, state = (Path(str(side) + suffix) for suffix in (".code", ".state"))
    code_key, state_key = digest(code), digest(state)
    container = src / (name + ".dxbc")
    inputs = [settings, name, code_key, state_key, digest(listing), digest(Path(str(side) + ".json")),
              digest(container) if container.is_file() else None]
    key = object_digest(inputs)
    receipt = out / ".compute-cache" / (key + ".json")
    if receipt.exists():
        try:
            row = json.loads(receipt.read_text(encoding="utf-8"))
            if row["options"] == settings["options"] and (
                    row["meta"]["fspx_code_sha256"], row["meta"]["fspx_state_sha256"]) == (code_key, state_key):
                module_hash = row["module"]
                if len(module_hash) == 64 and all(c in "0123456789abcdef" for c in module_hash):
                    if digest(out / (name + "_" + module_hash + ".spv")) == module_hash:
                        return row
        except (OSError, ValueError, KeyError, TypeError):
            pass

    asm = translator.translate(str(listing))
    with tempfile.TemporaryDirectory(dir=out / ".compute-cache") as tmp:
        sa, sv = Path(tmp) / "module.spvasm", Path(tmp) / "module.spv"
        sa.write_text(asm, encoding="utf-8", newline="\n")
        # Older translators support ordinary assembly but do not accept the compact keyword.
        # Explicit compact is gated by recipe() before either worker starts; never drop it.
        result = translator.assemble_spirv(str(sa), str(sv), asm_text=asm, **({"compact": True} if compact else {}))
        if result.returncode:
            raise translator.Unsupported((result.stderr.splitlines() or ["SPIR-V tool failed"])[0])
        meta = translator.shader_meta(translator.translate.last_gen, asm, src_path=str(listing))
        if (meta.get("stage"), meta.get("fspx_code_sha256"), meta.get("fspx_state_sha256")) != (
                "cs", code_key, state_key):
            raise ValueError("compute stage or FSPX identity mismatch")
        module_hash = digest(sv)
        translator.replace_retry(sv, out / (name + "_" + module_hash + ".spv"))
    row = {"meta": meta, "module": module_hash, "options": observed(compact)}
    translator.write_index(str(receipt), row)
    return row


def build_part(src, out, names, settings, report, shard):
    options = Options(**settings["options"])
    options.apply()  # identically on the POSIX parent/fork child and a newly spawned Windows interpreter
    if recipe(options) != settings or observed(options.compact_spirv) != settings["options"]:
        raise RuntimeError("compute worker recipe differs from coordinator")
    rows, failures = {}, {}
    for number, name in enumerate(names, 1):
        try:
            rows[name] = build_one(src, out, name, settings, options.compact_spirv)
        except (translator.Unsupported, MemoryError) as e:
            failures[name] = str(e) or type(e).__name__
        if number % 100 == 0 or number == len(names):
            print(f"worker {os.getpid()}: {number}/{len(names)}, {len(failures)} failed", flush=True)
    audit = {"shard": shard, "recipe_sha256": object_digest(settings), "names_sha256": object_digest(names),
             "options": observed(options.compact_spirv), "accepted": len(rows), "failed": len(failures)}
    translator.write_index(str(report), {"audit": audit, "rows": rows, "failures": failures})


def run_request(path):
    q = json.loads(path.read_text(encoding="utf-8"))
    build_part(Path(q["src"]), Path(q["out"]), q["names"], q["settings"], Path(q["report"]), q["shard"])


@contextlib.contextmanager
def output_lock(path):
    if os.name == "nt":
        import compute_workers_win as win
        with win.output_lock(path) as lock:
            yield lock
    else:
        import fcntl
        with path.open("ab") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield lock


def run_parts(src, out, names, settings, jobs, reports, mib, lock):
    if os.name == "nt":
        import compute_workers_win as win
        requests = []
        for i, report in enumerate(reports):
            request = report.with_suffix(".request")
            translator.write_index(str(request), {"src": str(src), "out": str(out), "names": names[i::jobs],
                                                 "settings": settings, "report": str(report), "shard": i})
            requests.append(request)
        win.run(requests, mib, lock)
        return
    pid = os.fork() if jobs == 2 else None
    if pid == 0:
        try:
            build_part(src, out, names[1::2], settings, reports[1], 1)
        except BaseException:
            import traceback
            traceback.print_exc()
            os._exit(1)
        os._exit(0)
    try:
        build_part(src, out, names[::jobs], settings, reports[0], 0)
    finally:
        if pid and os.waitpid(pid, 0)[1]:
            raise RuntimeError("second translation worker failed")


def build(src, out, names, settings, jobs, mib):
    cache = out / ".compute-cache"
    cache.mkdir(exist_ok=True)
    with output_lock(cache / "lock") as lock, tempfile.TemporaryDirectory(dir=cache) as tmp:
        reports = [Path(tmp) / str(i) for i in range(jobs)]
        run_parts(src, out, names, settings, jobs, reports, mib, lock)

        index, failures, pairs = {}, {}, {}
        audits = []
        for i, report in enumerate(reports):
            part = json.loads(report.read_text(encoding="utf-8"))
            accepted, failed = set(part["rows"]), set(part["failures"])
            expected = names[i::jobs]
            audit = {"shard": i, "recipe_sha256": object_digest(settings), "names_sha256": object_digest(expected),
                     "options": settings["options"], "accepted": len(accepted), "failed": len(failed)}
            if part["audit"] != audit or accepted & failed or accepted | failed != set(expected):
                raise ValueError("compute shard options or coverage mismatch")
            audits.append(audit)
            failures.update(part["failures"])
            for name, row in part["rows"].items():
                if row["options"] != settings["options"]:
                    raise ValueError("cached compute row options mismatch")
                module, meta = row["module"], row["meta"]
                pair = meta["fspx_code_sha256"], meta["fspx_state_sha256"]
                if pairs.setdefault(pair, module) != module:
                    raise ValueError("conflicting compute modules for one code/state pair")
                # The same binary can implement different code/state keys. Each needs a runtime entry.
                entry = name + "_" + module
                index[entry] = meta
        by_hash = {}
        for name, reason in failures.items():
            code = digest(src / "fspx" / (name + ".code"))
            state = digest(src / "fspx" / (name + ".state"))
            by_hash.setdefault(code, {})[state] = reason
        translator.write_index(str(out / "failures.json"), by_hash)
        # An audit, not a publication marker. Associate it with the actual index bytes after commit.
        candidate = Path(tmp)/"final-index.json"
        translator.write_index(str(candidate), index)
        index_hash = digest(candidate)
        translator.write_index(str(cache / "worker-receipt.json"),
                               {"schema": 1, "recipe": settings, "index_sha256": index_hash, "shards": audits})
        translator.replace_retry(candidate, out / "index.json")
        print(f"{len(index)}/{len(names)} valid, {len(failures)} failed")
        return int(bool(failures))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", type=Path, help="unpacked library with index.json, disasm/ and fspx/")
    ap.add_argument("--out", type=Path, required=True, help="dedicated runtime library subdirectory")
    ap.add_argument("--jobs", type=int, choices=(1, 2), default=2)
    ap.add_argument("--memory-mib", type=int, default=768,
                    help="per-process limit: virtual space (Linux) or committed memory (Windows); default 768 MiB")
    ap.add_argument("--descriptor-hash", choices=translator.DESCRIPTOR_HASHES, default="fnv1a",
                    help="bindless descriptor table hash shared with the runtime")
    ap.add_argument("--loop-limit", type=translator.spirv_loop_guard.parse_limit,
                    default=translator.spirv_loop_guard.DEFAULT_LIMIT, help="maximum loop visits (0 disables)")
    ap.add_argument("--compact-spirv", action="store_true", help="compact modules with spirv-opt before validation")
    ap.add_argument("--wave64-emulate", action="store_true")
    args = ap.parse_args()
    if args.memory_mib < 1:
        ap.error("--memory-mib must be positive")
    src, out = args.inputs.resolve(), args.out.resolve()
    if out == src or src in out.parents or out in src.parents:
        ap.error("input and output directories must be separate")
    if os.name == "nt":
        import compute_workers_win as win
        win.budget_coordinator(args.memory_mib)
    else:
        import resource
        os.setpriority(os.PRIO_PROCESS, 0, 19)
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        bound = min(n for n in (args.memory_mib << 20, soft, hard) if n != resource.RLIM_INFINITY)
        resource.setrlimit(resource.RLIMIT_AS, (bound, hard))
    options = Options(args.wave64_emulate, args.loop_limit, args.descriptor_hash, args.compact_spirv)
    options.apply()
    try:
        settings = recipe(options)
    except ValueError as e:
        ap.error(str(e))
    rows = json.loads((src / "index.json").read_text(encoding="utf-8"))
    names = []
    for row in rows:
        if row["stage"] != "cs":
            continue
        filename = row["file"]
        if Path(filename).name != filename or not filename.endswith(".dxbc") or "\\" in filename:
            ap.error("invalid compute filename in input index")
        if os.name == "nt" and (any(ord(c)<32 or c in '<>:"|?*' for c in filename) or
                                (os.path.isreserved(filename) if hasattr(os.path, "isreserved")
                                 else Path(filename).is_reserved())):
            ap.error("invalid Windows compute filename in input index")
        names.append(filename[:-5])
    if not names or len({os.path.normcase(n) for n in names}) != len(names):
        ap.error("expected unique compute filenames in the input index")
    out.mkdir(parents=True, exist_ok=True)
    return build(src, out, sorted(names), settings, args.jobs, args.memory_mib)


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except AttributeError:
        pass
    sys.exit(main())
