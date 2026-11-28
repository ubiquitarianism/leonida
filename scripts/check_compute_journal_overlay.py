"""Check a disposable journal against two locally generated compute PSOs. No game files needed.

Build d3d12_compute_journal_test first, then run this script with the path to that executable.
--require-hit also requires the driver's application-cache-hit bit (not a timing heuristic).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def entries(cache):
    d = cache / "pipeline_cache.d"
    if not d.exists():
        return {}
    out = {}
    for p in d.iterdir():
        require(re.fullmatch(r"cs-[0-9a-f]{16}\.bin", p.name) and p.is_file() and
                32 < p.stat().st_size <= 8 << 20, "unexpected file in one-PSO journal")
        out[p.name] = (hashlib.sha256(p.read_bytes()).digest(), p.stat().st_mtime_ns)
    return out


def run(exe, root, device, strict):
    seed = root / "cold"
    before = None
    results = {}
    adapter = None
    for label, second in (("cold", False), ("external-A", False),
                          ("new-B", True), ("local-B", True)):
        cache = root / label
        cache.mkdir()
        if label == "local-B":
            # Bring only B into a fresh process; no session or shared cache crosses this boundary.
            (cache / "pipeline_cache.d").mkdir()
            src, = (root / "new-B" / "pipeline_cache.d").iterdir()
            shutil.copy2(src, cache / "pipeline_cache.d" / src.name)
        local_before = entries(cache)
        env = {k: v for k, v in os.environ.items() if not k.upper().startswith("LEONIDA_")}
        env.update(LEONIDA_PIPELINE_CACHE_DIR=str(cache), LEONIDA_TRACE_PIPELINE_COMPILE="1",
                   LEONIDA_PIPELINE_CACHE_BYTES=str(8 << 20), LEONIDA_PIPELINE_JOURNAL_BYTES=str(8 << 20),
                   LEONIDA_PIPELINE_JOURNAL="1")
        if device:
            env["LEONIDA_VK_DEVICE"] = device
        if label != "cold":
            env["LEONIDA_PIPELINE_JOURNAL_SEED_DIR"] = str(seed / "pipeline_cache.d")
        p = subprocess.run([str(exe)] + (["--second-shader"] if second else []),
                           stdin=subprocess.DEVNULL, capture_output=True, text=True, errors="replace",
                           cwd=cache, env=env, timeout=60)
        if p.returncode == 77:
            print("SKIP: the authored compute fixture is unsupported on this device")
            return 77
        expected = 14 if second else 10
        require(p.returncode == 0 and
                f"compute journal: direct Dispatch(4,1,1) copied both expected words (4,{expected})"
                in p.stdout.splitlines(), f"{label}: GPU readback did not match")
        devices = [x.removeprefix("device: ") for x in p.stdout.splitlines() if x.startswith("device: ")]
        require(len(devices) == 1 and (adapter is None or adapter == devices[0]), "inconsistent GPU selection")
        adapter = devices[0]
        calls = [x for x in p.stderr.splitlines() if x.startswith("[rt] pipeline-compile trigger=")]
        require(len(calls) == 1, f"{label}: expected one compute creation")
        m = re.search(r" hit=(-1|0|1) bytes=\d+ variant=\d+ result=0 ", calls[0])
        require(m is not None, f"{label}: missing feedback field or unsuccessful create")
        hit = int(m[1])
        local = entries(cache)
        if label == "cold":
            require(len(local) == 1, "cold A was not journaled")
            source, = (seed / "pipeline_cache.d").iterdir()
            # Make an erroneous hourly LRU touch observable without making the probe depend on system time.
            old = time.time_ns() - 48 * 3600 * 10**9
            os.utime(source, ns=(old, old))
            before = entries(seed)
        elif label == "new-B":
            require(len(local) == 1 and not (local.keys() & before.keys()), "missing distinct local B")
        else:
            if strict:
                require(hit == 1, f"{label}: driver reported no application-cache hit")
            if hit == 1:
                require(local == local_before, f"{label}: cache hit changed disposable journal")
        require(entries(seed) == before, f"{label}: seed contents or last-write time changed")
        results[label] = {"readback": [4, expected], "driver_hit": hit, "local_entries": len(local)}
    print(json.dumps({"device": adapter, "runs": results}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path, help="built d3d12_compute_journal_test.exe")
    parser.add_argument("--device", help="optional LEONIDA_VK_DEVICE selector")
    parser.add_argument("--require-hit", action="store_true")
    args = parser.parse_args()
    exe = args.executable.resolve(strict=True)
    require(exe.name.lower() == "d3d12_compute_journal_test.exe", "expected the compute test, not the game")
    with tempfile.TemporaryDirectory(prefix="leonida-authored-journal-") as tmp:
        raise SystemExit(run(exe, Path(tmp), args.device, args.require_hit))
