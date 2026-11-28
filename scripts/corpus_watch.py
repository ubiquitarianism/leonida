#!/usr/bin/env python3
"""Counts-only corpus regression watch.

Runs `make_game.py --stages check,shaders` against your own copy, then records
COUNTS ONLY (carved / translated / spirv-val-validated, per stage) against the
checked-out commit. This is the same counting the progress dashboard uses, so
a drop between snapshots is a regression.

Nothing derived from game data is written to the results file: no shader names,
no carved file names, no paths into the game copy. Diagnostic logs stay local
and may contain paths; never publish them.

    python3 -I scripts/corpus_watch.py --game <pkg> --work <work> --out <out> \
        --report <folder outside the repo> --results <counts.jsonl> --once
    python3 -I scripts/corpus_watch.py ... --interval 6 --pull

Standard library only; all text writes name their line ending.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

# -I removes the script folder from sys.path, so add tools/ explicitly.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from report_paths import same_report_path  # noqa: E402

MIN_FREE = 40 * 1024**3


def corpus_counts(work):
    """Return counts-only dict, or None when the shader stage produced no indexes.

    Mirrors tools/progress_dashboard.py: entries in dxbc/index.json are carved;
    entries in spirv/index.json without an 'error' key passed spirv-val.
    """
    dxbc = os.path.join(work, "shaders", "dxbc", "index.json")
    spirv_path = os.path.join(work, "shaders", "spirv", "index.json")
    if not os.path.isfile(dxbc):
        old_dxbc = os.path.join(work, "dxbc", "index.json")
        if not os.path.isfile(old_dxbc):
            return None
        dxbc = old_dxbc
        spirv_path = os.path.join(work, "spirv", "index.json")

    with open(dxbc, encoding="utf-8") as f:
        carved = json.load(f)
    entries = carved if isinstance(carved, list) else list(carved.values())

    per_stage = {}
    validated = 0
    spirv_entries = 0
    if os.path.isfile(spirv_path):
        with open(spirv_path, encoding="utf-8") as f:
            spirv = json.load(f)
        spirv_entries = len(spirv)
        for meta in spirv.values():
            if isinstance(meta, dict) and "error" not in meta:
                stage = meta.get("stage") or "unknown"
                per_stage.setdefault(stage, [0, 0])[1] += 1
                validated += 1
    for entry in entries:
        stage = (entry.get("stage") if isinstance(entry, dict) else None) or "unknown"
        per_stage.setdefault(stage, [0, 0])[0] += 1

    return {"carved": len(entries), "spirv_entries": spirv_entries,
            "validated": validated, "per_stage": per_stage}


def outputs_collide(work, report, results, log):
    """Refuse an output that names an input or another output."""
    outputs = [p for p in (report, results, log) if p]
    for i, first in enumerate(outputs):
        for second in outputs[i + 1:]:
            if same_report_path(first, second):
                return True
    for out in outputs:
        if same_report_path(out, work):
            return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--game", required=True, help="your package folder")
    ap.add_argument("--work", required=True, help="work folder (outside the repo)")
    ap.add_argument("--out", required=True, help="built port folder (outside the repo)")
    ap.add_argument("--report", required=True, help="progress dashboard folder (outside the repo)")
    ap.add_argument("--results", required=True, help="counts-only JSONL file")
    ap.add_argument("--log", help="local diagnostic log (may contain paths; never publish)")
    ap.add_argument("--stages", default="check,shaders")
    ap.add_argument("--interval", type=float, metavar="HOURS", default=0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--pull", action="store_true", help="git pull --ff-only before each run")
    ap.add_argument("--github", action="store_true", help="also refresh dashboard PR/CI panels")
    ap.add_argument("--jobs", type=int)
    a = ap.parse_args()
    if not a.once and a.interval <= 0:
        ap.error("use --once or --interval HOURS")

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_path = a.log

    def log(msg):
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        if log_path:
            with open(log_path, "a", encoding="utf-8", newline="\n") as f:
                f.write(line + "\n")

    if outputs_collide(a.work, a.report, a.results, log_path):
        ap.error("an output names the work folder or another output")

    for tool in ("llvm-mc", "spirv-val", "git"):
        if not shutil.which(tool):
            ap.error(f"missing tool: {tool}")
    if not os.path.isdir(a.game):
        ap.error("game folder not found")
    probe = subprocess.run(
        ["llvm-mc", "-triple=amdgcn-amd-amdhsa", "-mcpu=gfx1013",
         "-filetype=obj", "-o", os.devnull],
        input="s_nop 0\n", text=True, capture_output=True)
    if probe.returncode != 0:
        ap.error("llvm-mc cannot assemble gfx1013 (AMDGPU target missing)")

    def run(cmd, timeout):
        return subprocess.run(cmd, cwd=repo, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, timeout=timeout)

    def run_once():
        if a.pull:
            r = run(["git", "pull", "--ff-only"], 300)
            if r.returncode != 0:
                log("git pull failed:\n" + r.stdout[-2000:])
                return 1
        sha = run(["git", "rev-parse", "HEAD"], 60).stdout.strip()
        log(f"commit {sha}")
        for folder in (a.work, a.out, a.report):
            os.makedirs(folder, exist_ok=True)
        cmd = [sys.executable, "-I", "make_game.py", "--game", a.game,
               "--work", a.work, "--out", a.out, "--stages", a.stages]
        if a.jobs:
            cmd += ["--jobs", str(a.jobs)]
        log("running make_game.py (paths omitted from the local log)")
        r = run(cmd, 6 * 3600)
        if log_path:
            with open(log_path, "a", encoding="utf-8", newline="\n") as f:
                f.write(r.stdout + "\n")
        log(f"make_game.py exit {r.returncode}")

        counts = corpus_counts(a.work)
        record = {"time": datetime.now(timezone.utc).isoformat(), "commit": sha,
                  "make_game_exit": r.returncode, "counts": counts}
        with open(a.results, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

        dash = [sys.executable, "-I", "tools/progress_dashboard.py",
                "--work", a.work, "--out", a.report, "--run-tests"]
        if a.github:
            dash.append("--github")
        d = run(dash, 900)
        log(f"dashboard exit {d.returncode}")

        if counts:
            log(f"COUNTS carved={counts['carved']} spirv={counts['spirv_entries']} "
                f"validated={counts['validated']}")
            for stage, (nc, nv) in sorted(counts["per_stage"].items()):
                log(f"  {stage}: {nv}/{nc} validated")
            with open(a.results, encoding="utf-8") as f:
                previous = [json.loads(x) for x in f if x.strip()]
            if len(previous) >= 2:
                before = previous[-2].get("counts") or {}
                log(f"DELTA validated {before.get('validated')} -> {counts['validated']}, "
                    f"carved {before.get('carved')} -> {counts['carved']}")
                if counts["validated"] < before.get("validated", 0):
                    log("REGRESSION: validated count decreased")
        else:
            log("no corpus indexes produced")
        log(f"counts-only record: {a.results}")
        return 0

    rc = run_once()
    if a.once or a.interval <= 0:
        return rc
    while True:
        time.sleep(a.interval * 3600)
        run_once()


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
