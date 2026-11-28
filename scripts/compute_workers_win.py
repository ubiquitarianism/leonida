"""Windows process/lock adapter for translate_pipeline_compute.py (CPython, no extra packages).

Only a small bootstrap runs before job assignment. It receives no input path or work until the
parent verifies its job and opens the gate. Parent death closes the gate or the kill-on-close job.
"""
import contextlib
import ctypes as c
from ctypes import wintypes as w
import json
import os
from pathlib import Path
import subprocess
import sys
import time

kernel = c.WinDLL("kernel32", use_last_error=True)
H, D, Z = w.HANDLE, w.DWORD, c.c_size_t


def api(name, result, *args):
    f = getattr(kernel, name)
    f.argtypes, f.restype = args, result
    return f


def check(value):
    if not value:
        raise c.WinError(c.get_last_error())
    return value


class Basic(c.Structure):
    _fields_ = [("time", c.c_int64 * 2), ("flags", D), ("min_ws", Z), ("max_ws", Z),
                ("processes", D), ("affinity", Z), ("priority", D), ("scheduling", D)]


class Extended(c.Structure):
    _fields_ = [("basic", Basic), ("io", c.c_uint64 * 6), ("process_mem", Z),
                ("job_mem", Z), ("peak_process", Z), ("peak_job", Z)]


class Accounting(c.Structure):
    _fields_ = [("time", c.c_int64 * 4), ("faults", D), ("total", D), ("active", D), ("ended", D)]


close = api("CloseHandle", w.BOOL, H)
create_job = api("CreateJobObjectW", H, H, w.LPCWSTR)
set_job = api("SetInformationJobObject", w.BOOL, H, D, H, D)
query_job = api("QueryInformationJobObject", w.BOOL, H, D, H, D, H)
assign = api("AssignProcessToJobObject", w.BOOL, H, H)
inside = api("IsProcessInJob", w.BOOL, H, H, c.POINTER(w.BOOL))
terminate = api("TerminateJobObject", w.BOOL, H, D)
current = api("GetCurrentProcess", H)
priority = api("GetPriorityClass", D, H)
create_file = api("CreateFileW", H, w.LPCWSTR, D, D, H, D, D, H)
LOW = subprocess.BELOW_NORMAL_PRIORITY_CLASS
PROCESS_MEMORY, PRIORITY, TWO_ACTIVE, KILL_ON_CLOSE = 0x100, 0x20, 0x8, 0x2000


def flags(worker):
    return PROCESS_MEMORY | PRIORITY | ((TWO_ACTIVE | KILL_ON_CLOSE) if worker else 0)


def verify(job, process, mib, worker):
    info, member = Extended(), w.BOOL()
    check(inside(process, job, c.byref(member)))
    check(query_job(job, 9, c.byref(info), c.sizeof(info), None))
    if (not member.value or info.basic.flags != flags(worker) or info.process_mem != mib << 20 or
            info.basic.priority != LOW or priority(process) != LOW or
            (worker and info.basic.processes != 2)):
        raise RuntimeError("Windows compute job limits not in force")


@contextlib.contextmanager
def job(mib, worker):
    if not 0 < mib <= ((1 << (c.sizeof(Z)*8))-1) >> 20:
        raise ValueError("invalid --memory-mib for Windows job")
    h = check(create_job(None, None))  # unnamed, never inherited; no breakaway
    try:
        info = Extended()
        info.basic.flags, info.basic.priority = flags(worker), LOW
        info.basic.processes, info.process_mem = (2 if worker else 0), mib << 20
        check(set_job(h, 9, c.byref(info), c.sizeof(info)))
        yield h
    finally:
        check(close(h))


def budget_coordinator(mib):
    # No KILL_ON_CLOSE for the caller itself. Its limits survive handle close until process exit.
    with job(mib, False) as h:
        check(assign(h, current()))
        verify(h, current(), mib, False)


@contextlib.contextmanager
def output_lock(path):
    # One nonblocking, no-share open; duplicated into workers so an interrupted parent cannot
    # release exclusivity before the last worker has actually left. Never delete/recreate the lock.
    h = create_file(str(path), 0xc0000000, 0, None, 4, 0x80, None)
    if h == H(-1).value:
        raise c.WinError(c.get_last_error())
    try:
        yield h
    finally:
        check(close(h))


def drain(h):
    check(terminate(h, 1))
    end = time.monotonic()+10
    while True:
        a = Accounting()
        check(query_job(h, 1, c.byref(a), c.sizeof(a), None))
        if not a.active:
            return
        if time.monotonic() >= end:
            raise RuntimeError("Windows compute job did not drain")
        time.sleep(.02)


@contextlib.contextmanager
def start(request, mib, lock):
    with job(mib, True) as h:
        si = subprocess.STARTUPINFO(lpAttributeList={"handle_list": [lock]})
        os.set_handle_inheritable(lock, True)
        try:
            p = subprocess.Popen([sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--shard"],
                                 stdin=subprocess.PIPE, startupinfo=si, close_fds=True, creationflags=LOW)
        finally:
            os.set_handle_inheritable(lock, False)
        attached = False
        try:
            check(assign(h, int(p._handle)))
            attached = True
            verify(h, int(p._handle), mib, True)
            # A single framed gate. EOF, even after a short write, is not permission to start.
            gate = (json.dumps([str(request), mib, lock])+"\n").encode("ascii")
            if len(gate)>65536:
                raise ValueError("compute worker gate too large")
            p.stdin.write(gate)
            p.stdin.close()
            yield p
        finally:
            try:
                if attached:
                    drain(h)  # also stop/wait for a descendant if the Python worker died early
                elif p.poll() is None:
                    p.kill()  # gated bootstrap, no descendant or input path
            finally:
                try:
                    if not p.stdin.closed:
                        p.stdin.close()
                finally:
                    p.wait(timeout=10)


def run(requests, mib, lock):
    with contextlib.ExitStack() as stack:
        procs = [stack.enter_context(start(p, mib, lock)) for p in requests]
        while True:
            codes = [p.poll() for p in procs]
            if any(rc not in (None, 0) for rc in codes):
                raise RuntimeError(f"Windows translation worker failed: {codes}")
            if all(rc == 0 for rc in codes):
                return
            time.sleep(.05)


def child():
    line = sys.stdin.buffer.readline(65537)
    if not line.endswith(b"\n") or len(line)>65536:
        raise ValueError("no complete Windows compute gate")
    request, mib, lock = json.loads(line)
    # NULL queries this worker's immediate job. No job handle in a worker can keep it alive after
    # coordinator death. No library/source inputs touched before membership and effective limits.
    verify(None, current(), mib, True)
    os.set_handle_inheritable(lock, False)
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from translate_pipeline_compute import run_request
        run_request(Path(request))
    finally:
        check(close(lock))


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    if sys.argv[1:] != ["--shard"]:
        raise SystemExit(2)
    child()
