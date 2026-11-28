// LEONIDA_SKIP_INT3 (shims/startup_trace.h): with the variable set and a trace open, an int3 is logged once per
// address and stepped over, so execution continues.
//
// Each case runs in a child process of its own (`int3_skip_test --case NAME`), so a case that crashes fails alone and
// each one opens the trace with its own environment:
//   real     a real int3 at two sites, one of them hit twice: execution continues, one event per address
//   windows  a synthetic breakpoint with Rip on the 0xCC (Windows): Rip ends one past it, one event
//   wine     Rip one byte past the 0xCC (Wine, old upstream 7599): the same result
//   null     null EXCEPTION_POINTERS, ExceptionRecord or ContextRecord: no skip and no crash
//   init     the variable is read when the trace opens: removed after open(), int3s are still skipped
//   off      without the variable, nothing is skipped
#include <windows.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>

#include "shims.h"
#include "startup_trace.h"

namespace {

int g_failed = 0;

void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "ok   " : "FAIL ", what);
    if (!ok) ++g_failed;
}

int count(const std::string& hay, const std::string& needle) {
    int n = 0;
    for (size_t at = hay.find(needle); at != std::string::npos; at = hay.find(needle, at + 1)) ++n;
    return n;
}

// noinline: the optimizer must not unroll the loop into two separate int3 sites.
__declspec(noinline) void site_a(int& reached) {
    __debugbreak();
    ++reached;
}
__declspec(noinline) void site_b(int& reached) {
    __debugbreak();
    ++reached;
}

// A trace file of this process; the text is read back after close().
struct Trace {
    std::wstring path;
    Trace() {
        wchar_t temp[MAX_PATH];
        GetTempPathW(MAX_PATH, temp);
        path = std::wstring(temp) + L"int3_skip_test_" + std::to_wstring(GetCurrentProcessId()) + L".jsonl";
        DeleteFileW(path.c_str());
        check(rt::startup_trace::open(path.c_str()), "open the startup trace file");
    }
    std::string close_and_read() {
        rt::startup_trace::close();
        std::string log;
        {
            std::ifstream in{std::filesystem::path(path)};
            std::stringstream text;
            text << in.rdbuf();
            log = text.str();
        }
        DeleteFileW(path.c_str());
        return log;
    }
};

// A synthetic EXCEPTION_BREAKPOINT for skip_int3, as the vectored handler receives it.
struct Breakpoint {
    EXCEPTION_RECORD record{};
    std::unique_ptr<CONTEXT> context = std::make_unique<CONTEXT>();  // heap: a 16-byte-aligned member pads the struct
    EXCEPTION_POINTERS pointers{};
    Breakpoint(const unsigned char* address, const unsigned char* rip) {
        record.ExceptionCode = EXCEPTION_BREAKPOINT;
        record.ExceptionAddress = const_cast<unsigned char*>(address);
        record.NumberParameters = 1;
        context->ContextFlags = CONTEXT_CONTROL;
        context->Rip = reinterpret_cast<DWORD64>(rip);
        pointers.ExceptionRecord = &record;
        pointers.ContextRecord = context.get();
    }
    bool skip() { return rt::startup_trace::detail::skip_int3(&pointers); }
    bool rip_is(const unsigned char* p) const { return context->Rip == reinterpret_cast<DWORD64>(p); }
};

// nop, int3, nop: the int3 at code + 1, with a byte on each side that is not one.
alignas(16) unsigned char g_code[4] = {0x90, 0xCC, 0x90, 0x90};

std::string event_at(const unsigned char* at) {
    const std::string where = rt::startup_trace::where(reinterpret_cast<uint64_t>(at));
    return "\"kind\":\"int3_skipped\",\"at\":" + rt::startup_trace::quote(where);
}

void case_real() {
    SetEnvironmentVariableW(L"LEONIDA_SKIP_INT3", L"1");
    Trace trace;
    int reached = 0;
    for (int i = 0; i < 2; ++i) site_a(reached);  // the same site twice: one event
    site_b(reached);                              // another site
    check(reached == 3, "execution continues past every int3");
    const std::string log = trace.close_and_read();
    check(count(log, "\"kind\":\"int3_skipped\"") == 2, "exactly two int3_skipped events (one per address)");
    check(count(log, "\"kind\":\"exception\"") == 0, "a skipped int3 is not also logged as an exception");
    check(log.find("\"at\":\"host:") != std::string::npos, "the event names the address");
}

void case_windows() {
    SetEnvironmentVariableW(L"LEONIDA_SKIP_INT3", L"1");
    Trace trace;
    const unsigned char* at = g_code + 1;
    Breakpoint first(at, at);
    check(first.skip(), "Rip on the 0xCC: skipped");
    check(first.rip_is(at + 1), "Rip on the 0xCC: Rip ends one byte past it");
    Breakpoint again(at, at);
    check(again.skip() && again.rip_is(at + 1), "the same site again: skipped the same way");
    Breakpoint no_int3(g_code + 3, g_code + 3);
    check(!no_int3.skip() && no_int3.rip_is(g_code + 3), "no 0xCC at or before Rip: not skipped, Rip unchanged");
    Breakpoint elsewhere(g_code, at);
    check(!elsewhere.skip() && elsewhere.rip_is(at), "ExceptionAddress not at the int3 or past it: not skipped");
    const std::string log = trace.close_and_read();
    check(count(log, "\"kind\":\"int3_skipped\"") == 1, "one int3_skipped event for the site");
    check(log.find(event_at(at)) != std::string::npos, "the event names the 0xCC byte");
}

void case_wine() {
    SetEnvironmentVariableW(L"LEONIDA_SKIP_INT3", L"1");
    Trace trace;
    const unsigned char* at = g_code + 1;
    Breakpoint past(at + 1, at + 1);  // ExceptionAddress and Rip one byte past the 0xCC
    check(past.skip(), "Rip one byte past the 0xCC: skipped");
    check(past.rip_is(at + 1), "Rip one byte past the 0xCC: Rip ends one byte past it");
    Breakpoint mixed(at, at + 1);  // ExceptionAddress on the 0xCC, Rip past it
    check(mixed.skip() && mixed.rip_is(at + 1), "ExceptionAddress on the 0xCC, Rip past it: skipped");
    const std::string log = trace.close_and_read();
    check(count(log, "\"kind\":\"int3_skipped\"") == 1, "one int3_skipped event for the site");
    check(log.find(event_at(at)) != std::string::npos, "the event names the 0xCC byte, not the byte after it");
}

// nop, int3, int3, nop: a run of int3 padding.
alignas(16) unsigned char g_padding[4] = {0x90, 0xCC, 0xCC, 0x90};

void case_padding() {
    SetEnvironmentVariableW(L"LEONIDA_SKIP_INT3", L"1");
    Trace trace;
    const unsigned char* at = g_padding + 1;
    Breakpoint mixed(at, at + 1);  // ExceptionAddress on the first 0xCC, Rip on the second
    check(mixed.skip() && mixed.rip_is(at + 1),
          "ExceptionAddress on an int3 followed by another, Rip past it: the int3 ExceptionAddress names is skipped");
    const std::string log = trace.close_and_read();
    check(count(log, "\"kind\":\"int3_skipped\"") == 1, "one int3_skipped event");
    check(log.find(event_at(at)) != std::string::npos, "the event names the int3 ExceptionAddress names");
}

void case_null() {
    SetEnvironmentVariableW(L"LEONIDA_SKIP_INT3", L"1");
    Trace trace;
    check(!rt::startup_trace::detail::skip_int3(nullptr), "null EXCEPTION_POINTERS: not skipped");
    Breakpoint no_record(g_code + 1, g_code + 1);
    no_record.pointers.ExceptionRecord = nullptr;
    check(!no_record.skip() && no_record.rip_is(g_code + 1), "null ExceptionRecord: not skipped");
    Breakpoint no_context(g_code + 1, g_code + 1);
    no_context.pointers.ContextRecord = nullptr;
    check(!no_context.skip(), "null ContextRecord: not skipped");
    const std::string log = trace.close_and_read();
    check(count(log, "\"kind\":\"int3_skipped\"") == 0, "no int3_skipped event");
}

void case_init() {
    SetEnvironmentVariableW(L"LEONIDA_SKIP_INT3", L"1");
    Trace trace;
    SetEnvironmentVariableW(L"LEONIDA_SKIP_INT3", nullptr);  // read when the trace opened, not at the first int3
    check(rt::startup_trace::detail::skip_int3_on.load(), "the flag holds the value read when the trace opened");
    Breakpoint bp(g_code + 1, g_code + 1);
    check(bp.skip() && bp.rip_is(g_code + 2), "an int3 after the variable is removed is still skipped");
    trace.close_and_read();
}

void case_off() {
    SetEnvironmentVariableW(L"LEONIDA_SKIP_INT3", nullptr);
    Trace trace;
    check(!rt::startup_trace::detail::skip_int3_on.load(), "without the variable the flag is off");
    Breakpoint bp(g_code + 1, g_code + 1);
    check(!bp.skip() && bp.rip_is(g_code + 1), "without the variable an int3 is not skipped");
    trace.close_and_read();
}

struct Case {
    const char* name;
    void (*run)();
};
constexpr Case kCases[] = {{"real", case_real}, {"windows", case_windows}, {"wine", case_wine},
                           {"padding", case_padding}, {"null", case_null}, {"init", case_init},
                           {"off", case_off}};

// Runs `this-exe --case NAME` and returns its exit code.
DWORD run_child(const char* name) {
    wchar_t exe[MAX_PATH];
    GetModuleFileNameW(nullptr, exe, MAX_PATH);
    std::wstring command = L"\"" + std::wstring(exe) + L"\" --case ";
    for (const char* c = name; *c; ++c) command += static_cast<wchar_t>(*c);
    STARTUPINFOW si{};
    si.cb = sizeof si;
    PROCESS_INFORMATION pi{};
    if (!CreateProcessW(exe, command.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi)) return 0xFFFFFFFF;
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 0xFFFFFFFF;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return code;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 3 && std::strcmp(argv[1], "--case") == 0) {
        for (const Case& c : kCases)
            if (std::strcmp(argv[2], c.name) == 0) {
                c.run();
                std::fflush(stdout);
                return g_failed ? 1 : 0;
            }
        std::printf("unknown case %s\n", argv[2]);
        return 2;
    }
    for (const Case& c : kCases) {
        std::printf("-- %s\n", c.name);
        std::fflush(stdout);
        const DWORD code = run_child(c.name);
        if (code != 0) std::printf("FAIL case %s exited with 0x%08lx\n", c.name, static_cast<unsigned long>(code));
        if (code != 0) ++g_failed;
    }
    std::printf("%s\n", g_failed ? "FAILED" : "all checks passed");
    return g_failed ? 1 : 0;
}
