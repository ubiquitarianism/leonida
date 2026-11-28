// Opt-in startup trace: gta6rt --trace FILE.
//
// One JSON object per line, flushed per event so the file survives a crash. Off by default; every entry point is a
// single relaxed load when it is off. Events:
//   {"seq":1,"t":0.000,"kind":"trace_open"}
//   {"seq":2,"t":0.001,"kind":"tls_index0","owner":"gta6rt.exe","moved_to":4}   (first executable mapped, once)
//   {"seq":3,"t":0.001,"kind":"map","module":"a.dll","base":"0x...","size":"0x2000","relocated":false}
//   {"seq":4,"t":0.002,"kind":"tls_callback","module":"a.dll","at":"a.dll+0x1200","reason":1}
//   {"seq":5,"t":0.002,"kind":"dllmain","module":"a.dll","result":1}
//   {"seq":6,"t":0.002,"kind":"delay_import_unimplemented","module":"a.dll","import":"x.dll!Bar"}   (at load)
//   {"seq":7,"t":0.010,"kind":"unimplemented_call","import":"x.dll!Foo","caller":"a.dll+0x3f10"}   (first call only)
//   {"seq":8,"t":0.011,"kind":"debug_string","text":"..."}
//   {"seq":9,"t":0.020,"kind":"exception","code":"0xc0000005","at":"a.dll+0x4a00","access":"read","address":"0x0",
//        "region":{"base":"0x0","allocation_base":"0x0","size":"0x7ffe0000","state":"0x10000","protect":"0x1",
//        "type":"0x0"},"regs":{...},"stack":["a.dll+0x4a00","a.dll+0x2210"]}
//   {"seq":10,"t":0.030,"kind":"exit","code":0}
//
// The exception handler is vectored, logs and returns EXCEPTION_CONTINUE_SEARCH: it never changes what the process
// does (except LEONIDA_SKIP_INT3, below).
// OutputDebugStringA/W raise DBG_PRINTEXCEPTION_C / DBG_PRINTEXCEPTION_WIDE_C when no debugger is attached, so the
// same handler records their text without a shim.
//
// The file can quote whatever the game prints: it stays on the player's machine (AGENTS.md, "No game data").
#pragma once

#include <windows.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <mutex>
#include <string>

#include "shims.h"

namespace rt::startup_trace {

namespace detail {
// What the exception handler touches lives at namespace scope and is constant-initialized: it exists before any code
// runs, on every thread, and is never destroyed. A block-scope static is initialized on its first use, and that can be
// on a title thread, where under Wine the MSVC thread-safe-static guard has been seen to pass while the object was
// still zero (old upstream 7599): LEONIDA_SKIP_INT3 stayed off and the int3 set and the trace lock were never
// constructed.
inline constinit std::atomic<bool> on{false};

// The trace lock. An SRW lock needs no constructor or destructor, so vectored exception handlers and late teardown
// paths can take it at any time, also after normal statics have been destroyed.
class Lock {
public:
    void lock() noexcept { AcquireSRWLockExclusive(&srw_); }
    void unlock() noexcept { ReleaseSRWLockExclusive(&srw_); }

private:
    SRWLOCK srw_ = SRWLOCK_INIT;
};
inline constinit Lock trace_lock{};
inline Lock& mutex() noexcept { return trace_lock; }
inline FILE* file = nullptr;
inline unsigned long long seq = 0;
inline std::chrono::steady_clock::time_point start;
inline void* veh = nullptr;
inline thread_local bool in_handler = false;
// With nothing to handle it, OutputDebugStringW's DBG_PRINTEXCEPTION_WIDE_C falls back to OutputDebugStringA on its
// own ANSI copy (ExceptionInformation[3]), a second, lossy exception for the same text. Every debug string on the
// thread resets this, so only the exception right after the wide one, with that same pointer, is skipped.
inline thread_local ULONG_PTR wide_fallback = 0;

inline void append_escaped(std::string& out, const char* s, size_t n) {
    static const char hex[] = "0123456789abcdef";
    for (size_t i = 0; i < n; ++i) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    out += "\\u00";
                    out += hex[c >> 4];
                    out += hex[c & 15];
                } else {
                    out += static_cast<char>(c);  // UTF-8 passes through
                }
        }
    }
}
}  // namespace detail

inline bool enabled() noexcept { return detail::on.load(std::memory_order_acquire); }

// A JSON string literal, quotes included.
inline std::string quote(const std::string& s) {
    std::string out = "\"";
    detail::append_escaped(out, s.data(), s.size());
    out += '"';
    return out;
}

inline std::string hex(uint64_t v) {
    char b[24];
    std::snprintf(b, sizeof b, "0x%llx", static_cast<unsigned long long>(v));
    return b;
}

// "module+0xRVA" for a mapped image or a host module, else the bare address.
inline std::string where(uint64_t address) { return shims::module_offset(address); }

// `fields` is the rest of the object, already JSON: `"module":"a.dll","result":1`. Empty is allowed.
inline void event(const char* kind, const std::string& fields = {}) {
    if (!enabled()) return;
    const std::lock_guard lock(detail::mutex());
    if (!detail::file) return;
    const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - detail::start).count();
    std::fprintf(detail::file, "{\"seq\":%llu,\"t\":%.6f,\"kind\":\"%s\"%s%s}\n", ++detail::seq, t, kind,
                 fields.empty() ? "" : ",", fields.c_str());
    std::fflush(detail::file);
}

namespace detail {
constexpr DWORD kDbgPrint = 0x40010006;      // DBG_PRINTEXCEPTION_C
constexpr DWORD kDbgPrintWide = 0x4001000A;  // DBG_PRINTEXCEPTION_WIDE_C
constexpr DWORD kCppException = 0xE06D7363;  // MSVC throw: logged with its address, never its payload
constexpr int kMaxFrames = 16;

inline std::string stack_from(const CONTEXT& start_context) {
    CONTEXT c = start_context;
    std::string out = "[";
    for (int i = 0; i < kMaxFrames && c.Rip; ++i) {
        if (i) out += ',';
        out += quote(where(c.Rip));
        DWORD64 image_base = 0;
        PRUNTIME_FUNCTION fn = RtlLookupFunctionEntry(c.Rip, &image_base, nullptr);
        if (!fn) {  // a leaf function: the return address is at [rsp]
            if (!c.Rsp || IsBadReadPtr(reinterpret_cast<void*>(c.Rsp), 8)) break;
            c.Rip = *reinterpret_cast<DWORD64*>(c.Rsp);
            c.Rsp += 8;
            continue;
        }
        void* handler_data = nullptr;
        DWORD64 frame = 0;
        RtlVirtualUnwind(UNW_FLAG_NHANDLER, image_base, c.Rip, fn, &c, &handler_data, &frame, nullptr);
    }
    return out + "]";
}

inline std::string utf8_from_wide(const wchar_t* w, int len) {
    const int bytes = len > 0 ? WideCharToMultiByte(CP_UTF8, 0, w, len, nullptr, 0, nullptr, nullptr) : 0;
    std::string s(bytes > 0 ? bytes : 0, '\0');
    if (bytes > 0) WideCharToMultiByte(CP_UTF8, 0, w, len, s.data(), bytes, nullptr, nullptr);
    return s;
}

// OutputDebugStringA text is in the ANSI code page, not UTF-8: convert it, or a byte such as 0xE9 would land in the
// file as is and break the JSON lines for a strict UTF-8 reader.
inline std::string utf8_from_ansi(const char* a, int len) {
    const int chars = len > 0 ? MultiByteToWideChar(CP_ACP, 0, a, len, nullptr, 0) : 0;
    if (chars <= 0) return {};
    std::wstring w(static_cast<size_t>(chars), L'\0');
    MultiByteToWideChar(CP_ACP, 0, a, len, w.data(), chars);
    return utf8_from_wide(w.data(), chars);
}

inline std::string debug_text(const EXCEPTION_RECORD& r) {
    // Parameters: [0] length in characters including the terminator, [1] the string.
    if (r.NumberParameters < 2 || !r.ExceptionInformation[1]) return {};
    const size_t n = static_cast<size_t>(r.ExceptionInformation[0]);
    if (r.ExceptionCode == kDbgPrintWide) {
        const auto* w = reinterpret_cast<const wchar_t*>(r.ExceptionInformation[1]);
        return utf8_from_wide(w, static_cast<int>(n ? n - 1 : wcslen(w)));
    }
    const auto* a = reinterpret_cast<const char*>(r.ExceptionInformation[1]);
    return utf8_from_ansi(a, static_cast<int>(n ? n - 1 : strlen(a)));
}

// LOCAL DIAGNOSTIC for one of the title's asserts (docs/re/answers.md, "transient pool"): at that assert's int3
// the title has just computed the peak of a list of resources with lifetimes against a fixed pool. The int3's
// location is measured locally and not kept here: LEONIDA_POOL_DIAG_AT names it as the trace prints it
// (`<module>+0x<offset>`), and the dump is off without it.
// rdi = the list (24-byte records: entry pointer, u32 size rounded to 64 KiB), r13 = the count, rax = the peak,
// rcx = the pool size. An entry is {record pointer, ..., u32 first pass at +16, u32 last pass at +20}; a record has
// its usage word at +14, kind word at +16 and, for a texture, width/height/depth words at +56/+58/+60, the dimension
// and DXGI format bytes at +62/+63 and the mip count at +66. Printed once, to stderr; every read is checked.
inline void dump_transient_pool(const CONTEXT& c) {
    static std::atomic<bool> done{false};
    if (done.exchange(true)) return;
    const auto read = [](uint64_t at, void* out, size_t n) {
        SIZE_T got = 0;
        return ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(at), out, n, &got) && got == n;
    };
    std::fprintf(stderr, "[rt] pool-diag: peak %llu B, pool %llu B, %llu entries (i size first last usage kind fmt WxHxD mips)\n",
                 static_cast<unsigned long long>(c.Rax), static_cast<unsigned long long>(c.Rcx),
                 static_cast<unsigned long long>(c.R13));
    for (uint64_t i = 0; i < c.R13 && i < 4096; ++i) {
        uint64_t entry = 0, record = 0;
        uint32_t size = 0, passes[2]{};
        uint8_t desc[16]{};
        uint16_t words[2]{};
        if (!read(c.Rdi + i * 24, &entry, 8) || !read(c.Rdi + i * 24 + 8, &size, 4) || !read(entry, &record, 8) ||
            !read(entry + 16, passes, 8) || !read(record + 14, words, 4) || !read(record + 56, desc, 16))
            break;
        uint16_t w, h, d;
        std::memcpy(&w, desc, 2);
        std::memcpy(&h, desc + 2, 2);
        std::memcpy(&d, desc + 4, 2);
        std::fprintf(stderr, "[rt] pool-diag: %llu %u %u %u 0x%x 0x%x %u %ux%ux%u %u\n",
                     static_cast<unsigned long long>(i), size, passes[0], passes[1], words[0], words[1],
                     unsigned(desc[7]), unsigned(w), unsigned(h), unsigned(d), unsigned(desc[10]));
    }
}

inline void log_exception(const EXCEPTION_POINTERS& info) {
    const EXCEPTION_RECORD& r = *info.ExceptionRecord;
    try {
        static const std::string pool_diag_at = [] {
            char at[128]{};
            const DWORD n = GetEnvironmentVariableA("LEONIDA_POOL_DIAG_AT", at, sizeof(at));
            return n && n < sizeof(at) ? std::string(at, n) : std::string();
        }();
        if (r.ExceptionCode == EXCEPTION_BREAKPOINT && info.ContextRecord && !pool_diag_at.empty() &&
            where(reinterpret_cast<uint64_t>(r.ExceptionAddress)).ends_with(pool_diag_at))
            dump_transient_pool(*info.ContextRecord);
        if (r.ExceptionCode == kDbgPrint || r.ExceptionCode == kDbgPrintWide) {
            const ULONG_PTR fallback = wide_fallback;
            const bool wide = r.ExceptionCode == kDbgPrintWide;
            wide_fallback = wide && r.NumberParameters >= 4 ? r.ExceptionInformation[3] : 0;
            if (!wide && fallback && r.NumberParameters >= 2 && r.ExceptionInformation[1] == fallback) return;
            std::string text = debug_text(r);
            while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) text.pop_back();
            event("debug_string", "\"text\":" + quote(text));
        } else {
            std::string f = "\"code\":" + quote(hex(r.ExceptionCode)) +
                            ",\"at\":" + quote(where(reinterpret_cast<uint64_t>(r.ExceptionAddress)));
            if (r.ExceptionCode == EXCEPTION_ACCESS_VIOLATION && r.NumberParameters >= 2) {
                const auto op = r.ExceptionInformation[0];
                f += ",\"access\":" + quote(op == 0 ? "read" : op == 1 ? "write" : "execute") +
                     ",\"address\":" + quote(hex(r.ExceptionInformation[1]));
                // The faulting page's region: committed or not, its protection, private or a mapped view.
                MEMORY_BASIC_INFORMATION m{};
                if (VirtualQuery(reinterpret_cast<const void*>(r.ExceptionInformation[1]), &m, sizeof(m)))
                    f += ",\"region\":{\"base\":" + quote(hex(reinterpret_cast<uint64_t>(m.BaseAddress))) +
                         ",\"allocation_base\":" + quote(hex(reinterpret_cast<uint64_t>(m.AllocationBase))) +
                         ",\"size\":" + quote(hex(m.RegionSize)) + ",\"state\":" + quote(hex(m.State)) +
                         ",\"protect\":" + quote(hex(m.Protect)) + ",\"type\":" + quote(hex(m.Type)) + "}";
                // The runs the fault page's neighbours sit in: a writable run that ends mid-granule is a commit
                // shorter than the console's 64 KiB granule, a run that ends at a granule boundary is not. VirtualQuery
                // reports each from the page asked about (its BaseAddress is that page), so a run is the page, the
                // allocation it belongs to and how far the run goes from there ("size").
                std::string runs;
                const uintptr_t page = static_cast<uintptr_t>(r.ExceptionInformation[1]) & ~uintptr_t(0xfff);
                for (int i = -1; i <= 1; ++i) {
                    const uintptr_t at = page + static_cast<uintptr_t>(static_cast<intptr_t>(i)) * 0x1000;
                    MEMORY_BASIC_INFORMATION n{};
                    if (!VirtualQuery(reinterpret_cast<const void*>(at), &n, sizeof(n))) continue;
                    if (!runs.empty()) runs += ",";
                    runs += "{\"at\":" + quote(hex(at)) + ",\"allocation_base\":" +
                            quote(hex(reinterpret_cast<uint64_t>(n.AllocationBase))) + ",\"size\":" +
                            quote(hex(n.RegionSize)) + ",\"state\":" + quote(hex(n.State)) + ",\"protect\":" +
                            quote(hex(n.Protect)) + ",\"type\":" + quote(hex(n.Type)) + "}";
                }
                if (!runs.empty()) f += ",\"runs\":[" + runs + "]";
            }
            if (r.ExceptionCode == kCppException) f += ",\"cpp\":true";
            if (info.ContextRecord) {
                // The registers at the fault: what an analyst needs to read a title assert (which value, which list).
                const CONTEXT& c = *info.ContextRecord;
                const std::pair<const char*, DWORD64> regs[] = {
                    {"rax", c.Rax}, {"rcx", c.Rcx}, {"rdx", c.Rdx}, {"rbx", c.Rbx}, {"rsi", c.Rsi}, {"rdi", c.Rdi},
                    {"r8", c.R8},   {"r9", c.R9},   {"r10", c.R10}, {"r11", c.R11}, {"r12", c.R12}, {"r13", c.R13},
                    {"r14", c.R14}, {"r15", c.R15}, {"rbp", c.Rbp}, {"rsp", c.Rsp}};
                f += ",\"regs\":{";
                for (size_t i = 0; i < std::size(regs); ++i)
                    f += std::string(i ? "," : "") + "\"" + regs[i].first + "\":" + quote(hex(regs[i].second));
                f += "}";
            }
            if (info.ContextRecord) f += ",\"stack\":" + stack_from(*info.ContextRecord);
            event("exception", f);
        }
    } catch (...) {
        // Out of memory while formatting: drop the event rather than disturb the process.
    }
}

// LEONIDA_SKIP_INT3 (diagnostic, with --trace): an int3 in guest code is logged once per address and stepped over,
// so a boot can be followed past the title's soft asserts (an int3 followed by code that carries on). Off by default:
// a console title never reaches one in a correct run, so each skipped address is a finding, not a fix.
//
// open() reads the variable, on the thread that opens the trace and before any guest code runs.
inline constinit std::atomic<bool> skip_int3_on{false};
inline bool skip_int3_requested() { return GetEnvironmentVariableW(L"LEONIDA_SKIP_INT3", nullptr, 0) != 0; }

// The addresses already logged: an open-addressing table of atomics, so the handler needs no constructor, lock or
// allocation for it. When it is full, a site is logged again rather than lost.
constexpr size_t kInt3Sites = 4096;  // a power of two: 12 bits of the hash below
inline constinit std::atomic<const void*> int3_sites[kInt3Sites]{};

inline bool first_int3_at(const void* at) {
    const auto key = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(at));
    size_t i = static_cast<size_t>((key * 0x9E3779B97F4A7C15ull) >> 52);
    for (size_t probe = 0; probe < kInt3Sites; ++probe, i = (i + 1) & (kInt3Sites - 1)) {
        const void* seen = int3_sites[i].load(std::memory_order_acquire);
        if (!seen && int3_sites[i].compare_exchange_strong(seen, at, std::memory_order_acq_rel)) return true;
        if (seen == at) return false;  // also when another thread has just claimed this slot for the same site
    }
    return true;
}

inline bool readable_byte(const unsigned char* p) {
    MEMORY_BASIC_INFORMATION m{};
    if (!p || VirtualQuery(p, &m, sizeof m) != sizeof m || m.State != MEM_COMMIT) return false;
    return (m.Protect & (PAGE_NOACCESS | PAGE_GUARD)) == 0;
}

// The int3 (0xCC) a breakpoint stopped at, or null. Windows reports Rip and ExceptionAddress on the 0xCC byte, and so
// did Wine 11.22 (Staging) in a probe here; Spacecer2's Wine 11.22 boot reported both one byte past it (old upstream
// 7599). The candidate taken is the one ExceptionAddress agrees with (on it, or one byte before it), the byte at Rip
// first, so the Windows form is handled as before. With Rip one past the 0xCC and ExceptionAddress on it, a run of
// 0xCC bytes still gives the int3 ExceptionAddress names, not the one at Rip. When both name the byte after the int3
// (the second form) and that byte is another 0xCC, the next int3 is taken, and stepped over with this one.
inline const unsigned char* int3_from(const EXCEPTION_RECORD& r, const CONTEXT& c) {
    const auto* rip = reinterpret_cast<const unsigned char*>(c.Rip);
    const auto* address = static_cast<const unsigned char*>(r.ExceptionAddress);
    if (!rip) return nullptr;
    for (const unsigned char* at : {rip, rip - 1})
        if ((address == at || address == at + 1) && readable_byte(at) && *at == 0xCC) return at;
    return nullptr;
}

// With LEONIDA_SKIP_INT3, steps over an int3: Rip ends one byte past the 0xCC in either form.
inline bool skip_int3(EXCEPTION_POINTERS* info) {
    if (!skip_int3_on.load(std::memory_order_relaxed)) return false;
    if (!info || !info->ExceptionRecord || !info->ContextRecord) return false;
    if (info->ExceptionRecord->ExceptionCode != EXCEPTION_BREAKPOINT) return false;
    const unsigned char* at = int3_from(*info->ExceptionRecord, *info->ContextRecord);
    if (!at) return false;
    if (first_int3_at(at)) event("int3_skipped", "\"at\":" + quote(where(reinterpret_cast<uint64_t>(at))));
    info->ContextRecord->Rip = reinterpret_cast<DWORD64>(at) + 1;
    return true;
}

inline LONG CALLBACK handler(EXCEPTION_POINTERS* info) {
    if (!enabled() || in_handler || !info || !info->ExceptionRecord) return EXCEPTION_CONTINUE_SEARCH;
    if (skip_int3(info)) return EXCEPTION_CONTINUE_EXECUTION;
    in_handler = true;
#ifdef _MSC_VER
    // Under /EHsc, catch(...) does not catch SEH faults (e.g. a bad DBG_PRINTEXCEPTION_C pointer or corrupt unwind).
    // Keep the __try frame free of C++ destructors so in_handler is always reset and the exception still propagates.
    __try {
        log_exception(*info);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
    }
#else
    struct Reset {
        ~Reset() { in_handler = false; }
    } reset;
    log_exception(*info);
#endif
    in_handler = false;
    return EXCEPTION_CONTINUE_SEARCH;
}
}  // namespace detail

inline void close();

// Creates the trace file and installs the exception handler. The file must not exist yet ("x" mode): the trace is
// opened before the loader reads anything, so overwriting would let `--trace` destroy an input, such as the image
// itself. Returns false if the file exists or cannot be created; `error`, when given, receives the errno.
inline bool open(const wchar_t* path, int* error = nullptr) {
    {
        const std::lock_guard lock(detail::mutex());
        if (detail::file) return true;
        const errno_t e = _wfopen_s(&detail::file, path, L"wxb");
        if (e != 0 || !detail::file) {
            detail::file = nullptr;
            if (error) *error = e ? e : EINVAL;
            return false;
        }
        detail::start = std::chrono::steady_clock::now();
        detail::seq = 0;
        detail::skip_int3_on.store(detail::skip_int3_requested(), std::memory_order_relaxed);
        static const int atexit_registered = std::atexit(&close);
        (void)atexit_registered;
    }
    detail::veh = AddVectoredExceptionHandler(1, &detail::handler);
    detail::on.store(true, std::memory_order_release);
    event("trace_open", "\"pid\":" + std::to_string(GetCurrentProcessId()));
    return true;
}

// Removes the handler and closes the file (tests and process exit via atexit).
inline void close() {
    detail::on.store(false, std::memory_order_release);
    if (detail::veh) RemoveVectoredExceptionHandler(detail::veh);
    detail::veh = nullptr;
    const std::lock_guard lock(detail::mutex());
    if (detail::file) std::fclose(detail::file);
    detail::file = nullptr;
}

}  // namespace rt::startup_trace
