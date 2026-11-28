#pragma once
// LEONIDA_SAMPLE_MS=<ms> (diagnostic, off by default): where the process's busy threads spend their CPU, without a
// debugger or perf. A thread wakes every <ms>, and for each thread of the process that used CPU since its last look
// suspends it, copies its context and resumes it at once; the stack is then unwound outside the suspension (through
// the .pdata every module registers, the title's included), under __try, since the thread runs on meanwhile. Every
// LEONIDA_SAMPLE_REPORT_S seconds (10) one block lists the busiest threads (CPU share) and the most frequent call
// chains (up to eight frames, innermost first): title frames as t+RVA of the function, others as module+RVA. Nothing is
// allocated or logged while a thread is suspended, so a suspended thread's locks are never waited for.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <tlhelp32.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "submit_phases.h"

namespace leonida::thread_sampler {

constexpr int kFrames = 8;
using Chain = std::array<std::uint64_t, kFrames>;  // function starts (or addresses), innermost first
// LEONIDA_SAMPLE_SHIMS=1 (with LEONIDA_SAMPLE_MS): each sample is unwound this deep to tell whether a title thread
// is inside our runtime (a frame of this executable called, at any depth, from title code), and where.
constexpr int kDeepFrames = 48;
using DeepChain = std::array<std::uint64_t, kDeepFrames>;


struct Sampler {
    std::uint8_t* title = nullptr;
    std::uint64_t title_size = 0;
    DWORD period_ms = 10;
    DWORD report_ms = 10000;
    bool waits = false;  // LEONIDA_SAMPLE_WAITS=1: every thread, busy or waiting (where the wall time goes)
    bool shims = false;  // LEONIDA_SAMPLE_SHIMS=1: per title thread, the share of its samples inside our runtime
    std::uint64_t self_base = 0, self_size = 0;  // this executable (the runtime)
    // LEONIDA_SAMPLE_DETAIL=<id>[,<id>...]: for these threads, the 10 most frequent chains, 16 frames deep, each as a
    // share of that thread's samples (thread ids are stable from boot to boot: the title's main, render and submit).
    std::vector<DWORD> detail;
    // LEONIDA_SAMPLE_FIND=<rva>[,<rva>...] (title function starts): which threads have the function on their stack
    // (48 frames deep), how often, and their most frequent chain under it, every report.
    std::vector<std::uint64_t> find;
};

// Per title thread (one with a title frame in its chain): how many samples were inside our runtime, and through which
// entry (the runtime frame the title called) and which runtime function was innermost.
struct ShimCounts {
    std::uint64_t title = 0;      // samples with a title frame
    std::uint64_t ours = 0;       // ... of which inside our runtime
    std::uint64_t ours_busy = 0;  // ... while the thread used CPU since the previous look
    std::uint64_t busy = 0;       // samples with a title frame while the thread used CPU
    std::map<std::pair<std::uint64_t, std::uint64_t>, std::uint64_t> entries;  // (our entry, title caller) -> samples
    std::map<std::uint64_t, std::uint64_t> inner;                              // innermost runtime function -> samples
    std::map<std::uint64_t, std::uint64_t> leaves;                             // innermost frame (any module) -> samples
};

// Classifies one deep chain (innermost first) for LEONIDA_SAMPLE_SHIMS.
inline void count_shims(const Sampler& s, const DeepChain& chain, int n, bool moved, ShimCounts& c) {
    const auto title = reinterpret_cast<std::uint64_t>(s.title);
    const auto in_title = [&](std::uint64_t a) { return a >= title && a < title + s.title_size; };
    const auto in_self = [&](std::uint64_t a) { return a >= s.self_base && a < s.self_base + s.self_size; };
    int first_title = -1;
    for (int i = 0; i < n; ++i)
        if (in_title(chain[i])) {
            first_title = i;
            break;
        }
    if (first_title < 0) return;
    ++c.title;
    c.busy += moved;
    int entry = -1, innermost = -1;
    for (int i = 0; i < first_title; ++i)
        if (in_self(chain[i])) {
            if (innermost < 0) innermost = i;
            entry = i;  // the outermost runtime frame below the title's
        }
    if (entry < 0) return;
    ++c.ours;
    c.ours_busy += moved;
    ++c.entries[{chain[entry], chain[first_title]}];
    ++c.inner[chain[innermost]];
    ++c.leaves[chain[0]];
}

// The chain of a context copied from a suspended thread: unwound under __try (its stack may be changing).
template <std::size_t N>
inline int unwind(CONTEXT context, std::array<std::uint64_t, N>& chain) {
    int n = 0;
    __try {
        for (; n < int(N); ++n) {
            const DWORD64 pc = context.Rip;
            if (!pc) break;
            DWORD64 image = 0;
            PRUNTIME_FUNCTION entry = RtlLookupFunctionEntry(pc, &image, nullptr);
            chain[n] = entry ? image + entry->BeginAddress : pc;
            if (!entry) {  // a leaf, or code with no unwind data: take the return address at the stack pointer
                context.Rip = *reinterpret_cast<const DWORD64*>(context.Rsp);
                context.Rsp += 8;
                continue;
            }
            void* handler_data = nullptr;
            DWORD64 frame = 0;
            RtlVirtualUnwind(UNW_FLAG_NHANDLER, image, pc, entry, &context, &handler_data, &frame, nullptr);
        }
    } __except (EXCEPTION_EXECUTE_HANDLER) {
    }
    return n;
}

inline std::string name_of(const Sampler& s, std::uint64_t at) {
    char text[96];
    const auto title = reinterpret_cast<std::uint64_t>(s.title);
    if (at >= title && at < title + s.title_size) {
        std::snprintf(text, sizeof(text), "t+%llx", static_cast<unsigned long long>(at - title));
        return text;
    }
    HMODULE module = nullptr;
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCWSTR>(at), &module) &&
        module) {
        wchar_t path[MAX_PATH] = {};
        GetModuleFileNameW(module, path, MAX_PATH);
        const wchar_t* base = path;
        for (const wchar_t* p = path; *p; ++p)
            if (*p == L'\\' || *p == L'/') base = p + 1;
        std::snprintf(text, sizeof(text), "%ls+%llx", base,
                      static_cast<unsigned long long>(at - reinterpret_cast<std::uint64_t>(module)));
        return text;
    }
    std::snprintf(text, sizeof(text), "0x%llx", static_cast<unsigned long long>(at));
    return text;
}

inline DWORD WINAPI run(LPVOID parameter) {
    const Sampler s = *static_cast<Sampler*>(parameter);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
    const DWORD self = GetCurrentThreadId(), pid = GetCurrentProcessId();
    struct Thread {
        HANDLE handle = nullptr;
        std::uint64_t cpu = 0;   // 100 ns, at the last look
        std::uint64_t used = 0;  // since the last report
        std::uint64_t samples = 0;
    };
    std::map<DWORD, Thread> threads;
    std::map<std::pair<DWORD, Chain>, std::uint64_t> chains;
    std::map<DWORD, ShimCounts> shim_counts;
    constexpr int kDetailFrames = 16;
    using DetailChain = std::array<std::uint64_t, kDetailFrames>;
    std::map<std::pair<DWORD, DetailChain>, std::uint64_t> detail_chains;
    std::map<std::pair<std::uint64_t, DWORD>, std::map<DetailChain, std::uint64_t>> found;  // (rva, thread) -> chains
    std::uint64_t total = 0;
    ULONGLONG listed = 0, reported = GetTickCount64();
    const auto cpu_of = [](HANDLE h) {
        FILETIME created{}, exited{}, kernel{}, user{};
        if (!GetThreadTimes(h, &created, &exited, &kernel, &user)) return std::uint64_t(0);
        const auto ticks = [](const FILETIME& f) { return (std::uint64_t(f.dwHighDateTime) << 32) | f.dwLowDateTime; };
        return ticks(kernel) + ticks(user);
    };
    for (;;) {
        Sleep(s.period_ms);
        const ULONGLONG now = GetTickCount64();
        if (now - listed >= 2000) {  // the thread list, every 2 s
            listed = now;
            const HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
            if (snap != INVALID_HANDLE_VALUE) {
                THREADENTRY32 e{sizeof(e)};
                for (BOOL ok = Thread32First(snap, &e); ok; ok = Thread32Next(snap, &e)) {
                    if (e.th32OwnerProcessID != pid || e.th32ThreadID == self || threads.count(e.th32ThreadID)) continue;
                    const HANDLE h = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION,
                                                FALSE, e.th32ThreadID);
                    if (h) threads[e.th32ThreadID] = {h, cpu_of(h)};
                    e.dwSize = sizeof(e);
                }
                CloseHandle(snap);
            }
        }
        for (auto it = threads.begin(); it != threads.end();) {
            Thread& t = it->second;
            const std::uint64_t cpu = cpu_of(t.handle);
            DWORD code = 0;
            if (GetExitCodeThread(t.handle, &code) && code != STILL_ACTIVE) {
                CloseHandle(t.handle);
                it = threads.erase(it);
                continue;
            }
            const bool moved = cpu != t.cpu;
            const bool busy = moved || s.waits;
            t.used += cpu - t.cpu;
            t.cpu = cpu;
            if (busy) {
                CONTEXT context{};
                context.ContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER;
                bool got = false;
                if (SuspendThread(t.handle) != DWORD(-1)) {
                    got = GetThreadContext(t.handle, &context) != FALSE;
                    ResumeThread(t.handle);
                }
                if (got) {
                    Chain chain{};
                    if (std::find(s.detail.begin(), s.detail.end(), it->first) != s.detail.end()) {
                        DetailChain deep{};
                        unwind(context, deep);
                        ++detail_chains[{it->first, deep}];
                    }
                    if (s.shims || !s.find.empty()) {
                        DeepChain deep{};
                        const int n = unwind(context, deep);
                        std::copy_n(deep.begin(), kFrames, chain.begin());
                        if (s.shims) count_shims(s, deep, n, moved, shim_counts[it->first]);
                        const auto title = reinterpret_cast<std::uint64_t>(s.title);
                        for (const std::uint64_t rva : s.find)
                            for (int f = 0; f < n; ++f)
                                if (deep[f] == title + rva) {
                                    DetailChain inner{};
                                    std::copy_n(deep.begin(), (std::min)(kDetailFrames, n), inner.begin());
                                    ++found[{rva, it->first}][inner];
                                    break;
                                }
                    } else {
                        unwind(context, chain);
                    }
                    ++chains[{it->first, chain}];
                    ++t.samples;
                    ++total;
                }
            }
            ++it;
        }
        if (now - reported < s.report_ms || !total) continue;
        const double seconds = double(now - reported) / 1000.0;
        reported = now;
        std::vector<std::pair<std::uint64_t, DWORD>> busiest;
        for (auto& [id, t] : threads)
            if (t.used) busiest.emplace_back(t.used, id);
        std::sort(busiest.rbegin(), busiest.rend());
        std::string line;
        for (std::size_t i = 0; i < busiest.size() && i < 8; ++i) {
            char text[64];
            std::snprintf(text, sizeof(text), "%s%lu %.0f%%", i ? ", " : "", busiest[i].second,
                          double(busiest[i].first) * 1e-5 / seconds);
            line += text;
        }
        std::fprintf(stderr, "[rt] sampler: %llu samples in %.0f s; CPU by thread: %s\n",
                     static_cast<unsigned long long>(total), seconds, line.c_str());
        std::vector<std::pair<std::uint64_t, const std::pair<DWORD, Chain>*>> top;
        for (const auto& c : chains) top.emplace_back(c.second, &c.first);
        std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        const auto print = [&](const auto& entry, std::uint64_t of, const char* what) {
            std::string frames;
            for (int f = 0; f < kFrames && entry.second->second[f]; ++f)
                frames += (f ? " < " : "") + name_of(s, entry.second->second[f]);
            std::fprintf(stderr, "[rt] sampler: %s %5.1f%% thread %lu: %s\n", what, 100.0 * double(entry.first) / double(of),
                         entry.second->first, frames.c_str());
        };
        for (std::size_t i = 0; i < top.size() && i < 16; ++i) print(top[i], total, " ");
        // Apart: the submitting thread's chains, and the two busiest threads', each as a share of that thread's samples.
        std::vector<DWORD> apart;
        if (const DWORD submitter = submit_phases::totals().thread.load(std::memory_order_relaxed)) apart.push_back(submitter);
        for (std::size_t i = 0; i < busiest.size() && i < 2; ++i)
            if (std::find(apart.begin(), apart.end(), busiest[i].second) == apart.end()) apart.push_back(busiest[i].second);
        // With waiting threads sampled: each thread's own most frequent chain, busiest first, so a thread that waits
        // (the title's main thread behind its render thread, a render thread behind a lock) shows where.
        if (s.waits) {
            std::vector<std::pair<std::uint64_t, DWORD>> all;
            for (auto& [id, t] : threads)
                if (t.samples) all.emplace_back(t.used, id);
            std::sort(all.rbegin(), all.rend());
            std::size_t listed = 0;
            for (const auto& [used, id] : all) {
                if (listed++ >= 24) break;
                const auto it = threads.find(id);
                if (it == threads.end() || !it->second.samples) continue;
                for (const auto& entry : top)
                    if (entry.second->first == id) {
                        print(entry, it->second.samples, "thread");
                        break;
                    }
            }
        }
        for (const DWORD id : apart) {
            const auto it = threads.find(id);
            if (it == threads.end() || !it->second.samples) continue;
            std::size_t shown = 0;
            for (const auto& entry : top)
                if (entry.second->first == id && shown++ < 8)
                    print(entry, it->second.samples, id == apart.front() && submit_phases::totals().thread.load() == id
                                                         ? "submit"
                                                         : "busy  ");
        }
        for (const DWORD id : s.detail) {
            std::vector<std::pair<std::uint64_t, const DetailChain*>> mine;
            std::uint64_t all = 0;
            for (const auto& [key, n] : detail_chains)
                if (key.first == id) {
                    mine.emplace_back(n, &key.second);
                    all += n;
                }
            std::sort(mine.begin(), mine.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
            for (std::size_t i = 0; i < mine.size() && i < 10; ++i) {
                std::string frames;
                for (int f = 0; f < kDetailFrames && (*mine[i].second)[f]; ++f)
                    frames += (f ? " < " : "") + name_of(s, (*mine[i].second)[f]);
                std::fprintf(stderr, "[rt] sampler: detail thread %lu %5.1f%%: %s\n", id,
                             100.0 * double(mine[i].first) / double(all), frames.c_str());
            }
        }
        detail_chains.clear();
        for (const auto& [key, chains_of] : found) {
            std::uint64_t all = 0;
            const DetailChain* top = nullptr;
            std::uint64_t top_n = 0;
            for (const auto& [c, n] : chains_of) {
                all += n;
                if (n > top_n) top_n = n, top = &c;
            }
            std::string frames;
            for (int f = 0; top && f < kDetailFrames && (*top)[f]; ++f) frames += (f ? " < " : "") + name_of(s, (*top)[f]);
            const auto t = threads.find(key.second);
            std::fprintf(stderr, "[rt] sampler: find t+%llx: thread %lu %llu samples of %llu; top chain (%llu): %s\n",
                         static_cast<unsigned long long>(key.first), key.second, static_cast<unsigned long long>(all),
                         static_cast<unsigned long long>(t == threads.end() ? 0 : t->second.samples),
                         static_cast<unsigned long long>(top_n), frames.c_str());
        }
        found.clear();
        if (s.shims) {
            // Title threads by their samples inside our runtime: where the title's own threads (its main, render and
            // Subrender workers, streaming) spend time in our shims, waiting or running.
            const auto top_of = [&](const auto& counts, std::size_t limit, bool pair_key) {
                std::vector<std::pair<std::uint64_t, std::string>> v;
                for (const auto& [key, n] : counts) {
                    std::string name;
                    if constexpr (requires { key.first; }) {
                        name = name_of(s, key.first) + "<-" + name_of(s, key.second);
                    } else {
                        name = name_of(s, key);
                    }
                    v.emplace_back(n, std::move(name));
                }
                (void)pair_key;
                std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
                std::string text;
                for (std::size_t i = 0; i < v.size() && i < limit; ++i) {
                    char n[24];
                    std::snprintf(n, sizeof(n), " %llu", static_cast<unsigned long long>(v[i].first));
                    text += (i ? ", " : "") + v[i].second + n;
                }
                return text;
            };
            std::vector<std::pair<std::uint64_t, DWORD>> order;
            ShimCounts all;
            for (const auto& [id, c] : shim_counts) {
                if (!c.title) continue;
                order.emplace_back(c.ours, id);
                all.title += c.title;
                all.ours += c.ours;
                all.ours_busy += c.ours_busy;
                all.busy += c.busy;
                for (const auto& [k, n] : c.entries) all.entries[k] += n;
                for (const auto& [k, n] : c.inner) all.inner[k] += n;
                for (const auto& [k, n] : c.leaves) all.leaves[k] += n;
            }
            std::sort(order.rbegin(), order.rend());
            std::fprintf(stderr, "[rt] sampler: shims: title-thread samples inside our runtime %llu of %llu (%.1f%%), "
                                 "%llu of them while the thread used CPU\n",
                         static_cast<unsigned long long>(all.ours), static_cast<unsigned long long>(all.title),
                         all.title ? 100.0 * double(all.ours) / double(all.title) : 0.0,
                         static_cast<unsigned long long>(all.ours_busy));
            for (std::size_t i = 0; i < order.size() && i < 12 && order[i].first; ++i) {
                const ShimCounts& c = shim_counts[order[i].second];
                std::fprintf(stderr, "[rt] sampler: shims thread %lu: %llu of %llu samples ours (%.0f%%; %llu busy), "
                                     "thread busy in %llu; entries: %s; innermost ours: %s; leaves: %s\n",
                             order[i].second, static_cast<unsigned long long>(c.ours),
                             static_cast<unsigned long long>(c.title), 100.0 * double(c.ours) / double(c.title),
                             static_cast<unsigned long long>(c.ours_busy), static_cast<unsigned long long>(c.busy),
                             top_of(c.entries, 5, true).c_str(), top_of(c.inner, 5, false).c_str(),
                             top_of(c.leaves, 3, false).c_str());
            }
            std::fprintf(stderr, "[rt] sampler: shims all: entries: %s\n", top_of(all.entries, 16, true).c_str());
            std::fprintf(stderr, "[rt] sampler: shims all: innermost ours: %s\n", top_of(all.inner, 16, false).c_str());
            std::fprintf(stderr, "[rt] sampler: shims all: leaves: %s\n", top_of(all.leaves, 12, false).c_str());
            shim_counts.clear();
        }
        chains.clear();
        total = 0;
        for (auto& [id, t] : threads) t.used = t.samples = 0;
    }
}

// From main, once the title is mapped. Does nothing without LEONIDA_SAMPLE_MS.
inline void start(std::uint8_t* title, std::uint64_t title_size) {
    char text[16] = {};
    const DWORD n = GetEnvironmentVariableA("LEONIDA_SAMPLE_MS", text, sizeof(text));
    if (!n || n >= sizeof(text)) return;
    static Sampler s;
    s.title = title;
    s.title_size = title_size;
    s.period_ms = (std::max)(1ul, std::strtoul(text, nullptr, 10));
    char report[16] = {};
    if (GetEnvironmentVariableA("LEONIDA_SAMPLE_REPORT_S", report, sizeof(report)) > 0)
        s.report_ms = (std::max)(1ul, std::strtoul(report, nullptr, 10)) * 1000;
    char waits[4] = {};
    s.waits = GetEnvironmentVariableA("LEONIDA_SAMPLE_WAITS", waits, sizeof(waits)) == 1 && waits[0] == '1';
    char detail[128] = {};
    if (const DWORD n = GetEnvironmentVariableA("LEONIDA_SAMPLE_DETAIL", detail, sizeof(detail)); n && n < sizeof(detail))
        for (char* at = detail; *at;) {
            char* end = nullptr;
            const unsigned long id = std::strtoul(at, &end, 10);
            if (end == at) break;
            if (id) s.detail.push_back(DWORD(id));
            at = *end == ',' ? end + 1 : end;
        }
    char finds[256] = {};
    if (const DWORD n = GetEnvironmentVariableA("LEONIDA_SAMPLE_FIND", finds, sizeof(finds)); n && n < sizeof(finds))
        for (char* at = finds; *at;) {
            char* end = nullptr;
            const std::uint64_t rva = std::strtoull(at, &end, 16);
            if (end == at) break;
            if (rva) s.find.push_back(rva);
            at = *end == ',' ? end + 1 : end;
        }
    char shims[4] = {};
    s.shims = GetEnvironmentVariableA("LEONIDA_SAMPLE_SHIMS", shims, sizeof(shims)) == 1 && shims[0] == '1';
    if (const auto self = reinterpret_cast<const std::uint8_t*>(GetModuleHandleW(nullptr))) {
        const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(self);
        const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(self + dos->e_lfanew);
        s.self_base = reinterpret_cast<std::uint64_t>(self);
        s.self_size = nt->OptionalHeader.SizeOfImage;
    }
    if (const HANDLE h = CreateThread(nullptr, 0, &run, &s, 0, nullptr)) {
        CloseHandle(h);
        std::fprintf(stderr, "[rt] sampler: every %lu ms, report every %lu s%s%s\n", s.period_ms, s.report_ms / 1000,
                     s.waits ? ", waiting threads too" : "", s.shims ? ", title threads inside our runtime" : "");
    }
}

}  // namespace leonida::thread_sampler
