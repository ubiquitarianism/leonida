// How many large SPIR-V modules vk_pipeline_check and the runtime let into the driver at once, by a memory budget.
//
// Measured on RADV GFX1201 (Mesa 25.3.6) with the Mesa shader cache off: a 5.7 MB translated ray-tracing compute
// module peaks at 1.96 GB of process memory compiling alone (~330 bytes per SPIR-V byte), a 0.36 MB one at 94 MB.
// Compiling the large ones strictly one at a time left the other threads idle: 40 modules (27 over 4 MiB) on 6
// threads took 308 s exclusively and 75 s in parallel, 7.0 GB and 8.7 GB peak. kBytesPerSpirvByte keeps a margin
// over the measurement. The NVIDIA 621.24 driver (RTX 3080) took 85 to 131 bytes per SPIR-V byte for synthetic compute
// modules of 128 KiB to 4 MiB (docs/d3d12_vulkan.md, "Large pipelines").
//
// The process grew past one module's peak even one at a time because glibc keeps freed memory in per-thread arenas
// (6 threads: 6.2 GB; MALLOC_ARENA_MAX=1: 2.1 GB, same modules). release_free_memory() hands it back after each
// large module.
#pragma once

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <list>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#if defined(__GLIBC__)
#include <malloc.h>
#endif

namespace leonida {

inline constexpr uint64_t kBytesPerSpirvByte = 512;

inline uint64_t large_module_estimate(uint64_t spirv_bytes) { return spirv_bytes * kBytesPerSpirvByte; }

#if defined(__linux__)
// The first unsigned number in a file ("max" and unreadable files give no value).
inline bool read_u64_file(const std::string& path, uint64_t& value) {
    std::FILE* f = std::fopen(path.c_str(), "r");
    if (!f) return false;
    unsigned long long v = 0;
    const bool ok = std::fscanf(f, "%llu", &v) == 1;
    std::fclose(f);
    value = v;
    return ok;
}

// The room left under the cgroup v2 memory.max of this process and of every cgroup above it (a container or
// systemd-run -p MemoryMax=...), or UINT64_MAX when none sets a limit. cgroup_root is a parameter for the tests.
inline uint64_t cgroup_memory_room(const std::string& self_cgroup_file = "/proc/self/cgroup",
                                   const std::string& cgroup_root = "/sys/fs/cgroup") {
    std::FILE* f = std::fopen(self_cgroup_file.c_str(), "r");
    if (!f) return UINT64_MAX;
    char line[4096];
    std::string rel;
    while (std::fgets(line, sizeof(line), f)) {
        std::string l(line);
        if (l.rfind("0::", 0) == 0) {   // the cgroup v2 line
            rel = l.substr(3);
            while (!rel.empty() && (rel.back() == '\n' || rel.back() == '\r')) rel.pop_back();
            break;
        }
    }
    std::fclose(f);
    uint64_t room = UINT64_MAX;
    for (std::string dir = rel; !dir.empty();) {
        // In a cgroup namespace, "/" can be a constrained cgroup mounted as the visible root.
        const std::string base = dir == "/" ? cgroup_root : cgroup_root + dir;
        uint64_t max = 0, current = 0;
        if (read_u64_file(base + "/memory.max", max) && read_u64_file(base + "/memory.current", current))
            room = std::min(room, max > current ? max - current : 0);
        const auto slash = dir.find_last_of('/');
        if (dir == "/" || slash == std::string::npos) break;
        dir.resize(slash == 0 ? 1 : slash);
    }
    return room;
}
#endif

// Memory the OS could give this process now, or 0 when it can't tell (then the caller keeps the old exclusive gate).
inline uint64_t available_memory_bytes() {
#ifdef _WIN32
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    return GlobalMemoryStatusEx(&status) ? uint64_t(status.ullAvailPhys) : 0;
#elif defined(__linux__)
    std::FILE* f = std::fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    unsigned long long kb = 0;
    bool found = false;
    while (std::fgets(line, sizeof(line), f))
        if (std::sscanf(line, "MemAvailable: %llu kB", &kb) == 1) { found = true; break; }
    std::fclose(f);
    if (!found) return 0;
    const uint64_t room = cgroup_memory_room();
    return std::max<uint64_t>(1, std::min(uint64_t(kb) * 1024, room));   // 0 would mean "unknown"
#else
    return 0;
#endif
}

// What large compiles may use of the free RAM and of the room under the commit limit: three quarters of the smaller. (On
// Windows an allocation that the limit has no room for fails, whatever RAM is free.)
inline uint64_t large_compile_share(uint64_t free_ram, uint64_t commit_room) {
    return std::min(free_ram, commit_room) / 4 * 3;
}

// The same of the memory that is free at this moment. 0 when it is unknown: then they compile one at a time.
inline uint64_t large_compile_budget_bytes() {
#ifdef _WIN32
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (!GlobalMemoryStatusEx(&status)) return 0;
    return large_compile_share(status.ullAvailPhys, status.ullAvailPageFile);
#else
    return large_compile_share(available_memory_bytes(), UINT64_MAX);
#endif
}

// LEONIDA_BIG_COMPILE_BUDGET_MB: a fixed budget in MiB instead of the share of the free memory (0: one at a time). Unset
// or empty, and text that is not a number: none. `valid` is false for the latter.
struct LargeCompileBudget {
    std::optional<uint64_t> fixed{};
    bool valid = true;
};

inline LargeCompileBudget large_compile_budget(const char* text) {
    if (!text || !*text) return {};
    uint64_t mib = 0;
    for (const char* c = text; *c; ++c) {
        if (*c < '0' || *c > '9') return {std::nullopt, false};
        mib = std::min<uint64_t>(mib * 10 + uint64_t(*c - '0'), UINT64_MAX >> 20);
    }
    return {mib << 20, true};
}

// Return memory freed by the driver's compiler to the OS (glibc keeps it in the arena of the thread that freed it).
inline void release_free_memory() {
#if defined(__GLIBC__)
    malloc_trim(0);
#endif
}

// Lets large modules in while their estimates fit the budget and fewer than max_inflight are in. One always gets in
// when none is, so a module bigger than the whole budget still compiles (alone among the large ones).
//
// The second constructor is the runtime's: the budget is read again whenever a place is given or returned, and the
// modules get in in the order they came, those a submission waits for (demand) before the others (speculative). While the
// first in line does not fit, the next behind it that does may pass, as long as all that have passed it would fit the
// budget together with it. So, with the budget as it is, it waits at most for those that were in when it became the first,
// however many come after it, and one larger than the budget is not passed. A speculative one gets in only while no demand
// waits and no submission has a Demand, and only within a quarter of the budget, so that a demand finds the rest free. An
// estimate counts in full until its module leaves, although the driver has allocated part of it by then and the budget is
// that much smaller: what is in has not taken all it is going to.
class LargeModuleGate {
public:
    enum class Work { demand, speculative };

    LargeModuleGate(uint64_t budget, unsigned max_inflight)
        : budget_(budget), max_inflight_(max_inflight ? max_inflight : 1) {}
    explicit LargeModuleGate(std::function<uint64_t()> budget)
        : budget_(0), max_inflight_(UINT32_MAX), read_budget_(std::move(budget)) {}

    bool admits(uint64_t estimate) const {
        std::lock_guard<std::mutex> g(mu_);
        return admits_locked(estimate, budget());
    }
    void enter(uint64_t estimate, Work work = Work::demand) {
        std::unique_lock<std::mutex> l(mu_);
        if (!read_budget_) {
            cv_.wait(l, [&] { return admits_locked(estimate, limit_for(work)); });
        } else {
            auto& queue = waiting_[work == Work::demand ? 0 : 1];
            const auto me = queue.insert(queue.end(), Waiter{estimate});
            cv_.wait(l, [&] {
                return (work == Work::demand || (waiting_[0].empty() && !demanded_)) && next_locked(work) == &*me;
            });
            if (me != queue.begin()) queue.front().passed += estimate;
            queue.erase(me);
        }
        used_ += estimate;
        ++inflight_;
        if (!waiting_[0].empty() || !waiting_[1].empty()) cv_.notify_all();   // the next in the queue may fit as well
    }
    void leave(uint64_t estimate) {
        {
            std::lock_guard<std::mutex> g(mu_);
            used_ -= estimate;
            --inflight_;
        }
        cv_.notify_all();
    }
    unsigned inflight() const {
        std::lock_guard<std::mutex> g(mu_);
        return inflight_;
    }
    size_t waiting(Work work) const {   // in order: the threads in enter(), with a budget that is read
        std::lock_guard<std::mutex> g(mu_);
        return waiting_[work == Work::demand ? 0 : 1].size();
    }
    uint64_t budget() const { return read_budget_ ? read_budget_() : budget_; }
    unsigned max_inflight() const { return max_inflight_; }

    // A submission that has large modules to build: until it ends, the places that its modules leave between one and the
    // next are not taken by speculative ones.
    class Demand {
    public:
        explicit Demand(LargeModuleGate& gate) : gate_(gate) {
            std::lock_guard<std::mutex> g(gate_.mu_);
            ++gate_.demanded_;
        }
        ~Demand() {
            {
                std::lock_guard<std::mutex> g(gate_.mu_);
                --gate_.demanded_;
            }
            gate_.cv_.notify_all();
        }
        Demand(const Demand&) = delete;
        Demand& operator=(const Demand&) = delete;

    private:
        LargeModuleGate& gate_;
    };

    // Holds a place from construction until destruction (or release()), then hands freed memory back.
    class Ticket {
    public:
        Ticket() = default;
        Ticket(LargeModuleGate& gate, uint64_t estimate, Work work = Work::demand) : gate_(&gate), estimate_(estimate) {
            gate.enter(estimate, work);
        }
        Ticket(Ticket&& o) noexcept : gate_(o.gate_), estimate_(o.estimate_) { o.gate_ = nullptr; }
        Ticket& operator=(Ticket&& o) noexcept {
            if (this != &o) { release(); gate_ = o.gate_; estimate_ = o.estimate_; o.gate_ = nullptr; }
            return *this;
        }
        Ticket(const Ticket&) = delete;
        Ticket& operator=(const Ticket&) = delete;
        ~Ticket() { release(); }
        void release() {
            if (!gate_) return;
            gate_->leave(estimate_);
            gate_ = nullptr;
            release_free_memory();
        }
    private:
        LargeModuleGate* gate_ = nullptr;
        uint64_t estimate_ = 0;
    };

private:
    struct Waiter {
        uint64_t estimate;
        uint64_t passed = 0;   // the estimates of those that have passed it while it was the first in line
    };

    uint64_t limit_for(Work work) const { return work == Work::demand ? budget() : budget() / 4; }
    bool admits_locked(uint64_t estimate, uint64_t limit) const {
        return inflight_ == 0 || (inflight_ < max_inflight_ && used_ + estimate <= limit);
    }
    // The one in line that gets in now, if any: the first, or else the first that fits and may pass it.
    const Waiter* next_locked(Work work) const {
        const auto& queue = waiting_[work == Work::demand ? 0 : 1];
        const uint64_t limit = limit_for(work);
        for (const Waiter& w : queue) {
            const Waiter& first = queue.front();
            if (&w != &first && first.passed + w.estimate + first.estimate > limit) continue;
            if (admits_locked(w.estimate, limit)) return &w;
        }
        return nullptr;
    }

    const uint64_t budget_;
    const unsigned max_inflight_;
    const std::function<uint64_t()> read_budget_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::list<Waiter> waiting_[2];   // demand, speculative: one for each thread in enter(), in the order they came
    uint64_t used_ = 0;
    unsigned inflight_ = 0;
    unsigned demanded_ = 0;   // the Demands
};

}  // namespace leonida
