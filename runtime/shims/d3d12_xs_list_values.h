#pragma once
// Console command-list memory writes and waits (docs/d3d12_xs_console_abi.md §5.5): the runtime half of list slots
// 71-74 (newer layout; 69-72 in the older one). Included by d3d12_xs_device.cpp only; the list and queue proxies
// there record and submit, this header keeps what has to outlive a call: the writes and waits in flight and the
// threads that settle them.
//
// What the console does. `WriteValueBottomOfPipeX`-style writes store 32 or 64 bits into title memory once the GPU
// work recorded before them has finished; a wait stops the queue's command processor until the 32 or 64 bits at an
// address compare true against a reference. The title chains jobs across its queues this way: a compute list waits
// at its start for a value a direct list writes as its first calls, and a direct list waits for values compute lists
// write near their end (measured on the retail title at the story load: `wait64(address, >=, tag)` on addresses
// other lists `write64(address, tag)`).
//
// What this runtime does. Neither has a D3D12 or Vulkan command. The proxy records them per list part (a list is
// cut at a wait recorded after work, d3d12_xs_device.cpp "list cut"), and the queue proxy submits part by part
// (writes recorded before a part's first wait land before it):
// - A part's writes land after the part has run: the queue proxy signals the queue's `written` fence behind the
//   lists of a group and the pump, a thread of its own, stores the values with the CPU once that fence is reached
//   (kernel_writes::store: one locked access for an aligned value, as any host write into title memory, so the
//   slot-133 shadows see it; an arena offset and a page xmem keeps PAGE_NOACCESS land through src/value_write_sink.h,
//   see land()). Later than the console, never earlier: a write in the middle of a part lands when the
//   whole part is done. Writes land in the order they were submitted, across queues too: the fences are read newest
//   first, so a write whose work waited for an older one (a relay, a title fence) is never seen done before it.
//   A timestamp write (write64 flags 1 and 2) lands the GPU's timestamp for its place in the list (Timestamps).
// - A wait that is met when its part is submitted costs nothing. One that a write still in flight will meet is
//   turned into a GPU wait on that write's fence (relay: the queue waits on the writer's `written` value, so the
//   GPU orders the two queues without a CPU round trip). Any other wait makes the queue wait on its `gate` fence,
//   which is signalled from the CPU once the memory compares true or the wait has timed out. The Vulkan layer
//   holds a queue's later work, Signals included, behind an unsatisfied fence wait without blocking the caller
//   (src/queue_holds.h), so the submitting thread never waits here.
// - The thread that signals a gate runs the work the layer held behind it (QueueHolds: the held lists' submission,
//   the shadow refresh, a deferred Present). That is the releaser, a second thread: the pump only decides, so
//   writes keep landing and other gates keep timing out while held work runs.
// Bounded: a gate is signalled after LEONIDA_VALUE_WAIT_TIMEOUT_MS (default 200 ms) of idle wait time, counted from
// when the gates before it on its queue opened and paused (up to a finite cap scaled with the gate's own budget) while
// the releaser is running held work, while an acyclic matching producer write is in flight, or while unrelated
// unblocked GPU work is running (src/list_value_wait_budget.h); logged once per address. Short timeouts (4 ms) go to an
// address that timed out three times in a row, and, once four gates in a row timed out (the title waits for values
// no modelled write stores, a new address each frame), to every wait whose address was never met: a frame then pays
// a few milliseconds, not the full timeout, per such wait. A wait cut short is watched for the rest of the full
// timeout: met in that time, its address keeps the full timeout from then on. A gate cut short that is met within
// its short timeout gives every wait the full timeout again.
// LEONIDA_LIST_VALUES=0 drops both methods (the behaviour before this file).

#include <windows.h>
#include <d3d12.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <new>
#include <span>
#include <utility>
#include <vector>

#include "d3d12_xs_device.h"          // console_value_met, ListValueStats
#include "../src/list_value_wait_budget.h"  // leonida::list_value_wait: work-aware timeout clock & producer graph
#include "../src/title_page_watch.h"  // leonida::kernel_writes::store
#include "../src/value_write_sink.h"  // leonida::value_sink: the arena and xmem's PAGE_NOACCESS pages

namespace rt::d3d12_xs::list_values {

struct Wait {
    std::uint64_t address;
    std::uint64_t reference;
    std::uint32_t comparison;
    bool wide;  // 64 bits; else the low 32 bits of memory and reference are compared
};

enum class WriteKind : std::uint8_t { value, gpu_timestamp, system_timestamp };

// The GPU timestamps of write64 flags 1 and 2: one timestamp query heap and one readback buffer per device, used as
// a ring of slots. A timestamp write takes a slot where the title records it: EndQuery(TIMESTAMP) on the real list
// (the Vulkan layer writes it at the bottom of the pipe, after the work recorded before it) and a ResolveQueryData of
// that one slot into the readback right after. The pump reads the slot once the part's fence has passed. The ticks
// are those of the queue's GetTimestampFrequency, which is what the queue proxy answers the title's frequency query
// with (newer layout slot 18).
class Timestamps {
public:
    static constexpr std::uint32_t kSlots = 4096;  // a few per frame: a slot comes back after hundreds of frames

    Timestamps(const Timestamps&) = delete;
    Timestamps& operator=(const Timestamps&) = delete;
    ~Timestamps() {
        if (readback_) {
            if (values_) readback_->Unmap(0, nullptr);
            readback_->Release();
        }
        if (heap_) heap_->Release();
    }

    // Null when the device cannot make the heap or the buffer: timestamps then come from the CPU clock at landing.
    static std::shared_ptr<Timestamps> create(ID3D12Device* device) {
        if (!device) return nullptr;
        auto stamps = std::shared_ptr<Timestamps>(new (std::nothrow) Timestamps);
        if (!stamps) return nullptr;
        D3D12_QUERY_HEAP_DESC heap{};
        heap.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        heap.Count = kSlots;
        if (FAILED(device->CreateQueryHeap(&heap, __uuidof(ID3D12QueryHeap),
                                           reinterpret_cast<void**>(&stamps->heap_))) ||
            !stamps->heap_)
            return nullptr;
        D3D12_HEAP_PROPERTIES readback{};
        readback.Type = D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC buffer{};
        buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        buffer.Width = kSlots * sizeof(std::uint64_t);
        buffer.Height = 1;
        buffer.DepthOrArraySize = 1;
        buffer.MipLevels = 1;
        buffer.SampleDesc.Count = 1;
        buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        void* mapped = nullptr;
        if (FAILED(device->CreateCommittedResource(&readback, D3D12_HEAP_FLAG_NONE, &buffer,
                                                   D3D12_RESOURCE_STATE_COPY_DEST, nullptr, __uuidof(ID3D12Resource),
                                                   reinterpret_cast<void**>(&stamps->readback_))) ||
            !stamps->readback_ || FAILED(stamps->readback_->Map(0, nullptr, &mapped)) || !mapped)
            return nullptr;
        stamps->values_ = static_cast<volatile std::uint64_t*>(mapped);
        return stamps;
    }

    // A slot for a timestamp recorded now, its readback value cleared (a slot the GPU has not written reads 0).
    std::uint32_t take() {
        const std::uint32_t slot = next_.fetch_add(1, std::memory_order_relaxed) % kSlots;
        values_[slot] = 0;
        return slot;
    }

    ID3D12QueryHeap* heap() const { return heap_; }
    ID3D12Resource* readback() const { return readback_; }
    std::uint64_t read(std::uint32_t slot) const { return slot < kSlots ? values_[slot] : 0; }

private:
    Timestamps() = default;
    ID3D12QueryHeap* heap_ = nullptr;
    ID3D12Resource* readback_ = nullptr;
    volatile std::uint64_t* values_ = nullptr;
    std::atomic<std::uint32_t> next_{0};
};

struct Write {
    std::uint64_t address;
    std::uint64_t value;
    std::uint8_t bytes;  // 4 or 8
    WriteKind kind;
    // A timestamp the GPU takes at the write's place in its list (null: the CPU clock at landing; a write that lands
    // before its part has run drops it, see ValueSubmission::part).
    std::shared_ptr<const Timestamps> stamps = nullptr;
    std::uint32_t slot = 0;
};

struct Counters {
    std::atomic<std::uint64_t> writes_recorded{0}, writes_landed{0}, writes_failed{0};
    std::atomic<std::uint64_t> waits_recorded{0}, waits_met_at_submit{0}, waits_relayed{0};
    std::atomic<std::uint64_t> gates_held{0}, gates_met{0}, gates_timed_out{0};
    std::atomic<std::uint64_t> cuts{0}, cuts_deferred{0}, cut_failures{0};
    // Gates given the short timeout, times every wait was demoted to it, short waits met in the rest of the full
    // timeout (late), writes and waits on bundles (not modelled: a bundle is never cut).
    std::atomic<std::uint64_t> gates_short{0}, demotions{0}, late_met{0}, bundle_ops{0};
    // The releaser's gate signals and the time they took (the held work they ran), the pump's longest pass.
    std::atomic<std::uint64_t> releases{0}, release_us{0}, release_us_max{0}, pass_us_max{0}, release_stalls{0};
    // Queue slot 31 (CopyPageMappingsBatchX): applied at the call, deferred behind held work, deferred ones that
    // failed, and the longest the call waited for the queue's earlier work.
    std::atomic<std::uint64_t> remaps_now{0}, remaps_deferred{0}, remaps_failed{0}, remap_wait_us_max{0};
    // Timestamp writes (write64 flags 1 and 2) landed with the GPU's timestamp, and with the CPU clock at landing.
    std::atomic<std::uint64_t> stamps_gpu{0}, stamps_cpu{0};
    // List slot 71/72 writes handed to the GPU-label path (LEONIDA_GPU_LABELS) of a list these values do not track.
    std::atomic<std::uint64_t> labels_handed{0};
    // Writes landed through the fallbacks (src/value_write_sink.h): in the GPU address arena (an arena offset is never
    // stored to as a host address), and through xmem's own mapping of a page the title's view keeps PAGE_NOACCESS.
    // Both are also counted in writes_landed.
    std::atomic<std::uint64_t> writes_arena{0}, writes_title_pages{0};
};

inline Counters& counters() {
    static Counters* c = new Counters;  // immortal: the threads may still count during teardown
    return *c;
}

inline void bump(std::atomic<std::uint64_t>& counter, std::uint64_t by = 1) {
    counter.fetch_add(by, std::memory_order_relaxed);
}

inline void raise_to(std::atomic<std::uint64_t>& counter, std::uint64_t value) {
    std::uint64_t seen = counter.load(std::memory_order_relaxed);
    while (seen < value && !counter.compare_exchange_weak(seen, value, std::memory_order_relaxed)) {
    }
}

// LEONIDA_LIST_VALUES=0 turns both methods back into the no-ops they were. Read once; a test can set it (lists made
// afterwards follow it).
inline std::atomic<int>& enabled_state() {
    static std::atomic<int> state{-1};
    return state;
}
inline bool enabled() {
    int state = enabled_state().load(std::memory_order_relaxed);
    if (state < 0) {
        char value[8] = {};
        std::size_t length = 0;
        state = getenv_s(&length, value, sizeof(value), "LEONIDA_LIST_VALUES") != 0 || std::strcmp(value, "0") != 0;
        enabled_state().store(state, std::memory_order_relaxed);
    }
    return state != 0;
}
inline void set_enabled(bool on) { enabled_state().store(on ? 1 : 0, std::memory_order_relaxed); }

// The value a wait looks at, read through the kernel: memory that is not there is reported instead of faulting (a
// fault would run every vectored handler of the process, the D3D12 layer's write-fault handler among them, from a
// thread that may hold the pump's lock). The kernel's copy is not one access, so a value read while another thread
// stores it could mix the two: it is read until two reads agree.
inline bool read_value(std::uint64_t address, bool wide, std::uint64_t& value) {
    value = 0;
    if (!address) return false;
    const SIZE_T bytes = wide ? 8 : 4;
    const auto* at = reinterpret_cast<const void*>(static_cast<std::uintptr_t>(address));
    std::uint64_t last = 0;
    for (int attempt = 0; attempt < 4; ++attempt) {
        std::uint64_t now = 0;
        SIZE_T got = 0;
        if (!ReadProcessMemory(GetCurrentProcess(), at, &now, bytes, &got) || got != bytes) return false;
        if (attempt && now == last) break;
        last = now;
    }
    value = last;
    return true;
}

// Memory that cannot be read meets the wait: nothing could ever change it for the GPU either.
inline bool met_now(const Wait& wait) {
    std::uint64_t value = 0;
    if (!read_value(wait.address, wait.wide, value)) return true;
    return console_value_met(wait.comparison, value, wait.reference, wait.wide);
}

// One per queue proxy, made at its first submission of a list part with writes or an unmet wait. The fences are the
// real device's; their values only rise. `gate_value` and `written_value` are advanced by the submitting thread
// under the queue proxy's order lock; the threads only read the values they were handed.
struct QueueValues {
    ID3D12Fence* gate = nullptr;     // the queue waits on it at an unmet wait; the releaser signals it
    ID3D12Fence* written = nullptr;  // the queue signals it behind a group of parts that write
    std::uint64_t gate_value = 0;
    std::uint64_t written_value = 0;
    std::uint64_t last_gpu_stamp = 0;  // the pump's only: the last GPU timestamp landed for this queue

    QueueValues() = default;
    QueueValues(const QueueValues&) = delete;
    QueueValues& operator=(const QueueValues&) = delete;
    ~QueueValues() {
        if (gate) gate->Release();
        if (written) written->Release();
        if (queue_) queue_->Release();
    }

    // Null when a fence cannot be made: the caller then submits as before (waits run on, writes are dropped).
    static std::shared_ptr<QueueValues> create(ID3D12Device* device, ID3D12CommandQueue* queue) {
        if (!device) return nullptr;
        auto values = std::shared_ptr<QueueValues>(new (std::nothrow) QueueValues);
        if (!values) return nullptr;
        if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence),
                                       reinterpret_cast<void**>(&values->gate))) ||
            !values->gate ||
            FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence),
                                       reinterpret_cast<void**>(&values->written))) ||
            !values->written)
            return nullptr;
        LARGE_INTEGER frequency{};
        QueryPerformanceFrequency(&frequency);
        values->qpc_frequency_ = static_cast<std::uint64_t>(frequency.QuadPart);
        UINT64 gpu_frequency = 0;
        if (queue && SUCCEEDED(queue->GetTimestampFrequency(&gpu_frequency)) && gpu_frequency) {
            values->gpu_frequency_ = gpu_frequency;
            queue->AddRef();
            values->queue_ = queue;
            const auto started = std::chrono::steady_clock::now();
            values->calibrate();
            // A queue without calibrated timestamps answers by stalling itself (the layer's fallback): such a queue
            // is calibrated once, here, and not again from the pump.
            values->recalibrate_ = std::chrono::steady_clock::now() - started < std::chrono::milliseconds(2);
        }
        return values;
    }

    // The GPU's timestamp (in the queue's GetTimestampFrequency ticks) for a QueryPerformanceCounter value, from the
    // queue's clock calibration: gpu0 + (qpc - cpu0) * gpu_frequency / qpc_frequency. Without one, the QPC value.
    std::uint64_t gpu_ticks_at(std::uint64_t qpc) {
        const Calibration c = calibration();
        if (!c.cpu0 || !gpu_frequency_ || !qpc_frequency_) return qpc;
        return qpc >= c.cpu0 ? c.gpu0 + scale(qpc - c.cpu0, gpu_frequency_, qpc_frequency_)
                             : c.gpu0 - (std::min)(c.gpu0, scale(c.cpu0 - qpc, gpu_frequency_, qpc_frequency_));
    }

    // The QueryPerformanceCounter value at which the GPU's clock read `ticks`, the other way round. 0 without a
    // calibration.
    std::uint64_t qpc_at(std::uint64_t ticks) {
        const Calibration c = calibration();
        if (!c.cpu0 || !gpu_frequency_ || !qpc_frequency_) return 0;
        return ticks >= c.gpu0 ? c.cpu0 + scale(ticks - c.gpu0, qpc_frequency_, gpu_frequency_)
                               : c.cpu0 - (std::min)(c.cpu0, scale(c.gpu0 - ticks, qpc_frequency_, gpu_frequency_));
    }

    std::uint64_t gpu_frequency() const { return gpu_frequency_; }

private:
    struct Calibration {
        std::uint64_t gpu0 = 0, cpu0 = 0;
    };

    // value * to / from without overflowing 64 bits for any uptime.
    static std::uint64_t scale(std::uint64_t value, std::uint64_t to, std::uint64_t from) {
        const std::uint64_t whole = value / from, part = value % from;
        return whole * to + part * to / from;
    }

    void calibrate() {
        UINT64 gpu = 0, cpu = 0;
        if (!queue_ || FAILED(queue_->GetClockCalibration(&gpu, &cpu)) || !cpu) return;
        std::lock_guard lock(clock_mutex_);
        calibration_ = {gpu, cpu};
        calibrated_at_ = std::chrono::steady_clock::now();
    }

    // The two clocks drift apart by some parts per million: the pump calibrates again once a second.
    Calibration calibration() {
        if (recalibrate_ && std::chrono::steady_clock::now() - calibrated_at_.load() > std::chrono::seconds(1))
            calibrate();
        std::lock_guard lock(clock_mutex_);
        return calibration_;
    }

    ID3D12CommandQueue* queue_ = nullptr;  // the real queue (one reference), for GetClockCalibration
    std::uint64_t gpu_frequency_ = 0, qpc_frequency_ = 0;
    bool recalibrate_ = false;
    std::mutex clock_mutex_;
    Calibration calibration_;
    std::atomic<std::chrono::steady_clock::time_point> calibrated_at_{};
};

// The two threads that land writes and open gates. One pair per process, made on first use; stop() ends them before
// the process exits, so neither is cut off inside the layer (holding QueueHolds' pump lock, say).
// - The pump lands writes whose fence was reached and decides which gates open (met, timed out, or an action's
//   turn). It never signals a gate: it hands them, in each queue's order, to
// - the releaser, which signals them (after running the action of an action gate). The Signal runs the work the
//   layer held behind the gate, which can take a while; the pump goes on meanwhile.
class Pump {
public:
    static Pump& get() {
        static Pump* pump = new Pump;  // immortal: a late submission may still reach it during teardown
        return *pump;
    }

    // Writes that land once `queue->written` reaches `value`. The caller signalled that value on the queue and
    // asked the fence to set wake() on it (SetEventOnCompletion), so the pump needs no polling for writes.
    void add_writes(std::shared_ptr<QueueValues> queue, std::uint64_t value, std::vector<Write> writes) {
        if (writes.empty() || exiting_.load(std::memory_order_relaxed)) return;  // stop(): see there
        {
            std::lock_guard lock(mutex_);
            const std::uint64_t after_gate = max_pending_gate_locked(queue.get());
            writes_.push_back({std::move(queue), value, std::move(writes), after_gate});
        }
        start_or_settle();
    }

    // The queue waits on `queue->gate` for `value`: signalled once every wait is met, or the gate timed out. Its
    // time starts when the gates before it on its queue have opened (start_gate).
    void add_gate(std::shared_ptr<QueueValues> queue, std::uint64_t value, std::vector<Wait> waits) {
        if (exiting_.load(std::memory_order_relaxed)) return;  // stop(): never signalled, see there
        {
            std::lock_guard lock(mutex_);
            Gate gate;
            gate.queue = std::move(queue);
            gate.value = value;
            gate.waits = std::move(waits);
            gates_.push_back(std::move(gate));
        }
        start_or_settle();
    }

    // The queue waits on `queue->gate` for `value`, which the releaser signals after running `action`, once
    // `queue->written` reaches `after` (the caller signalled it on the queue and set wake() on it): something the
    // CPU does at a point of the queue's timeline, after the work before it and before the work after it. No
    // timeout. False when the threads do not run (the caller then does it the synchronous way).
    bool add_action(std::shared_ptr<QueueValues> queue, std::uint64_t value, std::uint64_t after,
                    std::function<void()> action) {
        if (!start()) return false;
        {
            std::lock_guard lock(mutex_);
            Gate gate;
            gate.queue = std::move(queue);
            gate.value = value;
            gate.started = true;
            gate.since = Clock::now();
            gate.last_tick = gate.since;
            gate.after = after;
            gate.action = std::move(action);
            gates_.push_back(std::move(gate));
        }
        SetEvent(wake_);
        return true;
    }

    // A write still in flight that will meet `wait` when it lands: the fence and value whose completion lands it
    // (the earliest such write). False when there is none.
    bool relay(const Wait& wait, ID3D12Fence*& fence, std::uint64_t& value) {
        std::lock_guard lock(mutex_);
        const std::uint8_t need = wait.wide ? 8 : 4;
        for (const WriteGroup& group : writes_)
            for (const Write& write : group.writes)
                if (write.kind == WriteKind::value && write.address == wait.address && write.bytes >= need &&
                    console_value_met(wait.comparison, write.value, wait.reference, wait.wide)) {
                    fence = group.queue->written;
                    value = group.value;
                    return true;
                }
        return false;
    }

    HANDLE wake() const { return wake_; }

    // stop() ran: nothing new is landed or opened (the process is ending).
    bool stopped() const { return exiting_.load(std::memory_order_relaxed); }

    // Starts the threads if they are not running yet; true while they run.
    bool start() {
        std::call_once(start_, [this] {
            if (wake_) {
                thread_ = CreateThread(nullptr, 0, &Pump::pump_main, this, 0, nullptr);
                if (thread_) releaser_ = CreateThread(nullptr, 0, &Pump::releaser_main, this, 0, nullptr);
                if (thread_ && !releaser_) {  // the pump alone would wait for a releaser forever
                    stop_.store(true, std::memory_order_relaxed);
                    SetEvent(wake_);
                    WaitForSingleObject(thread_, INFINITE);
                    CloseHandle(thread_);
                    thread_ = nullptr;
                }
            }
            if (!releaser_)
                std::fprintf(stderr, "[rt] d3d12_xs: no threads for list memory writes and waits (error %lu): "
                                     "writes land at submission and waits run on\n", GetLastError());
            else
                std::fprintf(stderr, "[rt] d3d12_xs: list memory writes land after their part ran, waits hold their "
                                     "queue (timeout %u ms)\n", timeout_ms_.load(std::memory_order_relaxed));
        });
        return releaser_ && !stop_.load(std::memory_order_relaxed);
    }

    void set_timeout_ms(unsigned ms) { timeout_ms_.store(ms ? ms : 1, std::memory_order_relaxed); }

    // Test helper: true once nothing is in flight, polling for up to `ms`.
    bool idle(unsigned ms) {
        const auto until = Clock::now() + std::chrono::milliseconds(ms);
        for (;;) {
            {
                std::lock_guard lock(mutex_);
                std::lock_guard release(release_mutex_);
                if (writes_.empty() && gates_.empty() && !busy_ && releases_.empty() && !releasing_) return true;
            }
            if (Clock::now() >= until) return false;
            if (wake_) SetEvent(wake_);
            Sleep(1);
        }
    }

    // Before the process exits: both threads finish what they are doing (a held submission the releaser is running,
    // say) and end, waiting up to `ms` for them. Gates not signalled yet stay shut and their actions do not run, and
    // writes and gates added afterwards are dropped: signalling any of them from another thread could open a gate
    // whose remap was skipped, or signal a gate value above one still to come. The queues' held work is not run at
    // exit.
    void stop(unsigned ms) {
        if (exiting_.exchange(true, std::memory_order_relaxed)) return;
        stop_.store(true, std::memory_order_relaxed);
        if (wake_) SetEvent(wake_);
        {
            std::lock_guard release(release_mutex_);
        }
        release_ready_.notify_all();
        HANDLE threads[2];
        DWORD count = 0;
        if (thread_) threads[count++] = thread_;
        if (releaser_) threads[count++] = releaser_;
        if (count && WaitForMultipleObjects(count, threads, TRUE, ms) == WAIT_TIMEOUT)
            std::fprintf(stderr, "[rt] d3d12_xs: list value threads still busy after %u ms at exit\n", ms);
    }

private:
    using Clock = std::chrono::steady_clock;
    static constexpr unsigned kDemoteAfter = 3;        // timeouts in a row before an address gets the short timeout
    static constexpr unsigned kDemoteAllAfter = 4;     // gates timed out in a row before every wait gets it
    static constexpr unsigned kDemotedTimeoutMs = 4;
    static constexpr unsigned kPollMs = 1;             // while a gate waits for memory nobody tells us about
    static constexpr std::size_t kMaxStreaks = 256;    // addresses whose timeouts are counted (least recently used go)
    static constexpr std::size_t kMaxLate = 64;        // waits cut short watched for the rest of the full timeout
    static constexpr std::uint64_t kStallMs = 1000;    // a gate Signal this long is reported (watch_releaser)

    struct WriteGroup {
        std::shared_ptr<QueueValues> queue;
        std::uint64_t value;
        std::vector<Write> writes;
        std::uint64_t after_gate = 0;  // highest pending gate value on `queue` ahead of this write group
    };
    struct Gate {
        std::shared_ptr<QueueValues> queue;
        std::uint64_t value = 0;
        std::vector<Wait> waits;
        leonida::list_value_wait::GateTimeoutState timeout{};  // work-aware timeout clock (set by start_gate)
        Clock::time_point last_tick{};
        Clock::time_point since{};
        bool started = false;          // the gates before it on its queue have opened: its time runs
        bool cut_short = false;        // given the short timeout
        std::uint64_t after = 0;       // action gates: the queue's `written` value the action runs behind
        std::function<void()> action;  // null: a memory wait
    };
    struct Release {
        std::shared_ptr<QueueValues> queue;
        std::uint64_t value;
        std::function<void()> action;
    };
    struct Streak {
        std::uint64_t address;
        unsigned timeouts;   // in a row
        bool logged;
        bool met;            // a wait for it was met once (in time or late): it keeps the full timeout when every
                             // wait is demoted
        std::uint64_t used;  // use_clock_ at its last use
    };
    struct Late {
        Wait wait;
        Clock::time_point until;
    };

    Pump() {
        wake_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        timer_ = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
        unsigned timeout = 200;
        char text[16] = {};
        std::size_t length = 0;
        if (getenv_s(&length, text, sizeof(text), "LEONIDA_VALUE_WAIT_TIMEOUT_MS") == 0 && text[0]) {
            const unsigned long parsed = std::strtoul(text, nullptr, 10);
            if (parsed > 0 && parsed < 600000) timeout = static_cast<unsigned>(parsed);
        }
        timeout_ms_.store(timeout, std::memory_order_relaxed);
    }

    // Without the threads (none could be made, or stop() ran) nothing could land later: what is queued is settled
    // right here, writes at once and gates opened, which is the order before this file, not a hang.
    void start_or_settle() {
        if (start()) {
            SetEvent(wake_);
            return;
        }
        if (exiting_.load(std::memory_order_relaxed)) return;
        // One thread at a time, so that the gates are signalled in order.
        std::lock_guard settle(settle_mutex_);
        pass(true);
    }

    static DWORD WINAPI pump_main(LPVOID self) {
        static_cast<Pump*>(self)->run();
        return 0;
    }
    static DWORD WINAPI releaser_main(LPVOID self) {
        static_cast<Pump*>(self)->release_loop();
        return 0;
    }

    void run() {
        while (!stop_.load(std::memory_order_relaxed)) {
            bool polling = false, writes = false;
            {
                std::lock_guard lock(mutex_);
                polling = !gates_.empty() || !late_.empty();
                writes = !writes_.empty();
            }
            if (polling && timer_) {
                LARGE_INTEGER due{};
                due.QuadPart = -static_cast<LONGLONG>(kPollMs) * 10000;  // relative, 100 ns units
                SetWaitableTimer(timer_, &due, 0, nullptr, nullptr, FALSE);
                const HANDLE handles[2] = {wake_, timer_};
                WaitForMultipleObjects(2, handles, FALSE, INFINITE);
            } else {
                // Writes wake us through SetEventOnCompletion; the timeout is only a safety net (and, while the
                // releaser is inside a Signal, how often watch_releaser looks at it).
                const bool releasing = signal_since_ms_.load(std::memory_order_relaxed) != 0;
                WaitForSingleObject(wake_, polling ? kPollMs : writes ? 50 : releasing ? 500 : INFINITE);
            }
            if (stop_.load(std::memory_order_relaxed)) break;
            pass(false);
            watch_releaser();
            summarize();
        }
    }

    // A gate Signal that does not return holds every later gate of every queue (the releaser signals them in turn):
    // said once per stall, with the thread, for a debugger.
    void watch_releaser() {
        const std::uint64_t since = signal_since_ms_.load(std::memory_order_relaxed);
        if (!since) {
            stall_logged_ = false;
            return;
        }
        // `since` is the start tick with its low bit set (never 0 while inside), so it can be one past now.
        const std::uint64_t now = GetTickCount64();
        const std::uint64_t ms = now > since ? now - since : 0;
        if (ms < kStallMs || stall_logged_) return;
        stall_logged_ = true;
        bump(counters().release_stalls);
        static std::atomic<unsigned> logged{0};
        if (logged.fetch_add(1, std::memory_order_relaxed) < 8)
            std::fprintf(stderr, "[rt] d3d12_xs: the releaser (thread %lu) has been inside one gate Signal (value "
                                 "%llu) for %llu ms: the work the layer held behind the gate has not run\n",
                         releaser_tid_.load(std::memory_order_relaxed),
                         static_cast<unsigned long long>(signal_value_.load(std::memory_order_relaxed)),
                         static_cast<unsigned long long>(ms));
    }

    // Signals the gates the pump opened, in the order it opened them (each queue's gate values rise in that order).
    void release_loop() {
        releaser_tid_.store(GetCurrentThreadId(), std::memory_order_relaxed);
        for (;;) {
            std::deque<Release> batch;
            {
                std::unique_lock lock(release_mutex_);
                release_ready_.wait(lock, [this] {
                    return !releases_.empty() || stop_.load(std::memory_order_relaxed);
                });
                if (stop_.load(std::memory_order_relaxed)) return;
                batch.swap(releases_);
                releasing_ = true;
            }
            SetEvent(wake_);  // the pump may be in an untimed wait: it watches the Signals from here on
            struct PendingReleaseGuard {
                std::atomic<std::size_t>* count;
                std::size_t remaining;
                ~PendingReleaseGuard() {
                    if (remaining) count->fetch_sub(remaining, std::memory_order_release);
                }
            } pending_guard{&pending_releases_, batch.size()};
            for (Release& release : batch) {
                if (stop_.load(std::memory_order_relaxed)) break;
                signal_value_.store(release.value, std::memory_order_relaxed);
                signal_since_ms_.store(GetTickCount64() | 1, std::memory_order_relaxed);  // never 0 while inside
                signal(release);
                signal_since_ms_.store(0, std::memory_order_relaxed);
                pending_releases_.fetch_sub(1, std::memory_order_release);
                --pending_guard.remaining;
            }
            std::lock_guard lock(release_mutex_);
            releasing_ = false;
        }
    }

    // The action (if any), then the gate's Signal, which runs whatever the layer held behind it.
    static void signal(const Release& release) {
        const auto started = Clock::now();
        if (release.action) release.action();
        release.queue->gate->Signal(release.value);
        const auto us = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - started).count());
        Counters& c = counters();
        bump(c.releases);
        bump(c.release_us, us);
        raise_to(c.release_us_max, us);
    }

    Streak* find_streak(std::uint64_t address) {
        for (Streak& streak : streaks_)
            if (streak.address == address) {
                streak.used = ++use_clock_;
                return &streak;
            }
        return nullptr;
    }

    Streak* add_streak(std::uint64_t address) {
        if (streaks_.size() < kMaxStreaks) {
            streaks_.push_back({address, 0, false, false, ++use_clock_});
            return &streaks_.back();
        }
        Streak* oldest = &*std::min_element(streaks_.begin(), streaks_.end(),
                                            [](const Streak& a, const Streak& b) { return a.used < b.used; });
        *oldest = {address, 0, false, false, ++use_clock_};
        return oldest;
    }

    // Write groups whose fence was reached, taken off the list in submission order (per queue, up to the first one
    // not reached). The fences are read newest group first, a fence again for an older group its last reading does
    // not cover: if a group's work waited for an older group's (on another queue), the older group's reading comes
    // later than the newer one's, so it cannot read as not done while the newer one reads as done. Called with
    // mutex_ held.
    void take_landed(bool force, std::vector<WriteGroup>& landing) {
        const std::size_t count = writes_.size();
        if (!count) return;
        std::vector<char> reached(count, force ? 1 : 0);
        if (!force) {
            struct Reading {
                const QueueValues* queue;
                std::uint64_t completed;
            };
            std::vector<Reading> readings;
            for (std::size_t i = count; i-- > 0;) {
                const WriteGroup& group = writes_[i];
                auto it = std::find_if(readings.begin(), readings.end(),
                                       [&](const Reading& r) { return r.queue == group.queue.get(); });
                if (it == readings.end()) {
                    readings.push_back({group.queue.get(), group.queue->written->GetCompletedValue()});
                    it = readings.end() - 1;
                } else if (it->completed < group.value) {
                    it->completed = group.queue->written->GetCompletedValue();
                }
                reached[i] = it->completed >= group.value;
            }
        }
        std::deque<WriteGroup> rest;
        std::vector<const QueueValues*> blocked;
        for (std::size_t i = 0; i < count; ++i) {
            WriteGroup& group = writes_[i];
            const QueueValues* queue = group.queue.get();
            const bool queue_blocked = std::find(blocked.begin(), blocked.end(), queue) != blocked.end();
            if (reached[i] && !queue_blocked) {
                landing.push_back(std::move(group));
            } else {
                if (!queue_blocked) blocked.push_back(queue);
                rest.push_back(std::move(group));
            }
        }
        writes_.swap(rest);
    }

    // Highest pending gate value currently in `gates_` for `queue` (0 if none). Called with mutex_ held.
    std::uint64_t max_pending_gate_locked(const QueueValues* queue) const {
        std::uint64_t highest = 0;
        for (const Gate& g : gates_)
            if (g.queue.get() == queue && g.value > highest) highest = g.value;
        return highest;
    }

    // The gates whose waits are met, whose idle time is up or whose action's turn has come, taken off the list in each
    // queue's order (a queue's gate values rise in submission order, so a later one must not be signalled first: it
    // would open the earlier one too). A memory gate's timeout clock pauses while held work is being released, while
    // an acyclic matching producer write is in flight, or (up to a cap) while unrelated unblocked GPU work is running
    // (src/list_value_wait_budget.h). `force` (no threads): every gate, now. Called with mutex_ held.
    void take_open(bool force, bool landed_any, std::vector<Release>& opening) {
        if (gates_.empty()) return;
        const auto now = Clock::now();
        namespace lvw = leonida::list_value_wait;

        // Build the lightweight producer/gate dependency view lazily only if a started memory gate is still unmet.
        bool graph_built = false;
        std::vector<lvw::PendingWrite> flat_writes;
        std::vector<lvw::PendingWriteGroup> view_writes;
        std::vector<lvw::WaitPredicate> flat_waits;
        std::vector<lvw::PendingGate> view_gates;
        std::vector<std::pair<std::size_t, std::size_t>> gate_wait_ranges;  // (offset, count) into flat_waits
        const auto ensure_graph = [&] {
            if (graph_built) return;
            graph_built = true;
            std::size_t total_writes = 0;
            for (const WriteGroup& wg : writes_) total_writes += wg.writes.size();
            flat_writes.reserve(total_writes);
            view_writes.reserve(writes_.size());
            for (const WriteGroup& wg : writes_) {
                const std::size_t start = flat_writes.size();
                for (const Write& w : wg.writes)
                    flat_writes.push_back({w.address, w.value, w.bytes, w.kind != WriteKind::value});
                view_writes.push_back({wg.queue.get(), wg.after_gate,
                                       std::span<const lvw::PendingWrite>(flat_writes).subspan(start, wg.writes.size())});
            }
            std::size_t total_waits = 0;
            for (const Gate& g : gates_)
                if (!g.action) total_waits += g.waits.size();
            flat_waits.reserve(total_waits);
            gate_wait_ranges.reserve(gates_.size());
            for (const Gate& g : gates_) {
                const std::size_t start = flat_waits.size();
                if (!g.action) {
                    for (const Wait& w : g.waits)
                        if (!met_now(w)) flat_waits.push_back({w.address, w.reference, w.comparison, w.wide});
                }
                gate_wait_ranges.emplace_back(start, flat_waits.size() - start);
            }
            view_gates.reserve(gates_.size());
            for (std::size_t i = 0; i < gates_.size(); ++i) {
                const auto [off, cnt] = gate_wait_ranges[i];
                view_gates.push_back({gates_[i].queue.get(), gates_[i].value, static_cast<bool>(gates_[i].action),
                                      std::span<const lvw::WaitPredicate>(flat_waits).subspan(off, cnt)});
            }
        };

        std::vector<const QueueValues*> blocked;
        std::vector<bool> opened(gates_.size(), false);
        for (std::size_t idx = 0; idx < gates_.size(); ++idx) {
            Gate& gate = gates_[idx];
            const QueueValues* queue = gate.queue.get();
            if (std::find(blocked.begin(), blocked.end(), queue) != blocked.end()) continue;
            bool open = force;
            if (gate.action) {
                open = open || queue->written->GetCompletedValue() >= gate.after;
            } else {
                if (!gate.started) start_gate(gate, now);
                bool met = true;
                for (const Wait& wait : gate.waits) met = met && met_now(wait);
                bool timed_out = false;
                if (!open && !met) {
                    const std::uint64_t dt_us =
                        now > gate.last_tick
                            ? static_cast<std::uint64_t>(
                                  std::chrono::duration_cast<std::chrono::microseconds>(now - gate.last_tick).count())
                            : 0u;
                    gate.last_tick = now;
                    const bool releaser_busy =
                        landed_any || !opening.empty() || pending_releases_.load(std::memory_order_acquire) != 0;
                    lvw::PauseReason reason = lvw::PauseReason::releaser_busy;
                    if (!releaser_busy) {
                        ensure_graph();
                        reason = lvw::pause_reason(false, view_writes, view_gates, idx, console_value_met);
                    }
                    timed_out = gate.timeout.advance(dt_us, reason);
                } else {
                    gate.last_tick = now;
                }
                open = open || met || timed_out;
                if (open) note_outcome(gate, met || force, now);
            }
            if (open) {
                opening.push_back({std::move(gate.queue), gate.value, std::move(gate.action)});
                opened[idx] = true;
            } else {
                blocked.push_back(queue);
            }
        }
        if (!opening.empty()) {
            std::deque<Gate> rest;
            for (std::size_t idx = 0; idx < gates_.size(); ++idx)
                if (!opened[idx]) rest.push_back(std::move(gates_[idx]));
            gates_.swap(rest);
        }
    }

    // Writes landed, then gates opened: the pump's one step. `force` runs on the caller's thread when the threads do
    // not, and signals there too.
    void pass(bool force) {
        const auto started = Clock::now();
        std::vector<WriteGroup> landing;
        {
            std::lock_guard lock(mutex_);
            busy_ = true;
            take_landed(force, landing);
        }
        const bool landed_any = !landing.empty();
        for (const WriteGroup& group : landing) land(group);
        std::vector<Release> opening;
        {
            std::lock_guard lock(mutex_);
            watch_late();
            take_open(force, landed_any, opening);
            if (!force && !opening.empty())
                pending_releases_.fetch_add(opening.size(), std::memory_order_release);
        }
        if (!opening.empty()) {
            if (force) {
                for (const Release& release : opening) signal(release);
            } else {
                {
                    std::lock_guard release(release_mutex_);
                    for (Release& r : opening) releases_.push_back(std::move(r));
                }
                release_ready_.notify_one();
            }
        }
        if (!force)
            raise_to(counters().pass_us_max,
                     static_cast<std::uint64_t>(
                         std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - started).count()));
        std::lock_guard lock(mutex_);
        busy_ = false;
    }

    // One line every 10 s while there is traffic, for boot logs.
    void summarize() {
        const auto now = Clock::now();
        if (now < next_summary_) return;
        next_summary_ = now + std::chrono::seconds(10);
        const Counters& c = counters();
        const auto get = [](const std::atomic<std::uint64_t>& v) {
            return static_cast<unsigned long long>(v.load(std::memory_order_relaxed));
        };
        std::fprintf(stderr,
                     "[rt] d3d12_xs: list values so far: %llu writes recorded, %llu landed, %llu failed; %llu waits "
                     "recorded, %llu met at submission, %llu relayed; %llu gates held, %llu met, %llu timed out "
                     "(%llu short, %llu demotions, %llu met late); %llu cuts (%llu deferred to a query's end, %llu "
                     "failed), %llu on bundles; releaser %llu signals, %.1f ms, longest %.2f ms, %llu stalls; longest pump pass "
                     "%.2f ms; %llu page remaps at the call (longest wait %.2f ms), %llu deferred (%llu failed); "
                     "timestamps %llu from the GPU, %llu from the CPU clock\n",
                     get(c.writes_recorded), get(c.writes_landed), get(c.writes_failed), get(c.waits_recorded),
                     get(c.waits_met_at_submit), get(c.waits_relayed), get(c.gates_held), get(c.gates_met),
                     get(c.gates_timed_out), get(c.gates_short), get(c.demotions), get(c.late_met), get(c.cuts),
                     get(c.cuts_deferred), get(c.cut_failures), get(c.bundle_ops), get(c.releases),
                     get(c.release_us) / 1000.0, get(c.release_us_max) / 1000.0, get(c.release_stalls),
                     get(c.pass_us_max) / 1000.0,
                     get(c.remaps_now), get(c.remap_wait_us_max) / 1000.0, get(c.remaps_deferred),
                     get(c.remaps_failed), get(c.stamps_gpu), get(c.stamps_cpu));
    }

    // A timestamp write's value: the GPU's timestamp for the write's place in its list when the GPU wrote one, else
    // the CPU's clock now. Flag 1 is in the queue's GetTimestampFrequency ticks; flag 2 (the system timestamp) is a
    // QueryPerformanceCounter value, the GPU's timestamp turned into one through the queue's clock calibration (never
    // later than now: the GPU got there before its fence was seen passed).
    static std::uint64_t stamp(QueueValues& queue, const Write& write) {
        LARGE_INTEGER now{};
        QueryPerformanceCounter(&now);
        const auto qpc = static_cast<std::uint64_t>(now.QuadPart);
        const std::uint64_t ticks = write.stamps ? write.stamps->read(write.slot) : 0;
        Counters& c = counters();
        if (write.kind == WriteKind::gpu_timestamp) {
            if (ticks) {
                bump(c.stamps_gpu);
                return ticks;
            }
            bump(c.stamps_cpu);
            return queue.gpu_ticks_at(qpc);
        }
        if (ticks) {
            if (const std::uint64_t at = queue.qpc_at(ticks)) {
                bump(c.stamps_gpu);
                return (std::min)(at, qpc);
            }
        }
        bump(c.stamps_cpu);
        return qpc;
    }

    // The first few groups with two GPU timestamps or more, and then one in 1000: how far apart the GPU took them, to
    // show in a boot log that the title's begin and end timestamps are its GPU work's, not one landing time.
    static void note_stamps(QueueValues& queue, const WriteGroup& group) {
        std::uint64_t first = 0, last = 0;
        unsigned count = 0;
        for (const Write& write : group.writes) {
            if (write.kind != WriteKind::gpu_timestamp || !write.stamps) continue;
            const std::uint64_t ticks = write.stamps->read(write.slot);
            if (!ticks) continue;
            if (!count++) first = ticks;
            last = ticks;
        }
        if (!count) return;
        const std::uint64_t previous = queue.last_gpu_stamp;
        queue.last_gpu_stamp = last;
        static std::atomic<std::uint64_t> groups{0};
        const std::uint64_t n = groups.fetch_add(1, std::memory_order_relaxed);
        if (count < 2 || (n >= 6 && n % 1000 != 0)) return;
        const double ms_per_tick = queue.gpu_frequency() ? 1000.0 / static_cast<double>(queue.gpu_frequency()) : 0.0;
        std::fprintf(stderr, "[rt] d3d12_xs: %u GPU timestamps landed together span %.3f ms (the first %.3f ms after "
                             "the queue's previous one)\n",
                     count, static_cast<double>(last - first) * ms_per_tick,
                     previous && first >= previous ? static_cast<double>(first - previous) * ms_per_tick : 0.0);
    }

    // A write's target, in this order (src/value_write_sink.h):
    // - an offset in the GPU address arena (a default-heap buffer): the device writes its own mapping. Asked before
    //   any host store, since a host store at an arena offset would write whatever host memory is at that number.
    //   One the arena refuses (its reserved start, a sparse buffer) is dropped.
    // - the title's memory: kernel_writes::store, as before.
    // - a page of xmem's that the title's view keeps PAGE_NOACCESS, where that store faults: xmem writes its own
    //   mapping of the physical page.
    void land(const WriteGroup& group) {
        note_stamps(*group.queue, group);
        for (const Write& write : group.writes) {
            const std::uint64_t value = write.kind == WriteKind::value ? write.value : stamp(*group.queue, write);
            void* const target = reinterpret_cast<void*>(static_cast<std::uintptr_t>(write.address));
            const leonida::value_sink::Landing arena = leonida::value_sink::arena(write.address, &value, write.bytes);
            bool landed = arena == leonida::value_sink::Landing::landed;
            if (landed) bump(counters().writes_arena);
            if (arena == leonida::value_sink::Landing::elsewhere) {
                landed = leonida::kernel_writes::store(target, value, write.bytes);
                if (!landed && leonida::value_sink::title_pages(target, &value, write.bytes)) {
                    landed = true;
                    bump(counters().writes_title_pages);
                }
            }
            if (landed) {
                bump(counters().writes_landed);
                continue;
            }
            bump(counters().writes_failed);
            static std::atomic<unsigned> logged{0};
            if (logged.fetch_add(1, std::memory_order_relaxed) < 4)
                std::fprintf(stderr, "[rt] d3d12_xs: a list's %u-byte write to 0x%llx faulted; it is dropped\n",
                             unsigned(write.bytes), static_cast<unsigned long long>(write.address));
        }
    }

    // The gates before it on its queue have opened: its time starts now. The short timeout goes to a gate with an
    // address that timed out three times in a row, and, while every wait is demoted, to one with an address no wait
    // was ever met for (an address the title waits on and some write does meet keeps its full timeout). Called with
    // mutex_ held.
    void start_gate(Gate& gate, Clock::time_point now) {
        const unsigned full = timeout_ms_.load(std::memory_order_relaxed);
        bool cut_short = false, unknown = false;
        for (const Wait& wait : gate.waits) {
            const Streak* streak = find_streak(wait.address);
            if (streak && streak->timeouts >= kDemoteAfter) cut_short = true;
            if (!streak || !streak->met) unknown = true;
        }
        cut_short = (cut_short || (demoted_ && unknown)) && kDemotedTimeoutMs < full;
        if (cut_short) bump(counters().gates_short);
        gate.started = true;
        gate.cut_short = cut_short;
        gate.since = now;
        gate.last_tick = now;
        const unsigned gate_ms = cut_short ? kDemotedTimeoutMs : full;
        const auto caps = leonida::list_value_wait::default_pause_caps(gate_ms, cut_short);
        gate.timeout.reset(static_cast<std::uint64_t>(gate_ms) * 1000u, caps.active_us, caps.max_us);
    }

    // A wait for `address` was met (in time, or late): the address has met history and its timeouts in a row end.
    // Called with mutex_ held.
    void note_met(std::uint64_t address) {
        Streak* streak = find_streak(address);
        if (!streak) streak = add_streak(address);
        streak->met = true;
        streak->timeouts = 0;
    }

    // A gate cut short by the demotion of every wait was met within its short timeout: waits are met again, every
    // wait gets the full timeout. (Not a gate with the full timeout, nor a late meet: those leave the waits that
    // nothing meets cut short.) Called with mutex_ held.
    void lift_demotion(std::uint64_t address) {
        if (!demoted_) return;
        demoted_ = false;
        static std::atomic<unsigned> logged{0};
        if (logged.fetch_add(1, std::memory_order_relaxed) < 8)
            std::fprintf(stderr, "[rt] d3d12_xs: a wait for 0x%llx cut short was met: waits get %u ms again\n",
                         static_cast<unsigned long long>(address), timeout_ms_.load(std::memory_order_relaxed));
    }

    // The waits cut short, watched until their full timeout would have run out. Called with mutex_ held.
    void watch_late() {
        if (late_.empty()) return;
        const auto now = Clock::now();
        for (auto it = late_.begin(); it != late_.end();) {
            if (met_now(it->wait)) {
                bump(counters().late_met);
                note_met(it->wait.address);
                it = late_.erase(it);
            } else if (now >= it->until) {
                it = late_.erase(it);
            } else {
                ++it;
            }
        }
    }

    // Called with mutex_ held.
    void note_outcome(const Gate& gate, bool met, Clock::time_point now) {
        if (met) {
            bump(counters().gates_met);
            timeouts_in_a_row_ = 0;
            for (const Wait& wait : gate.waits) note_met(wait.address);
            if (gate.cut_short && !gate.waits.empty()) lift_demotion(gate.waits.front().address);
            if (logged_met_ < 4 && !gate.waits.empty()) {  // the first few, to show the model at work in a boot log
                ++logged_met_;
                const double ms = std::chrono::duration<double, std::milli>(now - gate.since).count();
                std::fprintf(stderr, "[rt] d3d12_xs: a queue's wait for 0x%llx was met after %.2f ms\n",
                             static_cast<unsigned long long>(gate.waits.front().address), ms);
            }
            return;
        }
        bump(counters().gates_timed_out);
        const unsigned full = timeout_ms_.load(std::memory_order_relaxed);
        if (++timeouts_in_a_row_ >= kDemoteAllAfter && !demoted_ && kDemotedTimeoutMs < full) {
            demoted_ = true;
            bump(counters().demotions);
            static std::atomic<unsigned> logged{0};
            if (logged.fetch_add(1, std::memory_order_relaxed) < 8)
                std::fprintf(stderr, "[rt] d3d12_xs: %u waits in a row timed out: every wait gets %u ms until one "
                                     "cut short is met (addresses met before keep %u ms)\n", timeouts_in_a_row_,
                             kDemotedTimeoutMs, full);
        }
        bool watched = false;
        for (const Wait& wait : gate.waits) {
            if (met_now(wait)) continue;
            if (gate.cut_short && !watched && late_.size() < kMaxLate) {
                late_.push_back({wait, now + std::chrono::milliseconds(full)});
                watched = true;
            }
            Streak* streak = find_streak(wait.address);
            if (!streak) streak = add_streak(wait.address);
            ++streak->timeouts;
            std::uint64_t current = 0;
            read_value(wait.address, wait.wide, current);
            if (!streak->logged) {
                streak->logged = true;
                static std::atomic<unsigned> addresses{0};
                if (addresses.fetch_add(1, std::memory_order_relaxed) < 16)
                    std::fprintf(stderr,
                                 "[rt] d3d12_xs: a queue stood %u ms at a wait for 0x%llx (%u-bit, comparison %u, "
                                 "reference 0x%llx, memory 0x%llx); it runs on (logged once per address)\n",
                                 gate.cut_short ? kDemotedTimeoutMs : full,
                                 static_cast<unsigned long long>(wait.address), wait.wide ? 64u : 32u,
                                 wait.comparison, static_cast<unsigned long long>(wait.reference),
                                 static_cast<unsigned long long>(current));
            }
            if (streak->timeouts == kDemoteAfter) {
                static std::atomic<unsigned> demoted{0};
                if (demoted.fetch_add(1, std::memory_order_relaxed) < 16)
                    std::fprintf(stderr, "[rt] d3d12_xs: the wait for 0x%llx timed out %u times in a row; it gets "
                                         "%u ms until it is met again\n",
                                 static_cast<unsigned long long>(wait.address), kDemoteAfter, kDemotedTimeoutMs);
            }
        }
    }

    std::mutex mutex_;  // the lists below; never held while a fence is signalled or a write lands
    std::once_flag start_;
    HANDLE wake_ = nullptr;
    HANDLE timer_ = nullptr;
    HANDLE thread_ = nullptr;
    HANDLE releaser_ = nullptr;
    std::atomic<unsigned> timeout_ms_{200};
    std::deque<WriteGroup> writes_;
    std::deque<Gate> gates_;
    std::vector<Streak> streaks_;
    std::vector<Late> late_;
    std::uint64_t use_clock_ = 0;
    unsigned timeouts_in_a_row_ = 0;  // gates that timed out since one was met
    bool demoted_ = false;            // every wait gets the short timeout until one is met
    bool busy_ = false;               // a pass is landing or handing over what it took off the lists
    unsigned logged_met_ = 0;
    Clock::time_point next_summary_ = Clock::now() + std::chrono::seconds(10);  // the pump thread's only
    std::mutex release_mutex_;        // releases_ and releasing_
    std::condition_variable release_ready_;
    std::deque<Release> releases_;
    bool releasing_ = false;
    std::atomic<std::size_t> pending_releases_{0};  // queued + active releases, read lock-free in take_open()
    std::atomic<std::uint64_t> signal_since_ms_{0};  // GetTickCount64 when the releaser's current Signal began, or 0
    std::atomic<std::uint64_t> signal_value_{0};
    std::atomic<DWORD> releaser_tid_{0};
    bool stall_logged_ = false;  // the pump thread's only
    std::mutex settle_mutex_;          // the caller's-thread settling without threads (start_or_settle)
    std::atomic<bool> exiting_{false};  // stop() ran
    std::atomic<bool> stop_{false};    // the threads end
};

// What a console list set since its last Reset, so the next part of a list cut at a wait starts with it: a D3D12
// list starts from defaults, and a draw after the wait would otherwise run without the pipeline, root arguments,
// targets or viewports the title set before the wait. Every value is what the real list received (already translated
// by the proxy: real heaps, device addresses, layer descriptor handles). A bundle's state is kept too: what it sets
// stays set in the list that executes it (inherit).
// shortcut: render targets are set again from their descriptor handles, so a descriptor the title rewrote since it
// set the targets names the new view, not the one set.
class ListState {
public:
    enum RootKind : std::uint8_t { kNone, kTable, kCbv, kSrv, kUav };

    ListState() = default;
    ListState(const ListState&) = delete;
    ListState& operator=(const ListState&) = delete;

    void clear() {
        set_ = 0;
        viewport_count_ = scissor_count_ = heap_count_ = 0;
        vb_mask_ = so_mask_ = 0;
        for (Side& side : sides_) side.clear();
    }

    void pipeline(std::uint64_t pso) {
        pso_ = pso;
        set_ |= kPipeline;
    }
    void topology(std::uint64_t topology) {
        topology_ = topology;
        set_ |= kTopology;
    }
    void stencil_ref(std::uint64_t ref) {
        stencil_ = ref;
        set_ |= kStencil;
    }
    void blend_factor(const FLOAT* factor) {
        set_ |= kBlend;
        blend_null_ = !factor;
        if (factor) std::memcpy(blend_, factor, sizeof(blend_));
    }
    void viewports(UINT count, const D3D12_VIEWPORT* viewports) {
        viewport_count_ = viewports ? (std::min)(count, UINT(kMaxViewports)) : 0;
        if (viewport_count_) std::memcpy(viewports_, viewports, viewport_count_ * sizeof(D3D12_VIEWPORT));
    }
    void scissors(UINT count, const D3D12_RECT* rects) {
        scissor_count_ = rects ? (std::min)(count, UINT(kMaxViewports)) : 0;
        if (scissor_count_) std::memcpy(scissors_, rects, scissor_count_ * sizeof(D3D12_RECT));
    }
    void heaps(UINT count, const std::uint64_t* real_heaps) {
        heap_count_ = real_heaps ? (std::min)(count, 2u) : 0;
        for (UINT i = 0; i < heap_count_; ++i) heaps_[i] = reinterpret_cast<ID3D12DescriptorHeap*>(real_heaps[i]);
    }
    // A different root signature leaves the side's arguments undefined (D3D12): they are forgotten.
    void root_signature(bool graphics, std::uint64_t signature) {
        Side& side = sides_[graphics];
        if (!side.signature_set || side.signature != signature) side.clear();
        side.signature_set = true;
        side.signature = signature;
    }
    void root_arg(bool graphics, UINT index, RootKind kind, std::uint64_t value) {
        if (index >= kMaxRootArgs) return;
        Side& side = sides_[graphics];
        side.kinds[index] = kind;
        side.values[index] = value;
        side.arg_mask |= std::uint64_t{1} << index;
    }
    // A root signature holds at most 64 DWORDs, so no constant sits at an offset of 64 or more.
    void root_constants(bool graphics, UINT index, UINT count, const std::uint32_t* values, UINT offset) {
        if (!values || index >= kMaxRootArgs || offset >= kMaxConstants) return;
        count = (std::min)(count, kMaxConstants - offset);
        if (!count) return;
        Side& side = sides_[graphics];
        std::unique_ptr<std::uint32_t[]>& row = side.constants[index];
        if (!row) row.reset(new (std::nothrow) std::uint32_t[kMaxConstants]);
        if (!row) return;
        std::memcpy(row.get() + offset, values, count * sizeof(std::uint32_t));
        side.constant_mask[index] |= (count == 64 ? ~std::uint64_t{0} : ((std::uint64_t{1} << count) - 1)) << offset;
        side.constant_params |= std::uint64_t{1} << index;
    }
    void index_buffer(const D3D12_INDEX_BUFFER_VIEW* view) {
        set_ |= kIndexBuffer;
        ib_null_ = !view;
        if (view) ib_ = *view;
    }
    void vertex_buffers(UINT start, UINT count, const D3D12_VERTEX_BUFFER_VIEW* views) {
        for (UINT i = 0; i < count && start + i < kMaxVertexBuffers; ++i) {
            vbs_[start + i] = views ? views[i] : D3D12_VERTEX_BUFFER_VIEW{};
            vb_mask_ |= 1u << (start + i);
        }
    }
    void so_targets(UINT start, UINT count, const D3D12_STREAM_OUTPUT_BUFFER_VIEW* views) {
        for (UINT i = 0; i < count && start + i < kMaxSoTargets; ++i) {
            so_[start + i] = views ? views[i] : D3D12_STREAM_OUTPUT_BUFFER_VIEW{};
            so_mask_ |= 1u << (start + i);
        }
    }
    void render_targets(UINT count, const D3D12_CPU_DESCRIPTOR_HANDLE* rtvs, BOOL single,
                        const D3D12_CPU_DESCRIPTOR_HANDLE* dsv) {
        set_ |= kTargets;
        rt_single_ = single;
        rt_count_ = rtvs ? (std::min)(count, UINT(D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT)) : 0;
        const UINT handles = rt_single_ ? (rt_count_ ? 1u : 0u) : rt_count_;
        for (UINT i = 0; i < handles; ++i) rtvs_[i] = rtvs[i];
        rt_has_dsv_ = dsv != nullptr;
        if (dsv) dsv_ = *dsv;
    }
    void predication(std::uint64_t resource, std::uint64_t offset, std::uint64_t operation) {
        set_ |= kPredication;
        predication_[0] = resource;
        predication_[1] = offset;
        predication_[2] = operation;
    }

    // ExecuteBundle: what the bundle set stays set in this list afterwards (its pipeline, topology, blend factor,
    // stencil reference, input buffers, root signatures and root arguments; a bundle starts with this list's
    // descriptor heaps and cannot set targets, viewports, stream output or predication). A bundle that sets no root
    // signature works on this list's, whose arguments it changes.
    void inherit(const ListState& bundle) {
        if (bundle.set_ & kPipeline) pipeline(bundle.pso_);
        if (bundle.set_ & kTopology) topology(bundle.topology_);
        if (bundle.set_ & kStencil) stencil_ref(bundle.stencil_);
        if (bundle.set_ & kBlend) blend_factor(bundle.blend_null_ ? nullptr : bundle.blend_);
        if (bundle.set_ & kIndexBuffer) index_buffer(bundle.ib_null_ ? nullptr : &bundle.ib_);
        for (UINT i = 0; i < kMaxVertexBuffers; ++i)
            if (bundle.vb_mask_ >> i & 1) vertex_buffers(i, 1, &bundle.vbs_[i]);
        for (int graphics = 0; graphics < 2; ++graphics) {
            const Side& from = bundle.sides_[graphics];
            if (from.signature_set) root_signature(graphics, from.signature);
            for (UINT index = 0; index < kMaxRootArgs; ++index)
                if (from.arg_mask >> index & 1)
                    root_arg(graphics, index, static_cast<RootKind>(from.kinds[index]), from.values[index]);
            for (UINT index = 0; index < kMaxRootArgs; ++index) {
                const std::uint64_t mask = from.constant_mask[index];
                if (!(from.constant_params >> index & 1) || !mask) continue;
                for (UINT offset = 0; offset < kMaxConstants; ++offset)
                    if (mask >> offset & 1) root_constants(graphics, index, 1, &from.constants[index][offset], offset);
            }
        }
    }

    // Sets everything recorded on `list`, a list fresh from Reset: heaps and root signatures before the arguments
    // that depend on them.
    void replay(ID3D12GraphicsCommandList* list) const {
        if (heap_count_) list->SetDescriptorHeaps(heap_count_, heaps_);
        for (int graphics = 0; graphics < 2; ++graphics) {
            const Side& side = sides_[graphics];
            if (side.signature_set) {
                auto* signature = reinterpret_cast<ID3D12RootSignature*>(side.signature);
                graphics ? list->SetGraphicsRootSignature(signature) : list->SetComputeRootSignature(signature);
            }
            for (UINT index = 0; index < kMaxRootArgs; ++index) {
                if (!(side.arg_mask >> index & 1)) continue;
                const std::uint64_t value = side.values[index];
                switch (side.kinds[index]) {
                case kTable:
                    graphics ? list->SetGraphicsRootDescriptorTable(index, D3D12_GPU_DESCRIPTOR_HANDLE{value})
                             : list->SetComputeRootDescriptorTable(index, D3D12_GPU_DESCRIPTOR_HANDLE{value});
                    break;
                case kCbv:
                    graphics ? list->SetGraphicsRootConstantBufferView(index, value)
                             : list->SetComputeRootConstantBufferView(index, value);
                    break;
                case kSrv:
                    graphics ? list->SetGraphicsRootShaderResourceView(index, value)
                             : list->SetComputeRootShaderResourceView(index, value);
                    break;
                case kUav:
                    graphics ? list->SetGraphicsRootUnorderedAccessView(index, value)
                             : list->SetComputeRootUnorderedAccessView(index, value);
                    break;
                default:
                    break;
                }
            }
            // Each run of constants set in one call (one constant: the single setter).
            for (UINT index = 0; index < kMaxRootArgs; ++index) {
                const std::uint64_t mask = side.constant_mask[index];
                if (!(side.constant_params >> index & 1) || !mask) continue;
                const std::uint32_t* row = side.constants[index].get();
                for (UINT offset = 0; offset < kMaxConstants;) {
                    if (!(mask >> offset & 1)) {
                        ++offset;
                        continue;
                    }
                    UINT end = offset + 1;
                    while (end < kMaxConstants && (mask >> end & 1)) ++end;
                    if (end - offset == 1) {
                        graphics ? list->SetGraphicsRoot32BitConstant(index, row[offset], offset)
                                 : list->SetComputeRoot32BitConstant(index, row[offset], offset);
                    } else {
                        graphics ? list->SetGraphicsRoot32BitConstants(index, end - offset, row + offset, offset)
                                 : list->SetComputeRoot32BitConstants(index, end - offset, row + offset, offset);
                    }
                    offset = end;
                }
            }
        }
        if (set_ & kPipeline) list->SetPipelineState(reinterpret_cast<ID3D12PipelineState*>(pso_));
        if (set_ & kTopology) list->IASetPrimitiveTopology(static_cast<D3D12_PRIMITIVE_TOPOLOGY>(topology_));
        if (viewport_count_) list->RSSetViewports(viewport_count_, viewports_);
        if (scissor_count_) list->RSSetScissorRects(scissor_count_, scissors_);
        if (set_ & kBlend) list->OMSetBlendFactor(blend_null_ ? nullptr : blend_);
        if (set_ & kStencil) list->OMSetStencilRef(static_cast<UINT>(stencil_));
        if (set_ & kIndexBuffer) list->IASetIndexBuffer(ib_null_ ? nullptr : &ib_);
        for (UINT i = 0; i < kMaxVertexBuffers; ++i)
            if (vb_mask_ >> i & 1) list->IASetVertexBuffers(i, 1, &vbs_[i]);
        for (UINT i = 0; i < kMaxSoTargets; ++i)
            if (so_mask_ >> i & 1) list->SOSetTargets(i, 1, &so_[i]);
        if (set_ & kTargets)
            list->OMSetRenderTargets(rt_count_, rt_count_ ? rtvs_ : nullptr, rt_single_,
                                     rt_has_dsv_ ? &dsv_ : nullptr);
        if (set_ & kPredication)
            list->SetPredication(reinterpret_cast<ID3D12Resource*>(predication_[0]), predication_[1],
                                 static_cast<D3D12_PREDICATION_OP>(predication_[2]));
    }

private:
    static constexpr UINT kMaxViewports = D3D12_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE;
    static constexpr UINT kMaxRootArgs = 64;
    static constexpr UINT kMaxConstants = 64;  // DWORDs of one root parameter
    static constexpr UINT kMaxVertexBuffers = 32;
    static constexpr UINT kMaxSoTargets = 4;
    enum : std::uint32_t {
        kPipeline = 1, kTopology = 2, kStencil = 4, kBlend = 8, kIndexBuffer = 16, kTargets = 32, kPredication = 64,
    };
    struct Side {
        bool signature_set = false;
        std::uint64_t signature = 0;
        std::uint64_t arg_mask = 0;
        std::uint64_t values[kMaxRootArgs];
        std::uint8_t kinds[kMaxRootArgs];
        // Root constants by parameter and offset: a row of 64 DWORDs per parameter, made at its first constant and
        // kept across Resets; `constant_mask` says which DWORDs of it are set.
        std::uint64_t constant_params = 0;
        std::uint64_t constant_mask[kMaxRootArgs] = {};
        std::unique_ptr<std::uint32_t[]> constants[kMaxRootArgs];
        void clear() {
            signature_set = false;
            arg_mask = 0;
            for (UINT index = 0; index < kMaxRootArgs; ++index)
                if (constant_params >> index & 1) constant_mask[index] = 0;
            constant_params = 0;
        }
    };

    std::uint32_t set_ = 0;
    std::uint64_t pso_ = 0, topology_ = 0, stencil_ = 0;
    FLOAT blend_[4] = {};
    bool blend_null_ = false, ib_null_ = false, rt_has_dsv_ = false;
    BOOL rt_single_ = FALSE;
    UINT viewport_count_ = 0, scissor_count_ = 0, heap_count_ = 0, rt_count_ = 0;
    D3D12_VIEWPORT viewports_[kMaxViewports];
    D3D12_RECT scissors_[kMaxViewports];
    ID3D12DescriptorHeap* heaps_[2] = {};
    Side sides_[2];
    D3D12_INDEX_BUFFER_VIEW ib_{};
    std::uint32_t vb_mask_ = 0, so_mask_ = 0;
    D3D12_VERTEX_BUFFER_VIEW vbs_[kMaxVertexBuffers];
    D3D12_STREAM_OUTPUT_BUFFER_VIEW so_[kMaxSoTargets];
    D3D12_CPU_DESCRIPTOR_HANDLE rtvs_[D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT] = {};
    D3D12_CPU_DESCRIPTOR_HANDLE dsv_{};
    std::uint64_t predication_[3] = {};
};

}  // namespace rt::d3d12_xs::list_values
