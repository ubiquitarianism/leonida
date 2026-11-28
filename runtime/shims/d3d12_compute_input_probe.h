#pragma once

#include "../src/command_list.h"
#include "../src/fsdraw_probe.h"
#include "../src/title_va_alias.h"
#include "../src/title_page_watch.h"
#include "xmem.h"
#include <windows.h>
#include <algorithm>
#include <atomic>
#include <array>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace rt::compute_probe {
inline std::string_view filters() {
    static const char* value = std::getenv("LEONIDA_COMPUTE_INPUTS");
    return value ? value : "";
}
inline bool enabled() { return !filters().empty(); }
inline bool matches(const char* name) {
    if (!name) return false;
    auto parts = filters();
    while (!parts.empty()) {
        const auto end = parts.find(',');
        const auto part = parts.substr(0, end);
        if (!part.empty() && std::string_view(name).find(part) != std::string_view::npos) return true;
        if (end == parts.npos) break;
        parts.remove_prefix(end + 1);
    }
    return false;
}

// Each decimal pair selects a table root dword and a 32-byte descriptor index.
inline const auto& slots() {
    static const auto pairs = [] {
        std::vector<std::pair<unsigned, unsigned>> result;
        const char* p = std::getenv("LEONIDA_COMPUTE_INPUT_SLOTS");
        if (!p) return result;
        const char* end = p + std::strlen(p);
        while (p != end && result.size() < 256) {
            unsigned root{}, slot{};
            auto r = std::from_chars(p, end, root);
            if (r.ec != std::errc{} || r.ptr == end || *r.ptr != '/') break;
            auto s = std::from_chars(r.ptr + 1, end, slot);
            if (s.ec != std::errc{} || (s.ptr != end && *s.ptr != ',')) break;
            if (root < 32 && slot < (1u << 21)) result.emplace_back(root, slot);
            p = s.ptr == end ? end : s.ptr + 1;
        }
        return result;
    }();
    return pairs;
}

inline constexpr const char* writers[] = {"unobserved", "CreateConstantBufferView", "CreateShaderResourceView",
    "CreateUnorderedAccessView", "CopyDescriptors", "slot70", "slot71"};
inline std::uint8_t writer_code(std::string_view name) {
    for (std::uint8_t i = 1; i < std::size(writers); ++i)
        if (name == writers[i]) return i;
    return 0;
}

inline bool read_cpu(std::uint64_t address, void* out, std::size_t size) {
    SIZE_T read{};
    return address && ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(address),
                                        out, size, &read) && read == size;
}

struct Backing {
    const char* kind;
    std::uint64_t title{}, host{}, available{};
};

inline bool read_backing(Backing b, std::uint64_t offset, void* out, std::size_t size) {
    if (!b.host || offset > b.available || size > b.available - offset) return false;
    // The device owns the arena mapping for its lifetime. Linux cross-process reads can reject a
    // device mapping even though the current process can read it directly.
    if (std::string_view(b.kind) == "arena" || std::string_view(b.kind) == "arena-shadow" ||
        std::string_view(b.kind) == "window-shadow") {
        std::memcpy(out, reinterpret_cast<const void*>(b.host + offset), size);
        return true;
    }
    return read_cpu(b.host + offset, out, size);
}

template<class State>
Backing backing(State& state, std::uint64_t base) {
    auto& aliases = leonida::title_va_aliases();
    std::uint64_t source{}, room{};
    const bool alias = aliases.source_of_vsharp(base, source, room);
    if (base < state.arena.size) {
        std::lock_guard lock(state.arena.mutex);
        // The block map's own reader (src/arena_blocks.h): whether base is inside a free block, and where the next
        // block after it starts, without holding an iterator the block map could cut.
        std::uint64_t block_offset = 0, block_size = 0, next = state.arena.size;
        if (state.arena.blocks.free_at(base, block_offset, block_size, next)) return {"arena-free"};
        return {alias ? "arena-shadow" : "arena", source, reinterpret_cast<std::uint64_t>(state.arena.mapped) + base,
                alias ? std::min<std::uint64_t>(room, next - base) : next - base};
    }
    if (alias) {
        std::uint64_t host{}, bytes{};
        {
            std::lock_guard lock(state.title_window_mutex);
            if (state.external_windows.lookup(state.arena.address + base, host, bytes))
                return {"window-shadow", source, host, std::min(room, bytes)};
        }
        if (!state.context->diagnostic_buffer(source, host, bytes)) return {"alias-only", source};
        return {host == source ? "import" : host ? "external-shadow" : "external-buffer", source, host,
                std::min(room, bytes)};
    }
    std::uint64_t lo{}, hi{};
    if (state.geometry.cap()) {
        std::uint64_t mapped{}, size{}, source{};
        bool populated{};
        if (state.geometry.lookup(state.arena.address + base, mapped, size, &source, &populated))
            return {populated ? "geometry" : "geometry-pending", source, mapped, size};
    }
    if (aliases.title_memory(base, lo, hi)) return {"untranslated-title", base, 0, hi - base};
    const auto normalized = (base ^ (1ull << 47)) - (1ull << 47);
    std::uint64_t host{}, bytes{};
    if (state.context->diagnostic_buffer(normalized, host, bytes)) return {"registered-buffer", 0, host, bytes};
    return {"unknown"};
}

// LEONIDA_COMPUTE_HEAP_RECORDS=<table>/<slot>:<stride>:<dword>:<heap>[:<first>:<count>][,...]
// Records of `stride` bytes in the buffer at table root s<table>, descriptor <slot>, whose dword <dword> holds a
// directly indexed resource-heap index in its low 24 bits; s<heap> holds the heap's 32-bit shader VA. <first> and
// <count> name the root dwords that hold the dispatch's first record and record count; without them the first
// records of the buffer are read.
struct HeapRecordSpec {
    unsigned table{}, slot{}, stride{}, dword{}, heap{};
    unsigned first = ~0u, count = ~0u;
};
inline const auto& heap_record_specs() {
    static const auto specs = [] {
        std::vector<HeapRecordSpec> result;
        const char* p = std::getenv("LEONIDA_COMPUTE_HEAP_RECORDS");
        if (!p) return result;
        const char* end = p + std::strlen(p);
        while (p != end && result.size() < 8) {
            unsigned v[7]{};
            unsigned n = 0;
            for (; n < 7; ++n) {
                auto r = std::from_chars(p, end, v[n]);
                if (r.ec != std::errc{}) break;
                p = r.ptr;
                const char want = n == 0 ? '/' : ':';
                if (p == end || *p == ',' || *p != want) { ++n; break; }
                ++p;
            }
            if ((n == 5 || n == 7) && v[0] < 32 && v[4] < 32 && v[2] >= 4 && v[3] < v[2] / 4 &&
                (n == 5 || (v[5] < 32 && v[6] < 32)))
                result.push_back({v[0], v[1], v[2], v[3], v[4], n == 7 ? v[5] : ~0u, n == 7 ? v[6] : ~0u});
            while (p != end && *p != ',') ++p;
            if (p != end) ++p;
        }
        return result;
    }();
    return specs;
}

// Every descriptor the probe read, by shader address, with the GPU-visible words it saw at submission. After a device
// loss they are read again: a slot rewritten while a dispatch that names it was still queued shows up there.
struct Seen {
    std::uint64_t seq{};
    std::size_t cmd{};
    unsigned reg{}, slot{};
    std::array<std::uint32_t, 8> words{};
};
inline std::mutex& seen_mutex() { static std::mutex m; return m; }
inline std::map<std::uint64_t, Seen>& seen() { static std::map<std::uint64_t, Seen> m; return m; }
inline void remember(std::uint64_t at, std::uint64_t seq, std::size_t cmd, unsigned reg, unsigned slot,
                     const std::array<std::uint32_t, 8>& words) {
    std::lock_guard lock(seen_mutex());
    auto& m = seen();
    if (m.size() >= 65536 && !m.count(at)) return;
    m[at] = {seq, cmd, reg, slot, words};
}

template<class State>
bool gpu_words_now(State& state, std::uint64_t at, std::array<std::uint32_t, 8>& out) {
    if (at >= state.arena.size) {
        std::unique_lock lock(state.console_heap_mutex, std::try_to_lock);  // a lost device may hold it
        if (lock.owns_lock())
            for (const auto& heap : state.console_heaps) {
                const auto delta = at - heap->shader_va;
                if (heap->copy && delta <= heap->bytes && sizeof(out) <= heap->bytes - delta) {
                    std::memcpy(out.data(), heap->copy + delta, sizeof(out));
                    return true;
                }
            }
    }
    return leonida::fsdraw::arena_words(at, out.data(), out.size());
}

template<class State>
void recheck(State& state) {
    std::lock_guard lock(seen_mutex());
    std::size_t same = 0, changed = 0, unreadable = 0;
    for (const auto& [at, s] : seen()) {
        std::array<std::uint32_t, 8> now{};
        if (!gpu_words_now(state, at, now)) { ++unreadable; continue; }
        if (std::equal(now.begin(), now.begin() + 4, s.words.begin())) { ++same; continue; }
        ++changed;
        // No backing lookup here: the thread that saw the loss may hold the arena or window locks.
        std::fprintf(stderr, "[rt] compute-input: after loss, s%u/%u (last seen execute=%llu cmd=%zu) changed: "
                     "was [%s] now [%s]\n", s.reg, s.slot, (unsigned long long)s.seq, s.cmd,
                     leonida::fsdraw::hex_words(s.words.data(), 4).c_str(),
                     leonida::fsdraw::hex_words(now.data(), 4).c_str());
    }
    std::fprintf(stderr, "[rt] compute-input: after loss, %zu probed descriptor(s) unchanged, %zu changed, "
                 "%zu unreadable\n", same, changed, unreadable);
    // Every GPU-visible descriptor heap, for buffer V#s whose base is still the title's own memory: heap_base + that
    // address is no allocation, so a shader that writes through one loses the device.
    auto& aliases = leonida::title_va_aliases();
    auto scan = [&](const char* what, std::uint64_t shader_va, const std::uint8_t* words, std::uint64_t bytes) {
        std::size_t raw = 0;
        for (std::uint64_t at = 0; at + 32 <= bytes; at += 32) {
            std::uint32_t w[4];
            std::memcpy(w, words + at, sizeof(w));
            if (!w[2] || (w[3] >> 30)) continue;  // no records: every access is out of bounds
            const auto base = std::uint64_t(w[0]) | (std::uint64_t(w[1] & 0xffff) << 32);
            std::uint64_t lo{}, hi{}, title{}, room{};
            if (base < state.arena.size || !aliases.title_memory(base, lo, hi) ||
                aliases.source_of_vsharp(base, title, room))
                continue;
            const auto stride = (w[1] >> 16) & 0x3fffu;
            if (++raw <= 2048)
                std::fprintf(stderr, "[rt] compute-input: after loss, raw title V# in %s heap 0x%llx slot %llu: "
                             "[%08x %08x %08x %08x] base=0x%llx end=0x%llx\n", what, (unsigned long long)shader_va,
                             (unsigned long long)(at / 32), w[0], w[1], w[2], w[3], (unsigned long long)base,
                             (unsigned long long)(base + std::uint64_t(w[2]) * (stride ? stride : 1)));
        }
        std::fprintf(stderr, "[rt] compute-input: after loss, %s heap 0x%llx (%llu KiB): %zu raw title V#s\n", what,
                     (unsigned long long)shader_va, (unsigned long long)(bytes >> 10), raw);
    };
    {
        std::unique_lock lock(state.console_heap_mutex, std::try_to_lock);
        if (!lock.owns_lock()) std::fputs("[rt] compute-input: after loss, console heaps locked\n", stderr);
        else
            for (const auto& heap : state.console_heaps)
                if (heap->copy) scan("console", heap->shader_va, heap->copy, heap->bytes);
    }
    {
        std::unique_lock lock(state.visible_heap_mutex, std::try_to_lock);
        if (!lock.owns_lock()) std::fputs("[rt] compute-input: after loss, arena heaps locked\n", stderr);
        else
            for (const auto& [offset, heap] : state.visible_heaps)
                if (heap.type == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV && heap.end <= state.arena.size)
                    scan("arena", offset, state.arena.mapped + offset, heap.end - offset);
    }
}

template<class State>
Backing descriptor(State& state, std::uint64_t seq, std::size_t list, std::size_t cmd,
                const std::array<std::uint32_t, 32>& roots, unsigned reg, unsigned slot,
                std::array<std::uint32_t, 8>* words = nullptr) {
    const auto at = std::uint64_t(roots[reg]) + std::uint32_t(slot * 32u);
    std::array<std::uint32_t, 8> raw{}, gpu{};
    const char* origin = "no-heap";
    std::uint64_t index = 0, title_at = 0;
    bool table = false, have_gpu = false;
    {
        std::lock_guard lock(state.probe_mutex);
        for (const auto& heap : state.probes) {
            const auto delta = at - heap->shader_va;
            if (delta / 32 >= heap->slots || delta % 32) continue;
            index = delta / 32;
            title_at = heap->console_va + delta;
            table = read_cpu(title_at, raw.data(), sizeof(raw));
            if (!table) break;
            const auto h = State::probe_hash(reinterpret_cast<const std::uint8_t*>(raw.data()));
            origin = heap->api[index] == h && !heap->writer.empty() ? writers[heap->writer[index]] :
                     heap->api[index] ? "changed-since-API" : "no-API-write-recorded";
            if (heap->shadowed && at >= state.arena.size) {
                gpu = raw;
                have_gpu = true;
            }
            break;
        }
    }
    if (state.geometry.cap() && table && at >= state.arena.size) {
        std::lock_guard lock(state.console_heap_mutex);
        for (const auto& heap : state.console_heaps) {
            const auto delta = at - heap->shader_va;
            if (heap->copy && delta <= heap->bytes && sizeof(gpu) <= heap->bytes - delta) {
                std::memcpy(gpu.data(), heap->copy + delta, sizeof(gpu));
                have_gpu = true;
                break;
            }
        }
    }
    if (!have_gpu) have_gpu = leonida::fsdraw::arena_words(at, gpu.data(), gpu.size());
    if (have_gpu) remember(at, seq, cmd, reg, slot, gpu);
    const auto prefix = "[rt] compute-input: execute=%llu list=%zu cmd=%zu s%u/%u ";
    std::fprintf(stderr, prefix, (unsigned long long)seq, list, cmd, reg, slot);
    if (!have_gpu) {
        std::fprintf(stderr, "table=0x%llx no GPU-visible bytes\n", (unsigned long long)at);
        return {"missing-table"};
    }
    if (words) *words = gpu;
    std::fprintf(stderr, "heap-slot=%llu table-title=0x%llx writer=%s words=[%s] ",
        (unsigned long long)index, (unsigned long long)title_at, origin,
        leonida::fsdraw::hex_words(gpu.data(), gpu.size()).c_str());
    if (table && raw != gpu)
        std::fprintf(stderr, "title-words=[%s] ", leonida::fsdraw::hex_words(raw.data(), raw.size()).c_str());
    if (gpu[3] >> 30) {
        std::fprintf(stderr, "texture key=%s\n", leonida::fsdraw::has_tsharp(gpu.data()) ? "registered" : "absent");
        return {"texture"};
    }
    if (!(gpu[0] | gpu[1] | gpu[2] | gpu[3])) { std::fputs("null\n", stderr); return {"null"}; }
    const auto base = std::uint64_t(gpu[0]) | (std::uint64_t(gpu[1] & 0xffff) << 32);
    const auto stride = (gpu[1] >> 16) & 0x3fffu;
    const auto span = std::uint64_t(gpu[2]) * (stride ? stride : 1);
    auto b = backing(state, base);
    const auto signed_base = (base ^ (1ull << 47)) - (1ull << 47);
    std::fprintf(stderr, "V#=0x%llx bda=0x%llx stride=%u records=%u storage=%s title=0x%llx",
        (unsigned long long)base, (unsigned long long)(state.arena.address + signed_base), stride, gpu[2],
        b.kind, (unsigned long long)b.title);
    std::array<std::uint32_t, 16> data{};
    const auto n = std::min<std::uint64_t>({span, b.available, sizeof(data)}) / 4;
    if (n && read_backing(b, 0, data.data(), 4 * n))
        std::fprintf(stderr, " cpu-sample=[%s]", leonida::fsdraw::hex_words(data.data(), n).c_str());
    if (std::string_view(b.kind) == "untranslated-title") {
        std::uint64_t created_at{};
        const bool from_cbv = leonida::fsdraw::cbv_origin(base, created_at);
        const auto source = leonida::title_gpu_views::to_cpu(base);
        const auto aliased = leonida::title_va_aliases().shader_va(source);
        MEMORY_BASIC_INFORMATION region{};
        VirtualQuery(reinterpret_cast<const void*>(source), &region, sizeof(region));
        std::fprintf(stderr, " source=0x%llx source-alias=0x%llx state=0x%lx protect=0x%lx",
                     (unsigned long long)source, (unsigned long long)aliased, region.State, region.Protect);
        if (from_cbv) std::fprintf(stderr, " recorded-CBV-location=0x%llx", (unsigned long long)created_at);
        const auto size = std::min<std::uint64_t>(span, sizeof(data)) / 4;
        if (size && region.State == MEM_COMMIT && !(region.Protect & (PAGE_NOACCESS | PAGE_GUARD)) &&
            read_cpu(source, data.data(), size * 4))
            std::fprintf(stderr, " source-only=[%s]", leonida::fsdraw::hex_words(data.data(), size).c_str());
    }
    std::fputc('\n', stderr);
    b.available = std::min(b.available, span);
    return b;
}

// What a heap V# whose base no backing claimed points at: the title range that holds it, how its pages are mapped
// (MEM_MAPPED: xmem physical pages, the pool CopyPageMappingsBatchX remaps; MEM_PRIVATE: plain title memory), how
// many other addresses share its first page, and whether the layer knows a buffer or a GPU copy there.
template<class State>
void heap_target(State& state, std::uint64_t base, std::uint64_t span) {
    const auto cpu = leonida::title_gpu_views::to_cpu(base, span);
    std::uint64_t lo{}, hi{};
    const bool title = leonida::title_va_aliases().title_memory(cpu, lo, hi);
    MEMORY_BASIC_INFORMATION region{};
    const bool queried = VirtualQuery(reinterpret_cast<const void*>(cpu), &region, sizeof(region)) != 0;
    std::vector<std::uintptr_t> peers;
    rt::xmem::geometry_aliases(cpu & ~std::uint64_t(0xffff), peers);
    std::uint64_t host{}, room{};
    const bool buffer = state.context->diagnostic_buffer(cpu, host, room);
    const auto copy = state.geometry.cap() ? state.geometry.find(cpu, span) : 0;
    std::fprintf(stderr, " target: cpu=0x%llx other-view=0x%llx title-range=%s", (unsigned long long)cpu,
                 (unsigned long long)leonida::title_gpu_views::other_view(cpu), title ? "yes" : "no");
    if (title)
        std::fprintf(stderr, " range=[0x%llx,+0x%llx) offset=0x%llx", (unsigned long long)lo,
                     (unsigned long long)(hi - lo), (unsigned long long)(cpu - lo));
    if (queried)
        std::fprintf(stderr, " state=0x%lx type=%s protect=0x%lx alloc-base=0x%llx region=0x%llx", region.State,
                     region.Type == MEM_MAPPED ? "mapped" : region.Type == MEM_PRIVATE ? "private" :
                     region.Type == MEM_IMAGE ? "image" : "none", region.Protect,
                     (unsigned long long)reinterpret_cast<std::uintptr_t>(region.AllocationBase),
                     (unsigned long long)region.RegionSize);
    std::fprintf(stderr, " page-peers=%zu layer-buffer=%s known-va=%s geometry-copy=0x%llx span-fits-range=%s",
                 peers.size(), buffer ? "yes" : "no", state.context->knows_buffer_va(base) ? "yes" : "no",
                 (unsigned long long)copy, title && span <= hi - cpu ? "yes" : "no");
}

// The heap slots a table of records selects for one dispatch (LEONIDA_COMPUTE_HEAP_RECORDS). Records are read on
// the CPU from the title's copy when the table has one (the GPU copy can lag it by a submission), at most 64 distinct
// heap indices per dispatch.
template<class State>
void heap_records(State& state, std::uint64_t seq, std::size_t list, std::size_t cmd,
                  const std::array<std::uint32_t, 32>& roots, const HeapRecordSpec& spec) {
    constexpr std::size_t kMaxIndices = 64, kMaxRecordLines = 16, kMaxRecords = 4096;
    std::fprintf(stderr, "[rt] compute-heap-record: execute=%llu cmd=%zu spec=s%u/%u stride=%u dword=%u heap=s%u "
                 "table follows\n", (unsigned long long)seq, cmd, spec.table, spec.slot, spec.stride, spec.dword,
                 spec.heap);
    const Backing table = descriptor(state, seq, list, cmd, roots, spec.table, spec.slot);
    if (!table.host && !table.title) return;
    const std::uint64_t available = table.available;
    const std::uint64_t records = available / spec.stride;
    std::uint64_t first = 0, count = std::min<std::uint64_t>(records, kMaxRecords);
    if (spec.first < 32 && spec.count < 32) {
        first = roots[spec.first];
        count = std::min<std::uint64_t>(roots[spec.count], kMaxRecords);
    }
    if (first >= records) count = 0;
    else count = std::min(count, records - first);
    struct Use { std::uint64_t first_record = 0, records = 0; std::set<unsigned> high; };
    std::map<unsigned, Use> uses;
    std::uint64_t read = 0, unreadable = 0, differs = 0, skipped = 0;
    std::vector<std::uint32_t> words(spec.stride / 4), gpu(spec.stride / 4);
    for (std::uint64_t r = first; r < first + count; ++r) {
        const auto at = r * spec.stride;
        const bool from_title = table.title && read_cpu(table.title + at, words.data(), spec.stride);
        const bool from_gpu = read_backing(table, at, gpu.data(), spec.stride);
        if (!from_title && !from_gpu) { ++unreadable; continue; }
        if (!from_title) words = gpu;
        else differs += from_gpu && words != gpu;
        ++read;
        const auto value = words[spec.dword];
        const unsigned index = value & 0xffffffu;
        if (read <= kMaxRecordLines && spec.first < 32)
            std::fprintf(stderr, "[rt] compute-heap-record: execute=%llu cmd=%zu record=%llu from=%s value=0x%08x "
                         "index=%u high=%u words=[%s]\n", (unsigned long long)seq, cmd, (unsigned long long)r,
                         from_title ? "title" : "gpu-copy", value, index, value >> 24,
                         leonida::fsdraw::hex_words(words.data(), words.size()).c_str());
        auto it = uses.find(index);
        if (it == uses.end()) {
            if (uses.size() >= kMaxIndices) { ++skipped; continue; }
            it = uses.emplace(index, Use{r}).first;
        }
        ++it->second.records;
        it->second.high.insert(value >> 24);
    }
    std::map<std::string, unsigned> classes;
    for (const auto& [index, use] : uses) {
        std::fprintf(stderr, "[rt] compute-heap-record: execute=%llu cmd=%zu heap-index=%u records=%llu "
                     "first-record=%llu high=", (unsigned long long)seq, cmd, index,
                     (unsigned long long)use.records, (unsigned long long)use.first_record);
        for (unsigned h : use.high) std::fprintf(stderr, "%u;", h);
        std::fputc('\n', stderr);
        std::array<std::uint32_t, 8> v{};
        const Backing b = descriptor(state, seq, list, cmd, roots, spec.heap, index, &v);
        std::string kind = b.kind;
        const auto base = std::uint64_t(v[0]) | (std::uint64_t(v[1] & 0xffff) << 32);
        const auto stride = (v[1] >> 16) & 0x3fffu;
        const bool buffer = (v[0] | v[1] | v[2] | v[3]) && !(v[3] >> 30);
        if (buffer && kind != "arena" && kind != "arena-shadow" && kind != "import" && kind != "geometry" &&
            kind != "window-shadow") {
            std::fprintf(stderr, "[rt] compute-heap-record: execute=%llu cmd=%zu heap-index=%u base=0x%llx",
                         (unsigned long long)seq, cmd, index, (unsigned long long)base);
            heap_target(state, base, std::uint64_t(v[2]) * (stride ? stride : 1));
            std::fputc('\n', stderr);
        }
        ++classes[kind];
    }
    std::fprintf(stderr, "[rt] compute-heap-summary: execute=%llu cmd=%zu spec=s%u/%u:%u:%u:s%u records=%llu+%llu "
                 "read=%llu unreadable=%llu title-differs-from-copy=%llu distinct=%zu over-cap=%llu classes:",
                 (unsigned long long)seq, cmd, spec.table, spec.slot, spec.stride, spec.dword, spec.heap,
                 (unsigned long long)first, (unsigned long long)count, (unsigned long long)read,
                 (unsigned long long)unreadable, (unsigned long long)differs, uses.size(),
                 (unsigned long long)skipped);
    for (const auto& [kind, n] : classes) std::fprintf(stderr, " %s=%u", kind.c_str(), n);
    std::fputc('\n', stderr);
}

// The two measured culling kernels share this record format. Read a bounded CPU snapshot
// of the actual int offsets for each dispatched group. A prior GPU writer can still be in flight.
template<class State>
void culling_records(State& state, std::uint64_t seq, std::size_t list, std::size_t cmd,
                     const std::array<std::uint32_t, 32>& roots, unsigned groups, Backing indices, Backing records) {
    auto word = [](Backing b, std::uint64_t off, unsigned& out) {
        return read_backing(b, off, &out, sizeof(out));
    };
    std::set<unsigned> indices_seen;
    for (unsigned group = 0; group < std::min(groups, 256u); ++group) {
        unsigned record{};
        if (!word(indices, 4ull * group, record)) break;
        record &= ~3u;
        unsigned branches[2]{}, vb{}, count{};
        const bool valid = word(records, record, branches[0]) && word(records, record + 4ull, branches[1]) &&
                           word(records, record + 0x28ull, vb);
        std::fprintf(stderr, "[rt] compute-record: execute=%llu cmd=%zu group=%u offset=0x%x accessible=%u",
                     (unsigned long long)seq, cmd, group, record, valid);
        if (valid) {
            std::fprintf(stderr, " vertex-slot=%u", vb);
            unsigned byte_offset{}, first{}, count_instances{};
            if (word(records, record + 0x2cull, byte_offset) && word(records, record + 0x3cull, first) &&
                word(records, record + 0x40ull, count_instances))
                std::fprintf(stderr, " byte-offset=%u first=%u instances=%u", byte_offset, first, count_instances);
            indices_seen.insert(vb);
            for (unsigned branch : branches) {
                std::uint64_t node = branch & ~3u;
                unsigned indices_pair[2]{};
                if (word(records, node + 4, count)) std::fprintf(stderr, " node=0x%llx count4=%u", (unsigned long long)node, count);
                if (word(records, node + 0x374, count)) std::fprintf(stderr, " count374=%u", count);
                if (word(records, node + 0x39c, indices_pair[0]) && word(records, node + 0x3a0, indices_pair[1])) {
                    indices_seen.insert(indices_pair[0]);
                    indices_seen.insert(indices_pair[1]);
                }
            }
        }
        std::fputc('\n', stderr);
    }
    std::vector<std::uint32_t> alternatives;
    {
        std::lock_guard lock(state.probe_mutex);
        for (const auto& heap : state.probes)
            if (heap->shadowed && heap->shader_va <= UINT32_MAX && roots[7] != heap->shader_va)
                alternatives.push_back(static_cast<std::uint32_t>(heap->shader_va));
    }
    for (unsigned index : indices_seen) {
        descriptor(state, seq, list, cmd, roots, 7, index);
        for (auto heap : alternatives) {
            std::fprintf(stderr, "[rt] compute-record: comparison only, heap=0x%x index=%u\n", heap, index);
            auto expected = roots;
            expected[7] = heap;
            descriptor(state, seq, list, cmd, expected, 7, index);
        }
    }
}

template<class State>
void inspect(State& state, std::span<const leonida::CommandList* const> lists, std::uint64_t seq, const char* queue) {
    for (std::size_t l = 0; l < lists.size(); ++l) {
        const auto& cmds = lists[l]->commands();
        for (std::size_t i = 0; i < cmds.size(); ++i) {
            const auto* d = std::get_if<leonida::Dispatch>(&cmds[i]);
            const auto* ind = std::get_if<leonida::ExecuteIndirect>(&cmds[i]);
            if (!d && (!ind || ind->kind != leonida::IndirectKind::dispatch)) continue;
            const char* name = state.context->diagnostic_compute_name(d ? d->pipeline : ind->pipeline);
            if (!matches(name)) continue;
            // LEONIDA_COMPUTE_INPUT_LIMIT=<n>: the first n matching dispatches in full, then one in 1,000 (a culling
            // program runs several times a submission; 0 or unset: every one, as before).
            static const unsigned limit = [] {
                const char* v = std::getenv("LEONIDA_COMPUTE_INPUT_LIMIT");
                return v ? unsigned(std::strtoul(v, nullptr, 10)) : 0u;
            }();
            static std::atomic<unsigned> matched{0};
            if (const unsigned n = ++matched; limit && n > limit && n % 1000) continue;
            const auto& roots = d ? d->user_data : ind->user_data;
            std::fprintf(stderr, "[rt] compute-input: execute=%llu list=%zu cmd=%zu queue=%s name=%s pso=%p %s groups=%u,%u,%u "
                         "heap-base=0x%llx roots=[%s]\n", (unsigned long long)seq, l, i, queue, name, reinterpret_cast<void*>(d ? d->pipeline : ind->pipeline),
                         d ? "direct" : "indirect", d ? d->x : 0, d ? d->y : 0, d ? d->z : 0,
                         (unsigned long long)state.arena.address, leonida::fsdraw::hex_words(roots.data(), roots.size()).c_str());
            Backing indices{}, records{};
            for (const auto& [reg, slot] : slots()) {
                const auto b = descriptor(state, seq, l, i, roots, reg, slot);
                if (reg == 5 && slot == 0) indices = b;
                if (reg == 4 && slot == 86) records = b;
            }
            for (const auto& spec : heap_record_specs()) heap_records(state, seq, l, i, roots, spec);
            if (d && (std::string_view(name).ends_with("lib_cs_01405") || std::string_view(name).ends_with("lib_cs_01410")))
                culling_records(state, seq, l, i, roots, d->x, indices, records);
        }
    }
}
} // namespace rt::compute_probe
