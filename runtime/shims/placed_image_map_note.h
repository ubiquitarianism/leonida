// Diagnostics for the address a title reads after mapping a placed image.
// The exposure lock passes its computed mip offset in D3D12_RANGE::Begin.
#pragma once

#include <windows.h>
#include <d3d12.h>
#include <atomic>
#include <charconv>
#include <cstdint>
#include <string_view>

namespace rt::placed_image_map_note {

// XMem writers are serialized by the fault lock. Never wait for that lock in
// a diagnostic; a concurrent reservation change is reported as unknown.
inline std::atomic<uint64_t> generation{}, gpu{}, cpu{}, size{};
inline void publish_pair(uint64_t gpu_base, uint64_t cpu_base, uint64_t bytes) {
    ++generation;
    gpu = gpu_base;
    cpu = cpu_base;
    size = bytes;
    ++generation;
}

inline bool title_pointer(uintptr_t address, uintptr_t& result) {
    const auto before = generation.load();
    const auto g = gpu.load(), c = cpu.load(), n = size.load();
    result = g && address >= g && address - g < n ? c + (address - g) : address;
    return !(before & 1) && generation.load() == before;
}

enum class Reason { readable, unbacked, inaccessible, invalid, empty, unspecified, unknown, count };

inline Reason classify(uintptr_t start, size_t bytes) {
    if (!bytes) return Reason::empty;
    if (bytes - 1 > UINTPTR_MAX - start) return Reason::invalid;
    // Bound query cost on fragmented reservations. Never probe the bytes.
    for (unsigned i = 0; i < 8; ++i) {
        MEMORY_BASIC_INFORMATION m{};
        if (!VirtualQuery(reinterpret_cast<void*>(start), &m, sizeof(m))) return Reason::unknown;
        if (m.State != MEM_COMMIT) return Reason::unbacked;
        if ((m.Protect & PAGE_GUARD) || !(m.Protect & (PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY |
                PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY))) return Reason::inaccessible;
        const auto remaining = m.RegionSize - (start - reinterpret_cast<uintptr_t>(m.BaseAddress));
        if (bytes <= remaining) return Reason::readable;
        bytes -= remaining;
        start += remaining;
    }
    return Reason::unknown;
}

inline void note(void* pointer, unsigned subresource, const D3D12_RANGE* range, bool exposure_shape) {
    const DWORD saved_error = GetLastError();
    const auto raw = reinterpret_cast<uintptr_t>(pointer);
    const auto begin = range ? range->Begin : 0, end = range ? range->End : 0;
    uintptr_t title{};
    Reason why = Reason::unspecified;
    if (range) {
        if (begin > UINTPTR_MAX - raw) why = Reason::invalid;
        else if (!title_pointer(raw + begin, title)) why = Reason::unknown;
        else if (begin > end) why = Reason::invalid;
        else why = classify(title, end - begin);
    }
    // Common images must not exhaust the exposure samples before world start.
    static std::atomic<unsigned> counts[2][unsigned(Reason::count)]{};
    auto& count = counts[exposure_shape][unsigned(why)];
    if (count.load(std::memory_order_relaxed) < 16 && count.fetch_add(1, std::memory_order_relaxed) < 16) {
        constexpr std::string_view names[]{"readable", "uncommitted", "inaccessible", "invalid-range",
                                           "empty-range", "unspecified-range", "unknown"};
        char line[320];
        char* p = line;
        auto put = [&](std::string_view s) { for (char ch : s) *p++ = ch; };
        auto hex = [&](uintptr_t n) { put("0x"); p = std::to_chars(p, line + sizeof(line), n, 16).ptr; };
        put("[rt] placed-image Map "); put(names[unsigned(why)]);
        put(exposure_shape ? " exposure-shape raw=" : " raw="); hex(raw);
        put(" subresource="); hex(subresource);
        put(" begin="); hex(begin); put(" end="); hex(end);
        put(" title="); hex(title); put("\n");
        DWORD written{};
        WriteFile(GetStdHandle(STD_ERROR_HANDLE), line, static_cast<DWORD>(p - line), &written, nullptr);
    }
    SetLastError(saved_error);
}
} // namespace rt::placed_image_map_note
