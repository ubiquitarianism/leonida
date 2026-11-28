#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <map>
#include <set>
#include <tuple>
#include <vector>

namespace rt::d3d12 {

// Caller holds the registry lock. Counters describe current placements, excluding superseded GPU aliases.
template<class Registry>
void log_title_image_uses(const Registry& textures, std::size_t retired_count, std::uint64_t retired_bytes, std::uint64_t shared_bytes) {
    std::size_t unbound = 0, unbound_placed = 0;
    std::uint64_t unbound_bytes = 0, unbound_placed_bytes = 0;
    // Two aliases of a stream-pool address resolve to the same CPU pages. Compare only live registry entries.
    std::set<std::uint64_t> placed_addresses;
    std::uint64_t attachments = 0, duplicates = 0, gpu_written = 0, cpu_unfilled = 0;
    std::size_t attachment_count = 0, duplicate_count = 0;
    std::uint64_t cold = 0, cold_4k = 0, replayed = 0, shader_only = 0;
    struct AllocationClass { std::uint64_t bytes = 0, unbound = 0, cold = 0; unsigned count = 0; };
    using Shape = std::tuple<std::uint64_t, unsigned, unsigned, unsigned, unsigned>;
    std::map<Shape, AllocationClass> classes;
    for (const auto& [address, t] : textures) {
        if (t->owned) {
            unbound += t->unbound_bytes != 0;
            unbound_bytes += t->unbound_bytes;
            continue;
        }
        unbound_placed += t->unbound_bytes != 0;
        unbound_placed_bytes += t->unbound_bytes;
        placed_addresses.insert(t->cpu);
        const auto& d = t->resource->desc();
        auto& c = classes[{d.Width, d.Height, unsigned(d.Format), unsigned(d.Flags), d.SampleDesc.Count}];
        ++c.count;
        c.bytes += t->backing_bytes;
        c.unbound += t->unbound_bytes;
        if (t->resource->replay_seen()) replayed += t->backing_bytes;
        else if (t->resource->shader_views()) shader_only += t->backing_bytes;
        else if (!t->mapped && !t->fills) {
            c.cold += t->backing_bytes;
            cold += t->backing_bytes;
            if (t->attachment_4k) cold_4k += t->backing_bytes;
        }
        if (t->attachment_4k) {
            ++attachment_count;
            attachments += t->backing_bytes;
        }
        if (t->gpu_written) gpu_written += t->backing_bytes;
        else if (!t->fills) cpu_unfilled += t->backing_bytes;
    }
    for (const auto& [address, t] : textures) {
        if (!t->owned || !placed_addresses.contains(t->cpu)) continue;
        ++duplicate_count;
        duplicates += t->backing_bytes;
    }
    constexpr double mib = 1.0 / 1048576.0;
    std::fprintf(stderr, "[rt] mem-probe: %zu unbound slot-133 images, %.1f MiB backing deferred, %.1f MiB shared zero images\n",
                 unbound_placed, unbound_placed_bytes * mib, shared_bytes * mib);
    std::fprintf(stderr, "[rt] mem-probe: slot-133 %.1f MiB referenced by replay/present, %.1f MiB with only "
                         "shader descriptors; %.1f MiB with no replay, shader descriptor, map or fill "
                         "(%.1f MiB 4K attachments)\n", replayed * mib, shader_only * mib, cold * mib, cold_4k * mib);
    std::vector<std::pair<Shape, AllocationClass>> ranked(classes.begin(), classes.end());
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
        return a.second.bytes + a.second.unbound > b.second.bytes + b.second.unbound;
    });
    for (std::size_t i = 0; i < std::min<std::size_t>(8, ranked.size()); ++i) {
        const auto& [shape, c] = ranked[i];
        const auto& [w, h, fmt, flags, samples] = shape;
        std::fprintf(stderr, "[rt] mem-probe: slot-133 %llux%u DXGI %u flags 0x%x samples %u: %u images %.1f MiB "
                             "backed, %.1f MiB deferred (%.1f MiB of backing untouched)\n",
                     (unsigned long long)w, h, fmt, flags, samples, c.count, c.bytes * mib, c.unbound * mib, c.cold * mib);
    }
    std::fprintf(stderr, "[rt] mem-probe: %zu unbound slot-70 images, %.1f MiB backing deferred; "
                         "%zu retired awaiting release %.1f MiB; %zu slot-70 images sharing slot-133 base %.1f MiB; "
                         "slot-133 GPU-writable %.1f MiB including %zu 4K attachments %.1f MiB; "
                         "CPU-fillable without a fill %.1f MiB\n",
                 unbound, unbound_bytes * mib, retired_count, retired_bytes * mib,
                 duplicate_count, duplicates * mib, gpu_written * mib, attachment_count, attachments * mib,
                 cpu_unfilled * mib);
}

}  // namespace rt::d3d12
