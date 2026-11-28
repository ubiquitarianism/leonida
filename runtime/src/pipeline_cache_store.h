// Persisting the runtime's VkPipelineCache across launches (D3D-704).
//
// Two files sit beside gta6rt.exe, and they are not interchangeable:
//
//   pipeline_cache.bin         the offline corpus seed the `pipelines` build stage writes (#367). Read only.
//   pipeline_cache.session.bin what the runtime compiled last time. Written on device teardown.
//
// Keeping them apart means a run can never replace the multi-GB corpus cache with the handful of pipelines it
// happened to compile. On the next launch both are loaded and merged, so the offline entries and the previous
// session's entries are available together.
//
// This header is pure stdlib on purpose (it does not include vulkan.h): pipeline_cache_store_test builds on hosts
// where CMake has no Vulkan headers. The byte layout below is VkPipelineCacheHeaderVersionOne, whose 32 bytes are
// part of Vulkan's ABI, not something a driver chooses.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

#include "cached_pso.h"

namespace leonida::pipeline_cache_store {

// A session cache holds what the boots so far compiled: 1.28 GB after a few world frames on NVIDIA and still growing,
// where a 256 MiB cap kept every later build from being saved. The limit applies to what the session adds: a cache
// seeded from the offline corpus file comes back from the driver as the whole union, so the seed's size is not
// counted (over_save_limit), and a 9.7 GB seed plus a few hundred MB of new builds is still saved. Past the limit the
// previous session file is left alone; raise it with LEONIDA_PIPELINE_CACHE_BYTES when a session really compiles that
// much.
inline constexpr std::size_t default_save_limit = std::size_t{4096} << 20;  // 4 GiB

// Whether a cache of `size` bytes, created from an offline seed of `seed` bytes (0: none), adds more than `limit`.
inline bool over_save_limit(std::size_t size, std::size_t seed, std::size_t limit) {
    return size - (seed < size ? seed : size) > limit;
}

struct SaveLimit {
    std::size_t bytes = default_save_limit;
    bool valid = true;  // false when the text was not a byte count; bytes then stays at the default
};

// LEONIDA_PIPELINE_CACHE_BYTES: a decimal byte count, 0 turning the session file off entirely. Empty or absent
// means the default; stray text, a sign or an overflow is reported by the caller and the default is used.
inline SaveLimit parse_save_limit(const char* text) {
    if (!text || !*text) return {};
    std::size_t value = 0;
    for (const char* p = text; *p; ++p) {
        if (*p < '0' || *p > '9') return {default_save_limit, false};
        const std::size_t digit = static_cast<std::size_t>(*p - '0');
        if (value > (std::numeric_limits<std::size_t>::max() - digit) / 10) return {default_save_limit, false};
        value = value * 10 + digit;
    }
    return {value, true};
}

inline constexpr std::size_t header_bytes = 32;
inline constexpr std::uint32_t header_version_one = 1;

inline std::uint32_t read_u32(const std::uint8_t* p) {
    return std::uint32_t(p[0]) | (std::uint32_t(p[1]) << 8) | (std::uint32_t(p[2]) << 16) |
           (std::uint32_t(p[3]) << 24);
}

// True when the blob's header names this physical device and pipeline-cache UUID, which is what a driver requires
// before it will read the opaque payload. The header is little-endian, like the host.
inline bool header_matches(const void* data, std::size_t size, const cached_pso::Identity& identity) {
    if (!data || size < header_bytes) return false;
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    return read_u32(bytes) == header_bytes && read_u32(bytes + 4) == header_version_one &&
           read_u32(bytes + 8) == identity.vendor && read_u32(bytes + 12) == identity.device &&
           std::memcmp(bytes + 16, identity.cache_uuid, sizeof(identity.cache_uuid)) == 0;
}

}  // namespace leonida::pipeline_cache_store
