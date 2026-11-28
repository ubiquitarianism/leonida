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
// A third, small file, pipeline_cache.session.seed, says which seed a session file was written beside (SessionSeed):
// the driver writes back the whole union, so such a session file already holds every pipeline of that seed, and the
// next launch can load it alone instead of the seed plus a second cache of nearly the same size merged into it
// (session_holds_seed).
//
// This header is pure stdlib on purpose (it does not include vulkan.h): pipeline_cache_store_test builds on hosts
// where CMake has no Vulkan headers. The byte layout below is VkPipelineCacheHeaderVersionOne, whose 32 bytes are
// part of Vulkan's ABI, not something a driver chooses.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <string_view>

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

// Which pipeline_cache.bin a cache was created from: its size and last-write time (FILETIME, 100 ns ticks) when it
// was loaded. The pipelines build stage writes a new seed through a rename, which gives it a new time; a seed copied
// over with the same time and size is the same file in practice. Zero: no offline seed.
struct SeedStamp {
    std::uint64_t bytes = 0;
    std::uint64_t write_time = 0;
    bool operator==(const SeedStamp&) const = default;
};

// pipeline_cache.session.seed: the seed a session file was written beside, and that session file's size. Written
// after the session file is published, so a session file published since (by a crash between the two, or an older
// build that does not know of this file) has, in practice, another size.
struct SessionSeed {
    SeedStamp seed;
    std::uint64_t session_bytes = 0;
};

inline constexpr std::string_view session_seed_magic = "leonida pipeline_cache.session.seed 1";

// Four lines of text: the magic, then `seed_bytes N`, `seed_time N` and `session_bytes N`, in that order.
inline std::string format_session_seed(const SessionSeed& recorded) {
    return std::string(session_seed_magic) + "\nseed_bytes " + std::to_string(recorded.seed.bytes) + "\nseed_time " +
           std::to_string(recorded.seed.write_time) + "\nsession_bytes " + std::to_string(recorded.session_bytes) +
           "\n";
}

// The file's contents back, or nothing when they are not exactly what format_session_seed writes: a file cut short,
// edited or from another version is not trusted, and the next launch loads the seed and merges as before.
inline std::optional<SessionSeed> parse_session_seed(std::string_view text) {
    const auto line = [&text](std::string_view& out) {
        const auto end = text.find('\n');
        if (end == std::string_view::npos) return false;
        out = text.substr(0, end);
        text.remove_prefix(end + 1);
        return true;
    };
    const auto number = [&line](std::string_view key, std::uint64_t& out) {
        std::string_view got;
        if (!line(got) || got.size() <= key.size() + 1 || got.substr(0, key.size()) != key || got[key.size()] != ' ')
            return false;
        std::uint64_t value = 0;
        for (const char c : got.substr(key.size() + 1)) {
            if (c < '0' || c > '9') return false;
            const auto digit = std::uint64_t(c - '0');
            if (value > (std::numeric_limits<std::uint64_t>::max() - digit) / 10) return false;
            value = value * 10 + digit;
        }
        out = value;
        return true;
    };
    std::string_view magic;
    SessionSeed recorded;
    if (!line(magic) || magic != session_seed_magic || !number("seed_bytes", recorded.seed.bytes) ||
        !number("seed_time", recorded.seed.write_time) || !number("session_bytes", recorded.session_bytes) ||
        !text.empty())
        return std::nullopt;
    return recorded;
}

// Whether a session file of `session_bytes` holds every pipeline of the offline seed `seed` as it is now, so it can
// be loaded instead of the seed and the merge: it was written beside exactly this seed and has not changed size since
// (the driver's union, which is never smaller than the seed). Anything else, or no record, is the merge as before.
inline bool session_holds_seed(const std::optional<SessionSeed>& recorded, const SeedStamp& seed,
                               std::uint64_t session_bytes) {
    return recorded && seed.bytes >= header_bytes && recorded->seed == seed &&
           recorded->session_bytes == session_bytes && session_bytes >= seed.bytes;
}

// Where a device's shared cache came from: the offline seed it holds (zero for none) and the size of the blob it was
// created from, the seed itself or a session file that holds it.
struct CacheOrigin {
    SeedStamp seed;
    std::uint64_t loaded_bytes = 0;
};

// Whether a session file of `written` bytes, written back from a cache of `origin`, may be recorded as holding the
// seed. Not without a seed, and not when it is smaller than the seed or than the blob the cache was created from: the
// driver's union of a cache it was given does not shrink, so a smaller one lost entries on the way (a damaged file
// the driver took only part of). Recorded, that loss would be carried to every later launch, because the seed would
// not be loaded again to put the entries back; without a record the next launch loads the seed and merges.
inline bool may_record_session_seed(const CacheOrigin& origin, std::uint64_t written) {
    return origin.seed.bytes && written >= origin.seed.bytes && written >= origin.loaded_bytes;
}

}  // namespace leonida::pipeline_cache_store
