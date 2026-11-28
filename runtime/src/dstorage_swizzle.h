#pragma once
#include <cstdint>
#include <cstring>
#include <span>

namespace leonida::dstorage {
// The public Xbox-GDK-Samples DirectStorageTextureShuffling example stores each BC component
// plane across the whole request. The MDU interleaves a stripe of each plane per group. The
// sample's compute shader does the remaining intra-group shuffle; surface tiling is already baked in.
inline constexpr unsigned stripes[][5] = {{}, {1, 1, 2}, {1, 1, 6}, {1, 3, 1, 1, 2}, {1, 3, 1, 3}};

inline size_t group_size(unsigned mode) { return mode == 1 ? 16384 : 32768; }

inline bool unshuffle(unsigned mode, std::span<const uint8_t> input, std::span<uint8_t> out) {
    if (!mode || mode > 4 || input.size() != out.size() || input.size() % group_size(mode)) return false;
    const size_t group = group_size(mode), groups = input.size() / group;
    size_t source = 0, offset = 0;
    for (unsigned pages : stripes[mode]) {
        if (!pages) break;
        const size_t bytes = pages * 4096;
        for (size_t i = 0; i < groups; ++i) {
            std::memcpy(out.data() + i * group + offset, input.data() + source, bytes);
            source += bytes;
        }
        offset += bytes;
    }
    return true;
}
} // namespace leonida::dstorage
