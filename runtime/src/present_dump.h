#pragma once

// LEONIDA_PRESENT_DUMP: the final swapchain texels, encoded on a worker after their frame fence signals.
#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace leonida::present_dump {

inline unsigned texel_size(VkFormat format) {
    switch (format) {
    case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_B8G8R8A8_SRGB:
    case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_SRGB:
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32: case VK_FORMAT_A2R10G10B10_UNORM_PACK32: return 4;
    case VK_FORMAT_R16G16B16A16_SFLOAT: return 8;
    case VK_FORMAT_R32G32B32A32_SFLOAT: return 16;
    default: return 0;
    }
}

inline std::uint8_t unorm8(float v) {
    return v > 0 ? std::uint8_t(std::min(v, 1.0f) * 255 + 0.5f) : 0;
}

inline std::uint8_t half_unorm8(std::uint16_t bits) {
    const int exponent = bits >> 10;
    const int fraction = bits & 1023;
    if (bits >= 0x8000 || !exponent || (exponent == 31 && fraction)) return 0;
    return unorm8(std::ldexp(float(1024 + fraction), exponent - 25));
}

inline void rgba_row(std::uint8_t* dst, const std::uint8_t* src, unsigned width, VkFormat format, bool pq) {
    if (format == VK_FORMAT_R8G8B8A8_UNORM || format == VK_FORMAT_R8G8B8A8_SRGB) {
        std::memcpy(dst, src, size_t(width) * 4);
        return;
    }
    const unsigned stride = texel_size(format);
    for (unsigned x = 0; x < width; ++x, src += stride, dst += 4) {
        if (format == VK_FORMAT_B8G8R8A8_UNORM || format == VK_FORMAT_B8G8R8A8_SRGB) {
            dst[0] = src[2]; dst[1] = src[1]; dst[2] = src[0]; dst[3] = src[3];
        } else if (format == VK_FORMAT_R16G16B16A16_SFLOAT) {
            std::uint16_t v[4];
            std::memcpy(v, src, sizeof(v));
            std::transform(v, v + 4, dst, half_unorm8);
        } else if (format == VK_FORMAT_R32G32B32A32_SFLOAT) {
            float v[4];
            std::memcpy(v, src, sizeof(v));
            std::transform(v, v + 4, dst, unorm8);
        } else {
            // A cheap HDR10 preview: decode ST.2084 per component, clip at 203 nits and encode for SDR.
            // Keep the source primaries. A 10-bit SDR swapchain just scales its code values to eight bits.
            static const auto pq_to_sdr = [] {
                std::array<std::uint8_t, 1024> table{};
                for (unsigned i = 0; i < table.size(); ++i) {
                    const double p = std::pow(i / 1023.0, 1.0 / 78.84375);
                    const double linear = (10000.0 / 203.0) *
                        std::pow(std::max(0.0, p - 0.8359375) / (18.8515625 - 18.6875 * p), 1.0 / 0.1593017578125);
                    const double srgb = linear <= 0.0031308 ? 12.92 * linear : 1.055 * std::pow(linear, 1.0 / 2.4) - 0.055;
                    table[i] = unorm8(float(srgb));
                }
                return table;
            }();
            std::uint32_t packed;
            std::memcpy(&packed, src, sizeof(packed));
            for (unsigned channel = 0; channel < 3; ++channel) {
                const unsigned bits = (packed >> (10 * channel)) & 1023;
                dst[channel] = pq ? pq_to_sdr[bits] : std::uint8_t((bits * 255 + 511) / 1023);
            }
            if (format == VK_FORMAT_A2R10G10B10_UNORM_PACK32) std::swap(dst[0], dst[2]);
            dst[3] = std::uint8_t((packed >> 30) * 85);
        }
    }
}

inline std::array<std::uint8_t, 4> big_endian(std::uint32_t n) {
    return {std::uint8_t(n >> 24), std::uint8_t(n >> 16), std::uint8_t(n >> 8), std::uint8_t(n)};
}

inline bool chunk(std::FILE* file, const char (&type)[5], std::span<const std::uint8_t> data,
                  std::span<const std::uint8_t> prefix = {}) {
    static constexpr auto crc_table = [] {
        std::array<std::uint32_t, 256> table{};
        for (unsigned i = 0; i < table.size(); ++i) {
            unsigned c = i;
            for (int bit = 0; bit < 8; ++bit) c = (c >> 1) ^ ((c & 1) ? 0xedb88320u : 0);
            table[i] = c;
        }
        return table;
    }();
    std::uint32_t crc = ~0u;
    const auto write = [&](std::span<const std::uint8_t> bytes) {
        for (auto b : bytes) crc = crc_table[(crc ^ b) & 255] ^ (crc >> 8);
        return bytes.empty() || std::fwrite(bytes.data(), 1, bytes.size(), file) == bytes.size();
    };
    return std::fwrite(big_endian(std::uint32_t(prefix.size() + data.size())).data(), 1, 4, file) == 4 &&
           write({reinterpret_cast<const std::uint8_t*>(type), 4}) && write(prefix) && write(data) &&
           std::fwrite(big_endian(~crc).data(), 1, 4, file) == 4;
}

inline bool write_png(std::FILE* file, VkExtent2D extent, VkFormat format, bool pq, const std::uint8_t* src) {
    constexpr std::uint8_t signature[]{137, 80, 78, 71, 13, 10, 26, 10};
    std::array<std::uint8_t, 13> ihdr{};
    const auto w = big_endian(extent.width), h = big_endian(extent.height);
    std::copy(w.begin(), w.end(), ihdr.begin());
    std::copy(h.begin(), h.end(), ihdr.begin() + 4);
    ihdr[8] = 8;
    ihdr[9] = 6;  // 8-bit RGBA
    constexpr std::uint8_t zlib_header[]{0x78, 0x01};
    if (std::fwrite(signature, 1, sizeof(signature), file) != sizeof(signature) ||
        !chunk(file, "IHDR", ihdr) || !chunk(file, "IDAT", zlib_header)) return false;

    // IDAT chunks concatenate to one zlib stream. Each scanline starts with PNG filter 0 and occupies
    // one or more stored DEFLATE blocks (16-bit little-endian length and its ones' complement).
    std::vector<std::uint8_t> row(1 + size_t(extent.width) * 4);
    std::uint32_t a = 1, b = 0;
    for (unsigned y = 0; y < extent.height; ++y, src += size_t(extent.width) * texel_size(format)) {
        rgba_row(row.data() + 1, src, extent.width, format, pq);
        std::span<const std::uint8_t> rest(row);
        while (!rest.empty()) {
            auto block = rest.first(std::min<size_t>(rest.size(), 65535));
            rest = rest.subspan(block.size());
            const auto len = std::uint16_t(block.size());
            const std::uint8_t header[]{std::uint8_t(y + 1 == extent.height && rest.empty()),
                std::uint8_t(len), std::uint8_t(len >> 8), std::uint8_t(~len), std::uint8_t((len >> 8) ^ 255)};
            for (auto byte : block) {
                a = (a + byte) % 65521;
                b = (b + a) % 65521;
            }
            if (!chunk(file, "IDAT", block, header)) return false;
        }
    }
    return chunk(file, "IDAT", big_endian((b << 16) | a)) && chunk(file, "IEND", {});
}

class PresentDump {
public:
    PresentDump() {
#ifdef _WIN32
        wchar_t* env = nullptr;
        size_t length = 0;
        if (_wdupenv_s(&env, &length, L"LEONIDA_PRESENT_DUMP") != 0) {
            bad_value();
            return;
        }
        const std::unique_ptr<wchar_t, decltype(&std::free)> owned(env, &std::free);
#else
        const auto* env = std::getenv("LEONIDA_PRESENT_DUMP");
#endif
        if (!env) return;
        try {
            auto value = std::filesystem::path(env).native();
            std::uint64_t interval = 60;
            const auto comma = value.rfind(',');
            if (comma != value.npos) {
                const auto n = std::filesystem::path(value.substr(comma + 1)).string();
                const auto end = n.data() + n.size();
                const auto parsed = std::from_chars(n.data(), end, interval);
                if (parsed.ec != std::errc{} || parsed.ptr != end || !interval) {
                    bad_value();
                    return;
                }
                value.resize(comma);
            }
            if (value.empty()) {
                bad_value();
                return;
            }
            dir_ = std::filesystem::absolute(value);
            std::filesystem::create_directories(dir_);
            interval_ = interval;
        } catch (const std::exception&) { bad_value(); }
    }

    bool enabled() const { return interval_ != 0; }
    bool due(std::uint64_t frames_shown) const {
        const auto n = interval_.load();
        return n && frames_shown % n == 0;
    }
    void disable(const char* why) {
        if (interval_.exchange(0)) std::fprintf(stderr, "[rt] present: %s; LEONIDA_PRESENT_DUMP disabled\n", why);
    }
    void save(std::uint64_t frame, VkExtent2D extent, VkFormat format, bool pq, const std::uint8_t* src, size_t size) {
        try {
            // Bound memory to one outstanding write even at N=1. Teardown joins the final writer.
            if (writer_.joinable()) writer_.join();
            if (!enabled()) return;
            char name[48];
            std::snprintf(name, sizeof(name), "present_%06llu.png", static_cast<unsigned long long>(frame));
            writer_ = std::jthread([this, frame, extent, format, pq, path = dir_ / name,
                                    bytes = std::vector<std::uint8_t>(src, src + size)] {
                try {
                    if (publish(path, extent, format, pq, bytes.data())) {
                        const auto utf8 = path.u8string();
                        std::fprintf(stderr, "[rt] present: dumped frame %llu to %s\n",
                                     static_cast<unsigned long long>(frame), reinterpret_cast<const char*>(utf8.c_str()));
                        return;
                    }
                } catch (const std::exception&) {}
                disable("could not write the present dump");
            });
        } catch (const std::exception&) { disable("could not queue the present dump"); }
    }

private:
    static void bad_value() {
        static std::once_flag once;
        std::call_once(once, [] {
            std::fprintf(stderr, "[rt] present: invalid LEONIDA_PRESENT_DUMP (expected a writable dir[,N], N > 0); disabled\n");
        });
    }
    static bool publish(const std::filesystem::path& path, VkExtent2D extent, VkFormat format, bool pq,
                        const std::uint8_t* src) {
        // Never open an existing output for writing: it could be a link to a file outside the dump directory.
        // Build a complete sibling exclusively, then replace the directory entry in a single rename.
        auto tmp = path;
        tmp += ".tmp";
#ifdef _WIN32
        std::FILE* file = nullptr;
        _wfopen_s(&file, tmp.c_str(), L"wbx");
#else
        auto* file = std::fopen(tmp.c_str(), "wbx");
#endif
        if (!file) return false;
        bool ok = false;
        try { ok = write_png(file, extent, format, pq, src); } catch (const std::exception&) {}
        if (std::fclose(file) != 0) ok = false;
        std::error_code ec;
        if (ok) {
            std::filesystem::rename(tmp, path, ec);
            if (!ec) return true;
        }
        std::filesystem::remove(tmp, ec);
        return false;
    }

    std::filesystem::path dir_;
    std::atomic<std::uint64_t> interval_{0};
    std::jthread writer_;  // joined before the state it uses is destroyed
};

}  // namespace leonida::present_dump
