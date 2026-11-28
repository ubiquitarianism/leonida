// Offline inference of built-in BPTC predictor bits from legal anchor indices
// and BC7 mode-1 shared p-bits. No texture contents or fitted values are built in.
#include "../runtime/src/bcpack.h"
#include "../runtime/src/bcpack_stream.h"
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

namespace {
using namespace leonida::bcpack;
using namespace leonida::bcpack::detail;
using Map = std::array<uint8_t, 16>;
using Votes = std::array<uint64_t, 16>;

struct IndexChain {
    int root = -1;
    std::array<Map, 32> values{};
};
struct Red {
    unsigned subset{};
    // Mode 1 compares bit 1. Higher initial bits cannot carry down into it.
    std::array<uint8_t, 4> values{};
};
struct EndpointChain {
    int root = -1;
    std::array<Red, 6> red{};
};
struct Evidence {
    Votes indices[256][32]{};
    Votes parity[256][3]{};
};

void scan_indices(IndexChain& out, const IndexChain* parent, int root,
                  const Symbols& a, const Symbols& b, int bits, bool two, int anc1, int anc2,
                  Evidence& evidence) {
    out.root = root;
    if (root < 0) return;
    constexpr int secondary[] = {6, 7, 8, 10};
    for (int pos = 0; pos < (two ? 32 : 16); ++pos) {
        const int px = pos % 16, y = px / 4, x = px % 4;
        const int lane = pos < 16 ? 11 + y : secondary[y];
        const int symbol = (x < 2 ? a[lane] : b[lane]) >> ((x % 2) ^ (y < 3));
        const int delta = unzig((symbol & 1) | ((symbol >> 1) & 2) | ((symbol >> 2) & 4) | ((symbol >> 3) & 8));
        const int shift = pos < 16 ? 4 - bits : 2;
        auto& values = out.values[pos];
        for (unsigned h = 0; h < 16; ++h) {
            const int base = parent ? parent->values[pos][h] : h;
            values[h] = uint8_t((((base >> shift) + delta) * (1 << shift)) & 15);
        }
        if (px == 0 || pos == anc1 || pos == anc2) {
            auto& votes = evidence.indices[root][pos];
            for (unsigned h = 0; h < 16; ++h) votes[h] += values[h] >> 3;
        }
    }
    if (!two) std::fill(out.values.begin()+16, out.values.end(), Map{});
}

void scan_endpoints(EndpointChain& out, const EndpointChain* parent, int root,
                    const Symbols& a, const Symbols& b, const BC7Format& fmt,
                    int rot, Evidence& evidence) {
    out.root = root;
    if (root < 0) return;
    const int bits = (rot == 1 ? fmt.alpha : fmt.color) + (fmt.pbits != 0);
    const int shift = 8 - bits;
    constexpr int red_lane[] = {2, 6, 10};
    for (int sub = 0; sub < fmt.subsets; ++sub) {
        for (int e = 0; e < 2; ++e) {
            const int k = 2 * sub + e;
            auto& red = out.red[k];
            red.subset = parent ? parent->red[k].subset : sub;
            const int delta = unzig((e ? b : a)[red_lane[sub]]);
            for (int h = 0; h < 4; ++h) {
                const int base = parent ? parent->red[k].values[h] : h;
                red.values[h] = uint8_t(((base >> shift) + delta) * (1 << shift));
            }
        }
        if (fmt.pbits == 2) {
            const auto& left = out.red[2 * sub];
            const auto& right = out.red[2 * sub + 1];
            auto& votes = evidence.parity[root][left.subset];
            for (int x = 0; x < 4; ++x)
                for (int y = 0; y < 4; ++y)
                    votes[x*4 + y] += ((left.values[x] ^ right.values[y]) >> 1) & 1;
        }
    }
    for (int k = 2*fmt.subsets; k < 6; ++k) out.red[k] = out.red[k % 2];
}

bool train(unsigned codec, span<const uint8_t> bytes, size_t size, Evidence& evidence) {
    auto indices = std::make_unique<std::array<IndexChain, 256>>();
    std::array<EndpointChain, 256> endpoints{};
    return visit_blocks(codec, bytes, size, [&](size_t i, const Symbols& a, const Symbols& b) {
        int bits, anc1 = -1, anc2 = -1;
        bool two = false;
        const unsigned di = interleave(a[1], b[1]);
        if (codec == 7) {
            if (a[15] >= 32 || b[15] >= 16) return false;
            const auto [tag, part, rot, isel] = bc7Config[a[15]*16 + b[15]];
            const int mode = tag & 127;
            if (mode > 7 || part >= 64) return false;
            const auto& fmt = bc7_formats[mode];
            bits = fmt.index;
            two = mode == 4 || mode == 5;
            if (fmt.subsets == 2) anc1 = anchor2[part];
            if (fmt.subsets == 3) { anc1 = anchor3a[part]; anc2 = anchor3b[part]; }
            const unsigned de = (di + unzig(interleave(a[0], b[0]))) & 255;
            const auto* prev = de && de <= i ? &endpoints[(i-de)&255] : nullptr;
            const int root = de > i ? int(256+i-de) : prev ? prev->root : -1;
            scan_endpoints(endpoints[i&255], prev, root, a, b, fmt, rot, evidence);
        } else {
            if (a[15] >= 14) return false;
            const bool split = bc6Modes[a[15]].regions == 2;
            bits = split ? 3 : 4;
            if (split) anc1 = anchor2[std::min<unsigned>(b[15], 31)];
        }
        const auto* prev = di && di <= i ? &(*indices)[(i-di)&255] : nullptr;
        const int root = di > i ? int(256+i-di) : prev ? prev->root : -1;
        scan_indices((*indices)[i&255], prev, root, a, b, bits, two, anc1, anc2, evidence);
        return true;
    });
}

// The binary protocol is private to build_dstorage_history.py. Stream one bounded
// chunk at a time. Only a fully parsed chunk can contribute votes.
int run() {
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    auto evidence = std::make_unique<std::array<Evidence, 2>>();
    auto chunk = std::make_unique<Evidence>();
    unsigned accepted = 0, rejected = 0;
    uint32_t header[3];
    size_t n;
    while ((n = std::fread(header, 1, sizeof header, stdin)) != 0) {
        const auto [codec, size, packed] = header;
        if (n != sizeof header || (codec != 6 && codec != 7) || !size || size%16 ||
            size > 32u<<20 || !packed || packed > 32u<<20) return 2;
        std::vector<uint8_t> bytes(packed);
        if (std::fread(bytes.data(), 1, packed, stdin) != packed) return 3;
        std::memset(chunk.get(), 0, sizeof(Evidence));
        if (!train(codec, bytes, size, *chunk)) {
            ++rejected;  // discard this stream's evidence, including valid-looking prefixes
            continue;
        }
        ++accepted;
        auto& total = (*evidence)[codec-6];
        for (unsigned k = 0; k < 256; ++k) {
            for (unsigned j = 0; j < (codec == 6 ? 16u : 32u); ++j)
                for (unsigned h = 0; h < 16; ++h) total.indices[k][j][h] += chunk->indices[k][j][h];
            if (codec == 7)
                for (unsigned j = 0; j < 3; ++j)
                    for (unsigned h = 0; h < 16; ++h) total.parity[k][j][h] += chunk->parity[k][j][h];
        }
    }
    if (std::ferror(stdin)) return 4;
    std::fprintf(stderr, "BPTC constraint pass: %u valid streams, %u unsupported or malformed\n", accepted, rejected);
    for (const auto& e : *evidence) {
        if (std::fwrite(e.indices, 1, sizeof e.indices, stdout) != sizeof e.indices ||
            std::fwrite(e.parity, 1, sizeof e.parity, stdout) != sizeof e.parity) return 5;
    }
    return std::fflush(stdout) ? 5 : 0;
}
}
int main() { return run(); }
