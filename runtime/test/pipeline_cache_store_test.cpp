// The pure decisions behind D3D-704: how large a session cache may be (LEONIDA_PIPELINE_CACHE_BYTES), whether a
// blob's VkPipelineCacheHeaderVersionOne names this device, and whether a session file holds the offline seed it was
// written beside (pipeline_cache.session.seed). No Vulkan, no GPU, no files.
#include "../src/pipeline_cache_store.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>

#define CHECK(condition) do {                                        \
    if (!(condition)) {                                              \
        std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
        return 1;                                                    \
    }                                                                \
} while (0)

namespace store = leonida::pipeline_cache_store;

static void put32(std::uint8_t* p, std::uint32_t value) {
    for (int i = 0; i < 4; ++i) p[i] = static_cast<std::uint8_t>(value >> (8 * i));
}

static int test_limit() {
    using store::default_save_limit;
    // Absent, empty and the default knob.
    CHECK(store::parse_save_limit(nullptr).bytes == default_save_limit);
    CHECK(store::parse_save_limit(nullptr).valid);
    CHECK(store::parse_save_limit("").bytes == default_save_limit);
    // 0 is the documented "off", not a typo, and leading zeros are still a number.
    CHECK(store::parse_save_limit("0").bytes == 0);
    CHECK(store::parse_save_limit("0").valid);
    CHECK(store::parse_save_limit("007").bytes == 7);
    CHECK(store::parse_save_limit("1048576").bytes == 1048576);
    const std::string largest = std::to_string(std::numeric_limits<std::size_t>::max());
    CHECK(store::parse_save_limit(largest.c_str()).bytes == std::numeric_limits<std::size_t>::max());
    // Anything that is not a plain byte count falls back to the default and says the value was rejected.
    for (const char* text : {"-1", "+1", " 1", "1 ", "12x", "0x10", "1.5", "99999999999999999999999999"}) {
        const auto limit = store::parse_save_limit(text);
        CHECK(!limit.valid);
        CHECK(limit.bytes == default_save_limit);
    }
    // The limit counts what a session adds to an offline seed, not the seed the driver hands back with it.
    constexpr std::size_t gib = std::size_t{1} << 30;
    CHECK(!store::over_save_limit(3 * gib, 0, 4 * gib));
    CHECK(store::over_save_limit(5 * gib, 0, 4 * gib));
    CHECK(!store::over_save_limit(10 * gib, 9 * gib, 4 * gib));    // a 9 GB seed plus 1 GB of new builds is saved
    CHECK(store::over_save_limit(14 * gib, 9 * gib, 4 * gib));     // 5 GB added is over 4
    CHECK(!store::over_save_limit(4 * gib, 9 * gib, 1));           // smaller than the seed: nothing added
    CHECK(!store::over_save_limit(4 * gib + 1, 1, 4 * gib));       // exactly at the limit once the seed is taken off
    return 0;
}

static int test_header() {
    leonida::cached_pso::Identity identity;
    identity.vendor = 0x1002;
    identity.device = 0x73ff;
    identity.driver = 0x02000d0b;

    std::uint8_t header[store::header_bytes] = {};
    put32(header, store::header_bytes);
    put32(header + 4, store::header_version_one);
    put32(header + 8, identity.vendor);
    put32(header + 12, identity.device);
    for (int i = 0; i < 16; ++i) header[16 + i] = identity.cache_uuid[i] = static_cast<std::uint8_t>(i + 1);
    CHECK(store::header_matches(header, sizeof(header), identity));
    // Extra payload past the header does not matter: the driver reads the header and then its own data.
    std::uint8_t blob[store::header_bytes + 8] = {};
    std::memcpy(blob, header, sizeof(header));
    CHECK(store::header_matches(blob, sizeof(blob), identity));

    // Too short and null are "no cache", never a read past the end.
    CHECK(!store::header_matches(header, store::header_bytes - 1, identity));
    CHECK(!store::header_matches(nullptr, store::header_bytes, identity));

    // The four fields a driver checks before trusting a blob from another device, driver or header version.
    auto differs = [&](int offset, std::uint32_t value) {
        std::uint8_t changed[store::header_bytes];
        std::memcpy(changed, header, sizeof(changed));
        put32(changed + offset, value);
        return !store::header_matches(changed, sizeof(changed), identity);
    };
    CHECK(differs(0, store::header_bytes + 4));
    CHECK(differs(4, store::header_version_one + 1));
    CHECK(differs(8, identity.vendor + 1));
    CHECK(differs(12, identity.device + 1));
    std::uint8_t uuid[store::header_bytes];
    std::memcpy(uuid, header, sizeof(uuid));
    uuid[16] ^= 0xff;
    CHECK(!store::header_matches(uuid, sizeof(uuid), identity));
    // vendorID/deviceID are u32s little-endian; the same bytes read the other way name another device.
    leonida::cached_pso::Identity swapped = identity;
    swapped.vendor = 0x02100000;
    CHECK(!store::header_matches(header, sizeof(header), swapped));
    return 0;
}

// pipeline_cache.session.seed and the decision it serves: a session file written beside exactly this seed, and not
// changed since, is loaded alone instead of the seed plus a merge.
static int test_session_seed() {
    // The RX 7900 XT figures: a 1.88 GB seed and a session file 124 KB over it.
    const store::SeedStamp seed{1883920688, 134087654321098765ull};
    const store::SessionSeed recorded{seed, 1884044916};
    const std::string text = store::format_session_seed(recorded);
    CHECK(text == "leonida pipeline_cache.session.seed 1\nseed_bytes 1883920688\nseed_time 134087654321098765\n"
                  "session_bytes 1884044916\n");
    const auto parsed = store::parse_session_seed(text);
    CHECK(parsed && parsed->seed == seed && parsed->session_bytes == recorded.session_bytes);
    CHECK(store::session_holds_seed(parsed, seed, recorded.session_bytes));
    // The extremes survive the text.
    constexpr std::uint64_t most = std::numeric_limits<std::uint64_t>::max();
    const auto largest = store::parse_session_seed(store::format_session_seed({{most, most}, most}));
    CHECK(largest && largest->seed.bytes == most && largest->seed.write_time == most && largest->session_bytes == most);

    // A different seed: another size, or the same size with another time (a new corpus build, a seed copied in).
    CHECK(!store::session_holds_seed(parsed, {seed.bytes + 1, seed.write_time}, recorded.session_bytes));
    CHECK(!store::session_holds_seed(parsed, {seed.bytes - 1, seed.write_time}, recorded.session_bytes));
    CHECK(!store::session_holds_seed(parsed, {seed.bytes, seed.write_time + 1}, recorded.session_bytes));
    CHECK(!store::session_holds_seed(parsed, {seed.bytes, 0}, recorded.session_bytes));
    // No seed now (zero), or a file too short to be a pipeline cache.
    CHECK(!store::session_holds_seed(parsed, {}, recorded.session_bytes));
    const auto tiny = store::parse_session_seed(store::format_session_seed({{8, 5}, 64}));
    CHECK(tiny && !store::session_holds_seed(tiny, {8, 5}, 64));
    CHECK(store::session_holds_seed(store::parse_session_seed(store::format_session_seed({{32, 5}, 32})), {32, 5}, 32));
    // The session file changed size since the record: published again without it (a crash in between, an older
    // build) or replaced by hand.
    CHECK(!store::session_holds_seed(parsed, seed, recorded.session_bytes + 1));
    CHECK(!store::session_holds_seed(parsed, seed, recorded.session_bytes - 1));
    CHECK(!store::session_holds_seed(parsed, seed, 0));
    // A session file smaller than the seed was not the driver's union of it, whatever the record says.
    const auto smaller = store::parse_session_seed(store::format_session_seed({seed, seed.bytes - 1}));
    CHECK(smaller && !store::session_holds_seed(smaller, seed, seed.bytes - 1));
    // Exactly the seed's size (a session that added nothing) is still the union.
    const auto equal = store::parse_session_seed(store::format_session_seed({seed, seed.bytes}));
    CHECK(store::session_holds_seed(equal, seed, seed.bytes));
    // No record.
    CHECK(!store::session_holds_seed(std::nullopt, seed, recorded.session_bytes));

    // A truncated or corrupt record is no record.
    for (std::size_t cut = 0; cut < text.size(); ++cut) CHECK(!store::parse_session_seed(text.substr(0, cut)));
    CHECK(!store::parse_session_seed(text + "extra\n"));
    CHECK(!store::parse_session_seed(text + "\n"));
    for (const char* bad : {
             "leonida pipeline_cache.session.seed 2\nseed_bytes 1\nseed_time 1\nsession_bytes 1\n",  // another version
             "leonida pipeline_cache.session.seed 1\nseed_time 1\nseed_bytes 1\nsession_bytes 1\n",  // out of order
             "leonida pipeline_cache.session.seed 1\nseed_bytes 1\nseed_time 1\n",                   // a line short
             "leonida pipeline_cache.session.seed 1\nseed_bytes \nseed_time 1\nsession_bytes 1\n",   // no number
             "leonida pipeline_cache.session.seed 1\nseed_bytes -1\nseed_time 1\nsession_bytes 1\n", // a sign
             "leonida pipeline_cache.session.seed 1\nseed_bytes 1x\nseed_time 1\nsession_bytes 1\n", // stray text
             "leonida pipeline_cache.session.seed 1\nseed_bytes  1\nseed_time 1\nsession_bytes 1\n", // two spaces
             "leonida pipeline_cache.session.seed 1\nseed_bytes=1\nseed_time 1\nsession_bytes 1\n",  // another separator
             "leonida pipeline_cache.session.seed 1\r\nseed_bytes 1\r\nseed_time 1\r\nsession_bytes 1\r\n",  // CRLF
             "leonida pipeline_cache.session.seed 1\nseed_bytes 18446744073709551616\nseed_time 1\nsession_bytes 1\n",
         })
        CHECK(!store::parse_session_seed(bad));
    std::string flipped = text;
    flipped[flipped.find("seed_time ") + 10] = '\x01';
    CHECK(!store::parse_session_seed(flipped));

    // Whether the session file just written may be recorded. From the seed itself: anything at least the seed's size.
    const store::CacheOrigin from_seed{seed, seed.bytes};
    CHECK(store::may_record_session_seed(from_seed, seed.bytes));
    CHECK(store::may_record_session_seed(from_seed, recorded.session_bytes));
    CHECK(!store::may_record_session_seed(from_seed, seed.bytes - 1));
    // From a session file that holds the seed: a write-back smaller than that file lost entries (a damaged blob the
    // driver took only part of), even when it is still larger than the seed, so it is not recorded and the next
    // launch merges the seed again.
    const store::CacheOrigin from_session{seed, recorded.session_bytes};
    CHECK(store::may_record_session_seed(from_session, recorded.session_bytes));
    CHECK(store::may_record_session_seed(from_session, recorded.session_bytes + 4096));
    CHECK(!store::may_record_session_seed(from_session, recorded.session_bytes - 1));
    CHECK(!store::may_record_session_seed(from_session, seed.bytes + 1));
    CHECK(!store::may_record_session_seed(from_session, 0));
    // No seed: never recorded, whatever the sizes.
    CHECK(!store::may_record_session_seed({}, recorded.session_bytes));
    CHECK(!store::may_record_session_seed({{}, 64}, recorded.session_bytes));
    return 0;
}

int main() {
    if (const int failed = test_limit()) return failed;
    if (const int failed = test_session_seed()) return failed;
    return test_header();
}
