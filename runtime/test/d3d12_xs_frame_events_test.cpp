// Portable C++20 tests for runtime/src/d3d12_xs_frame_events.h (no Windows or Vulkan dependency).
#include "../src/d3d12_xs_frame_events.h"

#include <cstdint>
#include <cstdio>

#define CHECK(condition) do {                                        \
    if (!(condition)) {                                              \
        std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
        return 1;                                                    \
    }                                                                \
} while (0)

int main() {
    using namespace leonida::d3d12_xs;

    // 1. Struct ABI geometry.
    CHECK(sizeof(XboxFrameEvent) == 0x38);
    XboxFrameEvent events[6] = {};
    events[0].event_type = 0x1000u;
    events[0].status = 0x5Au;
    events[0].timestamp = 0x123456789ABCDEF0ull;

    // 2. Valid non-empty query clears *num_events to 0 without disturbing caller buffer bytes.
    std::uint32_t count = 6;
    CHECK(query_frame_events_empty(&count, events) == FrameEventQueryStatus::ok);
    CHECK(count == 0);
    CHECK(events[0].event_type == 0x1000u && events[0].status == 0x5Au &&
          events[0].timestamp == 0x123456789ABCDEF0ull);

    // 3. Zero-capacity query with null buffer succeeds cleanly.
    count = 0;
    CHECK(query_frame_events_empty(&count, nullptr) == FrameEventQueryStatus::ok);
    CHECK(count == 0);

    // 4. Null count pointer and non-zero capacity with null buffer are rejected without mutating *num_events.
    CHECK(query_frame_events_empty(nullptr, events) == FrameEventQueryStatus::null_count);
    count = 4;
    CHECK(query_frame_events_empty(&count, nullptr) == FrameEventQueryStatus::null_buffer);
    CHECK(count == 4);

    std::printf("d3d12_xs_frame_events: 0x38-byte record geometry and GetFrameEventsX validation passed\n");
    return 0;
}
