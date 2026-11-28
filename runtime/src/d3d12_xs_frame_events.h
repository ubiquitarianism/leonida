#pragma once

// Portable Xbox Series D3D12 device slot 81 (GetFrameEventsX) ABI model (no Windows or Vulkan dependency).
//
// Observed call shape (static ABI inspection, offsets and stride only):
//   HRESULT GetFrameEventsX(uint64_t frameToken, uint32_t flags, uint32_t* pNumEvents, XboxFrameEvent* pEvents)
//
// Each output event record has stride 0x38 bytes:
//   +0x00: uint32_t event_type
//   +0x08: uint8_t  status (0 = ok)
//   +0x18: uint64_t timestamp

#include <cstddef>
#include <cstdint>

namespace leonida::d3d12_xs {

// Device slot 81 (GetFrameEventsX) output event record (exact stride 0x38 bytes).
struct XboxFrameEvent {
    std::uint32_t event_type = 0;     // +0x00
    std::uint32_t reserved0 = 0;      // +0x04
    std::uint8_t status = 0;          // +0x08 (0 = ok)
    std::uint8_t pad[15] = {};        // +0x09..+0x17
    std::uint64_t timestamp = 0;      // +0x18
    std::uint64_t reserved1[3] = {};  // +0x20..+0x37
};
static_assert(sizeof(XboxFrameEvent) == 0x38);
static_assert(offsetof(XboxFrameEvent, event_type) == 0x00);
static_assert(offsetof(XboxFrameEvent, status) == 0x08);
static_assert(offsetof(XboxFrameEvent, timestamp) == 0x18);

enum class FrameEventQueryStatus : std::uint32_t {
    ok = 0,
    null_count = 1,
    null_buffer = 2,
};

// Validates GetFrameEventsX output parameters and reports zero pending frame events.
// With no hardware flip-event history yet, reporting *num_events = 0 is cleanly handled by callers.
inline FrameEventQueryStatus query_frame_events_empty(std::uint32_t* num_events, XboxFrameEvent* events) {
    if (!num_events) return FrameEventQueryStatus::null_count;
    if (*num_events > 0 && !events) return FrameEventQueryStatus::null_buffer;
    *num_events = 0;
    return FrameEventQueryStatus::ok;
}

}  // namespace leonida::d3d12_xs
