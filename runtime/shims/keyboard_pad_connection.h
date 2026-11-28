#pragma once

#include <cstdint>

namespace rt::gameinput::detail {

// Owned by the poll lock. Keep the virtual device ID until SDL confirms detachment;
// a successful attach followed by a failed open still leaves a device to release.
template <class Id, class Handle>
struct KeyboardPadConnection {
    Id id{};
    Handle js{};

    template <class Backend>
    bool update(bool enabled, Backend& b) {
        if (enabled != enabled_) {
            enabled_ = enabled;
            waiting_ = warned_ = false;  // changing mode never waits for the previous mode's retry
        }
        if (enabled && js) return true;
        if (!enabled && !id) return false;
        if (waiting_ && b.now() - failed_at_ < 1'000'000) return false;

        if (js) {  // switching off a connected pad
            b.close(js);
            js = {};
        }
        if (id) {  // switching off, or cleanup was incomplete on the previous poll
            if (!release(b)) return wait(b);
            if (announced_) b.removed();
            announced_ = false;
        }
        if (!enabled) return false;

        id = b.attach();
        if (id) js = b.open(id);
        if (!js) {
            if (!warned_) b.failed();  // preserve the attach/open error before cleanup
            warned_ = true;
            if (id) release(b);
            return wait(b);
        }
        waiting_ = warned_ = false;
        announced_ = true;
        b.attached();
        return true;
    }

private:
    template <class Backend>
    bool release(Backend& b) {
        // An already-removed ID also fails SDL_DetachVirtualJoystick. Retain only a still-live virtual device.
        if (!b.detach(id) && b.is_virtual(id)) return false;
        id = {};
        return true;
    }
    template <class Backend>
    bool wait(Backend& b) {
        failed_at_ = b.now();  // measure from completion, not from the start of a slow attempt
        waiting_ = true;
        return false;
    }
    std::uint64_t failed_at_ = 0;
    bool enabled_ = false, waiting_ = false, warned_ = false, announced_ = false;
};

}  // namespace rt::gameinput::detail
