#pragma once

// LEONIDA_LOCAL_DIAGNOSTICS=1 turns on the boot diagnostics that write to stderr from the draw, clear, present and
// descriptor paths (the lines tagged present-probe, fsdraw, heap-probe, desc-probe and "PresentX diag"). They were
// written to follow one frame of the retail title from its draws to the window and cost a lock and a line on paths
// that run for every draw, so they are off unless asked for. Read once; any value that does not start with '0'
// turns them on. No Windows dependency: the headers that use it are built into portable tests too.

#include <cstddef>
#include <cstdlib>

namespace leonida {

inline bool local_diagnostics() {
    static const bool on = [] {
#ifdef _WIN32
        char value[8]{};
        std::size_t length = 0;
        return getenv_s(&length, value, sizeof value, "LEONIDA_LOCAL_DIAGNOSTICS") == 0 && length > 1 &&
               value[0] != '0';
#else
        const char* value = std::getenv("LEONIDA_LOCAL_DIAGNOSTICS");
        return value && *value && *value != '0';
#endif
    }();
    return on;
}

}  // namespace leonida
