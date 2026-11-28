#pragma once
#include <windows.h>

namespace leonida::dstorage {
inline bool trace_enabled() {
    static const bool enabled = [] {
        char value[2]{};
        return GetEnvironmentVariableA("LEONIDA_DSTORAGE_TRACE", value, sizeof value) == 1 && value[0] == '1';
    }();
    return enabled;
}
} // namespace leonida::dstorage
