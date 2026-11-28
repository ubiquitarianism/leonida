// Process-local policy for this shim's ordinary indirect graphics recorder, not a device feature.
#pragma once

#include <windows.h>
#include <cwchar>
#include <cstdio>

namespace rt::d3d12 {
inline bool indirect_draws_disabled() {
    static const bool disabled = [] {
        wchar_t value[4] = {};
        // GetEnvironmentVariableW returns the required size (including NUL) on overflow.
        if (GetEnvironmentVariableW(L"LEONIDA_INDIRECT_DRAWS", value, 4) != 3) return false;
        if (std::wmemcmp(value, L"raw", 3) == 0)
            std::fprintf(stderr, "[rt] LEONIDA_INDIRECT_DRAWS=raw is obsolete; using guarded stateful draws\n");
        return std::wmemcmp(value, L"off", 3) == 0;
    }();
    return disabled;
}
}  // namespace rt::d3d12
