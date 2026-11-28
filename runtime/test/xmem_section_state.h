#pragma once

#include <windows.h>

inline DWORD expected_xmem_section_state() {
    char value[4]{};
    const DWORD length = GetEnvironmentVariableA("LEONIDA_XMEM_SEC_COMMIT", value, sizeof(value));
    const bool wine = GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "wine_get_version") != nullptr;
    return wine && !(length == 1 && value[0] == '0') ? MEM_COMMIT : MEM_RESERVE;
}
