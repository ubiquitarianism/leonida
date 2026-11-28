// Registry shim: the console model the title reads from the registry.
//
// GTA VI does not only call XSystemGetDeviceType. Code linked into the executable (and into libHttpClient.GDK.dll
// and XCurl.dll, which carry the same statically linked GDK helper) opens
//     HKEY_LOCAL_MACHINE\Software\Microsoft\Windows NT\CurrentVersion\OEM
// with RegOpenKeyW / RegOpenKeyExW, reads the DWORD "ConsoleMode" with RegQueryValueExW (0x64 Series X, 0x60 Series
// S; a missing value means a PC, and a PC takes the PC memory and resolution budgets), and calls RegCloseKey. Under
// Proton the value could be written into the prefix, and on a Windows PC it needs administrator rights. Leonida
// answers it itself, from the same setting as XSystemGetDeviceType (LEONIDA_DEVICE_TYPE, console_identity.h).
//
// Only that one key and that one value are ours: a pseudo key stands for the key, and everything else (every other
// key, value, handle) goes to the host's advapi32 unchanged. For a device type that has no ConsoleMode (a PC, an
// Xbox One) the key is not ours either, so the host registry answers and a missing key stays missing.
//
// Bound for ADVAPI32.dll and for the registry API sets (XCurl.dll imports api-ms-win-core-registry-l1-1-0 and
// l2-1-0, measured with tools/pe_inspect.py); rt::shims::owns_dll keeps advapi32 and the api sets host modules.
//
// Ported with credit from VIctory crates/victory-hle/src/modules/xsystem.rs (mod device_type, commit 2849477,
// GPL-3.0-or-later): the pseudo-key design and the way a null buffer, the size and the type are answered. The
// hook in xbox-emulator crates/compat/compat-common/src/gdk/hook.rs (commit 0284e83, GPL-3.0) did the same with
// an import patch. Written here in C++ against the documented RegQueryValueExW contract.
#include <windows.h>

#include <atomic>
#include <cstdio>
#include <cstring>

#include "console_identity.h"
#include "console_registry.h"
#include "shims.h"

namespace rt::shims::console_registry {
namespace {

// HKEY_LOCAL_MACHINE as the title passes it: the predefined handle value (sign-extended 0x80000002).
const HKEY kLocalMachine = HKEY_LOCAL_MACHINE;
constexpr wchar_t kOemKey[] = L"Software\\Microsoft\\Windows NT\\CurrentVersion\\OEM";
constexpr wchar_t kConsoleMode[] = L"ConsoleMode";

// The pseudo key's handle value: the address of a static, so it is never a kernel handle value.
char g_oem_key;

std::atomic<const HostRegistry*> g_test_host{nullptr};

HostRegistry load_advapi32() {
    HostRegistry h{};
    if (const HMODULE m = LoadLibraryExW(L"advapi32.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32)) {
        h.open_key = reinterpret_cast<decltype(h.open_key)>(GetProcAddress(m, "RegOpenKeyW"));
        h.open_key_ex = reinterpret_cast<decltype(h.open_key_ex)>(GetProcAddress(m, "RegOpenKeyExW"));
        h.query_value_ex = reinterpret_cast<decltype(h.query_value_ex)>(GetProcAddress(m, "RegQueryValueExW"));
        h.close_key = reinterpret_cast<decltype(h.close_key)>(GetProcAddress(m, "RegCloseKey"));
    }
    return h;
}

const HostRegistry& host() {
    if (const HostRegistry* t = g_test_host.load()) return *t;
    static const HostRegistry real = load_advapi32();  // thread-safe; a failed load leaves null pointers
    return real;
}

bool equals_ignore_case(LPCWSTR a, const wchar_t* b) {
    return a && CompareStringOrdinal(a, -1, b, -1, TRUE) == CSTR_EQUAL;
}

// Whether the title asks for our key: HKLM and the OEM path (case-insensitive, as the registry is), and only while the
// configured device type has a ConsoleMode to give.
bool is_ours(HKEY key, LPCWSTR sub_key) {
    DWORD mode;
    return key == kLocalMachine && equals_ignore_case(sub_key, kOemKey) &&
           console_identity::console_mode_for(console_identity::configured_device_type(), mode);
}

}  // namespace

HKEY pseudo_key() { return reinterpret_cast<HKEY>(&g_oem_key); }

void set_host_registry_for_test(const HostRegistry* h) { g_test_host.store(h); }

LSTATUS WINAPI reg_open_key_w(HKEY key, LPCWSTR sub_key, PHKEY result) {
    if (key == pseudo_key()) {
        // The OEM key has no subkeys; opening it with no subkey path gives the same key, as the registry does.
        if (!result) return ERROR_INVALID_PARAMETER;
        if (sub_key && *sub_key) return ERROR_FILE_NOT_FOUND;
        *result = pseudo_key();
        return ERROR_SUCCESS;
    }
    if (result && is_ours(key, sub_key)) {
        *result = pseudo_key();
        return ERROR_SUCCESS;
    }
    return host().open_key ? host().open_key(key, sub_key, result) : ERROR_CALL_NOT_IMPLEMENTED;
}

LSTATUS WINAPI reg_open_key_ex_w(HKEY key, LPCWSTR sub_key, DWORD options, REGSAM desired, PHKEY result) {
    if (key == pseudo_key()) {
        if (!result) return ERROR_INVALID_PARAMETER;
        if (sub_key && *sub_key) return ERROR_FILE_NOT_FOUND;
        *result = pseudo_key();
        return ERROR_SUCCESS;
    }
    if (result && is_ours(key, sub_key)) {
        *result = pseudo_key();
        return ERROR_SUCCESS;
    }
    return host().open_key_ex ? host().open_key_ex(key, sub_key, options, desired, result)
                              : ERROR_CALL_NOT_IMPLEMENTED;
}

LSTATUS WINAPI reg_query_value_ex_w(HKEY key, LPCWSTR value_name, LPDWORD reserved, LPDWORD type, LPBYTE data,
                                    LPDWORD size) {
    if (key != pseudo_key())
        return host().query_value_ex ? host().query_value_ex(key, value_name, reserved, type, data, size)
                                     : ERROR_CALL_NOT_IMPLEMENTED;

    // Our key holds one value. The default (unnamed) value and every other name do not exist.
    DWORD mode;
    if (!equals_ignore_case(value_name, kConsoleMode) ||
        !console_identity::console_mode_for(console_identity::configured_device_type(), mode))
        return ERROR_FILE_NOT_FOUND;

    if (data && !size) return ERROR_INVALID_PARAMETER;  // a buffer with no size to say how big it is
    if (type) *type = REG_DWORD;
    if (!size) return ERROR_SUCCESS;  // the caller wants only the type (or nothing)

    const DWORD room = *size;
    *size = sizeof(DWORD);
    if (!data) return ERROR_SUCCESS;  // a size query
    if (room < sizeof(DWORD)) return ERROR_MORE_DATA;
    std::memcpy(data, &mode, sizeof(DWORD));  // the buffer may be unaligned
    // Once per distinct answer, so a boot log shows what the title was told without a line per read.
    static std::atomic<DWORD> logged{0};
    if (logged.exchange(mode) != mode)
        std::fprintf(stderr, "[rt] registry: ConsoleMode answered 0x%x (LEONIDA_DEVICE_TYPE=%u)\n",
                     static_cast<unsigned>(mode), console_identity::configured_device_type());
    return ERROR_SUCCESS;
}

LSTATUS WINAPI reg_close_key(HKEY key) {
    if (key == pseudo_key()) return ERROR_SUCCESS;
    return host().close_key ? host().close_key(key) : ERROR_CALL_NOT_IMPLEMENTED;
}

}  // namespace rt::shims::console_registry

#define REG_ENTRIES(dll)                                                                                        \
    {dll, "RegOpenKeyW", reinterpret_cast<void*>(&rt::shims::console_registry::reg_open_key_w)},               \
    {dll, "RegOpenKeyExW", reinterpret_cast<void*>(&rt::shims::console_registry::reg_open_key_ex_w)},          \
    {dll, "RegQueryValueExW", reinterpret_cast<void*>(&rt::shims::console_registry::reg_query_value_ex_w)},    \
    {dll, "RegCloseKey", reinterpret_cast<void*>(&rt::shims::console_registry::reg_close_key)}

// The classic import library and the API sets the registry functions live in.
SHIM_TABLE(console_registry,
    REG_ENTRIES("advapi32.dll"),
    REG_ENTRIES("api-ms-win-core-registry-l1-1-0.dll"),
    REG_ENTRIES("api-ms-win-core-registry-l1-1-1.dll"),
    REG_ENTRIES("api-ms-win-core-registry-l1-1-2.dll"),
    REG_ENTRIES("api-ms-win-core-registry-l2-1-0.dll"),
    REG_ENTRIES("api-ms-win-core-registry-l2-1-1.dll"),
);
