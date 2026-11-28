// The registry answer for the console model: HKLM\Software\Microsoft\Windows NT\CurrentVersion\OEM, ConsoleMode.
// See console_registry.cpp. This header is the test seam; the title reaches the shim through rt::shims::find.
#pragma once
#include <windows.h>

namespace rt::shims::console_registry {

// The host registry functions every call that is not ours is forwarded to. The defaults are advapi32's, resolved on
// first use; a test replaces them to see what is forwarded.
struct HostRegistry {
    LSTATUS(WINAPI* open_key)(HKEY, LPCWSTR, PHKEY);
    LSTATUS(WINAPI* open_key_ex)(HKEY, LPCWSTR, DWORD, REGSAM, PHKEY);
    LSTATUS(WINAPI* query_value_ex)(HKEY, LPCWSTR, LPDWORD, LPDWORD, LPBYTE, LPDWORD);
    LSTATUS(WINAPI* close_key)(HKEY);
};
void set_host_registry_for_test(const HostRegistry* host);  // nullptr: advapi32 again

// The pseudo key handed out for the OEM key. Never a host handle value.
HKEY pseudo_key();

LSTATUS WINAPI reg_open_key_w(HKEY key, LPCWSTR sub_key, PHKEY result);
LSTATUS WINAPI reg_open_key_ex_w(HKEY key, LPCWSTR sub_key, DWORD options, REGSAM desired, PHKEY result);
LSTATUS WINAPI reg_query_value_ex_w(HKEY key, LPCWSTR value_name, LPDWORD reserved, LPDWORD type, LPBYTE data,
                                    LPDWORD size);
LSTATUS WINAPI reg_close_key(HKEY key);

}  // namespace rt::shims::console_registry
