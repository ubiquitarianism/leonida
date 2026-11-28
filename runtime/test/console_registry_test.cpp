// Tests for runtime/shims/console_registry.cpp: the title's registry read of the console model.
// The shim is reached the way the loader reaches it, through rt::shims::find, under every DLL name the title's
// modules import the registry functions from.
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

#include "console_identity.h"
#include "console_registry.h"
#include "shims.h"

namespace {

using namespace rt::shims;

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

using OpenFn = LSTATUS(WINAPI*)(HKEY, LPCWSTR, PHKEY);
using OpenExFn = LSTATUS(WINAPI*)(HKEY, LPCWSTR, DWORD, REGSAM, PHKEY);
using QueryFn = LSTATUS(WINAPI*)(HKEY, LPCWSTR, LPDWORD, LPDWORD, LPBYTE, LPDWORD);
using CloseFn = LSTATUS(WINAPI*)(HKEY);

const wchar_t kOem[] = L"Software\\Microsoft\\Windows NT\\CurrentVersion\\OEM";
const wchar_t kCurrentVersion[] = L"Software\\Microsoft\\Windows NT\\CurrentVersion";

// What the title's import binds to: the shim registered for advapi32.dll.
OpenFn open_w() { return reinterpret_cast<OpenFn>(find("advapi32.dll", "RegOpenKeyW")); }
OpenExFn open_ex_w() { return reinterpret_cast<OpenExFn>(find("advapi32.dll", "RegOpenKeyExW")); }
QueryFn query_w() { return reinterpret_cast<QueryFn>(find("advapi32.dll", "RegQueryValueExW")); }
CloseFn close_k() { return reinterpret_cast<CloseFn>(find("advapi32.dll", "RegCloseKey")); }

void set_device_type(const char* text) { SetEnvironmentVariableA("LEONIDA_DEVICE_TYPE", text); }

// Opens our key the way the title does and reads ConsoleMode into `mode`; returns the query's status.
LSTATUS read_console_mode(DWORD& mode) {
    HKEY key = nullptr;
    if (open_w()(HKEY_LOCAL_MACHINE, kOem, &key) != ERROR_SUCCESS) return ERROR_PATH_NOT_FOUND;
    DWORD type = 0, size = sizeof(mode);
    mode = 0;
    const LSTATUS r = query_w()(key, L"ConsoleMode", nullptr, &type, reinterpret_cast<LPBYTE>(&mode), &size);
    close_k()(key);
    return r == ERROR_SUCCESS && (type != REG_DWORD || size != sizeof(DWORD)) ? ERROR_BAD_FORMAT : r;
}

// A host registry that records what it was asked and answers a status no real registry returns.
constexpr LSTATUS kFromHost = 0x5A5A;
struct Calls {
    int open = 0, open_ex = 0, query = 0, close = 0;
    HKEY last_key = nullptr;
    LPCWSTR last_sub = nullptr;
    LPCWSTR last_name = nullptr;
    DWORD last_options = 0, last_desired = 0;
    LPBYTE last_data = nullptr;
    LPDWORD last_size = nullptr;
} g_calls;

LSTATUS WINAPI fake_open(HKEY k, LPCWSTR sub, PHKEY) {
    ++g_calls.open;
    g_calls.last_key = k;
    g_calls.last_sub = sub;
    return kFromHost;
}
LSTATUS WINAPI fake_open_ex(HKEY k, LPCWSTR sub, DWORD options, REGSAM desired, PHKEY) {
    ++g_calls.open_ex;
    g_calls.last_key = k;
    g_calls.last_sub = sub;
    g_calls.last_options = options;
    g_calls.last_desired = desired;
    return kFromHost;
}
LSTATUS WINAPI fake_query(HKEY k, LPCWSTR name, LPDWORD, LPDWORD, LPBYTE data, LPDWORD size) {
    ++g_calls.query;
    g_calls.last_key = k;
    g_calls.last_name = name;
    g_calls.last_data = data;
    g_calls.last_size = size;
    return kFromHost;
}
LSTATUS WINAPI fake_close(HKEY k) {
    ++g_calls.close;
    g_calls.last_key = k;
    return kFromHost;
}
const console_registry::HostRegistry kFakeHost{fake_open, fake_open_ex, fake_query, fake_close};

void test_registered_under_every_name() {
    const char* const dlls[] = {"advapi32.dll",
                                "api-ms-win-core-registry-l1-1-0.dll", "api-ms-win-core-registry-l1-1-1.dll",
                                "api-ms-win-core-registry-l1-1-2.dll", "api-ms-win-core-registry-l2-1-0.dll",
                                "api-ms-win-core-registry-l2-1-1.dll"};
    const char* const names[] = {"RegOpenKeyW", "RegOpenKeyExW", "RegQueryValueExW", "RegCloseKey"};
    bool all = true;
    for (const char* d : dlls)
        for (const char* n : names) all = all && find(d, n) != nullptr;
    check(all, "the four registry functions resolve under advapi32 and the registry API sets");
    check(find("advapi32.dll", "RegOpenKeyW") == find("api-ms-win-core-registry-l2-1-0.dll", "RegOpenKeyW"),
          "an API set and advapi32 bind the same shim");
    check(find("advapi32.dll", "RegSetValueExW") == nullptr, "other advapi32 functions are not shimmed");
    check(!owns_dll("advapi32.dll") && !owns_dll("api-ms-win-core-registry-l1-1-0.dll"),
          "advapi32 and the API sets stay host modules (no pseudo-handle for LoadLibrary)");
}

void test_our_key_for_each_device_type() {
    struct Case { const char* env; DWORD mode; const char* what; } cases[] = {
        {"6", 0x60, "device type 6 (Series S) -> ConsoleMode 0x60"},
        {"7", 0x64, "device type 7 (Series X) -> ConsoleMode 0x64"},
        {"8", 0x68, "device type 8 (devkit) -> ConsoleMode 0x68"},
        {nullptr, 0x64, "no setting -> Series X, ConsoleMode 0x64 (as XSystemGetDeviceType)"},
        {"99", 0x64, "a device type out of range is ignored -> 0x64"},
    };
    for (const auto& c : cases) {
        set_device_type(c.env);
        DWORD mode = 0;
        const LSTATUS r = read_console_mode(mode);
        check(r == ERROR_SUCCESS && mode == c.mode, c.what);
    }
    // The same setting drives XSystemGetDeviceType: ConsoleMode for 6 and 7 is the one the device type names.
    for (uint32_t t = 0; t <= 8; ++t) {
        char text[4];
        std::snprintf(text, sizeof text, "%u", t);
        set_device_type(text);
        DWORD mode = 0;
        const bool has = console_identity::console_mode_for(console_identity::configured_device_type(), mode);
        check(console_identity::configured_device_type() == t && has == (t >= 6),
              "configured_device_type follows LEONIDA_DEVICE_TYPE; only the Series types have a ConsoleMode");
    }
    set_device_type(nullptr);
}

void test_pc_and_xbox_one_are_not_ours() {
    console_registry::set_host_registry_for_test(&kFakeHost);
    for (const char* env : {"0", "1", "2", "5"}) {
        set_device_type(env);
        g_calls = {};
        HKEY key = nullptr;
        const LSTATUS r = open_w()(HKEY_LOCAL_MACHINE, kOem, &key);
        check(r == kFromHost && g_calls.open == 1 && key == nullptr,
              "a device type without a ConsoleMode leaves the OEM key to the host registry");
    }
    set_device_type(nullptr);
    console_registry::set_host_registry_for_test(nullptr);
}

void test_size_query_and_type() {
    set_device_type("7");
    HKEY key = nullptr;
    check(open_ex_w()(HKEY_LOCAL_MACHINE, kOem, 0, KEY_READ, &key) == ERROR_SUCCESS &&
              key == console_registry::pseudo_key(),
          "RegOpenKeyExW hands out the pseudo key");

    DWORD type = 0, size = 0;
    check(query_w()(key, L"ConsoleMode", nullptr, &type, nullptr, &size) == ERROR_SUCCESS && type == REG_DWORD &&
              size == sizeof(DWORD),
          "a size query (null data) reports 4 bytes and REG_DWORD");

    size = 0xFFFF;  // the size query ignores what was in there
    check(query_w()(key, L"ConsoleMode", nullptr, nullptr, nullptr, &size) == ERROR_SUCCESS && size == 4,
          "a size query with no type pointer works");

    type = 0;
    check(query_w()(key, L"ConsoleMode", nullptr, &type, nullptr, nullptr) == ERROR_SUCCESS && type == REG_DWORD,
          "null data and null size report just the type");
    check(query_w()(key, L"ConsoleMode", nullptr, nullptr, nullptr, nullptr) == ERROR_SUCCESS,
          "everything null is a plain existence check");

    // Room for more than a DWORD: the size comes back as 4.
    uint8_t big[8];
    std::memset(big, 0xEE, sizeof big);
    size = sizeof big;
    check(query_w()(key, L"ConsoleMode", nullptr, &type, big, &size) == ERROR_SUCCESS && size == 4 &&
              big[0] == 0x64 && big[1] == 0 && big[2] == 0 && big[3] == 0 && big[4] == 0xEE,
          "a larger buffer gets exactly 4 bytes and the size comes back as 4");

    // An unaligned buffer.
    uint8_t raw[8] = {};
    size = 4;
    check(query_w()(key, L"ConsoleMode", nullptr, &type, raw + 1, &size) == ERROR_SUCCESS && raw[1] == 0x64 &&
              raw[0] == 0 && raw[5] == 0,
          "an unaligned buffer is written byte-wise");

    check(close_k()(key) == ERROR_SUCCESS, "RegCloseKey on the pseudo key succeeds");
    set_device_type(nullptr);
}

void test_more_data() {
    set_device_type("6");
    HKEY key = nullptr;
    check(open_w()(HKEY_LOCAL_MACHINE, kOem, &key) == ERROR_SUCCESS, "open the OEM key (Series S)");
    for (DWORD room : {0u, 1u, 2u, 3u}) {
        uint8_t buf[4] = {0xEE, 0xEE, 0xEE, 0xEE};
        DWORD size = room, type = 0;
        const LSTATUS r = query_w()(key, L"ConsoleMode", nullptr, &type, buf, &size);
        check(r == ERROR_MORE_DATA && size == 4 && type == REG_DWORD && buf[0] == 0xEE && buf[1] == 0xEE &&
                  buf[2] == 0xEE && buf[3] == 0xEE,
              "a short buffer returns ERROR_MORE_DATA, the needed size and the type, and writes nothing");
    }
    uint8_t buf[4] = {};
    DWORD size = 4;
    check(query_w()(key, L"ConsoleMode", nullptr, nullptr, buf, &size) == ERROR_SUCCESS && buf[0] == 0x60,
          "a buffer of exactly 4 bytes is enough (and Series S gives 0x60)");
    close_k()(key);
    set_device_type(nullptr);
}

void test_bad_arguments_and_names() {
    set_device_type("7");
    HKEY key = nullptr;
    open_w()(HKEY_LOCAL_MACHINE, kOem, &key);
    uint8_t buf[4];
    DWORD type = 0;
    check(query_w()(key, L"ConsoleMode", nullptr, &type, buf, nullptr) == ERROR_INVALID_PARAMETER,
          "a data buffer with no size pointer is ERROR_INVALID_PARAMETER");
    DWORD size = 4;
    check(query_w()(key, L"consolemode", nullptr, &type, buf, &size) == ERROR_SUCCESS && buf[0] == 0x64,
          "the value name is case-insensitive");
    size = 4;
    check(query_w()(key, L"Other", nullptr, &type, buf, &size) == ERROR_FILE_NOT_FOUND, "another value does not exist");
    check(query_w()(key, nullptr, nullptr, &type, buf, &size) == ERROR_FILE_NOT_FOUND,
          "the default value does not exist");
    check(query_w()(key, L"", nullptr, &type, buf, &size) == ERROR_FILE_NOT_FOUND, "an empty value name does not exist");

    HKEY sub = nullptr;
    check(open_w()(key, L"Sub", &sub) == ERROR_FILE_NOT_FOUND && open_ex_w()(key, L"Sub", 0, KEY_READ, &sub) ==
                                                                      ERROR_FILE_NOT_FOUND,
          "the pseudo key has no subkeys");
    check(open_w()(key, nullptr, &sub) == ERROR_SUCCESS && sub == console_registry::pseudo_key(),
          "opening the pseudo key with no subkey path gives the key again");

    HKEY again = nullptr;
    const wchar_t mixed[] = L"software\\MICROSOFT\\windows nt\\CurrentVersion\\oem";
    check(open_w()(HKEY_LOCAL_MACHINE, mixed, &again) == ERROR_SUCCESS && again == console_registry::pseudo_key(),
          "the key path is case-insensitive");
    check(close_k()(key) == ERROR_SUCCESS && close_k()(key) == ERROR_SUCCESS, "closing the pseudo key twice is harmless");
    set_device_type(nullptr);
}

void test_other_keys_fall_through() {
    set_device_type("7");
    console_registry::set_host_registry_for_test(&kFakeHost);
    HKEY key = nullptr;

    g_calls = {};
    check(open_w()(HKEY_LOCAL_MACHINE, kCurrentVersion, &key) == kFromHost && g_calls.open == 1 &&
              g_calls.last_key == HKEY_LOCAL_MACHINE && g_calls.last_sub == kCurrentVersion,
          "RegOpenKeyW of another HKLM key reaches the host with its arguments");
    g_calls = {};
    check(open_w()(HKEY_CURRENT_USER, kOem, &key) == kFromHost && g_calls.open == 1 &&
              g_calls.last_key == HKEY_CURRENT_USER,
          "the OEM path under HKCU is not ours");
    g_calls = {};
    check(open_w()(HKEY_LOCAL_MACHINE, L"Software\\Microsoft\\Windows NT\\CurrentVersion\\OEM\\Sub", &key) == kFromHost,
          "a subkey of the OEM path is not ours");
    check(open_w()(HKEY_LOCAL_MACHINE, nullptr, &key) == kFromHost, "HKLM with no subkey is not ours");
    g_calls = {};
    check(open_ex_w()(HKEY_LOCAL_MACHINE, kCurrentVersion, 3, KEY_QUERY_VALUE | KEY_WOW64_64KEY, &key) == kFromHost &&
              g_calls.open_ex == 1 && g_calls.last_options == 3 &&
              g_calls.last_desired == (KEY_QUERY_VALUE | KEY_WOW64_64KEY) && g_calls.last_sub == kCurrentVersion,
          "RegOpenKeyExW of another key reaches the host with options and access");
    check(open_w()(HKEY_LOCAL_MACHINE, kOem, nullptr) == kFromHost,
          "the OEM key with a null result pointer is left to the host to reject");

    // A value query on a host key, even for the name ConsoleMode: forwarded unchanged.
    HKEY host_key = reinterpret_cast<HKEY>(static_cast<uintptr_t>(0x1234));
    uint8_t buf[4];
    DWORD size = 4, type = 0;
    g_calls = {};
    check(query_w()(host_key, L"ConsoleMode", nullptr, &type, buf, &size) == kFromHost && g_calls.query == 1 &&
              g_calls.last_key == host_key && g_calls.last_data == buf && g_calls.last_size == &size,
          "RegQueryValueExW on a host key is forwarded, even for the name ConsoleMode");
    g_calls = {};
    check(close_k()(host_key) == kFromHost && g_calls.close == 1 && g_calls.last_key == host_key,
          "RegCloseKey on a host key is forwarded");
    g_calls = {};
    check(close_k()(HKEY_LOCAL_MACHINE) == kFromHost && g_calls.close == 1, "a predefined key is closed by the host");

    // Our own calls never reach the host.
    g_calls = {};
    HKEY ours = nullptr;
    DWORD mode = 0;
    size = 4;
    check(open_w()(HKEY_LOCAL_MACHINE, kOem, &ours) == ERROR_SUCCESS &&
              query_w()(ours, L"ConsoleMode", nullptr, &type, reinterpret_cast<LPBYTE>(&mode), &size) == ERROR_SUCCESS &&
              close_k()(ours) == ERROR_SUCCESS && mode == 0x64 &&
              g_calls.open + g_calls.open_ex + g_calls.query + g_calls.close == 0,
          "the OEM key's open, query and close never reach the host");

    console_registry::set_host_registry_for_test(nullptr);
    set_device_type(nullptr);
}

// The real advapi32 behind the shim (a Windows host, Wine): another key opens, reads and closes as it does directly.
void test_real_registry_behind_the_shim() {
    set_device_type("7");
    HKEY key = nullptr;
    const LSTATUS r = open_w()(HKEY_LOCAL_MACHINE, kCurrentVersion, &key);
    check(r == ERROR_SUCCESS && key != nullptr && key != console_registry::pseudo_key(),
          "the real HKLM\\...\\CurrentVersion opens through the shim as a host key");
    if (r == ERROR_SUCCESS) {
        DWORD size = 0, type = 0;
        const LSTATUS shim = query_w()(key, L"ProductName", nullptr, &type, nullptr, &size);
        DWORD size2 = 0, type2 = 0;
        const LSTATUS real = RegQueryValueExW(key, L"ProductName", nullptr, &type2, nullptr, &size2);
        check(shim == real && size == size2 && type == type2, "a value of a host key reads the same through the shim");
        check(close_k()(key) == ERROR_SUCCESS, "the host key closes through the shim");
    }

    // The OEM key of a PC-class device type: whatever the host registry says, byte for byte.
    set_device_type("1");
    HKEY a = nullptr, b = nullptr;
    const LSTATUS shim = open_ex_w()(HKEY_LOCAL_MACHINE, kOem, 0, KEY_READ, &a);
    const LSTATUS real = RegOpenKeyExW(HKEY_LOCAL_MACHINE, kOem, 0, KEY_READ, &b);
    check(shim == real, "with a PC device type the OEM key is the host's: same status as a direct call");
    if (shim == ERROR_SUCCESS) close_k()(a);
    if (real == ERROR_SUCCESS) RegCloseKey(b);
    set_device_type(nullptr);
}

}  // namespace

int main() {
    test_registered_under_every_name();
    test_our_key_for_each_device_type();
    test_pc_and_xbox_one_are_not_ours();
    test_size_query_and_type();
    test_more_data();
    test_bad_arguments_and_names();
    test_other_keys_fall_through();
    test_real_registry_behind_the_shim();
    std::printf("%d check(s), %d failure(s)\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
