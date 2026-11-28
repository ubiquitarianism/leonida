// The host GameInput path: mapped code that imports GameInputCreate from GameInput.dll, with no GameInput shim.
//
// docs/gameinput.md: without the SDL3 shim, the loader binds GameInputCreate to the host's GameInput.dll. This checks
// that path with a synthetic DLL (test/fixtures/gameinput_host.cpp) mapped by rt::Loader:
//   - its GameInput.dll import binds to the host's copy (this test links no shim table, so it cannot be a shim);
//   - GameInputCreate succeeds from mapped code, and the library answers GetCurrentTimestamp;
//   - GetCurrentReading says either "reading" or "no reading found": no crash, no other error. With a gamepad attached
//     GetGamepadState must succeed on the reading; the test passes with or without one.
// Skips (exit 77) only where Windows itself cannot load GameInput.dll (asked before mapping, with LoadLibraryExW).
// When the DLL is there, an import that binds to nothing is a failure, not a skip. No game files.
//
// Wine has a GameInput.dll of its own that loads and implements nothing: its GameInputCreate answers E_NOTIMPL
// (wine-staging 11.21). There the import binding is still checked, and the calls into the library are skipped.
#include <windows.h>
#include <unknwn.h>

#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "host_wine.h"
#include "pe_loader.h"

namespace {

using ProbeFn = int (*)(long*, unsigned long long*, long*, int*);
constexpr unsigned long kReadingNotFound = 0x838A0003UL;  // GAMEINPUT_E_READING_NOT_FOUND

int g_failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    std::fflush(stdout);
    if (!ok) ++g_failures;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: gameinput_host_test <gameinput_host.dll>\n");
        return 2;
    }
    const std::filesystem::path fixture = argv[1];

    // Decide the skip from the machine, not from the loader's answer: a broken host binding must fail the test.
    HMODULE const host_gameinput = LoadLibraryExW(L"GameInput.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!host_gameinput) {
        const DWORD error = GetLastError();
        if (error == ERROR_MOD_NOT_FOUND || error == ERROR_FILE_NOT_FOUND || error == ERROR_PATH_NOT_FOUND) {
            std::printf("SKIP: no GameInput.dll on this machine\n");
            return 77;
        }
        std::printf("FAIL GameInput.dll is present but does not load (error %lu)\n", static_cast<unsigned long>(error));
        return 1;
    }

    // Asked of the host's DLL directly and before anything is mapped, for the same reason: what is skipped must not
    // depend on the loader's answer. On Windows nothing is asked, and every failure below stays a failure.
    bool host_is_a_stub = false;
    if (hostwine::is_wine()) {
        using CreateFn = HRESULT(WINAPI*)(IUnknown**);
        if (const auto create = reinterpret_cast<CreateFn>(GetProcAddress(host_gameinput, "GameInputCreate"))) {
            IUnknown* input = nullptr;
            host_is_a_stub = create(&input) == E_NOTIMPL;
            if (input) input->Release();
        }
    }

    try {
        rt::Loader loader(fixture.parent_path());
        rt::Module& m = loader.load(fixture);

        // The SDK's import library imports by ordinal, so match the DLL name, not the function name.
        rt::Binding binding = rt::Binding::Unimplemented;
        int seen = 0;
        for (const auto& r : m.imports)
            if (rt::lower(r.dll) == "gameinput.dll") {
                binding = r.binding;
                ++seen;
            }
        check(seen == 1 && binding == rt::Binding::Host, "the GameInput.dll import binds to the host's copy");

        if (host_is_a_stub) {
            std::printf("SKIP the calls into GameInput.dll (Wine's has no implementation: GameInputCreate answers "
                        "E_NOTIMPL)\n");
        } else {
            loader.initialise();
            const auto probe = reinterpret_cast<ProbeFn>(m.exports.at("gameinput_host_probe"));
            long create_hr = -1, reading_hr = -1;
            unsigned long long timestamp = 0;
            int state_ok = 0;
            const int created = probe(&create_hr, &timestamp, &reading_hr, &state_ok);
            std::printf("     create hr=0x%08lX, timestamp=%llu, reading hr=0x%08lX\n",
                        static_cast<unsigned long>(create_hr), timestamp, static_cast<unsigned long>(reading_hr));
            check(created == 1 && create_hr == 0, "GameInputCreate succeeds from mapped code");
            check(timestamp != 0, "GetCurrentTimestamp returns a time");
            const bool pad = created == 1 && reading_hr == 0;
            check(pad || static_cast<unsigned long>(reading_hr) == kReadingNotFound,
                  "GetCurrentReading says either 'reading' or 'no reading found'");
            std::printf("     gamepad reading: %s\n",
                        pad ? "yes (a gamepad is attached)" : "none (no gamepad attached)");
            if (pad) check(state_ok == 1, "GetGamepadState succeeds on a real reading");
        }
    } catch (const std::exception& e) {
        std::printf("FAIL unexpected exception: %s\n", e.what());
        ++g_failures;
    }
    FreeLibrary(host_gameinput);
    std::printf("gameinput_host_test: %s (%d failures)\n", g_failures ? "FAILED" : "OK", g_failures);
    return g_failures ? 1 : 0;
}
