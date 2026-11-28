// Checks the registered xapu.dll shim against the measured call shapes
// (docs/audio_hal_exe_calls.md, PR #6981).
//
// Every export is resolved through rt::shims::find so the test exercises
// the same shim table the runtime installs. The measured call shapes:
// XApuConnect(out24*, out16*, handle**), XApuEnqueueCommand(handle, cmd*, 0),
// XApuGetResultCount(handle, 0), XApuDequeueResult(handle, out*),
// XApuDisconnect(handle).
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

#include "shims.h"

namespace {

using XApuConnectFn = HRESULT(__stdcall*)(void*, void*, void**);
using XApuDisconnectFn = HRESULT(__stdcall*)(void*);
using XApuEnqueueCommandFn = HRESULT(__stdcall*)(void*, const uint32_t*, uint64_t);
using XApuGetResultCountFn = uint32_t(__stdcall*)(void*, uint64_t);
using XApuDequeueResultFn = HRESULT(__stdcall*)(void*, void*);

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

}  // namespace

int main() {
    std::printf("xapu_test: registered xapu.dll shim\n");

    const auto connect = reinterpret_cast<XApuConnectFn>(rt::shims::find("xapu.dll", "XApuConnect"));
    const auto disconnect = reinterpret_cast<XApuDisconnectFn>(rt::shims::find("xapu.dll", "XApuDisconnect"));
    const auto enqueue = reinterpret_cast<XApuEnqueueCommandFn>(rt::shims::find("xapu.dll", "XApuEnqueueCommand"));
    const auto count = reinterpret_cast<XApuGetResultCountFn>(rt::shims::find("xapu.dll", "XApuGetResultCount"));
    const auto dequeue = reinterpret_cast<XApuDequeueResultFn>(rt::shims::find("xapu.dll", "XApuDequeueResult"));

    check(connect != nullptr, "XApuConnect is registered");
    check(disconnect != nullptr, "XApuDisconnect is registered");
    check(enqueue != nullptr, "XApuEnqueueCommand is registered");
    check(count != nullptr, "XApuGetResultCount is registered");
    check(dequeue != nullptr, "XApuDequeueResult is registered");
    check(rt::shims::owns_dll("xapu.dll"), "the registry owns xapu.dll");
    if (!connect || !disconnect || !enqueue || !count || !dequeue) return 1;

    // XApuConnect: measured (out24*, out16*, handle**) with 3 args.
    // Returns a handle and fills out16[0..1] with {buffer, size}.
    // out24 is zeroed.
    uint8_t out24[24];
    uint64_t out16[2] = {};
    std::memset(out24, 0xcc, sizeof(out24));
    void* handle = nullptr;
    check(SUCCEEDED(connect(out24, out16, &handle)) && handle != nullptr,
          "XApuConnect succeeds with a handle");
    check(out24[0] == 0 && out24[23] == 0,
          "XApuConnect zeroes the 24-byte output");
    check(connect(out24, out16, nullptr) == E_INVALIDARG,
          "XApuConnect(nullptr handle) returns E_INVALIDARG");

    if (!handle) return 1;
    auto* base = reinterpret_cast<uint8_t*>(out16[0]);
    check(base != nullptr && out16[1] >= 0x14000 + 128 * 0x1E00,
          "XApuConnect returns the shared buffer and its size");
    if (base) {
        const size_t last = 0x14000 + 127 * 0x1E00 + 0x1E00 - 1;
        check(base[0] == 0 && base[last] == 0,
              "the shared buffer is zeroed");
        base[127 * 0x280] = 1;
        check(base[127 * 0x280] == 1,
              "the shared buffer is writable at slot 127's input");
    }

    // XApuEnqueueCommand(handle, cmd*, 0): 3 args, the third is flags=0.
    for (uint32_t type = 1; type <= 3; ++type) {
        uint32_t command[16] = {type, 5};
        check(enqueue(handle, command, 0) == S_OK,
              "XApuEnqueueCommand accepts a command");
    }

    // XApuGetResultCount(handle, 0): 2 args, the second is flags=0.
    check(count(handle, 0) == 0,
          "XApuGetResultCount returns 0 (no results)");

    // XApuDequeueResult(handle, out*): 2 args, returns a negative HRESULT.
    uint8_t result[0x50] = {};
    check(FAILED(dequeue(handle, result)),
          "XApuDequeueResult on the empty queue returns a negative HRESULT");

    // XApuDisconnect(handle): 1 arg.
    check(disconnect(handle) == S_OK,
          "XApuDisconnect succeeds");

    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}