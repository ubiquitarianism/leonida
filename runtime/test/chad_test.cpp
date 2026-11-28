// Checks the registered chad.dll shim against the measured call shapes
// (docs/audio_hal_exe_calls.md, PR #6981).
//
// Every export is resolved through rt::shims::find so the test exercises
// the same shim table the runtime installs. The measured call shapes:
// XDspConnectWithMaximumStreamLimit(buffer, u32, u32, max_streams, flags, handle**),
// XDspActivate(handle, ...), XDspSubmitCommand(a, b), XDspDeactivate(handle, ...),
// XDspDisconnect(handle).
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

#include "shims.h"

namespace {

using XDspConnectFn = HRESULT(__stdcall*)(void*, uint32_t, uint32_t, uint32_t, uint32_t, void**);
using XDspActivateFn = HRESULT(__stdcall*)(void*, uint64_t, uint64_t, uint64_t, uint64_t);
using XDspDeactivateFn = HRESULT(__stdcall*)(void*, uint64_t, uint64_t, uint64_t);
using XDspDisconnectFn = HRESULT(__stdcall*)(void*);
using XDspSubmitFn = HRESULT(__stdcall*)(void*, void*);

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

}  // namespace

int main() {
    std::printf("chad_test: registered chad.dll shim\n");

    const auto dsp_connect = reinterpret_cast<XDspConnectFn>(
        rt::shims::find("chad.dll", "XDspConnectWithMaximumStreamLimit"));
    const auto dsp_activate = reinterpret_cast<XDspActivateFn>(
        rt::shims::find("chad.dll", "XDspActivate"));
    const auto dsp_deactivate = reinterpret_cast<XDspDeactivateFn>(
        rt::shims::find("chad.dll", "XDspDeactivate"));
    const auto dsp_disconnect = reinterpret_cast<XDspDisconnectFn>(
        rt::shims::find("chad.dll", "XDspDisconnect"));
    const auto dsp_submit = reinterpret_cast<XDspSubmitFn>(
        rt::shims::find("chad.dll", "XDspSubmitCommand"));

    check(dsp_connect != nullptr, "XDspConnectWithMaximumStreamLimit is registered");
    check(dsp_activate != nullptr, "XDspActivate is registered");
    check(dsp_deactivate != nullptr, "XDspDeactivate is registered");
    check(dsp_disconnect != nullptr, "XDspDisconnect is registered");
    check(dsp_submit != nullptr, "XDspSubmitCommand is registered");
    check(rt::shims::owns_dll("chad.dll"), "the registry owns chad.dll");
    if (!dsp_connect || !dsp_activate || !dsp_deactivate || !dsp_disconnect || !dsp_submit) return 1;

    // XDspConnectWithMaximumStreamLimit: measured 6 args
    // (buffer, baseBufferLength, aggImpulseResp, max_streams, max_channels, handle**).
    // The shim validates baseBufferLength >= 0x10000, max_streams >= 1, etc.
    // The title passes 0x80 (128) streams and 1 channel.
    alignas(64) static uint8_t buffer[0x10000];
    void* handle = nullptr;
    const HRESULT cr = dsp_connect(buffer, 0x10000, 0, 0x80, 1, &handle);
    check(cr == S_OK && handle != nullptr,
          "XDspConnectWithMaximumStreamLimit returns S_OK and a handle");

    // XDspActivate(handle, ...): 5 args, success is a return of 0.
    check(dsp_activate(handle, 0, 0, 0, 0) == S_OK,
          "XDspActivate returns S_OK");

    // XDspSubmitCommand(a, b): 2 args, which of the two is the connection is unknown.
    check(dsp_submit(handle, buffer) == S_OK,
          "XDspSubmitCommand returns S_OK");

    // XDspDeactivate(handle, ...): 4 args, success is a return of 0.
    check(dsp_deactivate(handle, 0, 0, 0) == S_OK,
          "XDspDeactivate returns S_OK");

    // XDspDisconnect(handle): 1 arg, success is a return of 0.
    check(dsp_disconnect(handle) == S_OK,
          "XDspDisconnect returns S_OK");

    // Null handle returns E_POINTER.
    check(dsp_connect(buffer, 0x10000, 0, 0x80, 1, nullptr) == E_POINTER,
          "XDspConnectWithMaximumStreamLimit(nullptr handle) returns E_POINTER");

    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}