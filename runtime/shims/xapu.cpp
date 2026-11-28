// xapu.dll: Audio Processing Unit (XAPU). Hardware-accelerated Opus decoding and
// sample-rate conversion on Xbox Series.
//
// On a PC there is no such hardware. This shim accepts connections and commands,
// and reports results as if they completed immediately with silence/identity.
//
// Measured from the call sites in the executable, read statically (shapes and
// argument counts only; see docs/audio_hal_exe_calls.md and PR #6981):
//   XApuConnect(void* out24, void* out16, void** handle)      3 args
//   XApuEnqueueCommand(void* handle, const uint32_t* cmd, u64) 3 args
//   XApuGetResultCount(void* handle, u64)                      2 args
//   XApuDequeueResult(void* handle, void* out)                 2 args
//   XApuDisconnect(void* handle)                               1 arg
//
// The 24-byte output is zeroed, the 16-byte output holds {base, size} of a
// shared buffer (0x14000 + 128 * 0x1E00 bytes), commands are accepted and
// dropped, the result queue stays empty.

#include <windows.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <string>

#include "shims.h"
#include "startup_trace.h"

namespace {

constexpr size_t kApuBufferSize = 0x14000 + 128 * 0x1E00;
constexpr HRESULT XAPU_E_PENDING_RESULTS = 0x8004E801;

// ---- Opaque handle ----

struct XApuClient {
    void* buffer;
};

// ---- Startup trace helper ----

void trace_call(const char* api, const std::string& detail) {
    if (!rt::startup_trace::enabled()) return;
    static std::mutex mutex;
    static std::set<std::string> seen;
    {
        std::lock_guard lock(mutex);
        if (!seen.insert(std::string(api) + detail).second) return;
    }
    rt::startup_trace::event("xapu",
                             std::string("\"api\":\"") + api + "\"" + detail);
}

// ---- XApuConnect ----
//
// Measured: XApuConnect(void* out24, void* out16, void** handle)
//   out24[0..23] is zeroed. out16[0..1] = {buffer_base, buffer_size}.
//   The handle is an opaque pointer.
//   Success is SUCCEEDED(hr).

HRESULT __stdcall XApuConnect(void* out24, void* out16, void** handle) {
    trace_call("XApuConnect", "");
    if (!handle) return E_INVALIDARG;
    *handle = nullptr;

    void* buffer = VirtualAlloc(nullptr, kApuBufferSize,
                                MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!buffer) return E_OUTOFMEMORY;

    if (out24) std::memset(out24, 0, 24);
    if (out16) {
        const uint64_t info[2] = {reinterpret_cast<uint64_t>(buffer), kApuBufferSize};
        std::memcpy(out16, info, sizeof(info));
    }

    *handle = new XApuClient{buffer};
    return S_OK;
}

// ---- XApuDequeueResult ----
//
// Measured: XApuDequeueResult(void* handle, void* result)
//   Without hardware there are no results. Return a negative HRESULT that the
//   caller handles (it stops the dequeue loop).

HRESULT __stdcall XApuDequeueResult(void* handle, void* commandResult) {
    (void)handle;
    (void)commandResult;
    return XAPU_E_PENDING_RESULTS;
}

// ---- XApuDisconnect ----
//
// Measured: XApuDisconnect(void* handle)

HRESULT __stdcall XApuDisconnect(void* handle) {
    trace_call("XApuDisconnect", "");
    if (!handle) return E_INVALIDARG;
    auto* client = static_cast<XApuClient*>(handle);
    VirtualFree(client->buffer, 0, MEM_RELEASE);
    delete client;
    return S_OK;
}

// ---- XApuEnqueueCommand ----
//
// Measured: XApuEnqueueCommand(void* handle, const uint32_t* command, uint64_t flags)
//   The return value is never read. The command's first dword is a type (1, 2, 3 seen),
//   the second a stream slot. The 3rd argument is a flags word (always 0 at the call sites).

HRESULT __stdcall XApuEnqueueCommand(void* handle, const uint32_t* command,
                                      uint64_t flags) {
    (void)flags;
    trace_call("XApuEnqueueCommand",
               ",\"type\":" + std::to_string(command ? command[0] : ~0u));
    return handle && command ? S_OK : E_INVALIDARG;
}

// ---- XApuGetResultCount ----
//
// Measured: XApuGetResultCount(void* handle, uint64_t flags)
//   Returns uint32_t: 0 means no results ready.

uint32_t __stdcall XApuGetResultCount(void* handle, uint64_t flags) {
    (void)handle;
    (void)flags;
    return 0;
}

}  // namespace

SHIM_TABLE(xapu,
    {"xapu.dll", "XApuConnect", reinterpret_cast<void*>(&XApuConnect)},
    {"xapu.dll", "XApuDequeueResult", reinterpret_cast<void*>(&XApuDequeueResult)},
    {"xapu.dll", "XApuDisconnect", reinterpret_cast<void*>(&XApuDisconnect)},
    {"xapu.dll", "XApuEnqueueCommand", reinterpret_cast<void*>(&XApuEnqueueCommand)},
    {"xapu.dll", "XApuGetResultCount", reinterpret_cast<void*>(&XApuGetResultCount)},
);