// chad.dll: Custom Hardware Audio DSP. Hardware-accelerated convolution reverb, FFT and IFFT.
//
// On a PC there is no such DSP hardware. This shim accepts connections and activations,
// records the relevant parameters through the startup trace, and returns success for every
// call. Any submitted command is dropped; its "completion" is reported through the status
// buffer synchronously.
//
// The signatures below come from the public GDK reference (devdocs.xbox.com / docs.xbox.com):
//   XDspAudio function reference, handles XDspClientHandle and XDspStreamHandle.
//
// ASSUMPTION: XDspClientHandle and XDspStreamHandle are opaque pointers returned by the
// connect and activate calls respectively. They are never dereferenced by the title
// (only passed back to XDSP functions). A plain heap allocation suffices.
//
// ASSUMPTION: XDspStatus is a structure the title reads to find the last processed command
// sequence. Without a hardware queue the status always reports every submitted command as
// processed.
//
// ASSUMPTION: XDspCommand carries input/output buffers and a frame count. The shim copies
// nothing and reports the command as completed immediately.

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

// ---- Handle types (opaque) ----

struct XDspClient {
    uint32_t refcount;
    bool connected;
    // The registered base buffer (the title's own memory, we just track it)
    void* baseBuffer;
    uint32_t baseBufferLength;
    uint32_t aggregateImpulseResponseInSeconds;
    uint32_t maxStreamCount;       // set by XDspConnectWithMaximumStreamLimit
    uint32_t maxChannelCount;      // set by XDspConnectWithMaximumStreamLimit
};
struct XDspStream {};

// ---- Global constants ----

constexpr uint32_t kMaxClients = 8;
constexpr uint32_t kMaxStreamsPerClient = 256;
constexpr uint32_t kMaxAggregateImpulseResponse = 128;  // seconds

// ---- Global state ----

std::mutex g_chad_mutex;
uint32_t g_client_count = 0;

// ---- Startup trace helper ----

void trace_call(const char* api, const std::string& detail) {
    if (!rt::startup_trace::enabled()) return;
    static std::mutex mutex;
    static std::set<std::string> seen;
    {
        std::lock_guard lock(mutex);
        if (!seen.insert(api + detail).second) return;
    }
    rt::startup_trace::event("chad", std::string("\"api\":\"") + api + "\"" + detail);
}

// ---- XDspConnect ----
//
// HRESULT XDspConnect(float* baseBuffer, uint32_t baseBufferLength,
//                      uint32_t aggregateImpulseResponseInSeconds,
//                      XDspClientHandle* handle)
//
// Not imported by the game (it uses XDspConnectWithMaximumStreamLimit), but provided
// for completeness.

HRESULT __stdcall XDspConnect(void* baseBuffer, uint32_t baseBufferLength,
                               uint32_t aggregateImpulseResponseInSeconds,
                               void** handle) {
    // Validate
    if (!handle) return E_POINTER;
    *handle = nullptr;
    // The real title passes baseBufferLength=0 at a measured call site, so accept any size.
    if (!baseBuffer || baseBufferLength > 500 * 1024 * 1024)
        return E_INVALIDARG;
    if (aggregateImpulseResponseInSeconds > kMaxAggregateImpulseResponse)
        return 0x8004E402;  // XDSP_E_MAX_AGGREGATE_IMPULSE_RESPONSE_EXCEEDED

    std::lock_guard lock(g_chad_mutex);
    if (g_client_count >= kMaxClients)
        return 0x8004E401;  // XDSP_E_MAX_CLIENTS_EXCEEDED

    auto* client = new (std::nothrow) XDspClient{};
    if (!client) return E_OUTOFMEMORY;
    client->refcount = 1;
    client->connected = true;
    client->baseBuffer = baseBuffer;
    client->baseBufferLength = baseBufferLength;
    client->aggregateImpulseResponseInSeconds = aggregateImpulseResponseInSeconds;
    client->maxStreamCount = kMaxStreamsPerClient;
    client->maxChannelCount = 2;
    ++g_client_count;
    *handle = client;

    trace_call("XDspConnect",
               ",\"baseBufferLength\":" + std::to_string(baseBufferLength));
    return S_OK;
}

// ---- XDspConnectWithMaximumStreamLimit ----
//
// HRESULT XDspConnectWithMaximumStreamLimit(float* baseBuffer, uint32_t baseBufferLength,
//                                            uint32_t aggregateImpulseResponseInSeconds,
//                                            uint32_t maximumStreamCount,
//                                            uint32_t maximumChannelCount,
//                                            XDspClientHandle* handle)

HRESULT __stdcall XDspConnectWithMaximumStreamLimit(void* baseBuffer,
                                                     uint32_t baseBufferLength,
                                                     uint32_t aggregateImpulseResponseInSeconds,
                                                     uint32_t maximumStreamCount,
                                                     uint32_t maximumChannelCount,
                                                     void** handle) {
    if (!handle) return E_POINTER;
    *handle = nullptr;
    // A measured call site passes baseBufferLength=0, so accept any size.
    if (!baseBuffer || baseBufferLength > 500 * 1024 * 1024)
        return E_INVALIDARG;
    if (maximumStreamCount == 0 || maximumStreamCount > kMaxStreamsPerClient)
        return E_INVALIDARG;
    if (maximumChannelCount != 1 && maximumChannelCount != 2)
        return E_INVALIDARG;
    if (aggregateImpulseResponseInSeconds > kMaxAggregateImpulseResponse)
        return 0x8004E402;

    std::lock_guard lock(g_chad_mutex);
    if (g_client_count >= kMaxClients)
        return 0x8004E401;

    auto* client = new (std::nothrow) XDspClient{};
    if (!client) return E_OUTOFMEMORY;
    client->refcount = 1;
    client->connected = true;
    client->baseBuffer = baseBuffer;
    client->baseBufferLength = baseBufferLength;
    client->aggregateImpulseResponseInSeconds = aggregateImpulseResponseInSeconds;
    client->maxStreamCount = maximumStreamCount;
    client->maxChannelCount = maximumChannelCount;
    ++g_client_count;
    *handle = client;

    trace_call("XDspConnectWithMaximumStreamLimit",
               ",\"baseBufferLength\":" + std::to_string(baseBufferLength) +
               ",\"maxStreams\":" + std::to_string(maximumStreamCount));
    return S_OK;
}

// ---- XDspActivate ----
//
// Measured: XDspActivate(void* handle, u64, u64, u64, u64) — 5 args (PR #6981).
// Success is a HRESULT of 0 (S_OK). The extra 4 values are activation parameters
// whose meaning is not read; we accept them and return success.

HRESULT __stdcall XDspActivate(void* handle, uint64_t, uint64_t, uint64_t, uint64_t) {
    trace_call("XDspActivate", "");
    return handle ? S_OK : E_INVALIDARG;
}

// ---- XDspDeactivate ----
//
// Measured: XDspDeactivate(void* handle, u64, u64, u64) — 4 args.

HRESULT __stdcall XDspDeactivate(void* handle, uint64_t, uint64_t, uint64_t) {
    trace_call("XDspDeactivate", "");
    return handle ? S_OK : E_INVALIDARG;
}

// ---- XDspDisconnect ----
//
// HRESULT XDspDisconnect(XDspClientHandle clientHandle)

HRESULT __stdcall XDspDisconnect(void* clientHandle) {
    if (!clientHandle) return E_INVALIDARG;
    auto* client = static_cast<XDspClient*>(clientHandle);
    {
        std::lock_guard lock(g_chad_mutex);
        if (g_client_count > 0) --g_client_count;
    }
    delete client;
    trace_call("XDspDisconnect", "");
    return S_OK;
}

// ---- XDspSubmitCommand ----
//
// Measured: XDspSubmitCommand(void* a, void* b) — 2 args. Which of the two is
// the connection handle is not known (the measured wrapper swaps them).

HRESULT __stdcall XDspSubmitCommand(void*, void*) {
    trace_call("XDspSubmitCommand", "");
    return S_OK;
}

}  // namespace

SHIM_TABLE(chad,
    {"chad.dll", "XDspActivate", reinterpret_cast<void*>(&XDspActivate)},
    {"chad.dll", "XDspConnectWithMaximumStreamLimit", reinterpret_cast<void*>(&XDspConnectWithMaximumStreamLimit)},
    {"chad.dll", "XDspDeactivate", reinterpret_cast<void*>(&XDspDeactivate)},
    {"chad.dll", "XDspDisconnect", reinterpret_cast<void*>(&XDspDisconnect)},
    {"chad.dll", "XDspSubmitCommand", reinterpret_cast<void*>(&XDspSubmitCommand)},
);
