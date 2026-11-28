// chad.dll: the console's audio DSP command queue. No PC has one, so this gives the title a connection whose calls
// succeed and zeroed DSP status: nothing the title submits is processed. xapu.dll, the APU command queue, is in
// xapu_queue.cpp.
//
// Read from the executable's call sites (counts in docs/runtime_imports.md), statically, never run:
//   XDspConnectWithMaximumStreamLimit(buffer, u32, u32, 0x80, 1, handle*)   1 site; success is a return of 0. The
//                                             buffer comes from the title's XMemAlloc and stays the title's.
//   XDspActivate(handle, desc*, out*, out*)   1 site; success is a return of 0, else the caller returns -1. The two
//                                             outputs are fields of the caller's object. The title later reads
//                                             dwords at +4 and +0xc through one of them with no null check, and
//                                             passes the other to XDspSubmitCommand.
//   XDspSubmitCommand(stream, cmd*, id*)      1 site; success is a return of 0. The title zeroes *id before the call
//                                             and afterwards waits until the status dword at +0xc is 0 and the one
//                                             at +4 equals *id.
//   XDspDeactivate, XDspDisconnect            1 site each, tail calls: the caller returns what they return.
//
// HYPOTHESIS: XDspActivate's outputs point at zeroed memory owned by the connection. XDspSubmitCommand leaves *id at
// the title's 0, so the zeroed status reads as "command 0 done, no error" and the title does not wait.
#include <windows.h>
#include <intrin.h>

#include <cstdint>
#include <cstring>
#include <mutex>
#include <new>
#include <set>
#include <string>

#include "shims.h"
#include "startup_trace.h"

namespace {

struct DspConnection {
    uint8_t status[256];
    uint8_t stream[256];
};

// gta6rt --trace FILE: the first call of each function, with the caller, so the first boot shows what the title sends
// without a line per call.
void note(const char* api, uint64_t value, const void* caller) {
    if (!rt::startup_trace::enabled()) return;
    static std::mutex mutex;
    static std::set<std::pair<std::string, uint64_t>> seen;
    {
        const std::lock_guard lock(mutex);
        if (!seen.insert({api, value}).second) return;
    }
    rt::startup_trace::event(
        "audio_queue", std::string("\"api\":") + rt::startup_trace::quote(api) +
                           ",\"value\":" + rt::startup_trace::quote(rt::startup_trace::hex(value)) + ",\"caller\":" +
                           rt::startup_trace::quote(rt::startup_trace::where(reinterpret_cast<uint64_t>(caller))));
}

HRESULT __stdcall XDspConnectWithMaximumStreamLimit(void* buffer, uint32_t a, uint32_t b, uint32_t max_streams,
                                                    uint32_t flags, void** handle) {
    note("XDspConnectWithMaximumStreamLimit", max_streams, _ReturnAddress());
    (void)buffer;
    (void)a;
    (void)b;
    (void)flags;
    if (!handle) return E_INVALIDARG;
    *handle = new (std::nothrow) DspConnection{};
    return *handle ? S_OK : E_OUTOFMEMORY;
}

HRESULT __stdcall XDspDisconnect(void* handle) {
    note("XDspDisconnect", 0, _ReturnAddress());
    delete static_cast<DspConnection*>(handle);
    return S_OK;
}

HRESULT __stdcall XDspActivate(void* handle, const void* description, void** status, void** stream) {
    note("XDspActivate", 0, _ReturnAddress());
    (void)description;
    auto* c = static_cast<DspConnection*>(handle);
    if (!c) return E_INVALIDARG;
    if (status) *status = c->status;
    if (stream) *stream = c->stream;
    return S_OK;
}

HRESULT __stdcall XDspDeactivate(void* handle, uint64_t, uint64_t, uint64_t) {
    note("XDspDeactivate", 0, _ReturnAddress());
    (void)handle;
    return S_OK;
}

// *id stays the 0 the title wrote, which the zeroed status block already reports as done.
HRESULT __stdcall XDspSubmitCommand(void*, void*, uint32_t*) {
    note("XDspSubmitCommand", 0, _ReturnAddress());
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
