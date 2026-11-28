// Tests for runtime/shims/silent_audio.cpp: the silent output endpoint behind CoCreateInstance(MMDeviceEnumerator).
//
// The host is a stand-in enumerator (with no endpoints, or with one), so no host audio device or sound server is ever
// touched. The last case goes through the com_shim CoCreateInstance with LEONIDA_AUDIO=silent, which makes no host
// enumerator for render questions.
#include <windows.h>
#include <objbase.h>
#include <mmreg.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <propsys.h>
#include <spatialaudioclient.h>

#include <cstdio>
#include <cstring>
#include <cwchar>

#include "shims.h"
#include "silent_audio.h"

namespace {

namespace sa = rt::silent_audio;

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

constexpr HRESULT kNotFound = HRESULT_FROM_WIN32(ERROR_NOT_FOUND);
constexpr GUID kSubtypeFloat{0x00000003, 0x0000, 0x0010, {0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71}};
constexpr PROPERTYKEY kKeyFriendlyName{{0xa45c254e, 0xdf1c, 0x4efd, {0x80, 0x20, 0x67, 0xd1, 0x46, 0xa8, 0x50, 0xe0}},
                                       14};
constexpr PROPERTYKEY kKeyDeviceFormat{{0xf19f064d, 0x082c, 0x4e27, {0xbc, 0x73, 0x68, 0x82, 0xa1, 0xbb, 0x8e, 0x4c}},
                                       0};

// ---- stand-in host -------------------------------------------------------------------------------------------

class HostDevice final : public IMMDevice {
public:
    ULONG STDMETHODCALLTYPE AddRef() override { return static_cast<ULONG>(InterlockedIncrement(&refs_)); }
    ULONG STDMETHODCALLTYPE Release() override {
        const LONG n = InterlockedDecrement(&refs_);
        if (n == 0) delete this;
        return static_cast<ULONG>(n);
    }
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppv) override {
        if (!ppv) return E_POINTER;
        *ppv = nullptr;
        if (!IsEqualIID(riid, __uuidof(IUnknown)) && !IsEqualIID(riid, __uuidof(IMMDevice))) return E_NOINTERFACE;
        *ppv = this;
        AddRef();
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE Activate(REFIID, DWORD, PROPVARIANT*, void**) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE OpenPropertyStore(DWORD, IPropertyStore**) override { return E_NOTIMPL; }
    HRESULT STDMETHODCALLTYPE GetId(LPWSTR* id) override {
        const wchar_t text[] = L"host-device";
        *id = static_cast<wchar_t*>(CoTaskMemAlloc(sizeof text));
        std::memcpy(*id, text, sizeof text);
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE GetState(DWORD* state) override {
        *state = DEVICE_STATE_ACTIVE;
        return S_OK;
    }

private:
    LONG refs_ = 1;
};

class HostCollection final : public IMMDeviceCollection {
public:
    explicit HostCollection(bool one) : one_(one) {}
    ULONG STDMETHODCALLTYPE AddRef() override { return static_cast<ULONG>(InterlockedIncrement(&refs_)); }
    ULONG STDMETHODCALLTYPE Release() override {
        const LONG n = InterlockedDecrement(&refs_);
        if (n == 0) delete this;
        return static_cast<ULONG>(n);
    }
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID, void**) override { return E_NOINTERFACE; }
    HRESULT STDMETHODCALLTYPE GetCount(UINT* n) override {
        *n = one_ ? 1 : 0;
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE Item(UINT i, IMMDevice** d) override {
        *d = nullptr;
        if (!one_ || i != 0) return E_INVALIDARG;
        *d = new HostDevice();
        return S_OK;
    }

private:
    LONG refs_ = 1;
    bool one_;
};

// A host enumerator: with no endpoints at all (as a machine without an output), or with one render endpoint. It
// counts what it was asked, per flow.
class HostEnumerator final : public IMMDeviceEnumerator {
public:
    explicit HostEnumerator(bool has_render) : has_render_(has_render) {}
    ULONG STDMETHODCALLTYPE AddRef() override { return static_cast<ULONG>(InterlockedIncrement(&refs_)); }
    ULONG STDMETHODCALLTYPE Release() override {
        const LONG n = InterlockedDecrement(&refs_);
        if (n == 0) delete this;
        return static_cast<ULONG>(n);
    }
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppv) override {
        if (!ppv) return E_POINTER;
        *ppv = nullptr;
        if (!IsEqualIID(riid, __uuidof(IUnknown)) && !IsEqualIID(riid, __uuidof(IMMDeviceEnumerator)))
            return E_NOINTERFACE;
        *ppv = this;
        AddRef();
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE EnumAudioEndpoints(EDataFlow flow, DWORD, IMMDeviceCollection** out) override {
        ++(flow == eCapture ? capture_calls : render_calls);
        *out = new HostCollection(flow == eRender && has_render_);
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE GetDefaultAudioEndpoint(EDataFlow flow, ERole, IMMDevice** out) override {
        ++(flow == eCapture ? capture_calls : render_calls);
        *out = nullptr;
        if (flow != eRender || !has_render_) return kNotFound;
        *out = new HostDevice();
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE GetDevice(LPCWSTR, IMMDevice** out) override {
        ++other_calls;
        *out = nullptr;
        return kNotFound;
    }
    HRESULT STDMETHODCALLTYPE RegisterEndpointNotificationCallback(IMMNotificationClient*) override {
        ++other_calls;
        return S_OK;
    }
    HRESULT STDMETHODCALLTYPE UnregisterEndpointNotificationCallback(IMMNotificationClient*) override {
        ++other_calls;
        return S_OK;
    }
    LONG refs() const { return refs_; }

    int render_calls = 0;
    int capture_calls = 0;
    int other_calls = 0;

private:
    LONG refs_ = 1;
    bool has_render_;
};

HostEnumerator* g_lazy_host = nullptr;
int g_lazy_made = 0;
IMMDeviceEnumerator* lazy_factory() {
    ++g_lazy_made;
    g_lazy_host->AddRef();
    return g_lazy_host;
}

class Notifications final : public IMMNotificationClient {
public:
    ULONG STDMETHODCALLTYPE AddRef() override { return 2; }
    ULONG STDMETHODCALLTYPE Release() override { return 1; }
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID, void**) override { return E_NOINTERFACE; }
    HRESULT STDMETHODCALLTYPE OnDeviceStateChanged(LPCWSTR, DWORD) override { return ++calls, S_OK; }
    HRESULT STDMETHODCALLTYPE OnDeviceAdded(LPCWSTR) override { return ++calls, S_OK; }
    HRESULT STDMETHODCALLTYPE OnDeviceRemoved(LPCWSTR) override { return ++calls, S_OK; }
    HRESULT STDMETHODCALLTYPE OnDefaultDeviceChanged(EDataFlow, ERole, LPCWSTR) override { return ++calls, S_OK; }
    HRESULT STDMETHODCALLTYPE OnPropertyValueChanged(LPCWSTR, const PROPERTYKEY) override { return ++calls, S_OK; }
    int calls = 0;
};

// ---- helpers --------------------------------------------------------------------------------------------------

bool is_silent_device(IMMDevice* d) {
    if (!d) return false;
    LPWSTR id = nullptr;
    const bool ok = SUCCEEDED(d->GetId(&id)) && id && std::wcscmp(id, sa::kDeviceId) == 0;
    CoTaskMemFree(id);
    return ok;
}

IMMDeviceEnumerator* serve(IMMDeviceEnumerator* host, HRESULT host_hr, sa::Mode mode, sa::HostFactory factory) {
    IMMDeviceEnumerator* e = nullptr;
    const HRESULT hr =
        sa::serve_enumerator(host, host_hr, mode, factory, __uuidof(IMMDeviceEnumerator), reinterpret_cast<void**>(&e));
    return SUCCEEDED(hr) ? e : nullptr;
}

UINT collection_count(IMMDeviceEnumerator* e, EDataFlow flow, DWORD mask, bool* silent_first = nullptr) {
    IMMDeviceCollection* c = nullptr;
    if (FAILED(e->EnumAudioEndpoints(flow, mask, &c)) || !c) return 0xFFFFFFFF;
    UINT n = 0;
    c->GetCount(&n);
    if (silent_first) {
        IMMDevice* d = nullptr;
        *silent_first = n > 0 && SUCCEEDED(c->Item(0, &d)) && is_silent_device(d);
        if (d) d->Release();
    }
    c->Release();
    return n;
}

// ---- the enumerator -------------------------------------------------------------------------------------------

void test_host_without_endpoints() {
    auto* host = new HostEnumerator(false);
    IMMDeviceEnumerator* e = serve(host, S_OK, sa::Mode::Auto, nullptr);
    check(e != nullptr && e != host, "no host render endpoint: the wrapper answers, not the host");
    if (!e) return;
    const uint32_t served_before = sa::devices_served();

    IMMDevice* d = nullptr;
    check(e->GetDefaultAudioEndpoint(eRender, eConsole, &d) == S_OK && is_silent_device(d),
          "GetDefaultAudioEndpoint(eRender, eConsole) is the silent device");
    check(host->render_calls >= 1, "the host was asked first");
    check(sa::devices_served() == served_before + 1, "one silent device handed out");
    if (d) {
        DWORD state = 0;
        check(d->GetState(&state) == S_OK && state == DEVICE_STATE_ACTIVE, "the silent device is active");
        IMMEndpoint* ep = nullptr;
        EDataFlow flow = eCapture;
        check(SUCCEEDED(d->QueryInterface(__uuidof(IMMEndpoint), reinterpret_cast<void**>(&ep))) &&
                  ep->GetDataFlow(&flow) == S_OK && flow == eRender,
              "IMMEndpoint says eRender");
        if (ep) ep->Release();

        IPropertyStore* store = nullptr;
        check(d->OpenPropertyStore(STGM_READWRITE, &store) == E_ACCESSDENIED && !store,
              "the property store is read-only");
        check(d->OpenPropertyStore(STGM_READ, &store) == S_OK && store, "OpenPropertyStore(STGM_READ)");
        if (store) {
            PROPVARIANT v;
            PropVariantInit(&v);
            check(store->GetValue(kKeyFriendlyName, &v) == S_OK && v.vt == VT_LPWSTR &&
                      std::wcscmp(v.pwszVal, sa::kFriendlyName) == 0,
                  "PKEY_Device_FriendlyName");
            PropVariantClear(&v);
            check(store->GetValue(kKeyDeviceFormat, &v) == S_OK && v.vt == VT_BLOB &&
                      v.blob.cbSize == sizeof(WAVEFORMATEXTENSIBLE),
                  "PKEY_AudioEngine_DeviceFormat is a WAVEFORMATEXTENSIBLE blob");
            if (v.vt == VT_BLOB) {
                const auto* f = reinterpret_cast<const WAVEFORMATEXTENSIBLE*>(v.blob.pBlobData);
                check(f->Format.nChannels == 8 && f->Format.nSamplesPerSec == 48000 && f->Format.wBitsPerSample == 32 &&
                          f->dwChannelMask == 0x63F && IsEqualGUID(f->SubFormat, kSubtypeFloat),
                      "device format: 48 kHz float 7.1");
            }
            PropVariantClear(&v);
            DWORD n = 0;
            check(store->GetCount(&n) == S_OK && n >= 2, "the store lists its keys");
            store->Release();
        }
        d->Release();
    }

    bool silent_first = false;
    check(collection_count(e, eRender, DEVICE_STATE_ACTIVE, &silent_first) == 1 && silent_first,
          "EnumAudioEndpoints(eRender, ACTIVE) lists the silent device");
    check(collection_count(e, eRender, DEVICE_STATE_DISABLED) == 0,
          "EnumAudioEndpoints(eRender, DISABLED) lists nothing");
    const int capture_before = host->capture_calls;
    check(collection_count(e, eCapture, DEVICE_STATEMASK_ALL) == 0 && host->capture_calls == capture_before + 1,
          "EnumAudioEndpoints(eCapture) is the host's answer");
    check(collection_count(e, eAll, DEVICE_STATE_ACTIVE) == 1, "EnumAudioEndpoints(eAll) has the silent device");
    check(e->GetDefaultAudioEndpoint(eCapture, eConsole, &d) == kNotFound && !d,
          "GetDefaultAudioEndpoint(eCapture) is the host's answer");
    check(e->GetDefaultAudioEndpoint(eAll, eConsole, &d) == E_INVALIDARG, "GetDefaultAudioEndpoint(eAll) is refused");
    check(e->GetDevice(sa::kDeviceId, &d) == S_OK && is_silent_device(d), "GetDevice(silent id)");
    if (d) d->Release();
    d = nullptr;
    check(e->GetDevice(L"host-device", &d) == kNotFound && host->other_calls == 1, "GetDevice(other id) goes to the host");

    Notifications n;
    check(e->RegisterEndpointNotificationCallback(&n) == S_OK, "RegisterEndpointNotificationCallback accepted");
    check(e->UnregisterEndpointNotificationCallback(&n) == S_OK && n.calls == 0,
          "UnregisterEndpointNotificationCallback; the callback never fired");
    check(e->RegisterEndpointNotificationCallback(nullptr) == E_POINTER, "a null callback is E_POINTER");

    e->Release();
    check(host->refs() == 1, "the wrapper released the host");
    host->Release();
}

void test_host_with_endpoint() {
    auto* host = new HostEnumerator(true);
    IMMDeviceEnumerator* e = serve(host, S_OK, sa::Mode::Auto, nullptr);
    check(e == host, "a host with a render endpoint is returned unchanged");
    if (e) e->Release();

    const int render_before = host->render_calls;
    e = serve(host, S_OK, sa::Mode::Silent, nullptr);
    check(e != nullptr && e != host && host->render_calls == render_before,
          "LEONIDA_AUDIO=silent: the wrapper answers without asking the host");
    if (e) {
        IMMDevice* d = nullptr;
        check(e->GetDefaultAudioEndpoint(eRender, eMultimedia, &d) == S_OK && is_silent_device(d) &&
                  host->render_calls == render_before,
              "forced: GetDefaultAudioEndpoint(eRender) is silent, the host's endpoint is not used");
        if (d) d->Release();
        bool silent_first = false;
        check(collection_count(e, eRender, DEVICE_STATE_ACTIVE, &silent_first) == 1 && silent_first &&
                  host->render_calls == render_before,
              "forced: EnumAudioEndpoints(eRender) lists only the silent device");
        e->Release();
    }
    host->Release();
}

void test_no_host_enumerator() {
    IMMDeviceEnumerator* e = serve(nullptr, REGDB_E_CLASSNOTREG, sa::Mode::Auto, nullptr);
    check(e != nullptr, "no host enumerator: the wrapper still serves");
    if (!e) return;
    IMMDevice* d = nullptr;
    check(e->GetDefaultAudioEndpoint(eRender, eConsole, &d) == S_OK && is_silent_device(d),
          "no host enumerator: render is the silent device");
    if (d) d->Release();
    d = nullptr;
    check(e->GetDefaultAudioEndpoint(eCapture, eConsole, &d) == kNotFound && !d,
          "no host enumerator: capture is ERROR_NOT_FOUND");
    check(collection_count(e, eCapture, DEVICE_STATE_ACTIVE) == 0, "no host enumerator: no capture devices");
    e->Release();

    // Forced, the host is made only when a capture question comes.
    g_lazy_host = new HostEnumerator(true);
    e = serve(nullptr, S_OK, sa::Mode::Silent, &lazy_factory);
    if (e) {
        check(e->GetDefaultAudioEndpoint(eRender, eConsole, &d) == S_OK && is_silent_device(d) && g_lazy_made == 0,
              "forced: render makes no host enumerator");
        if (d) d->Release();
        d = nullptr;
        e->GetDefaultAudioEndpoint(eCapture, eConsole, &d);
        check(g_lazy_made == 1 && g_lazy_host->capture_calls == 1, "forced: a capture question makes the host's");
        if (d) d->Release();
        e->Release();
    }
    check(g_lazy_host->refs() == 1, "forced: the lazily made host is released");
    g_lazy_host->Release();
}

// ---- the stream -----------------------------------------------------------------------------------------------

IAudioClient* activate(IMMDevice* d, REFIID riid) {
    void* out = nullptr;
    return SUCCEEDED(d->Activate(riid, CLSCTX_ALL, nullptr, &out)) ? static_cast<IAudioClient*>(out) : nullptr;
}

void test_render_stream() {
    IMMDeviceEnumerator* e = serve(nullptr, REGDB_E_CLASSNOTREG, sa::Mode::Auto, nullptr);
    IMMDevice* d = nullptr;
    if (!e || FAILED(e->GetDefaultAudioEndpoint(eRender, eConsole, &d)) || !d) {
        check(false, "silent device for the stream tests");
        return;
    }
    e->Release();

    for (const IID& iid : {__uuidof(IAudioClient), __uuidof(IAudioClient2), __uuidof(IAudioClient3)}) {
        IAudioClient* c = activate(d, iid);
        check(c != nullptr, "Activate(IAudioClient / 2 / 3)");
        if (c) c->Release();
    }
    void* other = reinterpret_cast<void*>(1);
    check(d->Activate(__uuidof(IMMDeviceEnumerator), CLSCTX_ALL, nullptr, &other) == E_NOINTERFACE && !other,
          "Activate(another interface) is E_NOINTERFACE");

    IAudioClient3* c = nullptr;
    d->Activate(__uuidof(IAudioClient3), CLSCTX_ALL, nullptr, reinterpret_cast<void**>(&c));
    d->Release();
    if (!c) return;

    UINT32 frames = 0;
    check(c->GetBufferSize(&frames) == AUDCLNT_E_NOT_INITIALIZED, "GetBufferSize before Initialize");
    check(c->Start() == AUDCLNT_E_NOT_INITIALIZED, "Start before Initialize");

    WAVEFORMATEX* mix = nullptr;
    check(c->GetMixFormat(&mix) == S_OK && mix && mix->wFormatTag == WAVE_FORMAT_EXTENSIBLE && mix->nChannels == 8 &&
              mix->nSamplesPerSec == 48000 && mix->wBitsPerSample == 32 && mix->nBlockAlign == 32,
          "GetMixFormat: 48 kHz float, 8 channels");
    if (!mix) return;
    WAVEFORMATEX* closest = reinterpret_cast<WAVEFORMATEX*>(1);
    check(c->IsFormatSupported(AUDCLNT_SHAREMODE_SHARED, mix, &closest) == S_OK && !closest,
          "IsFormatSupported(mix format) is S_OK");
    WAVEFORMATEX stereo{WAVE_FORMAT_PCM, 2, 44100, 44100 * 4, 4, 16, 0};
    check(c->IsFormatSupported(AUDCLNT_SHAREMODE_SHARED, &stereo, &closest) == S_FALSE && closest &&
              closest->nChannels == 8,
          "IsFormatSupported(stereo 44.1 kHz) is S_FALSE with the mix format");
    CoTaskMemFree(closest);
    check(c->IsFormatSupported(AUDCLNT_SHAREMODE_EXCLUSIVE, mix, nullptr) == AUDCLNT_E_UNSUPPORTED_FORMAT,
          "no exclusive mode");
    REFERENCE_TIME def = 0, minimum = 0;
    check(c->GetDevicePeriod(&def, &minimum) == S_OK && def == 100000 && minimum > 0 && minimum <= def,
          "GetDevicePeriod: 10 ms");

    check(c->Initialize(AUDCLNT_SHAREMODE_SHARED, AUDCLNT_STREAMFLAGS_EVENTCALLBACK, 0, 0, mix, nullptr) == S_OK,
          "Initialize(shared, event callback, mix format)");
    check(c->Initialize(AUDCLNT_SHAREMODE_SHARED, 0, 0, 0, mix, nullptr) == AUDCLNT_E_ALREADY_INITIALIZED,
          "a second Initialize is refused");
    CoTaskMemFree(mix);
    check(c->GetBufferSize(&frames) == S_OK && frames >= 960, "GetBufferSize: at least two periods");
    REFERENCE_TIME latency = 0;
    check(c->GetStreamLatency(&latency) == S_OK && latency > 0, "GetStreamLatency");
    check(c->Start() == AUDCLNT_E_EVENTHANDLE_NOT_SET, "Start without an event handle is refused");

    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    check(c->SetEventHandle(event) == S_OK, "SetEventHandle");

    IAudioRenderClient* render = nullptr;
    IAudioClock* clock = nullptr;
    ISimpleAudioVolume* simple = nullptr;
    IAudioStreamVolume* stream = nullptr;
    check(c->GetService(__uuidof(IAudioRenderClient), reinterpret_cast<void**>(&render)) == S_OK && render,
          "GetService(IAudioRenderClient)");
    check(c->GetService(__uuidof(IAudioClock), reinterpret_cast<void**>(&clock)) == S_OK && clock,
          "GetService(IAudioClock)");
    check(c->GetService(__uuidof(ISimpleAudioVolume), reinterpret_cast<void**>(&simple)) == S_OK && simple,
          "GetService(ISimpleAudioVolume)");
    check(c->GetService(__uuidof(IAudioStreamVolume), reinterpret_cast<void**>(&stream)) == S_OK && stream,
          "GetService(IAudioStreamVolume)");
    if (!render || !clock || !simple || !stream) return;
    UINT32 channels = 0;
    float level = 0.0f;
    check(stream->GetChannelCount(&channels) == S_OK && channels == 8 && stream->SetChannelVolume(7, 0.5f) == S_OK &&
              stream->GetChannelVolume(7, &level) == S_OK && level == 0.5f,
          "IAudioStreamVolume keeps a channel volume");
    check(simple->SetMasterVolume(0.25f, nullptr) == S_OK && simple->GetMasterVolume(&level) == S_OK && level == 0.25f,
          "ISimpleAudioVolume keeps the master volume");
    UINT64 freq = 0;
    check(clock->GetFrequency(&freq) == S_OK && freq == 48000, "IAudioClock counts frames at 48 kHz");

    // Fill the whole buffer before Start, as a title primes its stream.
    BYTE* data = nullptr;
    check(render->ReleaseBuffer(1, 0) == AUDCLNT_E_OUT_OF_ORDER, "ReleaseBuffer without GetBuffer");
    check(render->GetBuffer(frames + 1, &data) == AUDCLNT_E_BUFFER_TOO_LARGE, "GetBuffer larger than the buffer");
    check(render->GetBuffer(frames, &data) == S_OK && data, "GetBuffer(whole buffer)");
    if (data) std::memset(data, 0x7f, static_cast<size_t>(frames) * 32);
    check(render->GetBuffer(1, &data) == AUDCLNT_E_OUT_OF_ORDER, "a second GetBuffer before ReleaseBuffer");
    check(render->ReleaseBuffer(frames + 1, 0) == AUDCLNT_E_INVALID_SIZE, "ReleaseBuffer of more than was got");
    check(render->ReleaseBuffer(frames, 0) == S_OK, "ReleaseBuffer(whole buffer)");
    UINT32 padding = 0;
    check(c->GetCurrentPadding(&padding) == S_OK && padding == frames, "padding is the whole buffer before Start");
    Sleep(30);
    check(c->GetCurrentPadding(&padding) == S_OK && padding == frames, "nothing drains before Start");
    check(render->GetBuffer(1, &data) == AUDCLNT_E_BUFFER_TOO_LARGE, "a full buffer takes no more");

    // Run for a few periods, refilling on every event, as an event-driven title does.
    LARGE_INTEGER f, t0, t1;
    QueryPerformanceFrequency(&f);
    check(c->Start() == S_OK, "Start");
    check(c->Start() == AUDCLNT_E_NOT_STOPPED, "a second Start is refused");
    QueryPerformanceCounter(&t0);
    int signals = 0;
    UINT32 lowest = frames;
    UINT64 written = frames;
    for (int i = 0; i < 12; ++i) {
        if (WaitForSingleObject(event, 500) != WAIT_OBJECT_0) break;
        ++signals;
        if (FAILED(c->GetCurrentPadding(&padding))) break;
        if (padding < lowest) lowest = padding;
        const UINT32 room = frames - padding;
        if (room && SUCCEEDED(render->GetBuffer(room, &data))) {
            std::memset(data, 0, static_cast<size_t>(room) * 32);
            if (SUCCEEDED(render->ReleaseBuffer(room, 0))) written += room;
        }
    }
    QueryPerformanceCounter(&t1);
    const double seconds = static_cast<double>(t1.QuadPart - t0.QuadPart) / static_cast<double>(f.QuadPart);
    std::printf("  12 waits: %d signals in %.1f ms, lowest padding %u of %u, %llu frames written\n", signals,
                seconds * 1000.0, lowest, frames, static_cast<unsigned long long>(written));
    check(signals == 12, "the event is signalled every period while running");
    check(seconds > 0.05 && seconds < 2.0, "the periods come at about real time (12 waits in 50 ms .. 2 s)");
    check(lowest < frames, "the padding drains while running");
    check(written > frames, "the drained room was refilled");
    UINT64 pos = 0, qpc = 0;
    check(clock->GetPosition(&pos, &qpc) == S_OK && pos > 0 && pos <= written && qpc > 0,
          "IAudioClock position advances, never past what was written");
    // The clock runs at the stream's rate: frames played track the time since Start (within the buffer).
    const double expected = seconds * 48000.0;
    check(static_cast<double>(pos) > expected * 0.5 && static_cast<double>(pos) <= expected * 1.5 + frames,
          "the position follows real time at 48 kHz");

    check(c->Stop() == S_OK, "Stop");
    UINT32 stopped = 0, later = 0;
    c->GetCurrentPadding(&stopped);
    Sleep(30);
    c->GetCurrentPadding(&later);
    check(stopped == later, "nothing drains after Stop");
    while (WaitForSingleObject(event, 0) == WAIT_OBJECT_0) {
    }
    check(WaitForSingleObject(event, 60) == WAIT_TIMEOUT, "no event after Stop");
    check(c->Stop() == S_FALSE, "a second Stop is S_FALSE");
    check(c->Reset() == S_OK && c->GetCurrentPadding(&padding) == S_OK && padding == 0, "Reset empties the buffer");
    check(clock->GetPosition(&pos, nullptr) == S_OK && pos == 0, "Reset rewinds the clock");

    stream->Release();
    simple->Release();
    clock->Release();
    render->Release();
    check(c->Release() == 0, "the client is released last");
    CloseHandle(event);
}

void test_polled_stream_drains() {
    IMMDeviceEnumerator* e = serve(nullptr, REGDB_E_CLASSNOTREG, sa::Mode::Auto, nullptr);
    IMMDevice* d = nullptr;
    if (!e || FAILED(e->GetDefaultAudioEndpoint(eRender, eConsole, &d)) || !d) return;
    e->Release();
    IAudioClient* c = activate(d, __uuidof(IAudioClient));
    d->Release();
    if (!c) return;
    WAVEFORMATEX stereo{WAVE_FORMAT_IEEE_FLOAT, 2, 48000, 48000 * 8, 8, 32, 0};
    check(c->Initialize(AUDCLNT_SHAREMODE_SHARED, 0, 1000000, 0, &stereo, nullptr) == S_OK,
          "Initialize(shared, polled, stereo float, 100 ms)");
    HANDLE unexpected = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    check(c->SetEventHandle(unexpected) == AUDCLNT_E_EVENTHANDLE_NOT_EXPECTED,
          "SetEventHandle on a polled stream is refused");
    CloseHandle(unexpected);
    UINT32 frames = 0;
    c->GetBufferSize(&frames);
    check(frames >= 4800, "a 100 ms buffer holds 4800 frames");
    IAudioRenderClient* render = nullptr;
    c->GetService(__uuidof(IAudioRenderClient), reinterpret_cast<void**>(&render));
    BYTE* data = nullptr;
    if (render && SUCCEEDED(render->GetBuffer(frames, &data))) render->ReleaseBuffer(frames, AUDCLNT_BUFFERFLAGS_SILENT);
    c->Start();
    Sleep(50);
    UINT32 padding = frames;
    c->GetCurrentPadding(&padding);
    check(padding < frames, "a polled stream drains on its clock");
    Sleep(150);
    c->GetCurrentPadding(&padding);
    check(padding == 0, "a polled stream runs dry when nothing more is written");
    c->Stop();
    if (render) render->Release();
    c->Release();
}

// ---- the spatial stream ---------------------------------------------------------------------------------------

void test_spatial_stream() {
    IMMDeviceEnumerator* e = serve(nullptr, REGDB_E_CLASSNOTREG, sa::Mode::Auto, nullptr);
    IMMDevice* d = nullptr;
    if (!e || FAILED(e->GetDefaultAudioEndpoint(eRender, eConsole, &d)) || !d) {
        check(false, "silent device for the spatial tests");
        return;
    }
    e->Release();
    ISpatialAudioClient2* c2 = nullptr;
    check(d->Activate(__uuidof(ISpatialAudioClient2), CLSCTX_ALL, nullptr, reinterpret_cast<void**>(&c2)) == S_OK && c2,
          "Activate(ISpatialAudioClient2)");
    if (c2) c2->Release();
    ISpatialAudioClient* c = nullptr;
    check(d->Activate(__uuidof(ISpatialAudioClient), CLSCTX_ALL, nullptr, reinterpret_cast<void**>(&c)) == S_OK && c,
          "Activate(ISpatialAudioClient)");
    d->Release();
    if (!c) return;

    AudioObjectType mask = AudioObjectType_None;
    UINT32 dynamic = 99;
    check(c->GetNativeStaticObjectTypeMask(&mask) == S_OK && static_cast<uint32_t>(mask) == 0x1FE,
          "the native static objects are the 7.1 bed");
    check(c->GetMaxDynamicObjectCount(&dynamic) == S_OK && dynamic == 0, "no dynamic objects");
    float x = 0, y = 0, z = 0;
    check(c->GetStaticObjectPosition(AudioObjectType_FrontLeft, &x, &y, &z) == S_OK && x < 0 && z < 0,
          "front left is to the left and in front");
    IAudioFormatEnumerator* formats = nullptr;
    WAVEFORMATEX* format = nullptr;
    UINT32 count = 0;
    check(c->GetSupportedAudioObjectFormatEnumerator(&formats) == S_OK && formats &&
              formats->GetCount(&count) == S_OK && count == 1 && formats->GetFormat(0, &format) == S_OK && format &&
              format->nChannels == 1 && format->nSamplesPerSec == 48000 && format->wBitsPerSample == 32 &&
              format->wFormatTag == WAVE_FORMAT_IEEE_FLOAT,
          "one object format: mono 48 kHz float");
    if (!format) return;
    check(c->IsAudioObjectFormatSupported(format) == S_OK, "IsAudioObjectFormatSupported(its format)");
    WAVEFORMATEX stereo{WAVE_FORMAT_IEEE_FLOAT, 2, 48000, 48000 * 8, 8, 32, 0};
    check(c->IsAudioObjectFormatSupported(&stereo) == AUDCLNT_E_UNSUPPORTED_FORMAT, "a stereo object is refused");
    UINT32 frames = 0;
    check(c->GetMaxFrameCount(format, &frames) == S_OK && frames == 480, "GetMaxFrameCount: 480 (10 ms)");
    check(c->IsSpatialAudioStreamAvailable(__uuidof(ISpatialAudioObjectRenderStream), nullptr) == S_OK,
          "the object render stream is available");

    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    SpatialAudioObjectRenderStreamActivationParams params{};
    params.ObjectFormat = format;
    params.StaticObjectTypeMask =
        static_cast<AudioObjectType>(AudioObjectType_FrontLeft | AudioObjectType_FrontRight | AudioObjectType_LowFrequency);
    params.Category = AudioCategory_GameEffects;
    params.EventHandle = event;
    PROPVARIANT pv;
    PropVariantInit(&pv);
    pv.vt = VT_BLOB;
    pv.blob.cbSize = sizeof params;
    pv.blob.pBlobData = reinterpret_cast<BYTE*>(&params);
    ISpatialAudioObjectRenderStream* stream = nullptr;
    params.EventHandle = nullptr;
    check(c->ActivateSpatialAudioStream(&pv, __uuidof(ISpatialAudioObjectRenderStream),
                                        reinterpret_cast<void**>(&stream)) == E_INVALIDARG && !stream,
          "a stream without an event handle is refused");
    params.EventHandle = event;
    check(c->ActivateSpatialAudioStream(&pv, __uuidof(ISpatialAudioObjectRenderStream),
                                        reinterpret_cast<void**>(&stream)) == S_OK && stream,
          "ActivateSpatialAudioStream(FL | FR | LFE, event)");
    formats->Release();
    c->Release();
    if (!stream) return;

    UINT32 available = 99;
    check(stream->BeginUpdatingAudioObjects(&available, &frames) == S_OK && available == 0 && frames == 480,
          "BeginUpdatingAudioObjects: 480 frames, no dynamic objects");
    check(stream->BeginUpdatingAudioObjects(&available, &frames) == SPTLAUDCLNT_E_OUT_OF_ORDER,
          "a second Begin is out of order");
    ISpatialAudioObject* fl = nullptr;
    ISpatialAudioObject* fr = nullptr;
    ISpatialAudioObject* other = nullptr;
    check(stream->ActivateSpatialAudioObject(AudioObjectType_FrontLeft, &fl) == S_OK && fl, "activate front left");
    check(stream->ActivateSpatialAudioObject(AudioObjectType_FrontRight, &fr) == S_OK && fr, "activate front right");
    check(stream->ActivateSpatialAudioObject(AudioObjectType_FrontLeft, &other) == SPTLAUDCLNT_E_OBJECT_ALREADY_ACTIVE,
          "front left twice is refused");
    check(stream->ActivateSpatialAudioObject(AudioObjectType_BackLeft, &other) ==
              SPTLAUDCLNT_E_STATIC_OBJECT_NOT_AVAILABLE,
          "a type outside the stream's mask is refused");
    check(stream->ActivateSpatialAudioObject(AudioObjectType_Dynamic, &other) == SPTLAUDCLNT_E_NO_MORE_OBJECTS,
          "no dynamic object");
    check(stream->EndUpdatingAudioObjects() == S_OK, "EndUpdatingAudioObjects");
    check(stream->EndUpdatingAudioObjects() == SPTLAUDCLNT_E_OUT_OF_ORDER, "a second End is out of order");
    if (!fl || !fr) return;
    BYTE* buffer = nullptr;
    UINT32 length = 0;
    check(fl->GetBuffer(&buffer, &length) == SPTLAUDCLNT_E_OUT_OF_ORDER, "GetBuffer outside an update");

    check(stream->Start() == S_OK, "Start");
    LARGE_INTEGER f, t0, t1;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t0);
    int passes = 0;
    bool buffers_ok = true;
    for (int i = 0; i < 10; ++i) {
        if (WaitForSingleObject(event, 500) != WAIT_OBJECT_0) break;
        if (FAILED(stream->BeginUpdatingAudioObjects(&available, &frames))) break;
        for (ISpatialAudioObject* o : {fl, fr}) {
            buffer = nullptr;
            length = 0;
            if (FAILED(o->GetBuffer(&buffer, &length)) || !buffer || length != frames * 4) buffers_ok = false;
            else std::memset(buffer, 0, length);
            o->SetVolume(0.5f);
            o->SetPosition(0.0f, 0.0f, -1.0f);
        }
        if (i == 9) fr->SetEndOfStream(0);
        if (SUCCEEDED(stream->EndUpdatingAudioObjects())) ++passes;
    }
    QueryPerformanceCounter(&t1);
    const double seconds = static_cast<double>(t1.QuadPart - t0.QuadPart) / static_cast<double>(f.QuadPart);
    std::printf("  spatial: %d passes in %.1f ms\n", passes, seconds * 1000.0);
    check(passes == 10, "the spatial event is signalled every pass while running");
    check(seconds > 0.04 && seconds < 2.0, "the passes come at about real time (10 in 40 ms .. 2 s)");
    check(buffers_ok, "every object gets a 480-frame mono float buffer in each pass");
    BOOL active = TRUE;
    check(fr->IsActive(&active) == S_OK && !active, "an object is inactive after the pass it ended in");
    check(fl->IsActive(&active) == S_OK && active, "the other object stays active");
    check(stream->Reset() == SPTLAUDCLNT_E_STREAM_NOT_STOPPED, "Reset while running is refused");
    check(stream->Stop() == S_OK, "Stop");
    while (WaitForSingleObject(event, 0) == WAIT_OBJECT_0) {
    }
    check(WaitForSingleObject(event, 60) == WAIT_TIMEOUT, "no spatial event after Stop");
    check(stream->Reset() == S_OK, "Reset");
    fr->Release();
    if (SUCCEEDED(stream->BeginUpdatingAudioObjects(&available, &frames))) {
        check(stream->ActivateSpatialAudioObject(AudioObjectType_FrontRight, &other) == S_OK && other,
              "an ended type can be activated again");
        if (other) other->Release();
        stream->EndUpdatingAudioObjects();
    }
    fl->Release();
    check(stream->Release() == 0, "the stream is released last");
    CloseHandle(event);
}

// ---- through CoCreateInstance ---------------------------------------------------------------------------------

using CoCreateInstFn = HRESULT(WINAPI*)(REFCLSID, LPUNKNOWN, DWORD, REFIID, LPVOID*);

void test_co_create_instance_forced() {
    auto create = reinterpret_cast<CoCreateInstFn>(rt::shims::find("api-ms-win-core-com-l1-1-0.dll", "CoCreateInstance"));
    check(create != nullptr, "CoCreateInstance resolves");
    if (!create) return;
    const HRESULT init = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    // Forced: no host enumerator is made for a render question, so this touches no host audio at all.
    SetEnvironmentVariableA("LEONIDA_AUDIO", "Silent");
    check(sa::mode_from_env() == sa::Mode::Silent, "LEONIDA_AUDIO=Silent is the silent mode (any case)");
    IMMDeviceEnumerator* e = nullptr;
    const HRESULT hr = create(sa::kClsidMMDeviceEnumerator, nullptr, CLSCTX_ALL, __uuidof(IMMDeviceEnumerator),
                              reinterpret_cast<void**>(&e));
    check(SUCCEEDED(hr) && e, "CoCreateInstance(MMDeviceEnumerator) with LEONIDA_AUDIO=silent");
    if (e) {
        IMMDevice* d = nullptr;
        check(e->GetDefaultAudioEndpoint(eRender, eConsole, &d) == S_OK && is_silent_device(d),
              "LEONIDA_AUDIO=silent: the default render endpoint is the silent one");
        IAudioClient* c = d ? activate(d, __uuidof(IAudioClient)) : nullptr;
        check(c != nullptr, "and it activates an IAudioClient");
        if (c) c->Release();
        if (d) d->Release();
        e->Release();
    }
    SetEnvironmentVariableA("LEONIDA_AUDIO", "host");
    check(sa::mode_from_env() == sa::Mode::Auto, "any other LEONIDA_AUDIO value is the default");
    SetEnvironmentVariableA("LEONIDA_AUDIO", nullptr);
    check(sa::mode_from_env() == sa::Mode::Auto, "unset LEONIDA_AUDIO is the default");
    if (SUCCEEDED(init)) CoUninitialize();
}

}  // namespace

int main() {
    test_host_without_endpoints();
    test_host_with_endpoint();
    test_no_host_enumerator();
    test_render_stream();
    test_polled_stream_drains();
    test_spatial_stream();
    test_co_create_instance_forced();
    std::printf("%d check(s), %d failure(s)\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
