// The silent output endpoint: an in-process WASAPI render device that throws every sample away.
//
// The title creates its MMDeviceEnumerator through CoCreateInstance (com_shim.cpp). When the host has no render
// endpoint, GetDefaultAudioEndpoint(eRender, ...) fails with 0x80070490 (ERROR_NOT_FOUND); the title's device start
// then leaks its own output-device lock and its audio engine never mixes, so the boot stops. A console always has an
// output, so the title only meets that path off-console. The enumerator served here answers render questions with a
// device of its own when the host has none (or always, with LEONIDA_AUDIO=silent): an ISpatialAudioClient (what the
// title activates) and an IAudioClient, whose streams consume frames in real time and signal their event every
// period. It never opens a host audio device.
// Capture questions keep the host's answer.
#pragma once

#include <windows.h>
#include <objbase.h>
#include <mmdeviceapi.h>

#include <cstdint>

namespace rt::silent_audio {

// {BCDE0395-E52F-467C-8E3D-C4579291692E}
inline constexpr CLSID kClsidMMDeviceEnumerator{
    0xBCDE0395, 0xE52F, 0x467C, {0x8E, 0x3D, 0xC4, 0x57, 0x92, 0x91, 0x69, 0x2E}};

// The endpoint ID GetId returns (the form of a host render endpoint ID).
inline constexpr wchar_t kDeviceId[] = L"{0.0.0.00000000}.{5a4c2a5e-1e0d-4c1b-9a51-6c656f6e6964}";
inline constexpr wchar_t kFriendlyName[] = L"Leonida silent output";

// The mix format: 48 kHz, 32-bit float, 7.1 (eight channels, KSAUDIO_SPEAKER_7POINT1_SURROUND).
inline constexpr uint32_t kSampleRate = 48000;
inline constexpr uint16_t kChannels = 8;
inline constexpr uint32_t kChannelMask = 0x63F;
// The engine period: 10 ms (default), 3 ms (minimum), in 100 ns units.
inline constexpr int64_t kDefaultPeriodHns = 100000;
inline constexpr int64_t kMinimumPeriodHns = 30000;

enum class Mode {
    Auto,    // the host's render endpoint when it has one, the silent one when it has none
    Silent,  // LEONIDA_AUDIO=silent: the silent endpoint even when the host has one
};

// LEONIDA_AUDIO, read on every call: "silent" (any case) is Mode::Silent, anything else Mode::Auto.
Mode mode_from_env();

// Makes the host's enumerator on demand (Mode::Silent creates none until a capture question needs one); may be null.
using HostFactory = IMMDeviceEnumerator* (*)();

// What CoCreateInstance(CLSID_MMDeviceEnumerator) returns. `host` is the host's enumerator or null, with `host_hr`
// the result of creating it; it is not consumed (what is kept is AddRef'd). In Mode::Auto, a host that answers
// GetDefaultAudioEndpoint(eRender, eConsole) is returned itself, unchanged; otherwise the wrapper answers.
HRESULT serve_enumerator(IMMDeviceEnumerator* host, HRESULT host_hr, Mode mode, HostFactory factory, REFIID riid,
                         void** ppv);

// The CoCreateInstance hook: the mode from the environment, the host's enumerator (not made in Mode::Silent), then
// serve_enumerator.
HRESULT create_enumerator(DWORD cls_context, REFIID riid, void** ppv);

// For tests: how many silent devices have been handed out.
uint32_t devices_served();

}  // namespace rt::silent_audio
