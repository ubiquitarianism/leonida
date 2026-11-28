// Checks the decoded path of the registered xapu.dll shim (runtime/shims/xapu_queue.cpp) with
// runtime/opus_decoder built: the title's ConnectInput, activate/process/deactivate, one result per
// command in order, the 0x50-byte result layout, and a generated 440 Hz stereo packet decoded to
// 960 frames of f32 PCM, then clipped by first_frame / frame_count / output_bytes. The packet is
// encoded here; no game data.
#include <windows.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "leonida_opus_decoder.h"
#include "shims.h"

namespace {

using XApuConnectFn = HRESULT(__stdcall*)(const void*, void*, void**);
using XApuDisconnectFn = HRESULT(__stdcall*)(void*, uint32_t);
using XApuEnqueueCommandFn = HRESULT(__stdcall*)(void*, const void*, uint32_t*);
using XApuGetResultCountFn = uint32_t(__stdcall*)(void*, uint32_t*);
using XApuDequeueResultFn = HRESULT(__stdcall*)(void*, void*);

constexpr HRESULT kQueueFull = static_cast<HRESULT>(0x8AC81000u);
constexpr HRESULT kInvalidStream = static_cast<HRESULT>(0x8AC81003u);

#pragma pack(push, 1)
struct ConnectInput {
    uint32_t processing, streams, commands_per_stream, memory_bytes, frames_per_output, options;
};
struct ConnectOutput {
    void* memory;
    HANDLE result_event;
};
struct CommandId {
    uint32_t kind, stream, sequence, options;
    uint64_t context;
};
struct ActivateCommand {
    CommandId id;
    uint32_t channels, init_bytes;
    const void* init;
    float start_pitch;
    uint32_t processing_bytes;
    void* processing;
};
struct ProcessCommand {
    CommandId id;
    float pitch_ramp, target_pitch;
    uint32_t first_frame, frame_count, input_bytes;
    const uint8_t* input;
    uint32_t output_bytes;
    uint8_t* output;
};
struct Result {
    CommandId id;
    uint32_t state;
    uint64_t state_data, usec;
    HRESULT error;
    uint32_t frames, output_bytes;
    uint8_t* output;
    float pitch;
    uint32_t input_bytes;
    const uint8_t* input;
};
#pragma pack(pop)
static_assert(sizeof(ActivateCommand) == 56 && sizeof(ProcessCommand) == 64 && sizeof(Result) == 0x50, "layouts");

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

template <typename T>
T get(const char* dll, const char* name) {
    void* fn = rt::shims::find(dll, name);
    check(fn != nullptr, name);
    return reinterpret_cast<T>(fn);
}

CommandId id(uint32_t kind, uint32_t stream, uint32_t options = 0) { return {kind, stream, 7, options, 0x1234}; }

}  // namespace

int main() {
    std::printf("xapu_opus_test: decoded xapu.dll path\n");
    const auto connect = get<XApuConnectFn>("xapu.dll", "XApuConnect");
    const auto disconnect = get<XApuDisconnectFn>("xapu.dll", "XApuDisconnect");
    const auto enqueue = get<XApuEnqueueCommandFn>("xapu.dll", "XApuEnqueueCommand");
    const auto count = get<XApuGetResultCountFn>("xapu.dll", "XApuGetResultCount");
    const auto dequeue = get<XApuDequeueResultFn>("xapu.dll", "XApuDequeueResult");
    if (g_failures) return 1;

    constexpr uint32_t kMemory = 0x14000 + 128 * 0x1E00;
    const ConnectInput in{1, 128, 2, kMemory, 0, 0};
    const ConnectInput in_copy = in;
    ConnectOutput out{};
    void* apu = nullptr;
    check(SUCCEEDED(connect(&in, &out, &apu)) && apu, "XApuConnect for Opus decoding");
    check(std::memcmp(&in, &in_copy, sizeof in) == 0, "XApuConnect leaves its input alone");
    auto* base = static_cast<uint8_t*>(out.memory);
    if (!apu || !base) return 1;
    check(base[0] == 0 && base[kMemory - 1] == 0, "the shared memory is zeroed");

    const ActivateCommand activate{id(1, 3), 2, 0, nullptr, 1.0f, 0, nullptr};
    check(enqueue(apu, &activate, nullptr) == S_OK, "activate stream 3 (stereo)");

    std::vector<float> tone(960 * 2);
    for (uint32_t i = 0; i < 960; ++i) {
        const float sample = static_cast<float>(std::sin(double(i) * 440.0 * 6.283185307179586 / 48000.0)) * 0.5f;
        tone[2 * i] = sample;
        tone[2 * i + 1] = sample;
    }
    uint8_t encoded[1500];
    size_t encoded_len = 0;
    check(leonida_opus_encode_test(tone.data(), 960, 2, encoded, sizeof encoded, &encoded_len) == LEONIDA_OPUS_OK &&
              encoded_len > 0,
          "a synthetic 20 ms stereo Opus packet encodes");
    std::memcpy(base, encoded, encoded_len);
    auto* const pcm = base + 0x14000 + 3 * 0x1E00;
    std::memset(pcm, 0x55, 0x1E00 + 64);
    ProcessCommand process{id(3, 3, 1), 0.0f, 1.0f, 0, 960, static_cast<uint32_t>(encoded_len), base, 0x1E00, pcm};
    uint32_t total = 0;
    check(enqueue(apu, &process, &total) == S_OK && total == 2, "process on stream 3; two commands accepted");
    check(enqueue(apu, &process, &total) == kQueueFull && total == 2,
          "a third command on a stream of two is 0x8AC81000");
    check(count(apu, nullptr) == 2, "two results ready");

    Result r{};
    check(dequeue(apu, &r) == S_OK && r.id.kind == 1 && r.id.stream == 3 && r.error == S_OK,
          "results come in command order: the activate first");
    check(dequeue(apu, &r) == S_OK && r.id.kind == 3 && r.error == S_OK, "the process result follows");
    uint32_t dword34 = 0;
    std::memcpy(&dword34, reinterpret_cast<const uint8_t*>(&r) + 0x34, 4);
    check(r.frames == 960 && dword34 == 960 * 8 && r.output == pcm && r.input == base &&
              r.input_bytes == static_cast<uint32_t>(encoded_len),
          "960 stereo float frames, the byte count at +0x34, the buffers echoed");
    float energy = 0.0f;
    const auto* floats = reinterpret_cast<const float*>(pcm);
    for (uint32_t i = 0; i < 960 * 2; ++i) energy += floats[i] * floats[i];
    energy /= 960 * 2;
    check(energy > 0.01f, "the decoded tone has energy");
    check(pcm[960 * 8] == 0x55, "nothing past the decoded range is written");
    check(count(apu, nullptr) == 0, "the queue is empty after both results");

    // first_frame, frame_count and what output_bytes holds each clip the decoded frames.
    process.first_frame = 500;
    process.frame_count = 4000;
    enqueue(apu, &process, nullptr);
    dequeue(apu, &r);
    check(r.error == S_OK && r.frames == 460 && r.output_bytes == 460 * 8, "frames from first_frame to the packet's end");
    process.frame_count = 300;
    enqueue(apu, &process, nullptr);
    dequeue(apu, &r);
    check(r.frames == 300 && r.output_bytes == 300 * 8, "no more than frame_count");
    process.frame_count = 4000;
    process.output_bytes = 100 * 8 + 7;
    enqueue(apu, &process, nullptr);
    dequeue(apu, &r);
    check(r.frames == 100 && r.output_bytes == 800, "and no more than the output holds");

    // Errors carried in the result.
    ProcessCommand empty = process;
    empty.input_bytes = 0;
    check(enqueue(apu, &empty, nullptr) == S_OK && dequeue(apu, &r) == S_OK && r.error == E_INVALIDARG,
          "an empty input: E_INVALIDARG in the result");
    empty = process;
    empty.output = nullptr;
    check(enqueue(apu, &empty, nullptr) == S_OK && dequeue(apu, &r) == S_OK && r.error == E_INVALIDARG,
          "a null output: E_INVALIDARG in the result");
    ProcessCommand stray = process;
    stray.id.stream = 4;
    check(enqueue(apu, &stray, nullptr) == S_OK && dequeue(apu, &r) == S_OK && r.error == kInvalidStream,
          "a process on a stream never activated: 0x8AC81003 in the result");

    // Deactivate closes the stream; activating again creates a fresh decoder.
    const CommandId deactivate = id(2, 3);
    check(enqueue(apu, &deactivate, nullptr) == S_OK && dequeue(apu, &r) == S_OK && r.error == S_OK,
          "deactivate gives a result and closes the stream");
    check(enqueue(apu, &process, nullptr) == S_OK && dequeue(apu, &r) == S_OK && r.error == kInvalidStream,
          "after deactivate the stream takes no process command");
    check(enqueue(apu, &activate, nullptr) == S_OK && dequeue(apu, &r) == S_OK && r.error == S_OK,
          "the stream can be activated again");
    check(disconnect(apu, 0x20) == S_OK, "XApuDisconnect(handle, 0x20)");

    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
