#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <thread>
#include <vector>

namespace rt {

// One thread's call chain, taken without stopping the thread for longer than a context and stack copy.
class StackSampler {
public:
    static constexpr std::size_t kMaxFrames = 40;
    static constexpr std::size_t kMaxStackBytes = 8 * 1024 * 1024;

    StackSampler();  // Allocates the stack copy once, so sample() allocates nothing.

    // `thread` needs THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT. While it is suspended this makes system calls only
    // (GetThreadContext, VirtualQuery, ReadProcessMemory): no heap, stdio or user-mode locks, which the suspended
    // thread may hold. After ResumeThread it unwinds the copy with RtlLookupFunctionEntry/RtlVirtualUnwind, reading
    // stack words from the copy rather than the live stack, which has moved on. Writes up to kMaxFrames addresses
    // (the instruction pointer, then return addresses) and returns how many; 0 if the thread could not be sampled.
    std::size_t sample(void* thread, std::uint64_t* frames);

private:
    std::unique_ptr<std::byte[]> stack_;
};

// Parses LEONIDA_STACKS: comma-separated whole seconds. Returns false (and leaves `seconds` empty) on bad text.
bool parse_stack_times(std::string_view text, std::vector<unsigned>& seconds);

// LEONIDA_STACKS: snapshots of every thread at the listed times since this call, printed to stderr. Destroy the
// sampler before unmapping the title image.
std::jthread start_thread_stacks();

// The call chain of thread `tid` now, or of every other thread when `tid` is 0, printed as LEONIDA_STACKS does under
// the tag `why`. One call at a time; the device-loss diagnostics call it (vulkan_device_fault.h thread_stack_hook).
void print_thread_stacks_now(std::uint32_t tid, const char* why) noexcept;

}  // namespace rt
