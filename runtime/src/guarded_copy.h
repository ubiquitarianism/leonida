#pragma once
// Copies and reads made inside a __try frame that must turn an access violation into an error (the shadow and
// title-texture copies of the D3D12 layer, DirectStorage request copies, the kernel-write retry).
//
// The faulting access has to happen in a function of ours, not inlined into the frame with the __try:
// - Under Wine an access violation raised inside the CRT's memcpy or memmove (Wine's ucrtbase) never reaches the
//   __except of the frame that called it, and the process ends ("Unhandled page fault ... starting debugger").
//   guarded_copy_test --crt shows it: the same fault raised in a function of ours is caught. A Linux boot ended that way
//   in the slot-133 shadow refresh, reading a title page the title was remapping (CopyPageMappingsBatchX).
// - clang-cl, without /EHa, catches only what a call inside the __try raises, not an inline load or store.
// So the bytes move here: never inlined, and no_builtin keeps the compiler from turning a loop back into a call to
// memcpy or memmove.
#include <cstddef>
#include <cstdint>
#include <intrin.h>

// clang-cl: no_builtin keeps the loops below from becoming CRT calls. MSVC has no such attribute (and accepts no
// __attribute__ at all); it does not turn __movsb or a loop over volatile bytes into memcpy/memmove, so nothing is needed.
#if defined(__clang__)
#define LEONIDA_GUARDED_NO_BUILTIN __attribute__((no_builtin))
#else
#define LEONIDA_GUARDED_NO_BUILTIN
#endif

namespace leonida::guarded {

// Forward copy, REP MOVSB: the source and destination do not overlap, or the destination is below the source.
__declspec(noinline) LEONIDA_GUARDED_NO_BUILTIN inline void copy(void* to, const void* from, std::size_t bytes) {
    __movsb(static_cast<unsigned char*>(to), static_cast<const unsigned char*>(from), bytes);
}

// memmove: forward when that is safe, else from the end down, byte by byte (rare: only an overlapping move upwards).
__declspec(noinline) LEONIDA_GUARDED_NO_BUILTIN inline void move(void* to, const void* from, std::size_t bytes) {
    auto* d = static_cast<volatile unsigned char*>(to);
    const auto* s = static_cast<const volatile unsigned char*>(from);
    const auto dst = reinterpret_cast<std::uintptr_t>(to), src = reinterpret_cast<std::uintptr_t>(from);
    if (dst <= src || dst >= src + bytes) {
        __movsb(static_cast<unsigned char*>(to), static_cast<const unsigned char*>(from), bytes);
        return;
    }
    for (std::size_t i = bytes; i-- > 0;) d[i] = s[i];
}

// An aligned 32- or 64-bit store in one locked access (XCHG): a thread polling the value, or reading it with one
// load, sees the old value or the new one, never half of each (a REP MOVSB may store byte by byte).
__declspec(noinline) inline void store32(void* to, std::uint32_t value) {
    _InterlockedExchange(static_cast<volatile long*>(to), static_cast<long>(value));
}
__declspec(noinline) inline void store64(void* to, std::uint64_t value) {
    _InterlockedExchange64(static_cast<volatile long long*>(to), static_cast<long long>(value));
}

// One word read: for a __try that only asks whether memory still holds what it did (an OVERLAPPED the title may have
// freed).
__declspec(noinline) inline std::uintptr_t read_word(const void* at) {
    return *static_cast<const volatile std::uintptr_t*>(at);
}

}  // namespace leonida::guarded
