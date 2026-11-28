// gameinput.dll without SDL3 (gameinput_no_sdl.cpp): the test seam of its scripted pad.
#pragma once
#include <cstdint>

namespace rt::gameinput_no_sdl {

// The clock the scripted pad reads, in microseconds (GetCurrentTimestamp's scale). Set it before the first
// GameInputCreate, which starts the script's clock; nullptr restores the performance counter.
using ClockFn = uint64_t (*)();
void set_clock(ClockFn now_us);

}  // namespace rt::gameinput_no_sdl
